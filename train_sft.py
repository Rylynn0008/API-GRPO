#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_sft_v4.py
===============
SFT-v4：从 Qwen2.5-7B-Instruct 基座从头训练，四字段输出格式。

输出字段：
  label / primary_rule / trigger_location / fix_hint

相比 v3 的变化：
  - 从基座模型从头训练（不继承任何已有 checkpoint）
  - 增加 trigger_location 和 fix_hint 字段（替换 fix_code）
  - 3 epochs（从头训练需要更多轮次）
  - 评估新增 location_acc、fix_hint_rate 指标

用法：
  python train_sft_v4.py \
      --dataset_dir ./sft_data_v4 \
      --output_dir  ./loras/crypto_sft_v4 \
      --cuda_device 0
"""

import os, re, json, argparse, random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--cuda_device",           type=str,   default="0")
parser.add_argument("--model_name",            type=str,
                    default="YOUR_BASE_MODEL_PATH")
parser.add_argument("--dataset_dir",           type=str,   required=True)
parser.add_argument("--output_dir",            type=str,   default=None)
parser.add_argument("--train_epochs",          type=int,   default=3)
parser.add_argument("--batch_size",            type=int,   default=4)
parser.add_argument("--gradient_accumulation", type=int,   default=4)
parser.add_argument("--learning_rate",         type=float, default=2e-5)
parser.add_argument("--max_seq_length",        type=int,   default=2048)
parser.add_argument("--lora_rank",             type=int,   default=32)
parser.add_argument("--lora_alpha",            type=int,   default=64)
parser.add_argument("--load_in_4bit",          action="store_true",
                    help="Use 4bit quantization to save memory")
parser.add_argument("--rare_rule_threshold",   type=int,   default=50)
parser.add_argument("--data_fraction",         type=float, default=1.0,
                    help="Fraction of training data to use (for weak baseline)")
parser.add_argument("--skip_training",         action="store_true")
parser.add_argument("--model_to_eval",         type=str,   default=None)
parser.add_argument("--max_eval_samples",      type=int,   default=None)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

# ─── 常量 ────────────────────────────────────────────────────────────────────

ALL_RULES = [
    "Rule-1a","Rule-1b","Rule-1c","Rule-1d",
    "Rule-2a","Rule-2b","Rule-2c","Rule-2d",
    "Rule-3a","Rule-3b",
    "Rule-4a","Rule-4b","Rule-4c","Rule-4d","Rule-4e",
    "Rule-5a","Rule-5b","Rule-5c",
    "Rule-6a","Rule-6b",
    "none",
]

GROUP_RULES = {
    "G1": ["Rule-1a","Rule-1b","Rule-1c","Rule-1d"],
    "G2": ["Rule-2a","Rule-2b","Rule-2c","Rule-2d"],
    "G3": ["Rule-3a","Rule-3b"],
    "G4": ["Rule-4a","Rule-4b","Rule-4c","Rule-4d","Rule-4e"],
    "G5": ["Rule-5a","Rule-5b","Rule-5c"],
    "G6": ["Rule-6a","Rule-6b"],
}

print("=" * 70)
print("Crypto API Misuse Detection — SFT v4")
print("  Output: label + primary_rule + trigger_location + fix_hint")
print("=" * 70)
print(f"  Base model : {args.model_name}")
print(f"  Dataset    : {args.dataset_dir}")
print(f"  LR={args.learning_rate}  LoRA r={args.lora_rank}  Epochs={args.train_epochs}")
print(f"  Batch={args.batch_size}×{args.gradient_accumulation}  MaxSeq={args.max_seq_length}")

# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


data_dir  = Path(args.dataset_dir)
train_raw = load_jsonl(data_dir / "train_sft.jsonl")
val_raw   = load_jsonl(data_dir / "val_sft.jsonl")
test_raw  = load_jsonl(data_dir / "test_sft.jsonl")

# 数据采样（用于弱基线）
if args.data_fraction < 1.0:
    import random
    random.seed(42)
    n_samples = int(len(train_raw) * args.data_fraction)
    train_raw = random.sample(train_raw, n_samples)
    print(f"\n  [Weak baseline] Sampled {args.data_fraction*100:.0f}% training data: {n_samples} samples")

print(f"\n  Train: {len(train_raw)}  Val: {len(val_raw)}  Test: {len(test_raw)}")

# ─── 稀有规则过采样 ───────────────────────────────────────────────────────────

rule_count = Counter(s["_meta"]["primary_rule"] for s in train_raw)
print(f"\n  Train rule distribution:")
rare_rules = {r for r, c in rule_count.items() if c < args.rare_rule_threshold}
for r, c in sorted(rule_count.items()):
    flag = " ← rare (×4)" if r in rare_rules else ""
    print(f"    {r:10s}: {c:4d}{flag}")

# location A级统计
loc_a = sum(1 for s in train_raw
            if s["_meta"].get("location_grade") == "A")
loc_b = sum(1 for s in train_raw
            if s["_meta"].get("location_grade") == "B")
print(f"\n  trigger_location quality: A={loc_a}, B={loc_b}, "
      f"none={len(train_raw)-loc_a-loc_b}")


def apply_weighted_sampling(data: List[Dict]) -> List[Dict]:
    out = []
    for s in data:
        rule = s["_meta"]["primary_rule"]
        copies = 4 if rule in rare_rules else 1
        out.extend([s] * copies)
    random.shuffle(out)
    return out


train_weighted = apply_weighted_sampling(train_raw)
print(f"\n  After rare-rule weighting: {len(train_raw)} → {len(train_weighted)}")

# ─── 模型加载 ─────────────────────────────────────────────────────────────────

if not args.skip_training:
    print(f"\nLoading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model_name,
        max_seq_length = args.max_seq_length,
        dtype          = None if args.load_in_4bit else torch.bfloat16,
        load_in_4bit   = args.load_in_4bit,
        device_map     = "auto",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_rank,
        target_modules = ["q_proj","k_proj","v_proj","o_proj",
                          "gate_proj","up_proj","down_proj"],
        lora_alpha     = args.lora_alpha,
        lora_dropout   = 0.05,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # ─── Tokenize ─────────────────────────────────────────────────────────────

    ASSISTANT_START = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False)

    def tokenize_fn(examples):
        texts = []
        for conv in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        enc = tokenizer(
            texts,
            max_length     = args.max_seq_length,
            padding        = False,
            truncation     = True,
            return_tensors = None,
        )
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    def mask_prompt_labels(batch):
        astart = ASSISTANT_START
        for i in range(len(batch["labels"])):
            ids  = batch["input_ids"][i]
            lbls = batch["labels"][i]
            pos  = None
            for j in range(len(ids) - len(astart)):
                if ids[j:j + len(astart)] == astart:
                    pos = j + len(astart)
                    break
            if pos is not None:
                for j in range(pos):
                    lbls[j] = -100
            batch["labels"][i] = lbls
        return batch

    def to_hf_dataset(raw: List[Dict]) -> Dataset:
        ds = Dataset.from_list([{"conversations": s["conversations"]} for s in raw])
        ds = ds.map(tokenize_fn, batched=True, batch_size=200,
                    remove_columns=["conversations"], desc="Tokenizing")
        ds = ds.map(mask_prompt_labels, batched=True, batch_size=200,
                    desc="Masking prompts")
        return ds

    print("\nTokenizing datasets...")
    train_ds = to_hf_dataset(train_weighted)
    val_ds   = to_hf_dataset(val_raw)

    # ─── Training ─────────────────────────────────────────────────────────────

    if args.output_dir is None:
        model_short    = args.model_name.rstrip("/").split("/")[-1]
        args.output_dir = (
            f"/path/to/project/loras/"
            f"crypto_sft_v4/{model_short}_r{args.lora_rank}_e{args.train_epochs}"
        )

    n_train     = len(train_ds)
    eff_batch   = args.batch_size * args.gradient_accumulation
    steps_epoch = n_train // eff_batch
    max_steps   = steps_epoch * args.train_epochs

    print(f"\nTraining: {n_train} samples, {steps_epoch} steps/epoch, {max_steps} total")

    training_args = TrainingArguments(
        per_device_train_batch_size  = args.batch_size,
        per_device_eval_batch_size   = args.batch_size,
        gradient_accumulation_steps  = args.gradient_accumulation,
        warmup_steps                 = max(1, int(0.05 * max_steps)),
        num_train_epochs             = args.train_epochs,
        learning_rate                = args.learning_rate,
        bf16                         = torch.cuda.is_bf16_supported(),
        fp16                         = not torch.cuda.is_bf16_supported(),
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",
        logging_steps                = max(1, steps_epoch // 10),
        evaluation_strategy          = "steps",
        eval_steps                   = max(1, steps_epoch // 2),
        save_strategy                = "steps",
        save_steps                   = max(1, steps_epoch // 2),
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
        greater_is_better            = False,
        save_total_limit             = 2,
        seed                         = 42,
        output_dir                   = args.output_dir,
        report_to                    = "tensorboard",
    )

    trainer = SFTTrainer(
        model          = model,
        tokenizer      = tokenizer,
        train_dataset  = train_ds,
        eval_dataset   = val_ds,
        data_collator  = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, return_tensors="pt"),
        max_seq_length = args.max_seq_length,
        args           = training_args,
    )

    print("\n" + "=" * 70)
    print("Training started...")
    print("=" * 70 + "\n")
    stats = trainer.train()

    lora_path = args.output_dir + "_final"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"\nLoRA saved → {lora_path}")
    print(f"Train loss: {stats.training_loss:.4f}  Steps: {stats.global_step}")
    trained_path = lora_path

    del trainer, model, tokenizer
    import gc; gc.collect()
    torch.cuda.empty_cache()
else:
    trained_path = args.model_to_eval
    if not trained_path:
        raise ValueError("--model_to_eval required with --skip_training")

# ─── 评估 ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Evaluation Phase")
print("=" * 70)

print(f"\nLoading model for inference: {trained_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = args.model_name,
    max_seq_length = args.max_seq_length,
    dtype          = torch.bfloat16,
    load_in_4bit   = False,
)
model = PeftModel.from_pretrained(model, trained_path, device_map={"": 0})
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")


# ─── 解析输出 ─────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> Dict:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1).strip())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        # 降级解析
        rm = re.search(r'"primary_rule"\s*:\s*"([^"]+)"', m.group(1))
        lm = re.search(r'"label"\s*:\s*(\d)',              m.group(1))
        return {
            "primary_rule":     rm.group(1) if rm else "parse_error",
            "label":            int(lm.group(1)) if lm else -1,
            "trigger_location": "",
            "fix_hint":         "",
        }


def compute_rule_f1(true_rules: List[str], pred_rules: List[str]) -> Dict:
    results = {}
    for rule in sorted(set(true_rules + pred_rules)):
        tp = sum(1 for t, p in zip(true_rules, pred_rules) if t == rule and p == rule)
        fp = sum(1 for t, p in zip(true_rules, pred_rules) if t != rule and p == rule)
        fn = sum(1 for t, p in zip(true_rules, pred_rules) if t == rule and p != rule)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[rule] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": sum(1 for t in true_rules if t == rule),
        }
    return results


# ─── location_acc 计算 ────────────────────────────────────────────────────────

def compute_location_acc(
    results: List[Dict],
    gold_meta: List[Dict],
) -> Dict:
    """
    只对 A 级 location（gold location 含 "at line"）的样本计算命中率。
    命中条件：gold_api ∈ pred_trigger_location（大小写不敏感）。
    """
    a_total = 0
    a_hit   = 0
    b_total = 0
    b_hit   = 0

    for res, meta in zip(results, gold_meta):
        gold_loc   = meta.get("trigger_location", "none")
        pred_loc   = res.get("pred_trigger_location", "")
        grade      = meta.get("location_grade", "none")

        if grade == "none" or gold_loc == "none":
            continue

        # 提取 gold_api（"at line" 前的部分）
        gold_api = gold_loc.split(" at line")[0].split(" (location")[0].strip()

        hit = gold_api.lower() in (pred_loc or "").lower()

        if grade == "A":
            a_total += 1
            if hit:
                a_hit += 1
        elif grade == "B":
            b_total += 1
            if hit:
                b_hit += 1

    return {
        "location_acc_A": a_hit / a_total if a_total > 0 else 0.0,
        "location_acc_B": b_hit / b_total if b_total > 0 else 0.0,
        "a_total": a_total, "a_hit": a_hit,
        "b_total": b_total, "b_hit": b_hit,
    }


# ─── 主评估函数 ───────────────────────────────────────────────────────────────

def evaluate(raw_data: List[Dict], split_name: str, out_file: str) -> Dict:
    if args.max_eval_samples and len(raw_data) > args.max_eval_samples:
        raw_data = random.sample(raw_data, args.max_eval_samples)

    true_rules,  pred_rules  = [], []
    true_labels, pred_labels = [], []
    parse_errors = 0
    fix_hint_present = 0
    results      = []
    gold_metas   = []

    print(f"\nEvaluating {split_name} ({len(raw_data)} samples)...")

    for s in tqdm(raw_data, desc=split_name):
        conv  = s["conversations"]
        meta  = s["_meta"]
        msgs  = [conv[0], conv[1]]   # system + user

        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids      = inputs,
                max_new_tokens = 1024,
                temperature    = 0.1,
                top_p          = 0.9,
                do_sample      = True,
                pad_token_id   = tokenizer.pad_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )
        gen  = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen)

        t_rule  = meta["primary_rule"]
        t_label = meta["label"]
        p_rule  = pred.get("primary_rule", "parse_error")
        p_label = pred.get("label", 1 if p_rule not in ("none", "parse_error") else 0)
        p_loc   = pred.get("trigger_location", "")
        p_hint  = pred.get("fix_hint", "")

        if p_rule == "parse_error":
            parse_errors += 1
        if t_label == 1 and p_hint and p_hint.strip() not in ("", "none"):
            fix_hint_present += 1

        true_rules.append(t_rule);   pred_rules.append(p_rule)
        true_labels.append(t_label); pred_labels.append(
            1 if p_rule not in ("none", "parse_error") else 0)

        results.append({
            "true_rule":             t_rule,
            "pred_rule":             p_rule,
            "true_label":            t_label,
            "pred_label":            p_label,
            "correct_rule":          t_rule == p_rule,
            "gold_trigger_location": meta.get("trigger_location", ""),
            "pred_trigger_location": p_loc,
            "pred_fix_hint":         p_hint,
            "location_grade":        meta.get("location_grade", "none"),
            "response":              gen[:800],
        })
        gold_metas.append(meta)

    # ── 核心指标 ──
    n         = len(results)
    label_acc = sum(1 for tl, pl in zip(true_labels, pred_labels) if tl == pl) / n
    rule_acc  = sum(1 for tr, pr in zip(true_rules,  pred_rules)  if tr == pr) / n
    rf1       = compute_rule_f1(true_rules, pred_rules)
    pos_f1s   = [v["f1"] for r, v in rf1.items()
                 if r != "none" and v["support"] > 0]
    macro_f1  = sum(pos_f1s) / len(pos_f1s) if pos_f1s else 0.0

    n_pos         = sum(1 for tl in true_labels if tl == 1)
    fix_hint_rate = fix_hint_present / n_pos if n_pos > 0 else 0.0

    # ── location 指标 ──
    loc_metrics = compute_location_acc(results, gold_metas)

    # ── group recall ──
    group_recall = {}
    for grp, rules in GROUP_RULES.items():
        grp_true = [t for t in true_rules if t in rules]
        grp_corr = [t for t, p in zip(true_rules, pred_rules) if t in rules and t == p]
        group_recall[grp] = len(grp_corr) / len(grp_true) if grp_true else 0.0

    # ── consistency（label 与 rule 自洽率）──
    consistent = sum(
        1 for tl, pl, pr in zip(true_labels, pred_labels, pred_rules)
        if not (pl == 1 and pr == "none") and not (pl == 0 and pr not in ("none", "parse_error"))
    )
    consistency_rate = consistent / n

    print(f"\n{'─'*60}")
    print(f"  [{split_name}]  n={n}  parse_errors={parse_errors}")
    print(f"  Label Acc      : {label_acc:.4f}")
    print(f"  Rule Acc       : {rule_acc:.4f}")
    print(f"  Macro-F1       : {macro_f1:.4f}")
    print(f"  Fix Hint Rate  : {fix_hint_rate:.4f}  ({fix_hint_present}/{n_pos} pos)")
    print(f"  Location Acc A : {loc_metrics['location_acc_A']:.4f}"
          f"  ({loc_metrics['a_hit']}/{loc_metrics['a_total']})")
    print(f"  Location Acc B : {loc_metrics['location_acc_B']:.4f}"
          f"  ({loc_metrics['b_hit']}/{loc_metrics['b_total']})")
    print(f"  Consistency    : {consistency_rate:.4f}")

    print(f"\n  Per-Rule F1:")
    for rule in ALL_RULES:
        if rule not in rf1:
            continue
        v = rf1[rule]
        if v["support"] == 0:
            continue
        star = " ★" if v["support"] < 30 else ""
        print(f"    {rule:10s}: F1={v['f1']:.3f}  P={v['precision']:.3f}"
              f"  R={v['recall']:.3f}  sup={v['support']}{star}")

    print(f"\n  Group Recall:")
    for grp, rec in sorted(group_recall.items()):
        print(f"    {grp}: {rec:.3f}")

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "split":            split_name,
            "n":                n,
            "label_acc":        label_acc,
            "rule_acc":         rule_acc,
            "macro_f1":         macro_f1,
            "fix_hint_rate":    fix_hint_rate,
            "location_acc_A":   loc_metrics["location_acc_A"],
            "location_acc_B":   loc_metrics["location_acc_B"],
            "consistency_rate": consistency_rate,
            "group_recall":     group_recall,
            "per_rule_f1":      rf1,
            "parse_errors":     parse_errors,
            "results":          results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  → saved: {out_file}")

    return {
        "label_acc":        label_acc,
        "rule_acc":         rule_acc,
        "macro_f1":         macro_f1,
        "fix_hint_rate":    fix_hint_rate,
        "location_acc_A":   loc_metrics["location_acc_A"],
        "consistency_rate": consistency_rate,
        "group_recall":     group_recall,
    }


# ─── 运行评估 ─────────────────────────────────────────────────────────────────

eval_dir     = Path(trained_path) / "eval_v4"
val_metrics  = evaluate(val_raw,  "Val",  str(eval_dir / "val_results.json"))
test_metrics = evaluate(test_raw, "Test", str(eval_dir / "test_results.json"))

print("\n" + "=" * 70)
print("Final Summary")
print("=" * 70)
for split_name, m in [("Val", val_metrics), ("Test", test_metrics)]:
    print(f"\n  [{split_name}]")
    print(f"    Label Acc       : {m['label_acc']:.4f}")
    print(f"    Rule Acc        : {m['rule_acc']:.4f}")
    print(f"    Macro-F1        : {m['macro_f1']:.4f}")
    print(f"    Fix Hint Rate   : {m['fix_hint_rate']:.4f}")
    print(f"    Location Acc(A) : {m['location_acc_A']:.4f}")
    print(f"    Consistency     : {m['consistency_rate']:.4f}")
    print(f"    Group Recall    : " +
          "  ".join(f"{g}={v:.2f}" for g, v in sorted(m["group_recall"].items())))

print(f"\nModel : {trained_path}")
print(f"Eval  : {eval_dir}")
print("\n[SFT-v4 complete. Next step: GRPO-v4 from this checkpoint.]")
