#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse, random, sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

import torch
from datasets import Dataset
from tqdm import tqdm

# ─── 参数 ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--sft_model",        type=str,
                    default="YOUR_SFT_MODEL_PATH",
                    )
parser.add_argument("--dataset_dir",      type=str,   required=True)
parser.add_argument("--output_dir",       type=str,
                    default="YOUR_GRPO_OUTPUT_DIR")
parser.add_argument("--max_seq_length",   type=int,   default=2048)
parser.add_argument("--max_new_tokens",   type=int,   default=768)
parser.add_argument("--train_epochs",     type=int,   default=3)
parser.add_argument("--grad_accum",       type=int,   default=4)
parser.add_argument("--learning_rate",    type=float, default=3e-6,
                    )
parser.add_argument("--lora_rank",        type=int,   default=16)
parser.add_argument("--lora_alpha",       type=int,   default=32)
parser.add_argument("--num_generations",  type=int,   default=8)
parser.add_argument("--rare_threshold",   type=int,   default=50)
parser.add_argument("--skip_eval",        action="store_true")
parser.add_argument("--max_eval_samples", type=int,   default=None)
args = parser.parse_args()

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


ALL_RULES = [
    "Rule-1a","Rule-1b","Rule-1c","Rule-1d",
    "Rule-2a","Rule-2b","Rule-2c","Rule-2d",
    "Rule-3a","Rule-3b",
    "Rule-4a","Rule-4b","Rule-4c","Rule-4d","Rule-4e",
    "Rule-5a","Rule-5b","Rule-5c",
    "Rule-6a","Rule-6b",
    "none",
]

RULE_TO_GROUP = {
    "Rule-1a":"G1","Rule-1b":"G1","Rule-1c":"G1","Rule-1d":"G1",
    "Rule-2a":"G2","Rule-2b":"G2","Rule-2c":"G2","Rule-2d":"G2",
    "Rule-3a":"G3","Rule-3b":"G3",
    "Rule-4a":"G4","Rule-4b":"G4","Rule-4c":"G4","Rule-4d":"G4","Rule-4e":"G4",
    "Rule-5a":"G5","Rule-5b":"G5","Rule-5c":"G5",
    "Rule-6a":"G6","Rule-6b":"G6",
    "none":"none",
}

GROUP_RULES = {
    "G1": ["Rule-1a","Rule-1b","Rule-1c","Rule-1d"],
    "G2": ["Rule-2a","Rule-2b","Rule-2c","Rule-2d"],
    "G3": ["Rule-3a","Rule-3b"],
    "G4": ["Rule-4a","Rule-4b","Rule-4c","Rule-4d","Rule-4e"],
    "G5": ["Rule-5a","Rule-5b","Rule-5c"],
    "G6": ["Rule-6a","Rule-6b"],
}

print("="*70)
print("Crypto API Misuse Detection — GRPO v4 (Annotate-Grounded Fix Reward)")
print("="*70)
print(f"  Init model : {args.sft_model}")
print(f"  Dataset    : {args.dataset_dir}")
print(f"  LR         : {args.learning_rate}  LoRA rank: {args.lora_rank}")
print(f"  G (gens)   : {args.num_generations}  World size: {WORLD_SIZE}")
print()
print("  Reward changes vs v3:")
print("    v3: fix_code non-empty → +0.3 (unconditional)")
print("    v4: fix_code passes annotate() → +2.0 (verified effective)")
print("        fix_code non-empty but still vuln → +0.1 (attempt credit)")


sys.path.insert(0, str(Path(__file__).parent))
try:
    from annotate import annotate as _annotate_fn
    ANNOTATE_AVAILABLE = True
    print("\n  [OK] annotate() loaded — executable fix reward ENABLED")
except ImportError as e:
    ANNOTATE_AVAILABLE = False
    print(f"\n  [WARN] annotate() not available ({e}) — fallback to v3 reward")

def check_fix_valid(fix_code: str, gold_rule: str) -> Optional[bool]:
    """
    用 annotate() 验证 fix_code 是否真正消除了违规。
    返回:
      True  — 修复有效（annotate 不再报 gold_rule）
      False — 修复无效（annotate 仍报 gold_rule）
      None  — annotate 不可用（fallback）
    """
    if not ANNOTATE_AVAILABLE:
        return None
    try:
        result = _annotate_fn(fix_code)
        after_rule = result["primary_rule"] if result else None
        return after_rule != gold_rule
    except Exception:
        return None


def load_jsonl(path) -> List[Dict]:
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

print(f"\n  Train: {len(train_raw)}  Val: {len(val_raw)}  Test: {len(test_raw)}")


rule_count = Counter(s["_meta"]["primary_rule"] for s in train_raw)
rare_rules = {r for r, c in rule_count.items() if c < args.rare_threshold}
print(f"\n  Rare rules (< {args.rare_threshold} samples, +1.5 bonus): {sorted(rare_rules)}")


def apply_weighted_sampling(data: List[Dict]) -> List[Dict]:
    out = []
    for s in data:
        rule = s["_meta"]["primary_rule"]
        copies = 4 if rule in rare_rules else 1
        out.extend([s] * copies)
    random.shuffle(out)
    return out

train_weighted = apply_weighted_sampling(train_raw)
print(f"  After weighting: {len(train_raw)} → {len(train_weighted)} samples")


from unsloth import FastLanguageModel, PatchFastRL
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

PatchFastRL("GRPO", FastLanguageModel)

BASE_MODEL = "/storage/yaohongyu/ZWTGRPO/Qwen2.5-7B-Instruct"
print(f"\nLoading base model: {BASE_MODEL} (4bit)")
print(f"Loading LoRA adapter: {args.sft_model}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = args.max_seq_length,
    dtype          = None,
    load_in_4bit   = True,
)
model = PeftModel.from_pretrained(model, args.sft_model)
model.enable_input_require_grads()
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)
model.gradient_checkpointing_enable()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
tokenizer.padding_side = "left"
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


def make_prompt(sample: Dict) -> str:
    conv = sample["conversations"]
    msgs = [conv[0], conv[1]]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)

def build_grpo_dataset(raw: List[Dict]) -> Dataset:
    rows = []
    for s in raw:
        rows.append({
            "prompt":     make_prompt(s),
            "gold_rule":  s["_meta"]["primary_rule"],
            "gold_label": s["_meta"]["label"],
        })
    return Dataset.from_list(rows)

train_ds = build_grpo_dataset(train_weighted)
val_ds   = build_grpo_dataset(val_raw[:min(200, len(val_raw))])
print(f"  GRPO train dataset: {len(train_ds)}")

# ─── Reward Function ──────────────────────────────────────────────────────────

def parse_answer(text: str) -> Dict:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1).strip())
        if not isinstance(obj, dict):
            return {}
        if "primary_rule" in obj and not isinstance(obj["primary_rule"], str):
            obj["primary_rule"] = str(obj["primary_rule"])
        return obj
    except Exception:
        rm = re.search(r'"primary_rule"\s*:\s*"([^"]+)"', m.group(1))
        lm = re.search(r'"label"\s*:\s*(\d)',             m.group(1))
        if rm or lm:
            return {
                "primary_rule": rm.group(1) if rm else "parse_error",
                "label":        int(lm.group(1)) if lm else -1,
            }
        return {}


def has_fix_code(parsed: Dict) -> bool:
    fc = parsed.get("fix_hint", "")
    return isinstance(fc, str) and len(fc.strip()) > 10


def compute_reward(
    completion: str,
    gold_rule:  str,
    gold_label: int,
) -> float:
    
    r = 0.0

    # ── 格式奖励 ──────────────────────────────────────────────────────────
    has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", completion, re.DOTALL))
    has_answer    = bool(re.search(r"<answer>.*?</answer>",    completion, re.DOTALL))
    if has_reasoning:
        r += 0.1
    if has_answer:
        r += 0.1
    if not has_answer:
        return r

    # ── 解析 ────────────────────────────────────────────────────────────────
    parsed = parse_answer(completion)
    if not parsed or "primary_rule" not in parsed:
        return r

    pred_rule = parsed.get("primary_rule", "parse_error")
    inferred_pred_label = 0 if pred_rule in ("none", "parse_error") else 1

    # ── 检测方向 ──────────────────────────────────────────────────────────
    if inferred_pred_label != gold_label:
        r += -0.5 if (gold_label == 1) else -0.2
        return r

    gold_grp = RULE_TO_GROUP.get(gold_rule, "none")
    pred_grp = RULE_TO_GROUP.get(pred_rule, "none")

    if pred_rule == gold_rule:
        # 规则精确匹配
        r += 3.0 + (1.5 if gold_rule in rare_rules else 0.0)

        # ── Fix Reward（v4 核心）────────────────────────────────────────
        if gold_label == 1 and has_fix_code(parsed):
            fix_code = parsed["fix_hint"]
            valid = check_fix_valid(fix_code, gold_rule)
            if valid is True:
                # annotate() 确认修复有效：高奖励
                r += 2.0
            elif valid is False:
                # 生成了 fix_code 但仍触发规则：低奖励（鼓励尝试）
                r += 0.1
            else:
                # annotate 不可用（fallback）：按 v3 给 +0.3
                r += 0.3

    elif gold_grp == pred_grp:
        # 组内混淆：部分分
        r += 1.0
    else:
        # 检测到有误用但规则组错误
        r += 0.3

    return r


def reward_fn(completions: List[str], prompts: List[str] = None,
              gold_rule: List[str] = None, gold_label: List[int] = None,
              **kwargs) -> List[float]:
    rewards = []
    for i, comp in enumerate(completions):
        gr = gold_rule[i]  if gold_rule  else "none"
        gl = gold_label[i] if gold_label else 0
        rewards.append(compute_reward(comp, gr, gl))
    return rewards

# ─── GRPO 训练 ────────────────────────────────────────────────────────────────

from trl import GRPOTrainer, GRPOConfig

n_train   = len(train_ds)
max_steps = n_train // (args.num_generations * args.grad_accum) * args.train_epochs

print(f"\nTraining plan:")
print(f"  Samples={n_train}  G={args.num_generations}  GPUs={WORLD_SIZE}")
print(f"  Max steps={max_steps}  epochs={args.train_epochs}")

grpo_config = GRPOConfig(
    num_generations             = args.num_generations,
    max_completion_length       = args.max_new_tokens,
    temperature                 = 0.3,
    per_device_train_batch_size = args.num_generations,
    gradient_accumulation_steps = args.grad_accum,
    num_train_epochs            = args.train_epochs,
    max_steps                   = max_steps,
    learning_rate               = args.learning_rate,
    bf16                        = torch.cuda.is_bf16_supported(),
    fp16                        = not torch.cuda.is_bf16_supported(),
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    logging_steps               = max(1, max_steps // 20),
    save_steps                  = max(1, max_steps // 4),
    save_total_limit            = 2,
    output_dir                  = args.output_dir,
    report_to                   = "tensorboard",
    seed                        = 42,
    use_vllm                    = False,
    beta                        = 0.01,
)

trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = reward_fn,
    args             = grpo_config,
    train_dataset    = train_ds,
    eval_dataset     = val_ds,
)

print("\n" + "="*70)
print("GRPO v4 Training started...")
print("="*70 + "\n")
trainer.train()

# ─── 保存 ────────────────────────────────────────────────────────────────────

out_final = args.output_dir + "_final"
del trainer
import gc; gc.collect()
model.save_pretrained(out_final)
tokenizer.save_pretrained(out_final)
print(f"\nGRPO v4 LoRA saved → {out_final}")

# ─── 评估 ────────────────────────────────────────────────────────────────────

if args.skip_eval:
    print("Skipping eval (--skip_eval).")
    exit(0)

if LOCAL_RANK != 0:
    exit(0)

print("\n" + "="*70)
print("Evaluation Phase (rank 0 only)")
print("="*70)

FastLanguageModel.for_inference(model)
tokenizer.padding_side = "right"


def extract_answer(text: str) -> Dict:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(1).strip())
        if not isinstance(obj, dict):
            return {}
        if "primary_rule" in obj and not isinstance(obj["primary_rule"], str):
            obj["primary_rule"] = str(obj["primary_rule"])
        return obj
    except Exception:
        rm = re.search(r'"primary_rule"\s*:\s*"([^"]+)"', m.group(1))
        lm = re.search(r'"label"\s*:\s*(\d)',              m.group(1))
        return {
            "primary_rule": rm.group(1) if rm else "parse_error",
            "label":        int(lm.group(1)) if lm else -1,
        }


def compute_rule_f1(true_rules: List[str], pred_rules: List[str]):
    results = {}
    for rule in sorted(set(true_rules + pred_rules), key=str):
        tp = sum(1 for t, p in zip(true_rules, pred_rules) if t == rule and p == rule)
        fp = sum(1 for t, p in zip(true_rules, pred_rules) if t != rule and p == rule)
        fn = sum(1 for t, p in zip(true_rules, pred_rules) if t == rule and p != rule)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        results[rule] = {"precision": prec, "recall": rec, "f1": f1,
                         "support": sum(1 for t in true_rules if t == rule)}
    return results


def evaluate(raw_data: List[Dict], split_name: str, out_file: str):
    if args.max_eval_samples and len(raw_data) > args.max_eval_samples:
        raw_data = random.sample(raw_data, args.max_eval_samples)

    true_rules, pred_rules = [], []
    true_labels, pred_labels = [], []
    parse_errors = 0
    fix_present  = 0
    fix_valid_count = 0
    results = []

    print(f"\nEvaluating {split_name} ({len(raw_data)} samples)...")
    for s in tqdm(raw_data, desc=split_name):
        conv = s["conversations"]
        msgs = [conv[0], conv[1]]

        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids      = inputs,
                max_new_tokens = args.max_new_tokens,
                temperature    = 0.1,
                top_p          = 0.9,
                do_sample      = True,
                pad_token_id   = tokenizer.pad_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )
        gen  = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(gen)

        meta    = s["_meta"]
        t_rule  = meta["primary_rule"]
        t_label = meta["label"]
        p_rule  = pred.get("primary_rule", "parse_error")
        p_label = 0 if p_rule in ("none", "parse_error") else 1

        if p_rule == "parse_error":
            parse_errors += 1

        fix_code = pred.get("fix_hint", "")
        fix_has  = isinstance(fix_code, str) and len(fix_code.strip()) > 10
        fix_ok   = False
        if t_label == 1 and fix_has:
            fix_present += 1
            v = check_fix_valid(fix_code, t_rule)
            if v is True:
                fix_valid_count += 1
                fix_ok = True

        true_rules.append(t_rule);   pred_rules.append(p_rule)
        true_labels.append(t_label); pred_labels.append(p_label)
        results.append({
            "true_rule": t_rule, "pred_rule": p_rule,
            "true_label": t_label, "pred_label": p_label,
            "correct_rule": t_rule == p_rule,
            "has_fix": fix_has,
            "fix_valid": fix_ok,
            "fix_code": fix_code,
            "original_code": s["conversations"][1]["content"][-800:],
            "response": gen[:600],
        })

    n = len(results)
    label_acc = sum(1 for tl, pl in zip(true_labels, pred_labels) if tl == pl) / n
    rule_acc  = sum(1 for tr, pr in zip(true_rules,  pred_rules)  if tr == pr) / n
    rf1       = compute_rule_f1(true_rules, pred_rules)
    pos_f1s   = [v["f1"] for r, v in rf1.items() if r != "none" and v["support"] > 0]
    macro_f1  = sum(pos_f1s) / len(pos_f1s) if pos_f1s else 0.0

    n_pos       = sum(1 for tl in true_labels if tl == 1)
    fix_rate    = fix_present     / n_pos if n_pos > 0 else 0.0
    fix_validity= fix_valid_count / n_pos if n_pos > 0 else 0.0

    group_recall = {}
    for grp, rules in GROUP_RULES.items():
        grp_true = [t for t in true_rules if t in rules]
        grp_corr = [t for t, p in zip(true_rules, pred_rules) if t in rules and t == p]
        group_recall[grp] = len(grp_corr) / len(grp_true) if grp_true else 0.0

    print(f"\n{'─'*60}")
    print(f"  [{split_name}] n={n}  parse_errors={parse_errors}")
    print(f"  Label Acc    : {label_acc:.3f}")
    print(f"  Rule Acc     : {rule_acc:.3f}")
    print(f"  Macro-F1     : {macro_f1:.3f}  (pos rules only)")
    print(f"  Fix Rate     : {fix_rate:.3f}  ({fix_present}/{n_pos} pos samples with fix_code)")
    print(f"  Fix Validity : {fix_validity:.3f}  ({fix_valid_count}/{n_pos} verified by annotate)")
    print(f"\n  Per-Rule F1:")
    for rule in ALL_RULES:
        if rule not in rf1:
            continue
        v = rf1[rule]
        if v["support"] == 0 and rule == "none":
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
            "split": split_name, "n": n,
            "label_acc": label_acc, "rule_acc": rule_acc,
            "macro_f1": macro_f1, "fix_rate": fix_rate,
            "fix_validity": fix_validity,
            "group_recall": group_recall, "per_rule_f1": rf1,
            "parse_errors": parse_errors, "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  → saved: {out_file}")
    return {"label_acc": label_acc, "rule_acc": rule_acc,
            "macro_f1": macro_f1, "fix_rate": fix_rate,
            "fix_validity": fix_validity, "group_recall": group_recall}


eval_dir = Path(out_final) / "eval_v4"
val_m    = evaluate(val_raw,  "Val",  str(eval_dir / "val_results.json"))
test_m   = evaluate(test_raw, "Test", str(eval_dir / "test_results.json"))

print("\n" + "="*70)
print("Final Summary")
print("="*70)
for split, m in [("Val", val_m), ("Test", test_m)]:
    print(f"\n  [{split}]")
    print(f"    Label Acc    : {m['label_acc']:.3f}")
    print(f"    Rule Acc     : {m['rule_acc']:.3f}")
    print(f"    Macro-F1     : {m['macro_f1']:.3f}")
    print(f"    Fix Rate     : {m['fix_rate']:.3f}")
    print(f"    Fix Validity : {m['fix_validity']:.3f}  ← annotate()-verified")
    print(f"    Group: " + "  ".join(f"{g}={v:.2f}" for g, v in sorted(m["group_recall"].items())))

print(f"\nModel: {out_final}")
print(f"Eval : {eval_dir}")
