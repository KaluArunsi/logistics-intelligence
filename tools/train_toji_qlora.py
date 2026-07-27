#!/usr/bin/env python3
"""Train Toji with LoRA/QLoRA on messages JSONL produced by build_toji_finetune_dataset.

Design:
- Supports `quantization=auto|qlora|lora`.
- `auto` picks QLoRA when CUDA + bitsandbytes are available, else LoRA.
- Uses prompt-masked labels so loss applies only to assistant outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    if role in {"assistant", "user", "system"}:
        return role
    return "user"


def _render_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for row in messages:
        role = _norm_role(row.get("role"))
        content = str(row.get("content") or "").strip()
        parts.append(f"<|{role}|>\n{content}\n")
    return "".join(parts).strip() + "\n"


def _deterministic_sample(rows: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return rows
    scored: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        token = _hash_text(f"{seed}:{json.dumps(row, sort_keys=True, ensure_ascii=True)}")
        scored.append((token, row))
    scored.sort(key=lambda x: x[0])
    return [row for _, row in scored[:cap]]


@dataclass
class Sample:
    sample_id: str
    task_type: str
    prompt: str
    answer: str


def _build_samples(path: Path, max_samples: int, seed: int) -> list[Sample]:
    rows = list(_iter_jsonl(path))
    rows = _deterministic_sample(rows, max_samples, seed=seed)

    out: list[Sample] = []
    for row in rows:
        sample_id = str(row.get("id") or "")[:64]
        task_type = str(row.get("task_type") or "unknown")
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            continue

        last_assistant_idx = -1
        for idx, msg in enumerate(msgs):
            if isinstance(msg, dict) and _norm_role(msg.get("role")) == "assistant":
                last_assistant_idx = idx
        if last_assistant_idx < 0:
            continue

        assistant_msg = msgs[last_assistant_idx]
        if not isinstance(assistant_msg, dict):
            continue
        answer = str(assistant_msg.get("content") or "").strip()
        if not answer:
            continue

        prompt_msgs = [m for m in msgs[:last_assistant_idx] if isinstance(m, dict)]
        prompt_msgs.append({"role": "assistant", "content": ""})
        prompt = _render_messages(prompt_msgs)

        if not sample_id:
            sample_id = _hash_text(prompt + "\n" + answer)[:24]
        out.append(
            Sample(
                sample_id=sample_id,
                task_type=task_type,
                prompt=prompt,
                answer=answer,
            )
        )
    return out


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _detect_quantization_mode(requested: str) -> tuple[str, dict[str, Any]]:
    details: dict[str, Any] = {"requested": requested}
    if requested not in {"auto", "qlora", "lora"}:
        raise ValueError("quantization must be one of: auto, qlora, lora")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for training. Install dependencies first.") from exc

    has_cuda = bool(torch.cuda.is_available())
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    details["has_cuda"] = has_cuda
    details["has_mps"] = has_mps

    has_bnb = False
    try:
        import bitsandbytes  # noqa: F401

        has_bnb = True
    except Exception:
        has_bnb = False
    details["has_bitsandbytes"] = has_bnb

    if requested == "qlora":
        if not has_cuda or not has_bnb:
            raise RuntimeError(
                "QLoRA requires CUDA + bitsandbytes. On Apple Silicon, use --quantization lora."
            )
        return "qlora", details
    if requested == "lora":
        return "lora", details

    if has_cuda and has_bnb:
        return "qlora", details
    return "lora", details


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", default="data/toji_finetune/toji_v2_messages_train.jsonl")
    p.add_argument("--val-file", default="data/toji_finetune/toji_v2_messages_val.jsonl")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output-dir", default="outputs/toji_qlora")
    p.add_argument("--run-name", default="toji_qwen25_7b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=0, help="0 means all.")
    p.add_argument("--max-val-samples", type=int, default=0, help="0 means all.")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--quantization", default="auto", help="auto|qlora|lora")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--plan-only", action="store_true", help="Load and summarize datasets without training.")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    random.seed(args.seed)

    train_path = Path(args.train_file).resolve()
    val_path = Path(args.val_file).resolve()
    if not train_path.exists():
        raise SystemExit(f"Train file not found: {train_path}")

    train_samples = _build_samples(train_path, max_samples=args.max_train_samples, seed=args.seed)
    val_samples: list[Sample] = []
    if val_path.exists():
        val_samples = _build_samples(val_path, max_samples=args.max_val_samples, seed=args.seed + 1)

    mode, mode_details = _detect_quantization_mode(args.quantization)

    plan = {
        "generated_at": _now_iso(),
        "run_name": args.run_name,
        "base_model": args.base_model,
        "quantization_mode": mode,
        "quantization_details": mode_details,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "max_seq_len": args.max_seq_len,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    run_root = Path(args.output_dir).resolve() / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))

    print(json.dumps(plan, indent=2))
    if args.plan_only:
        return 0

    if not train_samples:
        raise SystemExit("No train samples found after parsing.")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise SystemExit(
            "Missing training dependencies. Install from requirements-toji-qlora.txt before running training."
        ) from exc

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    quant_cfg = None
    device_map: Optional[str] = None
    optim = "adamw_torch"
    if mode == "qlora":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = "auto"
        optim = "paged_adamw_8bit"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise SystemExit("Tokenizer has no pad/eos/unk token; cannot continue.")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        quantization_config=quant_cfg,
        device_map=device_map,
    )
    if mode == "qlora":
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_split_csv(args.lora_target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id

    class PromptMaskedDataset(torch.utils.data.Dataset):
        def __init__(self, samples: list[Sample], max_len: int):
            self.samples = samples
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            row = self.samples[idx]
            prompt_ids = tokenizer(row.prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row.answer, add_special_tokens=False)["input_ids"] + [eos_id]

            if len(answer_ids) >= self.max_len:
                answer_ids = answer_ids[: max(1, self.max_len - 1)] + [eos_id]

            prompt_budget = max(0, self.max_len - len(answer_ids))
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

            input_ids = prompt_ids + answer_ids
            labels = ([-100] * len(prompt_ids)) + answer_ids
            attn = [1] * len(input_ids)

            return {
                "input_ids": input_ids,
                "attention_mask": attn,
                "labels": labels,
            }

    class Collator:
        def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(x["input_ids"]) for x in batch)
            input_ids = []
            attn = []
            labels = []
            for row in batch:
                pad = max_len - len(row["input_ids"])
                input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
                attn.append(row["attention_mask"] + [0] * pad)
                labels.append(row["labels"] + ([-100] * pad))
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    train_ds = PromptMaskedDataset(train_samples, max_len=args.max_seq_len)
    val_ds = PromptMaskedDataset(val_samples, max_len=args.max_seq_len) if val_samples else None

    train_args_kwargs = {
        "output_dir": str(run_root / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_steps": args.eval_steps if val_ds is not None else None,
        "load_best_model_at_end": bool(val_ds),
        "metric_for_best_model": "eval_loss" if val_ds is not None else None,
        "greater_is_better": False if val_ds is not None else None,
        "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        "fp16": bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        "optim": optim,
        "report_to": [],
        "seed": args.seed,
        "remove_unused_columns": False,
    }

    # Transformers changed this name in newer releases.
    eval_strategy_value = "steps" if val_ds is not None else "no"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        train_args_kwargs["evaluation_strategy"] = eval_strategy_value
    elif "eval_strategy" in ta_params:
        train_args_kwargs["eval_strategy"] = eval_strategy_value
    else:
        raise RuntimeError("TrainingArguments is missing both evaluation_strategy and eval_strategy.")

    train_args = TrainingArguments(**train_args_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if val_ds is not None else {}

    adapter_dir = run_root / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(run_root / "tokenizer")

    summary = {
        "completed_at": _now_iso(),
        "run_name": args.run_name,
        "base_model": args.base_model,
        "quantization_mode": mode,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_metrics": dict(train_result.metrics),
        "eval_metrics": eval_metrics,
        "adapter_dir": str(adapter_dir),
        "tokenizer_dir": str(run_root / "tokenizer"),
    }
    (run_root / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
