#!/usr/bin/env python3
"""Verify downloaded Toji training artifacts from Colab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _must(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        help="Path to run directory (e.g. outputs/toji_qlora/toji_qwen25_7b_colab_v1).",
    )
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    _must(run_dir, "run directory")

    adapter = run_dir / "adapter"
    tokenizer = run_dir / "tokenizer"
    plan = run_dir / "training_plan.json"
    summary = run_dir / "training_summary.json"

    _must(adapter, "adapter directory")
    _must(tokenizer, "tokenizer directory")
    _must(plan, "training_plan.json")
    _must(summary, "training_summary.json")
    _must(adapter / "adapter_config.json", "adapter_config.json")

    has_weights = (adapter / "adapter_model.safetensors").exists() or (adapter / "adapter_model.bin").exists()
    if not has_weights:
        raise SystemExit(f"Missing adapter weights in: {adapter}")

    with open(summary) as f:
        summary_payload = json.load(f)

    out = {
        "ok": True,
        "run_dir": str(run_dir),
        "adapter_dir": str(adapter),
        "tokenizer_dir": str(tokenizer),
        "train_samples": summary_payload.get("train_samples"),
        "val_samples": summary_payload.get("val_samples"),
        "quantization_mode": summary_payload.get("quantization_mode"),
        "base_model": summary_payload.get("base_model"),
        "next_step": (
            "PYTHONPATH=. .venv/bin/python tools/package_toji_ollama_adapter.py "
            f"--base-model-tag qwen2.5:7b --adapter-dir {adapter} --toji-tag toji:qwen2.5-7b-v1 --create"
        ),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
