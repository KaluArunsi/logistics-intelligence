#!/usr/bin/env python3
"""Package a trained Toji LoRA adapter for Ollama deployment.

This script writes a Modelfile that references:
- a base Ollama model tag (e.g. qwen2.5:7b)
- a local adapter directory from train_toji_qlora.py output

Optionally executes `ollama create`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are Toji, the product-native logistics intelligence model. "
    "Respect deterministic schema gates, be explicit about confidence, and stay grounded to "
    "report context, manifest semantics, and user-provided intake provenance."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model-tag", default="qwen2.5:7b")
    p.add_argument("--adapter-dir", required=True, help="Path to PEFT adapter directory.")
    p.add_argument("--toji-tag", default="toji:qwen2.5-7b-v1")
    p.add_argument("--out-modelfile", default="outputs/toji_qlora/Modelfile")
    p.add_argument("--system-prompt-file", default="", help="Optional file with SYSTEM prompt text.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--num-predict", type=int, default=256)
    p.add_argument("--create", action="store_true", help="Run `ollama create` after writing Modelfile.")
    return p


def _read_system_prompt(path: str) -> str:
    if not path:
        return DEFAULT_SYSTEM_PROMPT
    p = Path(path).resolve()
    if not p.exists():
        raise SystemExit(f"System prompt file not found: {p}")
    txt = p.read_text().strip()
    return txt or DEFAULT_SYSTEM_PROMPT


def _validate_adapter_dir(path: Path) -> None:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Adapter directory not found: {path}")
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing adapter_config.json in: {path}")
    has_weights = any((path / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not has_weights:
        raise SystemExit(f"Missing adapter_model.safetensors/bin in: {path}")


def main() -> int:
    args = _build_parser().parse_args()
    adapter_dir = Path(args.adapter_dir).resolve()
    _validate_adapter_dir(adapter_dir)
    system_prompt = _read_system_prompt(args.system_prompt_file)

    modelfile_path = Path(args.out_modelfile).resolve()
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)

    modelfile = "\n".join(
        [
            f"FROM {args.base_model_tag}",
            f"ADAPTER {adapter_dir}",
            f'PARAMETER temperature {args.temperature}',
            f'PARAMETER top_p {args.top_p}',
            f'PARAMETER num_predict {args.num_predict}',
            f'SYSTEM """{system_prompt}"""',
            "",
        ]
    )
    modelfile_path.write_text(modelfile)

    metadata = {
        "generated_at": _now_iso(),
        "toji_tag": args.toji_tag,
        "base_model_tag": args.base_model_tag,
        "adapter_dir": str(adapter_dir),
        "modelfile": str(modelfile_path),
        "create_executed": bool(args.create),
    }

    if args.create:
        cmd = ["ollama", "create", args.toji_tag, "-f", str(modelfile_path)]
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        metadata["create_returncode"] = proc.returncode
        metadata["create_stdout"] = proc.stdout[-4000:]
        metadata["create_stderr"] = proc.stderr[-4000:]
        if proc.returncode != 0:
            print(json.dumps(metadata, indent=2))
            raise SystemExit(proc.returncode)

    metadata_path = modelfile_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
