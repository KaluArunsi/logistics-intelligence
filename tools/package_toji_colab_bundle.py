#!/usr/bin/env python3
"""Package Toji fine-tune assets for easy Colab GPU upload.

Creates a single tar.gz bundle containing the dataset splits and optional stats file.
This is for transfer convenience only; training compute still runs in Colab.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _must_exist(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default=".", help="Repository root.")
    p.add_argument("--dataset-dir", default="data/toji_finetune", help="Dataset directory.")
    p.add_argument("--prefix", default="toji_v2", help="Dataset prefix.")
    p.add_argument("--out-dir", default="dist/colab", help="Output directory.")
    args = p.parse_args()

    base = Path(args.base).resolve()
    ds_dir = (base / args.dataset_dir).resolve()
    out_dir = (base / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        ds_dir / f"{args.prefix}_messages_train.jsonl",
        ds_dir / f"{args.prefix}_messages_val.jsonl",
        ds_dir / f"{args.prefix}_messages_test.jsonl",
    ]
    optional = [
        ds_dir / f"{args.prefix}_stats.json",
        (base / "requirements-toji-qlora.txt").resolve(),
        (base / "tools" / "train_toji_qlora.py").resolve(),
        (base / "tools" / "eval_toji_qlora.py").resolve(),
    ]

    for path in required:
        _must_exist(path)

    bundle_name = f"{args.prefix}_colab_bundle_{_now_stamp()}.tar.gz"
    bundle_path = out_dir / bundle_name
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in required:
            tar.add(path, arcname=f"dataset/{path.name}")
        for path in optional:
            if path.exists():
                subdir = "dataset" if path.parent == ds_dir else "repo_assets"
                tar.add(path, arcname=f"{subdir}/{path.name}")

    summary = {
        "bundle_path": str(bundle_path),
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": _sha256(bundle_path),
        "included_required": [str(p) for p in required],
        "included_optional": [str(p) for p in optional if p.exists()],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
