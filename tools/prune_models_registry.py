#!/usr/bin/env python3
"""Prune models registry to align with latest run snapshots or active specs.

This tool updates `models/<industry>/specs/models.json` while keeping an archive
copy and a structured prune report. It does not delete model artifact files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _latest_run_id(base: Path, industry: str) -> str:
    run_root = base / "reports" / "runs" / industry
    runs = sorted(p.name for p in run_root.glob(f"{industry}_*") if p.is_dir())
    if not runs:
        raise RuntimeError(f"No run directories found for industry '{industry}'")
    return runs[-1]


def _keep_sets_from_latest_run(base: Path, industry: str, run_id: str) -> tuple[set[str], set[str]]:
    reports_dir = base / "reports" / "runs" / industry / run_id / "reports"
    l0_results = _read_json(reports_dir / "l0_results.json")
    l1_results = _read_json(reports_dir / "l1_results.json")

    keep_l0_dataset_ids = {
        dataset_id
        for dataset_id, row in l0_results.items()
        if isinstance(row, dict) and row.get("status") == "trained" and bool(row.get("passed"))
    }
    keep_l1_categories = {
        category
        for category, row in l1_results.items()
        if isinstance(row, dict) and bool(row.get("passed"))
    }
    if not keep_l0_dataset_ids:
        raise RuntimeError(
            f"Latest run '{run_id}' has no passed L0 workers; refusing prune in latest-run mode."
        )
    return keep_l0_dataset_ids, keep_l1_categories


def _keep_sets_from_active_specs(base: Path, industry: str) -> tuple[set[str], set[str]]:
    specs = _read_json(base / "config" / "industries" / industry / f"{industry}_dataset_specs.json")
    categories_yaml = base / "config" / "industries" / industry / "categories.yaml"
    try:
        import yaml
        categories_cfg = yaml.safe_load(categories_yaml.read_text()) or {}
    except Exception:
        categories_cfg = {}

    datasets = specs.get("datasets", {}) or {}
    keep_l0_dataset_ids = {
        dataset_id
        for dataset_id, row in datasets.items()
        if not bool((row or {}).get("skip_training", False))
        and bool(((row or {}).get("training", {}) or {}).get("worker_enabled", True))
    }
    keep_l1_categories = set(((categories_cfg.get("categories", {}) or {}).keys()))
    if not keep_l0_dataset_ids:
        raise RuntimeError("Active-spec mode resolved zero L0 datasets; refusing prune.")
    return keep_l0_dataset_ids, keep_l1_categories


def prune_registry(
    base: Path,
    industry: str,
    mode: str,
    run_id: str | None,
    apply: bool,
) -> dict[str, Any]:
    models_path = base / "models" / industry / "specs" / "models.json"
    models = _read_json(models_path)

    if mode == "latest-run":
        resolved_run = run_id or _latest_run_id(base, industry)
        keep_l0, keep_l1 = _keep_sets_from_latest_run(base, industry, resolved_run)
    else:
        resolved_run = None
        keep_l0, keep_l1 = _keep_sets_from_active_specs(base, industry)

    kept: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    removed_reasons: dict[str, str] = {}

    for model_id, row in models.items():
        level = row.get("level")
        if level == 0:
            dataset_id = row.get("dataset_id")
            if dataset_id in keep_l0:
                kept[model_id] = row
            else:
                removed[model_id] = row
                removed_reasons[model_id] = "l0_not_in_keep_set"
        elif level == 1:
            category = row.get("category")
            if category in keep_l1:
                kept[model_id] = row
            else:
                removed[model_id] = row
                removed_reasons[model_id] = "l1_not_in_keep_set"
        else:
            # Unknown levels are dropped by default for consistency.
            removed[model_id] = row
            removed_reasons[model_id] = "unknown_level"

    result = {
        "industry": industry,
        "mode": mode,
        "run_id": resolved_run,
        "source_models": len(models),
        "kept_models": len(kept),
        "removed_models": len(removed),
        "kept_l0": sum(1 for v in kept.values() if v.get("level") == 0),
        "kept_l1": sum(1 for v in kept.values() if v.get("level") == 1),
        "removed_l0": sum(1 for v in removed.values() if v.get("level") == 0),
        "removed_l1": sum(1 for v in removed.values() if v.get("level") == 1),
    }

    if not apply:
        result["status"] = "dry-run"
        return result

    ts = _now_ts()
    archive_dir = models_path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_models_path = archive_dir / f"models_pre_prune_{ts}.json"
    prune_report_path = archive_dir / f"prune_report_{ts}.json"

    _write_json(archive_models_path, models)
    _write_json(models_path, kept)
    _write_json(
        prune_report_path,
        {
            "summary": result,
            "removed_model_ids": sorted(removed.keys()),
            "removed_reasons": removed_reasons,
            "archive_models_path": str(archive_models_path),
            "pruned_models_path": str(models_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    result.update(
        {
            "status": "applied",
            "archive_models_path": str(archive_models_path),
            "prune_report_path": str(prune_report_path),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune models registry.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--industry", required=True)
    parser.add_argument(
        "--mode",
        default="latest-run",
        choices=["latest-run", "active-spec"],
        help="latest-run keeps only models from latest passed run snapshot.",
    )
    parser.add_argument("--run-id", default=None, help="Run id override for latest-run mode.")
    parser.add_argument("--apply", action="store_true", help="Write changes. Default is dry-run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    result = prune_registry(
        base=base,
        industry=args.industry,
        mode=args.mode,
        run_id=args.run_id,
        apply=args.apply,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

