#!/usr/bin/env python3
"""
Retarget low-value primary targets for active trucking_delivery workers.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


LOW_VALUE_EXACT = {
    "year",
    "period",
    "series_no",
    "row_labels",
    "price_base",
    "unit_mult",
    "decimal",
    "decimals",
    "ranking",
    "rank",
    "index",
    "dot_number",
}

LOW_VALUE_PATTERNS = [
    re.compile(r".*_ranking$"),
    re.compile(r"^rank_.*"),
    re.compile(r"^row_.*"),
    re.compile(r".*_label(s)?$"),
    re.compile(r"^label(s)?$"),
]

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _is_low_value(name: str) -> bool:
    token = str(name).strip().lower()
    if token in LOW_VALUE_EXACT:
        return True
    if token in {"id", "row_id", "record_id", "uuid", "uid"}:
        return True
    if token.endswith("_id"):
        return True
    return any(p.match(token) for p in LOW_VALUE_PATTERNS)


def _build_category_tokens(industry_cfg: dict[str, Any], categories_cfg: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for cat, cfg in (industry_cfg.get("categories", {}) or {}).items():
        for name in (cfg.get("typical_targets") or []):
            for tok in re.split(r"[^a-z0-9]+", str(name).lower()):
                if tok:
                    out[cat].add(tok)
    for cat, cfg in (categories_cfg.get("categories", {}) or {}).items():
        for kw in (cfg.get("keywords") or []):
            for tok in re.split(r"[^a-z0-9]+", str(kw).lower()):
                if tok:
                    out[cat].add(tok)
    return out


def _best_target(
    sample: pl.DataFrame,
    category: str,
    category_tokens: dict[str, set[str]],
) -> tuple[str | None, str | None]:
    tokens = category_tokens.get(category, set())
    best: tuple[float, str, str] | None = None
    for name in sample.columns:
        lname = name.lower()
        if _is_low_value(lname):
            continue
        dtype = sample.schema.get(name)
        if dtype not in NUMERIC_DTYPES:
            continue
        nunique = int(sample.get_column(name).n_unique())
        if nunique < 4:
            continue
        score = 0.0
        score += 2.0
        if lname in {"obs_value", "value", "figure", "count", "hours", "delay", "cost", "revenue", "capacity", "ton"}:
            score += 2.0
        if any(tok and tok in lname for tok in tokens):
            score += 1.5
        if re.search(r"(delay|dwell|wait|volume|ton|value|cost|yield|margin|capacity|util|incident|injury|claim|temp|fuel|speed|flow)", lname):
            score += 1.0
        if re.search(r"(lat|lon|longitude|latitude|postcode|zip|fips|code$)", lname):
            score -= 3.0
        task_type = "classification" if nunique <= 20 else "regression"
        if best is None or score > best[0]:
            best = (score, name, task_type)
    if not best or best[0] < 1.0:
        return None, None
    return best[1], best[2]


def run() -> dict[str, Any]:
    base = Path(__file__).resolve().parents[1]
    industry = "trucking_delivery"
    spec_path = base / "config" / "industries" / industry / f"{industry}_dataset_specs.json"
    industry_cfg = yaml.safe_load((base / "config" / "industries" / industry / "industry.yaml").read_text()) or {}
    categories_cfg = yaml.safe_load((base / "config" / "industries" / industry / "categories.yaml").read_text()) or {}

    spec_obj = json.loads(spec_path.read_text())
    datasets = spec_obj.get("datasets", {})
    tokens = _build_category_tokens(industry_cfg, categories_cfg)

    changed: list[dict[str, Any]] = []
    skipped = 0
    for dataset_id, cfg in datasets.items():
        training = cfg.get("training") or {}
        if not training.get("worker_enabled", False):
            continue
        targets = cfg.get("targets") or []
        if not targets:
            continue
        primary_idx = 0
        for i, t in enumerate(targets):
            if t.get("primary"):
                primary_idx = i
                break
        primary = dict(targets[primary_idx])
        primary_name = str(primary.get("name") or "")
        if not _is_low_value(primary_name):
            continue

        files = list((base / "data" / "processed" / industry).glob(f"**/{dataset_id}.parquet.zstd"))
        if not files:
            skipped += 1
            continue
        sample = pl.read_parquet(files[0], n_rows=5000)
        category = str((cfg.get("categories") or [cfg.get("category") or "unknown"])[0])
        new_target, new_task = _best_target(sample, category=category, category_tokens=tokens)
        if not new_target:
            skipped += 1
            continue

        old_primary = dict(primary)
        old_primary["primary"] = False
        new_primary = {"name": new_target, "task_type": new_task or "regression", "primary": True}
        dedup = [new_primary]
        for t in [old_primary, *targets]:
            name = str(t.get("name") or "")
            if not name or name == new_target:
                continue
            is_duplicate = any(str(x.get("name") or "") == name for x in dedup)
            if not is_duplicate:
                t2 = dict(t)
                t2["primary"] = False
                dedup.append(t2)
        cfg["targets"] = dedup
        datasets[dataset_id] = cfg
        changed.append(
            {
                "dataset_id": dataset_id,
                "category": category,
                "old_target": primary_name,
                "new_target": new_target,
                "new_task_type": new_task,
            }
        )

    spec_obj["datasets"] = datasets
    spec_path.write_text(json.dumps(spec_obj, indent=2) + "\n")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": industry,
        "changed_count": len(changed),
        "skipped_count": skipped,
        "changes": changed,
    }
    out = base / "reports" / "trucking_delivery_retarget_selected_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
