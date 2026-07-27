#!/usr/bin/env python3
"""
Repair dataset target specs to enforce business-aligned multi-target workers.

Primary use:
  - Ensure each worker has valid targets.
  - Auto-create missing derived *_band targets.
  - Enforce primary/secondary target fallback ordering.
  - Apply lightweight category corrections for obvious misplacements.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _load_yaml(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        parsed = yaml.safe_load(f)
    return parsed if parsed is not None else default


def _is_low_value(name: str) -> bool:
    token = str(name).strip().lower()
    if token in LOW_VALUE_EXACT:
        return True
    return any(p.match(token) for p in LOW_VALUE_PATTERNS)


def _is_id_like(name: str) -> bool:
    token = str(name).strip().lower()
    return (
        token in {"id", "row_id", "record_id", "uuid", "uid", "index"}
        or token.endswith("_id")
        or bool(re.match(r".*id\d*$", token))
        or "fips" in token
    )


def _category_override(dataset_id: str, current: str) -> str:
    ds = dataset_id.lower()
    if "border_crossing_entry" in ds:
        return "cross_border_flow"
    if any(k in ds for k in ["grade_crossing", "accident_incident", "injury_illness", "casualty", "equipment_accident"]):
        return "carrier_safety_risk"
    if any(k in ds for k in ["temperature", "cold_chain", "reefer", "spoilage"]):
        return "cold_chain_integrity"
    return current


def _build_category_tokens(industry_cfg: dict, categories_cfg: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for cat, cfg in (industry_cfg.get("categories", {}) or {}).items():
        for name in cfg.get("typical_targets", []) or []:
            for tok in re.split(r"[^a-z0-9]+", str(name).lower()):
                if tok and tok not in {"flag", "bucket", "tier", "risk", "event"}:
                    out[cat].add(tok)
    for cat, cfg in (categories_cfg.get("categories", {}) or {}).items():
        for kw in cfg.get("keywords", []) or []:
            for tok in re.split(r"[^a-z0-9]+", str(kw).lower()):
                if tok and tok not in {"risk", "flow"}:
                    out[cat].add(tok)
    # Global metric indicators.
    metric_tokens = {
        "value", "obs", "figure", "count", "hours", "hour", "delay", "dwell",
        "wait", "time", "volume", "ton", "tons", "tonnage", "teu",
        "cost", "price", "revenue", "margin", "yield", "capacity",
        "utilization", "rate", "speed", "distance", "duration", "passenger",
        "incidents", "accident", "injury", "claims", "temperature",
    }
    for cat in set(out.keys()):
        out[cat].update(metric_tokens)
    return out


def _best_metric_column(
    dataset_id: str,
    category: str,
    schema_names: list[str],
    schema_dtypes: list[pl.DataType],
    category_tokens: dict[str, set[str]],
) -> str | None:
    token_set = category_tokens.get(category, set())
    best: tuple[float, str] | None = None
    for name, dtype in zip(schema_names, schema_dtypes):
        lname = name.lower()
        if _is_id_like(lname) or _is_low_value(lname):
            continue
        score = 0.0
        if dtype in NUMERIC_DTYPES:
            score += 2.0
        if lname in {"obs_value", "value", "figure", "count", "hours", "hour"}:
            score += 3.0
        if any(tok and tok in lname for tok in token_set):
            score += 1.5
        if re.search(r"(delay|dwell|wait|volume|ton|teu|value|cost|yield|margin|capacity|util|incident|injury|claim|temp)", lname):
            score += 1.5
        if re.search(r"(latitude|longitude|^lat$|^lon$|easting|northing|postcode|zip|fips|code$)", lname):
            score -= 4.0
        if re.fullmatch(r"(19|20)\d{2}", lname):
            score -= 4.0
        if lname in {"price_base", "decimal", "decimals", "unit_mult", "series_no"}:
            score -= 5.0
        if best is None or score > best[0]:
            best = (score, name)

    if not best or best[0] < 1.0:
        return None
    # Ensure chosen metric is numeric so regression + threshold banding are valid.
    picked = best[1]
    dtype_map = {n: d for n, d in zip(schema_names, schema_dtypes)}
    if dtype_map.get(picked) not in NUMERIC_DTYPES:
        return None
    return picked


def _ensure_derived_band(derived_targets: dict, base_column: str) -> str:
    band_name = f"{base_column}_band"
    if band_name not in derived_targets:
        derived_targets[band_name] = {
            "source": base_column,
            "dtype": "threshold_split",
            "high_label": "high",
            "low_label": "low",
        }
    return band_name


def _resolve_name(name: str, columns_map: dict[str, str]) -> str | None:
    if name in columns_map.values():
        return name
    return columns_map.get(name.lower())


def run(industry: str, apply: bool) -> None:
    base = Path(__file__).resolve().parents[1]
    spec_path = base / "config" / "industries" / industry / f"{industry}_dataset_specs.json"
    industry_cfg_path = base / "config" / "industries" / industry / "industry.yaml"
    categories_cfg_path = base / "config" / "industries" / industry / "categories.yaml"
    processed_root = base / "data" / "processed" / industry

    spec_obj = _load_json(spec_path, {})
    datasets = spec_obj.get("datasets", {})
    industry_cfg = _load_yaml(industry_cfg_path, {})
    categories_cfg = _load_yaml(categories_cfg_path, {})
    default_tasks = {k: v.get("task_type", "regression") for k, v in (categories_cfg.get("categories", {}) or {}).items()}
    category_tokens = _build_category_tokens(industry_cfg, categories_cfg)

    changes: list[dict[str, Any]] = []
    missing_files = 0

    for dataset_id, cfg in sorted(datasets.items()):
        existing_training = dict(cfg.get("training") or {})
        if existing_training.get("worker_enabled") is False or bool(cfg.get("skip_training")):
            # Preserve explicitly disabled/low-signal workers.
            continue

        files = list(processed_root.glob(f"**/{dataset_id}.parquet.zstd"))
        if not files:
            missing_files += 1
            continue
        file_path = files[0]
        try:
            schema = pl.scan_parquet(file_path).collect_schema()
            schema_names = list(schema.names())
            schema_dtypes = list(schema.dtypes())
        except Exception:
            continue

        columns_map = {c.lower(): c for c in schema_names}
        dtype_map = {n: d for n, d in zip(schema_names, schema_dtypes)}
        current_category = str(cfg.get("category") or "operations")
        new_category = _category_override(dataset_id, current_category)
        category = new_category
        default_task = default_tasks.get(category, "regression")

        derived_targets = dict(cfg.get("derived_targets") or {})
        input_targets = list(cfg.get("targets") or [])
        valid_targets: list[dict[str, Any]] = []
        local_change = {"dataset_id": dataset_id, "updated": []}

        # Keep existing targets only if resolvable and business-meaningful.
        for target in input_targets:
            raw_name = str(target.get("name", "")).strip()
            task = str(target.get("task_type", default_task))
            if not raw_name:
                continue
            if _is_low_value(raw_name):
                local_change["updated"].append(f"drop_low_value_target:{raw_name}")
                continue
            if raw_name.isdigit():
                local_change["updated"].append(f"drop_numeric_name_target:{raw_name}")
                continue

            resolved = _resolve_name(raw_name, columns_map)
            if resolved:
                valid_targets.append({"name": resolved, "task_type": task, "primary": False})
                continue

            if raw_name.endswith("_band"):
                base_name = raw_name[: -len("_band")]
                base_resolved = _resolve_name(base_name, columns_map)
                if base_resolved and dtype_map.get(base_resolved) in NUMERIC_DTYPES:
                    band_name = _ensure_derived_band(derived_targets, base_resolved)
                    valid_targets.append({"name": band_name, "task_type": "classification", "primary": False})
                    local_change["updated"].append(f"add_missing_derived:{band_name}")
                    continue

            local_change["updated"].append(f"drop_missing_target:{raw_name}")

        # If no valid targets survived, infer from schema.
        if not valid_targets:
            metric_col = _best_metric_column(
                dataset_id=dataset_id,
                category=category,
                schema_names=schema_names,
                schema_dtypes=schema_dtypes,
                category_tokens=category_tokens,
            )
            if metric_col:
                band_name = _ensure_derived_band(derived_targets, metric_col)
                if default_task == "classification":
                    valid_targets = [
                        {"name": band_name, "task_type": "classification", "primary": True},
                        {"name": metric_col, "task_type": "regression", "primary": False},
                    ]
                else:
                    valid_targets = [
                        {"name": metric_col, "task_type": "regression", "primary": True},
                        {"name": band_name, "task_type": "classification", "primary": False},
                    ]
                local_change["updated"].append(f"infer_targets_from_metric:{metric_col}")

        # Ensure paired fallback target (band + raw) only when needed.
        if len(valid_targets) == 1:
            names = {t["name"] for t in valid_targets}
            primary = valid_targets[0]
            if primary["name"].endswith("_band"):
                base_name = primary["name"][: -len("_band")]
                if base_name in schema_names and base_name not in names:
                    valid_targets.append({"name": base_name, "task_type": "regression", "primary": False})
                    local_change["updated"].append(f"add_raw_fallback:{base_name}")
            else:
                if primary["name"] in schema_names and dtype_map.get(primary["name"]) in NUMERIC_DTYPES:
                    band = _ensure_derived_band(derived_targets, primary["name"])
                    if band not in names:
                        valid_targets.append({"name": band, "task_type": "classification", "primary": False})
                        local_change["updated"].append(f"add_band_fallback:{band}")

        # Reorder to align with category default task.
        if valid_targets:
            if default_task == "classification":
                valid_targets.sort(key=lambda t: (0 if t["task_type"] == "classification" else 1, 0 if t["name"].endswith("_band") else 1))
            else:
                valid_targets.sort(key=lambda t: (0 if t["task_type"] == "regression" else 1, 1 if t["name"].endswith("_band") else 0))
            for i, t in enumerate(valid_targets):
                t["primary"] = i == 0

        # Write back if changes exist.
        if category != current_category:
            cfg["category"] = category
            cfg["categories"] = [category]
            local_change["updated"].append(f"category:{current_category}->{category}")

        if valid_targets:
            if cfg.get("targets") != valid_targets:
                cfg["targets"] = valid_targets
                local_change["updated"].append("targets_rewritten")
        if derived_targets:
            if cfg.get("derived_targets") != derived_targets:
                cfg["derived_targets"] = derived_targets
                local_change["updated"].append("derived_targets_updated")

        training = dict(existing_training)
        changed_training = False
        for key, value in {
            "worker_enabled": True,
            "allow_target_fallback": True,
            "fallback_on_fail": True,
            "max_target_trials": 4,
        }.items():
            if training.get(key) != value:
                training[key] = value
                changed_training = True
        if changed_training and (local_change["updated"] or len(cfg.get("targets") or []) > 1):
            cfg["training"] = training
            local_change["updated"].append("training_fallback_enforced")

        # Keep targets clear if still unresolved after all attempts.
        if not cfg.get("targets"):
            cfg["training"] = {**(cfg.get("training") or {}), "worker_enabled": False, "gating": False, "skip_training": True}
            local_change["updated"].append("worker_disabled_no_target")

        if local_change["updated"]:
            changes.append(local_change)

    report = {
        "industry": industry,
        "datasets_total": len(datasets),
        "datasets_changed": len(changes),
        "missing_processed_files": missing_files,
        "changes": changes,
    }
    report_path = base / "reports" / f"{industry}_target_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    if apply:
        spec_obj["datasets"] = datasets
        spec_path.write_text(json.dumps(spec_obj, indent=2) + "\n")

    print(f"Spec path:  {spec_path}")
    print(f"Report:     {report_path}")
    print(f"Changed:    {len(changes)} datasets")
    print(f"Missing:    {missing_files} files")
    if not apply:
        print("Dry run only; re-run with --apply to persist changes.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(industry=args.industry, apply=args.apply)


if __name__ == "__main__":
    main()
