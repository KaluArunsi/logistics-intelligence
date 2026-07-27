#!/usr/bin/env python3
"""Build router manifests with hierarchy: industry -> category -> workers -> columns per worker."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _to_abs(base: Path, maybe_rel: str | None) -> Path | None:
    if not maybe_rel:
        return None
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _pick_primary_target(spec: dict[str, Any]) -> dict[str, Any]:
    targets = spec.get("targets") or []
    if not targets:
        return {}
    for row in targets:
        if isinstance(row, dict) and bool(row.get("primary")):
            return row
    first = targets[0]
    return first if isinstance(first, dict) else {}


def _normalize_key(value: str) -> str:
    return value.strip().lower()


def _read_alias_registry(base: Path, industry: str) -> tuple[Path, list[dict[str, Any]]]:
    alias_path = base / "config" / "router" / "alias_registry" / f"{industry}_column_aliases.json"
    payload = _read_json(alias_path)
    rows = payload.get("aliases", []) if isinstance(payload, dict) else []

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        canonical = str(row.get("canonical_column", "")).strip()
        alias = str(row.get("alias", "")).strip()
        if not canonical or not alias:
            continue
        if _normalize_key(canonical) == _normalize_key(alias):
            continue

        status = str(row.get("status", "candidate")).strip().lower()
        if status not in {"candidate", "approved", "rejected"}:
            status = "candidate"

        categories = [str(v).strip() for v in (row.get("categories") or []) if str(v).strip()]
        worker_dataset_ids = [
            str(v).strip()
            for v in (row.get("worker_dataset_ids") or [])
            if str(v).strip()
        ]

        normalized.append(
            {
                "canonical_column": canonical,
                "alias": alias,
                "status": status,
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "source": str(row.get("source", "")).strip() or None,
                "dtype_expected": row.get("dtype_expected"),
                "unit_expected": row.get("unit_expected"),
                "categories": categories,
                "worker_dataset_ids": worker_dataset_ids,
                "first_seen": row.get("first_seen"),
                "last_seen": row.get("last_seen"),
                "hit_count": int(row.get("hit_count", 0) or 0),
            }
        )

    return alias_path, normalized


def _alias_applies(alias_row: dict[str, Any], category: str, dataset_id: str) -> bool:
    categories = alias_row.get("categories") or []
    if categories and category not in categories:
        return False
    worker_dataset_ids = alias_row.get("worker_dataset_ids") or []
    if worker_dataset_ids and dataset_id not in worker_dataset_ids:
        return False
    return True


def _worker_synonyms(
    alias_rows: list[dict[str, Any]],
    category: str,
    dataset_id: str,
    canonical_candidates: list[str],
) -> dict[str, Any]:
    canonical_lookup = {_normalize_key(c): c for c in canonical_candidates if c}
    canonical_to_aliases: dict[str, list[str]] = {}
    alias_to_canonical: dict[str, str] = {}

    for row in alias_rows:
        if row.get("status") != "approved":
            continue
        if not _alias_applies(row, category=category, dataset_id=dataset_id):
            continue

        canonical_raw = str(row.get("canonical_column") or "").strip()
        alias_raw = str(row.get("alias") or "").strip()
        if not canonical_raw or not alias_raw:
            continue

        canonical = canonical_lookup.get(_normalize_key(canonical_raw))
        if not canonical:
            continue

        alias_list = canonical_to_aliases.setdefault(canonical, [])
        if alias_raw not in alias_list:
            alias_list.append(alias_raw)

        # Keep first canonical assignment for a given alias to avoid ambiguity.
        alias_to_canonical.setdefault(alias_raw, canonical)

    for canonical, aliases in canonical_to_aliases.items():
        canonical_to_aliases[canonical] = sorted(set(aliases))

    return {
        "canonical_to_aliases": canonical_to_aliases,
        "alias_to_canonical": alias_to_canonical,
    }


def build_manifest(base: Path, industry: str) -> dict[str, Any]:
    categories_cfg = _read_yaml(base / "config" / "industries" / industry / "categories.yaml")
    specs_cfg = _read_json(base / "config" / "industries" / industry / f"{industry}_dataset_specs.json")
    models_cfg = _read_json(base / "models" / industry / "specs" / "models.json")
    datasets_meta = _read_json(base / "data" / "metadata" / industry / f"{industry}_datasets.json")
    alias_path, alias_rows = _read_alias_registry(base, industry)

    categories_map = categories_cfg.get("categories", {})
    dataset_specs = specs_cfg.get("datasets", {})
    active_dataset_ids = set(dataset_specs.keys())

    trained_worker_by_dataset: dict[str, dict[str, Any]] = {}
    for model_id, model in models_cfg.items():
        if model.get("level") != 0:
            continue
        dataset_id = str(model.get("dataset_id") or "")
        if not dataset_id:
            continue
        if dataset_id not in active_dataset_ids:
            # Ignore stale models that are no longer part of active dataset specs.
            continue
        spec = dataset_specs.get(dataset_id, {})
        training_cfg = spec.get("training") or {}
        worker_enabled = bool(training_cfg.get("worker_enabled", True)) if isinstance(training_cfg, dict) else True
        skip_training = bool(spec.get("skip_training", False))
        if skip_training or not worker_enabled:
            # Keep disabled/skip datasets out of active router worker pool.
            continue
        trained_worker_by_dataset[dataset_id] = {"model_id": model_id, "model": model}

    categories: list[dict[str, Any]] = []
    for category_name, cat_cfg in categories_map.items():
        task_type = str(cat_cfg.get("task_type", "mixed"))
        keywords = list(cat_cfg.get("keywords", []))

        workers: list[dict[str, Any]] = []
        for model_id, model in models_cfg.items():
            if model.get("level") != 0:
                continue
            if model.get("category") != category_name:
                continue

            dataset_id = str(model.get("dataset_id") or "")
            if dataset_id not in active_dataset_ids:
                # Keep router manifest aligned with active spec-backed datasets only.
                continue
            spec = dataset_specs.get(dataset_id, {})
            training_cfg = spec.get("training") or {}
            worker_enabled = bool(training_cfg.get("worker_enabled", True)) if isinstance(training_cfg, dict) else True
            skip_training = bool(spec.get("skip_training", False))
            if skip_training or not worker_enabled:
                # Exclude disabled/skip workers from active trained-worker routing list.
                continue
            dataset_meta = datasets_meta.get(dataset_id, {})
            dataset_columns = list(dataset_meta.get("columns") or [])
            model_feature_columns = list(model.get("feature_columns") or [])
            worker_synonyms = _worker_synonyms(
                alias_rows=alias_rows,
                category=category_name,
                dataset_id=dataset_id,
                canonical_candidates=sorted(set(dataset_columns) | set(model_feature_columns)),
            )

            fe_path = base / "models" / industry / "L0" / f"{model_id}_fe.json"
            fe = _read_json(fe_path) if fe_path.exists() else {}
            encodings = fe.get("encodings") or {}
            text_features = fe.get("text_features") or {}

            spec = dataset_specs.get(dataset_id, {})
            first_target = _pick_primary_target(spec)
            derived = spec.get("derived_target") or {}
            training = spec.get("training") or {}

            workers.append(
                {
                    "worker_model_id": model_id,
                    "worker_dataset_id": dataset_id,
                    "task_type": model.get("task_type"),
                    "passed_benchmark": bool(model.get("passed_benchmark", False)),
                    "target_column": model.get("target_column"),
                    "n_features": int(model.get("n_features") or 0),
                    "paths": {
                        "model_path": model.get("file_path"),
                        "data_file": model.get("data_file"),
                        "feature_metadata_path": str(fe_path.relative_to(base)) if fe_path.exists() else None,
                    },
                    "columns": {
                        "model_feature_columns": model_feature_columns,
                        "dataset_columns": dataset_columns,
                        "synonyms": worker_synonyms,
                        "feature_metadata": {
                            "kept_columns": list(fe.get("kept_columns") or []),
                            "dropped_columns": list(fe.get("dropped_columns") or []),
                            "dummy_columns": list(fe.get("dummy_columns") or []),
                            "encoded_columns": sorted(encodings.keys()),
                            "text_feature_columns": sorted(text_features.keys()),
                        },
                    },
                    "spec_target": {
                        "target_column": first_target.get("name"),
                        "target_task_type": first_target.get("task_type"),
                        "derived_target_name": derived.get("name"),
                        "gating": training.get("gating") if isinstance(training, dict) else None,
                    },
                }
            )

        workers.sort(key=lambda w: w["worker_model_id"])
        prepared_workers: list[dict[str, Any]] = []
        for dataset_id, spec in dataset_specs.items():
            spec_category = spec.get("category")
            spec_categories = spec.get("categories") or ([spec_category] if spec_category else [])
            if category_name not in spec_categories:
                continue

            dataset_meta = datasets_meta.get(dataset_id, {})
            trained = trained_worker_by_dataset.get(dataset_id)
            model = (trained or {}).get("model") or {}
            model_id = (trained or {}).get("model_id")
            first_target = _pick_primary_target(spec)
            derived = spec.get("derived_target") or {}
            training = spec.get("training") or {}
            skip_training = bool(spec.get("skip_training", False))
            worker_enabled = bool(training.get("worker_enabled", True)) if isinstance(training, dict) else True

            if skip_training or not worker_enabled:
                status = "skipped"
            elif model_id:
                status = "trained"
            else:
                status = "pending"

            prepared_workers.append(
                {
                    "worker_dataset_id": dataset_id,
                    "status": status,
                    "categories": list(spec_categories),
                    "target_column": first_target.get("name"),
                    "target_task_type": first_target.get("task_type"),
                    "derived_target_name": derived.get("name"),
                    "gating": training.get("gating") if isinstance(training, dict) else None,
                    "worker_enabled": worker_enabled,
                    "worker_disable_reason": training.get("worker_disable_reason") if isinstance(training, dict) else None,
                    "skip_training": skip_training,
                    "trained_model_id": model_id if status == "trained" else None,
                    "passed_benchmark": bool(model.get("passed_benchmark", False)) if model else False,
                    "paths": {
                        "data_file": dataset_meta.get("file_path"),
                        "trained_model_path": model.get("file_path") if status == "trained" and model else None,
                    },
                    "columns": {
                        "dataset_columns": list(dataset_meta.get("columns") or []),
                    },
                }
            )

        prepared_workers.sort(key=lambda w: w["worker_dataset_id"])
        categories.append(
            {
                "category": category_name,
                "task_type": task_type,
                "keywords": keywords,
                "worker_count": len(workers),
                "prepared_worker_count": len(prepared_workers),
                "workers": workers,
                "prepared_workers": prepared_workers,
            }
        )

    categories.sort(key=lambda c: c["category"])
    approved_rows = [row for row in alias_rows if row.get("status") == "approved"]
    canonical_to_aliases: dict[str, list[str]] = {}
    for row in approved_rows:
        canonical = str(row.get("canonical_column") or "").strip()
        alias = str(row.get("alias") or "").strip()
        if not canonical or not alias:
            continue
        aliases = canonical_to_aliases.setdefault(canonical, [])
        if alias not in aliases:
            aliases.append(alias)
    for canonical, aliases in canonical_to_aliases.items():
        canonical_to_aliases[canonical] = sorted(aliases)

    return {
        "version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": industry,
        "alias_registry": {
            "path": str(alias_path.relative_to(base)) if alias_path.exists() else None,
            "total_aliases": len(alias_rows),
            "approved_aliases": len([r for r in alias_rows if r.get("status") == "approved"]),
            "candidate_aliases": len([r for r in alias_rows if r.get("status") == "candidate"]),
            "rejected_aliases": len([r for r in alias_rows if r.get("status") == "rejected"]),
            "canonical_to_aliases": canonical_to_aliases,
        },
        "categories": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--industry", required=True)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    industry = args.industry

    manifest = build_manifest(base, industry)
    out = base / "config" / "router" / "manifests" / f"{industry}_router_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
