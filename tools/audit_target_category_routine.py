#!/usr/bin/env python3
"""
Routine audit for dataset target/category selection.

Workflow:
1) Read first N rows for each dataset file.
2) Cross-check source metadata (OpenML live API + Kaggle cached metadata).
3) Recommend target(s), task type, and category.
4) Emit machine-readable + markdown reports.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import polars as pl
import yaml


@dataclass
class TargetRecommendation:
    name: str
    task_type: str
    confidence: str
    reason: str


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


def _find_spec_path(base: Path, industry: str) -> Path:
    direct = base / "config" / "industries" / industry / f"{industry}_dataset_specs.json"
    if direct.exists():
        return direct
    matches = sorted((base / "config" / "industries" / industry).glob("*_dataset_specs.json"))
    if not matches:
        raise FileNotFoundError(f"No dataset specs found for industry '{industry}'")
    return matches[0]


def _load_kaggle_cache(base: Path, industry: str) -> dict[str, dict]:
    """Map slug -> cached Kaggle metadata file (owner/slug stored in payload)."""
    meta_dir = base / "data" / "metadata" / industry / "kaggle_meta"
    out: dict[str, dict] = {}
    if not meta_dir.exists():
        return out
    for path in sorted(meta_dir.glob("*.json")):
        payload = _load_json(path, {})
        stem = path.stem
        if "__" not in stem:
            continue
        _, slug = stem.split("__", 1)
        out[slug] = payload
    return out


def _fetch_openml_dataset(dataset_id: str) -> dict[str, Any]:
    """Fetch OpenML dataset metadata using the public JSON API."""
    url = f"https://www.openml.org/api/v1/json/data/{dataset_id}"
    with urlopen(url, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    desc = payload.get("data_set_description", {})
    return {
        "openml_id": dataset_id,
        "name": desc.get("name"),
        "default_target_attribute": desc.get("default_target_attribute"),
        "url": f"https://www.openml.org/d/{dataset_id}",
        "status": "ok",
    }


def _source_from_dataset_id(dataset_id: str) -> str:
    if dataset_id.startswith("kaggle__"):
        return "kaggle"
    if dataset_id.startswith("openml__"):
        return "openml"
    if dataset_id.startswith("uci__"):
        return "uci"
    return "unknown"


def _slug_from_dataset_id(dataset_id: str) -> str | None:
    # kaggle__<slug>__<file>
    m = re.match(r"^kaggle__([^_].*?)__", dataset_id)
    if not m:
        return None
    return m.group(1)


def _openml_id_from_dataset_id(dataset_id: str) -> str | None:
    # openml__<id>_...
    m = re.match(r"^openml__(\d+)_", dataset_id)
    if not m:
        return None
    return m.group(1)


def _read_sample(path: Path, n_rows: int) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path, n_rows=n_rows)
    return pl.read_parquet(path, n_rows=n_rows)


def _is_id_like(col: str) -> bool:
    c = col.lower()
    return (
        c in {"id", "row_id", "record_id", "index", "uuid", "uid"}
        or c.endswith("_id")
        or bool(re.match(r".*id\d*$", c))
    )


def _category_scores(search_text: str, categories_cfg: dict) -> dict[str, int]:
    scores: dict[str, int] = {}
    text = search_text.lower()
    for name, cfg in (categories_cfg.get("categories", {}) or {}).items():
        kws = cfg.get("keywords", []) or []
        scores[name] = sum(1 for kw in kws if str(kw).lower() in text)
    return scores


def _recommend_category(
    dataset_id: str,
    columns: list[str],
    current_category: str | None,
    categories_cfg: dict,
    internet_evidence: dict,
) -> tuple[str, str, str]:
    extra = []
    for key in ("name", "title", "subtitle", "description", "summary"):
        val = internet_evidence.get(key)
        if val:
            extra.append(str(val))
    search_text = " ".join([dataset_id, " ".join(columns), " ".join(extra)])
    scores = _category_scores(search_text, categories_cfg)
    if not scores:
        return (current_category or "operations", "low", "no category config available")
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_name, top_score = best[0]
    second_score = best[1][1] if len(best) > 1 else 0
    if top_score == 0:
        return (current_category or top_name, "low", "no keyword evidence")
    if top_score >= 3 and top_score - second_score >= 1:
        return (top_name, "high", f"keyword score {top_score}")
    return (top_name, "medium", f"keyword score {top_score}")


def _task_from_series(series: pl.Series) -> str:
    if series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
        n_unique = series.n_unique()
        if n_unique <= 20:
            return "classification"
        return "regression"
    return "classification"


def _target_candidates(df: pl.DataFrame) -> list[tuple[str, str, float, str]]:
    class_tokens = (
        "label", "target", "class", "fraud", "risk", "converted", "conversion",
        "delay", "abandon", "status", "sentiment", "review", "rating", "score",
        "flag", "reorder", "reordered",
    )
    reg_tokens = (
        "price", "amount", "sales", "revenue", "cost", "quantity", "time",
        "duration", "billing", "margin", "gmv", "units",
    )
    out: list[tuple[str, str, float, str]] = []
    n_rows = max(df.height, 1)
    for col in df.columns:
        if _is_id_like(col):
            continue
        s = df[col]
        non_null = max(1, int(s.is_not_null().sum()))
        unique = int(s.n_unique())
        null_ratio = 1.0 - (non_null / float(n_rows))
        name_l = col.lower()
        cls_hits = sum(1 for t in class_tokens if t in name_l)
        reg_hits = sum(1 for t in reg_tokens if t in name_l)
        inferred_task = _task_from_series(s)

        if inferred_task == "classification":
            if unique < 2 or unique > 100:
                continue
            score = 0.4 + (0.2 * cls_hits) - (0.1 * null_ratio)
            if reg_hits:
                score -= 0.05 * reg_hits
            out.append((col, "classification", score, f"unique={unique}, class_hits={cls_hits}"))
        else:
            if unique < 10:
                continue
            score = 0.4 + (0.2 * reg_hits) - (0.1 * null_ratio)
            if cls_hits:
                score -= 0.05 * cls_hits
            out.append((col, "regression", score, f"unique={unique}, reg_hits={reg_hits}"))
    return sorted(out, key=lambda x: x[2], reverse=True)


def _recommend_target(
    df: pl.DataFrame,
    current_target: str | None,
    current_task: str | None,
    internet_evidence: dict,
) -> TargetRecommendation | None:
    cols_lower = {c.lower(): c for c in df.columns}

    internet_target = internet_evidence.get("default_target_attribute")
    if internet_target:
        t = str(internet_target).strip()
        resolved = cols_lower.get(t.lower())
        if resolved:
            series = df[resolved]
            return TargetRecommendation(
                name=resolved,
                task_type=_task_from_series(series),
                confidence="high",
                reason="source metadata default target",
            )

    if current_target and current_target in df.columns:
        task = current_task or _task_from_series(df[current_target])
        return TargetRecommendation(
            name=current_target,
            task_type=str(task),
            confidence="medium",
            reason="existing spec target present in sample",
        )

    candidates = _target_candidates(df)
    if not candidates:
        return None
    col, task_type, _, reason = candidates[0]
    return TargetRecommendation(
        name=col,
        task_type=task_type,
        confidence="medium",
        reason=f"sample heuristic ({reason})",
    )


def run(industry: str, sample_rows: int, apply: bool) -> None:
    base = Path(__file__).resolve().parents[1]
    spec_path = _find_spec_path(base, industry)
    spec_obj = _load_json(spec_path, {})
    datasets = spec_obj.get("datasets", {})

    categories_cfg = _load_yaml(base / "config" / "industries" / industry / "categories.yaml", {})
    kaggle_cache = _load_kaggle_cache(base, industry)
    openml_meta_cache: dict[str, dict] = {}

    processed_root = base / "data" / "processed" / industry
    sampled_root = base / "data" / "sampled" / industry

    report_rows: list[dict] = []
    updates = 0

    for dataset_id, spec in sorted(datasets.items()):
        candidates = []
        for root in (processed_root, sampled_root):
            if not root.exists():
                continue
            candidates.extend(root.rglob(f"{dataset_id}.parquet.zstd"))
            candidates.extend(root.rglob(f"{dataset_id}.parquet"))
            candidates.extend(root.rglob(f"{dataset_id}.csv"))
        if not candidates:
            report_rows.append({
                "dataset_id": dataset_id,
                "status": "missing_file",
            })
            continue
        data_path = candidates[0]
        try:
            sample = _read_sample(data_path, sample_rows)
        except Exception as exc:
            report_rows.append({
                "dataset_id": dataset_id,
                "status": "read_failed",
                "error": str(exc),
            })
            continue

        source = _source_from_dataset_id(dataset_id)
        internet_evidence: dict[str, Any] = {"source": source}

        if source == "openml":
            oid = _openml_id_from_dataset_id(dataset_id)
            if oid:
                if oid not in openml_meta_cache:
                    try:
                        openml_meta_cache[oid] = _fetch_openml_dataset(oid)
                    except Exception as exc:
                        openml_meta_cache[oid] = {
                            "openml_id": oid,
                            "status": "error",
                            "error": str(exc),
                            "url": f"https://www.openml.org/d/{oid}",
                        }
                internet_evidence.update(openml_meta_cache[oid])
        elif source == "kaggle":
            slug = _slug_from_dataset_id(dataset_id)
            if slug and slug in kaggle_cache:
                payload = kaggle_cache[slug]
                internet_evidence.update({
                    "status": "cached",
                    "title": payload.get("title"),
                    "subtitle": payload.get("subtitle"),
                    "description": payload.get("description"),
                    "url": payload.get("url"),
                    "ref": payload.get("ref"),
                })
            else:
                internet_evidence.update({"status": "missing_kaggle_cache"})

        current_category = (spec.get("categories") or [spec.get("category") or "unknown"])[0]
        targets = spec.get("targets") or []
        current_target = None
        current_task = None
        for t in targets:
            if t.get("primary", False):
                current_target = t.get("name")
                current_task = t.get("task_type")
                break
        if current_target is None and targets:
            current_target = targets[0].get("name")
            current_task = targets[0].get("task_type")

        rec_target = _recommend_target(sample, current_target, current_task, internet_evidence)
        rec_category, category_conf, category_reason = _recommend_category(
            dataset_id=dataset_id,
            columns=sample.columns,
            current_category=current_category,
            categories_cfg=categories_cfg,
            internet_evidence=internet_evidence,
        )

        target_changed = bool(rec_target and rec_target.name != current_target)
        task_changed = bool(rec_target and rec_target.task_type != (current_task or rec_target.task_type))
        category_changed = rec_category != current_category

        if apply and rec_target:
            if current_target is None or current_target not in sample.columns:
                spec["targets"] = [{
                    "name": rec_target.name,
                    "task_type": rec_target.task_type,
                    "primary": True,
                }]
                updates += 1
            if current_category in (None, "unknown") and rec_category:
                spec["category"] = rec_category
                spec["categories"] = [rec_category]
                updates += 1

        report_rows.append({
            "dataset_id": dataset_id,
            "status": "ok",
            "source": source,
            "file_path": str(data_path),
            "sample_rows": sample.height,
            "columns": sample.columns,
            "current_category": current_category,
            "recommended_category": rec_category,
            "recommended_category_confidence": category_conf,
            "recommended_category_reason": category_reason,
            "current_target": current_target,
            "current_task_type": current_task,
            "recommended_target": (
                {
                    "name": rec_target.name,
                    "task_type": rec_target.task_type,
                    "confidence": rec_target.confidence,
                    "reason": rec_target.reason,
                }
                if rec_target
                else None
            ),
            "target_changed": target_changed,
            "task_changed": task_changed,
            "category_changed": category_changed,
            "internet_evidence": internet_evidence,
        })

    if apply:
        spec_obj["datasets"] = datasets
        spec_path.write_text(json.dumps(spec_obj, indent=2) + "\n")

    out_dir = base / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_report = out_dir / f"{industry}_target_category_audit.json"
    md_report = out_dir / f"{industry}_target_category_audit.md"
    json_report.write_text(json.dumps({
        "industry": industry,
        "sample_rows": sample_rows,
        "datasets_scanned": len(report_rows),
        "spec_updates_applied": updates,
        "report": report_rows,
    }, indent=2) + "\n")

    changed = [r for r in report_rows if r.get("status") == "ok" and (r.get("target_changed") or r.get("category_changed") or r.get("task_changed"))]
    lines = [
        f"# {industry.title()} Target/Category Audit",
        "",
        f"- Datasets scanned: `{len(report_rows)}`",
        f"- Candidate changes found: `{len(changed)}`",
        f"- Spec updates applied: `{updates}`",
        "",
        "## Candidate Changes",
    ]
    for row in changed:
        rec_t = row.get("recommended_target") or {}
        lines.append(
            f"- `{row['dataset_id']}` | category `{row['current_category']}` -> `{row['recommended_category']}` | "
            f"target `{row.get('current_target')}` -> `{rec_t.get('name')}` ({rec_t.get('task_type')})"
        )
    md_report.write_text("\n".join(lines) + "\n")

    print(f"JSON report: {json_report}")
    print(f"MD report:   {md_report}")
    print(f"Datasets scanned: {len(report_rows)}")
    print(f"Candidate changes: {len(changed)}")
    print(f"Spec updates applied: {updates}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", required=True)
    parser.add_argument("--sample-rows", type=int, default=100)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(industry=args.industry, sample_rows=args.sample_rows, apply=args.apply)


if __name__ == "__main__":
    main()
