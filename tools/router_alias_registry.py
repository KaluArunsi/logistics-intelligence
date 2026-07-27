#!/usr/bin/env python3
"""Manage router alias registry and feedback events for self-healing column synonyms."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"candidate", "approved", "rejected"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value: str) -> str:
    return str(value).strip().lower()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _registry_path(base: Path, industry: str) -> Path:
    return base / "config" / "router" / "alias_registry" / f"{industry}_column_aliases.json"


def _feedback_path(base: Path, industry: str) -> Path:
    return base / "reports" / "router_feedback" / f"{industry}_alias_events.jsonl"


def _load_registry(base: Path, industry: str) -> dict[str, Any]:
    path = _registry_path(base, industry)
    payload = _read_json(path)
    if not payload:
        return {
            "version": 1,
            "industry": industry,
            "updated_at": _now_iso(),
            "aliases": [],
        }
    payload.setdefault("version", 1)
    payload.setdefault("industry", industry)
    payload.setdefault("updated_at", _now_iso())
    payload.setdefault("aliases", [])
    return payload


def _save_registry(base: Path, industry: str, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now_iso()
    _write_json(_registry_path(base, industry), payload)


def _append_feedback_event(base: Path, industry: str, event: dict[str, Any]) -> None:
    path = _feedback_path(base, industry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _ensure_unique_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned:
            continue
        key = _norm(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def record_alias(
    base: Path,
    industry: str,
    canonical_column: str,
    alias: str,
    categories: list[str],
    worker_dataset_ids: list[str],
    confidence: float,
    source: str,
    dtype_expected: str | None,
    unit_expected: str | None,
    status: str,
    note: str | None,
) -> int:
    canonical_column = canonical_column.strip()
    alias = alias.strip()
    if not canonical_column or not alias:
        raise ValueError("canonical_column and alias are required")
    if _norm(canonical_column) == _norm(alias):
        raise ValueError("canonical_column and alias must be different")

    categories = _ensure_unique_list(categories)
    worker_dataset_ids = _ensure_unique_list(worker_dataset_ids)
    status = status.strip().lower()
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of: {sorted(VALID_STATUSES)}")

    registry = _load_registry(base, industry)
    aliases = registry.get("aliases", [])
    event_time = _now_iso()
    changed = False

    match_idx = None
    for idx, row in enumerate(aliases):
        if not isinstance(row, dict):
            continue
        if _norm(row.get("canonical_column", "")) != _norm(canonical_column):
            continue
        if _norm(row.get("alias", "")) != _norm(alias):
            continue
        match_idx = idx
        break

    if match_idx is None:
        aliases.append(
            {
                "canonical_column": canonical_column,
                "alias": alias,
                "status": status,
                "confidence": float(confidence),
                "source": source,
                "dtype_expected": dtype_expected,
                "unit_expected": unit_expected,
                "categories": categories,
                "worker_dataset_ids": worker_dataset_ids,
                "first_seen": event_time,
                "last_seen": event_time,
                "hit_count": 1,
                "notes": [note] if note else [],
            }
        )
        changed = True
    else:
        row = aliases[match_idx]
        row["status"] = status
        row["confidence"] = max(float(row.get("confidence", 0.0) or 0.0), float(confidence))
        row["source"] = source or row.get("source")
        if dtype_expected:
            row["dtype_expected"] = dtype_expected
        if unit_expected:
            row["unit_expected"] = unit_expected
        row["categories"] = _ensure_unique_list(list(row.get("categories", [])) + categories)
        row["worker_dataset_ids"] = _ensure_unique_list(
            list(row.get("worker_dataset_ids", [])) + worker_dataset_ids
        )
        row["last_seen"] = event_time
        row["hit_count"] = int(row.get("hit_count", 0) or 0) + 1
        if note:
            notes = list(row.get("notes", []))
            notes.append(note)
            row["notes"] = notes[-20:]
        changed = True

    if changed:
        registry["aliases"] = aliases
        _save_registry(base, industry, registry)

    event = {
        "event_time": event_time,
        "event_type": "alias_inference",
        "industry": industry,
        "canonical_column": canonical_column,
        "alias": alias,
        "status": status,
        "confidence": float(confidence),
        "source": source,
        "dtype_expected": dtype_expected,
        "unit_expected": unit_expected,
        "categories": categories,
        "worker_dataset_ids": worker_dataset_ids,
        "note": note,
    }
    _append_feedback_event(base, industry, event)
    return 0


def promote_aliases(base: Path, industry: str, min_hit_count: int, min_confidence: float) -> int:
    registry = _load_registry(base, industry)
    aliases = registry.get("aliases", [])
    promoted = 0
    for row in aliases:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).lower() != "candidate":
            continue
        hit_count = int(row.get("hit_count", 0) or 0)
        confidence = float(row.get("confidence", 0.0) or 0.0)
        if hit_count >= min_hit_count and confidence >= min_confidence:
            row["status"] = "approved"
            row["promoted_at"] = _now_iso()
            promoted += 1

    registry["aliases"] = aliases
    _save_registry(base, industry, registry)

    event = {
        "event_time": _now_iso(),
        "event_type": "alias_promotion_run",
        "industry": industry,
        "promoted_count": promoted,
        "min_hit_count": int(min_hit_count),
        "min_confidence": float(min_confidence),
    }
    _append_feedback_event(base, industry, event)
    print(
        json.dumps(
            {
                "industry": industry,
                "promoted_count": promoted,
                "min_hit_count": int(min_hit_count),
                "min_confidence": float(min_confidence),
            }
        )
    )
    return 0


def summarize_registry(base: Path, industry: str) -> int:
    registry = _load_registry(base, industry)
    aliases = registry.get("aliases", [])
    summary = {"candidate": 0, "approved": 0, "rejected": 0}
    for row in aliases:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "candidate")).lower()
        if status not in summary:
            status = "candidate"
        summary[status] += 1
    print(
        json.dumps(
            {
                "industry": industry,
                "total_aliases": len([r for r in aliases if isinstance(r, dict)]),
                "status_counts": summary,
                "registry_path": str(_registry_path(base, industry)),
                "feedback_path": str(_feedback_path(base, industry)),
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--industry", required=True)
    record_parser.add_argument("--canonical-column", required=True)
    record_parser.add_argument("--alias", required=True)
    record_parser.add_argument("--category", action="append", default=[])
    record_parser.add_argument("--worker-dataset-id", action="append", default=[])
    record_parser.add_argument("--confidence", type=float, default=0.70)
    record_parser.add_argument("--source", default="llm_inference")
    record_parser.add_argument("--dtype-expected", default=None)
    record_parser.add_argument("--unit-expected", default=None)
    record_parser.add_argument("--status", default="candidate", choices=sorted(VALID_STATUSES))
    record_parser.add_argument("--note", default=None)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--industry", required=True)
    promote_parser.add_argument("--min-hit-count", type=int, default=3)
    promote_parser.add_argument("--min-confidence", type=float, default=0.90)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--industry", required=True)

    args = parser.parse_args()
    base = Path(args.base).resolve()

    if args.command == "record":
        return record_alias(
            base=base,
            industry=args.industry,
            canonical_column=args.canonical_column,
            alias=args.alias,
            categories=args.category,
            worker_dataset_ids=args.worker_dataset_id,
            confidence=args.confidence,
            source=args.source,
            dtype_expected=args.dtype_expected,
            unit_expected=args.unit_expected,
            status=args.status,
            note=args.note,
        )
    if args.command == "promote":
        return promote_aliases(
            base=base,
            industry=args.industry,
            min_hit_count=args.min_hit_count,
            min_confidence=args.min_confidence,
        )
    if args.command == "summary":
        return summarize_registry(base=base, industry=args.industry)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
