#!/usr/bin/env python3
"""Review/promote target semantics candidates."""

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
    return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())


def _registry_path(base: Path, industry: str) -> Path:
    return base / "config" / "router" / "target_semantics" / f"{industry}_target_semantics.json"


def _feedback_path(base: Path, industry: str) -> Path:
    return base / "reports" / "router_feedback" / f"{industry}_target_semantic_events.jsonl"


def _load_registry(base: Path, industry: str) -> dict[str, Any]:
    path = _registry_path(base, industry)
    if not path.exists():
        return {
            "version": 1,
            "industry": industry,
            "updated_at": _now_iso(),
            "targets": [],
        }
    with open(path) as f:
        payload = json.load(f)
    payload.setdefault("version", 1)
    payload.setdefault("industry", industry)
    payload.setdefault("updated_at", _now_iso())
    payload.setdefault("targets", [])
    return payload


def _save_registry(base: Path, industry: str, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _now_iso()
    path = _registry_path(base, industry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _append_event(base: Path, industry: str, event: dict[str, Any]) -> None:
    path = _feedback_path(base, industry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")


def _summary(payload: dict[str, Any]) -> dict[str, int]:
    counts = {"candidate": 0, "approved": 0, "rejected": 0}
    for row in payload.get("targets", []) or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "candidate").lower()
        if status not in counts:
            status = "candidate"
        counts[status] += 1
    return counts


def _set_target_status(
    *,
    base: Path,
    industry: str,
    target_column: str,
    status: str,
    note: str,
) -> dict[str, Any]:
    payload = _load_registry(base, industry)
    rows = payload.get("targets") or []
    changed = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _norm(row.get("target_column", "")) != _norm(target_column):
            continue
        row["status"] = status
        row["reviewed_at"] = _now_iso()
        notes = list(row.get("review_notes") or [])
        if note:
            notes.append(note)
        row["review_notes"] = notes[-30:]
        changed += 1
    payload["targets"] = rows
    _save_registry(base, industry, payload)
    _append_event(
        base,
        industry,
        {
            "event_time": _now_iso(),
            "event_type": "target_semantic_manual_review",
            "industry": industry,
            "target_column": target_column,
            "status": status,
            "changed_rows": changed,
            "note": note,
        },
    )
    return {
        "industry": industry,
        "target_column": target_column,
        "status": status,
        "changed_rows": changed,
        "registry_path": str(_registry_path(base, industry)),
        "feedback_path": str(_feedback_path(base, industry)),
        "status_counts": _summary(payload),
    }


def _promote_by_threshold(
    *,
    base: Path,
    industry: str,
    min_hit_count: int,
    min_confidence: float,
) -> dict[str, Any]:
    payload = _load_registry(base, industry)
    rows = payload.get("targets") or []
    promoted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "candidate").lower() != "candidate":
            continue
        hits = int(row.get("hit_count", 0) or 0)
        conf = float(row.get("confidence", 0.0) or 0.0)
        if hits >= int(min_hit_count) and conf >= float(min_confidence):
            row["status"] = "approved"
            row["promoted_at"] = _now_iso()
            promoted += 1
    payload["targets"] = rows
    _save_registry(base, industry, payload)
    _append_event(
        base,
        industry,
        {
            "event_time": _now_iso(),
            "event_type": "target_semantic_promotion_run",
            "industry": industry,
            "promoted_count": promoted,
            "min_hit_count": int(min_hit_count),
            "min_confidence": float(min_confidence),
        },
    )
    return {
        "industry": industry,
        "promoted_count": promoted,
        "min_hit_count": int(min_hit_count),
        "min_confidence": float(min_confidence),
        "registry_path": str(_registry_path(base, industry)),
        "feedback_path": str(_feedback_path(base, industry)),
        "status_counts": _summary(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote/review target semantics registry entries.")
    parser.add_argument("--base", default=".", help="Repository root path.")
    parser.add_argument("--industry", required=True, help="Industry key (e.g., ecommerce).")
    parser.add_argument("--target", default="", help="Specific target column to review.")
    parser.add_argument(
        "--status",
        default="approved",
        choices=sorted(VALID_STATUSES),
        help="Status to set when --target is provided.",
    )
    parser.add_argument("--note", default="", help="Review note to append.")
    parser.add_argument("--min-hit-count", type=int, default=3, help="Batch promotion threshold.")
    parser.add_argument("--min-confidence", type=float, default=0.9, help="Batch promotion threshold.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    industry = str(args.industry).strip()
    if not industry:
        raise SystemExit("--industry is required")

    if str(args.target or "").strip():
        result = _set_target_status(
            base=base,
            industry=industry,
            target_column=str(args.target).strip(),
            status=str(args.status).strip().lower(),
            note=str(args.note or "").strip(),
        )
    else:
        result = _promote_by_threshold(
            base=base,
            industry=industry,
            min_hit_count=int(args.min_hit_count),
            min_confidence=float(args.min_confidence),
        )

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
