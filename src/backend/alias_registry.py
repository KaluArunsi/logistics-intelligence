"""
Runtime alias registry manager for self-healing router synonyms.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value: str) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


class AliasRegistryManager:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.registry_dir = self.base_path / "config" / "router" / "alias_registry"
        self.feedback_dir = self.base_path / "reports" / "router_feedback"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def _registry_path(self, industry: str) -> Path:
        return self.registry_dir / f"{industry}_column_aliases.json"

    def _feedback_path(self, industry: str) -> Path:
        return self.feedback_dir / f"{industry}_alias_events.jsonl"

    def load(self, industry: str) -> dict[str, Any]:
        path = self._registry_path(industry)
        if not path.exists():
            return {
                "version": 1,
                "industry": industry,
                "updated_at": _now_iso(),
                "aliases": [],
            }
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {}
        payload.setdefault("version", 1)
        payload.setdefault("industry", industry)
        payload.setdefault("updated_at", _now_iso())
        payload.setdefault("aliases", [])
        return payload

    def save(self, industry: str, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        path = self._registry_path(industry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _append_feedback(self, industry: str, event: dict[str, Any]) -> None:
        path = self._feedback_path(industry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(event) + "\n")

    def record_aliases(
        self,
        *,
        industry: str,
        category: Optional[str],
        aliases: list[dict[str, Any]],
        worker_dataset_ids: Optional[list[str]] = None,
        source: str = "llm_inference",
    ) -> dict[str, Any]:
        if not aliases:
            return {"recorded": 0, "approved": 0, "candidate": 0, "rejected": 0}

        worker_dataset_ids = list(worker_dataset_ids or [])
        payload = self.load(industry)
        rows = list(payload.get("aliases") or [])

        approved = 0
        candidate = 0
        rejected = 0
        recorded = 0

        for item in aliases:
            canonical = str(item.get("canonical_column") or "").strip()
            alias = str(item.get("alias") or "").strip()
            confidence = float(item.get("confidence", 0.0) or 0.0)
            note = str(item.get("note") or "").strip()
            evidence = [str(x).strip() for x in (item.get("evidence") or []) if str(x).strip()]
            if note and note not in evidence:
                evidence.append(note)
            if not canonical or not alias:
                continue
            if _norm(canonical) == _norm(alias):
                continue

            recorded += 1
            now_iso = _now_iso()
            existing = None
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if _norm(row.get("canonical_column", "")) != _norm(canonical):
                    continue
                if _norm(row.get("alias", "")) != _norm(alias):
                    continue
                existing = row
                break

            if existing is None:
                existing = {
                    "canonical_column": canonical,
                    "alias": alias,
                    "status": "candidate",
                    "confidence": confidence,
                    "source": source,
                    "categories": [category] if category else [],
                    "worker_dataset_ids": list(worker_dataset_ids),
                    "first_seen": now_iso,
                    "last_seen": now_iso,
                    "hit_count": 1,
                    "notes": [note] if note else [],
                    "semantic_evidence": evidence[:20],
                }
                rows.append(existing)
            else:
                existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), confidence)
                existing["source"] = source
                existing["last_seen"] = now_iso
                existing["hit_count"] = int(existing.get("hit_count", 0) or 0) + 1
                cats = set(str(x) for x in (existing.get("categories") or []) if str(x).strip())
                if category:
                    cats.add(category)
                existing["categories"] = sorted(cats)
                worker_set = set(str(x) for x in (existing.get("worker_dataset_ids") or []) if str(x).strip())
                worker_set.update(str(x) for x in worker_dataset_ids if str(x).strip())
                existing["worker_dataset_ids"] = sorted(worker_set)
                if note:
                    notes = list(existing.get("notes") or [])
                    notes.append(note)
                    existing["notes"] = notes[-20:]
                if evidence:
                    prior = list(existing.get("semantic_evidence") or [])
                    prior.extend(evidence)
                    deduped: list[str] = []
                    seen_evidence: set[str] = set()
                    for ev in prior:
                        token = str(ev).strip()
                        if not token or token in seen_evidence:
                            continue
                        seen_evidence.add(token)
                        deduped.append(token)
                    existing["semantic_evidence"] = deduped[-30:]

            # Candidate-only default for runtime self-healing.
            # Explicit review/promote tooling is responsible for approval.
            status = str(existing.get("status") or "").lower()
            if status not in {"approved", "rejected"}:
                existing["status"] = "candidate"

            status = str(existing.get("status") or "candidate")
            if status == "approved":
                approved += 1
            elif status == "rejected":
                rejected += 1
            else:
                candidate += 1

            self._append_feedback(
                industry,
                {
                    "event_time": now_iso,
                    "event_type": "alias_inference",
                    "industry": industry,
                    "category": category,
                    "canonical_column": canonical,
                    "alias": alias,
                    "confidence": confidence,
                    "status": status,
                    "source": source,
                    "worker_dataset_ids": list(worker_dataset_ids),
                    "note": note,
                    "semantic_evidence": evidence[:10],
                },
            )

        payload["aliases"] = rows
        self.save(industry, payload)
        return {
            "recorded": recorded,
            "approved": approved,
            "candidate": candidate,
            "rejected": rejected,
        }

    def aliases_for_category(
        self,
        *,
        industry: str,
        category: Optional[str] = None,
        include_candidates: bool = True,
    ) -> dict[str, str]:
        payload = self.load(industry)
        alias_to_canonical: dict[str, str] = {}

        for row in payload.get("aliases", []) or []:
            if not isinstance(row, dict):
                continue
            status = str(row.get("status", "candidate")).lower()
            if status not in {"approved", "candidate"}:
                continue
            if status == "candidate" and not include_candidates:
                continue
            if status == "candidate":
                conf = float(row.get("confidence", 0.0) or 0.0)
                hits = int(row.get("hit_count", 0) or 0)
                if conf < 0.95 or hits < 3:
                    continue

            categories = set(str(x) for x in (row.get("categories") or []) if str(x).strip())
            if category and categories and category not in categories:
                continue

            canonical = str(row.get("canonical_column") or "").strip()
            alias = str(row.get("alias") or "").strip()
            if not canonical or not alias:
                continue
            alias_to_canonical[alias] = canonical

        return alias_to_canonical
