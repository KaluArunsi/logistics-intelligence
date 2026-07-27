"""
Router-manifest based schema/category matcher.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from .alias_registry import AliasRegistryManager


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


class RouterManifestMatcher:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.alias_registry = AliasRegistryManager(self.base_path)
        self._manifests: dict[str, dict[str, Any]] = {}
        self._indexes: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        manifests_dir = self.base_path / "config" / "router" / "manifests"
        self._manifests = {}
        self._indexes = {}
        for path in sorted(manifests_dir.glob("*_router_manifest.json")):
            payload = json.loads(path.read_text())
            industry = str(payload.get("industry", "")).strip()
            if not industry:
                continue
            self._manifests[industry] = payload
            self._indexes[industry] = self._build_index(payload, industry=industry)

    def available_industries(self) -> list[str]:
        return sorted(self._manifests.keys())

    def _build_index(self, payload: dict[str, Any], *, industry: str) -> dict[str, Any]:
        categories_index: dict[str, Any] = {}
        for cat in payload.get("categories", []) or []:
            category = str(cat.get("category") or "")
            if not category:
                continue

            col_counter: Counter[str] = Counter()
            norm_to_display: dict[str, str] = {}
            alias_to_canonical: dict[str, str] = {}
            workers = cat.get("workers", []) or []

            for worker in workers:
                cols = worker.get("columns", {}) or {}
                for c in cols.get("dataset_columns", []) or []:
                    raw = str(c).strip()
                    norm = normalize_token(raw)
                    if norm:
                        col_counter[norm] += 1
                        if norm not in norm_to_display:
                            norm_to_display[norm] = raw
                for c in cols.get("model_feature_columns", []) or []:
                    raw = str(c).strip()
                    norm = normalize_token(raw)
                    if norm:
                        col_counter[norm] += 1
                        if norm not in norm_to_display:
                            norm_to_display[norm] = raw
                syn = cols.get("synonyms", {}) or {}
                for alias, canonical in (syn.get("alias_to_canonical") or {}).items():
                    a = normalize_token(alias)
                    c = normalize_token(canonical)
                    if a and c:
                        alias_to_canonical[a] = c
                        canonical_raw = str(canonical).strip()
                        if canonical_raw and c not in norm_to_display:
                            norm_to_display[c] = canonical_raw

            canonical_to_aliases = (
                (payload.get("alias_registry") or {}).get("canonical_to_aliases") or {}
            )
            for canonical, aliases in canonical_to_aliases.items():
                c = normalize_token(canonical)
                if not c:
                    continue
                canonical_raw = str(canonical).strip()
                if canonical_raw and c not in norm_to_display:
                    norm_to_display[c] = canonical_raw
                for alias in aliases or []:
                    a = normalize_token(alias)
                    if a:
                        alias_to_canonical[a] = c

            # Spec §13: only approved aliases affect canonical manifests/routing.
            # Candidates are kept for review/promote workflow, not live routing.
            runtime_aliases = self.alias_registry.aliases_for_category(
                industry=industry,
                category=category,
                include_candidates=False,
            )
            for alias, canonical in runtime_aliases.items():
                a = normalize_token(alias)
                c = normalize_token(canonical)
                if a and c:
                    alias_to_canonical[a] = c
                    canonical_raw = str(canonical).strip()
                    if canonical_raw and c not in norm_to_display:
                        norm_to_display[c] = canonical_raw

            # Use top recurring columns to avoid huge sparse unions.
            representative_norms = [name for name, _ in col_counter.most_common(30)]
            representative_columns = [norm_to_display.get(name, name) for name in representative_norms]
            categories_index[category] = {
                "task_type": cat.get("task_type"),
                "keywords": list(cat.get("keywords") or []),
                "representative_columns": representative_columns,
                "representative_norms": representative_norms,
                "representative_norm_to_column": norm_to_display,
                "alias_to_canonical": alias_to_canonical,
                "workers": workers,
            }
        return {"categories": categories_index}

    def _worker_overlap(self, worker: dict[str, Any], provided: set[str]) -> tuple[float, list[str], list[str]]:
        cols = worker.get("columns", {}) or {}
        raw_columns: list[str] = []
        for field in ("dataset_columns", "model_feature_columns"):
            for c in cols.get(field, []) or []:
                raw = str(c).strip()
                if raw:
                    raw_columns.append(raw)

        norm_to_raw: dict[str, str] = {}
        for raw in raw_columns:
            norm = normalize_token(raw)
            if norm and norm not in norm_to_raw:
                norm_to_raw[norm] = raw

        worker_norms = list(norm_to_raw.keys())
        if not worker_norms:
            return 0.0, [], []
        matched_norms = [n for n in worker_norms if n in provided]
        missing_norms = [n for n in worker_norms if n not in provided]
        # Keep overlap bounded to [0, 1] while still capping denominator to reduce
        # over-penalizing very wide schemas during routing.
        capped_denominator = max(1, min(25, len(worker_norms)))
        capped_matches = min(len(matched_norms), capped_denominator)
        score = capped_matches / capped_denominator
        matched = [norm_to_raw[n] for n in matched_norms]
        missing = [norm_to_raw[n] for n in missing_norms]
        return score, matched, missing

    def _category_match(self, category_data: dict[str, Any], provided: set[str]) -> dict[str, Any]:
        alias_map = category_data.get("alias_to_canonical", {}) or {}
        canonical_provided = set(provided)
        for col in list(provided):
            mapped = alias_map.get(col)
            if mapped:
                canonical_provided.add(mapped)

        representative_norms = list(category_data.get("representative_norms") or [])
        if not representative_norms:
            representative_norms = [
                normalize_token(c)
                for c in (category_data.get("representative_columns") or [])
                if str(c).strip()
            ]
        representative_norm_to_column = category_data.get("representative_norm_to_column") or {}
        matched_norms = [norm for norm in representative_norms if norm in canonical_provided]
        missing_norms = [norm for norm in representative_norms if norm not in canonical_provided]
        category_score = len(matched_norms) / max(1, len(representative_norms))
        representative_missing = [
            representative_norm_to_column.get(norm, norm)
            for norm in missing_norms
        ]
        matched = [
            representative_norm_to_column.get(norm, norm)
            for norm in matched_norms
        ]

        worker_scores: list[dict[str, Any]] = []
        best_worker_score = 0.0
        best_worker_missing: list[str] = []
        for worker in category_data.get("workers", []) or []:
            worker_score, worker_matched, worker_missing = self._worker_overlap(worker, canonical_provided)
            if worker_score <= 0:
                continue
            if worker_score > best_worker_score:
                best_worker_score = worker_score
                best_worker_missing = worker_missing
            worker_scores.append(
                {
                    "worker_model_id": worker.get("worker_model_id"),
                    "worker_dataset_id": worker.get("worker_dataset_id"),
                    "score": round(float(worker_score), 4),
                    "matched_columns": worker_matched[:10],
                    "missing_columns": worker_missing[:15],
                }
            )

        worker_scores.sort(key=lambda row: row["score"], reverse=True)
        if worker_scores:
            best_worker_missing = list(worker_scores[0].get("missing_columns") or [])

        # Use the better of category-level and best worker-level score.
        # A valid worker-specific dataset with 8 matching columns should not
        # be penalized because the category union has 30 representative columns.
        score = max(best_worker_score, category_score)
        missing_for_guidance = best_worker_missing or representative_missing

        return {
            "score": round(float(score), 4),
            "category_score": round(float(category_score), 4),
            "best_worker_score": round(float(best_worker_score), 4),
            "matched_fields": matched,
            "missing_fields": missing_for_guidance[:15],
            "fallback_models": [row["worker_model_id"] for row in worker_scores[:5] if row.get("worker_model_id")],
            "top_workers": worker_scores[:5],
        }

    def match(
        self,
        provided_columns: list[str],
        industry: Optional[str] = None,
        semantic_shadow: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        provided = {normalize_token(c) for c in provided_columns if str(c).strip()}
        if not provided:
            return {
                "decision": "REJECT",
                "industry": industry,
                "top_category": None,
                "schema_coverage": 0.0,
                "deterministic_score": 0.0,
                "semantic_score": 0.0,
                "blended_score": 0.0,
                "missing_fields": [],
                "fallback_models": [],
                "next_actions": ["upload_or_provide_columns"],
                "candidates": [],
            }

        semantic_by_category: dict[str, float] = {}
        if isinstance(semantic_shadow, dict):
            rows = semantic_shadow.get("category_hypotheses")
            if not isinstance(rows, list) or not rows:
                rows = semantic_shadow.get("categories")
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                category = str(row.get("category") or "").strip()
                if not category:
                    continue
                score = float(row.get("confidence", row.get("score", 0.0)) or 0.0)
                semantic_by_category[category] = max(0.0, min(1.0, score))

        industries = [industry] if industry else self.available_industries()
        best: Optional[dict[str, Any]] = None
        candidates: list[dict[str, Any]] = []

        for ind in industries:
            idx = self._indexes.get(ind, {})
            categories = idx.get("categories", {}) or {}
            for category, category_data in categories.items():
                details = self._category_match(category_data, provided)
                row = {
                    "industry": ind,
                    "category": category,
                    "score": details["score"],
                    "deterministic_score": details["score"],
                    "semantic_score": semantic_by_category.get(category, 0.0),
                    "blended_score": round((details["score"] * 0.85) + (semantic_by_category.get(category, 0.0) * 0.15), 4),
                    "missing_fields": details["missing_fields"],
                    "fallback_models": details["fallback_models"],
                    "top_workers": details["top_workers"],
                    "task_type": category_data.get("task_type"),
                }
                candidates.append(row)
                if best is None:
                    best = row
                else:
                    row_blended = float(row.get("blended_score") or row.get("score") or 0.0)
                    best_blended = float(best.get("blended_score") or best.get("score") or 0.0)
                    row_det = float(row.get("deterministic_score") or row.get("score") or 0.0)
                    best_det = float(best.get("deterministic_score") or best.get("score") or 0.0)
                    if row_blended > best_blended or (row_blended == best_blended and row_det > best_det):
                        best = row

        candidates.sort(
            key=lambda row: (
                float(row.get("blended_score") or row.get("score") or 0.0),
                float(row.get("deterministic_score") or row.get("score") or 0.0),
            ),
            reverse=True,
        )
        best = best or {
            "industry": industry,
            "category": None,
            "score": 0.0,
            "deterministic_score": 0.0,
            "semantic_score": 0.0,
            "blended_score": 0.0,
            "missing_fields": [],
            "fallback_models": [],
            "top_workers": [],
            "task_type": None,
        }

        score = float(best.get("deterministic_score", best.get("score", 0.0)))
        semantic_score = float(best.get("semantic_score") or 0.0)
        blended_score = float(best.get("blended_score") or score)
        min_accept = max(0.0, min(1.0, float(os.getenv("MIN_MATCH_SCORE", "0.80"))))
        partial_floor = max(0.0, min(min_accept, float(os.getenv("PARTIAL_MATCH_FLOOR", "0.20"))))
        if score >= min_accept:
            decision = "ACCEPT"
            next_actions = ["run_predict"]
        elif score >= partial_floor:
            decision = "PARTIAL_ACCEPT"
            next_actions = ["provide_missing", "run_fallback"]
        else:
            decision = "REJECT"
            next_actions = ["upload_better_schema", "provide_missing"]

        return {
            "decision": decision,
            "industry": best.get("industry"),
            "top_category": best.get("category"),
            "schema_coverage": round(score, 4),
            "deterministic_score": round(score, 4),
            "semantic_score": round(semantic_score, 4),
            "blended_score": round(blended_score, 4),
            "missing_fields": list(best.get("missing_fields") or []),
            "fallback_models": list(best.get("fallback_models") or []),
            "top_workers": list(best.get("top_workers") or []),
            "task_type": best.get("task_type"),
            "next_actions": next_actions,
            "candidates": candidates[:10],
        }

    def category_columns(self, industry: str, category: str) -> list[str]:
        payload = self._manifests.get(industry) or {}
        for cat in payload.get("categories", []) or []:
            if str(cat.get("category")) != category:
                continue
            cols = set()
            for worker in cat.get("workers", []) or []:
                meta = worker.get("columns") or {}
                for col in meta.get("dataset_columns", []) or []:
                    if str(col).strip():
                        cols.add(str(col))
                for col in meta.get("model_feature_columns", []) or []:
                    if str(col).strip():
                        cols.add(str(col))
            return sorted(cols)
        return []

    def category_worker_dataset_ids(self, industry: str, category: str) -> list[str]:
        payload = self._manifests.get(industry) or {}
        for cat in payload.get("categories", []) or []:
            if str(cat.get("category")) != category:
                continue
            out = []
            for worker in cat.get("workers", []) or []:
                dataset_id = str(worker.get("worker_dataset_id") or "").strip()
                if dataset_id:
                    out.append(dataset_id)
            return sorted(set(out))
        return []
