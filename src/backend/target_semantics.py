"""
Target semantics resolver and registry manager.

Infers meaning for ambiguous model targets using:
- target/feature naming semantics
- lightweight feature-importance signals from FE metadata
- target distribution hints from worker data files
- historical runtime outcomes from past reports
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "").strip() if ch.isalnum())


def _split_tokens(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    step = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    parts = re.split(r"[^a-zA-Z0-9]+", step)
    out = []
    for p in parts:
        p = p.strip().lower()
        if p:
            out.append(p)
    return out


class TargetSemanticsResolver:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.registry_dir = self.base_path / "config" / "router" / "target_semantics"
        self.feedback_dir = self.base_path / "reports" / "router_feedback"
        self.reports_dir = self.base_path / "exports" / "runtime_reports"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_cache: dict[str, dict[str, Any]] = {}
        self._worker_index_cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._history_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})

    def _registry_path(self, industry: str) -> Path:
        return self.registry_dir / f"{industry}_target_semantics.json"

    def _feedback_path(self, industry: str) -> Path:
        return self.feedback_dir / f"{industry}_target_semantic_events.jsonl"

    def load_registry(self, industry: str) -> dict[str, Any]:
        path = self._registry_path(industry)
        if not path.exists():
            return {
                "version": 1,
                "industry": industry,
                "updated_at": _now_iso(),
                "targets": [],
            }
        try:
            payload = json.loads(path.read_text())
        except Exception:
            payload = {}
        payload.setdefault("version", 1)
        payload.setdefault("industry", industry)
        payload.setdefault("updated_at", _now_iso())
        payload.setdefault("targets", [])
        return payload

    def save_registry(self, industry: str, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now_iso()
        path = self._registry_path(industry)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def _append_event(self, industry: str, event: dict[str, Any]) -> None:
        path = self._feedback_path(industry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _manifest(self, industry: str) -> dict[str, Any]:
        cached = self._manifest_cache.get(industry)
        if cached is not None:
            return cached
        path = self.base_path / "config" / "router" / "manifests" / f"{industry}_router_manifest.json"
        payload = json.loads(path.read_text()) if path.exists() else {"industry": industry, "categories": []}
        self._manifest_cache[industry] = payload
        return payload

    def _worker_index(self, industry: str) -> dict[str, dict[str, Any]]:
        cached = self._worker_index_cache.get(industry)
        if cached is not None:
            return cached
        out: dict[str, dict[str, Any]] = {}
        manifest = self._manifest(industry)
        for cat in manifest.get("categories", []) or []:
            category = str(cat.get("category") or "").strip()
            for worker in cat.get("workers", []) or []:
                wid = str(worker.get("worker_model_id") or "").strip()
                if not wid:
                    continue
                row = dict(worker)
                row["_category"] = category
                out[wid] = row
        self._worker_index_cache[industry] = out
        return out

    @staticmethod
    def _extract_importance_features(fe_meta: dict[str, Any], fallback_features: list[str]) -> list[str]:
        # Try common shapes for importance payloads.
        for key in ("feature_importances", "feature_importance", "importances", "importance"):
            value = fe_meta.get(key)
            if isinstance(value, dict):
                scored: list[tuple[str, float]] = []
                for k, v in value.items():
                    try:
                        scored.append((str(k), float(v)))
                    except Exception:
                        continue
                scored.sort(key=lambda x: abs(x[1]), reverse=True)
                return [name for name, _ in scored[:20]]
            if isinstance(value, list):
                scored = []
                for row in value:
                    if isinstance(row, dict):
                        name = str(row.get("feature") or row.get("name") or "").strip()
                        try:
                            score = float(row.get("importance", row.get("value", 0.0)) or 0.0)
                        except Exception:
                            score = 0.0
                        if name:
                            scored.append((name, score))
                if scored:
                    scored.sort(key=lambda x: abs(x[1]), reverse=True)
                    return [name for name, _ in scored[:20]]
        kept = [str(x) for x in (fe_meta.get("kept_columns") or []) if str(x).strip()]
        if kept:
            return kept[:20]
        return list(fallback_features)[:20]

    @staticmethod
    def _read_fe_metadata(base_path: Path, worker: dict[str, Any]) -> dict[str, Any]:
        fe_meta_path = str((worker.get("paths") or {}).get("feature_metadata_path") or "").strip()
        if not fe_meta_path:
            return {}
        path = (base_path / fe_meta_path).resolve()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    @staticmethod
    def _distribution_hints(base_path: Path, worker: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        paths = worker.get("paths") or {}
        data_file = str(paths.get("data_file") or "").strip()
        target = str(worker.get("target_column") or "").strip()
        if not data_file or not target:
            return {}, []
        path = (base_path / data_file).resolve()
        if not path.exists():
            return {}, []
        try:
            lower = path.name.lower()
            if lower.endswith(".parquet") or lower.endswith(".parquet.zstd") or lower.endswith(".pq"):
                df = pl.scan_parquet(path).limit(2000).collect()
            elif lower.endswith(".csv") or lower.endswith(".txt"):
                df = pl.read_csv(path, infer_schema_length=1000, ignore_errors=True).head(2000)
            elif lower.endswith(".json"):
                df = pl.read_json(path).head(2000)
            else:
                return {}, []
            if target not in df.columns:
                return {}, []
            series = df[target].drop_nulls()
            if series.len() == 0:
                return {}, []
            hints: list[str] = []
            summary: dict[str, Any] = {"n": int(series.len()), "dtype": str(series.dtype)}
            if series.dtype.is_numeric():
                mean_v = float(series.mean() or 0.0)
                min_v = float(series.min() or 0.0)
                max_v = float(series.max() or 0.0)
                summary.update(
                    {
                        "mean": round(mean_v, 6),
                        "min": round(min_v, 6),
                        "max": round(max_v, 6),
                        "std": round(float(series.std() or 0.0), 6),
                    }
                )
                if min_v >= 0 and max_v <= 1.0:
                    hints.append("bounded_ratio_target")
                if max_v > 100000:
                    hints.append("high_magnitude_target")
                if min_v >= 0:
                    hints.append("non_negative_target")
            else:
                summary["n_unique"] = int(series.n_unique())
                hints.append("categorical_or_text_target")
            return summary, hints
        except Exception:
            return {}, []

    def _historical_index(self, max_reports: int = 120, ttl_seconds: int = 180) -> dict[str, dict[str, Any]]:
        cached_at, cached = self._history_cache
        now = time.time()
        if now - cached_at <= max(1, int(ttl_seconds)) and cached:
            return cached
        if not self.reports_dir.exists():
            return {}
        files = sorted(self.reports_dir.glob("rep_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        by_worker: dict[str, dict[str, Any]] = {}
        scanned = 0
        for path in files:
            if scanned >= max_reports:
                break
            scanned += 1
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            runtime = payload.get("runtime_inference") or {}
            rows = runtime.get("l0_workers") or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                wid = str(row.get("worker_model_id") or "").strip()
                if not wid:
                    continue
                entry = by_worker.setdefault(
                    wid,
                    {
                        "means": [],
                        "confs": [],
                        "task_types": {},
                    },
                )
                try:
                    entry["means"].append(float(row.get("prediction_mean") or 0.0))
                    entry["confs"].append(float(row.get("confidence_mean") or 0.0))
                except Exception:
                    pass
                task = str(row.get("task_type") or "").strip().lower()
                if task:
                    tt = entry["task_types"]
                    tt[task] = tt.get(task, 0) + 1

        out: dict[str, dict[str, Any]] = {}
        for wid, entry in by_worker.items():
            means = list(entry.get("means") or [])
            confs = list(entry.get("confs") or [])
            task_types = dict(entry.get("task_types") or {})
            if not means:
                continue
            dominant = sorted(task_types.items(), key=lambda x: x[1], reverse=True)[0][0] if task_types else None
            out[wid] = {
                "runs": len(means),
                "prediction_mean_avg": round(sum(means) / max(1, len(means)), 6),
                "prediction_mean_abs_avg": round(sum(abs(x) for x in means) / max(1, len(means)), 6),
                "confidence_avg": round(sum(confs) / max(1, len(confs)), 6) if confs else None,
                "dominant_task_type": dominant,
            }
        self._history_cache = (now, out)
        return out

    def _historical_worker_stats(self, worker_model_id: str, max_reports: int = 120) -> dict[str, Any]:
        index = self._historical_index(max_reports=max_reports)
        return dict(index.get(worker_model_id) or {})

    @staticmethod
    def _meaning_heuristics(
        target_column: str,
        feature_columns: list[str],
        importance_features: list[str],
        distribution_hints: list[str],
        historical_stats: dict[str, Any],
    ) -> tuple[str, float, list[str]]:
        target_tokens = _split_tokens(target_column)
        all_tokens = set(target_tokens)
        for col in feature_columns[:40]:
            all_tokens.update(_split_tokens(col))
        for col in importance_features[:20]:
            all_tokens.update(_split_tokens(col))

        candidates: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        def vote(label: str, weight: float, reason: str) -> None:
            candidates[label] = candidates.get(label, 0.0) + weight
            evidence.setdefault(label, []).append(reason)

        tnorm = _norm(target_column)
        if tnorm in {"tp", "txnprocessed", "transactionsprocessed"}:
            vote("transactions processed", 0.90, f"target_token:{target_column}")
        if "rul" in tnorm:
            vote("remaining useful life", 0.88, "target_token:rul")
        if "eta" in tnorm or "delay" in tnorm:
            vote("arrival delay / ETA performance", 0.82, "target_token:eta_or_delay")
        if "risk" in tnorm or "severity" in tnorm:
            vote("operational risk level", 0.84, "target_token:risk_or_severity")
        if "sales" in tnorm or "revenue" in tnorm or "gmv" in tnorm:
            vote("sales / revenue outcome", 0.82, "target_token:sales_or_revenue")
        if "cost" in tnorm or "spend" in tnorm:
            vote("cost / spend outcome", 0.80, "target_token:cost_or_spend")
        if "count" in tnorm or "volume" in tnorm or "units" in tnorm:
            vote("volume / count outcome", 0.76, "target_token:count_or_volume")
        if tnorm in {"value", "value1", "value2", "obsvalue", "obs"}:
            vote("observed value metric", 0.60, "ambiguous_target_token")

        semantic_bags = {
            "transactions processed": {"transaction", "transactions", "txn", "processed", "orders", "throughput"},
            "volume / count outcome": {"volume", "count", "units", "shipments", "deliveries", "teu", "tons"},
            "arrival delay / ETA performance": {"eta", "arrival", "delay", "travel", "transit", "duration", "time"},
            "sales / revenue outcome": {"sales", "revenue", "gmv", "basket", "aov", "conversion", "checkout"},
            "cost / spend outcome": {"cost", "spend", "price", "fare", "expense", "fuel"},
            "operational risk level": {"risk", "incident", "safety", "fraud", "violation", "severity"},
        }
        for label, bag in semantic_bags.items():
            overlap = sorted(all_tokens.intersection(bag))
            if overlap:
                w = min(0.45, 0.12 * len(overlap))
                vote(label, w, f"feature_token_overlap:{','.join(overlap[:6])}")

        for hint in distribution_hints:
            if hint == "bounded_ratio_target":
                vote("rate / probability metric", 0.25, "distribution:bounded_ratio_target")
            elif hint == "high_magnitude_target":
                vote("sales / revenue outcome", 0.12, "distribution:high_magnitude_target")
            elif hint == "non_negative_target":
                vote("volume / count outcome", 0.08, "distribution:non_negative_target")

        dominant_task = str(historical_stats.get("dominant_task_type") or "").lower()
        if dominant_task == "classification":
            vote("classification label", 0.15, "historical:classification_worker")
        if historical_stats.get("runs", 0) >= 5:
            vote("operational target metric", 0.08, "historical:stable_worker_usage")

        if not candidates:
            return "operational target metric", 0.45, ["fallback:no_signal"]

        top = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[0]
        meaning, score = top
        conf = min(0.97, max(0.35, score))
        return meaning, round(conf, 4), (evidence.get(meaning) or ["heuristic_vote"])[:10]

    def _upsert_target_semantic(
        self,
        *,
        industry: str,
        category: str,
        worker_model_id: str,
        worker_dataset_id: Optional[str],
        target_column: str,
        meaning: str,
        confidence: float,
        evidence: list[str],
        feature_evidence: list[str],
        distribution_summary: dict[str, Any],
        historical_summary: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self.load_registry(industry)
        rows = list(payload.get("targets") or [])
        now_iso = _now_iso()
        existing = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            if _norm(row.get("target_column", "")) == _norm(target_column):
                existing = row
                break

        if existing is None:
            existing = {
                "target_column": target_column,
                "meaning": meaning,
                "status": "candidate",
                "confidence": float(confidence),
                "source": "target_semantics_resolver",
                "categories": [category] if category else [],
                "worker_model_ids": [worker_model_id] if worker_model_id else [],
                "worker_dataset_ids": [worker_dataset_id] if worker_dataset_id else [],
                "feature_evidence": feature_evidence[:20],
                "distribution_summary": distribution_summary,
                "historical_summary": historical_summary,
                "evidence": evidence[:20],
                "first_seen": now_iso,
                "last_seen": now_iso,
                "hit_count": 1,
            }
            rows.append(existing)
        else:
            existing["confidence"] = max(float(existing.get("confidence", 0.0) or 0.0), float(confidence))
            existing["last_seen"] = now_iso
            existing["hit_count"] = int(existing.get("hit_count", 0) or 0) + 1
            if not str(existing.get("meaning") or "").strip():
                existing["meaning"] = meaning
            categories = set(str(x) for x in (existing.get("categories") or []) if str(x).strip())
            if category:
                categories.add(category)
            existing["categories"] = sorted(categories)
            worker_ids = set(str(x) for x in (existing.get("worker_model_ids") or []) if str(x).strip())
            if worker_model_id:
                worker_ids.add(worker_model_id)
            existing["worker_model_ids"] = sorted(worker_ids)
            ds_ids = set(str(x) for x in (existing.get("worker_dataset_ids") or []) if str(x).strip())
            if worker_dataset_id:
                ds_ids.add(worker_dataset_id)
            existing["worker_dataset_ids"] = sorted(ds_ids)
            merged_evidence = list(existing.get("evidence") or []) + list(evidence or [])
            dedup_evidence: list[str] = []
            seen = set()
            for ev in merged_evidence:
                token = str(ev).strip()
                if not token or token in seen:
                    continue
                seen.add(token)
                dedup_evidence.append(token)
            existing["evidence"] = dedup_evidence[-30:]
            if feature_evidence:
                existing["feature_evidence"] = list(feature_evidence[:20])
            if distribution_summary:
                existing["distribution_summary"] = dict(distribution_summary)
            if historical_summary:
                existing["historical_summary"] = dict(historical_summary)
            status = str(existing.get("status") or "").lower()
            if status not in {"approved", "rejected"}:
                existing["status"] = "candidate"

        payload["targets"] = rows
        self.save_registry(industry, payload)
        status = str(existing.get("status") or "candidate")
        event = {
            "event_time": now_iso,
            "event_type": "target_semantic_inference",
            "industry": industry,
            "category": category,
            "worker_model_id": worker_model_id,
            "worker_dataset_id": worker_dataset_id,
            "target_column": target_column,
            "meaning": meaning,
            "confidence": float(confidence),
            "status": status,
            "evidence": evidence[:12],
        }
        self._append_event(industry, event)
        return {
            "target_column": target_column,
            "meaning": meaning,
            "confidence": round(float(confidence), 4),
            "status": status,
            "evidence": evidence[:12],
            "worker_model_id": worker_model_id,
            "category": category,
        }

    def infer_for_worker(
        self,
        *,
        industry: str,
        worker_model_id: str,
        category: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        worker_idx = self._worker_index(industry)
        worker = worker_idx.get(worker_model_id)
        if not worker:
            return None
        target_column = str(worker.get("target_column") or "").strip()
        if not target_column:
            return None

        resolved_category = str(category or worker.get("_category") or "").strip()
        existing = None
        registry = self.load_registry(industry)
        for row in registry.get("targets", []) or []:
            if not isinstance(row, dict):
                continue
            if _norm(row.get("target_column", "")) == _norm(target_column):
                existing = row
                break
        if existing is not None and str(existing.get("status") or "").lower() == "approved":
            return {
                "target_column": target_column,
                "meaning": str(existing.get("meaning") or "").strip() or "operational target metric",
                "confidence": round(float(existing.get("confidence", 0.0) or 0.0), 4),
                "status": "approved",
                "evidence": list(existing.get("evidence") or [])[:12],
                "worker_model_id": worker_model_id,
                "category": resolved_category,
            }

        columns_meta = worker.get("columns") or {}
        feature_columns = [str(x) for x in (columns_meta.get("model_feature_columns") or []) if str(x).strip()]
        fe_meta = self._read_fe_metadata(self.base_path, worker)
        importance_features = self._extract_importance_features(fe_meta, feature_columns)
        distribution_summary, distribution_hints = self._distribution_hints(self.base_path, worker)
        historical_summary = self._historical_worker_stats(worker_model_id)

        meaning, confidence, evidence = self._meaning_heuristics(
            target_column=target_column,
            feature_columns=feature_columns,
            importance_features=importance_features,
            distribution_hints=distribution_hints,
            historical_stats=historical_summary,
        )

        return self._upsert_target_semantic(
            industry=industry,
            category=resolved_category,
            worker_model_id=worker_model_id,
            worker_dataset_id=str(worker.get("worker_dataset_id") or "").strip() or None,
            target_column=target_column,
            meaning=meaning,
            confidence=confidence,
            evidence=evidence,
            feature_evidence=importance_features[:12],
            distribution_summary=distribution_summary,
            historical_summary=historical_summary,
        )

    def infer_from_match(
        self,
        *,
        industry: str,
        match: dict[str, Any],
        top_k_workers: int = 3,
    ) -> dict[str, Any]:
        workers = list(match.get("top_workers") or [])
        candidates: list[dict[str, Any]] = []
        for row in workers[: max(1, int(top_k_workers))]:
            worker_model_id = str(row.get("worker_model_id") or "").strip()
            if not worker_model_id:
                continue
            guess = self.infer_for_worker(
                industry=industry,
                worker_model_id=worker_model_id,
                category=str(match.get("top_category") or "").strip(),
            )
            if guess:
                candidates.append(guess)
        candidates.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
        top_guess = candidates[0] if candidates else None
        return {
            "industry": industry,
            "top_guess": top_guess,
            "candidates": candidates[:10],
        }
