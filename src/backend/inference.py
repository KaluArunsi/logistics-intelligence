"""
Runtime inference adapter: L0 workers -> L1 expert aggregation.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import joblib
import numpy as np
import polars as pl
import torch
from scipy import sparse

if TYPE_CHECKING:
    from src.models.l1_experts.ensemble import L0WorkerEnsemble
    from src.models.l1_experts.trainer import L1Trainer

_RUNTIME_MODEL_IMPORT_ERROR: Exception | None = None
try:
    from src.models.l1_experts.ensemble import L0WorkerEnsemble  # type: ignore[assignment]
    from src.models.l1_experts.trainer import L1Trainer  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - deployment fallback path
    # Keep import-time safe for deployments that run Toji unified flow only.
    L0WorkerEnsemble = Any  # type: ignore[misc,assignment]
    L1Trainer = Any  # type: ignore[misc,assignment]
    _RUNTIME_MODEL_IMPORT_ERROR = exc

from .alias_registry import AliasRegistryManager


@dataclass
class WorkerRuntimeResult:
    worker_model_id: str
    task_type: str
    prediction_mean: float
    prediction_std: float
    confidence_mean: float
    n_rows: int
    errors: Optional[str] = None


class RuntimeInferenceEngine:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._alias_registry = AliasRegistryManager(self.base_path)
        self._manifest_cache: dict[str, dict[str, Any]] = {}
        self._l1_trainers: dict[str, L1Trainer] = {}
        self._ensemble_cache: dict[str, L0WorkerEnsemble] = {}
        self._runtime_models_available = _RUNTIME_MODEL_IMPORT_ERROR is None

    def _ensure_runtime_models(self) -> None:
        if self._runtime_models_available:
            return
        raise RuntimeError(
            "Legacy worker-model runtime is unavailable in this deployment. "
            "Use Toji unified inference flow."
        ) from _RUNTIME_MODEL_IMPORT_ERROR

    def _load_manifest(self, industry: str) -> dict[str, Any]:
        if industry in self._manifest_cache:
            return self._manifest_cache[industry]
        path = self.base_path / "config" / "router" / "manifests" / f"{industry}_router_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing router manifest for industry={industry}")
        payload = json.loads(path.read_text())
        self._manifest_cache[industry] = payload
        return payload

    def _get_ensemble(self, industry: str) -> L0WorkerEnsemble:
        self._ensure_runtime_models()
        if industry not in self._ensemble_cache:
            self._ensemble_cache[industry] = L0WorkerEnsemble(self.base_path, industry=industry)
        return self._ensemble_cache[industry]

    def _get_l1_trainer(self, industry: str) -> L1Trainer:
        self._ensure_runtime_models()
        if industry not in self._l1_trainers:
            self._l1_trainers[industry] = L1Trainer(self.base_path, industry=industry)
        return self._l1_trainers[industry]

    def _get_category_workers(self, manifest: dict[str, Any], category: str) -> list[dict[str, Any]]:
        for cat in manifest.get("categories", []) or []:
            if str(cat.get("category")) == category:
                return list(cat.get("workers") or [])
        return []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _deterministic_bucket(value: Any, mod: int = 256) -> float:
        token = str(value).encode("utf-8")
        digest = hashlib.sha256(token).hexdigest()
        return float(int(digest[:8], 16) % mod)

    @staticmethod
    def _decode_pred_if_needed(pred: np.ndarray, fe_metadata: dict[str, Any]) -> np.ndarray:
        label_classes = fe_metadata.get("label_encoder_classes")
        if not label_classes:
            return pred
        classes = np.asarray(label_classes, dtype=object)
        try:
            pred_idx = np.asarray(pred).astype(int)
            valid = (pred_idx >= 0) & (pred_idx < len(classes))
            out = np.asarray(pred, dtype=object)
            out[valid] = classes[pred_idx[valid]]
            return out
        except Exception:
            return pred

    @staticmethod
    def _to_numeric(pred: np.ndarray) -> np.ndarray:
        out = np.zeros(len(pred), dtype=np.float32)
        for i, value in enumerate(pred):
            try:
                out[i] = float(value)
            except Exception:
                out[i] = RuntimeInferenceEngine._deterministic_bucket(value)
        return out

    def _merge_synonyms(
        self,
        *,
        industry: str,
        category: str,
        worker_row: dict[str, Any],
    ) -> dict[str, Any]:
        columns_meta = worker_row.get("columns") or {}
        worker_synonyms = (columns_meta.get("synonyms") or {}).get("alias_to_canonical") or {}
        runtime_aliases = self._alias_registry.aliases_for_category(
            industry=industry,
            category=category,
            include_candidates=True,
        )
        alias_to_canonical = dict(runtime_aliases)
        alias_to_canonical.update({str(k): str(v) for k, v in worker_synonyms.items() if str(k).strip() and str(v).strip()})
        return {"alias_to_canonical": alias_to_canonical}

    def _apply_runtime_transforms(
        self,
        *,
        df: pl.DataFrame,
        feature_columns: list[str],
        fe_metadata: dict[str, Any],
        synonyms: dict[str, Any],
        ensemble: L0WorkerEnsemble,
    ):
        """
        Apply FE metadata transforms on runtime data and output worker-ready matrix.
        """
        local = df.clone()
        alias_to_canonical = (synonyms or {}).get("alias_to_canonical") or {}

        # Canonicalize alias columns.
        rename_map = {}
        existing = set(local.columns)
        for alias, canonical in alias_to_canonical.items():
            if alias in existing and canonical and canonical not in existing:
                rename_map[alias] = canonical
        if rename_map:
            local = local.rename(rename_map)

        # Ensure referenced columns exist.
        needed = set(feature_columns)
        needed.update((fe_metadata.get("encodings") or {}).keys())
        needed.update((fe_metadata.get("numeric_coercions") or {}).keys())
        for col in sorted(needed):
            if col not in local.columns:
                local = local.with_columns(pl.lit(None).alias(col))

        # FE filter replay if feasible.
        filt = fe_metadata.get("filter") or {}
        if isinstance(filt, dict) and filt.get("column") in local.columns:
            try:
                local = local.filter(pl.col(filt["column"]).is_in(filt.get("values", [])))
            except Exception:
                pass
        if local.height == 0:
            # Preserve at least one row shape for downstream consistency.
            local = df.head(1).clone()

        # Replay encodings like training.
        encodings = fe_metadata.get("encodings") or {}
        for col, raw_mapping in encodings.items():
            if col not in local.columns:
                continue
            method, mapping = ensemble._normalize_encoding_config(raw_mapping)
            if not mapping:
                continue
            if any(isinstance(k, str) for k in mapping.keys()):
                local = local.with_columns(pl.col(col).cast(pl.Utf8))
            default_val = 0.0 if method == "frequency" else 0
            local = local.with_columns(
                pl.col(col).replace_strict(mapping, default=default_val).alias(col)
            )

        # Replay numeric string coercions captured at L0 training.
        for col in (fe_metadata.get("numeric_coercions") or {}).keys():
            if col in local.columns:
                local = local.with_columns(
                    ensemble._numeric_string_to_float_expr(pl.col(col)).alias(col)
                )

        X = ensemble._prepare_features_with_metadata(local, feature_columns, fe_metadata)
        return X, local

    def _predict_worker(
        self,
        *,
        worker_row: dict[str, Any],
        df: pl.DataFrame,
        industry: str,
        category: str,
    ) -> tuple[np.ndarray, np.ndarray, WorkerRuntimeResult]:
        model_path = worker_row.get("paths", {}).get("model_path")
        if not model_path:
            raise ValueError(f"Missing model path for worker={worker_row.get('worker_model_id')}")
        model_abs = (self.base_path / model_path).resolve()
        if not model_abs.exists():
            raise FileNotFoundError(f"Model not found: {model_abs}")

        fe_meta_path = worker_row.get("paths", {}).get("feature_metadata_path")
        fe_metadata: dict[str, Any] = {}
        if fe_meta_path:
            p = (self.base_path / fe_meta_path).resolve()
            if p.exists():
                fe_metadata = json.loads(p.read_text())

        ensemble = self._get_ensemble(industry)
        feature_columns = list((worker_row.get("columns") or {}).get("model_feature_columns") or [])
        if not feature_columns:
            # fallback to dataset columns if feature column metadata is missing
            feature_columns = list((worker_row.get("columns") or {}).get("dataset_columns") or [])

        X, _ = self._apply_runtime_transforms(
            df=df,
            feature_columns=feature_columns,
            fe_metadata=fe_metadata,
            synonyms=self._merge_synonyms(industry=industry, category=category, worker_row=worker_row),
            ensemble=ensemble,
        )

        expected = int(worker_row.get("n_features") or 0)
        if expected > 0:
            current = int(X.shape[1]) if hasattr(X, "shape") else 0
            if current < expected:
                pad = sparse.csr_matrix((X.shape[0], expected - current)) if sparse.issparse(X) else np.zeros((X.shape[0], expected - current))
                X = sparse.hstack([X, pad]).tocsr() if sparse.issparse(X) else np.hstack([X, pad])
            elif current > expected:
                X = X[:, :expected] if sparse.issparse(X) else X[:, :expected]

        model = joblib.load(model_abs)
        task_type = str(worker_row.get("task_type") or "regression")

        if task_type == "classification":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                pred = model.predict(X)
            pred = self._decode_pred_if_needed(np.asarray(pred), fe_metadata)
            if hasattr(model, "predict_proba"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    proba = model.predict_proba(X)
                conf = np.asarray(proba).max(axis=1).astype(np.float32)
            else:
                conf = np.ones(len(pred), dtype=np.float32)
            pred_num = self._to_numeric(np.asarray(pred))
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                pred = model.predict(X)
            pred_num = np.asarray(pred, dtype=np.float32).reshape(-1)
            conf = np.ones(len(pred_num), dtype=np.float32)

        result = WorkerRuntimeResult(
            worker_model_id=str(worker_row.get("worker_model_id") or ""),
            task_type=task_type,
            prediction_mean=float(np.mean(pred_num)) if len(pred_num) else 0.0,
            prediction_std=float(np.std(pred_num)) if len(pred_num) else 0.0,
            confidence_mean=float(np.mean(conf)) if len(conf) else 0.0,
            n_rows=int(len(pred_num)),
            errors=None,
        )
        return pred_num, conf, result

    @staticmethod
    def _torch_predict_regression(model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
        with torch.no_grad():
            xb = torch.FloatTensor(X).to(device)
            out = model(xb).cpu().numpy().reshape(-1)
        return out.astype(np.float32)

    @staticmethod
    def _torch_predict_classification(
        model: torch.nn.Module, X: np.ndarray, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            xb = torch.FloatTensor(X).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = probs.argmax(axis=1).astype(np.int32)
        return pred, probs

    def _aggregate_l1_regression(
        self,
        *,
        l1_model: torch.nn.Module,
        slot_count: int,
        worker_ids: list[str],
        worker_preds: dict[str, np.ndarray],
        worker_confs: dict[str, np.ndarray],
        worker_norm: dict[str, Any],
        device: torch.device,
    ) -> np.ndarray:
        n_rows = min(len(arr) for arr in worker_preds.values()) if worker_preds else 0
        if n_rows == 0:
            return np.array([], dtype=np.float32)

        weighted_sum = np.zeros(n_rows, dtype=np.float32)
        weight_total = np.zeros(n_rows, dtype=np.float32)

        for idx, worker_id in enumerate(worker_ids):
            if worker_id not in worker_preds:
                continue
            x = np.zeros((n_rows, slot_count * 2), dtype=np.float32)
            pred_arr = worker_preds[worker_id][:n_rows]
            conf_arr = worker_confs[worker_id][:n_rows]
            x[:, idx * 2] = pred_arr
            x[:, idx * 2 + 1] = conf_arr
            y_hat = self._torch_predict_regression(l1_model, x, device=device)

            norm = worker_norm.get(worker_id, {}) if isinstance(worker_norm, dict) else {}
            mean = float(norm.get("mean", 0.0))
            std = float(norm.get("std", 1.0))
            if abs(std) < 1e-8:
                std = 1.0
            y_denorm = (y_hat * std) + mean

            weighted_sum += y_denorm * conf_arr
            weight_total += conf_arr

        safe_weight = np.where(weight_total <= 1e-8, 1.0, weight_total)
        return weighted_sum / safe_weight

    def _aggregate_l1_classification(
        self,
        *,
        l1_model: torch.nn.Module,
        slot_count: int,
        worker_ids: list[str],
        worker_preds: dict[str, np.ndarray],
        worker_confs: dict[str, np.ndarray],
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_rows = min(len(arr) for arr in worker_preds.values()) if worker_preds else 0
        if n_rows == 0:
            return np.array([], dtype=np.int32), np.zeros((0, 0), dtype=np.float32)

        weighted_probs: Optional[np.ndarray] = None
        weight_total = np.zeros((n_rows, 1), dtype=np.float32)

        for idx, worker_id in enumerate(worker_ids):
            if worker_id not in worker_preds:
                continue
            x = np.zeros((n_rows, slot_count * 2), dtype=np.float32)
            pred_arr = worker_preds[worker_id][:n_rows]
            conf_arr = worker_confs[worker_id][:n_rows].reshape(-1, 1)
            x[:, idx * 2] = pred_arr
            x[:, idx * 2 + 1] = conf_arr.reshape(-1)
            _, probs = self._torch_predict_classification(l1_model, x, device=device)

            if weighted_probs is None:
                weighted_probs = np.zeros_like(probs, dtype=np.float32)
            weighted_probs += probs * conf_arr
            weight_total += conf_arr

        if weighted_probs is None:
            return np.array([], dtype=np.int32), np.zeros((0, 0), dtype=np.float32)
        safe_weight = np.where(weight_total <= 1e-8, 1.0, weight_total)
        probs_avg = weighted_probs / safe_weight
        pred = probs_avg.argmax(axis=1).astype(np.int32)
        return pred, probs_avg

    def run(
        self,
        *,
        industry: str,
        category: str,
        df: pl.DataFrame,
        preferred_workers: Optional[list[str]] = None,
        max_workers: int = 6,
    ) -> dict[str, Any]:
        manifest = self._load_manifest(industry)
        workers = self._get_category_workers(manifest, category)
        if not workers:
            return {
                "industry": industry,
                "category": category,
                "l0_worker_count": 0,
                "l0_workers": [],
                "l1": None,
                "error": "No workers found for category",
            }

        preferred = set(preferred_workers or [])
        if preferred:
            workers = [w for w in workers if w.get("worker_model_id") in preferred] or workers
        workers = workers[:max_workers]

        worker_results: list[WorkerRuntimeResult] = []
        worker_preds: dict[str, np.ndarray] = {}
        worker_confs: dict[str, np.ndarray] = {}

        for row in workers:
            wid = str(row.get("worker_model_id") or "")
            try:
                pred, conf, result = self._predict_worker(
                    worker_row=row,
                    df=df,
                    industry=industry,
                    category=category,
                )
                worker_results.append(result)
                worker_preds[wid] = pred
                worker_confs[wid] = conf
            except Exception as exc:
                worker_results.append(
                    WorkerRuntimeResult(
                        worker_model_id=wid,
                        task_type=str(row.get("task_type") or "unknown"),
                        prediction_mean=0.0,
                        prediction_std=0.0,
                        confidence_mean=0.0,
                        n_rows=int(df.height),
                        errors=str(exc),
                    )
                )

        l1_summary: Optional[dict[str, Any]] = None
        try:
            trainer = self._get_l1_trainer(industry)
            l1_model, metadata = trainer.load_expert(category)
            slot_worker_ids = list(metadata.get("worker_ids") or [])
            slot_count = int(metadata.get("n_workers") or len(slot_worker_ids))
            task_type = str(metadata.get("task_type") or "regression")
            if slot_count <= 0:
                slot_count = len(slot_worker_ids)

            available = [wid for wid in slot_worker_ids if wid in worker_preds]
            if task_type == "classification":
                pred, probs = self._aggregate_l1_classification(
                    l1_model=l1_model,
                    slot_count=slot_count,
                    worker_ids=available,
                    worker_preds=worker_preds,
                    worker_confs=worker_confs,
                    device=trainer.device,
                )
                l1_summary = {
                    "task_type": task_type,
                    "workers_used": len(available),
                    "n_rows": int(len(pred)),
                    "prediction_mode": "classification",
                    "class_distribution": {
                        str(int(cls)): int(cnt)
                        for cls, cnt in zip(*np.unique(pred, return_counts=True))
                    } if len(pred) else {},
                    "mean_confidence": float(np.max(probs, axis=1).mean()) if probs.size else 0.0,
                }
            else:
                output = self._aggregate_l1_regression(
                    l1_model=l1_model,
                    slot_count=slot_count,
                    worker_ids=available,
                    worker_preds=worker_preds,
                    worker_confs=worker_confs,
                    worker_norm=metadata.get("worker_target_normalization") or {},
                    device=trainer.device,
                )
                if len(output):
                    percentiles = np.percentile(output, [5, 50, 95]).tolist()
                    l1_summary = {
                        "task_type": task_type,
                        "workers_used": len(available),
                        "n_rows": int(len(output)),
                        "prediction_mode": "regression",
                        "prediction_mean": float(np.mean(output)),
                        "prediction_std": float(np.std(output)),
                        "prediction_p05": float(percentiles[0]),
                        "prediction_p50": float(percentiles[1]),
                        "prediction_p95": float(percentiles[2]),
                        "prediction_min": float(np.min(output)),
                        "prediction_max": float(np.max(output)),
                    }
                else:
                    l1_summary = {
                        "task_type": task_type,
                        "workers_used": len(available),
                        "n_rows": 0,
                        "prediction_mode": "regression",
                    }
        except Exception as exc:
            l1_summary = {"error": str(exc)}

        return {
            "industry": industry,
            "category": category,
            "l0_worker_count": len(workers),
            "l0_workers": [
                {
                    "worker_model_id": r.worker_model_id,
                    "task_type": r.task_type,
                    "prediction_mean": r.prediction_mean,
                    "prediction_std": r.prediction_std,
                    "confidence_mean": r.confidence_mean,
                    "n_rows": r.n_rows,
                    "errors": r.errors,
                }
                for r in worker_results
            ],
            "l1": l1_summary,
        }
