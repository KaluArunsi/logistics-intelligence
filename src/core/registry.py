"""
Dataset and Model Registry

Tracks all datasets and trained models throughout the pipeline.
Provides a single source of truth for pipeline state.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
import os
import tempfile
import threading
from .logging import get_logger


@dataclass
class DatasetMeta:
    """Metadata for a single dataset."""
    id: str
    name: str
    source: str  # kaggle, bts, nasa, etc.
    category: str  # primary category
    n_rows: int
    n_cols: int
    columns: list[str]
    categories: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # classification, regression
    file_path: Optional[str] = None
    processed: bool = False
    ingested_at: Optional[str] = None
    fingerprint: Optional[str] = None
    duplicate_of: Optional[str] = None
    worker_dataset_id: Optional[str] = None


@dataclass
class ModelMeta:
    """Metadata for a trained model."""
    id: str
    dataset_id: str
    level: int  # 0 for workers, 1 for experts
    category: str
    algorithm: str
    categories: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    passed_benchmark: bool = False
    file_path: Optional[str] = None
    trained_at: Optional[str] = None
    # Feature metadata for L1 training
    feature_columns: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    n_features: int = 0
    task_type: Optional[str] = None
    data_file: Optional[str] = None  # Path to source parquet file
    run_id: Optional[str] = None


class Registry:
    """Central registry for datasets and models."""

    def __init__(self, base_path: Path, industry: str = "ecommerce"):
        self.base_path = Path(base_path)
        self.industry = industry
        self.logger = get_logger("core.registry")
        self._lock = threading.RLock()
        self.datasets_file = (
            self.base_path / "data" / "metadata" / self.industry / f"{self.industry}_datasets.json"
        )
        self.models_file = self.base_path / "models" / self.industry / "specs" / "models.json"
        self._datasets: dict[str, DatasetMeta] = {}
        self._models: dict[str, ModelMeta] = {}
        self._load()

    def _make_relative(self, path_str: Optional[str]) -> Optional[str]:
        """Store paths relative to base_path when possible."""
        if not path_str:
            return path_str
        try:
            path = Path(path_str)
            return str(path.relative_to(self.base_path))
        except ValueError:
            return path_str

    def resolve_path(self, path_str: Optional[str]) -> Optional[str]:
        """Resolve stored paths to absolute paths when needed."""
        if not path_str:
            return path_str
        path = Path(path_str)
        if path.is_absolute():
            return str(path)
        return str(self.base_path / path)

    def _load(self):
        """Load existing registry from disk."""
        if self.datasets_file.exists():
            with open(self.datasets_file) as f:
                data = json.load(f)
                self._datasets = {}
                allowed = set(DatasetMeta.__dataclass_fields__.keys())
                for k, v in data.items():
                    if not isinstance(v, dict):
                        continue
                    cleaned = {field: v[field] for field in allowed if field in v}
                    cleaned.setdefault("id", k)
                    try:
                        self._datasets[k] = DatasetMeta(**cleaned)
                    except TypeError:
                        # Skip malformed entries instead of crashing whole registry load.
                        continue

        if self.models_file.exists():
            with open(self.models_file) as f:
                data = json.load(f)
                self._models = {}
                allowed = set(ModelMeta.__dataclass_fields__.keys())
                for k, v in data.items():
                    if not isinstance(v, dict):
                        continue
                    cleaned = {field: v[field] for field in allowed if field in v}
                    cleaned.setdefault("id", k)
                    try:
                        self._models[k] = ModelMeta(**cleaned)
                    except TypeError:
                        continue

    def _reload_from_disk(self) -> None:
        """
        Reload registry state from disk before mutating writes.

        Multiple long-lived Registry instances exist across pipeline components.
        Reloading here prevents one instance from overwriting newer entries
        written by another instance.
        """
        self._datasets = {}
        self._models = {}
        self._load()

    def _to_jsonable(self, value):
        """Recursively convert values to JSON-safe Python primitives."""
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        # Handle numpy scalars without importing numpy as a hard dependency.
        if hasattr(value, "item") and callable(getattr(value, "item")):
            try:
                return self._to_jsonable(value.item())
            except Exception:
                pass
        if isinstance(value, Path):
            return str(value)
        return value

    def _write_json_atomic(self, path: Path, payload: dict) -> None:
        """Write JSON atomically to avoid partial/corrupted files on failure."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use a unique temp file per write to avoid collisions when
        # multiple pipeline components persist registry state concurrently.
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _save(self):
        """Persist registry to disk."""
        datasets_payload = {k: self._to_jsonable(asdict(v)) for k, v in self._datasets.items()}
        models_payload = {k: self._to_jsonable(asdict(v)) for k, v in self._models.items()}
        self._write_json_atomic(self.datasets_file, datasets_payload)
        self._write_json_atomic(self.models_file, models_payload)

    def register_dataset(self, meta: DatasetMeta) -> None:
        """Register a new dataset."""
        with self._lock:
            self._reload_from_disk()
            meta.ingested_at = datetime.now().isoformat()
            meta.file_path = self._make_relative(meta.file_path)
            self._datasets[meta.id] = meta
            self._save()
        self.logger.info("Registered dataset %s", meta.id)

    def get_dataset(self, dataset_id: str) -> Optional[DatasetMeta]:
        """Get dataset by ID."""
        with self._lock:
            return self._datasets.get(dataset_id)

    def list_datasets(self, category: Optional[str] = None, processed_only: bool = False) -> list[DatasetMeta]:
        """List datasets, optionally filtered."""
        with self._lock:
            datasets = list(self._datasets.values())
        if category:
            datasets = [d for d in datasets if d.category == category]
        if processed_only:
            datasets = [d for d in datasets if d.processed]
        return datasets

    def mark_processed(self, dataset_id: str, file_path: str) -> None:
        """Mark a dataset as processed and store its file path."""
        with self._lock:
            self._reload_from_disk()
            if dataset_id in self._datasets:
                self._datasets[dataset_id].processed = True
                self._datasets[dataset_id].file_path = self._make_relative(file_path)
                self._save()
                self.logger.info("Marked dataset processed %s", dataset_id)

    def register_model(self, meta: ModelMeta) -> None:
        """Register a trained model."""
        with self._lock:
            self._reload_from_disk()
            meta.trained_at = datetime.now().isoformat()
            meta.file_path = self._make_relative(meta.file_path)
            meta.data_file = self._make_relative(meta.data_file)
            meta.metrics = self._to_jsonable(meta.metrics)
            self._models[meta.id] = meta
            self._save()
        self.logger.info("Registered model %s", meta.id)

    def get_model(self, model_id: str) -> Optional[ModelMeta]:
        """Get model by ID."""
        with self._lock:
            return self._models.get(model_id)

    def list_models(self, level: Optional[int] = None, category: Optional[str] = None) -> list[ModelMeta]:
        """List models, optionally filtered."""
        with self._lock:
            models = list(self._models.values())
        if level is not None:
            models = [m for m in models if m.level == level]
        if category:
            models = [m for m in models if m.category == category]
        return models

    def all_l0_passed(self) -> bool:
        """Check if all L0 workers passed benchmarks."""
        l0_models = self.list_models(level=0)
        if not l0_models:
            return False
        return all(m.passed_benchmark for m in l0_models)

    def get_category_workers(self, category: str) -> list[ModelMeta]:
        """Get all L0 workers for a category."""
        return [m for m in self.list_models(level=0) if m.category == category]

    def summary(self) -> dict:
        """Get a summary of registry state."""
        l0_models = self.list_models(level=0)
        l1_models = self.list_models(level=1)
        return {
            "total_datasets": len(self._datasets),
            "processed_datasets": len([d for d in self._datasets.values() if d.processed]),
            "l0_models": len(l0_models),
            "l0_passed": len([m for m in l0_models if m.passed_benchmark]),
            "l1_models": len(l1_models),
            "l1_passed": len([m for m in l1_models if m.passed_benchmark]),
            "categories": list(set(d.category for d in self._datasets.values())),
        }
