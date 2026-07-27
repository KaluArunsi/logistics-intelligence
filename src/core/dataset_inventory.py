"""
Dataset inventory and fingerprinting for worker uniqueness.
"""

import hashlib
from pathlib import Path
from typing import Iterable

from ..data.scanner import DatasetScanner
from .logging import get_logger


class DatasetInventory:
    """Scans datasets and assigns worker groups based on fingerprint."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.scanner = DatasetScanner()
        self.logger = get_logger("core.dataset_inventory")

    def discover_files(self, directories: Iterable[Path]) -> list[Path]:
        """Find candidate dataset files in directories."""
        files: list[Path] = []
        for directory in directories:
            directory = Path(directory)
            if not directory.exists():
                continue
            for pattern in ["*.csv", "*.parquet", "*.parquet.zstd"]:
                files.extend(directory.rglob(pattern))
        self.logger.info("Discovered %s dataset files", len(files))
        return files

    def _fingerprint(self, file_path: Path, n_rows: int, columns: list[str]) -> str:
        """Create a stable fingerprint for dataset grouping."""
        try:
            size = file_path.stat().st_size
        except FileNotFoundError:
            size = 0
        cols = "|".join(sorted(columns))
        raw = f"{file_path.suffix}|{n_rows}|{len(columns)}|{size}|{cols}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def build_inventory(self, directories: Iterable[Path]) -> list[dict]:
        """Scan files and return a list of dataset entries with fingerprints."""
        entries = []
        files = self.discover_files(directories)
        for file_path in files:
            try:
                profile = self.scanner.scan_file(file_path)
            except Exception as exc:
                self.logger.warning("Failed to scan %s: %s", file_path, exc)
                continue

            dataset_id = self._derive_dataset_id(file_path)
            fingerprint = self._fingerprint(file_path, profile.n_rows, profile.columns)

            entries.append({
                "dataset_id": dataset_id,
                "name": dataset_id,
                "file_path": str(file_path),
                "n_rows": profile.n_rows,
                "n_cols": profile.n_cols,
                "columns": profile.columns,
                "fingerprint": fingerprint,
                "profile": profile,
            })

        self.logger.info("Built inventory for %s datasets", len(entries))
        return entries

    def assign_worker_groups(self, entries: list[dict]) -> dict:
        """Assign a canonical worker dataset per fingerprint."""
        groups: dict[str, list[dict]] = {}
        for entry in entries:
            groups.setdefault(entry["fingerprint"], []).append(entry)

        worker_map: dict[str, str] = {}
        for _, group in groups.items():
            group_sorted = sorted(group, key=lambda e: e["dataset_id"])
            worker_dataset_id = group_sorted[0]["dataset_id"]
            for entry in group_sorted:
                worker_map[entry["dataset_id"]] = worker_dataset_id

        return worker_map

    def _derive_dataset_id(self, file_path: Path) -> str:
        """Derive a stable dataset_id from path."""
        stem = file_path.name
        for suffix in [".parquet.zstd", ".parquet", ".csv"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        return stem
