"""
Target selection heuristics for datasets.
"""

from dataclasses import dataclass
import re
from typing import Optional

import polars as pl

from .logging import get_logger


@dataclass
class TargetSelection:
    target_column: Optional[str]
    task_type: Optional[str]
    feature_columns: list[str]
    reason: str


class TargetSelector:
    """Heuristics to pick a target column and task type."""

    def __init__(self):
        self.logger = get_logger("core.target_selector")
        self.max_class_count = 50

    @staticmethod
    def _is_id_like_column(name: str) -> bool:
        col = str(name).strip().lower()
        return bool(
            col in {"id", "row_id", "record_id", "uuid", "uid", "index"}
            or col.endswith("_id")
            or re.match(r".*id\d*$", col)
            or "fips" in col
            or col.endswith("code")
            or col.startswith("code_")
        )

    def select(self, df: pl.DataFrame, candidate_columns: list[str]) -> TargetSelection:
        """Select a target column from candidates with heuristics."""
        if not candidate_columns:
            return TargetSelection(None, None, [], "no_candidates")

        scored = []
        for col in candidate_columns:
            if self._is_id_like_column(col):
                continue
            try:
                null_pct = df[col].null_count() / max(df.height, 1)
                unique_count = df[col].n_unique()
                dtype = df[col].dtype
            except Exception:
                continue

            if unique_count < 2:
                continue

            score = 0.0
            if "target" in col or "label" in col:
                score += 2.0
            if "price" in col or "fare" in col or "revenue" in col:
                score += 1.5
            if "delay" in col or "rul" in col or "failure" in col:
                score += 1.5
            if null_pct < 0.2:
                score += 0.5
            if unique_count > 1:
                score += 0.5

            # Penalize high-cardinality categorical targets
            if dtype not in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                if unique_count > self.max_class_count:
                    score -= 5.0

            scored.append((score, col, dtype, unique_count))

        if not scored:
            return TargetSelection(None, None, [], "no_valid_candidates")

        scored.sort(reverse=True, key=lambda x: x[0])
        _, target_col, dtype, unique_count = scored[0]

        # Task type
        if dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            if unique_count <= 20:
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            if unique_count > self.max_class_count:
                return TargetSelection(None, None, [], "high_cardinality_target")
            task_type = "classification"

        feature_columns = [c for c in df.columns if c != target_col]
        reason = f"selected:{target_col}"
        self.logger.info("Selected target %s | task=%s", target_col, task_type)

        return TargetSelection(target_col, task_type, feature_columns, reason)
