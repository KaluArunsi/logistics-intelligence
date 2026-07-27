"""
Numeric Scaling - Standard, min-max, and robust scaling.
"""

import polars as pl
from dataclasses import dataclass


@dataclass
class ScalerStats:
    """Statistics for reversing scaling."""
    mean: float
    std: float
    min: float
    max: float


class NumericScaler:
    """Handles numeric feature scaling."""

    def __init__(self):
        self._stats: dict[str, ScalerStats] = {}

    def standard_scale(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Z-score normalization: (x - mean) / std."""
        if column not in df.columns:
            return df

        mean = df[column].mean()
        std = df[column].std()

        if mean is None:
            mean = 0.0
        if std is None or std == 0:
            std = 1.0

        self._stats[column] = ScalerStats(mean=mean, std=std, min=0, max=0)

        new_col = f"{column}_scaled"
        df = df.with_columns(
            ((pl.col(column) - mean) / std).alias(new_col)
        )
        return df

    def minmax_scale(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Scale to [0, 1] range."""
        if column not in df.columns:
            return df

        min_val = df[column].min()
        max_val = df[column].max()

        if min_val is None:
            min_val = 0.0
        if max_val is None:
            max_val = 0.0
        range_val = max_val - min_val

        if range_val == 0:
            range_val = 1.0

        self._stats[column] = ScalerStats(mean=0, std=0, min=min_val, max=max_val)

        new_col = f"{column}_scaled"
        df = df.with_columns(
            ((pl.col(column) - min_val) / range_val).alias(new_col)
        )
        return df

    def robust_scale(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Scale using median and IQR (robust to outliers)."""
        if column not in df.columns:
            return df

        median = df[column].median()
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        if median is None:
            median = 0.0
        if q1 is None:
            q1 = 0.0
        if q3 is None:
            q3 = 0.0
        iqr = q3 - q1

        if iqr == 0:
            iqr = 1.0

        new_col = f"{column}_scaled"
        df = df.with_columns(
            ((pl.col(column) - median) / iqr).alias(new_col)
        )
        return df

    def get_stats(self, column: str) -> ScalerStats:
        """Get scaling statistics for inverse transform."""
        return self._stats.get(column)
