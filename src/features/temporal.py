"""
Temporal Features - Extract time-based features from datetime columns.
"""

import polars as pl


class TemporalFeatures:
    """Extracts features from datetime columns."""

    def extract_all(self, df: pl.DataFrame, column: str) -> tuple[pl.DataFrame, list[str]]:
        """Extract all temporal features from a datetime column."""
        if column not in df.columns:
            return df, []

        # Ensure datetime type
        if df[column].dtype not in [pl.Date, pl.Datetime]:
            try:
                df = df.with_columns(pl.col(column).str.to_datetime().alias(column))
            except Exception:
                return df, []

        new_cols = []
        prefix = column

        # Basic components
        df = df.with_columns([
            pl.col(column).dt.year().alias(f"{prefix}_year"),
            pl.col(column).dt.month().alias(f"{prefix}_month"),
            pl.col(column).dt.day().alias(f"{prefix}_day"),
            pl.col(column).dt.weekday().alias(f"{prefix}_weekday"),
            pl.col(column).dt.hour().alias(f"{prefix}_hour"),
        ])
        new_cols.extend([f"{prefix}_year", f"{prefix}_month", f"{prefix}_day",
                        f"{prefix}_weekday", f"{prefix}_hour"])

        # Derived features
        df = df.with_columns([
            (pl.col(f"{prefix}_weekday") >= 6).cast(pl.Int8).alias(f"{prefix}_is_weekend"),
            pl.col(f"{prefix}_month").is_in([12, 1, 2]).cast(pl.Int8).alias(f"{prefix}_is_winter"),
            pl.col(f"{prefix}_month").is_in([6, 7, 8]).cast(pl.Int8).alias(f"{prefix}_is_summer"),
        ])
        new_cols.extend([f"{prefix}_is_weekend", f"{prefix}_is_winter", f"{prefix}_is_summer"])

        # Time of day categories
        df = df.with_columns(
            pl.when(pl.col(f"{prefix}_hour") < 6).then(pl.lit(0))
            .when(pl.col(f"{prefix}_hour") < 12).then(pl.lit(1))
            .when(pl.col(f"{prefix}_hour") < 18).then(pl.lit(2))
            .otherwise(pl.lit(3))
            .alias(f"{prefix}_time_of_day")
        )
        new_cols.append(f"{prefix}_time_of_day")

        return df, new_cols

    def add_cyclical(self, df: pl.DataFrame, column: str, max_val: int) -> pl.DataFrame:
        """Add cyclical encoding (sin/cos) for periodic features."""
        import math

        if column not in df.columns:
            return df

        df = df.with_columns([
            (pl.col(column) * 2 * math.pi / max_val).sin().alias(f"{column}_sin"),
            (pl.col(column) * 2 * math.pi / max_val).cos().alias(f"{column}_cos"),
        ])
        return df
