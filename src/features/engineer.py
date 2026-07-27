"""
Feature Engineering Orchestrator - Coordinates all feature transformations.
"""

import polars as pl
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class FeatureSpec:
    """Specification for features to generate."""
    target_column: str
    task_type: str  # classification, regression
    categorical_columns: list[str]
    numeric_columns: list[str]
    temporal_columns: list[str]
    encoding_strategy: str = "label"  # label, onehot, target
    scaling_strategy: str = "standard"  # standard, minmax, robust


class FeatureEngineer:
    """Orchestrates feature engineering for datasets."""

    def __init__(self):
        from .encoders import CategoricalEncoder
        from .scalers import NumericScaler
        from .temporal import TemporalFeatures

        self.encoder = CategoricalEncoder()
        self.scaler = NumericScaler()
        self.temporal = TemporalFeatures()

    def auto_detect_columns(self, df: pl.DataFrame) -> dict:
        """Auto-detect column types for feature engineering."""
        categorical = []
        numeric = []
        temporal = []

        temporal_patterns = ["date", "time", "year", "month", "day", "hour", "timestamp"]

        for col in df.columns:
            dtype = df[col].dtype
            col_lower = col.lower()

            # Check temporal patterns first
            if any(p in col_lower for p in temporal_patterns):
                temporal.append(col)
            elif dtype in [pl.Utf8, pl.String, pl.Categorical]:
                categorical.append(col)
            elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                          pl.Float32, pl.Float64]:
                # Low cardinality numeric = categorical
                unique = df[col].n_unique()
                if unique <= 20 and unique < len(df) * 0.05:
                    categorical.append(col)
                else:
                    numeric.append(col)
            elif dtype in [pl.Date, pl.Datetime, pl.Time]:
                temporal.append(col)

        return {
            "categorical": categorical,
            "numeric": numeric,
            "temporal": temporal
        }

    def engineer(
        self,
        df: pl.DataFrame,
        spec: Optional[FeatureSpec] = None,
        target_column: Optional[str] = None
    ) -> tuple[pl.DataFrame, list[str]]:
        """
        Apply feature engineering to a DataFrame.

        Returns (transformed_df, feature_columns).
        """
        if spec is None:
            cols = self.auto_detect_columns(df)
            spec = FeatureSpec(
                target_column=target_column or "",
                task_type="classification",
                categorical_columns=cols["categorical"],
                numeric_columns=cols["numeric"],
                temporal_columns=cols["temporal"]
            )

        feature_cols = []

        # Encode categoricals
        if spec.categorical_columns:
            for col in spec.categorical_columns:
                if col in df.columns and col != spec.target_column:
                    df = self.encoder.label_encode(df, col)
                    feature_cols.append(f"{col}_encoded")

        # Scale numerics
        if spec.numeric_columns:
            for col in spec.numeric_columns:
                if col in df.columns and col != spec.target_column:
                    df = self.scaler.standard_scale(df, col)
                    feature_cols.append(f"{col}_scaled")

        # Extract temporal features
        if spec.temporal_columns:
            for col in spec.temporal_columns:
                if col in df.columns:
                    df, new_cols = self.temporal.extract_all(df, col)
                    feature_cols.extend(new_cols)

        return df, feature_cols

    def prepare_for_training(
        self,
        df: pl.DataFrame,
        target_column: str,
        feature_columns: Optional[list[str]] = None
    ) -> tuple[pl.DataFrame, pl.Series]:
        """
        Prepare DataFrame for model training.

        Returns (X, y) as polars objects.
        """
        if feature_columns is None:
            # Use all numeric columns except target
            feature_columns = [c for c in df.columns
                             if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                             and c != target_column]

        X = df.select(feature_columns)
        y = df[target_column]

        return X, y
