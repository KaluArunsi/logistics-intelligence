"""
Cross-industry common features.
"""

import polars as pl


class CommonFeatures:
    """Features applicable across all industries."""

    def add_lag_features(self, df: pl.DataFrame, column: str, lags: list[int] | None = None) -> pl.DataFrame:
        """Add lag features for time series."""
        if column not in df.columns:
            return df
        if lags is None:
            lags = [1, 7, 30]

        for lag in lags:
            df = df.with_columns(
                pl.col(column).shift(lag).alias(f"{column}_lag_{lag}")
            )
        return df

    def add_rolling_features(
        self,
        df: pl.DataFrame,
        column: str,
        windows: list[int] | None = None,
    ) -> pl.DataFrame:
        """Add rolling statistics."""
        if column not in df.columns:
            return df
        if windows is None:
            windows = [7, 30]

        for window in windows:
            df = df.with_columns([
                pl.col(column).rolling_mean(window_size=window).alias(f"{column}_rolling_mean_{window}"),
                pl.col(column).rolling_std(window_size=window).alias(f"{column}_rolling_std_{window}"),
            ])
        return df

    def add_ratio_features(self, df: pl.DataFrame, numerator: str, denominator: str) -> pl.DataFrame:
        """Add ratio between two columns."""
        if numerator not in df.columns or denominator not in df.columns:
            return df

        df = df.with_columns(
            (pl.col(numerator) / pl.col(denominator).replace(0, 1)).alias(f"{numerator}_per_{denominator}")
        )
        return df

    def add_interaction_features(self, df: pl.DataFrame, col1: str, col2: str) -> pl.DataFrame:
        """Add interaction between two numeric columns."""
        if col1 not in df.columns or col2 not in df.columns:
            return df

        df = df.with_columns(
            (pl.col(col1) * pl.col(col2)).alias(f"{col1}_x_{col2}")
        )
        return df
