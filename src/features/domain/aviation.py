"""
Aviation-specific feature engineering.
"""

import polars as pl


class AviationFeatures:
    """Aviation domain-specific features."""

    # Major hub airports for congestion scoring
    HUB_AIRPORTS = ["ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "LAS", "PHX", "MCO", "SEA"]

    def add_delay_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add features related to flight delays."""
        # Delay categories
        if "arrival_delay" in df.columns:
            df = df.with_columns([
                (pl.col("arrival_delay") > 0).cast(pl.Int8).alias("is_delayed"),
                (pl.col("arrival_delay") > 15).cast(pl.Int8).alias("is_late_15"),
                (pl.col("arrival_delay") > 60).cast(pl.Int8).alias("is_late_60"),
            ])

        return df

    def add_airport_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add airport-related features."""
        # Hub airport flags
        for col in ["origin", "dest", "departure_airport", "arrival_airport"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).is_in(self.HUB_AIRPORTS).cast(pl.Int8).alias(f"{col}_is_hub")
                )

        return df

    def add_route_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add route-related features."""
        # Route identifier
        origin_cols = ["origin", "departure_airport"]
        dest_cols = ["dest", "arrival_airport"]

        origin = next((c for c in origin_cols if c in df.columns), None)
        dest = next((c for c in dest_cols if c in df.columns), None)

        if origin and dest:
            df = df.with_columns(
                (pl.col(origin) + "_" + pl.col(dest)).alias("route")
            )

        return df

    def engineer_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all aviation-specific features."""
        df = self.add_delay_features(df)
        df = self.add_airport_features(df)
        df = self.add_route_features(df)
        return df
