"""
Categorical Encoding - Label, one-hot, and target encoding.
"""

import polars as pl


class CategoricalEncoder:
    """Handles categorical variable encoding."""

    def __init__(self):
        self._label_maps: dict[str, dict] = {}

    def label_encode(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        """Convert categorical to numeric labels."""
        if column not in df.columns:
            return df

        # Get unique values and create mapping
        unique_vals = df[column].drop_nulls().unique().sort().to_list()
        label_map = {v: i for i, v in enumerate(unique_vals)}
        self._label_maps[column] = label_map

        # Apply mapping
        new_col = f"{column}_encoded"
        df = df.with_columns(
            pl.col(column).replace(label_map).cast(pl.Int32).alias(new_col)
        )
        return df

    def onehot_encode(self, df: pl.DataFrame, column: str, max_categories: int = 20) -> pl.DataFrame:
        """One-hot encode a categorical column."""
        if column not in df.columns:
            return df

        unique_vals = df[column].drop_nulls().unique().to_list()

        # Limit categories to avoid explosion
        if len(unique_vals) > max_categories:
            # Keep top N by frequency
            value_counts = df[column].value_counts().sort("count", descending=True)
            unique_vals = value_counts.head(max_categories)[column].to_list()

        # Create binary columns
        for val in unique_vals:
            col_name = f"{column}_{val}".replace(" ", "_").lower()
            df = df.with_columns(
                (pl.col(column) == val).cast(pl.Int8).alias(col_name)
            )

        return df

    def get_label_map(self, column: str) -> dict:
        """Get the label mapping for a column."""
        return self._label_maps.get(column, {})
