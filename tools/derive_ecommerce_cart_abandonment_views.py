#!/usr/bin/env python3
"""Derive trainable cart-abandonment views for ecommerce workers.

Converts tiny lookup tables into row-level supervised datasets by joining each
dimension with session-level fact events and a shared business label:
`abandoned_flag` (1 if abandonment timestamp exists, else 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


DATASET_IDS = {
    "fact": "kaggle__cart_abandonment__fact_table_csv",
    "customer": "kaggle__cart_abandonment__customer_table_csv",
    "device": "kaggle__cart_abandonment__device_table_csv",
    "product": "kaggle__cart_abandonment__product_table_csv",
    "date": "kaggle__cart_abandonment__date_table_csv",
}


def _dataset_path(processed_root: Path, dataset_id: str) -> Path:
    matches = list(processed_root.glob(f"**/{dataset_id}.parquet.zstd"))
    if not matches:
        raise FileNotFoundError(f"Processed dataset not found: {dataset_id}")
    return matches[0]


def _load(processed_root: Path, dataset_id: str) -> tuple[pl.DataFrame, Path]:
    path = _dataset_path(processed_root, dataset_id)
    return pl.read_parquet(path), path


def _date_features(date_df: pl.DataFrame) -> pl.DataFrame:
    return date_df.with_columns(
        pl.col("date").dt.month().alias("date_month"),
        pl.col("date").dt.weekday().alias("date_day_of_week"),
        pl.col("date").dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),
    ).select(["date_id", "date", "date_month", "date_day_of_week", "is_weekend"])


def run(base_path: Path, industry: str) -> dict:
    if industry != "ecommerce":
        raise ValueError("This derivation currently supports industry='ecommerce' only.")

    processed_root = base_path / "data" / "processed" / industry
    reports_dir = base_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fact_df, fact_path = _load(processed_root, DATASET_IDS["fact"])
    customer_df, customer_path = _load(processed_root, DATASET_IDS["customer"])
    device_df, device_path = _load(processed_root, DATASET_IDS["device"])
    product_df, product_path = _load(processed_root, DATASET_IDS["product"])
    date_df, date_path = _load(processed_root, DATASET_IDS["date"])

    date_features_df = _date_features(date_df)

    fact_labeled = fact_df.with_columns(
        pl.col("abandonment_time").is_not_null().cast(pl.Int8).alias("abandoned_flag")
    )

    full = (
        fact_labeled
        .join(
            customer_df.select(
                ["customer_id", "age", "gender", "city"]
            ),
            on="customer_id",
            how="left",
        )
        .join(device_df.select(["device_id", "device_type", "os"]), on="device_id", how="left")
        .join(
            product_df.select(
                ["product_id", "product_name", "category", "price"]
            ),
            on="product_id",
            how="left",
        )
        .join(date_features_df, on="date_id", how="left")
    )

    fact_view = full.select(
        [
            "session_id",
            "customer_id",
            "product_id",
            "device_id",
            "date_id",
            "quantity",
            "age",
            "gender",
            "city",
            "device_type",
            "os",
            "category",
            "price",
            "date_month",
            "date_day_of_week",
            "is_weekend",
            "abandoned_flag",
        ]
    )

    customer_view = full.select(
        [
            "customer_id",
            "age",
            "gender",
            "city",
            "quantity",
            "device_type",
            "os",
            "category",
            "price",
            "date_month",
            "date_day_of_week",
            "is_weekend",
            "abandoned_flag",
        ]
    )

    device_view = full.select(
        [
            "device_id",
            "device_type",
            "os",
            "quantity",
            "age",
            "gender",
            "city",
            "category",
            "price",
            "date_month",
            "date_day_of_week",
            "is_weekend",
            "abandoned_flag",
        ]
    )

    product_view = full.select(
        [
            "product_id",
            "product_name",
            "category",
            "price",
            "quantity",
            "age",
            "gender",
            "city",
            "device_type",
            "os",
            "date_month",
            "date_day_of_week",
            "is_weekend",
            "abandoned_flag",
        ]
    )

    date_view = full.select(
        [
            "date_id",
            "date",
            "date_month",
            "date_day_of_week",
            "is_weekend",
            "quantity",
            "age",
            "gender",
            "city",
            "device_type",
            "os",
            "category",
            "price",
            "abandoned_flag",
        ]
    )

    # Write derived views to a separate directory to preserve original source data.
    derived_dir = processed_root / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    derived_paths = {}
    for key in DATASET_IDS:
        derived_paths[key] = derived_dir / f"{DATASET_IDS[key]}.parquet.zstd"

    fact_view.write_parquet(derived_paths["fact"], compression="zstd")
    customer_view.write_parquet(derived_paths["customer"], compression="zstd")
    device_view.write_parquet(derived_paths["device"], compression="zstd")
    product_view.write_parquet(derived_paths["product"], compression="zstd")
    date_view.write_parquet(derived_paths["date"], compression="zstd")

    report = {
        "industry": industry,
        "source": "cart_abandonment_derivation",
        "input_rows": {
            "fact": int(fact_df.height),
            "customer": int(customer_df.height),
            "device": int(device_df.height),
            "product": int(product_df.height),
            "date": int(date_df.height),
        },
        "output_rows": {
            DATASET_IDS["fact"]: int(fact_view.height),
            DATASET_IDS["customer"]: int(customer_view.height),
            DATASET_IDS["device"]: int(device_view.height),
            DATASET_IDS["product"]: int(product_view.height),
            DATASET_IDS["date"]: int(date_view.height),
        },
        "target": "abandoned_flag",
        "paths": {
            DATASET_IDS["fact"]: str(derived_paths["fact"]),
            DATASET_IDS["customer"]: str(derived_paths["customer"]),
            DATASET_IDS["device"]: str(derived_paths["device"]),
            DATASET_IDS["product"]: str(derived_paths["product"]),
            DATASET_IDS["date"]: str(derived_paths["date"]),
        },
    }

    report_path = reports_dir / "ecommerce_cart_abandonment_derivation.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive trainable cart-abandonment ecommerce views.")
    parser.add_argument("--industry", default="ecommerce")
    parser.add_argument("--base-path", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    report = run(base_path=base_path, industry=args.industry)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

