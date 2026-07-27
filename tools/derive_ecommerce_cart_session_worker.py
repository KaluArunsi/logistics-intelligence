#!/usr/bin/env python3
"""
Build a trainable cart-abandonment session worker dataset.

Creates one enriched worker table from the cart abandonment fact + dimensions,
with aggregate features intended for L0 conversion prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def _find_dataset(base: Path, dataset_id: str) -> Path:
    matches = list((base / "data" / "processed" / "ecommerce").rglob(f"{dataset_id}.parquet.zstd"))
    if not matches:
        raise FileNotFoundError(f"Missing dataset: {dataset_id}")
    return matches[0]


def _safe_div(numer: pl.Expr, denom: pl.Expr) -> pl.Expr:
    return pl.when(denom == 0).then(0.0).otherwise(numer / denom)


def run() -> None:
    base = Path(__file__).resolve().parents[1]
    out_dir = base / "data" / "processed" / "ecommerce" / "conversion_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    fact_id = "kaggle__cart_abandonment__fact_table_csv"
    customer_id = "kaggle__cart_abandonment__customer_table_csv"
    product_id = "kaggle__cart_abandonment__product_table_csv"
    device_id = "kaggle__cart_abandonment__device_table_csv"
    date_id = "kaggle__cart_abandonment__date_table_csv"

    fact = pl.read_parquet(_find_dataset(base, fact_id))
    customer = pl.read_parquet(_find_dataset(base, customer_id))
    product = pl.read_parquet(_find_dataset(base, product_id))
    device = pl.read_parquet(_find_dataset(base, device_id))
    date_tbl = pl.read_parquet(_find_dataset(base, date_id))

    # Aggregate features from fact (target-independent).
    customer_agg = fact.group_by("customer_id").agg(
        pl.len().alias("customer_session_count"),
        pl.col("product_id").n_unique().alias("customer_unique_product_count"),
        pl.col("price").mean().alias("customer_avg_price"),
        pl.col("quantity").mean().alias("customer_avg_quantity"),
    )
    product_agg = fact.group_by("product_id").agg(
        pl.len().alias("product_session_count"),
        pl.col("customer_id").n_unique().alias("product_unique_customer_count"),
        pl.col("price").mean().alias("product_avg_price"),
    )
    device_agg = fact.group_by("device_id").agg(
        pl.len().alias("device_session_count"),
        pl.col("customer_id").n_unique().alias("device_unique_customer_count"),
    )
    city_agg = fact.group_by("city").agg(pl.len().alias("city_session_count"))

    # Bring in additional context columns from dimensions (deduplicated by key).
    customer_dim = customer.select(
        [
            "customer_id",
            "age",
            "gender",
            "city",
        ]
    ).unique(subset=["customer_id"], keep="first")
    product_dim = product.select(
        [
            "product_id",
            "category",
            "price",
            "quantity",
        ]
    ).unique(subset=["product_id"], keep="first")
    device_dim = device.select(
        [
            "device_id",
            "device_type",
            "os",
        ]
    ).unique(subset=["device_id"], keep="first")
    date_dim = date_tbl.select(
        [
            "date_id",
            "date_month",
            "date_day_of_week",
            "is_weekend",
        ]
    ).unique(subset=["date_id"], keep="first")

    enriched = (
        fact.join(customer_dim, on="customer_id", how="left", suffix="_cust")
        .join(product_dim, on="product_id", how="left", suffix="_prod")
        .join(device_dim, on="device_id", how="left", suffix="_dev")
        .join(date_dim, on="date_id", how="left", suffix="_date")
        .join(customer_agg, on="customer_id", how="left")
        .join(product_agg, on="product_id", how="left")
        .join(device_agg, on="device_id", how="left")
        .join(city_agg, on="city", how="left")
    )

    median_price = float(enriched.select(pl.col("price").median()).item() or 0.0)
    median_qty = float(enriched.select(pl.col("quantity").median()).item() or 0.0)

    enriched = enriched.with_columns(
        [
            _safe_div(pl.col("price"), pl.col("quantity")).alias("unit_price"),
            (pl.col("price") >= median_price).cast(pl.Int8).alias("is_high_price"),
            (pl.col("quantity") >= median_qty).cast(pl.Int8).alias("is_high_quantity"),
            (pl.col("date_month").fill_null(0).cast(pl.Float64) / 12.0).alias("month_norm"),
            (pl.col("date_day_of_week").fill_null(0).cast(pl.Float64) / 7.0).alias("weekday_norm"),
        ]
    )

    # Keep one clean copy of common duplicated columns.
    drop_cols = [c for c in enriched.columns if c.endswith("_cust") or c.endswith("_prod") or c.endswith("_dev") or c.endswith("_date")]
    if drop_cols:
        enriched = enriched.drop(drop_cols)

    dataset_id = "kaggle__cart_abandonment__session_joined_csv"
    out_path = out_dir / f"{dataset_id}.parquet.zstd"
    enriched.write_parquet(out_path, compression="zstd")

    report = {
        "dataset_id": dataset_id,
        "source_fact": fact_id,
        "source_dimensions": [customer_id, product_id, device_id, date_id],
        "rows": enriched.height,
        "columns": len(enriched.columns),
        "output_path": str(out_path.relative_to(base)),
    }
    reports_dir = base / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "ecommerce_cart_session_worker_derivation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    run()
