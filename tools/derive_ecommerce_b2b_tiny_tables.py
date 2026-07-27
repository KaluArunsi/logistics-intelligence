#!/usr/bin/env python3
"""Derive trainable B2B courier datasets from related ecommerce tables.

Transforms two tiny tables into row-level trainable datasets:
- `courier_company_rates_xlsx` -> expected charge regression dataset
- `expected_result_xlsx` -> overbilling classification dataset
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import polars as pl


def _read_parquet(root: Path, dataset_id: str) -> pl.DataFrame:
    candidates = list(root.rglob(f"{dataset_id}.parquet.zstd")) + list(root.rglob(f"{dataset_id}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Missing processed dataset file for {dataset_id}")
    return pl.read_parquet(candidates[0])


def _to_float_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.replace_all(",", "")
        .str.replace_all(r"[^0-9.\-]", "")
        .cast(pl.Float64, strict=False)
    )


def _derive_base_table(processed_root: Path) -> pl.DataFrame:
    invoice = _read_parquet(
        processed_root, "kaggle__b2b_ecommerce_courier_dataset__courier_company_invoice_xlsx"
    )
    order_report = _read_parquet(
        processed_root, "kaggle__b2b_ecommerce_courier_dataset__company_abc_order_report_xlsx"
    )
    sku_master = _read_parquet(
        processed_root, "kaggle__b2b_ecommerce_courier_dataset__company_abc_sku_master_xlsx"
    )
    pincode_zones = _read_parquet(
        processed_root, "kaggle__b2b_ecommerce_courier_dataset__company_abc_pincode_zones_xlsx"
    )
    rates = _read_parquet(
        processed_root, "kaggle__b2b_ecommerce_courier_dataset__courier_company_rates_xlsx"
    )
    if rates.height == 0:
        raise ValueError("Rates table is empty.")
    rate_row = rates.row(0, named=True)

    order_norm = order_report.with_columns(
        _to_float_expr("order_qty").fill_null(1.0).alias("order_qty_num")
    )
    sku_norm = sku_master.with_columns(_to_float_expr("weight_g").fill_null(0.0).alias("weight_g_num"))
    order_lines = (
        order_norm.join(sku_norm.select(["sku", "weight_g_num"]), on="sku", how="left")
        .with_columns(pl.col("weight_g_num").fill_null(0.0))
    )
    order_agg = order_lines.group_by("externorderno").agg(
        [
            pl.col("order_qty_num").sum().alias("total_order_qty"),
            pl.col("sku").n_unique().alias("distinct_sku_count"),
            (pl.col("order_qty_num") * pl.col("weight_g_num")).sum().alias("order_weight_g_sum"),
        ]
    )

    zone_map = pincode_zones.group_by(["warehouse_pincode", "customer_pincode"]).agg(
        pl.col("zone").first().alias("mapped_zone")
    )

    base = (
        invoice.with_columns(
            [
                _to_float_expr("charged_weight").fill_null(0.0).alias("charged_weight_kg"),
                _to_float_expr("billing_amount_rs").fill_null(0.0).alias("billing_amount_rs_num"),
                pl.col("zone").cast(pl.Utf8).str.to_lowercase().alias("zone_l"),
                pl.col("type_of_shipment").cast(pl.Utf8).alias("type_of_shipment"),
            ]
        )
        .join(order_agg, left_on="order_id", right_on="externorderno", how="left")
        .join(zone_map, on=["warehouse_pincode", "customer_pincode"], how="left")
        .with_columns(
            [
                pl.col("mapped_zone")
                .cast(pl.Utf8)
                .str.to_lowercase()
                .fill_null(pl.col("zone_l"))
                .alias("mapped_zone_l"),
                pl.when(pl.col("mapped_zone").is_null())
                .then(pl.lit(0))
                .otherwise((pl.col("zone_l") == pl.col("mapped_zone").cast(pl.Utf8).str.to_lowercase()).cast(pl.Int8))
                .alias("zone_match_flag"),
                pl.when(pl.col("type_of_shipment").cast(pl.Utf8).str.to_lowercase().str.contains("rto"))
                .then(pl.lit("forward_and_rto"))
                .otherwise(pl.lit("forward_only"))
                .alias("shipment_mode"),
                (pl.col("order_weight_g_sum") / 1000.0).fill_null(0.0).alias("order_weight_kg"),
                pl.col("total_order_qty").fill_null(0.0).alias("total_order_qty"),
                pl.col("distinct_sku_count").fill_null(0).cast(pl.Int64).alias("distinct_sku_count"),
            ]
        )
    )

    derived_rows: list[tuple[float, float, float, float, str]] = []
    for row in base.select(["charged_weight_kg", "mapped_zone_l"]).iter_rows(named=True):
        charged_weight = float(row["charged_weight_kg"] or 0.0)
        zone = str(row["mapped_zone_l"] or "d").lower()
        if zone not in {"a", "b", "c", "d", "e"}:
            zone = "d"

        # Industry billing convention: 0.5kg slabs.
        slab = max(0.5, math.ceil(charged_weight / 0.5) * 0.5)
        increments = max(0, int(round((slab - 0.5) / 0.5)))
        fixed = float(rate_row.get(f"fwd_{zone}_fixed", 0.0) or 0.0)
        additional = float(rate_row.get(f"fwd_{zone}_additional", 0.0) or 0.0)
        expected = fixed + (increments * additional)
        derived_rows.append((slab, fixed, additional, expected, f"fwd_{zone}"))

    base = base.with_columns(
        [
            pl.Series("weight_slab_kg", [r[0] for r in derived_rows]),
            pl.Series("rate_fixed_rs", [r[1] for r in derived_rows]),
            pl.Series("rate_additional_rs", [r[2] for r in derived_rows]),
            pl.Series("expected_charge_rs", [r[3] for r in derived_rows]),
            pl.Series("rate_key", [r[4] for r in derived_rows]),
        ]
    ).with_columns(
        [
            (pl.col("billing_amount_rs_num") - pl.col("expected_charge_rs")).alias("charge_delta_rs"),
            (pl.col("billing_amount_rs_num") - pl.col("expected_charge_rs")).abs().alias("charge_delta_abs_rs"),
            (pl.col("billing_amount_rs_num") > (pl.col("expected_charge_rs") + 1.0))
            .cast(pl.Int8)
            .alias("overbilled_flag"),
        ]
    )
    return base


def run(base_path: Path, industry: str) -> dict:
    if industry != "ecommerce":
        raise ValueError("This derivation script currently supports industry='ecommerce' only.")

    processed_root = base_path / "data" / "processed" / industry
    base_df = _derive_base_table(processed_root)

    rates_derived = base_df.select(
        [
            "order_id",
            "awb_code",
            "mapped_zone_l",
            "shipment_mode",
            "zone_match_flag",
            "charged_weight_kg",
            "weight_slab_kg",
            "total_order_qty",
            "distinct_sku_count",
            "order_weight_kg",
            "rate_fixed_rs",
            "rate_additional_rs",
            "expected_charge_rs",
        ]
    )
    expected_derived = base_df.select(
        [
            "order_id",
            "awb_code",
            "mapped_zone_l",
            "shipment_mode",
            "zone_match_flag",
            "charged_weight_kg",
            "weight_slab_kg",
            "total_order_qty",
            "distinct_sku_count",
            "order_weight_kg",
            "expected_charge_rs",
            "billing_amount_rs_num",
            "charge_delta_rs",
            "charge_delta_abs_rs",
            "overbilled_flag",
        ]
    )

    rates_out = (
        processed_root
        / "demand_signal"
        / "kaggle__b2b_ecommerce_courier_dataset__courier_company_rates_xlsx.parquet.zstd"
    )
    expected_out = (
        processed_root
        / "fulfillment_flow"
        / "kaggle__b2b_ecommerce_courier_dataset__expected_result_xlsx.parquet.zstd"
    )
    rates_out.parent.mkdir(parents=True, exist_ok=True)
    expected_out.parent.mkdir(parents=True, exist_ok=True)
    rates_derived.write_parquet(rates_out, compression="zstd")
    expected_derived.write_parquet(expected_out, compression="zstd")

    report = {
        "industry": industry,
        "derived_from": [
            "kaggle__b2b_ecommerce_courier_dataset__courier_company_invoice_xlsx",
            "kaggle__b2b_ecommerce_courier_dataset__company_abc_order_report_xlsx",
            "kaggle__b2b_ecommerce_courier_dataset__company_abc_sku_master_xlsx",
            "kaggle__b2b_ecommerce_courier_dataset__company_abc_pincode_zones_xlsx",
            "kaggle__b2b_ecommerce_courier_dataset__courier_company_rates_xlsx",
        ],
        "outputs": {
            "kaggle__b2b_ecommerce_courier_dataset__courier_company_rates_xlsx": {
                "rows": rates_derived.height,
                "cols": rates_derived.width,
                "path": str(rates_out),
            },
            "kaggle__b2b_ecommerce_courier_dataset__expected_result_xlsx": {
                "rows": expected_derived.height,
                "cols": expected_derived.width,
                "path": str(expected_out),
            },
        },
        "class_balance_expected_result": expected_derived["overbilled_flag"].value_counts().to_dict(as_series=False),
    }

    reports_dir = base_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "ecommerce_b2b_tiny_table_derivation.json"
    report_path.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive trainable B2B ecommerce tiny tables.")
    parser.add_argument("--industry", default="ecommerce", help="Industry name (default: ecommerce).")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    report = run(base_path=base, industry=args.industry)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

