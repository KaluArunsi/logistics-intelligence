#!/usr/bin/env python3
"""
Derive additional trucking_delivery worker datasets to close category and regional gaps.

Creates:
- 4 cold-chain derived workers (US)
- 2 Asia derived workers
- 3 Europe derived workers
- 2 Amazon last-mile derived workers (US, route-level flattened)

Also updates dataset specs and metadata entries for the derived datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class DerivedDataset:
    dataset_id: str
    category: str
    target: str
    task_type: str
    source: str
    file_path: str
    rows: int
    cols: int
    columns: list[str]


class Deriver:
    def __init__(self, base: Path):
        self.base = base
        self.industry = "trucking_delivery"
        self.processed_root = base / "data" / "processed" / self.industry
        self.spec_path = (
            base
            / "config"
            / "industries"
            / self.industry
            / f"{self.industry}_dataset_specs.json"
        )
        self.meta_path = (
            base
            / "data"
            / "metadata"
            / self.industry
            / f"{self.industry}_datasets.json"
        )
        self.report_path = base / "reports" / "trucking_delivery_gap_derivation_report.json"

        self.spec_obj = self._read_json(self.spec_path, default={"industry": self.industry, "datasets": {}})
        if "datasets" not in self.spec_obj or not isinstance(self.spec_obj["datasets"], dict):
            self.spec_obj["datasets"] = {}
        self.specs: dict[str, dict[str, Any]] = self.spec_obj["datasets"]

        meta_obj = self._read_json(self.meta_path, default={})
        if isinstance(meta_obj, dict) and "datasets" in meta_obj and isinstance(meta_obj["datasets"], dict):
            self.meta_obj = meta_obj
            self.meta = self.meta_obj["datasets"]
        elif isinstance(meta_obj, dict):
            self.meta_obj = {"datasets": meta_obj}
            self.meta = self.meta_obj["datasets"]
        else:
            self.meta_obj = {"datasets": {}}
            self.meta = self.meta_obj["datasets"]

        self.derived: list[DerivedDataset] = []

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text())

    @staticmethod
    def _to_float(expr: pl.Expr) -> pl.Expr:
        cleaned = (
            expr.cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r"[\u00A0\s]", "")
            .str.replace_all(r"[^0-9,.\-]", "")
        )
        return (
            pl.when(cleaned.str.contains(r"^-?\d{1,3}(\.\d{3})+,\d+$"))
            .then(cleaned.str.replace_all(r"\.", "").str.replace_all(",", "."))
            .when(cleaned.str.contains(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$"))
            .then(cleaned.str.replace_all(",", ""))
            .when(cleaned.str.contains(r"^-?\d+,\d+$"))
            .then(cleaned.str.replace_all(",", "."))
            .otherwise(cleaned)
            .cast(pl.Float64, strict=False)
        )

    def _read_dataset(self, dataset_id: str) -> pl.DataFrame:
        matches = list(self.processed_root.rglob(f"{dataset_id}.parquet.zstd"))
        if not matches:
            raise FileNotFoundError(f"Missing dataset: {dataset_id}")
        return pl.read_parquet(matches[0])

    def _write_derived(
        self,
        df: pl.DataFrame,
        dataset_id: str,
        category: str,
        target: str,
        task_type: str,
        source: str,
    ) -> None:
        out_dir = self.processed_root / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{dataset_id}.parquet.zstd"
        df.write_parquet(out_file, compression="zstd")

        self.specs[dataset_id] = {
            "category": category,
            "categories": [category],
            "targets": [
                {
                    "name": target,
                    "task_type": task_type,
                    "primary": True,
                }
            ],
            "features": {
                "include": [],
                "exclude": [],
                "fe": {},
                "text_features": 128,
            },
            "sampling": {
                "max_rows": 250000,
                "batch_size": 50000,
            },
            "training": {
                "gating": True,
                "worker_enabled": True,
            },
        }

        ingested_at = datetime.now(timezone.utc).isoformat()
        self.meta[dataset_id] = {
            "id": dataset_id,
            "name": dataset_id,
            "source": source,
            "category": category,
            "categories": [category],
            "n_rows": int(df.height),
            "n_cols": int(df.width),
            "columns": list(df.columns),
            "target_column": target,
            "task_type": task_type,
            "file_path": str(out_file.relative_to(self.base)),
            "processed": True,
            "ingested_at": ingested_at,
            "fingerprint": None,
            "duplicate_of": None,
            "worker_dataset_id": None,
        }

        self.derived.append(
            DerivedDataset(
                dataset_id=dataset_id,
                category=category,
                target=target,
                task_type=task_type,
                source=source,
                file_path=str(out_file.relative_to(self.base)),
                rows=int(df.height),
                cols=int(df.width),
                columns=list(df.columns),
            )
        )

    def derive_cold_chain(self) -> None:
        src_id = "us__bts__bts_f3sb_gw7h_cfs_temperature_control_file_2012_2022_csv"
        df = self._read_dataset(src_id).with_columns(
            [
                self._to_float(pl.col("ton")).alias("ton_num"),
                self._to_float(pl.col("val")).alias("val_num"),
                self._to_float(pl.col("avgmile")).alias("avgmile_num"),
                self._to_float(pl.col("shipdist")).alias("shipdist_num"),
                self._to_float(pl.col("shipwt")).alias("shipwt_num"),
                self._to_float(pl.col("tonpchg")).alias("tonpchg_num"),
                self._to_float(pl.col("valpchg")).alias("valpchg_num"),
            ]
        )

        by_mode = (
            df.group_by(["year", "dmode", "tmode", "naics"])
            .agg(
                [
                    pl.col("ton_num").sum().alias("target_ton_sum"),
                    pl.col("val_num").sum().alias("value_sum"),
                    pl.col("avgmile_num").mean().alias("avg_miles"),
                    pl.col("shipdist_num").mean().alias("avg_ship_distance"),
                    pl.col("shipwt_num").mean().alias("avg_ship_weight"),
                ]
            )
            .drop_nulls(subset=["target_ton_sum"])
            .filter(pl.col("target_ton_sum").is_not_nan())
        )
        self._write_derived(
            by_mode,
            dataset_id="us__bts__cfs_tempctrl_mode_year_ton_derived_csv",
            category="cold_chain_last_mile",
            target="target_ton_sum",
            task_type="regression",
            source="us",
        )

        by_value = (
            df.group_by(["year", "naics", "comm", "shipwt"])
            .agg(
                [
                    pl.col("val_num").sum().alias("target_value_sum"),
                    pl.col("ton_num").sum().alias("ton_sum"),
                    pl.col("avgmile_num").mean().alias("avg_miles"),
                    pl.col("shipdist_num").mean().alias("avg_ship_distance"),
                ]
            )
            .drop_nulls(subset=["target_value_sum"])
            .filter(pl.col("target_value_sum").is_not_nan())
        )
        self._write_derived(
            by_value,
            dataset_id="us__bts__cfs_tempctrl_commodity_value_derived_csv",
            category="cold_chain_last_mile",
            target="target_value_sum",
            task_type="regression",
            source="us",
        )

        by_distance = (
            df.group_by(["year", "shipdist", "shipwt", "dmode"])
            .agg(
                [
                    pl.col("avgmile_num").mean().alias("target_avg_miles"),
                    pl.col("ton_num").sum().alias("ton_sum"),
                    pl.col("val_num").sum().alias("value_sum"),
                ]
            )
            .drop_nulls(subset=["target_avg_miles"])
            .filter(pl.col("target_avg_miles").is_not_nan())
        )
        self._write_derived(
            by_distance,
            dataset_id="us__bts__cfs_tempctrl_distance_profile_derived_csv",
            category="cold_chain_last_mile",
            target="target_avg_miles",
            task_type="regression",
            source="us",
        )

        by_change = (
            df.group_by(["year", "dmode", "shipwt", "naics"])
            .agg(
                [
                    pl.col("tonpchg_num").mean().alias("target_ton_change"),
                    pl.col("valpchg_num").mean().alias("value_change"),
                    pl.col("ton_num").sum().alias("ton_sum"),
                ]
            )
            .drop_nulls(subset=["target_ton_change"])
            .filter(pl.col("target_ton_change").is_not_nan())
        )
        self._write_derived(
            by_change,
            dataset_id="us__bts__cfs_tempctrl_mode_change_derived_csv",
            category="cold_chain_last_mile",
            target="target_ton_change",
            task_type="regression",
            source="us",
        )

    def derive_amazon_last_mile(self) -> None:
        split_to_ids = {
            "train": (
                "us__amazon_lastmile__almrrc2021_data_training__model_build_inputs__actual_sequences_json",
                "us__amazon_lastmile__almrrc2021_data_training__model_build_inputs__invalid_sequence_scores_json",
            ),
            "eval": (
                "us__amazon_lastmile__almrrc2021_data_evaluation__model_score_inputs__eval_actual_sequences_json",
                "us__amazon_lastmile__almrrc2021_data_evaluation__model_score_inputs__eval_invalid_sequence_scores_json",
            ),
        }
        for split, (actual_id, score_id) in split_to_ids.items():
            actual_df = self._read_dataset(actual_id)
            score_df = self._read_dataset(score_id)
            common_cols = [c for c in score_df.columns if c in actual_df.columns]
            rows: list[dict[str, Any]] = []
            for col in common_cols:
                score = score_df[col][0]
                payload = actual_df[col][0]
                if score is None or payload is None or not isinstance(payload, dict):
                    continue
                stops = payload.get("actual")
                if not isinstance(stops, dict) or len(stops) < 2:
                    continue
                positions = [float(v) for v in stops.values() if isinstance(v, (int, float))]
                if len(positions) < 2:
                    continue
                n_stops = len(positions)
                pos_min = min(positions)
                pos_max = max(positions)
                rows.append(
                    {
                        "route_id": col.replace("routeid_", ""),
                        "n_stops": float(n_stops),
                        "pos_mean": float(sum(positions) / n_stops),
                        "pos_span": float(pos_max - pos_min),
                        "pos_max": float(pos_max),
                        "pos_min": float(pos_min),
                        "invalid_sequence_score": float(score),
                    }
                )
            if not rows:
                continue
            flat = pl.DataFrame(rows)
            q75 = float(flat.select(pl.col("invalid_sequence_score").quantile(0.75)).item())
            reg = flat
            cls = flat.with_columns(
                (pl.col("invalid_sequence_score") >= q75).cast(pl.Int8).alias("route_exception_flag")
            )

            self._write_derived(
                reg,
                dataset_id=f"us__amazon_lastmile__route_invalid_score_{split}_derived_csv",
                category="last_mile_sla",
                target="invalid_sequence_score",
                task_type="regression",
                source="us",
            )
            self._write_derived(
                cls,
                dataset_id=f"us__amazon_lastmile__route_exception_flag_{split}_derived_csv",
                category="parcel_exception_risk",
                target="route_exception_flag",
                task_type="classification",
                source="us",
            )

    def derive_asia_europe_quota_fill(self) -> None:
        asia_oecd_id = "asia__oecd__oecd_st_df_stfreight_asia_csv"
        asia_hk_id = (
            "asia__hong_kong__hk_10bbaec2_5faed511_key_statistics_on_business_performance_and_"
            "operating_characteristics_of_the_transportat_856432a012"
        )
        eu_oecd_id = "europe__oecd__oecd_st_df_sttraffic_europe_csv"
        eu_road_id = "europe__eurostat__eurostat_road_go_ia_rc_csv"
        eu_uk_id = (
            "europe__uk__uk_1699fb94_59853277_northern_ireland_goods_vehicle_operator_s_licence_records_"
            "northern_ireland_goods_vehicle_ope_6ccbd8f6a4"
        )

        asia_oecd = self._read_dataset(asia_oecd_id)
        asia_oecd = asia_oecd.with_columns(self._to_float(pl.col("obs_value")).alias("obs_value_num"))
        asia_derived = (
            asia_oecd.group_by(["ref_area", "transport_mode", "fuel", "time_period"])
            .agg(
                [
                    pl.col("obs_value_num").mean().alias("target_obs_value"),
                    pl.col("obs_value_num").std().fill_null(0.0).alias("obs_std"),
                ]
            )
            .drop_nulls(subset=["target_obs_value"])
            .filter(pl.col("target_obs_value").is_not_nan())
        )
        self._write_derived(
            asia_derived,
            dataset_id="asia__oecd__stfreight_mode_fuel_profile_derived_csv",
            category="fuel_energy_efficiency",
            target="target_obs_value",
            task_type="regression",
            source="asia",
        )

        asia_hk = self._read_dataset(asia_hk_id).with_columns(self._to_float(pl.col("figure")).alias("figure_num"))
        asia_hk_derived = (
            asia_hk.group_by(["period", "ind", "pe_grp"])
            .agg(
                [
                    pl.col("figure_num").mean().alias("target_workforce_volume"),
                    pl.col("figure_num").std().fill_null(0.0).alias("figure_std"),
                    pl.len().alias("records"),
                ]
            )
            .drop_nulls(subset=["target_workforce_volume"])
            .filter(pl.col("target_workforce_volume").is_not_nan())
        )
        self._write_derived(
            asia_hk_derived,
            dataset_id="asia__hong_kong__courier_workforce_profile_derived_csv",
            category="workforce_shift_planning",
            target="target_workforce_volume",
            task_type="regression",
            source="asia",
        )

        eu_oecd = self._read_dataset(eu_oecd_id).with_columns(self._to_float(pl.col("obs_value")).alias("obs_value_num"))
        eu_oecd_derived = (
            eu_oecd.group_by(["ref_area", "vehicle_type", "fuel", "time_period"])
            .agg(
                [
                    pl.col("obs_value_num").mean().alias("target_obs_value"),
                    pl.col("obs_value_num").std().fill_null(0.0).alias("obs_std"),
                ]
            )
            .drop_nulls(subset=["target_obs_value"])
            .filter(pl.col("target_obs_value").is_not_nan())
        )
        self._write_derived(
            eu_oecd_derived,
            dataset_id="europe__oecd__sttraffic_vehicle_fuel_profile_derived_csv",
            category="route_eta_reliability",
            target="target_obs_value",
            task_type="regression",
            source="europe",
        )

        eu_road = self._read_dataset(eu_road_id).with_columns(self._to_float(pl.col("obs_value")).alias("obs_value_num"))
        eu_road_derived = (
            eu_road.group_by(["geo", "c_load", "c_unload", "time_period"])
            .agg(
                [
                    pl.col("obs_value_num").mean().alias("target_lane_flow"),
                    pl.col("obs_value_num").std().fill_null(0.0).alias("flow_std"),
                ]
            )
            .drop_nulls(subset=["target_lane_flow"])
            .filter(pl.col("target_lane_flow").is_not_nan())
        )
        self._write_derived(
            eu_road_derived,
            dataset_id="europe__eurostat__cross_border_lane_flow_derived_csv",
            category="cross_border_trucking",
            target="target_lane_flow",
            task_type="regression",
            source="europe",
        )

        eu_uk = self._read_dataset(eu_uk_id).with_columns(
            [
                self._to_float(pl.col("numberofvehiclesauthorised")).alias("vehicles_auth"),
                self._to_float(pl.col("vehiclesspecified")).alias("vehicles_specified"),
                self._to_float(pl.col("numberoftrailersauthorised")).alias("trailers_auth"),
                self._to_float(pl.col("trailersspecified")).alias("trailers_specified"),
            ]
        )
        eu_uk_derived = (
            eu_uk.group_by(["licencetype", "operatortype", "ocaddress"])
            .agg(
                [
                    pl.col("vehicles_auth").mean().alias("target_shift_capacity"),
                    pl.col("vehicles_specified").mean().alias("vehicles_specified_mean"),
                    pl.col("trailers_auth").mean().alias("trailers_auth_mean"),
                    pl.col("trailers_specified").mean().alias("trailers_specified_mean"),
                    pl.len().alias("operator_count"),
                ]
            )
            .drop_nulls(subset=["target_shift_capacity"])
            .filter(pl.col("target_shift_capacity").is_not_nan())
        )
        self._write_derived(
            eu_uk_derived,
            dataset_id="europe__uk__operator_license_shift_capacity_derived_csv",
            category="workforce_shift_planning",
            target="target_shift_capacity",
            task_type="regression",
            source="europe",
        )

    def save(self) -> None:
        self.spec_obj["datasets"] = self.specs
        self.spec_path.write_text(json.dumps(self.spec_obj, indent=2) + "\n")
        self.meta_obj["datasets"] = self.meta
        self.meta_path.write_text(json.dumps(self.meta_obj, indent=2) + "\n")

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "industry": self.industry,
            "derived_count": len(self.derived),
            "datasets": [
                {
                    "dataset_id": d.dataset_id,
                    "category": d.category,
                    "target": d.target,
                    "task_type": d.task_type,
                    "source": d.source,
                    "rows": d.rows,
                    "cols": d.cols,
                    "file_path": d.file_path,
                }
                for d in self.derived
            ],
        }
        self.report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    deriver = Deriver(base)
    deriver.derive_cold_chain()
    deriver.derive_amazon_last_mile()
    deriver.derive_asia_europe_quota_fill()
    deriver.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
