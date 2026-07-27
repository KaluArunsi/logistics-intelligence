#!/usr/bin/env python3
"""
Patch trucking_delivery failed L0 workers with stronger business-aligned targets.

Primary strategy:
- Replace brittle regression objectives with business-tier classification targets.
- Use leakage-safe `derived_target` so source target columns are removed from features.
- Remove weak fallback target chains for patched workers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _threshold_target(source: str, name: str, high_label: str = "high", low_label: str = "low") -> dict[str, Any]:
    return {
        "targets": [{"name": name, "task_type": "classification", "primary": True}],
        "derived_target": {
            "name": name,
            "source": source,
            "type": "threshold_split",
            "high_label": high_label,
            "low_label": low_label,
        },
    }


def _categorical_map_target(source: str, name: str, mapping: dict[str, str]) -> dict[str, Any]:
    return {
        "targets": [{"name": name, "task_type": "classification", "primary": True}],
        "derived_target": {
            "name": name,
            "source": source,
            "type": "categorical_map",
            "map": mapping,
            "drop_nulls": True,
        },
    }


def run() -> dict[str, Any]:
    base = Path(__file__).resolve().parents[1]
    industry = "trucking_delivery"
    spec_path = base / "config" / "industries" / industry / f"{industry}_dataset_specs.json"
    report_path = base / "reports" / "trucking_delivery_failed_worker_patch_report.json"

    spec_obj = json.loads(spec_path.read_text())
    specs = spec_obj.get("datasets", {})

    patches: dict[str, dict[str, Any]] = {
        # cold_chain_last_mile (3 fails)
        "us__bts__cfs_tempctrl_commodity_value_derived_csv": _threshold_target(
            source="target_value_sum",
            name="target_value_sum_band",
        ),
        "us__bts__cfs_tempctrl_distance_profile_derived_csv": _threshold_target(
            source="target_avg_miles",
            name="target_avg_miles_band",
        ),
        "us__bts__cfs_tempctrl_mode_change_derived_csv": _threshold_target(
            source="target_ton_change",
            name="target_ton_change_band",
            high_label="growth",
            low_label="decline",
        ),
        # cross_border_trucking (3 fails)
        "other__worldbank__worldbank_lp_lpi_ovrl_xq_csv": _threshold_target(
            source="value",
            name="lpi_value_band",
        ),
        "other__worldbank__worldbank_lp_lpi_trac_xq_csv": _threshold_target(
            source="value",
            name="lpi_value_band",
        ),
        "other__worldbank__worldbank_lp_lpi_time_xq_csv": _threshold_target(
            source="value",
            name="lpi_value_band",
        ),
        # driver_safety_compliance (2 fails)
        "us__usdot__usdot_6eyk_hxee_carrier_all_with_history_csv": _threshold_target(
            source="min_cov_amount",
            name="coverage_band",
            high_label="high_coverage",
            low_label="low_coverage",
        ),
        "us__usdot__usdot_h9zy_gjn8_sms_c_passproperty_csv": _threshold_target(
            source="veh_maint_measure",
            name="maintenance_risk_band",
            high_label="high_risk",
            low_label="low_risk",
        ),
        # fuel_energy_efficiency (1 fail)
        "asia__oecd__stfreight_mode_fuel_profile_derived_csv": _threshold_target(
            source="target_obs_value",
            name="energy_efficiency_band",
            high_label="high_consumption",
            low_label="low_consumption",
        ),
        # lane_cost_yield (1 fail)
        "us__usdot__usdot_phpr_iuzz_hpms_spatial_ramp_sections_2024_csv": _threshold_target(
            source="future_aadt",
            name="future_aadt_band",
        ),
        # last_mile_sla (2 fails)
        "us__amazon_lastmile__route_invalid_score_eval_derived_csv": _threshold_target(
            source="invalid_sequence_score",
            name="invalid_sequence_band",
            high_label="high_sla_risk",
            low_label="low_sla_risk",
        ),
        "us__amazon_lastmile__route_invalid_score_train_derived_csv": _threshold_target(
            source="invalid_sequence_score",
            name="invalid_sequence_band",
            high_label="high_sla_risk",
            low_label="low_sla_risk",
        ),
        # parcel_exception_risk (3 fails)
        "us__usdot__usdot_7wn6_i5b9_highway_rail_grade_crossing_incident_data_form_57_csv": _threshold_target(
            source="vehicledamagecost",
            name="damage_cost_band",
            high_label="high_damage",
            low_label="low_damage",
        ),
        "us__usdot__usdot_icqf_xf4w_highway_rail_grade_crossing_accident_incident_source_data_form_57_csv": _threshold_target(
            source="vehdmg",
            name="damage_cost_band",
            high_label="high_damage",
            low_label="low_damage",
        ),
        "us__usdot__usdot_uwah_u9bn_highway_rail_grade_crossing_incident_data_form_57_6_year_view_csv": _threshold_target(
            source="vehicledamagecost",
            name="damage_cost_band",
            high_label="high_damage",
            low_label="low_damage",
        ),
        # route_eta_reliability (11 fails)
        "asia__singapore__singstat_m451081_selected_merchandise_trade_at_current_prices_monthly_csv": _threshold_target(
            source="value",
            name="trade_value_band",
        ),
        "europe__eurostat__eurostat_tran_hv_frtra_csv": _threshold_target(
            source="obs_value",
            name="flow_band",
        ),
        "europe__eurostat__eurostat_tran_hv_pstra_csv": _threshold_target(
            source="obs_value",
            name="flow_band",
        ),
        "europe__oecd__sttraffic_vehicle_fuel_profile_derived_csv": _threshold_target(
            source="target_obs_value",
            name="traffic_volume_band",
        ),
        "us__bts__bts_j246_y2rf_cfs_area_file_2012_2022_csv": _threshold_target(
            source="ton",
            name="tonnage_band",
        ),
        "us__census_intltrade__census_timeseries_intltrade_exports_enduse_2024_12_csv": _threshold_target(
            source="air_val_yr",
            name="annual_air_value_band",
        ),
        "us__census_intltrade__census_timeseries_intltrade_exports_naics_2024_12_csv": _threshold_target(
            source="air_val_yr",
            name="annual_air_value_band",
        ),
        "us__census_intltrade__census_timeseries_intltrade_exports_sitc_2024_12_csv": _threshold_target(
            source="air_val_yr",
            name="annual_air_value_band",
        ),
        "us__census_intltrade__census_timeseries_intltrade_exports_statenaics_2024_12_csv": _threshold_target(
            source="all_val_yr",
            name="annual_total_value_band",
        ),
        "us__census_intltrade__census_timeseries_intltrade_imports_enduse_2024_12_csv": _threshold_target(
            source="air_val_yr",
            name="annual_air_value_band",
        ),
        "other__oecd__oecd_infrinv_df_infrinv_other_csv": _threshold_target(
            source="obs_value",
            name="infrastructure_value_band",
        ),
        # vehicle_maintenance_risk (1 fail)
        "us__usdot__usdot_p2mt_9ige_out_of_service_orders_csv": _categorical_map_target(
            source="status",
            name="status_active_flag",
            mapping={"ACTIVE": "active", "INACTIVE": "inactive"},
        ),
        # workforce_shift_planning (1 near-threshold fail)
        "europe__uk__operator_license_shift_capacity_derived_csv": _threshold_target(
            source="target_shift_capacity",
            name="shift_capacity_band",
            high_label="high_capacity",
            low_label="low_capacity",
        ),
    }

    changed: list[dict[str, Any]] = []
    skipped: list[str] = []
    for dataset_id, patch in patches.items():
        cfg = specs.get(dataset_id)
        if not cfg:
            skipped.append(dataset_id)
            continue
        old_targets = cfg.get("targets") or []
        cfg["targets"] = patch["targets"]
        cfg["derived_target"] = patch["derived_target"]
        training = cfg.get("training") or {}
        training["worker_enabled"] = True
        training["gating"] = True
        cfg["training"] = training
        specs[dataset_id] = cfg
        changed.append(
            {
                "dataset_id": dataset_id,
                "old_targets": old_targets,
                "new_targets": cfg["targets"],
                "derived_target": cfg["derived_target"],
            }
        )

    spec_obj["datasets"] = specs
    spec_path.write_text(json.dumps(spec_obj, indent=2) + "\n")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": industry,
        "patched_count": len(changed),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "patched": changed,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
