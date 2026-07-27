#!/usr/bin/env python3
"""
Curate trucking_delivery active workers to satisfy hard constraints:
- Region quotas: US 50, Europe 35, Asia 30, Other 15
- Minimum 5 active workers per planned category (15 categories)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml


REGION_QUOTAS = {"us": 50, "europe": 35, "asia": 30, "other": 15}
MIN_PER_CATEGORY = 5
LOW_VALUE_EXACT = {
    "year",
    "period",
    "series_no",
    "row_labels",
    "price_base",
    "unit_mult",
    "decimal",
    "decimals",
    "ranking",
    "rank",
    "index",
    "dot_number",
}
LOW_VALUE_PATTERNS = [
    re.compile(r".*_ranking$"),
    re.compile(r"^rank_.*"),
    re.compile(r"^row_.*"),
    re.compile(r".*_label(s)?$"),
    re.compile(r"^label(s)?$"),
]


def _is_low_value_target(name: str | None) -> bool:
    token = str(name or "").strip().lower()
    if not token:
        return True
    if token in LOW_VALUE_EXACT:
        return True
    if token in {"id", "row_id", "record_id", "uuid", "uid"}:
        return True
    if token.endswith("_id"):
        return True
    return any(p.match(token) for p in LOW_VALUE_PATTERNS)


def _dataset_id_from_file(path: Path) -> str:
    name = path.name
    if name.endswith(".parquet.zstd"):
        return name[: -len(".parquet.zstd")]
    if name.endswith(".parquet"):
        return name[: -len(".parquet")]
    return path.stem


@dataclass
class Candidate:
    dataset_id: str
    region: str
    category: str
    rows: int
    cols: int
    target: str
    task_type: str
    file_path: Path


class Curator:
    def __init__(self, base: Path):
        self.base = base
        self.industry = "trucking_delivery"
        self.spec_path = (
            base
            / "config"
            / "industries"
            / self.industry
            / f"{self.industry}_dataset_specs.json"
        )
        self.industry_yaml = base / "config" / "industries" / self.industry / "industry.yaml"
        self.report_path = base / "reports" / "trucking_delivery_active_set_report.json"
        self.processed_root = base / "data" / "processed" / self.industry

        self.spec_obj = json.loads(self.spec_path.read_text())
        self.specs = self.spec_obj.get("datasets", {})

        industry_cfg = yaml.safe_load(self.industry_yaml.read_text()) or {}
        self.planned_categories: list[str] = list((industry_cfg.get("categories") or {}).keys())

        self.category_reassignments = self._build_category_reassignments()

    def _build_category_reassignments(self) -> dict[str, str]:
        return {
            # Last-mile SLA: route quality and stop-level reliability workers.
            "us__amazon_lastmile__route_invalid_score_train_derived_csv": "last_mile_sla",
            "us__amazon_lastmile__route_invalid_score_eval_derived_csv": "last_mile_sla",
            "us__usdot__usdot_42um_tgh5_hpms_spatial_all_sections_2024_csv": "last_mile_sla",
            "us__usdot__usdot_8vwk_s3iq_hpms_spatial_other_principal_arterial_sections_2024_csv": "last_mile_sla",
            "us__usdot__usdot_jz72_ehnf_hpms_spatial_other_freeway_expressway_sections_2024_csv": "last_mile_sla",
            # Keep pickup/dropoff with route reliability slices.
            "us__usdot__usdot_uta5_4eu5_bts_atri_freight_mobility_initiative_county_to_county_truck_travel_times_2024_experimental_csv": "pickup_dropoff_reliability",
            "us__usdot__usdot_ez58_m3b4_bts_atri_freight_mobility_initiative_county_to_county_truck_travel_times_2023_experimental_csv": "pickup_dropoff_reliability",
            "us__usdot__usdot_mayv_2qfz_bts_atri_freight_mobility_initiative_county_to_county_truck_travel_times_2021_experimental_csv": "pickup_dropoff_reliability",
            # Parcel exception risk: explicit route exception + incident-derived exception proxies.
            "us__amazon_lastmile__route_exception_flag_train_derived_csv": "parcel_exception_risk",
            "us__amazon_lastmile__route_exception_flag_eval_derived_csv": "parcel_exception_risk",
            "us__usdot__usdot_7wn6_i5b9_highway_rail_grade_crossing_incident_data_form_57_csv": "parcel_exception_risk",
            "us__usdot__usdot_uwah_u9bn_highway_rail_grade_crossing_incident_data_form_57_6_year_view_csv": "parcel_exception_risk",
            "us__usdot__usdot_icqf_xf4w_highway_rail_grade_crossing_accident_incident_source_data_form_57_csv": "parcel_exception_risk",
            # Workforce shift planning: operator licensing and driver supply signals.
            "asia__hong_kong__courier_workforce_profile_derived_csv": "workforce_shift_planning",
            "europe__uk__operator_license_shift_capacity_derived_csv": "workforce_shift_planning",
            "europe__uk__uk_1699fb94_59853277_northern_ireland_goods_vehicle_operator_s_licence_records_northern_ireland_goods_vehicle_ope_6ccbd8f6a4": "workforce_shift_planning",
            "europe__uk__uk_2a67d1ee_2a6fc4bd_traffic_commissioners_goods_and_public_service_vehicle_operator_licence_records_olbslicencer_6366599eba": "workforce_shift_planning",
            "us__usdot__usdot_dwyf_zaik_licensed_drivers_by_state_1949_2024_dl_201_csv": "workforce_shift_planning",
        }

    def _target_from_spec(self, spec: dict[str, Any]) -> tuple[str | None, str | None]:
        targets = spec.get("targets") or []
        if isinstance(targets, list) and targets:
            primary = next((t for t in targets if t.get("primary")), None) or targets[0]
            return primary.get("name"), primary.get("task_type")
        return None, None

    def _apply_reassignments(self) -> None:
        for dataset_id, new_category in self.category_reassignments.items():
            spec = self.specs.get(dataset_id)
            if not spec:
                continue
            spec["category"] = new_category
            spec["categories"] = [new_category]
            self.specs[dataset_id] = spec

    def _collect_candidates(self) -> list[Candidate]:
        by_dataset: dict[str, Path] = {}
        for p in self.processed_root.rglob("*.parquet.zstd"):
            did = _dataset_id_from_file(p)
            by_dataset[did] = p

        out: list[Candidate] = []
        for dataset_id, spec in self.specs.items():
            path = by_dataset.get(dataset_id)
            if path is None:
                continue
            target, task_type = self._target_from_spec(spec)
            if not target:
                continue
            if _is_low_value_target(target):
                continue
            try:
                lf = pl.scan_parquet(path)
                rows = int(lf.select(pl.len()).collect().item())
                cols = len(lf.collect_schema().names())
            except Exception:
                continue
            if rows < 30 or cols < 2:
                continue

            region = (dataset_id.split("__", 1)[0] if "__" in dataset_id else "unknown").lower()
            category = str((spec.get("categories") or [spec.get("category") or "unknown"])[0])
            out.append(
                Candidate(
                    dataset_id=dataset_id,
                    region=region,
                    category=category,
                    rows=rows,
                    cols=cols,
                    target=target,
                    task_type=task_type or "regression",
                    file_path=path,
                )
            )
        return out

    def _initial_region_selection(self, candidates: list[Candidate]) -> set[str]:
        selected: set[str] = set()
        by_region: dict[str, list[Candidate]] = {}
        for region in REGION_QUOTAS:
            region_rows = [c for c in candidates if c.region == region]
            region_rows.sort(key=lambda c: (-c.rows, c.dataset_id))
            by_region[region] = region_rows
            if len(region_rows) < REGION_QUOTAS[region]:
                raise RuntimeError(
                    f"Insufficient eligible candidates for region {region}: "
                    f"{len(region_rows)} < {REGION_QUOTAS[region]}"
                )
            for c in region_rows[: REGION_QUOTAS[region]]:
                selected.add(c.dataset_id)
        return selected

    @staticmethod
    def _counts(selected: set[str], all_candidates: list[Candidate]) -> tuple[Counter, Counter]:
        reg = Counter()
        cat = Counter()
        for c in all_candidates:
            if c.dataset_id in selected:
                reg[c.region] += 1
                cat[c.category] += 1
        return reg, cat

    def _find_swap_out(
        self,
        selected: set[str],
        candidates: list[Candidate],
        region: str,
        protected_categories: set[str],
    ) -> str | None:
        _, cat_counts = self._counts(selected, candidates)
        in_region = [
            c for c in candidates
            if c.dataset_id in selected and c.region == region and c.category not in protected_categories
        ]
        # Remove from the largest surplus category first.
        in_region.sort(key=lambda c: (-(cat_counts.get(c.category, 0) - MIN_PER_CATEGORY), c.rows, c.dataset_id))
        for c in in_region:
            if cat_counts.get(c.category, 0) > MIN_PER_CATEGORY:
                return c.dataset_id
        return None

    def _enforce_category_minimums(self, selected: set[str], candidates: list[Candidate]) -> set[str]:
        id_to_candidate = {c.dataset_id: c for c in candidates}
        for _ in range(500):
            reg_counts, cat_counts = self._counts(selected, candidates)
            deficits = [cat for cat in self.planned_categories if cat_counts.get(cat, 0) < MIN_PER_CATEGORY]
            if not deficits:
                # Validate region quotas remain exact.
                for region, quota in REGION_QUOTAS.items():
                    if reg_counts.get(region, 0) != quota:
                        raise RuntimeError(
                            f"Region quota drift after category balancing: {region}={reg_counts.get(region, 0)} expected {quota}"
                        )
                return selected

            cat = deficits[0]
            # Candidates not selected, sorted by rows descending.
            pool = [c for c in candidates if c.category == cat and c.dataset_id not in selected]
            pool.sort(key=lambda c: (-c.rows, c.dataset_id))
            if not pool:
                raise RuntimeError(f"Cannot satisfy category minimum for {cat}: no spare candidates")

            added = False
            for cand in pool:
                region = cand.region
                if region not in REGION_QUOTAS:
                    continue
                # If region has room (should not happen with exact initial fill), add directly.
                if reg_counts.get(region, 0) < REGION_QUOTAS[region]:
                    selected.add(cand.dataset_id)
                    added = True
                    break
                # Swap inside same region to preserve exact quotas.
                swap_out = self._find_swap_out(
                    selected=selected,
                    candidates=candidates,
                    region=region,
                    protected_categories=set(deficits),
                )
                if swap_out is None:
                    continue
                selected.remove(swap_out)
                selected.add(cand.dataset_id)
                added = True
                break

            if not added:
                raise RuntimeError(f"Failed to rebalance category {cat} within region quotas")

        raise RuntimeError("Category balancing exceeded iteration budget")

    def _persist_worker_enabled(self, selected: set[str]) -> None:
        for dataset_id, spec in self.specs.items():
            training = spec.get("training") or {}
            training["worker_enabled"] = dataset_id in selected
            # Keep active workers as gating workers; disabled workers stay non-gating.
            training["gating"] = bool(dataset_id in selected)
            spec["training"] = training
            self.specs[dataset_id] = spec

        self.spec_obj["datasets"] = self.specs
        self.spec_path.write_text(json.dumps(self.spec_obj, indent=2) + "\n")

    def run(self) -> dict[str, Any]:
        self._apply_reassignments()
        candidates = self._collect_candidates()
        selected = self._initial_region_selection(candidates)
        selected = self._enforce_category_minimums(selected, candidates)
        self._persist_worker_enabled(selected)

        reg_counts, cat_counts = self._counts(selected, candidates)
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "industry": self.industry,
            "selected_workers": len(selected),
            "region_counts": dict(reg_counts),
            "category_counts": dict(cat_counts),
            "min_per_category": MIN_PER_CATEGORY,
            "region_quotas": REGION_QUOTAS,
            "planned_categories": self.planned_categories,
            "missing_categories": [c for c in self.planned_categories if cat_counts.get(c, 0) < MIN_PER_CATEGORY],
            "selected_dataset_ids": sorted(selected),
        }
        self.report_path.write_text(json.dumps(report, indent=2) + "\n")
        return report


def main() -> int:
    base = Path(__file__).resolve().parents[1]
    curator = Curator(base)
    report = curator.run()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
