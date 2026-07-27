#!/usr/bin/env python3
"""
Internet-backed worker target audit for trucking_delivery.

Builds per-worker target recommendations aligned to business use-cases, with
source evidence URLs, and can optionally apply those target plans to dataset specs.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass
class TargetRule:
    pattern: str
    task_type: str
    priority: int
    rationale: str


CATEGORY_RULES: dict[str, list[TargetRule]] = {
    "route_eta_reliability": [
        TargetRule(r"tt_percentile_75|ttpercentile75|travel.*time|eta|delay|journey|endpoint", "regression", 100, "ETA and delay reliability"),
        TargetRule(r"on_time|late|arrival.*flag|breach", "classification", 80, "On-time SLA risk classification"),
    ],
    "last_mile_sla": [
        TargetRule(r"tt_percentile_75|travel.*time|delay|endpoint|service.*time", "regression", 100, "Delivery SLA latency"),
        TargetRule(r"exception|failed_attempt|breach|invalid_sequence|flag", "classification", 90, "Last-mile exception risk"),
    ],
    "pickup_dropoff_reliability": [
        TargetRule(r"tt_percentile_75|travel.*time|delay|service.*time|queue", "regression", 100, "Pickup/dropoff timing reliability"),
        TargetRule(r"missed|slot|breach|failed|flag", "classification", 85, "Pickup/dropoff failure risk"),
    ],
    "dispatch_capacity_balance": [
        TargetRule(r"capacity|flow|volume|count|vehicles|throughput|backlog", "regression", 100, "Capacity/load balance"),
        TargetRule(r"pressure|imbalance|overload|risk|band|class", "classification", 80, "Capacity stress tiering"),
    ],
    "driver_safety_compliance": [
        TargetRule(r"injur|fatal|crash|incident|violat|unsafe|out_of_service|inspection", "regression", 100, "Safety event intensity"),
        TargetRule(r"violat|risk|fail|unsafe|flag|class|band|status", "classification", 90, "Safety compliance risk state"),
    ],
    "fleet_utilization": [
        TargetRule(r"utilization|idle|empty|distance|mile|trip|flow|volume|count|obs_value|value", "regression", 100, "Fleet productivity"),
        TargetRule(r"utilization.*band|class|bucket|risk", "classification", 70, "Utilization tiering"),
    ],
    "vehicle_maintenance_risk": [
        TargetRule(r"maintenance|repair|fault|breakdown|equipment|service|vehiclesauthorised", "regression", 100, "Maintenance burden and exposure"),
        TargetRule(r"maintenance.*flag|failure|risk|band|class|status", "classification", 85, "Maintenance failure risk"),
    ],
    "fuel_energy_efficiency": [
        TargetRule(r"fuel|energy|consumption|efficiency|mpg|emission|co2|obs_value|value", "regression", 100, "Fuel/energy efficiency"),
        TargetRule(r"efficiency.*band|class|flag", "classification", 75, "Efficiency class/risk"),
    ],
    "lane_cost_yield": [
        TargetRule(r"cost|revenue|price|yield|margin|rate|value|obs_value", "regression", 100, "Lane economics"),
        TargetRule(r"yield.*band|margin.*band|price.*band|class", "classification", 80, "Lane yield segmentation"),
    ],
    "parcel_exception_risk": [
        TargetRule(r"exception|damage|claim|incident|invalid_sequence|injur|casinjrr|crossingusersinjured|totalinjured", "regression", 100, "Exception severity"),
        TargetRule(r"exception|damage|claim|risk|flag|class|band", "classification", 90, "Exception probability"),
    ],
    "reverse_logistics_returns": [
        TargetRule(r"return|restock|cycle|pickup|delay|value|obs_value", "regression", 100, "Return cycle performance"),
        TargetRule(r"return.*flag|restock.*flag|risk|class|band", "classification", 80, "Return risk segmentation"),
    ],
    "cross_border_trucking": [
        TargetRule(r"border|crossing|queue|time|delay|flow|volume|lpi|obs_value|value", "regression", 100, "Cross-border movement performance"),
        TargetRule(r"border.*risk|crossing.*risk|lpi.*band|class|flag", "classification", 85, "Cross-border risk/tier"),
    ],
    "urban_traffic_risk": [
        TargetRule(r"traffic|congestion|delay|incident|injur|casualty|obs_value|value", "regression", 100, "Urban traffic risk intensity"),
        TargetRule(r"congestion.*risk|incident.*risk|disruption|flag|class|band", "classification", 85, "Traffic disruption risk"),
    ],
    "cold_chain_last_mile": [
        TargetRule(r"temp|temperature|cold|excursion|spoil|ton|value|obs_value", "regression", 100, "Cold-chain quality/performance"),
        TargetRule(r"excursion|cold_chain_break|spoilage|flag|risk|class|band", "classification", 90, "Cold-chain breach risk"),
    ],
    "workforce_shift_planning": [
        TargetRule(r"target_shift_capacity|shift|coverage|overtime|driver|workforce|operator_count|vehiclesauthorised", "regression", 100, "Shift coverage planning"),
        TargetRule(r"shift_capacity_band|shortage|coverage_gap|flag|band|class", "classification", 80, "Shift shortage tiering"),
    ],
}


SOURCE_EVIDENCE: list[dict[str, Any]] = [
    {
        "prefix": "us__usdot",
        "name": "US DOT / FMCSA / FHWA / FRA",
        "urls": [
            "https://csa.fmcsa.dot.gov/",
            "https://www.fmcsa.dot.gov/mission/policy/what-csa",
            "https://www.fhwa.dot.gov/policyinformation/hpms/fieldmanual/",
            "https://railroads.dot.gov/safety-data/crossing-and-inventory-data-download",
        ],
    },
    {
        "prefix": "us__bts",
        "name": "BTS / ATRI / CFS",
        "urls": [
            "https://data.transportation.gov/Roadways-and-Bridges/BTS-ATRI-Freight-Mobility-Initiative-County-to-Coun/uta5-4eu5",
            "https://www.bts.gov/commodity-flow-survey",
        ],
    },
    {
        "prefix": "us__census_intltrade",
        "name": "US Census International Trade",
        "urls": [
            "https://www.census.gov/data/developers/data-sets/international-trade.html",
            "https://www.census.gov/foreign-trade/reference/products/catalog/nomenclature/index.html",
        ],
    },
    {
        "prefix": "europe__eurostat",
        "name": "Eurostat Transport Metadata",
        "urls": [
            "https://ec.europa.eu/eurostat/cache/metadata/en/road_esms.htm",
            "https://ec.europa.eu/eurostat/cache/metadata/en/rail_esms.htm",
        ],
    },
    {
        "prefix": "europe__oecd",
        "name": "OECD Transport Indicators",
        "urls": [
            "https://sdmx.oecd.org/",
            "https://www.oecd.org/en/data/insights/data-explainers/2024/05/transport-data.html",
        ],
    },
    {
        "prefix": "asia__oecd",
        "name": "OECD Transport Indicators",
        "urls": [
            "https://sdmx.oecd.org/",
            "https://www.oecd.org/en/data/insights/data-explainers/2024/05/transport-data.html",
        ],
    },
    {
        "prefix": "other__oecd",
        "name": "OECD Transport Indicators",
        "urls": [
            "https://sdmx.oecd.org/",
            "https://www.oecd.org/en/data/insights/data-explainers/2024/05/transport-data.html",
        ],
    },
    {
        "prefix": "other__worldbank",
        "name": "World Bank LPI",
        "urls": [
            "https://lpi.worldbank.org/",
            "https://www.worldbank.org/en/programs/world-bank-enterprise-surveys/brief/lpi",
        ],
    },
    {
        "prefix": "us__amazon_lastmile",
        "name": "Amazon Last Mile Challenge",
        "urls": [
            "https://registry.opendata.aws/amazon-last-mile-challenges/",
            "https://www.amazon.science/publications/learning-to-generalize-for-vehicle-routing",
        ],
    },
    {
        "prefix": "asia__singapore",
        "name": "Singapore Department of Statistics",
        "urls": [
            "https://www.singstat.gov.sg/",
            "https://tablebuilder.singstat.gov.sg/",
        ],
    },
    {
        "prefix": "asia__hong_kong",
        "name": "Hong Kong Census and Statistics Department",
        "urls": [
            "https://www.censtatd.gov.hk/en/scode200.html",
            "https://www.data.gov.hk/en-data/dataset/",
        ],
    },
    {
        "prefix": "europe__uk",
        "name": "UK Operator Licensing and Transport Data",
        "urls": [
            "https://www.gov.uk/government/collections/operator-licensing-statistics",
            "https://www.gov.uk/government/organisations/traffic-commissioners",
        ],
    },
]


ID_LIKE_PATTERNS = [
    r"(^|_)id($|_)",
    r"^index$",
    r"^row_",
    r"code$",
    r"^uuid$",
    r"^uid$",
]


def is_id_like(col: str) -> bool:
    c = col.lower()
    return any(re.search(p, c) for p in ID_LIKE_PATTERNS)


def infer_task_type(col: str, dtype: pl.DataType, preferred: str) -> str:
    c = col.lower()
    if any(k in c for k in ["flag", "band", "class", "status", "risk_tier", "bucket", "label"]):
        return "classification"
    if dtype in {pl.Utf8, pl.String, pl.Categorical, pl.Boolean}:
        return "classification"
    return preferred


def get_schema_columns(path_str: str) -> dict[str, pl.DataType]:
    try:
        lf = pl.scan_parquet(path_str)
        return lf.schema
    except Exception:
        return {}


def pick_targets(
    category: str,
    schema: dict[str, pl.DataType],
    current_targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cols = list(schema.keys())
    col_lc_map = {c.lower(): c for c in cols}
    rules = CATEGORY_RULES.get(category, [])
    scored: list[tuple[int, str, str, str]] = []

    for rule in rules:
        pat = re.compile(rule.pattern)
        for lc, orig in col_lc_map.items():
            if not pat.search(lc):
                continue
            if is_id_like(orig):
                continue
            task = infer_task_type(orig, schema[orig], rule.task_type)
            scored.append((rule.priority, orig, task, rule.rationale))

    # Keep current configured targets first-class to avoid regressions.
    for t in current_targets:
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        actual = col_lc_map.get(name.lower())
        if actual is None:
            continue
        task = str(t.get("task_type", infer_task_type(actual, schema.get(actual, pl.Utf8), "regression")))
        scored.append((110, actual, task, "Existing configured target"))

    # Deduplicate by name while keeping highest score.
    best: dict[str, tuple[int, str, str]] = {}
    for score, name, task, rationale in scored:
        prev = best.get(name)
        if prev is None or score > prev[0]:
            best[name] = (score, task, rationale)

    ordered = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))
    selected: list[dict[str, Any]] = []
    for idx, (name, (score, task, rationale)) in enumerate(ordered[:3]):
        selected.append(
            {
                "name": name,
                "task_type": task,
                "primary": idx == 0,
                "priority_score": score,
                "rationale": rationale,
            }
        )
    return selected


def resolve_source_evidence(dataset_id: str) -> dict[str, Any]:
    for source in SOURCE_EVIDENCE:
        if dataset_id.startswith(source["prefix"] + "__") or dataset_id.startswith(source["prefix"]):
            return source
    return {"name": "General transport ML references", "urls": []}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", default="trucking_delivery")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    spec_path = base / "config" / "industries" / args.industry / f"{args.industry}_dataset_specs.json"
    latest_complete_run = base / "reports" / "runs" / args.industry / f"{args.industry}_20260217_154513" / "reports" / "dataset_details.json"
    audit_path = base / "reports" / f"{args.industry}_worker_target_audit.json"

    spec_obj = json.loads(spec_path.read_text())
    datasets = spec_obj["datasets"] if "datasets" in spec_obj else spec_obj
    details_obj = json.loads(latest_complete_run.read_text())
    details_by_id = {d["id"]: d for d in details_obj.get("datasets", [])}

    audit_rows: list[dict[str, Any]] = []
    updated = 0
    skipped = 0

    for dataset_id, spec in sorted(datasets.items()):
        if spec.get("skip_training"):
            skipped += 1
            continue
        training = spec.get("training", {}) or {}
        if training.get("worker_enabled", True) is False:
            skipped += 1
            continue
        if spec.get("status") == "disabled":
            skipped += 1
            continue

        detail = details_by_id.get(dataset_id)
        file_path = (detail or {}).get("file_path")
        schema = get_schema_columns(file_path) if file_path else {}
        current_targets = spec.get("targets", []) or []
        category = spec.get("category") or ((spec.get("categories") or ["operations"])[0])
        source = resolve_source_evidence(dataset_id)
        recommended = pick_targets(category, schema, current_targets)

        if not recommended and current_targets:
            # Keep existing targets if schema introspection failed.
            recommended = [
                {
                    "name": t.get("name"),
                    "task_type": t.get("task_type", "regression"),
                    "primary": i == 0,
                    "priority_score": 50,
                    "rationale": "Fallback to existing configured target",
                }
                for i, t in enumerate(current_targets[:3])
                if t.get("name")
            ]

        if recommended:
            new_targets = []
            for i, rec in enumerate(recommended):
                new_targets.append(
                    {
                        "name": rec["name"],
                        "task_type": rec["task_type"],
                        "primary": i == 0,
                    }
                )
            audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    "category": category,
                    "business_use_case": category,
                    "source_family": source["name"],
                    "source_urls": source["urls"],
                    "current_targets": current_targets,
                    "recommended_targets": new_targets,
                    "rationale": [r["rationale"] for r in recommended],
                }
            )
            if args.apply:
                spec["targets"] = new_targets
                if len(new_targets) > 1:
                    training["allow_target_fallback"] = True
                    training["fallback_on_fail"] = True
                spec["training"] = training
                spec["target_alignment"] = {
                    "business_use_case": category,
                    "source_family": source["name"],
                    "source_urls": source["urls"],
                }
                updated += 1
        else:
            audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    "category": category,
                    "business_use_case": category,
                    "source_family": source["name"],
                    "source_urls": source["urls"],
                    "current_targets": current_targets,
                    "recommended_targets": [],
                    "rationale": ["No suitable target candidate from schema; manual review required."],
                }
            )

    audit_payload = {
        "industry": args.industry,
        "active_workers_audited": len(audit_rows),
        "workers_updated": updated if args.apply else 0,
        "workers_skipped": skipped,
        "audit_rows": audit_rows,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n")
    if args.apply:
        spec_path.write_text(json.dumps(spec_obj, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "industry": args.industry,
                "audited": len(audit_rows),
                "updated": updated if args.apply else 0,
                "skipped": skipped,
                "audit_report": str(audit_path),
                "spec_path": str(spec_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
