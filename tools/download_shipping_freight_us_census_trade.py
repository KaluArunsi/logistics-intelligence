#!/usr/bin/env python3
"""Download official U.S. Census international trade datasets for shipping_freight."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def _slug(value: str) -> str:
    out = []
    prev_sep = False
    for ch in value.lower().strip():
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        else:
            if not prev_sep:
                out.append("_")
                prev_sep = True
    return "".join(out).strip("_") or "dataset"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _choose_category_hint(path_bits: list[str]) -> str:
    key = "/".join(path_bits).lower()
    if "port" in key:
        return "port_terminal_congestion"
    if "state" in key:
        return "cross_border_flow"
    return "trade_volume_mix"


def _candidate_fields(variables: dict[str, Any]) -> list[str]:
    priority = [
        "CTY_CODE",
        "CTY_NAME",
        "DISTRICT",
        "DIST_NAME",
        "STATE",
        "E_ENDUSE",
        "E_ENDUSE_SDESC",
        "E_ENDUSE_LDESC",
        "I_ENDUSE",
        "I_ENDUSE_SDESC",
        "I_ENDUSE_LDESC",
        "E_COMMODITY",
        "E_COMMODITY_SDESC",
        "E_COMMODITY_LDESC",
        "I_COMMODITY",
        "I_COMMODITY_SDESC",
        "I_COMMODITY_LDESC",
        "NAICS",
        "SITC",
        "USDA",
        "PORT",
        "SUMMARY_LVL",
        "COMM_LVL",
    ]
    chosen: list[str] = [k for k in priority if k in variables]
    metric_vars = [
        k
        for k in variables.keys()
        if (("_VAL_" in k) or ("_WGT_" in k))
        and not k.startswith("LAST_")
    ]
    metric_vars = sorted(metric_vars)
    for k in metric_vars:
        if k not in chosen:
            chosen.append(k)
    for k in ("MONTH", "YEAR"):
        if k in variables and k not in chosen:
            chosen.append(k)
    # Keep URLs manageable while retaining dimensional richness.
    return chosen[:12]


def _update_shipping_metadata(base: Path, rows: list[dict[str, Any]]) -> None:
    meta_path = base / "data" / "metadata" / "shipping_freight" / "shipping_freight_datasets.json"
    payload = _read_json(meta_path)
    if "datasets" in payload and isinstance(payload.get("datasets"), dict):
        datasets = payload["datasets"]
    else:
        datasets = payload if isinstance(payload, dict) else {}

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        dataset_id = row["dataset_id"]
        category = row.get("category_hint") or "trade_volume_mix"
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": row.get("title") or dataset_id,
            "source": "us_census_intltrade_api",
            "category": category,
            "n_rows": int(row.get("rows", 0) or 0),
            "n_cols": len(row.get("columns") or []),
            "columns": row.get("columns", []),
            "categories": [category],
            "target_column": None,
            "task_type": None,
            "file_path": row.get("file_path"),
            "processed": False,
            "ingested_at": row.get("downloaded_at") or now,
            "fingerprint": None,
            "duplicate_of": None,
            "worker_dataset_id": None,
            "source_dataset_id": row.get("endpoint"),
            "region": "us",
            "attribution": "U.S. Census Bureau",
            "portal_category": "International Trade",
            "portal_url": row.get("api_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _load_catalog_endpoints(session: requests.Session, catalog_url: str, timeout_sec: int) -> list[str]:
    r = session.get(catalog_url, timeout=timeout_sec)
    r.raise_for_status()
    payload = r.json()
    out: list[str] = []
    for d in payload.get("dataset", []):
        parts = d.get("c_dataset") or []
        if not parts:
            continue
        out.append("/".join(parts))
    return out


def download_census_trade(base: Path, year: int, month: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "us" / "census_intltrade"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    now = datetime.now(timezone.utc).isoformat()

    catalogs = [
        "https://api.census.gov/data/timeseries/intltrade/exports/",
        "https://api.census.gov/data/timeseries/intltrade/imports/",
    ]
    endpoints: list[str] = []
    for c in catalogs:
        endpoints.extend(_load_catalog_endpoints(session, c, timeout_sec))
    endpoints = sorted(set(endpoints))

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for endpoint in endpoints:
        variables_url = f"https://api.census.gov/data/{endpoint}/variables.json"
        api_url = f"https://api.census.gov/data/{endpoint}"
        item = {
            "endpoint": endpoint,
            "api_url": api_url,
            "downloaded_at": now,
        }
        try:
            vr = session.get(variables_url, timeout=timeout_sec)
            vr.raise_for_status()
            variables = (vr.json() or {}).get("variables", {}) or {}
            fields = _candidate_fields(variables)
            if not fields:
                raise RuntimeError("no_fields")

            params = {
                "get": ",".join(fields),
                "YEAR": str(year),
                "MONTH": f"{month:02d}",
            }
            dr = session.get(api_url, params=params, timeout=timeout_sec)
            dr.raise_for_status()
            arr = dr.json()
            if not isinstance(arr, list) or len(arr) <= 1:
                raise RuntimeError("empty_result")
            header = [str(x) for x in arr[0]]
            records = [list(map(str, row)) for row in arr[1:]]
            if not records:
                raise RuntimeError("no_rows")

            short = _slug(endpoint.split("/")[-1])
            file_name = f"census__{_slug(endpoint)}__{year}_{month:02d}.csv"
            out_path = out_dir / file_name
            _write_csv(out_path, header, records)

            dataset_id = f"census__{_slug(endpoint)}__{year}_{month:02d}"
            category_hint = _choose_category_hint(endpoint.split("/"))
            row = {
                **item,
                "dataset_id": dataset_id,
                "title": f"Census {short} {year}-{month:02d}",
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": len(records),
                "columns": header,
                "category_hint": category_hint,
            }
            success_rows.append(row)
            print(f"[OK] {endpoint} rows={len(records)}")
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
            failed_rows.append(item)
            print(f"[FAIL] {endpoint} error={exc}")

    manifest = {
        "industry": "shipping_freight",
        "region": "us",
        "source": "us_census_intltrade_api",
        "generated_at": now,
        "year": year,
        "month": month,
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download U.S. Census intltrade datasets for shipping_freight.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--year", type=int, default=2024, help="Calendar year filter")
    parser.add_argument("--month", type=int, default=12, help="Calendar month filter (1-12)")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    year = max(2000, int(args.year))
    month = min(12, max(1, int(args.month)))
    success_rows, failed_rows = download_census_trade(
        base=base,
        year=year,
        month=month,
        timeout_sec=max(30, int(args.timeout_sec)),
    )
    print(
        json.dumps(
            {
                "success_count": len(success_rows),
                "failure_count": len(failed_rows),
                "manifest": "data/raw/shipping_freight/us/census_intltrade/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
