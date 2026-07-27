#!/usr/bin/env python3
"""Download OECD transport datasets and split into Asia/Europe/Other regional raw files."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


OECD_BASE = "https://sdmx.oecd.org/public/rest/data"

FLOW_CATEGORY = {
    "DSD_ST@DF_STFREIGHT": "trade_volume_mix",
    "DSD_ST@DF_STTRAFFIC": "trucking_capacity",
    "DSD_ST@DF_STINDICATORS": "lane_cost_yield",
    "DSD_TRENDS@DF_TRENDSFREIGHT": "trade_volume_mix",
    "DSD_TRENDS@DF_TRENDSCONT": "port_terminal_congestion",
    "DSD_TRENDS@DF_TRENDSSAFETY": "carrier_safety_risk",
    "DSD_INFRINV@DF_INFRINV": "fleet_utilization",
    "DSD_INDICATORS@DF_SAFETY": "carrier_safety_risk",
    "DSD_INDICATORS@DF_INFRASTRUCTURE": "inland_waterway_flow",
    "DSD_INDICATORS@DF_TRANSPORTINDICATORS": "eta_delay_risk",
    "DSD_INDICATORS@DF_MEASUREMENT": "last_mile_sla",
    "DSD_INDICATORS@DF_EQUIPMENT": "fleet_utilization",
}

# OECD/ISO3 region buckets for split output.
ASIA_CODES = {
    "AFG", "ARM", "AZE", "BHR", "BGD", "BRN", "BTN", "CHN", "CYP", "GEO", "HKG", "IDN", "IND", "IRN", "IRQ", "ISR",
    "JOR", "JPN", "KAZ", "KGZ", "KHM", "KOR", "KWT", "LAO", "LBN", "LKA", "MAC", "MMR", "MNG", "MYS", "NPL", "OMN",
    "PAK", "PHL", "PSE", "QAT", "SAU", "SGP", "SYR", "THA", "TJK", "TKM", "TLS", "TUR", "TWN", "UZB", "VNM", "YEM",
}

EUROPE_CODES = {
    "ALB", "AND", "AUT", "BEL", "BGR", "BIH", "BLR", "CHE", "CZE", "DEU", "DNK", "ESP", "EST", "FIN", "FRA", "GBR",
    "GRC", "HRV", "HUN", "IRL", "ISL", "ITA", "LTU", "LUX", "LVA", "MDA", "MKD", "MLT", "MNE", "NLD", "NOR", "POL",
    "PRT", "ROU", "RUS", "SRB", "SVK", "SVN", "SWE", "UKR", "XKX",
}


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


def _line_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _split_region(ref_area: str) -> str:
    code = (ref_area or "").strip().upper()
    if code in ASIA_CODES:
        return "asia"
    if code in EUROPE_CODES:
        return "europe"
    return "other"


def _download_flow_csv(session: requests.Session, flow_id: str, start_period: int, timeout_sec: int, out_tmp: Path) -> None:
    url = f"{OECD_BASE}/{flow_id}/"
    params = {"startPeriod": start_period, "format": "csvfile"}
    with session.get(url, params=params, timeout=timeout_sec, stream=True) as resp:
        resp.raise_for_status()
        with open(out_tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _split_flow_file(base: Path, flow_id: str, tmp_csv: Path, category_hint: str) -> list[dict[str, Any]]:
    flow_slug = _slug(flow_id).replace("dsd_", "")
    regions = ("asia", "europe", "other")
    writers: dict[str, csv.DictWriter] = {}
    files: dict[str, Any] = {}
    counts = {r: 0 for r in regions}
    outputs: list[dict[str, Any]] = []

    with open(tmp_csv, "r", encoding="utf-8", errors="ignore", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            return []
        if "REF_AREA" not in reader.fieldnames:
            return []
        fields = list(reader.fieldnames)

        for region in regions:
            out_dir = base / "data" / "raw" / "shipping_freight" / region / "oecd"
            out_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"oecd__{flow_slug}__{region}.csv"
            out_path = out_dir / file_name
            f = open(out_path, "w", encoding="utf-8", newline="")
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            writers[region] = w
            files[region] = (f, out_path, file_name)

        for row in reader:
            region = _split_region(str(row.get("REF_AREA") or ""))
            writers[region].writerow(row)
            counts[region] += 1

    for region in regions:
        f, out_path, file_name = files[region]
        f.close()
        if counts[region] <= 0:
            out_path.unlink(missing_ok=True)
            continue
        dataset_id = f"oecd__{flow_slug}__{region}"
        outputs.append(
            {
                "dataset_id": dataset_id,
                "title": flow_id,
                "flow_id": flow_id,
                "region": region,
                "category_hint": category_hint,
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": counts[region],
                "columns": _header(out_path),
            }
        )
    return outputs


def _update_shipping_metadata(base: Path, rows: list[dict[str, Any]], generated_at: str) -> None:
    meta_path = base / "data" / "metadata" / "shipping_freight" / "shipping_freight_datasets.json"
    payload = _read_json(meta_path)
    if "datasets" in payload and isinstance(payload.get("datasets"), dict):
        datasets = payload["datasets"]
    else:
        datasets = payload if isinstance(payload, dict) else {}

    for row in rows:
        dataset_id = row["dataset_id"]
        category = row.get("category_hint") or "trade_volume_mix"
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": row.get("title") or dataset_id,
            "source": "oecd_sdmx_transport",
            "category": category,
            "categories": [category],
            "n_rows": int(row.get("rows", 0) or 0),
            "n_cols": len(row.get("columns") or []),
            "columns": row.get("columns", []),
            "target_column": None,
            "task_type": None,
            "file_path": row.get("file_path"),
            "processed": False,
            "ingested_at": generated_at,
            "fingerprint": None,
            "duplicate_of": None,
            "worker_dataset_id": None,
            "source_dataset_id": row.get("flow_id"),
            "region": row.get("region"),
            "attribution": "OECD/ITF",
            "portal_category": "Transport Statistics",
            "portal_url": f"https://sdmx.oecd.org/public/rest/data/{row.get('flow_id')}/",
            "updated_at": generated_at,
        }

    _write_json(meta_path, datasets)


def download_oecd_multiregion(
    base: Path,
    start_period: int,
    timeout_sec: int,
    max_flows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    now = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    selected_flows = list(FLOW_CATEGORY.items())[: max(1, max_flows)]
    tmp_dir = base / "data" / "raw" / "shipping_freight" / "_tmp_oecd"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for flow_id, category_hint in selected_flows:
        tmp_csv = tmp_dir / f"{_slug(flow_id)}.csv"
        try:
            _download_flow_csv(
                session=session,
                flow_id=flow_id,
                start_period=start_period,
                timeout_sec=timeout_sec,
                out_tmp=tmp_csv,
            )
            if _line_count(tmp_csv) <= 1:
                raise RuntimeError("empty_or_header_only")
            rows = _split_flow_file(base=base, flow_id=flow_id, tmp_csv=tmp_csv, category_hint=category_hint)
            if not rows:
                raise RuntimeError("split_failed_or_no_ref_area")
            for row in rows:
                row["downloaded_at"] = now
                row["source_url"] = f"{OECD_BASE}/{flow_id}/"
                success_rows.append(row)
            print(f"[OK] {flow_id} -> {len(rows)} regional datasets")
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"flow_id": flow_id, "error": str(exc), "downloaded_at": now})
            print(f"[FAIL] {flow_id} error={exc}")
        finally:
            tmp_csv.unlink(missing_ok=True)

    _update_shipping_metadata(base, success_rows, generated_at=now)
    manifest = {
        "industry": "shipping_freight",
        "source": "oecd_sdmx_transport",
        "generated_at": now,
        "start_period": start_period,
        "max_flows": max_flows,
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    manifest_path = base / "data" / "raw" / "shipping_freight" / "other" / "oecd" / "_download_manifest.json"
    _write_json(manifest_path, manifest)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OECD transport datasets and split to Asia/Europe/Other.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--start-period", type=int, default=2015, help="Start period for OECD query")
    parser.add_argument("--timeout-sec", type=int, default=240, help="HTTP timeout in seconds")
    parser.add_argument("--max-flows", type=int, default=12, help="Maximum number of OECD flows to download")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_oecd_multiregion(
        base=base,
        start_period=max(1990, int(args.start_period)),
        timeout_sec=max(30, int(args.timeout_sec)),
        max_flows=max(1, int(args.max_flows)),
    )
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "source": "oecd_sdmx_transport",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/other/oecd/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
