#!/usr/bin/env python3
"""Download trucking/delivery-focused US official datasets from USDOT open data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SEARCH_URL = "https://data.transportation.gov/api/search/views.json"

QUERY_TERMS = [
    "truck",
    "trucking",
    "road",
    "motor carrier",
    "delivery",
    "courier",
    "border crossing",
    "traffic",
    "vehicle",
    "fleet",
    "driver safety",
]

INCLUDE_KEYWORDS = (
    "truck",
    "trucking",
    "road",
    "motor carrier",
    "delivery",
    "courier",
    "last mile",
    "vehicle",
    "fleet",
    "driver",
    "traffic",
    "border crossing",
    "port of entry",
    "travel time",
    "mobility initiative",
    "safety",
    "inspection",
)

# Keep strict exclusions to avoid freight/ocean/rail contamination.
EXCLUDE_KEYWORDS = (
    "ocean",
    "marine",
    "vessel",
    "container",
    "teu",
    "shipping",
    "ship",
    "port",
    "berth",
    "waterway",
    "barge",
    "rail",
    "intermodal",
    "aviation",
    "airport",
    "air cargo",
    "airline",
)

# Explicit truck datasets that contain "freight" naming but are trucking-focused.
ALLOWLIST_VIEW_IDS = {
    "uta5-4eu5",
    "ez58-m3b4",
    "d7b8-pmxm",
    "mayv-2qfz",
    "dggd-bg3y",
    "sn4k-eiea",
    "xx4g-5dg2",
}


@dataclass(frozen=True)
class Candidate:
    view_id: str
    title: str
    description: str
    category: str


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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _line_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _score(text: str, view_id: str) -> int:
    t = text.lower()
    if view_id in ALLOWLIST_VIEW_IDS:
        return 5
    if any(k in t for k in EXCLUDE_KEYWORDS):
        return -10
    return sum(1 for k in INCLUDE_KEYWORDS if k in t)


def _to_candidate(raw: dict[str, Any]) -> Candidate | None:
    view = raw.get("view") or {}
    vid = str(view.get("id") or "").strip()
    if not vid:
        return None
    title = str(view.get("name") or vid)
    desc = str(view.get("description") or "")
    category = str(view.get("category") or "")
    text = f"{title} {desc} {category}".lower()
    score = _score(text, vid)
    if score <= 0:
        return None
    return Candidate(view_id=vid, title=title, description=desc, category=category)


def discover(session: requests.Session, max_candidates: int) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []

    for q in QUERY_TERMS:
        params = {"q": q, "limit": 200}
        resp = session.get(SEARCH_URL, params=params, timeout=45)
        resp.raise_for_status()
        payload = resp.json()
        for row in payload.get("results") or []:
            cand = _to_candidate(row)
            if not cand:
                continue
            if cand.view_id in seen:
                continue
            seen.add(cand.view_id)
            out.append(cand)
            if len(out) >= max_candidates:
                return out
    return out


def _category_hint(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("eta", "delay", "travel time", "on-time", "arrival")):
        return "route_eta_reliability"
    if any(k in t for k in ("dispatch", "capacity", "demand", "throughput")):
        return "dispatch_capacity_balance"
    if any(k in t for k in ("last mile", "delivery", "courier", "parcel", "dropoff", "pickup")):
        return "last_mile_sla"
    if any(k in t for k in ("safety", "inspection", "violation", "incident", "collision")):
        return "driver_safety_compliance"
    if any(k in t for k in ("fleet", "vehicle", "tractor", "van", "utilization")):
        return "fleet_utilization"
    if any(k in t for k in ("maintenance", "repair", "breakdown", "equipment")):
        return "vehicle_maintenance_risk"
    if any(k in t for k in ("fuel", "diesel", "energy", "mpg", "emission")):
        return "fuel_energy_efficiency"
    if any(k in t for k in ("lane", "rate", "cost", "yield", "revenue", "expense")):
        return "lane_cost_yield"
    if any(k in t for k in ("pickup", "dropoff", "stop", "appointment")):
        return "pickup_dropoff_reliability"
    if any(k in t for k in ("exception", "claim", "damage", "loss", "failed")):
        return "parcel_exception_risk"
    if any(k in t for k in ("return", "reverse", "rma", "refund")):
        return "reverse_logistics_returns"
    if any(k in t for k in ("border", "crossing", "port of entry", "transborder")):
        return "cross_border_trucking"
    if any(k in t for k in ("traffic", "congestion", "urban", "camera")):
        return "urban_traffic_risk"
    if any(k in t for k in ("cold", "temperature", "reefer", "chilled")):
        return "cold_chain_last_mile"
    if any(k in t for k in ("shift", "workforce", "driver hours", "staff")):
        return "workforce_shift_planning"
    return "dispatch_capacity_balance"


def download(base: Path, max_datasets: int, max_rows: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "trucking_delivery" / "us" / "usdot"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    discovered = discover(session, max_candidates=max_datasets * 4)

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for cand in discovered:
        if len(success) >= max_datasets:
            break

        view_id = cand.view_id
        csv_url = f"https://data.transportation.gov/resource/{view_id}.csv?$limit={max_rows}"
        meta_url = f"https://data.transportation.gov/api/views/{view_id}.json"

        title = cand.title
        safe = _slug(title)
        file_name = f"usdot__{view_id}__{safe}.csv"
        out_path = out_dir / file_name
        item = {
            "view_id": view_id,
            "title": title,
            "source_url": f"https://data.transportation.gov/d/{view_id}",
            "csv_url": csv_url,
            "downloaded_at": now,
        }

        try:
            meta_resp = session.get(meta_url, timeout=timeout_sec)
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            category_hint = _category_hint(f"{title} {meta.get('description') or ''} {meta.get('category') or ''}")

            if out_path.exists() and out_path.stat().st_size > 0:
                line_count = _line_count(out_path)
            else:
                with session.get(csv_url, timeout=timeout_sec, stream=True) as r:
                    r.raise_for_status()
                    tmp = out_path.with_suffix(".csv.tmp")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    line_count = _line_count(tmp)
                    if line_count <= 1:
                        tmp.unlink(missing_ok=True)
                        raise RuntimeError("empty_or_header_only")
                    tmp.rename(out_path)

            columns = _header(out_path)
            rows = max(0, line_count - 1)
            dataset_id = f"usdot__{view_id}__{safe}"
            success.append({
                **item,
                "dataset_id": dataset_id,
                "category_hint": category_hint,
                "rows": rows,
                "columns": columns,
                "file_path": str(out_path.relative_to(base)),
                "portal_category": meta.get("category"),
                "attribution": meta.get("attribution"),
            })
        except Exception as exc:  # noqa: BLE001
            failed.append({**item, "error": str(exc)})
        time.sleep(0.08)

    return success, failed


def update_metadata(base: Path, rows: list[dict[str, Any]]) -> None:
    meta_path = base / "data" / "metadata" / "trucking_delivery" / "trucking_delivery_datasets.json"
    payload = _read_json(meta_path)
    datasets = payload.get("datasets") if isinstance(payload, dict) and isinstance(payload.get("datasets"), dict) else payload
    if not isinstance(datasets, dict):
        datasets = {}

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        dataset_id = row["dataset_id"]
        cat = row.get("category_hint") or "dispatch_capacity_balance"
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": row.get("title") or dataset_id,
            "source": "usdot_data_transportation_gov",
            "category": cat,
            "categories": [cat],
            "n_rows": int(row.get("rows", 0) or 0),
            "n_cols": len(row.get("columns") or []),
            "columns": row.get("columns", []),
            "target_column": None,
            "task_type": None,
            "file_path": row.get("file_path"),
            "processed": False,
            "ingested_at": row.get("downloaded_at") or now,
            "fingerprint": None,
            "duplicate_of": None,
            "worker_dataset_id": None,
            "source_dataset_id": row.get("view_id"),
            "region": "us",
            "attribution": row.get("attribution"),
            "portal_category": row.get("portal_category"),
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download US trucking/delivery official datasets.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--max-datasets", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=250000)
    parser.add_argument("--timeout-sec", type=int, default=45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download(base, args.max_datasets, args.max_rows, args.timeout_sec)

    update_metadata(base, success)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": "trucking_delivery",
        "source": "usdot",
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "downloaded": success,
        "failed": failed,
    }
    report_path = base / "reports" / "trucking_delivery_us_official_download_report.json"
    _write_json(report_path, out)
    print(json.dumps({
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "report_path": str(report_path.relative_to(base)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
