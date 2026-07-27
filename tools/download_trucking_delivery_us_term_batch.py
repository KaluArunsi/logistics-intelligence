#!/usr/bin/env python3
"""Batch download US trucking/delivery datasets from USDOT by term with small row caps."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SEARCH_URL = "https://data.transportation.gov/api/search/views.json"

TERMS = [
    "truck",
    "trucking",
    "motor carrier",
    "fmcsa",
    "csa",
    "sms inspection",
    "truck crash",
    "faf",
    "commodity flow survey",
    "delivery",
    "courier",
    "driver",
    "fleet",
    "traffic",
    "road",
    "highway",
    "border crossing",
]

INCLUDE = (
    "truck", "trucking", "motor carrier", "delivery", "courier", "driver", "fleet", "vehicle",
    "road", "highway", "traffic", "border", "crossing", "travel time", "mobility", "dispatch",
    "fmcsa", "csa", "safety", "inspection", "crash", "fatality", "commodity flow", "faf", "freight analysis framework",
)

EXCLUDE = (
    "ocean", "marine", "vessel", "container", "teu", "shipping", "ship", "port", "berth",
    "waterway", "barge", "aviation", "airport", "airline", "air cargo",
)


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


def _line_count(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _header(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _category_hint(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("eta", "delay", "travel time", "on-time")):
        return "route_eta_reliability"
    if any(k in t for k in ("dispatch", "capacity", "load", "assignment")):
        return "dispatch_capacity_balance"
    if any(k in t for k in ("delivery", "courier", "last mile", "parcel")):
        return "last_mile_sla"
    if any(k in t for k in ("safety", "inspection", "violation", "incident", "collision")):
        return "driver_safety_compliance"
    if any(k in t for k in ("fleet", "vehicle", "tractor", "van")):
        return "fleet_utilization"
    if any(k in t for k in ("maintenance", "repair", "fault")):
        return "vehicle_maintenance_risk"
    if any(k in t for k in ("fuel", "diesel", "energy", "mpg")):
        return "fuel_energy_efficiency"
    if any(k in t for k in ("cost", "rate", "yield", "margin", "lane")):
        return "lane_cost_yield"
    if any(k in t for k in ("pickup", "dropoff", "stop", "appointment")):
        return "pickup_dropoff_reliability"
    if any(k in t for k in ("claim", "damage", "loss", "exception")):
        return "parcel_exception_risk"
    if any(k in t for k in ("return", "reverse", "refund")):
        return "reverse_logistics_returns"
    if any(k in t for k in ("border", "crossing", "transborder")):
        return "cross_border_trucking"
    if any(k in t for k in ("traffic", "congestion", "urban", "camera")):
        return "urban_traffic_risk"
    if any(k in t for k in ("cold", "temperature", "reefer")):
        return "cold_chain_last_mile"
    if any(k in t for k in ("workforce", "shift", "hours of service", "staff")):
        return "workforce_shift_planning"
    return "dispatch_capacity_balance"


def _is_candidate(text: str) -> bool:
    low = text.lower()
    if any(k in low for k in EXCLUDE):
        return False
    return any(k in low for k in INCLUDE)


def discover_ids(session: requests.Session, per_term: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for term in TERMS:
        resp = session.get(SEARCH_URL, params={"q": term, "limit": 200}, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        picked = 0
        for row in payload.get("results") or []:
            view = row.get("view") or {}
            vid = str(view.get("id") or "")
            if not vid or vid in seen:
                continue
            title = str(view.get("name") or vid)
            desc = str(view.get("description") or "")
            text = f"{title} {desc}"
            if not _is_candidate(text):
                continue
            seen.add(vid)
            out.append((vid, title, desc))
            picked += 1
            if picked >= per_term:
                break
    return out


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


def run(base: Path, per_term: int, max_rows: int, timeout_sec: int, max_new: int) -> dict[str, Any]:
    out_dir = base / "data" / "raw" / "trucking_delivery" / "us" / "usdot"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    candidates = discover_ids(session, per_term=per_term)
    now = datetime.now(timezone.utc).isoformat()

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    added = 0

    for view_id, title_hint, desc_hint in candidates:
        if added >= max_new:
            break

        meta_url = f"https://data.transportation.gov/api/views/{view_id}.json"
        csv_url = f"https://data.transportation.gov/resource/{view_id}.csv?$limit={max_rows}"
        item = {
            "view_id": view_id,
            "source_url": f"https://data.transportation.gov/d/{view_id}",
            "csv_url": csv_url,
            "downloaded_at": now,
        }
        try:
            meta_resp = session.get(meta_url, timeout=(8, timeout_sec))
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            title = str(meta.get("name") or title_hint or view_id)
            safe = _slug(title)
            file_name = f"usdot__{view_id}__{safe}.csv"
            out_path = out_dir / file_name

            if out_path.exists() and out_path.stat().st_size > 0:
                line_count = _line_count(out_path)
            else:
                with session.get(csv_url, timeout=(8, timeout_sec), stream=True) as r:
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
            category_hint = _category_hint(f"{title} {meta.get('description') or desc_hint}")
            dataset_id = f"usdot__{view_id}__{safe}"
            success.append({
                **item,
                "dataset_id": dataset_id,
                "title": title,
                "category_hint": category_hint,
                "rows": rows,
                "columns": columns,
                "file_path": str(out_path.relative_to(base)),
                "portal_category": meta.get("category"),
                "attribution": meta.get("attribution"),
            })
            added += 1
            print(f"[ok] {added:03d}/{max_new} {view_id} -> {category_hint} ({rows} rows)")
        except Exception as exc:  # noqa: BLE001
            failed.append({**item, "error": str(exc)})
            print(f"[skip] {view_id}: {exc}")
        time.sleep(0.05)

    update_metadata(base, success)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": "trucking_delivery",
        "source": "usdot_term_batch",
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "downloaded": success,
        "failed": failed,
    }
    report_path = base / "reports" / "trucking_delivery_us_term_batch_report.json"
    _write_json(report_path, report)
    return {
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "report_path": str(report_path.relative_to(base)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USDOT term-batch trucking downloader")
    p.add_argument("--base", default=".")
    p.add_argument("--per-term", type=int, default=12)
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--timeout-sec", type=int, default=20)
    p.add_argument("--max-new", type=int, default=80)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    out = run(base, per_term=args.per_term, max_rows=args.max_rows, timeout_sec=args.timeout_sec, max_new=args.max_new)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
