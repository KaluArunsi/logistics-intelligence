#!/usr/bin/env python3
"""Download official U.S. BTS shipping/freight datasets from data.bts.gov."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class DatasetRef:
    view_id: str
    category_hint: str


# Curated BTS datasets focused on shipping/freight operations.
BTS_SHIPPING_DATASETS: list[DatasetRef] = [
    DatasetRef("keg4-3bc2", "cross_border_flow"),
    DatasetRef("btpt-uxhx", "cross_border_flow"),
    DatasetRef("xnav-e47e", "cross_border_flow"),
    DatasetRef("3jux-kwvh", "cross_border_flow"),
    DatasetRef("75qq-qmj6", "trade_volume_mix"),
    DatasetRef("h7pv-kjj5", "trucking_capacity"),
    DatasetRef("5rpz-kgm9", "port_terminal_congestion"),
    DatasetRef("uxyn-8v2z", "port_terminal_congestion"),
    DatasetRef("25e5-bvnb", "port_terminal_congestion"),
    DatasetRef("dasz-28ip", "inland_waterway_flow"),
    DatasetRef("3z7h-xatu", "port_terminal_congestion"),
    DatasetRef("bi2e-xh9z", "port_terminal_congestion"),
    DatasetRef("f3sb-gw7h", "cold_chain_integrity"),
    DatasetRef("j246-y2rf", "claims_damage_risk"),
    DatasetRef("gbe2-48iq", "rail_intermodal_flow"),
    DatasetRef("pydt-j7kj", "ocean_schedule_reliability"),
    DatasetRef("w6v2-vk5z", "ocean_schedule_reliability"),
    DatasetRef("5kxy-j6hw", "ocean_schedule_reliability"),
    DatasetRef("xkuc-f3hj", "ocean_schedule_reliability"),
    DatasetRef("d2st-9nd6", "ocean_schedule_reliability"),
    DatasetRef("ca7h-i9yt", "port_terminal_congestion"),
    DatasetRef("ke6h-ga46", "port_terminal_congestion"),
    DatasetRef("bw6n-ddqk", "fleet_utilization"),
    DatasetRef("r8cc-5x95", "lane_cost_yield"),
]


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
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


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
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": row.get("title") or dataset_id,
            "source": "bts_data_bts_gov",
            "category": row.get("category_hint") or "trade_volume_mix",
            "categories": [row.get("category_hint") or "trade_volume_mix"],
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
            "portal_url": row.get("portal_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def download_us_bts(base: Path, limit: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "us" / "bts"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    now = datetime.now(timezone.utc).isoformat()

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for ref in BTS_SHIPPING_DATASETS:
        view_id = ref.view_id
        meta_url = f"https://data.bts.gov/api/views/{view_id}.json"
        csv_url = f"https://data.bts.gov/resource/{view_id}.csv?$limit={limit}"
        item = {
            "view_id": view_id,
            "category_hint": ref.category_hint,
            "csv_url": csv_url,
            "downloaded_at": now,
        }
        try:
            meta_resp = session.get(meta_url, timeout=timeout_sec)
            meta_resp.raise_for_status()
            meta = meta_resp.json()

            title = str(meta.get("name") or view_id)
            safe = _slug(title)
            file_name = f"bts__{view_id}__{safe}.csv"
            out_path = out_dir / file_name

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
            row_count = max(0, line_count - 1)
            dataset_id = f"bts__{view_id}__{safe}"
            row = {
                **item,
                "dataset_id": dataset_id,
                "title": title,
                "attribution": meta.get("attribution"),
                "portal_category": meta.get("category"),
                "portal_url": f"https://data.bts.gov/d/{view_id}",
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": row_count,
                "columns": columns,
            }
            success_rows.append(row)
            print(f"[OK] {view_id} rows={row_count} file={file_name}")
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
            failed_rows.append(item)
            print(f"[FAIL] {view_id} error={exc}")

    manifest = {
        "industry": "shipping_freight",
        "region": "us",
        "source": "bts_data_bts_gov",
        "generated_at": now,
        "limit": limit,
        "timeout_sec": timeout_sec,
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official BTS shipping/freight datasets.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--limit", type=int, default=250000, help="Maximum rows to request per dataset")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_us_bts(base=base, limit=max(1000, args.limit), timeout_sec=max(20, args.timeout_sec))
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "source": "bts_data_bts_gov",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/us/bts/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

