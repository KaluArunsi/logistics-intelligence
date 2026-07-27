#!/usr/bin/env python3
"""Download curated US trucking/delivery datasets by known USDOT view IDs."""

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


USDOT_TRUCKING_DATASETS: list[DatasetRef] = [
    DatasetRef("v2un-y5se", "driver_safety_compliance"),
    DatasetRef("t23t-ueeq", "driver_safety_compliance"),
    DatasetRef("2ju5-8zxb", "driver_safety_compliance"),
    DatasetRef("dsuf-xcni", "driver_safety_compliance"),
    DatasetRef("pk4v-772c", "driver_safety_compliance"),
    DatasetRef("i5dw-jvsi", "driver_safety_compliance"),
    DatasetRef("vfi4-ceay", "driver_safety_compliance"),
    DatasetRef("mnwf-3med", "driver_safety_compliance"),
    DatasetRef("85tf-25kj", "driver_safety_compliance"),
    DatasetRef("rash-pd2d", "driver_safety_compliance"),
    DatasetRef("unww-uhxd", "driver_safety_compliance"),
    DatasetRef("65fa-qbkf", "driver_safety_compliance"),
    DatasetRef("m8i6-zdsy", "driver_safety_compliance"),
    DatasetRef("63rf-6igh", "driver_safety_compliance"),
    DatasetRef("aeeh-bp8c", "driver_safety_compliance"),
    DatasetRef("kjg3-diqy", "driver_safety_compliance"),
    DatasetRef("6eyk-hxee", "driver_safety_compliance"),
    DatasetRef("rbkj-cgst", "driver_safety_compliance"),
    DatasetRef("mt5m-skz3", "dispatch_capacity_balance"),
    DatasetRef("5n49-mh85", "dispatch_capacity_balance"),
    DatasetRef("gyti-3rm8", "dispatch_capacity_balance"),
    DatasetRef("ivp4-5mkt", "dispatch_capacity_balance"),
    DatasetRef("hf8z-xt9r", "fleet_utilization"),
    DatasetRef("j5uj-anzx", "lane_cost_yield"),
    DatasetRef("e5cn-ri8q", "lane_cost_yield"),
    DatasetRef("3qgg-2u2a", "lane_cost_yield"),
    DatasetRef("kbvr-tyu5", "lane_cost_yield"),
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def download(base: Path, max_rows: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "trucking_delivery" / "us" / "usdot"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    now = datetime.now(timezone.utc).isoformat()

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for ref in USDOT_TRUCKING_DATASETS:
        view_id = ref.view_id
        meta_url = f"https://data.transportation.gov/api/views/{view_id}.json"
        csv_url = f"https://data.transportation.gov/resource/{view_id}.csv?$limit={max_rows}"
        item: dict[str, Any] = {
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
            file_name = f"usdot__{view_id}__{safe}.csv"
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
            dataset_id = f"usdot__{view_id}__{safe}"
            row = {
                **item,
                "dataset_id": dataset_id,
                "title": title,
                "attribution": meta.get("attribution"),
                "portal_category": meta.get("category"),
                "source_url": f"https://data.transportation.gov/d/{view_id}",
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": row_count,
                "columns": columns,
            }
            success.append(row)
        except Exception as exc:  # noqa: BLE001
            failed.append({**item, "error": str(exc)})

    return success, failed


def update_metadata(base: Path, rows: list[dict[str, Any]]) -> None:
    meta_path = base / "data" / "metadata" / "trucking_delivery" / "trucking_delivery_datasets.json"
    payload = _read_json(meta_path)
    if "datasets" in payload and isinstance(payload.get("datasets"), dict):
        datasets = payload.get("datasets") or {}
    else:
        datasets = payload if isinstance(payload, dict) else {}

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
    parser = argparse.ArgumentParser(description="Download curated US trucking/delivery datasets.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--timeout-sec", type=int, default=35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()

    success, failed = download(base=base, max_rows=args.max_rows, timeout_sec=args.timeout_sec)
    update_metadata(base, success)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": "trucking_delivery",
        "source": "usdot_curated",
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "downloaded": success,
        "failed": failed,
    }
    report_path = base / "reports" / "trucking_delivery_us_curated_download_report.json"
    _write_json(report_path, report)

    print(json.dumps(
        {
            "downloaded_count": len(success),
            "failed_count": len(failed),
            "report_path": str(report_path.relative_to(base)),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
