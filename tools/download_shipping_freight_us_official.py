#!/usr/bin/env python3
"""Download official U.S. shipping/freight datasets from USDOT Data portal."""

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


# Curated shipping/freight-first list (official USDOT/BTS portal).
# Excludes aviation-heavy and generic non-freight transport tables.
USDOT_SHIPPING_DATASETS: list[DatasetRef] = [
    DatasetRef("y5ut-ibwt", "trade_volume_mix"),
    DatasetRef("kijm-95mr", "cross_border_flow"),
    DatasetRef("7mzw-a8si", "trade_volume_mix"),
    DatasetRef("qq62-cjjy", "trade_volume_mix"),
    DatasetRef("dkgi-gbeh", "claims_damage_risk"),
    DatasetRef("p3bt-a5up", "port_terminal_congestion"),
    DatasetRef("38t4-dnq3", "port_terminal_congestion"),
    DatasetRef("ngjm-b5rq", "port_terminal_congestion"),
    DatasetRef("vc8a-zq94", "port_terminal_congestion"),
    DatasetRef("ub6a-sqr5", "trade_volume_mix"),
    DatasetRef("mjx8-bw4c", "trade_volume_mix"),
    DatasetRef("kett-pmdy", "trade_volume_mix"),
    DatasetRef("n4xv-8upm", "trade_volume_mix"),
    DatasetRef("vpgu-kvxj", "trade_volume_mix"),
    DatasetRef("p7t5-fmvf", "trade_volume_mix"),
    DatasetRef("a5sc-aujx", "trade_volume_mix"),
    DatasetRef("yirh-jy8u", "trade_volume_mix"),
    DatasetRef("5g3r-xnzv", "trade_volume_mix"),
    DatasetRef("pa3c-z6h4", "trade_volume_mix"),
    DatasetRef("ht7p-2x5y", "rail_intermodal_flow"),
    DatasetRef("sn74-xpkp", "port_terminal_congestion"),
    DatasetRef("iahn-a7j4", "port_terminal_congestion"),
    DatasetRef("x6rh-cpwu", "ocean_schedule_reliability"),
    DatasetRef("nfsh-p62e", "port_terminal_congestion"),
    DatasetRef("iiy2-kmkn", "port_terminal_congestion"),
    DatasetRef("abu9-jbyq", "ocean_schedule_reliability"),
    DatasetRef("c7tj-sc2j", "trade_volume_mix"),
    DatasetRef("usuf-55kz", "trade_volume_mix"),
    DatasetRef("c44g-ntqk", "trade_volume_mix"),
    DatasetRef("uta5-4eu5", "eta_delay_risk"),
    DatasetRef("ez58-m3b4", "eta_delay_risk"),
    DatasetRef("d7b8-pmxm", "eta_delay_risk"),
    DatasetRef("mayv-2qfz", "eta_delay_risk"),
    DatasetRef("dggd-bg3y", "eta_delay_risk"),
    DatasetRef("sn4k-eiea", "eta_delay_risk"),
    DatasetRef("xx4g-5dg2", "eta_delay_risk"),
    DatasetRef("v2un-y5se", "carrier_safety_risk"),
    DatasetRef("t23t-ueeq", "carrier_safety_risk"),
    DatasetRef("2ju5-8zxb", "carrier_safety_risk"),
    DatasetRef("dsuf-xcni", "carrier_safety_risk"),
    DatasetRef("8wvp-gjhh", "carrier_safety_risk"),
    DatasetRef("bx7m-yn3v", "carrier_safety_risk"),
    DatasetRef("jm8x-ccxs", "carrier_safety_risk"),
    DatasetRef("uwah-u9bn", "carrier_safety_risk"),
    DatasetRef("b39d-rg8e", "rail_intermodal_flow"),
    DatasetRef("tf5k-fhu2", "cross_border_flow"),
    DatasetRef("u6iw-gzjf", "cross_border_flow"),
    DatasetRef("2a7t-n7sy", "cross_border_flow"),
    DatasetRef("pk4v-772c", "carrier_safety_risk"),
    DatasetRef("i5dw-jvsi", "carrier_safety_risk"),
    DatasetRef("vfi4-ceay", "carrier_safety_risk"),
    DatasetRef("mnwf-3med", "carrier_safety_risk"),
    # Extended official USDOT/BTS rail + motor-carrier + freight economics coverage.
    DatasetRef("icqf-xf4w", "carrier_safety_risk"),
    DatasetRef("byy5-w977", "carrier_safety_risk"),
    DatasetRef("7wn6-i5b9", "carrier_safety_risk"),
    DatasetRef("aqxq-n5hy", "carrier_safety_risk"),
    DatasetRef("kuvg-3uwp", "carrier_safety_risk"),
    DatasetRef("xp92-5xme", "rail_intermodal_flow"),
    DatasetRef("8uv2-y4is", "rail_intermodal_flow"),
    DatasetRef("85tf-25kj", "carrier_safety_risk"),
    DatasetRef("rash-pd2d", "carrier_safety_risk"),
    DatasetRef("unww-uhxd", "carrier_safety_risk"),
    DatasetRef("x5vg-yqea", "rail_intermodal_flow"),
    DatasetRef("hf8z-xt9r", "fleet_utilization"),
    DatasetRef("m2f8-22s6", "rail_intermodal_flow"),
    DatasetRef("vhwz-raag", "rail_intermodal_flow"),
    DatasetRef("65fa-qbkf", "carrier_safety_risk"),
    DatasetRef("m8i6-zdsy", "carrier_safety_risk"),
    DatasetRef("63rf-6igh", "carrier_safety_risk"),
    DatasetRef("aeeh-bp8c", "carrier_safety_risk"),
    DatasetRef("r495-tyji", "trade_volume_mix"),
    DatasetRef("5ti2-5uiv", "trade_volume_mix"),
    DatasetRef("mt5m-skz3", "trucking_capacity"),
    DatasetRef("kjg3-diqy", "carrier_safety_risk"),
    DatasetRef("j5uj-anzx", "lane_cost_yield"),
    DatasetRef("e5cn-ri8q", "lane_cost_yield"),
    DatasetRef("3qgg-2u2a", "lane_cost_yield"),
    DatasetRef("kbvr-tyu5", "lane_cost_yield"),
    DatasetRef("5n49-mh85", "trucking_capacity"),
    DatasetRef("6eyk-hxee", "carrier_safety_risk"),
    DatasetRef("q4tb-tbff", "trade_volume_mix"),
    DatasetRef("bu82-4pwz", "trade_volume_mix"),
    DatasetRef("56rv-9p75", "trade_volume_mix"),
    DatasetRef("udzf-9fvh", "trade_volume_mix"),
    DatasetRef("crem-w557", "trade_volume_mix"),
    DatasetRef("gyti-3rm8", "trucking_capacity"),
    DatasetRef("ivp4-5mkt", "trucking_capacity"),
    DatasetRef("rbkj-cgst", "carrier_safety_risk"),
    DatasetRef("vudg-jaa5", "port_terminal_congestion"),
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


def _update_shipping_metadata(
    base: Path,
    rows: list[dict[str, Any]],
) -> None:
    meta_path = base / "data" / "metadata" / "shipping_freight" / "shipping_freight_datasets.json"
    payload = _read_json(meta_path)
    # Registry expects a flat map: {dataset_id: DatasetMeta}
    if "datasets" in payload and isinstance(payload.get("datasets"), dict):
        datasets = payload.get("datasets") or {}
    else:
        datasets = payload if isinstance(payload, dict) else {}

    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        dataset_id = row["dataset_id"]
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": row.get("title") or dataset_id,
            "source": "usdot_data_transportation_gov",
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


def download_usdot_shipping(
    base: Path,
    limit: int,
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "us" / "usdot"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for ref in USDOT_SHIPPING_DATASETS:
        view_id = ref.view_id
        meta_url = f"https://data.transportation.gov/api/views/{view_id}.json"
        csv_url = f"https://data.transportation.gov/resource/{view_id}.csv?$limit={limit}"
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
                "portal_url": f"https://data.transportation.gov/d/{view_id}",
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
        "source": "usdot_data_transportation_gov",
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
    parser = argparse.ArgumentParser(description="Download official U.S. shipping/freight datasets.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--limit", type=int, default=250000, help="Max rows per dataset for Socrata CSV export")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success_rows, failed_rows = download_usdot_shipping(
        base=base,
        limit=max(1, args.limit),
        timeout_sec=max(30, args.timeout_sec),
    )
    print(
        json.dumps(
            {
                "success_count": len(success_rows),
                "failure_count": len(failed_rows),
                "manifest": "data/raw/shipping_freight/us/usdot/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
