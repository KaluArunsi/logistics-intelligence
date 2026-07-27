#!/usr/bin/env python3
"""Download 'other region' shipping/freight proxy datasets from World Bank indicators."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


WORLD_BANK_INDICATORS = [
    "LP.LPI.OVRL.XQ",
    "LP.LPI.CUST.XQ",
    "LP.LPI.INFR.XQ",
    "LP.LPI.ITRN.XQ",
    "LP.LPI.LOGS.XQ",
    "LP.LPI.TIME.XQ",
    "LP.LPI.TRAC.XQ",
    "IS.SHP.GCNW.XQ",
    "IS.SHP.GOOD.TU",
    "TX.VAL.MRCH.XD.WD",
    "TM.VAL.MRCH.XD.WD",
    "TG.VAL.TOTL.GD.ZS",
    "TX.VAL.TECH.CD",
    "BX.GSR.CCIS.CD",
    "IC.FRM.BRIB.ZS",
    "IC.CUS.DURS.EX",
]

WORLD_BANK_URL = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"


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


def _category_hint(indicator: str) -> str:
    if indicator.startswith("LP.LPI.CUST") or indicator.startswith("IC.CUS"):
        return "customs_compliance"
    if indicator.startswith("LP.LPI.TIME") or indicator.startswith("IC.FRM"):
        return "eta_delay_risk"
    if indicator.startswith("IS.SHP.GCNW"):
        return "ocean_schedule_reliability"
    if indicator.startswith("IS.SHP.GOOD"):
        return "trade_volume_mix"
    if indicator.startswith("LP.LPI."):
        return "lane_cost_yield"
    if indicator.startswith("TX.") or indicator.startswith("TM.") or indicator.startswith("TG."):
        return "trade_volume_mix"
    return "trade_volume_mix"


def _download_indicator(session: requests.Session, indicator: str) -> list[dict[str, Any]]:
    page = 1
    out: list[dict[str, Any]] = []
    while True:
        params = {"format": "json", "per_page": 20000, "page": page}
        payload = session.get(WORLD_BANK_URL.format(indicator=indicator), params=params, timeout=60).json()
        if not isinstance(payload, list) or len(payload) < 2:
            break
        meta = payload[0] or {}
        data = payload[1] or []
        if not data:
            break
        for row in data:
            out.append(
                {
                    "indicator_id": indicator,
                    "indicator_name": ((row.get("indicator") or {}).get("value") or ""),
                    "country_code": ((row.get("country") or {}).get("id") or ""),
                    "country_name": ((row.get("country") or {}).get("value") or ""),
                    "year": row.get("date"),
                    "value": row.get("value"),
                    "unit": row.get("unit"),
                    "obs_status": row.get("obs_status"),
                    "decimal": row.get("decimal"),
                }
            )
        if page >= int(meta.get("pages", 1) or 1):
            break
        page += 1
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
            "source": "world_bank_indicator_api",
            "category": category,
            "categories": [category],
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
            "source_dataset_id": row.get("indicator"),
            "region": "other",
            "attribution": "World Bank Open Data",
            "portal_category": "Indicators",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def download_worldbank(base: Path, max_datasets: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "other" / "worldbank"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    session = requests.Session()
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for indicator in WORLD_BANK_INDICATORS[:max_datasets]:
        item = {"indicator": indicator, "downloaded_at": now}
        try:
            rows = _download_indicator(session, indicator)
            if not rows:
                raise RuntimeError("no_rows")
            file_name = f"worldbank__{indicator.lower().replace('.', '_')}.csv"
            out_path = out_dir / file_name
            _write_csv(out_path, rows)
            row_count = max(0, _line_count(out_path) - 1)
            columns = _header(out_path)
            entry = {
                **item,
                "dataset_id": f"worldbank__{indicator.lower().replace('.', '_')}",
                "title": indicator,
                "source_url": WORLD_BANK_URL.format(indicator=indicator),
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": row_count,
                "columns": columns,
                "category_hint": _category_hint(indicator),
            }
            success_rows.append(entry)
            print(f"[OK] {indicator} rows={row_count}")
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
            failed_rows.append(item)
            print(f"[FAIL] {indicator} error={exc}")

    manifest = {
        "industry": "shipping_freight",
        "region": "other",
        "source": "world_bank_indicator_api",
        "generated_at": now,
        "requested_datasets": max_datasets,
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download 'other region' datasets from World Bank indicators.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--max-datasets", type=int, default=16, help="Maximum indicator datasets to download")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_worldbank(
        base=base,
        max_datasets=max(1, int(args.max_datasets)),
    )
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "region": "other",
                "source": "world_bank_indicator_api",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/other/worldbank/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
