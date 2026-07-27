#!/usr/bin/env python3
"""Download Europe shipping/freight datasets from Eurostat official APIs."""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


EUROSTAT_INVENTORY_URL = "https://ec.europa.eu/eurostat/api/dissemination/files/inventory?type=data"

PREFIX_CATEGORY = {
    "ROAD_GO_": "trucking_capacity",
    "RAIL_GO_": "rail_intermodal_flow",
    "IWW_GO_": "inland_waterway_flow",
    "MAR_GO_": "ocean_schedule_reliability",
    "TRAN_HV_": "trade_volume_mix",
}

SAFE_EUROSTAT_CODES = [
    "MAR_GO_AA",
    "MAR_GO_QM",
    "MAR_PA_AA",
    "MAR_PA_QM",
    "IWW_GO_ACSIZE",
    "IWW_GO_ACTYGO",
    "IWW_GO_ADAGO",
    "IWW_GO_ANAVE",
    "IWW_GO_APORT",
    "IWW_GO_ATYVE",
    "ROAD_GO_CA_C",
    "ROAD_GO_CA_HAC",
    "ROAD_GO_IA_LGTT",
    "ROAD_GO_IA_LTT",
    "ROAD_GO_IA_RC",
    "ROAD_GO_IA_UGTT",
    "ROAD_GO_IA_UTT",
    "ROAD_GO_NA_DCTG",
    "ROAD_GO_NA_DCTT",
    "RAIL_GO_CONSGMT",
    "RAIL_GO_CONTNBR",
    "RAIL_GO_CONTWGT",
    "RAIL_GO_DNGGOOD",
    "RAIL_GO_GRPGOOD",
    "RAIL_GO_INTCMGN",
    "RAIL_GO_INTGONG",
    "RAIL_GO_ITU",
    "RAIL_GO_QUARTAL",
    "RAIL_GO_TOTAL",
    "TRAN_HV_FRMOD",
    "TRAN_HV_FRTRA",
    "TRAN_HV_MS_FRMOD",
    "TRAN_HV_MS_FRMOD6",
    "TRAN_HV_PSMOD",
    "TRAN_HV_PSTRA",
]


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


def _category_for_code(code: str) -> str:
    code_u = code.upper()
    for prefix, category in PREFIX_CATEGORY.items():
        if code_u.startswith(prefix):
            return category
    return "trade_volume_mix"


def _select_codes(max_datasets: int) -> list[dict[str, str]]:
    text = requests.get(EUROSTAT_INVENTORY_URL, timeout=90).text
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    selected_by_code: dict[str, dict[str, str]] = {}
    for row in rows:
        code = str(row.get("Code") or "").strip()
        if not code:
            continue
        if not any(code.startswith(prefix) for prefix in PREFIX_CATEGORY):
            continue
        csv_url = str(row.get("Data download url (csv)") or "").strip()
        if not csv_url:
            continue
        selected_by_code[code] = {"code": code, "csv_url": csv_url}

    selected: list[dict[str, str]] = []
    for code in SAFE_EUROSTAT_CODES:
        if code in selected_by_code:
            selected.append(selected_by_code[code])
        if len(selected) >= max_datasets:
            break

    if len(selected) < max_datasets:
        fallback = sorted(
            [v for k, v in selected_by_code.items() if k not in {s["code"] for s in selected}],
            key=lambda r: r["code"],
        )
        selected.extend(fallback[: max(0, max_datasets - len(selected))])

    return selected[:max_datasets]


def _download_csv(session: requests.Session, url: str, out_path: Path, timeout_sec: int) -> int:
    if "startPeriod=" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}startPeriod=2015"
    with session.get(url, timeout=timeout_sec, stream=True) as r:
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
        return line_count


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
            "source": "eurostat_sdmx",
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
            "source_dataset_id": row.get("code"),
            "region": "europe",
            "attribution": "Eurostat",
            "portal_category": "Transport Statistics",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def download_eurostat(base: Path, max_datasets: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "europe" / "eurostat"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    selected = _select_codes(max_datasets=max_datasets)
    session = requests.Session()

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for row in selected:
        code = row["code"]
        csv_url = row["csv_url"]
        category = _category_for_code(code)
        file_name = f"eurostat__{code.lower()}.csv"
        out_path = out_dir / file_name

        item: dict[str, Any] = {
            "code": code,
            "csv_url": csv_url,
            "dataset_id": f"eurostat__{code.lower()}",
            "category_hint": category,
            "downloaded_at": now,
        }
        try:
            line_count = _line_count(out_path) if out_path.exists() and out_path.stat().st_size > 0 else _download_csv(
                session=session,
                url=csv_url,
                out_path=out_path,
                timeout_sec=timeout_sec,
            )
            columns = _header(out_path)
            row_count = max(0, line_count - 1)
            entry = {
                **item,
                "title": code,
                "source_url": csv_url,
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": row_count,
                "columns": columns,
            }
            success_rows.append(entry)
            print(f"[OK] {code} rows={row_count} file={file_name}")
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
            failed_rows.append(item)
            print(f"[FAIL] {code} error={exc}")

    manifest = {
        "industry": "shipping_freight",
        "region": "europe",
        "source": "eurostat_sdmx",
        "generated_at": now,
        "max_datasets": max_datasets,
        "timeout_sec": timeout_sec,
        "selected_count": len(selected),
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Europe shipping/freight datasets from Eurostat.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--max-datasets", type=int, default=30, help="Maximum number of Eurostat datasets to download")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_eurostat(
        base=base,
        max_datasets=max(1, int(args.max_datasets)),
        timeout_sec=max(20, int(args.timeout_sec)),
    )
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "region": "europe",
                "source": "eurostat_sdmx",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/europe/eurostat/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
