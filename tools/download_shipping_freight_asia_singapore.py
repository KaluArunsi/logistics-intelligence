#!/usr/bin/env python3
"""Download Asia shipping/freight datasets from Singapore official portals."""

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


API_LIST_URL = "https://api-production.data.gov.sg/v2/public/api/datasets"
SINGSTAT_TABLE_URL = "https://tablebuilder.singstat.gov.sg/api/table/tabledata/{table_id}"

ALLOWED_AGENCIES = {
    "Maritime and Port Authority of Singapore",
    "Singapore Department of Statistics",
    "Singapore Customs",
    "Land Transport Authority",
}

INCLUDE_KEYWORDS = (
    "sea cargo",
    "shipping",
    "vessel",
    "bunker",
    "throughput",
    "port",
    "trade",
    "export",
    "imports",
    "import",
    "container",
    "cargo",
    "goods vehicle",
    "wholesale trade",
)

EXCLUDE_KEYWORDS = (
    "school",
    "students",
    "household",
    "hdb",
    "union membership",
    "rehabilitation",
    "hiv",
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


def _extract_table_id(description: str) -> str | None:
    if not description:
        return None
    match = re.search(r"/table/TS/([A-Z0-9]+)", description)
    if match:
        return match.group(1)
    return None


def _is_candidate(row: dict[str, Any]) -> bool:
    agency = str(row.get("managedByAgencyName") or "").strip()
    if agency not in ALLOWED_AGENCIES:
        return False
    name = str(row.get("name") or "").lower()
    if not any(k in name for k in INCLUDE_KEYWORDS):
        return False
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    fmt = str(row.get("format") or "").upper()
    return fmt in {"CSV", "XLSX", "JSON"}


def _category_hint(title: str, topic: str, subject: str) -> str:
    text = f"{title} {topic} {subject}".lower()
    if "customs" in text:
        return "customs_compliance"
    if "goods vehicle" in text or "road freight" in text or "truck" in text:
        return "trucking_capacity"
    if "vessel" in text or "shipping" in text or "port" in text or "berth" in text:
        return "ocean_schedule_reliability"
    if "cargo throughput" in text or "cargo" in text or "container" in text:
        return "port_terminal_congestion"
    return "trade_volume_mix"


def _discover_candidates(session: requests.Session, max_pages: int, target_count: int) -> list[dict[str, Any]]:
    first = session.get(API_LIST_URL, params={"page": 1}, timeout=40).json().get("data") or {}
    pages = min(int(first.get("pages", 1) or 1), max_pages)
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for page in range(1, pages + 1):
        payload = session.get(API_LIST_URL, params={"page": page}, timeout=40).json().get("data") or {}
        for row in payload.get("datasets") or []:
            dataset_id = str(row.get("datasetId") or "")
            if not dataset_id or dataset_id in seen_ids:
                continue
            if _is_candidate(row):
                out.append(row)
                seen_ids.add(dataset_id)
        if len(out) >= target_count * 3:
            break
    return out


def _flatten_table_rows(
    payload: dict[str, Any],
    dataset_id: str,
    dataset_name: str,
    table_id: str,
) -> list[dict[str, Any]]:
    data = payload.get("Data") or {}
    rows = data.get("row") or []
    out: list[dict[str, Any]] = []
    for row in rows:
        series_no = str(row.get("seriesNo") or "")
        row_text = str(row.get("rowText") or "")
        uom = str(row.get("uoM") or "")
        footnote = str(row.get("footnote") or "")
        for cell in row.get("columns") or []:
            out.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "table_id": table_id,
                    "theme": str(data.get("theme") or ""),
                    "subject": str(data.get("subject") or ""),
                    "topic": str(data.get("topic") or ""),
                    "datasource": str(data.get("datasource") or ""),
                    "series_no": series_no,
                    "row_text": row_text,
                    "unit": uom,
                    "period": str(cell.get("key") or ""),
                    "value": str(cell.get("value") or ""),
                    "row_footnote": footnote,
                }
            )
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
            "source": "singapore_datagov_singstat",
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
            "source_dataset_id": row.get("table_id"),
            "region": "asia",
            "attribution": row.get("managed_by"),
            "portal_category": "Singapore Official Statistics",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def download_asia_singapore(base: Path, max_datasets: int, max_pages: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "asia" / "singapore"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://tablebuilder.singstat.gov.sg/",
        }
    )

    candidates = _discover_candidates(session=session, max_pages=max_pages, target_count=max_datasets)
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    seen_table_ids: set[str] = set()

    for row in candidates:
        if len(success_rows) >= max_datasets:
            break

        dataset_id = str(row.get("datasetId") or "")
        name = str(row.get("name") or dataset_id)
        meta_url = f"{API_LIST_URL}/{dataset_id}/metadata"
        item: dict[str, Any] = {
            "dataset_id": dataset_id,
            "name": name,
            "meta_url": meta_url,
            "downloaded_at": now,
        }
        try:
            meta = session.get(meta_url, timeout=40).json().get("data") or {}
            description = str(meta.get("description") or "")
            table_id = _extract_table_id(description)
            if not table_id:
                raise RuntimeError("no_table_id_in_metadata")
            if table_id in seen_table_ids:
                continue

            table_url = SINGSTAT_TABLE_URL.format(table_id=table_id)
            table_payload = session.get(table_url, timeout=40).json()
            if int(table_payload.get("StatusCode", 200) or 200) != 200:
                raise RuntimeError(f"table_api_status_{table_payload.get('StatusCode')}")

            flat_rows = _flatten_table_rows(
                payload=table_payload,
                dataset_id=dataset_id,
                dataset_name=name,
                table_id=table_id,
            )
            if not flat_rows:
                raise RuntimeError("no_rows_after_flatten")

            file_name = f"singstat__{table_id.lower()}__{_slug(name)}.csv"
            out_path = out_dir / file_name
            _write_csv(out_path, flat_rows)
            line_count = _line_count(out_path)
            row_count = max(0, line_count - 1)
            columns = _header(out_path)

            topic = str((table_payload.get("Data") or {}).get("topic") or "")
            subject = str((table_payload.get("Data") or {}).get("subject") or "")
            category = _category_hint(name, topic, subject)
            entry = {
                **item,
                "dataset_id": f"singstat__{table_id.lower()}__{_slug(name)}",
                "source_dataset_id": dataset_id,
                "table_id": table_id,
                "title": name,
                "managed_by": str(meta.get("managedBy") or row.get("managedByAgencyName") or ""),
                "source_url": f"https://tablebuilder.singstat.gov.sg/table/TS/{table_id}",
                "file_name": file_name,
                "file_path": str(out_path.relative_to(base)),
                "rows": row_count,
                "columns": columns,
                "category_hint": category,
            }
            success_rows.append(entry)
            seen_table_ids.add(table_id)
            print(f"[OK] {table_id} rows={row_count} file={file_name}")

            # Avoid data.gov.sg throttling bursts while reading metadata.
            time.sleep(0.25)
        except Exception as exc:  # noqa: BLE001
            item["error"] = str(exc)
            failed_rows.append(item)
            print(f"[FAIL] {dataset_id} error={exc}")
            time.sleep(0.5)

    manifest = {
        "industry": "shipping_freight",
        "region": "asia",
        "source": "singapore_datagov_singstat",
        "generated_at": now,
        "max_datasets": max_datasets,
        "max_pages": max_pages,
        "candidate_count": len(candidates),
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Asia shipping/freight datasets from Singapore official sources.")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--max-datasets", type=int, default=35, help="Maximum datasets to download")
    parser.add_argument("--max-pages", type=int, default=250, help="Maximum list pages to scan")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_asia_singapore(
        base=base,
        max_datasets=max(1, int(args.max_datasets)),
        max_pages=max(1, int(args.max_pages)),
    )
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "region": "asia",
                "source": "singapore_datagov_singstat",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/asia/singapore/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

