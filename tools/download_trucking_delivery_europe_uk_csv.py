#!/usr/bin/env python3
"""Download Europe trucking/delivery CSV datasets from data.gov.uk."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SEARCH_URL = "https://data.gov.uk/api/action/package_search"

QUERY_TERMS = [
    "road freight",
    "goods vehicle",
    "operator licence",
    "truck traffic",
    "delivery",
    "courier",
    "parcel",
    "driver safety",
]

TITLE_INCLUDE = (
    "road",
    "truck",
    "goods vehicle",
    "operator licence",
    "hgv",
    "lgv",
    "lorry",
    "van",
    "delivery",
    "courier",
    "parcel",
    "traffic",
    "driver",
    "fleet",
)

TITLE_EXCLUDE = (
    "marine",
    "vessel",
    "port",
    "ocean",
    "container",
    "rail",
    "shipping",
    "airport",
    "air",
    "health",
    "school",
    "housing",
)

ORG_INCLUDE = (
    "Department for Transport",
    "Driver and Vehicle Standards Agency",
    "OpenDataNI",
    "Department for Infrastructure",
    "Office for National Statistics",
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
    if any(k in t for k in ("delivery", "courier", "parcel", "last mile")):
        return "last_mile_sla"
    if any(k in t for k in ("operator licence", "goods vehicle", "fleet", "hgv", "lgv", "van")):
        return "fleet_utilization"
    if any(k in t for k in ("accident", "incident", "collision", "violation", "safety")):
        return "driver_safety_compliance"
    if any(k in t for k in ("traffic", "congestion", "road")):
        return "urban_traffic_risk"
    if any(k in t for k in ("border", "crossing")):
        return "cross_border_trucking"
    return "dispatch_capacity_balance"


def _is_candidate_package(pkg: dict[str, Any]) -> bool:
    title = str(pkg.get("title") or pkg.get("name") or "")
    if not title:
        return False
    tl = title.lower()
    if any(k in tl for k in TITLE_EXCLUDE):
        return False
    if not any(k in tl for k in TITLE_INCLUDE):
        return False

    org = str((pkg.get("organization") or {}).get("title") or "")
    return any(o.lower() in org.lower() for o in ORG_INCLUDE)


def _discover_candidates(session: requests.Session, max_pages: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for term in QUERY_TERMS:
        start = 0
        for _ in range(max_pages):
            params = {"q": term, "rows": 100, "start": start, "fq": "res_format:CSV"}
            resp = session.get(SEARCH_URL, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
            result = payload.get("result") or {}
            rows = result.get("results") or []
            if not rows:
                break

            for pkg in rows:
                pid = str(pkg.get("id") or "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                if _is_candidate_package(pkg):
                    out.append(pkg)

            start += len(rows)
            if start >= int(result.get("count", 0) or 0):
                break
    return out


def _csv_resources(pkg: dict[str, Any], per_package: int) -> list[dict[str, Any]]:
    resources = pkg.get("resources") or []
    csv_resources = []
    for r in resources:
        fmt = str(r.get("format") or "").lower()
        url = str(r.get("url") or "")
        if not url.startswith("http"):
            continue
        if fmt == "csv" or url.lower().endswith(".csv"):
            csv_resources.append(r)
    return csv_resources[: max(1, per_package)]


def download(base: Path, max_datasets: int, per_package_resources: int, max_pages: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "trucking_delivery" / "europe" / "uk"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    session = requests.Session()
    candidates = _discover_candidates(session, max_pages=max_pages)

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen_resource_ids: set[str] = set()

    for pkg in candidates:
        if len(success) >= max_datasets:
            break

        pkg_id = str(pkg.get("id") or "")
        title = str(pkg.get("title") or pkg.get("name") or pkg_id)
        org = str((pkg.get("organization") or {}).get("title") or "")

        for resource in _csv_resources(pkg, per_package_resources):
            if len(success) >= max_datasets:
                break

            resource_id = str(resource.get("id") or "")
            if not resource_id or resource_id in seen_resource_ids:
                continue
            seen_resource_ids.add(resource_id)

            url = str(resource.get("url") or "")
            safe_title = _slug(title)
            safe_name = _slug(str(resource.get("name") or resource_id))
            file_name = f"uk__{pkg_id[:8]}__{resource_id[:8]}__{safe_title}_{safe_name}.csv"
            out_path = out_dir / file_name
            item = {
                "package_id": pkg_id,
                "resource_id": resource_id,
                "title": title,
                "organization": org,
                "resource_name": resource.get("name"),
                "source_url": url,
                "downloaded_at": now,
            }

            try:
                if out_path.exists() and out_path.stat().st_size > 0:
                    line_count = _line_count(out_path)
                else:
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

                columns = _header(out_path)
                rows = max(0, line_count - 1)
                dataset_id = f"uk__{pkg_id[:8]}__{resource_id[:8]}__{safe_title}_{safe_name}"
                category_hint = _category_hint(f"{title} {resource.get('name') or ''}")
                success.append({
                    **item,
                    "dataset_id": dataset_id,
                    "category_hint": category_hint,
                    "rows": rows,
                    "columns": columns,
                    "file_path": str(out_path.relative_to(base)),
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
            "source": "uk_data_gov_ckan",
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
            "source_dataset_id": row.get("resource_id"),
            "region": "europe",
            "attribution": row.get("organization"),
            "portal_category": "UK Open Data",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download UK trucking/delivery CSV datasets.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--max-datasets", type=int, default=50)
    parser.add_argument("--per-package-resources", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download(base, args.max_datasets, args.per_package_resources, args.max_pages, args.timeout_sec)
    update_metadata(base, success)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": "trucking_delivery",
        "source": "uk_data_gov",
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "downloaded": success,
        "failed": failed,
    }
    report_path = base / "reports" / "trucking_delivery_europe_uk_download_report.json"
    _write_json(report_path, report)

    print(json.dumps({
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "report_path": str(report_path.relative_to(base)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
