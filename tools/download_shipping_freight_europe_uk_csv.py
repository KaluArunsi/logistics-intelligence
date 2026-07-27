#!/usr/bin/env python3
"""Download Europe shipping/freight datasets from data.gov.uk (CSV resources only)."""

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
    "freight",
    "road freight",
    "goods vehicle",
    "operator licence",
    "port freight",
    "rail freight",
    "logistics",
    "parcel delivery",
    "courier",
    "truck traffic",
    "marine accident",
]

ORG_INCLUDE = (
    "Department for Transport",
    "Driver and Vehicle Standards Agency",
    "OpenDataNI",
    "Department for Infrastructure",
    "HM Revenue & Customs",
    "Department for Business and Trade",
    "Office for National Statistics",
)

TITLE_INCLUDE = (
    "freight",
    "goods vehicle",
    "operator licence",
    "hgv",
    "lgv",
    "lorry",
    "truck",
    "van",
    "port",
    "shipping",
    "marine",
    "cargo",
    "logistics",
    "delivery",
    "courier",
    "parcel",
    "customs",
    "border",
    "trade",
    "rail freight",
    "inland waterway",
    "barge",
)

TITLE_EXCLUDE = (
    "housing delivery",
    "service delivery footprints",
    "maternal smoking",
    "school",
    "education",
    "hospital",
    "health",
    "noise exposure",
    "species",
    "habitat",
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


def _category_hint(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("parcel", "courier", "last mile", "delivery performance", "delivery")):
        return "last_mile_sla"
    if any(k in t for k in ("goods vehicle", "operator licence", "fleet", "hgv", "lgv", "lorry", "van")):
        return "fleet_utilization"
    if any(k in t for k in ("accident", "safety", "collision", "casualty", "incident")):
        return "carrier_safety_risk"
    if any(k in t for k in ("cold", "temperature", "reefer", "chilled")):
        return "cold_chain_integrity"
    if any(k in t for k in ("inland waterway", "barge", "canal", "river", "lock")):
        return "inland_waterway_flow"
    if any(k in t for k in ("customs", "clearance", "border")):
        return "customs_compliance"
    if any(k in t for k in ("port", "berth", "terminal", "shipping", "vessel", "marine")):
        return "port_terminal_congestion"
    if any(k in t for k in ("rail freight", "intermodal")):
        return "rail_intermodal_flow"
    if any(k in t for k in ("trade", "imports", "exports", "cargo", "container")):
        return "trade_volume_mix"
    if any(k in t for k in ("road freight", "truck traffic", "freight")):
        return "trucking_capacity"
    return "lane_cost_yield"


def _is_candidate_package(pkg: dict[str, Any]) -> bool:
    title = str(pkg.get("title") or pkg.get("name") or "")
    if not title:
        return False
    tl = title.lower()
    if any(x in tl for x in TITLE_EXCLUDE):
        return False
    if not any(x in tl for x in TITLE_INCLUDE):
        return False

    org = str((pkg.get("organization") or {}).get("title") or "")
    return any(o.lower() in org.lower() for o in ORG_INCLUDE)


def _discover_candidates(session: requests.Session, max_pages: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_pkg_ids: set[str] = set()

    for term in QUERY_TERMS:
        start = 0
        for _ in range(max_pages):
            params = {
                "q": term,
                "rows": 100,
                "start": start,
                "fq": "res_format:CSV",
            }
            payload: dict[str, Any] | None = None
            for attempt in range(4):
                try:
                    resp = session.get(SEARCH_URL, params=params, timeout=45)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception:  # noqa: BLE001
                    payload = None
                    time.sleep(0.6 + attempt * 0.5)
            if not payload:
                break
            result = payload.get("result") or {}
            rows = result.get("results") or []
            if not rows:
                break

            for pkg in rows:
                pkg_id = str(pkg.get("id") or "")
                if not pkg_id or pkg_id in seen_pkg_ids:
                    continue
                seen_pkg_ids.add(pkg_id)
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
        if fmt == "csv" or url.lower().endswith(".csv"):
            if url.startswith("http"):
                csv_resources.append(r)
    return csv_resources[: max(1, per_package)]


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
            "source": "uk_data_gov_ckan",
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
            "source_dataset_id": row.get("resource_id"),
            "region": "europe",
            "attribution": row.get("organization"),
            "portal_category": "UK Open Data",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def download_uk_csv(
    base: Path,
    max_datasets: int,
    per_package_resources: int,
    max_pages: int,
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "europe" / "uk"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    session = requests.Session()

    candidates = _discover_candidates(session=session, max_pages=max_pages)
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    seen_resource_ids: set[str] = set()
    seen_urls: set[str] = set()

    for pkg in candidates:
        if len(success_rows) >= max_datasets:
            break
        pkg_id = str(pkg.get("id") or "")
        title = str(pkg.get("title") or pkg.get("name") or pkg_id)
        org = str((pkg.get("organization") or {}).get("title") or "")
        cat = _category_hint(f"{title} {org}")

        for res in _csv_resources(pkg, per_package=per_package_resources):
            if len(success_rows) >= max_datasets:
                break
            rid = str(res.get("id") or "")
            url = str(res.get("url") or "")
            if not url or url in seen_urls:
                continue
            if rid and rid in seen_resource_ids:
                continue
            if rid:
                seen_resource_ids.add(rid)
            seen_urls.add(url)

            r_name = str(res.get("name") or "")
            safe = _slug(f"{title}_{r_name}")[:120]
            rid_part = rid[:8] if rid else _slug(url)[-8:]
            file_name = f"uk__{pkg_id[:8]}__{rid_part}__{safe}.csv"
            out_path = out_dir / file_name
            item = {
                "package_id": pkg_id,
                "resource_id": rid,
                "dataset_id": f"uk__{pkg_id[:8]}__{rid_part}__{safe}",
                "title": title,
                "resource_name": r_name,
                "organization": org,
                "source_url": url,
                "category_hint": cat,
                "downloaded_at": now,
            }
            try:
                if out_path.exists() and out_path.stat().st_size > 0:
                    line_count = _line_count(out_path)
                else:
                    with session.get(url, timeout=timeout_sec, stream=True) as resp:
                        resp.raise_for_status()
                        tmp = out_path.with_suffix(".csv.tmp")
                        with open(tmp, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                        line_count = _line_count(tmp)
                        if line_count <= 1:
                            tmp.unlink(missing_ok=True)
                            raise RuntimeError("empty_or_header_only")
                        tmp.rename(out_path)
                cols = _header(out_path)
                row_count = max(0, line_count - 1)
                success_rows.append(
                    {
                        **item,
                        "file_name": file_name,
                        "file_path": str(out_path.relative_to(base)),
                        "rows": row_count,
                        "columns": cols,
                    }
                )
                print(f"[OK] {title} | {r_name} rows={row_count}")
            except Exception as exc:  # noqa: BLE001
                failed_rows.append({**item, "error": str(exc)})
                print(f"[FAIL] {title} | {r_name} error={exc}")

    manifest = {
        "industry": "shipping_freight",
        "region": "europe",
        "source": "uk_data_gov_ckan",
        "generated_at": now,
        "max_datasets": max_datasets,
        "per_package_resources": per_package_resources,
        "max_pages": max_pages,
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download shipping/freight Europe datasets from data.gov.uk")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--max-datasets", type=int, default=35, help="Maximum datasets to download")
    parser.add_argument("--per-package-resources", type=int, default=2, help="Max CSV resources per package")
    parser.add_argument("--max-pages", type=int, default=3, help="Max pagination pages per query term")
    parser.add_argument("--timeout-sec", type=int, default=120, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_uk_csv(
        base=base,
        max_datasets=max(1, int(args.max_datasets)),
        per_package_resources=max(1, int(args.per_package_resources)),
        max_pages=max(1, int(args.max_pages)),
        timeout_sec=max(20, int(args.timeout_sec)),
    )
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "region": "europe",
                "source": "uk_data_gov_ckan",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/europe/uk/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
