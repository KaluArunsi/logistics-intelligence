#!/usr/bin/env python3
"""Download Asia trucking/delivery datasets from Hong Kong open data (CKAN)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SEARCH_URL = "https://data.gov.hk/en-data/api/3/action/package_search"

QUERY_TERMS = [
    "transport",
    "courier",
    "delivery",
    "vehicle",
    "traffic",
    "road",
    "logistics",
]

INCLUDE_KEYWORDS = (
    "transport",
    "courier",
    "delivery",
    "vehicle",
    "traffic",
    "road",
    "logistics",
    "operator",
)

EXCLUDE_KEYWORDS = (
    "marine",
    "vessel",
    "shipping",
    "container",
    "port",
    "air transport",
    "air cargo",
    "airport",
    "merchandise trade",
    "school",
    "health",
)

ALLOWED_ORGS = {
    "Census and Statistics Department",
    "Transport Department",
}


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


def _category_hint(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("courier", "delivery", "last mile")):
        return "last_mile_sla"
    if any(k in t for k in ("vehicle", "fleet", "operator", "licence")):
        return "fleet_utilization"
    if any(k in t for k in ("traffic", "congestion", "road")):
        return "urban_traffic_risk"
    if any(k in t for k in ("safety", "accident", "incident")):
        return "driver_safety_compliance"
    return "dispatch_capacity_balance"


def _is_candidate_package(pkg: dict[str, Any]) -> bool:
    org = str((pkg.get("organization") or {}).get("title") or "").strip()
    if org not in ALLOWED_ORGS:
        return False

    title = str(pkg.get("title") or pkg.get("name") or "").lower()
    if not title:
        return False
    if any(k in title for k in EXCLUDE_KEYWORDS):
        return False
    return any(k in title for k in INCLUDE_KEYWORDS)


def _pick_resources(pkg: dict[str, Any], per_package: int) -> list[dict[str, Any]]:
    resources = pkg.get("resources") or []
    supported = []
    for r in resources:
        fmt = str(r.get("format") or "").strip().lower()
        url = str(r.get("url") or "")
        if not url.startswith("http"):
            continue
        if fmt in {"json", "csv"}:
            supported.append(r)

    def score(r: dict[str, Any]) -> int:
        name = str(r.get("name") or "").lower()
        fmt = str(r.get("format") or "").strip().lower()
        s = 0
        if fmt == "json":
            s += 2
        if "english" in name:
            s += 2
        if "api" in name:
            s -= 1
        return s

    ranked = sorted(supported, key=score, reverse=True)
    return ranked[: max(1, per_package)]


def _discover_candidates(session: requests.Session) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_pkg_ids: set[str] = set()
    for term in QUERY_TERMS:
        payload = session.get(SEARCH_URL, params={"q": term, "rows": 100}, timeout=45).json()
        for pkg in (payload.get("result") or {}).get("results") or []:
            pkg_id = str(pkg.get("id") or "")
            if not pkg_id or pkg_id in seen_pkg_ids:
                continue
            seen_pkg_ids.add(pkg_id)
            if _is_candidate_package(pkg):
                out.append(pkg)
    return out


def download(base: Path, max_datasets: int, per_package_resources: int, timeout_sec: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "trucking_delivery" / "asia" / "hong_kong"
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.censtatd.gov.hk/",
        "Accept": "application/json,text/csv,*/*",
    })

    candidates = _discover_candidates(session)
    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen_resource_ids: set[str] = set()

    for pkg in candidates:
        if len(success) >= max_datasets:
            break

        pkg_id = str(pkg.get("id") or "")
        title = str(pkg.get("title") or pkg.get("name") or pkg_id)
        org = str((pkg.get("organization") or {}).get("title") or "")

        for resource in _pick_resources(pkg, per_package_resources):
            if len(success) >= max_datasets:
                break

            resource_id = str(resource.get("id") or "")
            if not resource_id or resource_id in seen_resource_ids:
                continue
            seen_resource_ids.add(resource_id)

            url = str(resource.get("url") or "")
            ext = ".json" if str(resource.get("format") or "").lower() == "json" else ".csv"
            safe_title = _slug(title)
            file_name = f"hk__{pkg_id[:8]}__{resource_id[:8]}__{safe_title}{ext}"
            out_path = out_dir / file_name
            item = {
                "package_id": pkg_id,
                "resource_id": resource_id,
                "title": title,
                "organization": org,
                "source_url": url,
                "downloaded_at": now,
            }

            try:
                if not out_path.exists() or out_path.stat().st_size == 0:
                    r = session.get(url, timeout=timeout_sec)
                    r.raise_for_status()
                    out_path.write_bytes(r.content)

                if out_path.suffix.lower() == ".csv":
                    line_count = _line_count(out_path)
                    if line_count <= 1:
                        raise RuntimeError("empty_or_header_only")
                    columns = _header(out_path)
                    rows = max(0, line_count - 1)
                else:
                    payload = json.loads(out_path.read_text(encoding="utf-8", errors="ignore"))
                    data = payload.get("result", {}).get("records") if isinstance(payload, dict) else []
                    rows = len(data) if isinstance(data, list) else 0
                    columns = list(data[0].keys()) if rows else []

                dataset_id = f"hk__{pkg_id[:8]}__{resource_id[:8]}__{safe_title}"
                category_hint = _category_hint(title)
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
            "source": "hong_kong_data_gov_hk_ckan",
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
            "region": "asia",
            "attribution": row.get("organization"),
            "portal_category": "Hong Kong Open Data",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download trucking/delivery datasets from Hong Kong open data.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--max-datasets", type=int, default=30)
    parser.add_argument("--per-package-resources", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download(base, args.max_datasets, args.per_package_resources, args.timeout_sec)
    update_metadata(base, success)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": "trucking_delivery",
        "source": "hong_kong_open_data",
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "downloaded": success,
        "failed": failed,
    }
    report_path = base / "reports" / "trucking_delivery_asia_hongkong_download_report.json"
    _write_json(report_path, report)

    print(json.dumps({
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "report_path": str(report_path.relative_to(base)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
