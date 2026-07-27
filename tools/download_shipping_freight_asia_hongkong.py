#!/usr/bin/env python3
"""Download Asia shipping/freight datasets from data.gov.hk (CKAN API)."""

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
    "logistics",
    "shipping",
    "marine",
    "cargo",
    "container",
    "courier",
    "vehicle",
    "trade transport",
]

ALLOWED_ORGS = {
    "Census and Statistics Department",
    "Marine Department",
    "Transport Department",
}

INCLUDE_KEYWORDS = (
    "transport",
    "logistics",
    "shipping",
    "marine",
    "cargo",
    "container",
    "courier",
    "delivery",
    "trade",
    "vehicle",
    "freight",
    "port",
    "vessel",
    "traffic",
)

EXCLUDE_KEYWORDS = (
    "auction",
    "school",
    "education",
    "housing",
    "hospital",
    "health",
    "influenza",
    "covid",
    "reproductive",
    "snapshot image",
    "traffic snapshot",
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


def _category_hint(title: str) -> str:
    text = title.lower()
    if any(k in text for k in ("courier", "delivery", "last mile", "on-time")):
        return "last_mile_sla"
    if any(k in text for k in ("vehicle", "operator licence", "fleet", "hgv", "lgv", "truck", "van")):
        return "fleet_utilization"
    if any(k in text for k in ("accident", "incident", "safety", "casualty", "violation")):
        return "carrier_safety_risk"
    if any(k in text for k in ("cold", "temperature", "reefer", "spoilage")):
        return "cold_chain_integrity"
    if any(k in text for k in ("inland waterway", "river", "canal", "barge", "lock")):
        return "inland_waterway_flow"
    if any(k in text for k in ("customs", "clearance", "inspection", "hs code")):
        return "customs_compliance"
    if any(k in text for k in ("border", "crossing", "port of entry")):
        return "cross_border_flow"
    if any(k in text for k in ("port", "terminal", "berth", "vessel", "ship", "marine")):
        return "port_terminal_congestion"
    if any(k in text for k in ("trade", "imports", "exports", "cargo", "container")):
        return "trade_volume_mix"
    return "trucking_capacity"


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
    if not supported:
        return []

    # Prefer JSON (avoids CSV 403 on some C&SD links) and English resources.
    def score(r: dict[str, Any]) -> int:
        name = str(r.get("name") or "").lower()
        fmt = str(r.get("format") or "").strip().lower()
        s = 0
        if fmt == "json":
            s += 4
        if "english" in name:
            s += 3
        if "traditional chinese" in name or "simplified chinese" in name:
            s -= 3
        if "api" in name:
            s -= 1
        return s

    ranked = sorted(supported, key=score, reverse=True)
    return ranked[: max(1, per_package)]


def _discover_candidates(session: requests.Session) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_pkg_ids: set[str] = set()
    for term in QUERY_TERMS:
        params = {"q": term, "rows": 100}
        payload = session.get(SEARCH_URL, params=params, timeout=45).json()
        for pkg in (payload.get("result") or {}).get("results") or []:
            pkg_id = str(pkg.get("id") or "")
            if not pkg_id or pkg_id in seen_pkg_ids:
                continue
            seen_pkg_ids.add(pkg_id)
            if _is_candidate_package(pkg):
                out.append(pkg)
    return out


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
            "source": "hong_kong_data_gov_hk_ckan",
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
            "region": "asia",
            "attribution": row.get("organization"),
            "portal_category": "Hong Kong Open Data",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def download_hk(
    base: Path,
    max_datasets: int,
    per_package_resources: int,
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "shipping_freight" / "asia" / "hong_kong"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.censtatd.gov.hk/",
            "Accept": "application/json,text/csv,*/*",
        }
    )
    candidates = _discover_candidates(session)
    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    seen_resource_ids: set[str] = set()

    for pkg in candidates:
        if len(success_rows) >= max_datasets:
            break

        pkg_id = str(pkg.get("id") or "")
        pkg_title = str(pkg.get("title") or pkg.get("name") or pkg_id)
        org = str((pkg.get("organization") or {}).get("title") or "")
        category = _category_hint(pkg_title)

        for res in _pick_resources(pkg, per_package=per_package_resources):
            if len(success_rows) >= max_datasets:
                break
            resource_id = str(res.get("id") or "")
            if not resource_id or resource_id in seen_resource_ids:
                continue
            seen_resource_ids.add(resource_id)

            url = str(res.get("url") or "")
            r_name = str(res.get("name") or "")
            fmt = str(res.get("format") or "").strip().lower()
            safe = _slug(f"{pkg_title}_{r_name}")[:120]
            file_name = f"hk__{pkg_id[:8]}__{resource_id[:8]}__{safe}.csv"
            out_path = out_dir / file_name
            item = {
                "package_id": pkg_id,
                "resource_id": resource_id,
                "dataset_id": f"hongkong__{pkg_id[:8]}__{resource_id[:8]}__{safe}",
                "title": pkg_title,
                "resource_name": r_name,
                "organization": org,
                "source_url": url,
                "category_hint": category,
                "downloaded_at": now,
            }
            try:
                if out_path.exists() and out_path.stat().st_size > 0:
                    line_count = _line_count(out_path)
                else:
                    tmp = out_path.with_suffix(".csv.tmp")
                    if fmt == "json":
                        resp = session.get(url, timeout=timeout_sec)
                        resp.raise_for_status()
                        payload = resp.json()
                        rows = payload.get("dataSet") or []
                        if not isinstance(rows, list) or not rows:
                            raise RuntimeError("json_no_rows")
                        fieldnames = sorted({k for row in rows if isinstance(row, dict) for k in row.keys()})
                        if not fieldnames:
                            raise RuntimeError("json_no_fields")
                        with open(tmp, "w", encoding="utf-8", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=fieldnames)
                            w.writeheader()
                            for row in rows:
                                if isinstance(row, dict):
                                    w.writerow({k: row.get(k) for k in fieldnames})
                    else:
                        with session.get(url, timeout=timeout_sec, stream=True) as resp:
                            resp.raise_for_status()
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
                print(f"[OK] {pkg_title} | {r_name} rows={row_count}")
            except Exception as exc:  # noqa: BLE001
                failed_rows.append({**item, "error": str(exc)})
                print(f"[FAIL] {pkg_title} | {r_name} error={exc}")

    manifest = {
        "industry": "shipping_freight",
        "region": "asia",
        "source": "hong_kong_data_gov_hk_ckan",
        "generated_at": now,
        "max_datasets": max_datasets,
        "per_package_resources": per_package_resources,
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "downloads": success_rows,
        "failures": failed_rows,
    }
    _write_json(out_dir / "_download_manifest.json", manifest)
    _update_shipping_metadata(base, success_rows)
    return success_rows, failed_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download shipping/freight Asia datasets from data.gov.hk")
    parser.add_argument("--base", default=".", help="Project root path")
    parser.add_argument("--max-datasets", type=int, default=30, help="Maximum datasets to download")
    parser.add_argument("--per-package-resources", type=int, default=1, help="Maximum CSV resources per package")
    parser.add_argument("--timeout-sec", type=int, default=120, help="HTTP timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download_hk(
        base=base,
        max_datasets=max(1, int(args.max_datasets)),
        per_package_resources=max(1, int(args.per_package_resources)),
        timeout_sec=max(20, int(args.timeout_sec)),
    )
    print(
        json.dumps(
            {
                "industry": "shipping_freight",
                "region": "asia",
                "source": "hong_kong_data_gov_hk_ckan",
                "success_count": len(success),
                "failure_count": len(failed),
                "manifest": "data/raw/shipping_freight/asia/hong_kong/_download_manifest.json",
            },
            indent=2,
        )
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
