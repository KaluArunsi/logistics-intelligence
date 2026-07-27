#!/usr/bin/env python3
"""Download Amazon Last Mile Routing Research Challenge (ALMRRC 2021) data."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BUCKET_LIST_URL = "https://amazon-last-mile-challenges.s3.amazonaws.com/?prefix=almrrc2021/"

FILE_ALLOWLIST = {
    "almrrc2021/Readme.txt": "route_eta_reliability",
    "almrrc2021/almrrc2021-data-training/model_build_inputs/route_data.json": "route_eta_reliability",
    "almrrc2021/almrrc2021-data-training/model_build_inputs/travel_times.json": "route_eta_reliability",
    "almrrc2021/almrrc2021-data-training/model_build_inputs/package_data.json": "last_mile_sla",
    "almrrc2021/almrrc2021-data-training/model_build_inputs/actual_sequences.json": "pickup_dropoff_reliability",
    "almrrc2021/almrrc2021-data-training/model_build_inputs/invalid_sequence_scores.json": "route_eta_reliability",
    "almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/eval_route_data.json": "route_eta_reliability",
    "almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json": "route_eta_reliability",
    "almrrc2021/almrrc2021-data-evaluation/model_apply_inputs/eval_package_data.json": "last_mile_sla",
    "almrrc2021/almrrc2021-data-evaluation/model_score_inputs/eval_actual_sequences.json": "pickup_dropoff_reliability",
    "almrrc2021/almrrc2021-data-evaluation/model_score_inputs/eval_invalid_sequence_scores.json": "route_eta_reliability",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _count_rows(path: Path) -> int:
    # Lightweight row estimate for metadata consistency.
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return 0
        if isinstance(data, dict):
            return len(data)
        if isinstance(data, list):
            return len(data)
        return 1
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def _count_cols(path: Path) -> int:
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return 0
        if isinstance(data, dict) and data:
            first = next(iter(data.values()))
            if isinstance(first, dict):
                return len(first)
            return 1
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return len(first)
            return 1
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        return len(next(reader, []))


def _dataset_id_from_key(key: str) -> str:
    stem = key.replace("almrrc2021/", "").replace("/", "__").replace(".", "_")
    return f"amazon_lastmile__{stem}"


def list_bucket_keys(session: requests.Session) -> list[str]:
    xml_text = session.get(BUCKET_LIST_URL, timeout=45).text
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(xml_text)
    return [k.text for k in root.findall("s:Contents/s:Key", ns) if k.text]


def list_bucket_sizes(session: requests.Session) -> dict[str, int]:
    xml_text = session.get(BUCKET_LIST_URL, timeout=45).text
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(xml_text)
    sizes: dict[str, int] = {}
    for node in root.findall("s:Contents", ns):
        key = node.find("s:Key", ns)
        size = node.find("s:Size", ns)
        if key is None or size is None or not key.text:
            continue
        try:
            sizes[key.text] = int(size.text or "0")
        except Exception:
            sizes[key.text] = 0
    return sizes


def download(base: Path, timeout_sec: int, max_file_mb: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = base / "data" / "raw" / "trucking_delivery" / "us" / "amazon_lastmile"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    keys = set(list_bucket_keys(session))
    sizes = list_bucket_sizes(session)
    now = datetime.now(timezone.utc).isoformat()

    success: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for key, category in FILE_ALLOWLIST.items():
        if key not in keys:
            failed.append({"key": key, "error": "missing_in_bucket"})
            continue

        size_bytes = int(sizes.get(key, 0))
        if size_bytes > max_file_mb * 1024 * 1024:
            failed.append(
                {
                    "key": key,
                    "error": "skipped_file_too_large",
                    "size_bytes": size_bytes,
                    "max_allowed_bytes": max_file_mb * 1024 * 1024,
                }
            )
            continue

        url = f"https://amazon-last-mile-challenges.s3.amazonaws.com/{key}"
        rel_name = key.replace("almrrc2021/", "")
        out_path = out_dir / rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not out_path.exists() or out_path.stat().st_size == 0:
                resp = session.get(url, timeout=timeout_sec)
                resp.raise_for_status()
                out_path.write_bytes(resp.content)

            rows = _count_rows(out_path)
            cols = _count_cols(out_path)
            dataset_id = _dataset_id_from_key(key)
            success.append(
                {
                    "dataset_id": dataset_id,
                    "key": key,
                    "category_hint": category,
                    "source_url": url,
                    "downloaded_at": now,
                    "file_path": str(out_path.relative_to(base)),
                    "rows": rows,
                    "n_cols": cols,
                    "size_bytes": size_bytes,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"key": key, "source_url": url, "error": str(exc)})

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
        cat = row.get("category_hint") or "route_eta_reliability"
        datasets[dataset_id] = {
            "id": dataset_id,
            "name": dataset_id,
            "source": "amazon_last_mile_routing_research_challenge",
            "category": cat,
            "categories": [cat],
            "n_rows": int(row.get("rows", 0) or 0),
            "n_cols": int(row.get("n_cols", 0) or 0),
            "columns": [],
            "target_column": None,
            "task_type": None,
            "file_path": row.get("file_path"),
            "processed": False,
            "ingested_at": row.get("downloaded_at") or now,
            "fingerprint": None,
            "duplicate_of": None,
            "worker_dataset_id": None,
            "source_dataset_id": row.get("key"),
            "region": "us",
            "attribution": "Amazon Last Mile Routing Research Challenge 2021",
            "portal_category": "Route Optimization",
            "portal_url": row.get("source_url"),
            "updated_at": now,
        }

    _write_json(meta_path, datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Amazon Last Mile Challenge files.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=100,
        help="Skip files larger than this size in MB to avoid oversized downloads.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    success, failed = download(base=base, timeout_sec=args.timeout_sec, max_file_mb=args.max_file_mb)
    update_metadata(base, success)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "industry": "trucking_delivery",
        "source": "amazon_lastmile",
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "downloaded": success,
        "failed": failed,
    }
    report_path = base / "reports" / "trucking_delivery_amazon_lastmile_download_report.json"
    _write_json(report_path, report)
    print(json.dumps({
        "downloaded_count": len(success),
        "failed_count": len(failed),
        "report_path": str(report_path.relative_to(base)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
