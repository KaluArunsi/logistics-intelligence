#!/usr/bin/env python3
"""Seed trucking_delivery raw data from shipping_freight corpus using a road-only filter."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


INCLUDE_PATTERNS = (
    r"truck",
    r"trucking",
    r"road_",
    r"road-",
    r"goods_vehicle",
    r"operator_licence",
    r"operator_license",
    r"courier",
    r"delivery",
    r"last.?mile",
    r"vehicle",
    r"fleet",
    r"driver",
    r"traffic",
    r"border_crossing",
    r"crossing",
    r"mobility",
    r"travel_time",
    r"dispatch",
    r"route",
    r"parcel",
    r"van",
    r"hgv",
    r"lgv",
    r"ridership",
    r"motor",
)

# Terms that indicate ocean/freight-heavy datasets outside trucking/delivery scope.
EXCLUDE_PATTERNS = (
    r"ocean",
    r"marine",
    r"vessel",
    r"contain",
    r"\bteu\b",
    r"\bport\b",
    r"berth",
    r"ship",
    r"shipping",
    r"waterway",
    r"barge",
    r"canal",
    r"ferry",
    r"cruise",
    r"rail_go",
    r"railroad",
    r"intermodal",
    r"census_intltrade",
    r"\bimports?\b",
    r"\bexports?\b",
    r"\btrade\b",
    r"wholesale",
    r"merchandise",
    r"air_cargo",
    r"airport",
)

# Keep explicit USDOT/BTS trucking & safety datasets even if names contain broad freight terms.
ALLOWLIST_IDS = {
    "uta5-4eu5",
    "ez58-m3b4",
    "d7b8-pmxm",
    "mayv-2qfz",
    "dggd-bg3y",
    "sn4k-eiea",
    "xx4g-5dg2",
    "uwah-u9bn",
    "icqf-xf4w",
    "7wn6-i5b9",
    "8wvp-gjhh",
    "bx7m-yn3v",
    "keg4-3bc2",
    "3jux-kwvh",
    "btpt-uxhx",
    "xnav-e47e",
    "2a7t-n7sy",
    "tf5k-fhu2",
    "u6iw-gzjf",
}


@dataclass
class Stats:
    copied: int = 0
    skipped: int = 0


def _parse_view_id(path: Path) -> str:
    # expected patterns like usdot__d7b8-pmxm__name.csv / bts__keg4-3bc2__name.csv
    stem = path.stem.lower()
    parts = stem.split("__")
    if len(parts) >= 2 and re.fullmatch(r"[a-z0-9]{4}-[a-z0-9]{4}", parts[1]):
        return parts[1]
    return ""


def _is_candidate(rel_posix: str) -> bool:
    low = rel_posix.lower()
    if low.endswith("_download_manifest.json") or low.endswith(".tmp"):
        return False

    vid = _parse_view_id(Path(rel_posix))
    if vid and vid in ALLOWLIST_IDS:
        return True

    if not any(re.search(p, low) for p in INCLUDE_PATTERNS):
        return False
    if any(re.search(p, low) for p in EXCLUDE_PATTERNS):
        return False
    return True


def seed(base: Path, clean: bool) -> dict:
    src = base / "data" / "raw" / "shipping_freight"
    dst = base / "data" / "raw" / "trucking_delivery"
    manifest = dst / "_seed_manifest.json"

    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    kept: list[str] = []
    skipped: list[str] = []

    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        rel_posix = rel.as_posix()
        if _is_candidate(rel_posix):
            out_path = dst / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out_path)
            kept.append(rel_posix)
            stats.copied += 1
        else:
            skipped.append(rel_posix)
            stats.skipped += 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(src.relative_to(base)),
        "destination": str(dst.relative_to(base)),
        "copied_count": stats.copied,
        "skipped_count": stats.skipped,
        "include_patterns": list(INCLUDE_PATTERNS),
        "exclude_patterns": list(EXCLUDE_PATTERNS),
        "allowlist_ids": sorted(ALLOWLIST_IDS),
        "copied_files": kept,
    }
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed trucking_delivery raw folder from shipping_freight.")
    parser.add_argument("--base", default=".", help="Project root")
    parser.add_argument("--no-clean", action="store_true", help="Do not wipe destination folder first")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()
    report = seed(base, clean=not args.no_clean)
    print(json.dumps({
        "copied_count": report["copied_count"],
        "skipped_count": report["skipped_count"],
        "manifest": "data/raw/trucking_delivery/_seed_manifest.json",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
