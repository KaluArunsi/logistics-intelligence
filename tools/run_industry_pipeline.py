#!/usr/bin/env python3
"""CLI wrapper to run industry pipeline in local or containerized environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cost_policy import CostPolicy
from src.core.industry_pipeline import IndustryPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run industry pipeline and print summary.")
    parser.add_argument("--base", default=".", help="Project root path.")
    parser.add_argument("--industry", required=True, help="Industry key (e.g., aviation, ecommerce).")
    parser.add_argument("--min-workers", type=int, default=30)
    parser.add_argument("--min-workers-per-category", type=int, default=3)
    parser.add_argument("--min-l1-categories", type=int, default=5)
    parser.add_argument("--l1-method", default="deep", choices=["deep", "reinforcement"])
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument(
        "--allow-cost-policy-violations",
        action="store_true",
        help="Continue even when STRICT_ZERO_BUDGET policy violations are detected.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base).resolve()

    cost_policy = CostPolicy.from_env()
    violations = cost_policy.validate()
    if violations and not args.allow_cost_policy_violations:
        print(json.dumps({"status": "blocked", "cost_policy_violations": violations}, indent=2))
        return 2
    if violations:
        print(json.dumps({"status": "warning", "cost_policy_violations": violations}, indent=2))

    pipeline = IndustryPipeline(base, industry=args.industry)
    summary = pipeline.run(
        min_workers=args.min_workers,
        min_workers_per_category=args.min_workers_per_category,
        min_l1_categories=args.min_l1_categories,
        l1_method=args.l1_method,
        max_rows=args.max_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
