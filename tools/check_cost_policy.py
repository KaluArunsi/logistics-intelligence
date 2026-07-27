#!/usr/bin/env python3
"""Validate zero-budget runtime flags."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cost_policy import CostPolicy


def main() -> int:
    policy = CostPolicy.from_env()
    violations = policy.validate()
    payload = {
        "policy": {
            "llm_provider_mode": policy.llm_provider_mode,
            "allow_paid_fallback": policy.allow_paid_fallback,
            "allow_paid_llm": policy.allow_paid_llm,
            "allow_paid_datastores": policy.allow_paid_datastores,
            "allow_paid_observability": policy.allow_paid_observability,
            "strict_zero_budget": policy.strict_zero_budget,
        },
        "violations": violations,
        "status": "ok" if not violations else "violations",
    }
    print(json.dumps(payload, indent=2))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
