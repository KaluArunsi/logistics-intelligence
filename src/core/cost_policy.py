"""Runtime cost policy helpers for zero-budget deployments."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CostPolicy:
    """Cost controls for deployment/runtime integrations."""

    llm_provider_mode: str
    allow_paid_fallback: bool
    allow_paid_llm: bool
    allow_paid_datastores: bool
    allow_paid_observability: bool
    strict_zero_budget: bool

    @classmethod
    def from_env(cls) -> "CostPolicy":
        return cls(
            llm_provider_mode=os.getenv("LLM_PROVIDER_MODE", "oss_only").strip().lower(),
            allow_paid_fallback=_env_bool("ALLOW_PAID_FALLBACK", default=False),
            allow_paid_llm=_env_bool("ALLOW_PAID_LLM", default=False),
            allow_paid_datastores=_env_bool("ALLOW_PAID_DATASTORES", default=False),
            allow_paid_observability=_env_bool("ALLOW_PAID_OBSERVABILITY", default=False),
            strict_zero_budget=_env_bool("STRICT_ZERO_BUDGET", default=True),
        )

    def validate(self) -> list[str]:
        """
        Return validation violations.

        In strict mode, any paid toggle enabled is considered a violation.
        """
        violations: list[str] = []
        if self.llm_provider_mode not in {"oss_only", "hybrid", "paid_only"}:
            violations.append(f"Invalid LLM_PROVIDER_MODE='{self.llm_provider_mode}'")

        if not self.strict_zero_budget:
            return violations

        if self.allow_paid_fallback:
            violations.append("ALLOW_PAID_FALLBACK=true while STRICT_ZERO_BUDGET=true")
        if self.allow_paid_llm:
            violations.append("ALLOW_PAID_LLM=true while STRICT_ZERO_BUDGET=true")
        if self.allow_paid_datastores:
            violations.append("ALLOW_PAID_DATASTORES=true while STRICT_ZERO_BUDGET=true")
        if self.allow_paid_observability:
            violations.append("ALLOW_PAID_OBSERVABILITY=true while STRICT_ZERO_BUDGET=true")
        if self.llm_provider_mode == "paid_only":
            violations.append("LLM_PROVIDER_MODE=paid_only while STRICT_ZERO_BUDGET=true")
        return violations

