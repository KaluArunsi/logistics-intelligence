"""
Benchmark Validation

Validates model performance against industry benchmarks.
Enforces the all-pass gate for L0 -> L1 progression.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    """Result of benchmark validation."""
    passed: bool
    metric_name: str
    actual: float
    threshold: float
    tier: str  # minimum, good, excellent


class BenchmarkValidator:
    """Validates model metrics against configured benchmarks."""

    def __init__(self, config_path: Optional[Path] = None, industry: Optional[str] = None):
        if config_path is None:
            base = Path(__file__).parent.parent.parent
            industry_config = None
            if industry:
                industry_config = base / "config" / "industries" / industry / "benchmarks.yaml"
            config_path = industry_config if industry_config and industry_config.exists() else (base / "config" / "benchmarks.yaml")

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def validate_classification(
        self,
        accuracy: float,
        f1_score: Optional[float] = None,
        mcc: Optional[float] = None,
        roc_auc: Optional[float] = None,
        tier: str = "minimum",
        primary_metric: Optional[str] = None
    ) -> list[BenchmarkResult]:
        """
        Validate classification metrics against benchmarks.

        For imbalanced datasets, F1/MCC are more reliable than accuracy.
        The primary_metric determines pass/fail status.
        """
        thresholds = self.config["classification"][tier]
        results = []

        # Determine which metric is primary (defaults to config setting)
        if primary_metric is None:
            primary_metric = self.config["classification"].get("primary_metric", "accuracy")

        def metric_label(name: str) -> str:
            return f"{name} (primary)" if name == primary_metric else name

        # Accuracy (always computed but may not be primary)
        results.append(BenchmarkResult(
            passed=accuracy >= thresholds["accuracy"],
            metric_name=metric_label("accuracy"),
            actual=accuracy,
            threshold=thresholds["accuracy"],
            tier=tier
        ))

        # F1 Score (preferred for imbalanced data)
        if f1_score is not None:
            results.append(BenchmarkResult(
                passed=f1_score >= thresholds["f1_score"],
                metric_name=metric_label("f1_score"),
                actual=f1_score,
                threshold=thresholds["f1_score"],
                tier=tier
            ))

        # Matthews Correlation Coefficient (best for imbalanced binary/multiclass)
        if mcc is not None and "mcc" in thresholds:
            results.append(BenchmarkResult(
                passed=mcc >= thresholds["mcc"],
                metric_name=metric_label("mcc"),
                actual=mcc,
                threshold=thresholds["mcc"],
                tier=tier
            ))

        if roc_auc is not None and "roc_auc" in thresholds:
            results.append(BenchmarkResult(
                passed=roc_auc >= thresholds["roc_auc"],
                metric_name=metric_label("roc_auc"),
                actual=roc_auc,
                threshold=thresholds["roc_auc"],
                tier=tier
            ))

        return results

    def validate_regression(
        self,
        r2: float,
        rmse_normalized: Optional[float] = None,
        mae_normalized: Optional[float] = None,
        tier: str = "minimum"
    ) -> list[BenchmarkResult]:
        """Validate regression metrics against benchmarks."""
        thresholds = self.config["regression"][tier]
        results = []

        results.append(BenchmarkResult(
            passed=r2 >= thresholds["r2"],
            metric_name="r2",
            actual=r2,
            threshold=thresholds["r2"],
            tier=tier
        ))

        if rmse_normalized is not None:
            # For RMSE, lower is better
            results.append(BenchmarkResult(
                passed=rmse_normalized <= thresholds["rmse_normalized"],
                metric_name="rmse_normalized",
                actual=rmse_normalized,
                threshold=thresholds["rmse_normalized"],
                tier=tier
            ))

        if mae_normalized is not None:
            # For MAE, lower is better
            results.append(BenchmarkResult(
                passed=mae_normalized <= thresholds["mae_normalized"],
                metric_name="mae_normalized",
                actual=mae_normalized,
                threshold=thresholds["mae_normalized"],
                tier=tier
            ))

        return results

    def validate_domain(
        self,
        domain: str,
        metrics: dict
    ) -> list[BenchmarkResult]:
        """Validate domain-specific benchmarks."""
        if domain not in self.config.get("domain", {}):
            return []

        thresholds = self.config["domain"][domain]
        results = []

        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                actual = metrics[metric_name]
                # Determine if higher or lower is better
                lower_is_better = metric_name in ["mape", "rmse", "mae", "rul_rmse_cycles"]
                if lower_is_better:
                    passed = actual <= threshold
                else:
                    passed = actual >= threshold

                results.append(BenchmarkResult(
                    passed=passed,
                    metric_name=metric_name,
                    actual=actual,
                    threshold=threshold,
                    tier="domain"
                ))

        return results

    def validate_model(
        self,
        task_type: str,
        metrics: dict,
        tier: str = "minimum",
        domain: Optional[str] = None
    ) -> tuple[bool, list[BenchmarkResult]]:
        """
        Validate a model's metrics against benchmarks.

        Returns (passed, results) where passed is True only if ALL benchmarks pass.
        """
        results = []

        if task_type == "classification":
            results.extend(self.validate_classification(
                accuracy=metrics.get("accuracy", 0),
                f1_score=metrics.get("f1_score"),
                mcc=metrics.get("mcc"),
                roc_auc=metrics.get("roc_auc"),
                tier=tier
            ))
        elif task_type == "regression":
            results.extend(self.validate_regression(
                r2=metrics.get("r2", 0),
                rmse_normalized=metrics.get("rmse_normalized"),
                mae_normalized=metrics.get("mae_normalized"),
                tier=tier
            ))

        if domain:
            results.extend(self.validate_domain(domain, metrics))

        # Pass/fail determined by primary metric only (marked with "(primary)" in name)
        primary_results = [r for r in results if "(primary)" in r.metric_name]
        if primary_results:
            passed = all(r.passed for r in primary_results)
        else:
            # Fallback: all metrics must pass if no primary specified
            passed = all(r.passed for r in results)
        return passed, results

    def format_results(self, results: list[BenchmarkResult]) -> str:
        """Format benchmark results for display."""
        lines = []
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{status}] {r.metric_name}: {r.actual:.4f} (threshold: {r.threshold:.4f})")
        return "\n".join(lines)
