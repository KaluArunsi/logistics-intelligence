"""
Main Pipeline Orchestrator

Coordinates the full training pipeline:
1. Data ingestion and processing
2. Feature engineering
3. L0 worker training with benchmark validation
4. L1 expert training (after L0 passes)
"""

import yaml
from pathlib import Path
from typing import Optional

from .registry import Registry
from .benchmark import BenchmarkValidator
from .logging import get_logger


class Pipeline:
    """
    Main orchestrator for the Logistics Intelligence pipeline.

    Stages:
    1. Data ingestion: Scan, download, process, compress
    2. Taxonomy: Categorize datasets, generate specs
    3. L0 Training: Train worker models, validate benchmarks
    4. L1 Training: Train category experts (requires L0 all-pass)
    """

    def __init__(self, base_path: Optional[Path] = None, industry: str = "ecommerce"):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent

        self.base_path = Path(base_path)
        self.industry = industry
        self.registry = Registry(self.base_path, industry=self.industry)
        self.validator = BenchmarkValidator(industry=self.industry)
        self.logger = get_logger("core.pipeline")

        # Load industry config
        config_path = self.base_path / "config" / "industries" / industry / "industry.yaml"
        if config_path.exists():
            with open(config_path) as f:
                self.industry_config = yaml.safe_load(f)
        else:
            self.industry_config = {}

        # Load category definitions
        categories_path = self.base_path / "config" / "industries" / industry / "categories.yaml"
        if categories_path.exists():
            with open(categories_path) as f:
                self.categories_config = yaml.safe_load(f)
        else:
            self.categories_config = {"categories": {}}

    def status(self) -> dict:
        """Get current pipeline status."""
        summary = self.registry.summary()
        summary["industry"] = self.industry
        summary["l0_gate_passed"] = self.registry.all_l0_passed()
        self.logger.info("Pipeline status: %s", summary)
        return summary

    def can_train_l1(self) -> bool:
        """Check if L1 training is allowed (all L0 must pass)."""
        self.logger.info("Checking if L1 can train")
        return self.registry.all_l0_passed()

    def get_category_for_dataset(self, columns: list[str], name: str) -> str:
        """
        Determine the category for a dataset based on columns and name.

        Uses keyword matching from categories.yaml.
        """
        name_lower = name.lower()
        col_text = " ".join(columns).lower()
        search_text = f"{name_lower} {col_text}"

        best_category = "operations"  # default
        best_score = 0

        for cat_name, cat_config in self.categories_config.get("categories", {}).items():
            keywords = cat_config.get("keywords", [])
            score = sum(1 for kw in keywords if kw in search_text)
            if score > best_score:
                best_score = score
                best_category = cat_name

        return best_category

    def get_task_type(self, category: str) -> str:
        """Get the task type (classification/regression) for a category."""
        cat_config = self.categories_config.get("categories", {}).get(category, {})
        return cat_config.get("task_type", "classification")

    def validate_l0_model(self, model_id: str) -> tuple[bool, str]:
        """
        Validate an L0 model against benchmarks.

        Returns (passed, report).
        """
        model = self.registry.get_model(model_id)
        if not model:
            return False, f"Model {model_id} not found"

        dataset = self.registry.get_dataset(model.dataset_id)
        task_type = dataset.task_type if dataset else "classification"

        passed, results = self.validator.validate_model(
            task_type=task_type,
            metrics=model.metrics,
            tier="minimum"
        )

        report = f"Model: {model_id}\n"
        report += f"Task: {task_type}\n"
        report += self.validator.format_results(results)

        return passed, report

    def print_status(self):
        """Print a formatted status report."""
        status = self.status()
        print(f"\n{'='*50}")
        print(f"Logistics Intelligence Pipeline - {status['industry'].upper()}")
        print(f"{'='*50}")
        print(f"Datasets: {status['processed_datasets']}/{status['total_datasets']} processed")
        print(f"L0 Workers: {status['l0_passed']}/{status['l0_models']} passed benchmarks")
        print(f"L1 Experts: {status['l1_passed']}/{status['l1_models']} passed benchmarks")
        print(f"Categories: {', '.join(status['categories']) if status['categories'] else 'None yet'}")
        print(f"\nL0 Gate: {'PASSED - Ready for L1' if status['l0_gate_passed'] else 'BLOCKED - L0 training incomplete'}")
        print(f"{'='*50}\n")
