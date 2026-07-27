"""Dataset spec loader and helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .logging import get_logger


@dataclass
class DatasetSpec:
    dataset_id: str
    category: str
    categories: list[str]
    targets: list[dict]
    features: dict
    sampling: dict


class DatasetSpecStore:
    """Loads dataset specs for an industry."""

    def __init__(self, base_path: Path, industry: str):
        self.base_path = Path(base_path)
        self.industry = industry
        self.logger = get_logger("core.dataset_specs")
        self.spec_path = self.base_path / "config" / "industries" / industry / f"{industry}_dataset_specs.json"
        self.specs = {}
        self._load()

    def _load(self):
        if not self.spec_path.exists():
            self.specs = {}
            return
        with open(self.spec_path) as f:
            data = json.load(f)
        self.specs = data.get("datasets", {})
        self.logger.info("Loaded %s dataset specs from %s", len(self.specs), self.spec_path)

    def get(self, dataset_id: str) -> Optional[dict]:
        return self.specs.get(dataset_id)
