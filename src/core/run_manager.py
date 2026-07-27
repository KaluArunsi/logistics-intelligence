"""
Run manager for reproducible pipeline runs and reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json

from .logging import get_logger


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    logs_dir: Path
    reports_dir: Path


class RunManager:
    """Creates and manages run directories and summary artifacts."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.logger = get_logger("core.run_manager")

    def start_run(self, industry: str) -> RunContext:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{industry}_{ts}"
        run_dir = self.base_path / "reports" / "runs" / industry / run_id
        logs_dir = run_dir / "logs"
        reports_dir = run_dir / "reports"
        logs_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Started run %s", run_id)
        return RunContext(run_id=run_id, run_dir=run_dir, logs_dir=logs_dir, reports_dir=reports_dir)

    def write_json(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info("Wrote json %s", path)

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        self.logger.info("Wrote text %s", path)
