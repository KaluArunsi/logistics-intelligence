"""
Retention cleanup for runtime artifacts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable


class RetentionManager:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._last_run = 0.0
        self.interval_seconds = int(os.getenv("RETENTION_SWEEP_INTERVAL_SEC", "900"))
        self.default_ttl_hours = int(os.getenv("RUNTIME_ARTIFACT_TTL_HOURS", "168"))

    def run_if_due(self) -> dict[str, int]:
        now = time.time()
        if (now - self._last_run) < float(self.interval_seconds):
            return {"checked": 0, "deleted": 0}
        self._last_run = now
        return self.cleanup()

    def cleanup(self) -> dict[str, int]:
        checked = 0
        deleted = 0
        paths_with_ttl = [
            (self.base_path / "exports" / "runtime_chats", int(os.getenv("CHAT_TTL_HOURS", "168"))),
            (self.base_path / "exports" / "runtime_reports", int(os.getenv("REPORT_TTL_HOURS", "336"))),
            (self.base_path / "exports" / "runtime_intake", int(os.getenv("INTAKE_TTL_HOURS", "72"))),
            (self.base_path / "exports" / "runtime_state", int(os.getenv("STATE_TTL_HOURS", "72"))),
            (self.base_path / "exports" / "runtime_debug", int(os.getenv("DEBUG_TTL_HOURS", "48"))),
        ]
        for root, ttl in paths_with_ttl:
            c, d = self._cleanup_path(root, ttl_hours=max(1, ttl))
            checked += c
            deleted += d
        return {"checked": checked, "deleted": deleted}

    @staticmethod
    def _iter_files(root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        return [p for p in root.rglob("*") if p.is_file()]

    def _cleanup_path(self, root: Path, *, ttl_hours: int) -> tuple[int, int]:
        checked = 0
        deleted = 0
        cutoff = time.time() - (float(ttl_hours) * 3600.0)
        for path in self._iter_files(root):
            checked += 1
            try:
                mtime = path.stat().st_mtime
                if mtime < cutoff:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except Exception:
                continue
        return checked, deleted
