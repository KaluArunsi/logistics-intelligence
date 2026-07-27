"""
Security helpers for backend runtime (rate limiting, abuse controls).
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = {}

    def allow(self, key: str, *, limit: int, window_seconds: int) -> tuple[bool, int]:
        """
        Sliding-window limiter.
        Returns (allowed, retry_after_seconds).
        """
        now = time.time()
        cutoff = now - float(window_seconds)
        retry_after = 0

        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= int(limit):
                oldest = bucket[0]
                retry_after = max(1, int((oldest + float(window_seconds)) - now))
                return False, retry_after

            bucket.append(now)
            if len(bucket) == 1:
                # Keep map small over time.
                self._gc(now=now)
            return True, retry_after

    def _gc(self, *, now: float) -> None:
        stale_keys = []
        for key, bucket in self._buckets.items():
            if not bucket:
                stale_keys.append(key)
                continue
            if now - bucket[-1] > 3600:
                stale_keys.append(key)
        for key in stale_keys:
            self._buckets.pop(key, None)
