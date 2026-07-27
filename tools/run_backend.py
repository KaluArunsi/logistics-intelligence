#!/usr/bin/env python3
"""Run the FastAPI backend for local development."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn


def _resolve_port(port_arg: str | None) -> int:
    """Resolve backend port from CLI argument and environment variables.

    Accepts raw ints ("8000"), env references ("$PORT", "${PORT}"), or blank.
    Falls back to the runtime PORT env var, then 8000.
    """
    raw = (port_arg or "").strip()
    env_port = os.environ.get("PORT", "").strip()

    if raw.startswith("${") and raw.endswith("}"):
        env_key = raw[2:-1].strip()
        raw = os.environ.get(env_key, "").strip()
    if raw.startswith("$"):
        env_key = raw[1:].strip()
        raw = os.environ.get(env_key, "").strip()

    if not raw:
        raw = env_port
    if not raw:
        return 8000

    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid --port value {port_arg!r}; resolved value was {raw!r}") from exc


def _resolve_worker_count(workers_arg: int | None, *, reload_enabled: bool) -> int:
    """Resolve uvicorn worker count with cloud-safe defaults."""
    if reload_enabled:
        return 1
    if workers_arg is not None:
        return max(1, int(workers_arg))
    raw = os.environ.get("UVICORN_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    # Railway/free-tier friendly default; set UVICORN_WORKERS to scale up.
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        default=None,
        help="Port number or env reference (examples: 8000, $PORT, ${PORT}).",
    )
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (ignored when --reload is set).")
    parser.add_argument("--base", default=".")
    args = parser.parse_args()
    try:
        port = _resolve_port(args.port)
    except ValueError as exc:
        parser.error(str(exc))

    base = Path(args.base).resolve()
    # Ensure uvicorn reload subprocess can import `src.backend.app`.
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    existing_py_path = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing_py_path.split(os.pathsep) if p]
    if str(base) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(base)] + parts)

    workers = _resolve_worker_count(args.workers, reload_enabled=args.reload)
    uvicorn_workers = None if args.reload or workers <= 1 else workers

    uvicorn.run(
        "src.backend.app:app",
        host=args.host,
        port=port,
        reload=args.reload,
        reload_dirs=[str(base)] if args.reload else None,
        workers=uvicorn_workers,
        app_dir=str(base),
        log_level="info",
        # Each worker gets its own thread pool for sync route handlers.
        # This prevents blocking LLM/matplotlib calls from starving healthz/session routes.
        timeout_keep_alive=75,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
