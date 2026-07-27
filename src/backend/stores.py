"""
In-memory stores for session, task, and report runtime state.
"""

from __future__ import annotations

import hmac
import json
import math
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Optional

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

SESSION_DEFAULT_STATE = "IDLE"


def _validate_store_id(value: str) -> str:
    """Ensure an ID is safe for filesystem use (no path traversal)."""
    clean = str(value or "").strip()
    if not clean or not _SAFE_ID_RE.match(clean):
        raise ValueError(f"Invalid ID: {clean!r}")
    return clean


def _sanitize_floats(obj: Any) -> Any:
    """Replace NaN/Infinity floats with None so JSON serialization succeeds."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _coarse_ua_bucket(user_agent: str) -> str:
    ua = (user_agent or "").lower()
    if "chrome" in ua:
        return "chrome"
    if "safari" in ua and "chrome" not in ua:
        return "safari"
    if "firefox" in ua:
        return "firefox"
    if "edge" in ua:
        return "edge"
    return "other"


def derive_stable_identity(secret: str, ip: str, user_agent: str) -> str:
    """Stable identity hash for cooldown tracking — does NOT include epoch.

    Same IP + UA bucket always produces the same hash regardless of time.
    Used to enforce the 24-hour cooldown across epoch boundaries.
    """
    payload = f"{ip}|{_coarse_ua_bucket(user_agent)}|stable".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()


def derive_ip_hash(secret: str, ip: str, user_agent: str, session_window_hours: int = 1) -> str:
    """Session-window hash — includes hourly epoch for session scoping."""
    epoch = int(utcnow().timestamp() // (session_window_hours * 3600))
    payload = f"{ip}|{_coarse_ua_bucket(user_agent)}|{epoch}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()
    return digest


@dataclass
class SessionRecord:
    ip_hash: str
    created_at: datetime
    expires_at: datetime
    cooldown_until: Optional[datetime] = None
    state: str = SESSION_DEFAULT_STATE
    last_successful_step_uri: Optional[str] = None
    active_task_id: Optional[str] = None
    report_id: Optional[str] = None
    artifact_uris: dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "upload_uri": None,
            "schema_uri": None,
            "match_uri": None,
            "routing_uri": None,
            "tell_us_uri": None,
            "tell_us_conversation_uri": None,
            "questions_uri": None,
            "augment_uri": None,
            "synthetic_meta_uri": None,
        }
    )

    def to_status_payload(self) -> dict[str, Any]:
        return {
            "has_session": True,
            "session_expires_at": iso(self.expires_at),
            "cooldown_until": iso(self.cooldown_until),
            "state": self.state,
            "last_successful_step_uri": self.last_successful_step_uri,
            "active_task_id": self.active_task_id,
            "report_id": self.report_id,
            "artifacts": dict(self.artifact_uris),
        }


class SessionStore:
    def __init__(self, session_ttl_minutes: int = 60, cooldown_hours: int = 24):
        self.session_ttl_minutes = max(10, int(session_ttl_minutes))
        self.cooldown_hours = cooldown_hours
        self._items: dict[str, SessionRecord] = {}
        # Cooldowns keyed by *stable* identity hash (no epoch) so they
        # survive hourly epoch rollovers and actually enforce 24h lockout.
        self._cooldowns: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def _check_stable_cooldown(self, stable_id: Optional[str]) -> Optional[datetime]:
        """Return cooldown_until if the stable identity is still locked."""
        if not stable_id:
            return None
        until = self._cooldowns.get(stable_id)
        if until and utcnow() < until:
            return until
        if until:
            del self._cooldowns[stable_id]
        return None

    def _set_stable_cooldown(self, stable_id: Optional[str], until: datetime) -> None:
        if stable_id:
            self._cooldowns[stable_id] = until

    def _normalize(self, rec: SessionRecord, stable_id: Optional[str] = None) -> SessionRecord:
        now = utcnow()
        # Check stable-identity cooldown first (cross-epoch enforcement).
        stable_until = self._check_stable_cooldown(stable_id)
        if stable_until:
            rec.cooldown_until = stable_until
            rec.state = "SESSION_LOCKED_COOLDOWN"
            return rec
        if rec.cooldown_until and now < rec.cooldown_until:
            rec.state = "SESSION_LOCKED_COOLDOWN"
            return rec
        if now > rec.expires_at:
            cd_until = now + timedelta(hours=self.cooldown_hours)
            rec.cooldown_until = cd_until
            rec.state = "SESSION_LOCKED_COOLDOWN"
            rec.active_task_id = None
            rec.last_successful_step_uri = None
            rec.artifact_uris = {k: None for k in rec.artifact_uris.keys()}
            self._set_stable_cooldown(stable_id, cd_until)
        return rec

    def get(self, ip_hash: str, stable_id: Optional[str] = None) -> Optional[SessionRecord]:
        with self._lock:
            rec = self._items.get(ip_hash)
            if rec is None:
                # Check if a different epoch-hash session exists under cooldown
                # for this stable identity.
                if stable_id and self._check_stable_cooldown(stable_id):
                    now = utcnow()
                    rec = SessionRecord(
                        ip_hash=ip_hash,
                        created_at=now,
                        expires_at=now,
                        cooldown_until=self._cooldowns[stable_id],
                        state="SESSION_LOCKED_COOLDOWN",
                    )
                    return rec
                return None
            rec = self._normalize(rec, stable_id)
            self._items[ip_hash] = rec
            return rec

    def start_or_resume(self, ip_hash: str, stable_id: Optional[str] = None) -> SessionRecord:
        with self._lock:
            # First check stable-identity cooldown (enforced across epoch boundaries).
            stable_until = self._check_stable_cooldown(stable_id)
            if stable_until:
                existing = self._items.get(ip_hash)
                if existing is not None:
                    existing.cooldown_until = stable_until
                    existing.state = "SESSION_LOCKED_COOLDOWN"
                    self._items[ip_hash] = existing
                    return existing
                now = utcnow()
                rec = SessionRecord(
                    ip_hash=ip_hash,
                    created_at=now,
                    expires_at=now,
                    cooldown_until=stable_until,
                    state="SESSION_LOCKED_COOLDOWN",
                )
                self._items[ip_hash] = rec
                return rec

            existing = self._items.get(ip_hash)
            if existing is not None:
                existing = self._normalize(existing, stable_id)
                self._items[ip_hash] = existing
                now = utcnow()
                if existing.cooldown_until and now < existing.cooldown_until:
                    return existing
                if existing.state != "SESSION_LOCKED_COOLDOWN":
                    return existing

            now = utcnow()
            rec = SessionRecord(
                ip_hash=ip_hash,
                created_at=now,
                expires_at=now + timedelta(minutes=self.session_ttl_minutes),
                state="SESSION_ACTIVE",
            )
            self._items[ip_hash] = rec
            return rec

    def update_state(
        self,
        ip_hash: str,
        state: str,
        *,
        step_uri: Optional[str] = None,
        active_task_id: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> Optional[SessionRecord]:
        with self._lock:
            rec = self._items.get(ip_hash)
            if rec is None:
                return None
            rec.state = state
            if step_uri is not None:
                rec.last_successful_step_uri = step_uri
            if active_task_id is not None:
                rec.active_task_id = active_task_id
            if report_id is not None:
                rec.report_id = report_id
            self._items[ip_hash] = rec
            return rec

    def update_artifact(self, ip_hash: str, key: str, uri: str) -> Optional[SessionRecord]:
        with self._lock:
            rec = self._items.get(ip_hash)
            if rec is None:
                return None
            rec.artifact_uris[key] = uri
            self._items[ip_hash] = rec
            return rec

    def purge_expired(self) -> int:
        """Remove sessions whose cooldown has also expired (#41)."""
        now = utcnow()
        stale: list[str] = []
        stale_cooldowns: list[str] = []
        with self._lock:
            for ip_hash, rec in self._items.items():
                # Remove if expired and past cooldown window
                if rec.cooldown_until and now > rec.cooldown_until:
                    stale.append(ip_hash)
                elif now > rec.expires_at + timedelta(hours=self.cooldown_hours + 1):
                    stale.append(ip_hash)
            for key in stale:
                self._items.pop(key, None)
            # Also purge expired stable-identity cooldowns.
            for stable_id, until in self._cooldowns.items():
                if now > until:
                    stale_cooldowns.append(stable_id)
            for key in stale_cooldowns:
                self._cooldowns.pop(key, None)
        return len(stale)


@dataclass
class TaskRecord:
    task_id: str
    industry: Optional[str]
    state: str
    progress: float
    updated_at: datetime
    error: Optional[str] = None
    report_id: Optional[str] = None
    owner_session: Optional[str] = None
    result: Optional[dict[str, Any]] = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "state": self.state,
            "progress": float(self.progress),
            "updated_at": iso(self.updated_at),
            "error": self.error,
            "report_id": self.report_id,
            "industry": self.industry,
        }
        if self.result is not None:
            payload["result"] = self.result
        return payload


_TASK_TERMINAL_STATES = {"REPORT_READY", "ERROR", "INSUFFICIENT_CONTEXT"}
_TASK_PERSIST_TTL_HOURS = 48


class TaskStore:
    def __init__(self, max_workers: int = 2, persist_dir: Optional[Path] = None):
        self._items: dict[str, TaskRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._event_seq = 0
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="li-task")
        self._persist_dir: Optional[Path] = Path(persist_dir) if persist_dir else None
        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk persistence helpers
    # ------------------------------------------------------------------

    def _task_path(self, task_id: str) -> Optional[Path]:
        if self._persist_dir is None:
            return None
        return self._persist_dir / f"{task_id}.json"

    def _write_task(self, rec: TaskRecord) -> None:
        path = self._task_path(rec.task_id)
        if path is None:
            return
        try:
            import tempfile as _tmpmod
            data = {
                "task_id": rec.task_id,
                "industry": rec.industry,
                "state": rec.state,
                "progress": rec.progress,
                "updated_at": rec.updated_at.isoformat(),
                "error": rec.error,
                "report_id": rec.report_id,
                "owner_session": rec.owner_session,
            }
            # Atomic write: write to temp file then rename
            import os as _os_mod
            fd, tmp_path = _tmpmod.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".task_"
            )
            closed = False
            try:
                _os_mod.write(fd, json.dumps(data).encode("utf-8"))
                _os_mod.close(fd)
                closed = True
                _os_mod.replace(tmp_path, str(path))
            except Exception:
                if not closed:
                    try:
                        _os_mod.close(fd)
                    except OSError:
                        pass
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except Exception:
            pass  # Never let persistence failure crash the request

    def _load_from_disk(self) -> None:
        if self._persist_dir is None:
            return
        cutoff = utcnow() - timedelta(hours=_TASK_PERSIST_TTL_HOURS)
        for path in self._persist_dir.glob("task_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                updated_at = datetime.fromisoformat(data["updated_at"])
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                # Purge stale tasks from disk
                if updated_at < cutoff:
                    path.unlink(missing_ok=True)
                    continue
                rec = TaskRecord(
                    task_id=data["task_id"],
                    industry=data.get("industry"),
                    state=data["state"],
                    progress=float(data.get("progress", 0.0)),
                    updated_at=updated_at,
                    error=data.get("error"),
                    report_id=data.get("report_id"),
                    owner_session=data.get("owner_session"),
                )
                # Tasks that were in-flight when the process died will never complete.
                # Mark them as ERROR so the frontend gets a clear signal.
                if rec.state not in _TASK_TERMINAL_STATES:
                    rec.state = "ERROR"
                    rec.error = "Server restarted while this task was running. Please try again."
                    rec.progress = 1.0
                    rec.updated_at = utcnow()
                    self._write_task(rec)
                self._items[rec.task_id] = rec
                self._events[rec.task_id] = []
            except Exception:
                pass  # Skip corrupt files silently

    def _purge_old_tasks(self) -> None:
        """Remove tasks older than TTL from memory and disk."""
        cutoff = utcnow() - timedelta(hours=_TASK_PERSIST_TTL_HOURS)
        with self._lock:
            stale = [tid for tid, rec in self._items.items() if rec.updated_at < cutoff]
            for tid in stale:
                self._items.pop(tid, None)
                self._events.pop(tid, None)
                path = self._task_path(tid)
                if path:
                    path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Public API (unchanged signatures)
    # ------------------------------------------------------------------

    def create(self, industry: Optional[str], owner_session: Optional[str]) -> TaskRecord:
        with self._lock:
            task_id = f"task_{uuid.uuid4().hex[:16]}"
            rec = TaskRecord(
                task_id=task_id,
                industry=industry,
                state="PREDICT_ENQUEUED",
                progress=0.0,
                updated_at=utcnow(),
                owner_session=owner_session,
            )
            self._items[task_id] = rec
            self._events[task_id] = []
            self._append_event(rec)
            self._write_task(rec)
            return rec

    def _append_event(self, rec: TaskRecord) -> None:
        self._event_seq += 1
        row = {"seq": self._event_seq, **rec.to_payload()}
        self._events.setdefault(rec.task_id, []).append(row)
        if len(self._events[rec.task_id]) > 100:
            self._events[rec.task_id] = self._events[rec.task_id][-100:]

    def set_state(
        self,
        task_id: str,
        state: str,
        *,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        with self._lock:
            rec = self._items.get(task_id)
            if rec is None:
                return None
            rec.state = state
            if progress is not None:
                rec.progress = max(0.0, min(1.0, float(progress)))
            rec.error = error
            if report_id is not None:
                rec.report_id = report_id
            rec.updated_at = utcnow()
            self._items[task_id] = rec
            self._append_event(rec)
            self._write_task(rec)
            return rec

    def set_result(self, task_id: str, result: dict[str, Any]) -> Optional[TaskRecord]:
        with self._lock:
            rec = self._items.get(task_id)
            if rec is None:
                return None
            rec.result = result
            rec.updated_at = utcnow()
            self._items[task_id] = rec
            self._append_event(rec)
            self._write_task(rec)
            return rec

    def get(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            return self._items.get(task_id)

    def get_events(self, task_id: str, since: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._events.get(task_id, [])
            return [row for row in rows if int(row.get("seq", 0)) > since]

    def submit(self, task_id: str, runner: Callable[[], None]) -> None:
        def _wrapped() -> None:
            try:
                runner()
            except Exception as exc:
                self.set_state(task_id, "ERROR", progress=1.0, error=str(exc))

        self.executor.submit(_wrapped)


class ReportStore:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.reports_dir = self.base_path / "exports" / "runtime_reports"
        self.debug_dir = self.base_path / "exports" / "runtime_debug"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def new_report_id(self) -> str:
        return f"rep_{uuid.uuid4().hex[:16]}"

    def report_path(self, report_id: str) -> Path:
        return self.reports_dir / f"{_validate_store_id(report_id)}.json"

    def write_report(self, payload: dict[str, Any], report_id: Optional[str] = None) -> str:
        report_id = _validate_store_id(report_id or self.new_report_id())
        out = self.reports_dir / f"{report_id}.json"
        out.write_text(json.dumps(_sanitize_floats(payload), indent=2) + "\n")
        return report_id

    def read_report(self, report_id: str) -> Optional[dict[str, Any]]:
        path = self.reports_dir / f"{_validate_store_id(report_id)}.json"
        if not path.exists():
            return None
        return _sanitize_floats(json.loads(path.read_text()))

    def update_report(self, report_id: str, payload: dict[str, Any]) -> bool:
        path = self.report_path(report_id)
        if not path.exists():
            return False
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return True

    def write_debug_bundle(self, task_id: str, payload: dict[str, Any]) -> str:
        task_id = _validate_store_id(task_id)
        data = dict(payload)
        data["expires_at"] = iso(utcnow() + timedelta(hours=48))
        out = self.debug_dir / f"{task_id}.json"
        out.write_text(json.dumps(data, indent=2) + "\n")
        return str(out)


class ChatStore:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.chat_dir = self.base_path / "exports" / "runtime_chats"
        self.summary_dir = self.chat_dir / "summaries"
        self.training_corpus_path = self.chat_dir / "training_corpus.jsonl"
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def chat_path(self, report_id: str) -> Path:
        return self.chat_dir / f"{_validate_store_id(report_id)}.jsonl"

    def summary_path(self, report_id: str) -> Path:
        return self.summary_dir / f"{_validate_store_id(report_id)}.json"

    def append(self, report_id: str, role: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        row = {
            "ts": iso(utcnow()),
            "role": role,
            "content": content,
            "metadata": metadata or {},
        }
        path = self.chat_path(report_id)
        with open(path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        self.append_global(
            conversation_id=report_id,
            role=role,
            content=content,
            metadata={"channel": "report_chat", **(metadata or {})},
        )

    def append_global(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        row = {
            "ts": iso(utcnow()),
            "conversation_id": str(conversation_id or "unknown"),
            "role": str(role or "assistant"),
            "content": str(content or ""),
            "metadata": metadata or {},
        }
        with open(self.training_corpus_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    def history(self, report_id: str, limit: int = 50) -> list[dict[str, Any]]:
        path = self.chat_path(report_id)
        if not path.exists():
            return []
        rows = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if limit <= 0:
            return rows
        return rows[-limit:]

    def write_summary(self, report_id: str, payload: dict[str, Any]) -> str:
        path = self.summary_path(report_id)
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return str(path)

    def read_summary(self, report_id: str) -> Optional[dict[str, Any]]:
        path = self.summary_path(report_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
