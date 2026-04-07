"""
session.py — In-memory session store protected by a threading.Lock.
Sessions expire automatically via a background daemon thread.
"""
import threading
import time
import uuid
import os
from typing import Dict, Any, Optional

_store: Dict[str, dict] = {}
_lock = threading.Lock()

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_MINUTES", "30")) * 60


# ── Public API ────────────────────────────────────────────────────────────────

def create_session(data: dict) -> str:
    """Create a new session, return its upload_id."""
    upload_id = str(uuid.uuid4())
    with _lock:
        _store[upload_id] = {
            "upload_id": upload_id,
            "created_at": time.time(),
            "stages_complete": {
                "upload": True,
                "column_map": False,
                "geocoding": False,
                "code_mapping": False,
                "normalization": False,
            },
            "flags": [],
            **data,
        }
    return upload_id


def get_session(upload_id: str) -> Optional[dict]:
    """Return the session dict, or None if not found."""
    with _lock:
        return _store.get(upload_id)


def require_session(upload_id: str) -> dict:
    """Return the session or raise a KeyError (to be caught as 404)."""
    session = get_session(upload_id)
    if session is None:
        raise KeyError(f"Session {upload_id!r} not found or expired")
    return session


def update_session(upload_id: str, updates: dict) -> bool:
    """Merge updates into an existing session. Returns False if not found."""
    with _lock:
        if upload_id not in _store:
            return False
        _store[upload_id].update(updates)
        return True


def patch_session_field(upload_id: str, field: str, value: Any) -> bool:
    """Set a single top-level field in a session."""
    return update_session(upload_id, {field: value})


def session_mark_stage(upload_id: str, stage: str) -> None:
    """Mark a pipeline stage as complete."""
    with _lock:
        if upload_id in _store:
            _store[upload_id]["stages_complete"][stage] = True


def append_flag(upload_id: str, flag: dict) -> None:
    """Append a single flag entry to the session's flag list."""
    with _lock:
        if upload_id in _store:
            _store[upload_id].setdefault("flags", []).append(flag)


def append_flags(upload_id: str, flags: list) -> None:
    """Append multiple flag entries."""
    with _lock:
        if upload_id in _store:
            _store[upload_id].setdefault("flags", []).extend(flags)


def remove_flag(upload_id: str, row_index: int, field: str) -> None:
    """Remove a flag entry matching (row_index, field)."""
    with _lock:
        if upload_id in _store:
            _store[upload_id]["flags"] = [
                f for f in _store[upload_id].get("flags", [])
                if not (f.get("row_index") == row_index and f.get("field") == field)
            ]


def delete_session(upload_id: str) -> bool:
    with _lock:
        if upload_id in _store:
            del _store[upload_id]
            return True
        return False


def list_sessions() -> list:
    """Return a summary list of all active sessions (for debug/admin)."""
    with _lock:
        return [
            {
                "upload_id": sid,
                "created_at": data["created_at"],
                "row_count": len(data.get("raw_rows", [])),
                "stages": data.get("stages_complete", {}),
            }
            for sid, data in _store.items()
        ]


# ── TTL Cleanup ───────────────────────────────────────────────────────────────

def start_ttl_cleanup(ttl_seconds: int = SESSION_TTL_SECONDS, interval_seconds: int = 300) -> None:
    """Start a background daemon thread that evicts expired sessions every `interval` seconds."""
    def _cleanup_loop():
        while True:
            time.sleep(interval_seconds)
            now = time.time()
            with _lock:
                expired_ids = [
                    sid for sid, data in _store.items()
                    if now - data.get("created_at", 0) > ttl_seconds
                ]
                for sid in expired_ids:
                    del _store[sid]
            if expired_ids:
                import logging
                logging.getLogger("session").info(
                    f"TTL cleanup: evicted {len(expired_ids)} expired session(s)"
                )

    t = threading.Thread(target=_cleanup_loop, daemon=True, name="session-ttl-cleanup")
    t.start()
