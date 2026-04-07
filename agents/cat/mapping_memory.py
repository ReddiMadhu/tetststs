"""
mapping_memory.py — Persistent, human-feedback-driven column mapping memory.

How it works:
  - Every time a user confirms a column mapping (/confirm-columns), all
    confirmed source_col → canonical pairs are saved into mapping_memory.json.
  - On the next upload, suggest_columns() checks memory FIRST (before fuzzy /
    LLM) and returns score=1.0 / method="memory" for any known source column.
  - Users can "forget" a bad memory via the DELETE /mapping-memory endpoint.

Memory key: normalize(source_col) + "::" + target_format
  e.g.  "building value::AIR"  →  "BuildingValue"
        "bldg class::RMS"       →  "BLDGCLASS"

Storage: bk/mapping_memory.json  (flat JSON, hand-editable)
"""

import json
import logging
import pathlib
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger("mapping_memory")

# ── Storage path (same directory as this file) ─────────────────────────────────
_MEMORY_FILE = pathlib.Path(__file__).parent / "mapping_memory.json"
_lock = threading.Lock()


# ── Normalisation helper ────────────────────────────────────────────────────────

def _normalize(col: str) -> str:
    """Lowercase, strip, collapse inner whitespace/underscores/hyphens."""
    col = col.lower().strip()
    col = re.sub(r"[\s_\-]+", " ", col)
    return col


def _make_key(source_col: str, target_format: str) -> str:
    return f"{_normalize(source_col)}::{target_format.upper()}"


# ── File I/O ────────────────────────────────────────────────────────────────────

def _load_raw() -> Dict[str, dict]:
    """Load the raw JSON dict from disk. Returns {} on first run or error."""
    if not _MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"mapping_memory: could not load {_MEMORY_FILE}: {exc}")
        return {}


def _save_raw(data: Dict[str, dict]) -> None:
    """Persist the full memory dict to disk atomically."""
    try:
        tmp = _MEMORY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_MEMORY_FILE)
    except Exception as exc:
        logger.error(f"mapping_memory: could not save {_MEMORY_FILE}: {exc}")


# ── Public API ──────────────────────────────────────────────────────────────────

def lookup_memory(
    source_cols: List[str],
    target_format: str,
) -> Dict[str, dict]:
    """
    For each source column that exists in memory, return a suggestion dict:
      { col: { "canonical": str, "score": 1.0, "method": "memory",
               "count": int, "last_confirmed": str } }
    Columns NOT in memory are omitted — they will fall through to fuzzy/LLM.
    """
    with _lock:
        data = _load_raw()

    hits: Dict[str, dict] = {}
    for col in source_cols:
        key = _make_key(col, target_format)
        entry = data.get(key)
        if entry and entry.get("canonical"):
            hits[col] = {
                "canonical": entry["canonical"],
                "score": 1.0,
                "method": "memory",
                "reason": (
                    f"Learned from {entry.get('confirmed_count', 1)} "
                    f"confirmed session(s)"
                ),
                "count": entry.get("confirmed_count", 1),
                "last_confirmed": entry.get("last_confirmed", ""),
            }
    return hits


def record_confirmed(
    column_map: Dict[str, Optional[str]],
    target_format: str,
) -> None:
    """
    Called after /confirm-columns succeeds.
    Persists every confirmed source→canonical pair (skips unmapped / None values).
    If the key already exists, increment confirmed_count and update canonical
    (user's latest manual override is always ground truth).
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with _lock:
        data = _load_raw()
        changed = False

        for src_col, canonical in column_map.items():
            if not canonical:           # skip unmapped columns
                continue
            key = _make_key(src_col, target_format)
            existing = data.get(key)

            if existing:
                existing["canonical"] = canonical           # always update to latest
                existing["confirmed_count"] = existing.get("confirmed_count", 1) + 1
                existing["last_confirmed"] = now_iso
            else:
                data[key] = {
                    "source_col_normalized": _normalize(src_col),
                    "target_format": target_format.upper(),
                    "canonical": canonical,
                    "confirmed_count": 1,
                    "first_confirmed": now_iso,
                    "last_confirmed": now_iso,
                }
            changed = True

        if changed:
            _save_raw(data)
            logger.info(
                f"mapping_memory: recorded {len([v for v in column_map.values() if v])} "
                f"pairs for target={target_format}"
            )


def forget_mapping(source_col: str, target_format: str) -> bool:
    """
    Remove a single memory entry.
    Returns True if the entry existed and was deleted, False if not found.
    """
    key = _make_key(source_col, target_format)
    with _lock:
        data = _load_raw()
        if key not in data:
            return False
        del data[key]
        _save_raw(data)
        logger.info(f"mapping_memory: forgot '{key}'")
        return True


def list_memory(target_format: Optional[str] = None) -> List[dict]:
    """
    Return all memory entries, optionally filtered by target_format.
    Each entry has: source_col_normalized, target_format, canonical,
                    confirmed_count, first_confirmed, last_confirmed.
    Sorted by confirmed_count descending (most-trusted first).
    """
    with _lock:
        data = _load_raw()

    entries = list(data.values())
    if target_format:
        entries = [e for e in entries if e.get("target_format", "").upper() == target_format.upper()]

    entries.sort(key=lambda e: e.get("confirmed_count", 0), reverse=True)
    return entries


def memory_stats() -> dict:
    """Return a quick summary dict."""
    with _lock:
        data = _load_raw()
    total = len(data)
    air = sum(1 for e in data.values() if e.get("target_format") == "AIR")
    rms = sum(1 for e in data.values() if e.get("target_format") == "RMS")
    return {"total": total, "AIR": air, "RMS": rms}
