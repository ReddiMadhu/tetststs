"""
duplicate_detector.py — Duplicate detection for COPE dictionary entries.

Primarily checks for exact code matches against existing dictionary data.
Also detects internal duplicates within a batch of uploaded entries.
"""
from typing import Any, Dict, List
from pydantic import BaseModel
from typing import Literal


class DuplicateMatch(BaseModel):
    existing_code: str
    existing_description: str
    existing_keywords: List[str]
    match_type: Literal["exact_code"]
    similarity_score: float  # 1.0 for exact code match


def find_duplicates(
    new_entries: Dict[str, dict],
    existing_data: dict,
    cope_type: str,
) -> Dict[str, List[DuplicateMatch]]:
    """
    Check new entries against existing dictionary data for exact code matches.

    Args:
        new_entries: Dict of code -> {description, keywords} to check.
        existing_data: The full merged dictionary (base + overrides).
        cope_type: One of 'construction', 'occupancy', 'protection', 'exposure'.

    Returns:
        Dict of new_code -> list of DuplicateMatch objects.
        Only codes with matches are included.
    """
    duplicates: Dict[str, List[DuplicateMatch]] = {}

    # For exposure type, flatten the nested sections to get all codes
    if cope_type == "exposure":
        flat_existing = _flatten_exposure(existing_data)
    else:
        flat_existing = {
            code: meta
            for code, meta in existing_data.items()
            if not code.startswith("_")
        }

    for new_code, new_meta in new_entries.items():
        if new_code.startswith("_"):
            continue

        matches: List[DuplicateMatch] = []

        # Exact code match
        if new_code in flat_existing:
            existing_meta = flat_existing[new_code]
            desc = _extract_description(existing_meta)
            keywords = _extract_keywords(existing_meta)
            matches.append(DuplicateMatch(
                existing_code=new_code,
                existing_description=desc,
                existing_keywords=keywords,
                match_type="exact_code",
                similarity_score=1.0,
            ))

        if matches:
            duplicates[new_code] = matches

    return duplicates


def find_internal_duplicates(
    entries: Dict[str, dict],
) -> Dict[str, List[str]]:
    """
    Detect duplicate codes within a single batch of entries.
    This catches cases where the same code appears multiple times in an upload.

    Since dict keys are unique, this checks if any code normalization
    (stripping, lowering) would cause collisions.

    Returns:
        Dict of normalized_code -> [original_codes] for any collisions.
    """
    seen: Dict[str, List[str]] = {}
    for code in entries:
        if code.startswith("_"):
            continue
        normalized = code.strip().upper()
        seen.setdefault(normalized, []).append(code)

    return {k: v for k, v in seen.items() if len(v) > 1}


def _flatten_exposure(data: dict) -> dict:
    """Flatten exposure nested structure into code -> meta dict."""
    flat = {}
    for section_key, section_data in data.items():
        if section_key.startswith("_"):
            continue
        if not isinstance(section_data, dict):
            continue
        codes = section_data.get("codes", {})
        aliases = section_data.get("aliases", {})
        for code, desc in codes.items():
            matching_aliases = [a for a, c in aliases.items() if str(c) == str(code)]
            flat[code] = {
                "description": desc,
                "keywords": matching_aliases,
                "section": section_key,
            }
    return flat


def _extract_description(meta: Any) -> str:
    """Extract description from various meta formats."""
    if isinstance(meta, dict):
        return meta.get("description", meta.get("iso_label", ""))
    return str(meta)


def _extract_keywords(meta: Any) -> List[str]:
    """Extract keywords list from various meta formats."""
    if isinstance(meta, dict):
        kw = meta.get("keywords", meta.get("aliases", []))
        if isinstance(kw, list):
            return kw
        if isinstance(kw, dict):
            return list(kw.keys())
    return []
