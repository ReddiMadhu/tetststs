"""
secondary_modifier_mapper.py
----------------------------
Lightweight standalone mapper for AIR and RMS secondary modifier fields.

Field breakdown by vendor:
  AIR (Touchstone):
    - roof_cover      (codes 0-11)  â€” AIR RoofCover field
    - wall_type       (codes 0-9)   â€” AIR WallType / WallSiding field
    - foundation_type (codes 0-12)  â€” shared with RMS, identical codes
    - soft_story      (codes 0-2)   â€” shared with RMS, identical codes

  RMS (Touchstone NA):
    - rms_roofsys     (codes 0-9)   â€” RMS Roofsys field (different scale than AIR)
    - rms_cladsys     (codes 0-10)  â€” RMS CLADSYS cladding field (different schema than AIR)
    - foundation_type (codes 0-12)  â€” shared with AIR, identical codes
    - soft_story      (codes 0-2)   â€” shared with AIR, identical codes

IMPORTANT: wall_type and rms_cladsys are NOT interchangeable.
           roof_cover and rms_roofsys are NOT interchangeable.
           Always use the correct field for the pipeline target (AIR vs RMS).

Usage:
    from agents.cat.secondary_modifier_mapper import SecondaryModifierMapper

    mapper = SecondaryModifierMapper()

    # AIR fields
    code = mapper.map_roof_cover("clay tile")           # -> 3
    code = mapper.map_wall_type("plywood")              # -> 3
    code = mapper.map_foundation_type("slab")           # -> 8
    code = mapper.map_soft_story("yes")                 # -> 2

    # RMS fields
    code = mapper.map_rms_roofsys("composition shingles")  # -> 8
    code = mapper.map_rms_cladsys("brick veneer")           # -> 1

    # All AIR fields at once from a row dict
    result = mapper.map_all({
        "roof_cover":      "TPO membrane",
        "wall_type":       "cast in place concrete",
        "foundation_type": "pile",
        "soft_story":      "no",
    })
    # -> {"roof_cover": 7, "wall_type": 8, "foundation_type": 9, "soft_story": 1,
    #     "roof_cover_desc": "Single ply membrane", ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------

_REF_PATH = Path(__file__).parent.parent.parent / "reference" / "secondary_modifiers.json"

def _load_ref() -> Dict[str, Any]:
    with open(_REF_PATH, encoding="utf-8") as f:
        return json.load(f)


_REF_DATA: Dict[str, Any] = {}  # loaded lazily on first use


def _ref() -> Dict[str, Any]:
    global _REF_DATA
    if not _REF_DATA:
        _REF_DATA = _load_ref()
    return _REF_DATA


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize(raw: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation except / and -."""
    s = raw.strip().lower()
    s = re.sub(r"[^\w\s/-]", " ", s)   # keep word chars, /, and -
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_valid_int(val: str, lo: int, hi: int) -> Optional[int]:
    """Return int if val represents an integer in [lo, hi], else None. Handles cases like '1.0' from pandas NaN float casting."""
    try:
        f = float(val.strip())
        if f.is_integer():
            i = int(f)
            if lo <= i <= hi:
                return i
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Core lookup logic (shared across all modifier types)
# ---------------------------------------------------------------------------

def _lookup(
    raw: str,
    field: str,
    use_llm: bool = True,
    llm_client=None,
) -> Dict[str, Any]:
    """
    3-stage lookup for a single modifier field.

    Stage 1: Integer pass-through â€” already a valid code, return immediately.
    Stage 2: Exact alias match â€” O(1) dict lookup after normalization.
    Stage 3: Keyword token scan â€” longest-match substring search.
    Stage 4: LLM fallback â€” only for roof_cover and foundation_type (most
             ambiguous). Returns 0 (Unknown) if LLM unavailable or disabled.

    Returns dict with keys: code (int), description (str), method (str),
                             confidence (float), original (str).
    """
    spec = _ref()[field]
    codes     = spec["codes"]
    lo, hi    = spec["valid_range"]
    aliases   = spec["aliases"]
    keywords  = spec["keyword_tokens"]

    original = raw
    normalized = _normalize(raw)

    # â”€â”€ Stage 1: Integer pass-through â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    maybe_int = _is_valid_int(original, lo, hi)
    if maybe_int is not None:
        return {
            "code":        maybe_int,
            "description": codes[str(maybe_int)],
            "method":      "integer",
            "confidence":  1.0,
            "original":    original,
        }

    # â”€â”€ Stage 2: Exact alias lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if normalized in aliases:
        code = aliases[normalized]
        return {
            "code":        code,
            "description": codes[str(code)],
            "method":      "alias",
            "confidence":  0.97,
            "original":    original,
        }

    # â”€â”€ Stage 3: Keyword token scan (longest match wins) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    best_code   = None
    best_len    = 0
    for token, code in keywords.items():
        if token in normalized and len(token) > best_len:
            best_code = code
            best_len  = len(token)

    if best_code is not None:
        return {
            "code":        best_code,
            "description": codes[str(best_code)],
            "method":      "keyword",
            "confidence":  0.85,
            "original":    original,
        }

    # â”€â”€ Stage 4: LLM fallback (ambiguous fields only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _llm_eligible_fields = {"roof_cover", "foundation_type", "rms_roofsys", "rms_cladsys"}
    if use_llm and field in _llm_eligible_fields and llm_client:
        llm_code = _llm_classify(normalized, field, codes, llm_client)
        if llm_code is not None:
            return {
                "code":        llm_code,
                "description": codes[str(llm_code)],
                "method":      "llm",
                "confidence":  0.80,
                "original":    original,
            }

    # â”€â”€ Default: Unknown (0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    logger.debug(f"[{field}] No match for '{original}' â€” defaulting to 0 (Unknown)")
    return {
        "code":        0,
        "description": codes["0"],
        "method":      "default",
        "confidence":  0.0,
        "original":    original,
    }


def _llm_classify(
    normalized: str,
    field: str,
    codes: Dict[str, str],
    llm_client,
) -> Optional[int]:
    """
    Minimal LLM call â€” returns a single integer code or None on failure.
    Uses plain text output (not JSON) to minimise latency and hallucination.
    """
    code_list = "\n".join(f"  {k}: {v}" for k, v in codes.items())
    field_label = field.replace("_", " ")
    prompt = (
        f"You are an expert at classifying building characteristics.\n"
        f"Map the following {field_label} description to exactly one integer from the list below.\n\n"
        f"Valid codes:\n{code_list}\n\n"
        f"Input: \"{normalized}\"\n\n"
        f"Return ONLY a single integer. No text, no explanation."
    )
    try:
        response = llm_client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config={"temperature": 0, "max_output_tokens": 8},
        )
        text = response.text.strip()
        val = int(re.sub(r"\D", "", text))
        lo, hi = _ref()[field]["valid_range"]
        if lo <= val <= hi:
            return val
    except Exception as e:
        logger.warning(f"LLM fallback failed for [{field}] '{normalized}': {e}")
    return None


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class SecondaryModifierMapper:
    """
    Lightweight mapper for AIR and RMS secondary modifier fields.

    AIR-specific fields (use for AIR target):
      roof_cover (0-11), wall_type (0-9)

    RMS-specific fields (use for RMS target):
      rms_roofsys (0-9), rms_cladsys (0-10)

    Shared fields â€” identical codes for both AIR and RMS:
      foundation_type (0-12), soft_story (0-2)

    The mapper normalises raw text â†’ canonical integer in four stages:
      1. Integer pass-through   (fastest â€” already a code)
      2. Exact alias dict       (O(1) lookup, 200+ entries)
      3. Keyword token scan     (longest-match substring)
      4. LLM fallback           (roof_cover & foundation_type only, optional)

    Args:
        use_llm:  Enable LLM stage 4 (default True if GEMINI_API_KEY present).
                  Set False for offline/test usage.
        llm_client: Pre-built google.genai client. Auto-created if None and use_llm=True.
    """

    # AIR-target fields
    AIR_FIELDS = ("roof_cover", "wall_type", "foundation_type", "soft_story")
    # RMS-target fields
    RMS_FIELDS = ("rms_roofsys", "rms_cladsys", "foundation_type", "soft_story")
    # All known fields (union)
    FIELDS = ("roof_cover", "wall_type", "foundation_type", "soft_story",
              "rms_roofsys", "rms_cladsys")

    def __init__(self, use_llm: bool = True, llm_client=None):
        # Warm the reference data cache
        _ref()

        self._llm_client = None
        if use_llm:
            if llm_client:
                self._llm_client = llm_client
            else:
                api_key = os.getenv("GEMINI_API_KEY", "")
                if api_key:
                    try:
                        import google.genai as genai
                        self._llm_client = genai.Client(api_key=api_key)
                        logger.info("SecondaryModifierMapper: LLM stage active.")
                    except ImportError:
                        logger.warning("google-genai not installed. LLM stage disabled.")
                else:
                    logger.debug("GEMINI_API_KEY not set. LLM stage disabled.")

    # â”€â”€ AIR per-field convenience methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def map_roof_cover(self, raw: str) -> int:
        """Map raw roof cover description â†’ AIR RoofCover code 0-11."""
        return self._map("roof_cover", raw)["code"]

    def map_wall_type(self, raw: str) -> int:
        """Map raw wall type description â†’ AIR WallType code 0-9."""
        return self._map("wall_type", raw)["code"]

    def map_foundation_type(self, raw: str) -> int:
        """Map raw foundation type description â†’ code 0-12 (shared AIR/RMS)."""
        return self._map("foundation_type", raw)["code"]

    def map_soft_story(self, raw: str) -> int:
        """Map raw soft story indicator â†’ code 0-2 (shared AIR/RMS)."""
        return self._map("soft_story", raw)["code"]

    # â”€â”€ RMS per-field convenience methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def map_rms_roofsys(self, raw: str) -> int:
        """Map raw roof description â†’ RMS Roofsys code 0-9 (NOT same as AIR roof_cover)."""
        return self._map("rms_roofsys", raw)["code"]

    def map_rms_cladsys(self, raw: str) -> int:
        """Map raw cladding description â†’ RMS CLADSYS code 0-10 (NOT same as AIR wall_type)."""
        return self._map("rms_cladsys", raw)["code"]

    # â”€â”€ Full detail (code + description + confidence + method) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def map_roof_cover_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("roof_cover", raw)

    def map_wall_type_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("wall_type", raw)

    def map_foundation_type_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("foundation_type", raw)

    def map_soft_story_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("soft_story", raw)

    def map_rms_roofsys_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("rms_roofsys", raw)

    def map_rms_cladsys_detail(self, raw: str) -> Dict[str, Any]:
        return self._map("rms_cladsys", raw)

    # â”€â”€ Batch: map all 4 AIR fields from a row dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def map_all(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Map all AIR secondary modifier fields from a row dict.

        Args:
            row: dict with any subset of keys:
                 "roof_cover", "wall_type", "foundation_type", "soft_story"
                 Values can be raw text or already integer strings.

        Returns:
            Flat dict with integer codes AND descriptions:
            {
              "roof_cover": 7,          "roof_cover_desc": "Single ply membrane",
              "wall_type":  8,          "wall_type_desc": "Cast-in-place concrete",
              "foundation_type": 8,     "foundation_type_desc": "Mat / slab",
              "soft_story": 1,          "soft_story_desc": "No",
              "_methods": {"roof_cover": "alias", ...},
              "_confidence": {"roof_cover": 0.97, ...},
            }
        """
        result: Dict[str, Any] = {}
        methods: Dict[str, str] = {}
        confidence: Dict[str, float] = {}

        for field in self.AIR_FIELDS:
            raw_val = str(row.get(field) or "").strip()
            if not raw_val:
                raw_val = "0"  # treat missing as Unknown

            detail = self._map(field, raw_val)
            result[field]             = detail["code"]
            result[f"{field}_desc"]   = detail["description"]
            methods[field]            = detail["method"]
            confidence[field]         = detail["confidence"]

        result["_methods"]    = methods
        result["_confidence"] = confidence
        return result

    def map_all_rms(self, row: Dict[str, str]) -> Dict[str, Any]:
        """
        Map all RMS secondary modifier fields from a row dict.

        Args:
            row: dict with any subset of keys:
                 "roof_cover" (mapped to rms_roofsys),
                 "wall_type"  (mapped to rms_cladsys),
                 "foundation_type", "soft_story"
                 Values can be raw text or already integer strings.

        Returns:
            Flat dict with integer codes AND descriptions (using RMS schemas):
            {
              "rms_roofsys": 8,       "rms_roofsys_desc": "Composition/FG/Asphalt Shingles",
              "rms_cladsys": 1,       "rms_cladsys_desc": "Brick veneer",
              "foundation_type": 8,   "foundation_type_desc": "Mat / slab",
              "soft_story": 1,        "soft_story_desc": "No",
              "_methods": {...}, "_confidence": {...}
            }
        """
        result: Dict[str, Any] = {}
        methods: Dict[str, str] = {}
        confidence: Dict[str, float] = {}

        # Roof: input key is "roof_cover", RMS schema is "rms_roofsys"
        roof_raw = str(row.get("roof_cover") or "").strip() or "0"
        roof_detail = self._map("rms_roofsys", roof_raw)
        result["rms_roofsys"]       = roof_detail["code"]
        result["rms_roofsys_desc"]  = roof_detail["description"]
        methods["rms_roofsys"]      = roof_detail["method"]
        confidence["rms_roofsys"]   = roof_detail["confidence"]

        # Wall: input key is "wall_type", RMS schema is "rms_cladsys"
        wall_raw = str(row.get("wall_type") or "").strip() or "0"
        wall_detail = self._map("rms_cladsys", wall_raw)
        result["rms_cladsys"]       = wall_detail["code"]
        result["rms_cladsys_desc"]  = wall_detail["description"]
        methods["rms_cladsys"]      = wall_detail["method"]
        confidence["rms_cladsys"]   = wall_detail["confidence"]

        # Shared fields
        for field in ("foundation_type", "soft_story"):
            raw_val = str(row.get(field) or "").strip() or "0"
            detail = self._map(field, raw_val)
            result[field]           = detail["code"]
            result[f"{field}_desc"] = detail["description"]
            methods[field]          = detail["method"]
            confidence[field]       = detail["confidence"]

        result["_methods"]    = methods
        result["_confidence"] = confidence
        return result

    # â”€â”€ Code description lookups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def describe(self, field: str, code: int) -> str:
        """Return the human-readable description for a given field + code."""
        codes = _ref()[field]["codes"]
        return codes.get(str(code), f"Unknown code {code}")

    def valid_codes(self, field: str) -> Dict[int, str]:
        """Return all valid codes for a field as {int: description}."""
        codes = _ref()[field]["codes"]
        return {int(k): v for k, v in codes.items()}

    # â”€â”€ Internal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _map(self, field: str, raw: str) -> Dict[str, Any]:
        if field not in self.FIELDS:
            raise ValueError(f"Unknown field '{field}'. Valid: {self.FIELDS}")
        # LLM fallback for RMS fields mirrors AIR equivalents
        _llm_eligible = {"roof_cover", "foundation_type", "rms_roofsys", "rms_cladsys"}
        if not raw or not raw.strip():
            spec = _ref()[field]
            return {
                "code": 0,
                "description": spec["codes"]["0"],
                "method": "empty",
                "confidence": 0.0,
                "original": raw,
            }
        return _lookup(raw, field, use_llm=bool(self._llm_client), llm_client=self._llm_client)

    def __repr__(self) -> str:
        llm = "enabled" if self._llm_client else "disabled"
        return f"SecondaryModifierMapper(llm={llm})"


# ---------------------------------------------------------------------------
# Module-level convenience (singleton for quick scripting use)
# ---------------------------------------------------------------------------

_default_mapper: Optional[SecondaryModifierMapper] = None


def get_mapper(use_llm: bool = True) -> SecondaryModifierMapper:
    """Return a cached (module-level) SecondaryModifierMapper instance."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = SecondaryModifierMapper(use_llm=use_llm)
    return _default_mapper


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    mapper = SecondaryModifierMapper(use_llm=False)

    # â”€â”€ AIR field tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    air_tests = [
        ("roof_cover",      "asphalt shingles"),
        ("roof_cover",      "clay tile"),
        ("roof_cover",      "TPO"),
        ("roof_cover",      "standing seam metal"),
        ("roof_cover",      "3"),            # integer pass-through
        ("wall_type",       "brick"),        # -> 1 Brick/unreinforced masonry
        ("wall_type",       "reinforced masonry"),  # -> 2
        ("wall_type",       "plywood"),      # -> 3
        ("wall_type",       "osb"),          # -> 5 Particle board/OSB
        ("wall_type",       "metal panels"), # -> 6
        ("wall_type",       "cast in place"),# -> 8 Cast-in-place concrete
        ("wall_type",       "gypsum board"), # -> 9
        ("foundation_type", "slab on grade"),
        ("foundation_type", "pile foundation"),
        ("foundation_type", "cripple wall"),
        ("soft_story",      "yes"),
        ("soft_story",      "no"),
        ("soft_story",      "unknown"),
    ]
    print("=== SecondaryModifierMapper smoke test -- AIR fields ===\n")
    for field, raw in air_tests:
        detail = mapper._map(field, raw)
        print(f"{field:20s}  '{raw:30s}'  -> {detail['code']:2d}  "
              f"({detail['description'][:35]})  [{detail['method']}, {detail['confidence']:.2f}]")

    # â”€â”€ RMS field tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rms_tests = [
        ("rms_roofsys", "slate"),               # -> 1
        ("rms_roofsys", "single ply membrane"), # -> 3
        ("rms_roofsys", "composition shingles"),# -> 8
        ("rms_roofsys", "clay tiles"),          # -> 6
        ("rms_roofsys", "wood shingles"),       # -> 7
        ("rms_roofsys", "swr"),                 # -> 9
        ("rms_cladsys", "brick veneer"),        # -> 1
        ("rms_cladsys", "metal sheathing"),     # -> 2
        ("rms_cladsys", "eifs"),                # -> 4
        ("rms_cladsys", "vinyl siding"),        # -> 8
        ("rms_cladsys", "stucco"),              # -> 9
    ]
    print("\n=== SecondaryModifierMapper smoke test -- RMS fields ===\n")
    for field, raw in rms_tests:
        detail = mapper._map(field, raw)
        print(f"{field:20s}  '{raw:30s}'  -> {detail['code']:2d}  "
              f"({detail['description'][:55]})  [{detail['method']}, {detail['confidence']:.2f}]")

    # â”€â”€ map_all() AIR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- map_all() AIR ---")
    air_row = {
        "roof_cover":       "TPO membrane",
        "wall_type":        "cast in place concrete",
        "foundation_type":  "slab",
        "soft_story":       "no",
    }
    result = mapper.map_all(air_row)
    for field in mapper.AIR_FIELDS:
        print(f"  {field}: {result[field]} ({result[field+'_desc']})  [{result['_methods'][field]}]")

    # â”€â”€ map_all_rms() RMS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n--- map_all_rms() RMS ---")
    rms_row = {
        "roof_cover":       "composition shingles",
        "wall_type":        "brick veneer",
        "foundation_type":  "slab",
        "soft_story":       "no",
    }
    rms_result = mapper.map_all_rms(rms_row)
    for field in ("rms_roofsys", "rms_cladsys", "foundation_type", "soft_story"):
        print(f"  {field}: {rms_result[field]} ({rms_result[field+'_desc']})  [{rms_result['_methods'][field]}]")



