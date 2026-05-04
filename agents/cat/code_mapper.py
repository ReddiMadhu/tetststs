"""
code_mapper.py â€” 6-stage construction/occupancy code mapping via LangGraph.

Stage 0  : ISO Fire Class fast-path (scheme="ISO" or exact ISO label detected)
Stage 0.5: RMSâ†’AIR cross-translation fast-path (target=AIR, scheme=RMS, exact RMS code)
Stage 1  : Structural conflict resolver (compound descriptions with hybrid structures)
Stage 2  : Priority-weighted deterministic keyword matching (from reference JSON)
Stage 3  : Gemini LLM with response_schema + temperature=0 (structured, hallucination-safe)
Stage 4  : TF-IDF cosine similarity (pre-built vectorizer pkl)
Stage 5  : Business rules default

LangGraph graph: iso_direct â†’ rms_direct â†’ conflict â†’ deterministic â†’ llm â†’ tfidf â†’ default â†’ END
Items only progress to the next stage if their confidence is below the threshold.

LLM Reliability Notes:
  - response_schema enforces valid code enum at sampler level (not just prompt-level)
  - temperature=0, top_p=0.1 for deterministic classification
  - Pydantic model_validate_json as second safety net
  - Retry-with-fallback (max 2 retries â†’ TF-IDF on failure)
  - System instruction includes 8 structural conflict rules
"""
import collections
import json
import logging
import os
import pathlib
import pickle
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics.pairwise import cosine_similarity

from agents.cat.construction_rules import ConflictResolver, ConflictResult
from rules import BusinessRulesConfig

logger = logging.getLogger("code_mapper")

# ── LLM rate limiter (15 calls / 60 s sliding window) ──────────────────────────────────
_RATE_LIMIT_CALLS = 15
_RATE_LIMIT_WINDOW = 60.0   # seconds
_gemini_call_times: collections.deque = collections.deque()


def _rate_limit_gemini() -> None:
    """Block until sending one more Gemini call stays within the rate limit."""
    now = time.monotonic()
    # Drop timestamps older than the window
    while _gemini_call_times and now - _gemini_call_times[0] >= _RATE_LIMIT_WINDOW:
        _gemini_call_times.popleft()

    if len(_gemini_call_times) >= _RATE_LIMIT_CALLS:
        # Oldest call that must expire before we can proceed
        oldest = _gemini_call_times[0]
        sleep_for = _RATE_LIMIT_WINDOW - (now - oldest) + 0.05   # tiny buffer
        if sleep_for > 0:
            logger.info(
                f"Gemini rate limit reached ({_RATE_LIMIT_CALLS} calls/min). "
                f"Sleeping {sleep_for:.1f}sâ€¦"
            )
            time.sleep(sleep_for)
        # Re-prune after sleep
        now = time.monotonic()
        while _gemini_call_times and now - _gemini_call_times[0] >= _RATE_LIMIT_WINDOW:
            _gemini_call_times.popleft()

    _gemini_call_times.append(time.monotonic())


# ── Paths ─────────────────────────────────────────────────────────────────────────────
_BASE = pathlib.Path(__file__).parent  # agents/cat/
_REF_DIR = _BASE.parent.parent / "reference"  # backend/reference/
_TFIDF_DIR = _BASE / "tfidf_cache"  # agents/cat/tfidf_cache/

# ── Module-level singletons (populated at startup) ────────────────────────────────
_occ_codes: Dict[str, Dict] = {}
_const_codes: Dict[str, Dict] = {}
_rms_occ_codes: Dict[str, Dict] = {}
_rms_const_codes: Dict[str, Dict] = {}
_abbreviations: Dict[str, str] = {}

_iso_map: Dict[str, Dict] = {}              # iso_class_key â†’ {air_class, aliases, ...}
_rms_to_air_map: Dict[str, Dict] = {}       # rms_code â†’ {air_class, ...}
_atc_to_air_map: Dict[str, Dict] = {}       # atc_class_str â†’ {air_code, description, ...}
_occ_raw_lookup: Dict[str, Dict] = {}       # normalized_raw_str â†’ {air_code, atc, confidence}
_occ_context_rules: Dict[str, Any] = {}     # term â†’ {default, contexts[]}
_const_raw_lookup: Dict[str, Dict] = {}     # normalized_raw_str â†’ {bldgclass, bldgscheme, final_category, confidence}

_tfidf_indexes: Dict[str, Dict] = {}        # key: "air_occ", "air_const", "rms_occ", "rms_const"

# ── Conflict Resolver singleton ───────────────────────────────────────────────────
_conflict_resolver = ConflictResolver()

# ── ATC / RMS Occupancy scheme detection ──────────────────────────────────────────
# In the actual data (sample rows 2-115):
#   RMS Occ Scheme column = "ATC"  (the scheme label)
#   RMS Occ Code column   = 1-54   (the RMS/ATC numeric class number)
# i.e. scheme="ATC", value="10" â†’ Entertainment (AIR 317)
# "ATC" is the primary label. Any numeric value 1-54 is auto-detected.
# "RMS" as a scheme label is also accepted as a fallback but ONLY for
# occupancy field â€” see _is_atc_scheme() for field-aware logic.
_ATC_EXACT_VALUES = {str(i) for i in list(range(1, 40)) + [42, 43, 44, 47, 48, 49, 50, 51, 52, 53, 54]}
_ATC_SCHEME_NAMES_EXPLICIT = {
    # Primary label seen in actual RMS export data
    "ATC", "ATC_CLASS", "ATC_OCC",
    # Common column name variants from SOV exports
    "OCCTYPE", "OCC_TYPE",
}
# These scheme names are ONLY treated as ATC when field=occupancy
_ATC_SCHEME_NAMES_OCC_ONLY = {
    "RMS", "RMS_OCC", "RMS_OCC_CODE", "RMSATC",
}

# ── ISO auto-detect label set ─────────────────────────────────────────────────────
_ISO_EXACT_LABELS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "F", "JM", "NC", "MNC", "MFR", "FR", "HTJM", "SNC", "SMNC",
}
_ISO_SCHEME_NAMES = {"ISO", "ISO_CLASS", "FIRE_CLASS", "ISF", "ISO_FIRE_CLASS"}

# ── Gemini Structured Output Schemas ──────────────────────────────────────────────

# Dynamic enum types built at startup from loaded code registries
# (defined here typed broadly; exact Literal built in build_tfidf_indexes)
_GeminiCodeEnum = str  # placeholder replaced at startup

class _LLMAlternative(BaseModel):
    code: str = Field(description="Alternative AIR or RMS code")
    confidence: float = Field(ge=0.0, le=0.95, description="Confidence for this alternative")


class _LLMCodeResult(BaseModel):
    """Single item result from the LLM classification."""
    code: str = Field(
        description=(
            "The best-matching construction/occupancy code from the provided list only. "
            "Must be an exact code string from the list â€” never invent codes."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=0.95,
        description="Classification confidence between 0.0 and 0.95. Lower for ambiguous descriptions.",
    )
    reasoning: str = Field(
        description=(
            "One concise sentence: (1) primary structural system identified, "
            "(2) any conflict resolution rule applied, (3) confidence rationale."
        )
    )
    alternatives: List[_LLMAlternative] = Field(
        default_factory=list,
        description="Up to 2 alternative codes. Empty list if only one strong match.",
    )


class _LLMBatchResult(BaseModel):
    """Batch result wrapper â€” one entry per input item."""
    items: List[_LLMCodeResult]


# ── Azure ChatOpenAI model (configured at startup after codes are loaded) ──────────
_llm_model: Optional[AzureChatOpenAI] = None

_LLM_SYSTEM_INSTRUCTION = """You are a licensed insurance structural risk analyst classifying property construction data.

MANDATORY CONFLICT RESOLUTION RULES (apply in this exact priority order):
  RULE 1 â€” URM GOVERNS (WEAKEST): Unreinforced masonry load-bearing walls ALWAYS govern
           regardless of secondary steel roof or other elements. â†’ Joisted Masonry
  RULE 2 â€” MIXED CONSTRUCTION: Two distinct load-bearing structural systems in one building
           OR combustible upper floors over concrete/masonry base â†’ Mixed Construction
  RULE 3 â€” COMBUSTIBLE DOWNGRADE: Wood roof/joists/trusses on masonry walls
           (not just insulation or partition) â†’ Joisted Masonry
  RULE 4 â€” FRAME GOVERNS: Structural frame (steel, concrete, wood, metal) beats
           non-structural elements: facade, veneer, infill, cladding, canopy, foundation
           Example: "Steel frame with masonry infill" â†’ Steel Frame
           Example: "Wood frame with brick siding/veneer" â†’ Frame (wood primary, siding is cladding)
           Example: "Wood framing with reinforced concrete foundation" â†’ Frame (foundation not structural frame)
  RULE 5 â€” TILT-UP GOVERNS: Tilt-up concrete walls + any roof type â†’ Masonry Non-Combustible
  RULE 6 â€” HEAVY TIMBER PRIMARY: Heavy timber structure + masonry secondary â†’ Heavy Timber
  RULE 7 â€” UNKNOWN FRAME: Only facade material known, frame is unknown â†’ Joisted Masonry (conservative)
  RULE 8 â€” INTERIOR PARTITION: Non-structural interior wood office/partition inside metal building
           does NOT trigger downgrade â†’ primary governs

CONSTRUCTION-SPECIFIC DISAMBIGUATION RULES:
  CONST-1: "Frame" alone (no steel/concrete qualifier) â†’ Wood Frame (RMS 1)
           Wood/residential context always assumed when no explicit material stated
  CONST-2: "Non-Combustible" / "NC" / "Non Comb" alone â†’ ISO Fire Class 3 (FIRE/BLDGSCHEME=FIRE, BLDGCLASS=3)
           Do NOT map to Masonry â€” Non-Combustible = Steel/Metal frame per ISO convention
  CONST-3: "Wood Frame + [brick/masonry] Siding/Veneer/Cladding" â†’ Frame/Wood (Rule 4: cladding not structural)
  CONST-4: "Wood Frame + brick walls" (structural masonry walls) â†’ Joisted Masonry
  CONST-5: "Concrete podium" or "Frame over podium" â†’ primary frame type governs; podium = foundation element
           Example: "Frame over 2-story concrete podium" â†’ Frame/Wood (RMS 1)
  CONST-6: "Fire Resistive" / "FR" â†’ FIRE class 6 (BLDGSCHEME=FIRE, BLDGCLASS=6)
  CONST-7: "Modified Fire Resistive" / "MFR" â†’ FIRE class 5 (BLDGSCHEME=FIRE, BLDGCLASS=5)
  CONST-8: "Joisted Masonry" / "JM" â†’ FIRE class 2 (BLDGSCHEME=FIRE, BLDGCLASS=2)
  CONST-9: "Masonry Non-Combustible" / "MNC" â†’ FIRE class 4 (BLDGSCHEME=FIRE, BLDGCLASS=4)
  CONST-10: "Heavy Timber" / "HT" / "Glulam" â†’ FIRE class 7 (BLDGSCHEME=FIRE, BLDGCLASS=7)

IMPORTANT CONSTRAINTS:
  - Only output codes from the provided code list — NEVER invent or guess codes
  - Maximum confidence is 0.95 — never output 1.0
  - If two codes are equally plausible, lower confidence to 0.65-0.70
  - Abbreviations in raw values reduce certainty — lower confidence by 0.05-0.10
"""


def _build_llm_model(valid_codes: List[str]) -> None:
    """Initialize Azure ChatOpenAI for construction/occupancy classification. Called at startup."""
    global _llm_model

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key        = os.getenv("AZURE_OPENAI_API_KEY", "")
    deployment     = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

    if not azure_endpoint or not api_key:
        logger.warning(
            "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set. "
            "LLM classification stage disabled — TF-IDF will be used as fallback."
        )
        return

    # JSON mode enforces structured output; Pydantic validates it as a second safety net.
    # Temperature=0 ensures deterministic classification results.
    _llm_model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    logger.info(f"Azure ChatOpenAI initialized for code_mapper (deployment={deployment}, temperature=0).")


# ── Startup loader ────────────────────────────────────────────────────────────────

def build_tfidf_indexes() -> None:
    """Load all reference JSON, TF-IDF pickle files, and ISO/RMS maps at startup."""
    global _occ_codes, _const_codes, _rms_occ_codes, _rms_const_codes
    global _abbreviations, _iso_map, _rms_to_air_map

    # Import dynamically to avoid circular dependency
    from ontology_router import _load_with_overrides

    def load_json(name: str) -> dict:
        p = _REF_DIR / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        logger.warning(f"Reference file not found: {p}")
        return {}

    _occ_codes = _load_with_overrides("occupancy", "AIR")
    _const_codes = _load_with_overrides("construction", "AIR")
    _abbreviations = load_json("abbreviations.json")

    # â”€â”€ Build RMS occupancy registry from atc_to_air_occ_map.json â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Keys are ATC numeric strings ("1".."54"); values carry description + air_code.
    # We expose these as the RMS occupancy code registry so the deterministic
    # matcher and LLM classify directly into ATC numeric codes, NOT Res4/Res3.
    _atc_raw = load_json("atc_to_air_occ_map.json").get("atc_to_air", {})
    _rms_occ_codes.clear()
    for atc_key, meta in _atc_raw.items():
        _rms_occ_codes[atc_key] = {
            "description": meta.get("description", ""),
            "keywords":    [meta.get("description", "").lower()],
            "air_code":    meta.get("air_code", ""),
        }
    
    # Merge RMS specific overrides (e.g., native RMS codes like Res1, Com1)
    _rms_occ_overrides = _load_with_overrides("occupancy", "RMS")
    for code, meta in _rms_occ_overrides.items():
        if code.startswith("_"): continue
        if code in _rms_occ_codes:
            existing = _rms_occ_codes[code].get("keywords", [])
            _rms_occ_codes[code]["keywords"] = list(dict.fromkeys(existing + meta.get("keywords", [])))
            if meta.get("description"):
                _rms_occ_codes[code]["description"] = meta["description"]
        else:
            _rms_occ_codes[code] = {
                "description": meta.get("description", ""),
                "keywords": meta.get("keywords", []),
                "air_code": "",
            }
    logger.info(f"RMS occ registry built from atc_to_air_occ_map + overrides: {len(_rms_occ_codes)} codes")

    # â”€â”€ Build RMS construction registry from rms_to_air_const_map.json â”€â”€â”€â”€â”€â”€â”€
    # Keys are RMS numeric/alpha-numeric codes ("1", "2B", "3A1", "4A", â€¦).
    # rms_desc is used as the description/keyword for the deterministic matcher.
    _rms_const_codes.clear()
    _rms_const_raw = load_json("rms_to_air_const_map.json")
    _lookup_priority = ["advanced_structural", "basic", "infrastructure", "tanks", "industrial_equipment"]
    for category in _lookup_priority:
        for rms_code, meta in _rms_const_raw.get(category, {}).items():
            if rms_code not in _rms_const_codes:  # advanced beats basic for same key
                desc = meta.get("rms_desc", "")
                _rms_const_codes[rms_code] = {
                    "description": desc,
                    "keywords":    [desc.lower()],
                    "air_code":    meta.get("air_class", ""),
                    "category":    category,
                }
                
    # Merge RMS specific overrides (e.g., native RMS codes like W1, S1)
    _rms_const_overrides = _load_with_overrides("construction", "RMS")
    for code, meta in _rms_const_overrides.items():
        if code.startswith("_"): continue
        if code in _rms_const_codes:
            existing = _rms_const_codes[code].get("keywords", [])
            _rms_const_codes[code]["keywords"] = list(dict.fromkeys(existing + meta.get("keywords", [])))
            if meta.get("description"):
                _rms_const_codes[code]["description"] = meta["description"]
        else:
            _rms_const_codes[code] = {
                "description": meta.get("description", ""),
                "keywords": meta.get("keywords", []),
                "air_code": "",
                "category": "override",
            }
    logger.info(f"RMS const registry built from rms_to_air_const_map + overrides: {len(_rms_const_codes)} codes")

    # Load new reference maps
    iso_data = load_json("iso_fire_class_map.json")
    _iso_map = iso_data.get("iso_to_air", {})

    rms_air_data = load_json("rms_to_air_const_map.json")
    # Flatten all categories into a single lookup dict
    for category in ["basic", "advanced_structural", "infrastructure", "tanks", "industrial_equipment"]:
        for code, mapping in rms_air_data.get(category, {}).items():
            _rms_to_air_map[code.upper()] = mapping

    # Load occupancy maps
    atc_data = load_json("atc_to_air_occ_map.json")
    _atc_to_air_map.update(atc_data.get("atc_to_air", {}))

    raw_lookup_data = load_json("occ_raw_string_lookup.json")
    _occ_raw_lookup.update(raw_lookup_data.get("lookup", {}))

    ctx_data = load_json("occ_context_rules.json")
    _occ_context_rules.update(ctx_data.get("rules", {}))

    # â”€â”€ Load RMS construction raw-string lookup (English descriptions â†’ scheme/class) â”€â”€
    const_lookup_data = load_json("rms_const_string_lookup.json")
    _const_raw_lookup.update(const_lookup_data.get("lookup", {}))
    logger.info(f"RMS const raw-string lookup: {len(_const_raw_lookup)} entries loaded.")

    # â”€â”€ Inject keyword aliases into basic RMS construction codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Basic RMS codes only carry their rms_desc as keyword (e.g. code 1 = "Wood").
    # We inject English synonym aliases so the deterministic scorer can directly
    # match common inputs like "frame", "non-combustible", "steel", etc.
    _RMS_CONST_KEYWORD_ALIASES: Dict[str, list] = {
        "0":  ["unknown", "unspecified", "tbd", "unclassified"],
        "1":  ["wood", "frame", "wood frame", "stick frame", "timber frame",
               "light frame", "light wood frame", "post frame", "pole barn",
               "adobe", "sip", "structural insulated panel", "modular wood"],
        "2":  ["masonry", "brick", "cmu", "block", "joisted masonry", "jm",
               "unreinforced masonry", "urm", "stone masonry", "concrete block",
               "masonry bearing", "brick masonry"],
        "3":  ["reinforced concrete", "concrete", "rc", "cast in place",
               "cip", "post tension", "flat slab", "waffle slab",
               "prestressed concrete", "concrete frame", "shear wall",
               "concrete shear wall"],
        "4":  ["steel", "steel frame", "structural steel", "metal frame",
               "light gauge steel", "lgs", "steel stud", "steel joist",
               "braced frame", "moment frame", "steel deck"],
        "5":  ["mobile home", "manufactured home", "trailer"],
        "2B": ["unreinforced masonry", "urm", "unconfined masonry", "plain masonry"],
        "2C": ["structural masonry", "reinforced masonry", "rm brick"],
        "2C1":["reinforced masonry shear wall", "rm shear wall"],
        "3A": ["cast in place concrete", "cip concrete", "rc concrete roof"],
        "3A1":["rc moment resisting frame", "rcmrf", "concrete moment frame"],
        "3A2":["rc mrf with shear walls", "concrete dual system"],
        "3A4":["rc shear wall", "concrete shear wall"],
        "3B": ["precast concrete", "precast", "pre-cast", "precast panel",
               "hollow core", "double tee", "prefab concrete"],
        "3B4":["tilt-up", "tilt up", "tilt wall", "tilt panel", "tiltup"],
        "4A": ["steel frame concrete roof", "steel concrete"],
        "4A1":["steel mrf", "steel moment resisting frame", "smrf"],
        "4A4":["concentrically braced frame", "cbf", "braced steel frame"],
        "4B": ["light metal frame", "light metal", "metal building",
               "pre-engineered metal", "pemb", "metal bldg"],
        "4C": ["steel frame wood roof", "steel frame metal roof"],
        "5A": ["mobile home without tie downs", "manufactured home no tie down"],
        "5B": ["mobile home with tie downs", "manufactured home tie down"],
    }
    for code, aliases in _RMS_CONST_KEYWORD_ALIASES.items():
        if code in _rms_const_codes:
            existing = _rms_const_codes[code].get("keywords", [])
            _rms_const_codes[code]["keywords"] = list(dict.fromkeys(existing + aliases))
    logger.info("RMS const keyword aliases injected.")

    logger.info(
        f"Occupancy maps ready: {len(_atc_to_air_map)} ATC classes, "
        f"{len(_occ_raw_lookup)} raw string lookups, {len(_occ_context_rules)} context rules."
    )

    # Build TF-IDF indexes
    pkl_names = {
        "air_occ":   "air_occ_vectorizer.pkl",
        "air_const": "air_const_vectorizer.pkl",
        "rms_occ":   "rms_occ_vectorizer.pkl",
        "rms_const": "rms_const_vectorizer.pkl",
    }
    for key, fname in pkl_names.items():
        p = _TFIDF_DIR / fname
        if p.exists():
            with open(p, "rb") as f:
                _tfidf_indexes[key] = pickle.load(f)
            logger.info(f"Loaded TF-IDF index: {key}")
        else:
            logger.warning(f"TF-IDF pkl not found: {p}. Run build_references.py first.")

    # Initialize Azure ChatOpenAI model with all valid codes
    all_codes = list(_const_codes.keys()) + list(_occ_codes.keys())
    _build_llm_model(all_codes)

    logger.info(
        f"Code mapper ready: {len(_occ_codes)} AIR occ, {len(_const_codes)} AIR const, "
        f"{len(_rms_occ_codes)} RMS occ, {len(_rms_const_codes)} RMS const codes loaded. "
        f"{len(_iso_map)} ISO classes, {len(_rms_to_air_map)} RMSâ†’AIR mappings."
    )


def _get_code_registry(target: str, field: str) -> Dict[str, Dict]:
    if target == "AIR" and field == "occupancy":
        return _occ_codes
    if target == "AIR" and field == "construction":
        return _const_codes
    if target == "RMS" and field == "occupancy":
        return _rms_occ_codes
    if target == "RMS" and field == "construction":
        return _rms_const_codes
    return {}


def _get_tfidf_key(target: str, field: str) -> str:
    field_key = "occ" if field == "occupancy" else "const"
    return f"{target.lower()}_{field_key}"


# â”€â”€ Abbreviation expansion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def expand_abbreviations(text: str) -> Tuple[str, bool]:
    """
    Replace whole-word abbreviations with full text.
    Returns (expanded_text, was_expanded).
    """
    if not text:
        return text, False
    expanded = text
    was_expanded = False
    # Sort by length descending to match longer abbreviations first
    for abbrev in sorted(_abbreviations.keys(), key=len, reverse=True):
        full = _abbreviations[abbrev]
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        new = re.sub(pattern, full, expanded, flags=re.IGNORECASE)
        if new != expanded:
            expanded = new
            was_expanded = True
    return expanded, was_expanded


def _is_atc_scheme(scheme: str, value: str, field: str = "occupancy") -> bool:
    """
    Detect ATC/RMS occupancy scheme input.

    Actual data format: RMS Occ Scheme = "ATC", RMS Occ Code = 1-54.
    scheme="ATC" is the primary trigger. "RMS" also accepted for occupancy.
    scheme="AIR" is always excluded (AIR codes are 3 digits, 300+).
    """
    upper_scheme = scheme.strip().upper()
    # AIR scheme is never an ATC class
    if upper_scheme == "AIR":
        return False
    if upper_scheme in _ATC_SCHEME_NAMES_EXPLICIT:
        return True
    # RMS labels are ATC only for occupancy (construction RMS = text codes W1/C1)
    if upper_scheme in _ATC_SCHEME_NAMES_OCC_ONLY and field == "occupancy":
        return True
    # Auto-detect: only when scheme is empty/unknown and field=occupancy
    if field == "occupancy" and not upper_scheme:
        return value.strip() in _ATC_EXACT_VALUES
    return False


def _lookup_atc(value: str) -> Optional[Dict]:
    """Look up ATC class value in the ATCâ†’AIR map."""
    stripped = value.strip()
    return _atc_to_air_map.get(stripped)


def _lookup_raw_occ_string(raw: str) -> Optional[Dict]:
    """Pre-deterministic raw string cache lookup. Normalizes whitespace and case."""
    normalized = re.sub(r'\s+', ' ', raw.lower().strip())
    return _occ_raw_lookup.get(normalized)


def _lookup_raw_const_string(raw: str) -> Optional[Dict]:
    """
    Pre-deterministic raw string lookup for RMS construction descriptions.
    Returns {bldgclass, bldgscheme, final_category, confidence} or None.
    Tries full match first, then progressive prefix shortening for compound inputs.
    """
    normalized = re.sub(r'\s+', ' ', raw.lower().strip())
    # 1. Full exact match
    hit = _const_raw_lookup.get(normalized)
    if hit:
        return hit
    # 2. Strip trailing parentheticals e.g. "wood frame (apartments)" â†’ "wood frame"
    stripped = re.sub(r'\s*\([^)]*\)\s*$', '', normalized).strip()
    if stripped != normalized:
        hit = _const_raw_lookup.get(stripped)
        if hit:
            return hit
    # 3. Slash-split: "wood frame / brick siding" â†’ try each part
    if '/' in normalized:
        parts = [p.strip() for p in normalized.split('/')]
        for part in parts:
            hit = _const_raw_lookup.get(part)
            if hit:
                return hit
    return None


def _resolve_occ_context(raw: str, context_row: Dict) -> Optional[Dict]:
    """
    Apply context-aware disambiguation rules for ambiguous occupancy terms.
    context_row: the full data row dict (may contain account type, description, industry).
    Returns {air_code, atc, reasoning} or None.
    """
    normalized = re.sub(r'\s+', ' ', raw.lower().strip())
    rule = _occ_context_rules.get(normalized)
    if not rule:
        return None

    # Collect all context signals from the row
    row_text = " ".join(str(v).lower() for v in context_row.values() if v)

    for ctx in rule.get("contexts", []):
        signals = ctx.get("context_signals", [])
        if any(sig.lower() in row_text for sig in signals):
            return {
                "air_code": ctx["air_code"],
                "atc": ctx["atc"],
                "reasoning": ctx.get("reasoning", "Context signal matched"),
                "confidence": 0.93,
            }

    # Return default if exists
    default = rule.get("default")
    if default:
        return {
            "air_code": default["air_code"],
            "atc": default.get("atc", 0),
            "reasoning": default.get("reasoning", "Default rule applied"),
            "confidence": default.get("confidence", 0.80),
        }
    return None


# â”€â”€ ISO scheme detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _is_iso_scheme(scheme: str, value: str) -> bool:
    """
    Detect ISO Fire Class input using both explicit scheme label AND exact value matching.
    Only exact single-token matches are auto-detected to avoid false positives.
    """
    if scheme.strip().upper() in _ISO_SCHEME_NAMES:
        return True
    stripped = value.strip().upper()
    return stripped in _ISO_EXACT_LABELS


def _lookup_iso(value: str) -> Optional[Dict]:
    """Find ISO mapping by value alias search."""
    stripped = value.strip().upper()
    # Direct key lookup
    if stripped in _iso_map:
        d = dict(_iso_map[stripped])
        d["iso_key"] = stripped
        d["isf_code"] = d.get("isf_class")  # From iso_fire_class_map.json
        return d
    # Search aliases
    numeric_key = stripped if stripped.isdigit() else None
    for key, data in _iso_map.items():
        aliases = [a.upper() for a in data.get("aliases", [])]
        if stripped in aliases or (numeric_key and key == numeric_key):
            d = dict(data)
            d["iso_key"] = key
            d["isf_code"] = d.get("isf_class")
            return d
    return None


# â”€â”€ RMSâ†’AIR cross-translation lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _lookup_rms_to_air(rms_code: str) -> Optional[Dict]:
    """Look up RMS code in cross-translation table. Tries exact and normalized forms."""
    normalized = rms_code.strip().upper()
    if normalized in _rms_to_air_map:
        return _rms_to_air_map[normalized]
    return None


# â”€â”€ Priority-weighted deterministic scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _score_code_candidate(text: str, code: str, meta: Dict) -> float:
    """
    Score a candidate code against query text using priority-weighted matching.

    Scoring tiers:
      1.0 â€” exact full text match to keyword
      0.7 â€” keyword phrase found in text (phrase-in-text)
      0.5 â€” text found in keyword phrase (text-in-phrase)
      0.3 â€” description phrase in text
      0.2 â€” single-word overlap in keyword list
    Returns total float score (can be > 1.0 for multi-keyword matches).
    """
    keywords = [kw.lower() for kw in meta.get("keywords", [])]
    description = meta.get("description", "").lower()
    score = 0.0

    for kw in keywords:
        if kw == text:
            score += 1.0       # Exact full match — highest priority
        elif kw in text:
            # Weight by specificity: longer keyword phrases score higher
            score += 0.7 + (len(kw.split()) - 1) * 0.05
        elif text in kw:
            score += 0.5
        elif any(word in text.split() for word in kw.split() if len(word) > 3):
            score += 0.2       # Partial word overlap (only on substantial words)

    if description and description in text:
        score += 0.3

    return round(score, 4)


def _deterministic_classify(
    text: str, registry: Dict[str, Dict], was_expanded: bool
) -> Tuple[Optional[str], float, Optional[Dict], List[Dict]]:
    """
    Run priority-weighted scoring over all codes in registry.
    Returns (best_code, confidence, best_meta, alternatives).
    """
    if not registry:
        return None, 0.0, None, []

    scored: List[Tuple[str, float, Dict]] = []
    for code, meta in registry.items():
        s = _score_code_candidate(text, code, meta)
        if s > 0:
            scored.append((code, s, meta))

    if not scored:
        return None, 0.0, None, []

    scored.sort(key=lambda x: x[1], reverse=True)
    best_code, best_score, best_meta = scored[0]

    # Confidence derivation from score
    if best_score >= 1.5:
        confidence = 0.95
    elif best_score >= 1.0:
        confidence = 0.90
    elif best_score >= 0.7:
        confidence = 0.85
    elif best_score >= 0.5:
        confidence = 0.75
    else:
        confidence = 0.65

    # Penalize if abbreviation was expanded (less certainty)
    if was_expanded:
        confidence = max(0.0, confidence - 0.10)

    # If top 2 scores are very close — ambiguity — lower confidence
    if len(scored) > 1:
        gap = scored[0][1] - scored[1][1]
        if gap < 0.2:
            confidence = max(0.0, confidence - 0.10)

    # Build alternatives (up to 2)
    alts = []
    for code, score, meta in scored[1:3]:
        if score > 0:
            alts.append({
                "code": code,
                "description": meta.get("description", ""),
                "confidence": round(confidence * (score / best_score) * 0.85, 3),
            })

    return best_code, min(0.95, confidence), best_meta, alts


# â”€â”€ LLM helper: retry-with-fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _call_llm_with_retry(
    prompt: str,
    valid_codes: List[str],
    max_retries: int = 2,
) -> Optional[_LLMBatchResult]:
    """
    Call Gemini with schema-constrained output and Pydantic validation.
    Returns parsed _LLMBatchResult or None on failure (fall through to TF-IDF).
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    if _llm_model is None:
        logger.warning("Azure ChatOpenAI not initialised - LLM stage skipped, TF-IDF fallback active.")
        return None

    messages = [
        SystemMessage(content=_LLM_SYSTEM_INSTRUCTION),
        HumanMessage(content=prompt),
    ]

    for attempt in range(max_retries + 1):
        try:
            _rate_limit_gemini()   # enforce â‰¤15 calls/min before each attempt
            response  = _llm_model.invoke(messages)
            raw_text  = response.content.strip()

            # Pydantic parse + validate
            parsed = _LLMBatchResult.model_validate_json(raw_text)

            # Secondary code validity check â€” reject any hallucinated codes
            for item in parsed.items:
                if item.code not in valid_codes:
                    logger.warning(
                        f"LLM returned invalid code '{item.code}' (attempt {attempt+1}). "
                        f"Retrying..." if attempt < max_retries else "Falling back to TF-IDF."
                    )
                    raise ValueError(f"Invalid code in LLM response: {item.code}")

            return parsed

        except Exception as e:
            # Catches parse errors (JSONDecodeError, ValidationError, ValueError)
            # AND API errors (ResourceExhausted/429, network timeouts, etc.)
            # All failures fall through gracefully to TF-IDF.
            err_type = type(e).__name__
            logger.warning(f"LLM attempt {attempt + 1}/{max_retries + 1} failed ({err_type}): {e}")
            if attempt == max_retries:
                return None

    return None


# â”€â”€ LangGraph state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class CodeMappingState(TypedDict):
    target: str
    field: str
    unique_items: List[Dict[str, Any]]
    results: Dict[str, Dict[str, Any]]
    pending_iso: List[int]          # after ATC fast-path
    pending_rms_direct: List[int]   # after ISO fast-path
    pending_conflict: List[int]     # after RMS direct
    pending_llm: List[int]
    pending_tfidf: List[int]
    pending_default: List[int]
    conflict_hints: Dict[str, Any]
    rules_config: Dict[str, Any]
    error_log: List[str]


# â”€â”€ Stage 0: ATC class fast-path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_atc_direct(state: CodeMappingState) -> CodeMappingState:
    """
    Fast-path for RMS/ATC class numeric inputs (scheme='ATC' or value is 1-54).
    Only applies to field='occupancy'. Maps directly to AIR code with confidence=1.0.
    All construction and unrecognised items pass through to iso_direct.
    """
    results = dict(state["results"])
    pending_iso: List[int] = []

    for item in state["unique_items"]:
        idx = item["index"]
        scheme = str(item.get("scheme") or "").strip()
        raw_value = str(item.get("value") or "").strip()

        if state["field"] != "occupancy" or not _is_atc_scheme(scheme, raw_value, state["field"]):
            pending_iso.append(idx)
            continue

        atc_data = _lookup_atc(raw_value.strip())
        if atc_data:
            air_code = atc_data["air_code"]
            registry = _get_code_registry(state["target"], state["field"])
            description = atc_data.get("description", registry.get(air_code, {}).get("description", ""))
            results[str(idx)] = {
                "code": air_code,
                "confidence": 1.0,
                "description": description,
                "method": "atc_direct",
                "original": raw_value,
                "alternatives": [],
                "reasoning": f"ATC class '{raw_value}' ({atc_data.get('description', '')}) â†’ AIR {air_code}",
                "abbreviation_expanded": False,
                "atc_class": raw_value,
            }
            logger.debug(f"ATC direct: item {idx} ATC '{raw_value}' â†’ AIR {air_code}")
        else:
            pending_iso.append(idx)

    return {**state, "results": results, "pending_iso": pending_iso}


# â”€â”€ Stage 1: ISO Fire Class fast-path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_iso_direct(state: CodeMappingState) -> CodeMappingState:
    """
    Fast-path for ISO Fire Class inputs (ISO 0-9 / F / JM / NC etc.).
    Only runs for field="construction" â€” ISO classes don't apply to occupancy.
    Items resolved here skip all downstream stages.
    """
    results = dict(state["results"])
    pending = state.get("pending_iso", [])           # key written by _node_atc_direct
    pending_rms_direct: List[int] = []

    for item in state["unique_items"]:
        idx = item["index"]
        scheme = str(item.get("scheme") or "").strip()
        raw_value = str(item.get("value") or "").strip()

        if state["field"] != "construction" or not _is_iso_scheme(scheme, raw_value):
            if idx in pending:
                pending_rms_direct.append(idx)
            continue

        iso_data = _lookup_iso(raw_value)
        if iso_data:
            if state["target"] == "AIR":
                # User wants Frame (Class 1) to remain "AIR 101".
                # For all other ISO-based matches, use the "ISF" schema.
                iso_key = iso_data.get("iso_key")
                if iso_key == "1" or iso_data.get("air_class") == "101":
                    out_code = "101"
                    scheme_ov = None
                elif iso_data.get("isf_code"):
                    out_code = iso_data["isf_code"]
                    scheme_ov = "ISF"
                else:
                    out_code = iso_data["air_class"]
                    scheme_ov = None
            else:
                out_code = iso_data["iso_key"]
                scheme_ov = "FIRE"
                
            results[str(idx)] = {
                "code": out_code,
                "confidence": 1.0,
                "description": iso_data.get("description", ""),
                "method": "iso_direct",
                "original": raw_value,
                "alternatives": [],
                "reasoning": f"ISO Fire Class '{raw_value}' ({iso_data.get('iso_label', '')}) â†’ {out_code}",
                "abbreviation_expanded": False,
                "iso_class": raw_value,
                "scheme_override": scheme_ov,
            }
            logger.debug(f"ISO direct: item {idx} '{raw_value}' â†’ {out_code} ({scheme_ov if scheme_ov else 'AIR'})")
        else:
            # Unknown ISO value â€” fall through
            pending_rms_direct.append(idx)

    return {**state, "results": results, "pending_rms_direct": pending_rms_direct}


# â”€â”€ Stage 0.5: RMSâ†’AIR cross-translation fast-path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_rms_direct(state: CodeMappingState) -> CodeMappingState:
    """
    Fast-path for RMSâ†’AIR construction cross-translation (construction field only).

    NOTE: For OCCUPANCY, RMS and ATC are the SAME system (numeric codes 1-54).
    Occupancy items with scheme=RMS should have already been resolved by
    _node_atc_direct. Any occupancy items reaching here are passed straight
    to the conflict stage â€” the construction RMS cross-translation table
    must not be applied to occupancy codes.
    """
    pending = state["pending_rms_direct"]
    results = dict(state["results"])
    pending_conflict: List[int] = []

    for item in state["unique_items"]:
        idx = item["index"]
        if idx not in pending:
            continue

        scheme = str(item.get("scheme") or "").strip().upper()
        raw_value = str(item.get("value") or "").strip()

        # Occupancy: RMS = ATC (same system) â€” already handled by _node_atc_direct.
        # Pass straight through; don't apply construction cross-translation table.
        if state["field"] == "occupancy":
            pending_conflict.append(idx)
            continue

        # Construction only: apply RMSâ†’AIR cross-translation when target=AIR
        if state["target"] != "AIR" or scheme not in {"RMS", "RMS_CONST", "RMS_OCC"}:
            pending_conflict.append(idx)
            continue

        rms_mapping = _lookup_rms_to_air(raw_value)
        if rms_mapping:
            out_code = rms_mapping["air_class"]
            final_scheme = None

            # User wants Non-Frame ISO classes (2-6) to use ISF schema in AIR mode.
            # RMS codes 1-6 map directly to ISO 1-6.
            if raw_value in {"2", "3", "4", "5", "6"}:
                iso_meta = _lookup_iso(raw_value)
                if iso_meta and iso_meta.get("isf_code"):
                    out_code = iso_meta["isf_code"]
                    final_scheme = "ISF"

            results[str(idx)] = {
                "code": out_code,
                "confidence": 0.98,
                "description": rms_mapping.get("air_desc", ""),
                "method": "rms_direct",
                "original": raw_value,
                "alternatives": [],
                "reasoning": f"RMS construction code '{raw_value}' ({rms_mapping.get('rms_desc', '')}) â†’ {out_code} (scheme={final_scheme or 'AIR'}) via cross-translation table",
                "abbreviation_expanded": False,
                "scheme_override": final_scheme,
            }
            logger.debug(f"RMS direct (construction): item {idx} '{raw_value}' â†’ {out_code} ({final_scheme or 'AIR'})")
        else:
            pending_conflict.append(idx)

    return {**state, "results": results, "pending_conflict": pending_conflict}



# â”€â”€ Stage 1: Structural conflict resolver â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_conflict(state: CodeMappingState) -> CodeMappingState:
    """
    Resolve compound/hybrid construction descriptions using structural priority rules.
    Only runs for field="construction". Occupancy items pass through immediately.
    High-confidence results (>= conflict_threshold) skip deterministic stage.
    """
    pending = state["pending_conflict"]
    if not pending:
        return {**state, "pending_llm": [], "conflict_hints": {}}

    threshold = state["rules_config"].get("deterministic_score_threshold", 0.85)
    results = dict(state["results"])
    conflict_hints: Dict[str, Any] = dict(state.get("conflict_hints", {}))
    pending_llm: List[int] = []

    items_by_index = {item["index"]: item for item in state["unique_items"]}

    for idx in pending:
        item = items_by_index.get(idx, {})
        raw_value = str(item.get("value") or "").strip()
        expanded, was_expanded = expand_abbreviations(raw_value)

        # Only apply conflict resolution to construction field
        if state["field"] != "construction":
            pending_llm.append(idx)
            continue

        conflict_result: Optional[ConflictResult] = _conflict_resolver.resolve(expanded)

        if conflict_result and conflict_result.confidence >= threshold:
            # High confidence â†’ use directly, skip deterministic
            if state["target"] == "AIR":
                out_code = conflict_result.air_code
                final_scheme = None
                
                # Check for ISF schema preference (excluding Frame/Class 1)
                iso_cls = str(conflict_result.iso_class or "")
                if iso_cls and iso_cls != "1":
                    iso_meta = _lookup_iso(iso_cls)
                    if iso_meta and iso_meta.get("isf_code"):
                        out_code = iso_meta["isf_code"]
                        final_scheme = "ISF"

                desc = _const_codes.get(out_code, {}).get("description", conflict_result.final_category)
                results[str(idx)] = {
                    "code": out_code,
                    "confidence": conflict_result.confidence,
                    "description": desc,
                    "method": "conflict_rule",
                    "original": raw_value,
                    "alternatives": conflict_result.alternatives,
                    "reasoning": conflict_result.reasoning,
                    "abbreviation_expanded": was_expanded,
                    "conflict_flag": conflict_result.conflict_flag,
                    "rule_applied": conflict_result.rule_applied,
                    "scheme_override": final_scheme,
                }
            else:
                # For RMS target: conflict resolver returns an AIR code, but we need
                # the RMS numeric code. Do NOT hardcode W1 â€” pass to LLM with the
                # conflict reasoning as a hint so it picks the correct RMS code.
                conflict_hints[str(idx)] = conflict_result
                pending_llm.append(idx)
                logger.debug(
                    f"RMS target conflict item {idx} '{raw_value}' deferred to LLM "
                    f"with conflict hint: {conflict_result.final_category}"
                )
        elif conflict_result:
            # Medium confidence â†’ store as hint for LLM, but still run deterministic
            conflict_hints[str(idx)] = conflict_result
            pending_llm.append(idx)
        else:
            # No conflict detected â†’ pass to deterministic
            pending_llm.append(idx)

    return {**state, "results": results, "pending_llm": pending_llm, "conflict_hints": conflict_hints}


# â”€â”€ Stage 2: Priority-weighted deterministic keyword matching â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_deterministic(state: CodeMappingState) -> CodeMappingState:
    """
    Priority-weighted keyword matching over the code registry.
    For occupancy: first checks raw-string lookup and context rules before scoring.
    """
    registry = _get_code_registry(state["target"], state["field"])
    threshold = state["rules_config"].get("deterministic_score_threshold", 0.85)

    results = dict(state["results"])
    pending_llm: List[int] = list(state.get("pending_llm", []))
    new_pending_llm: List[int] = []

    items_by_index = {item["index"]: item for item in state["unique_items"]}

    for idx in pending_llm:
        if results.get(str(idx)) is not None:
            continue

        item = items_by_index.get(idx, {})
        raw_value = str(item.get("value") or "").strip()
        expanded, was_expanded = expand_abbreviations(raw_value)
        text = expanded.lower()

        # â”€â”€ Occupancy: raw-string cache + context rules (pre-scoring) â”€â”€â”€â”€â”€â”€â”€â”€
        if state["field"] == "occupancy":
            raw_hit = _lookup_raw_occ_string(raw_value)
            if raw_hit and not raw_hit.get("context_dependent"):
                hit_confidence = raw_hit["confidence"]
                air_code = raw_hit["air_code"]
                description = registry.get(air_code, {}).get("description", "")
                results[str(idx)] = {
                    "code": air_code,
                    "confidence": hit_confidence,
                    "description": description,
                    "method": "rule",
                    "original": raw_value,
                    "alternatives": [],
                    "reasoning": f"Raw string lookup match (confidence {hit_confidence:.2f})",
                    "abbreviation_expanded": was_expanded,
                    "atc_class": str(raw_hit.get("atc", "")),
                }
                if hit_confidence < threshold:
                    new_pending_llm.append(idx)
                continue

            # Context-dependent: try context rules with full row context
            if raw_hit and raw_hit.get("context_dependent"):
                ctx_result = _resolve_occ_context(raw_value, item.get("context", {}))
                if ctx_result:
                    air_code = ctx_result["air_code"]
                    description = registry.get(air_code, {}).get("description", "")
                    results[str(idx)] = {
                        "code": air_code,
                        "confidence": ctx_result["confidence"],
                        "description": description,
                        "method": "rule",
                        "original": raw_value,
                        "alternatives": [],
                        "reasoning": ctx_result["reasoning"],
                        "abbreviation_expanded": was_expanded,
                        "atc_class": str(ctx_result.get("atc", "")),
                    }
                    if ctx_result["confidence"] < threshold:
                        new_pending_llm.append(idx)
                    continue

        # â”€â”€ Construction: raw-string lookup (pre-scoring) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # This fires for BOTH target=RMS and target=AIR (construction field).
        # Returns {bldgclass, bldgscheme, final_category, confidence}.
        if state["field"] == "construction":
            const_hit = _lookup_raw_const_string(raw_value)
            if not const_hit:
                const_hit = _lookup_raw_const_string(expanded)
            if const_hit:
                hit_conf = const_hit["confidence"]
                bldgclass = const_hit["bldgclass"]
                bldgscheme = const_hit["bldgscheme"]
                final_cat = const_hit.get("final_category", "")
                
                out_code = bldgclass
                final_scheme = bldgscheme

                if state["target"] == "AIR":
                    if bldgscheme == "RMS":
                        mapped = _lookup_rms_to_air(bldgclass)
                        if mapped:
                            # User wants Frame (Class 1) to remain "AIR 101".
                            # For all other RMS->ISO mappings, prefer ISF code.
                            if bldgclass == "1":
                                out_code = "101"
                                final_scheme = None
                            else:
                                iso_meta = _lookup_iso(bldgclass)
                                if iso_meta and iso_meta.get("isf_code"):
                                    out_code = iso_meta["isf_code"]
                                    final_scheme = "ISF"
                                else:
                                    out_code = mapped["air_class"]
                                    final_scheme = None 
                    elif bldgscheme == "FIRE":
                        # User wants Frame (Class 1) to remain "AIR 101".
                        if bldgclass == "1":
                            out_code = "101"
                            final_scheme = None
                        else:
                            mapped = _lookup_iso(bldgclass)
                            if mapped:
                                if mapped.get("isf_code"):
                                    out_code = mapped["isf_code"]
                                    final_scheme = "ISF"
                                else:
                                    out_code = mapped["air_class"]
                                    final_scheme = None

                results[str(idx)] = {
                    "code": out_code,
                    "confidence": hit_conf,
                    "description": final_cat,
                    "method": "rule",
                    "original": raw_value,
                    "alternatives": [],
                    "reasoning": f"Construction raw-string lookup: '{raw_value}' â†’ {final_cat} (scheme={final_scheme or 'AIR'}, class={out_code})",
                    "abbreviation_expanded": was_expanded,
                    "scheme_override": final_scheme,
                }
                logger.debug(
                    f"const_raw_lookup hit: idx={idx} '{raw_value}' â†’ "
                    f"{final_cat} BLDGSCHEME={bldgscheme} BLDGCLASS={bldgclass} conf={hit_conf:.2f}"
                )
                if hit_conf < threshold:
                    new_pending_llm.append(idx)
                continue
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        best_code, confidence, best_meta, alts = _deterministic_classify(
            text, registry, was_expanded
        )

        if best_code and best_meta:
            results[str(idx)] = {
                "code": best_code,
                "confidence": confidence,
                "description": best_meta.get("description", ""),
                "method": "rule",
                "original": raw_value,
                "alternatives": alts,
                "reasoning": f"Priority-weighted keyword match (scoreâ†’confidence {confidence:.2f})",
                "abbreviation_expanded": was_expanded,
            }

        r = results.get(str(idx))
        if r is None or r.get("confidence", 0) < threshold:
            new_pending_llm.append(idx)

    for idx in pending_llm:
        if results.get(str(idx)) is not None and idx not in new_pending_llm:
            r = results[str(idx)]
            if r.get("confidence", 0) < threshold:
                new_pending_llm.append(idx)

    return {**state, "results": results, "pending_llm": new_pending_llm}


# â”€â”€ Stage 3: Gemini LLM with structured output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_llm(state: CodeMappingState) -> CodeMappingState:
    """
    LLM classification with:
    - response_schema (JSON mode enforced)
    - temperature=0, top_p=0.1 (deterministic)
    - Pydantic validation as second safety net
    - Retry-with-fallback (2 retries â†’ TF-IDF)
    - System instruction with 8 structural conflict rules
    - Per-item conflict hints from Stage 1
    """
    pending = state["pending_llm"]
    if not pending:
        return {**state, "pending_tfidf": []}

    registry = _get_code_registry(state["target"], state["field"])
    llm_threshold = state["rules_config"].get("llm_confidence_threshold", 0.70)
    valid_codes = list(registry.keys())

    # Build compact code list (cap to avoid token overflow)
    code_list_text = "\n".join(
        f"  {code}: {meta.get('description', '')}"
        for code, meta in list(registry.items())[:100]
    )

    # Build per-item text with conflict hints
    items_by_index = {item["index"]: item for item in state["unique_items"]}
    conflict_hints = state.get("conflict_hints", {})
    items_text_parts = []

    for i, idx in enumerate(pending):
        item = items_by_index.get(idx, {})
        raw = str(item.get("value") or "")
        expanded, _ = expand_abbreviations(raw)
        scheme = item.get("scheme", "")
        ctx = item.get("context", {})
        year = ctx.get("year_built", "unknown")
        stories = ctx.get("stories", "unknown")

        hint_text = ""
        hint = conflict_hints.get(str(idx))
        if hint:
            hint_text = (
                f' [CONFLICT HINT: Rule "{hint.rule_applied}" suggests '
                f'code {hint.air_code} ({hint.final_category}), '
                f'confidence {hint.confidence:.2f}. Confirm or override with reasoning.]'
            )

        items_text_parts.append(
            f'{i+1}. scheme="{scheme}", value="{expanded}", '
            f'year_built={year}, stories={stories}{hint_text}'
        )

    items_text = "\n".join(items_text_parts)
    field_label = state["field"]
    target_label = state["target"]

    # Occupancy-specific system rule addendum
    occ_rule_addendum = ""
    if field_label == "occupancy":
        occ_rule_addendum = """
OCCUPANCY-SPECIFIC RULES (additional, apply after structural rules):
  OCC-1: "Shop" in county/municipal/government context â†’ General Services (343)
         "Shop" standalone â†’ Retail (312); "Mechanic/Auto Shop" â†’ Personal Services (314)
  OCC-2: "Pharmacy/Drug Store" standalone â†’ Retail (312);
         "Pharmacy" inside hospital â†’ Health Care (316)
  OCC-3: Multi-use: primary function governs; residential over commercial if policy is residential
  OCC-4: Strip tenant names (Amazon, Walmart, McDonald's) â€” classify by building function
  OCC-5: School level: Elementary/Primary/K-12/Daycare â†’ Primary (346); University/College â†’ Higher Ed (345)
  OCC-6: BBQ structures, outdoor kitchens, grills â†’ Restaurants (331)
  OCC-7: Utility subtypes: pumping/water treatment â†’ Water (362); sewer/wastewater â†’ Sewer (363);
         gas distribution â†’ Natural Gas (364); power/electric â†’ Electrical (361)
  OCC-8: All farm subtypes (dairy, poultry, grain, ranch, livestock) â†’ Agriculture (373)
  OCC-9: "Activity Center" in government context â†’ General Services (343); recreational â†’ Entertainment (317)
  OCC-10: Casinos â†’ Entertainment (317) using ATC 48; Parking garage â†’ Parking (318)
"""

    prompt = f"""Classify each property {field_label} description to {target_label} {field_label} codes.{occ_rule_addendum}
VALID CODES (use ONLY these -- output the exact code string):
{code_list_text}

ITEMS TO CLASSIFY:
{items_text}

Apply the structural conflict resolution rules from your system instructions before classifying.

Return ONLY valid JSON in this exact structure:
{{
  "items": [
    {{
      "code": "<exact_code_from_list>",
      "confidence": 0.00,
      "reasoning": "<primary use/occupancy + any context rule applied>",
      "alternatives": [
        {{"code": "<code2>", "confidence": 0.00}}
      ]
    }}
  ]
}}

RULES:
- items array must have exactly {len(pending)} elements, one per input item, in order
- confidence range: 0.0 to 0.95 maximum
- max 2 alternatives per item; empty array if not ambiguous
- lower confidence for abbreviation-based inputs or ambiguous descriptions
"""

    results = dict(state["results"])
    pending_tfidf: List[int] = []
    error_log = list(state.get("error_log", []))

    parsed = _call_llm_with_retry(prompt, valid_codes)

    if parsed:
        print("\n" + "="*50)
        print("=== LLM CLASSIFICATION RAW RESPONSES ===")
        print("="*50)
        print(parsed.model_dump_json(indent=2))
        print("="*50 + "\n")

    if parsed and len(parsed.items) == len(pending):
        for i, idx in enumerate(pending):
            entry = parsed.items[i]
            code = entry.code
            confidence = entry.confidence
            item = items_by_index.get(idx, {})
            raw = str(item.get("value") or "")

            if code in registry:
                description = registry[code].get("description", "")
                results[str(idx)] = {
                    "code": code,
                    "confidence": min(0.95, confidence),
                    "description": description,
                    "method": "llm",
                    "original": raw,
                    "alternatives": [
                        {"code": alt.code, "confidence": alt.confidence}
                        for alt in entry.alternatives[:2]
                        if alt.code in registry
                    ],
                    "reasoning": entry.reasoning,
                    "abbreviation_expanded": False,
                }
                if confidence < llm_threshold:
                    pending_tfidf.append(idx)
            else:
                # Shouldn't happen due to retry logic, but safety net
                logger.warning(f"LLM code '{code}' not in registry for item {idx}. â†’ TF-IDF.")
                pending_tfidf.append(idx)
    else:
        if not parsed:
            logger.warning("LLM failed after retries. Routing all LLM items to TF-IDF.")
            error_log.append("LLM failure: max retries exceeded")
        else:
            logger.warning(f"LLM returned {len(parsed.items)} items but expected {len(pending)}. â†’ TF-IDF all.")
            error_log.append(f"LLM length mismatch: got {len(parsed.items)}, expected {len(pending)}")
        pending_tfidf = list(pending)

    return {**state, "results": results, "pending_tfidf": pending_tfidf, "error_log": error_log}


# â”€â”€ Stage 4: TF-IDF cosine similarity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_tfidf(state: CodeMappingState) -> CodeMappingState:
    pending = state["pending_tfidf"]
    if not pending:
        return {**state, "pending_default": []}

    tfidf_key = _get_tfidf_key(state["target"], state["field"])
    index = _tfidf_indexes.get(tfidf_key)

    tfidf_threshold = state["rules_config"].get("tfidf_confidence_threshold", 0.50)
    registry = _get_code_registry(state["target"], state["field"])

    results = dict(state["results"])
    pending_default: List[int] = []
    items_by_index = {item["index"]: item for item in state["unique_items"]}

    for idx in pending:
        item = items_by_index.get(idx, {})
        raw = str(item.get("value") or "")
        expanded, was_expanded = expand_abbreviations(raw)

        if index is None:
            pending_default.append(idx)
            continue

        try:
            vec = index["vectorizer"]
            matrix = index["matrix"]
            code_list = index["codes"]

            query_vec = vec.transform([expanded.lower()])
            scores = cosine_similarity(query_vec, matrix)[0]
            top_indices = scores.argsort()[-3:][::-1]

            best_idx_pos = top_indices[0]
            best_sim = float(scores[best_idx_pos])
            best_code = code_list[best_idx_pos]

            # Map cosine similarity to confidence
            if best_sim >= 0.65:
                confidence = 0.68
            elif best_sim >= 0.45:
                confidence = 0.58
            elif best_sim >= 0.25:
                confidence = 0.48
            else:
                confidence = 0.40

            alts = []
            for ti in top_indices[1:]:
                if float(scores[ti]) > 0.1:
                    alts.append({
                        "code": code_list[ti],
                        "confidence": round(float(scores[ti]) * 0.65, 3),
                    })

            description = registry.get(best_code, {}).get("description", "")
            
            # Apply ISF schema preference for AIR targets in TF-IDF results
            out_code = best_code
            final_scheme = None
            if state["target"] == "AIR" and state["field"] == "construction":
                # For non-combustible (152), map to ISF 3.
                if out_code == "152":
                    out_code = "3"
                    final_scheme = "ISF"
            
            results[str(idx)] = {
                "code": out_code,
                "confidence": confidence,
                "description": description,
                "method": "tfidf",
                "original": raw,
                "alternatives": alts,
                "reasoning": f"TF-IDF cosine similarity: {best_sim:.3f}",
                "abbreviation_expanded": was_expanded,
                "scheme_override": final_scheme,
            }

            if confidence < tfidf_threshold:
                pending_default.append(idx)

        except Exception as exc:
            logger.warning(f"TF-IDF failed for item {idx}: {exc}")
            pending_default.append(idx)

    return {**state, "results": results, "pending_default": pending_default}


# â”€â”€ Stage 5: Default â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_default(state: CodeMappingState) -> CodeMappingState:
    pending = state["pending_default"]
    if not pending:
        return state

    registry = _get_code_registry(state["target"], state["field"])
    rules = state["rules_config"]
    items_by_index = {item["index"]: item for item in state["unique_items"]}

    if state["field"] == "occupancy":
        default_code = (
            rules.get("default_occ_code_air") if state["target"] == "AIR"
            else rules.get("default_occ_code_rms", "Com1")
        )
    else:
        default_code = (
            rules.get("default_const_code_air") if state["target"] == "AIR"
            else rules.get("default_const_code_rms", "W1")
        )

    description = registry.get(str(default_code), {}).get("description", "Unknown")
    results = dict(state["results"])

    for idx in pending:
        item = items_by_index.get(idx, {})
        raw = str(item.get("value") or "")
        results[str(idx)] = {
            "code": str(default_code),
            "confidence": 0.0,
            "description": description,
            "method": "default",
            "original": raw,
            "alternatives": [],
            "reasoning": "No match found across all stages; default code applied",
            "abbreviation_expanded": False,
        }

    return {**state, "results": results}


# â”€â”€ Build and compile the 6-stage graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_code_mapping_graph():
    g = StateGraph(CodeMappingState)
    g.add_node("atc_direct",   _node_atc_direct)
    g.add_node("iso_direct",   _node_iso_direct)
    g.add_node("rms_direct",   _node_rms_direct)
    g.add_node("conflict",     _node_conflict)
    g.add_node("deterministic",_node_deterministic)
    g.add_node("llm",          _node_llm)
    g.add_node("tfidf",        _node_tfidf)
    g.add_node("default",      _node_default)

    g.set_entry_point("atc_direct")
    g.add_edge("atc_direct",   "iso_direct")
    g.add_edge("iso_direct",   "rms_direct")
    g.add_edge("rms_direct",   "conflict")
    g.add_edge("conflict",     "deterministic")
    g.add_edge("deterministic","llm")
    g.add_edge("llm",          "tfidf")
    g.add_edge("tfidf",        "default")
    g.add_edge("default",      END)
    return g.compile()


_code_mapping_graph = _build_code_mapping_graph()


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def map_codes(
    unique_items: List[Dict[str, Any]],
    target: str,
    field: str,
    rules_config: BusinessRulesConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Run the 6-stage code mapping pipeline for a list of unique items.

    Stages:
      0   iso_direct    â€” ISO Fire Class 0-9 fast-path (confidence 1.0)
      0.5 rms_direct    â€” RMSâ†’AIR cross-translation (confidence 0.98)
      1   conflict      â€” Structural conflict resolution (compound descriptions)
      2   deterministic â€” Priority-weighted keyword scoring
      3   llm           â€” Gemini with response_schema + temperature=0
      4   tfidf         â€” Cosine similarity fallback
      5   default       â€” Business rules default

    unique_items: [{index: int, scheme: str, value: str, context: {year_built, stories}}]
    Returns: {index_str â†’ result_dict}
    """
    initial: CodeMappingState = {
        "target": target,
        "field": field,
        "unique_items": unique_items,
        "results": {},
        "pending_iso": [],
        "pending_rms_direct": [],
        "pending_conflict": [],
        "pending_llm": [],
        "pending_tfidf": [],
        "pending_default": [],
        "conflict_hints": {},
        "rules_config": rules_config.model_dump(),
        "error_log": [],
    }
    final_state = _code_mapping_graph.invoke(initial)
    return final_state["results"]


def extract_unique_pairs(rows: List[Dict], scheme_col: str, value_col: str) -> List[Dict]:
    """
    Extract deduped (scheme, value) pairs from rows, with context (year, stories).
    Returns a list suitable for map_codes().
    """
    seen: Dict[Tuple, int] = {}
    items = []

    for row in rows:
        scheme = str(row.get(scheme_col) or "").strip()
        value = str(row.get(value_col) or "").strip()
        if not value:
            continue
        key = (scheme, value)
        if key not in seen:
            seen[key] = len(items)
            items.append({
                "index": len(items),
                "scheme": scheme,
                "value": value,
                "context": {
                    "year_built": row.get("Year_Built_Final") or row.get("YearBuilt"),
                    "stories": row.get("No_of_Stories_Final") or row.get("NumberOfStories"),
                },
            })

    return items


def build_row_key(scheme: str, value: str) -> str:
    return f"{scheme}|{value}"


