"""
construction_rules.py — Structural conflict resolver for construction code mapping.

Encodes insurance underwriting rules for classifying hybrid/compound construction
descriptions. Implements the 30-scenario conflict matrix from the Construction_Mapping.xlsx
reference with rule-based priority logic.

Rule priority order (highest to lowest):
  1. URM Governs (weakest structural system)
  2. Mixed Construction (two distinct load-bearing systems)
  3. Combustible Downgrade (wood roof/joists on masonry walls)
  4. Frame Governs (structural frame beats facade/infill/veneer)
  5. Tilt-Up Governs (tilt-up walls with any roof)
  6. Heavy Timber Governs (primary HT beats masonry secondary)
  7. Unknown Frame Conservative Downgrade
  8. Minor Combustible Allowed (non-load-bearing partition, interior)
"""
import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("construction_rules")

_BASE = pathlib.Path(__file__).parent  # agents/cat/
_RULES_FILE = _BASE.parent.parent / "reference" / "construction_conflict_rules.json"  # backend/reference/

# ── Token sets (all lowercase for matching) ────────────────────────────────────

# Structural frame tokens — if present, they govern over non-structural elements
FRAME_TOKENS: List[str] = [
    "steel frame", "steel moment frame", "steel braced frame",
    "concrete frame", "rc frame", "reinforced concrete frame",
    "wood frame", "light wood frame", "timber frame",
    "heavy timber", "metal building", "pre-engineered metal building",
    "tilt-up", "tilt up", "tilt wall",
]

# Non-structural / facade tokens — never govern the structural classification
NON_STRUCTURAL_TOKENS: List[str] = [
    "masonry infill", "brick veneer", "masonry veneer", "stone veneer",
    "glass curtain wall", "curtain wall", "glass facade",
    "precast panels", "precast cladding",
    "masonry exterior", "masonry front", "masonry facade",
    "concrete infill", "infill walls",
    "canopy", "steel canopy",
    "foundation", "concrete foundation",
    "wood mezzanine", "wood partition", "wood office interior",
    "interior wood office",
]

# Combustible load-bearing tokens — trigger downgrade when present on masonry
COMBUSTIBLE_STRUCTURAL: List[str] = [
    "wood roof", "wood joist", "wood truss roof", "combustible roof",
    "wood deck", "wood planking", "wood joists",
    "wood upper floor", "wood upper level", "wood upper",
    "wood floors", "wood floor", "wood framing", "wood penthouse",
    "wood mezzanine",  # only combustible if load-bearing; see rule logic
]

# Masonry wall tokens — need combustible roof to trigger Joisted Masonry
MASONRY_WALL_TOKENS: List[str] = [
    "masonry walls", "brick walls", "cmu walls", "concrete masonry unit walls",
    "concrete block walls", "block walls", "masonry load-bearing",
    "masonry bearing", "urm walls", "unreinforced masonry walls",
]

# URM trigger tokens — always the weakest system
URM_TOKENS: List[str] = [
    "unreinforced masonry", "urm", "unconfined masonry", "plain masonry",
]

# Tilt-up triggers
TILT_UP_TOKENS: List[str] = [
    "tilt-up", "tilt up", "tilt wall", "tilt panel",
]

# Heavy timber triggers
HEAVY_TIMBER_TOKENS: List[str] = [
    "heavy timber", "ht", "hvy timber", "glulam", "glulam timber",
    "cross laminated timber", "clt", "post and beam timber",
]

# Mixed construction indicators — conjunction words that suggest compound description
CONJUNCTION_TOKENS: List[str] = [
    " with ", " over ", " and ", " + ", " & ", " attached to ",
    " with attached ", " lower level", " upper level", " upper floor",
    " podium", " penthouse", " extension", " addition",
]

# Tokens indicating the element is secondary/minor (non-load-bearing clues)
MINOR_QUALIFIER: List[str] = [
    "decorative", "cosmetic", "non-structural", "exterior only",
    "facade only", "cladding only", "interior partition", "office partition",
]

# Tokens that suggest "unknown frame" scenario
UNKNOWN_FRAME_TOKENS: List[str] = [
    "unknown frame", "unknown interior", "frame unknown", "unknown construction",
    "unspecified frame", "insufficient data",
]

# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class ConflictResult:
    final_category: str
    air_code: str
    iso_class: str
    confidence: float
    reasoning: str
    conflict_flag: bool
    rule_applied: str
    alternatives: List[Dict] = field(default_factory=list)


# ── Scenario lookup (from JSON) ────────────────────────────────────────────────

_scenario_index: Dict[str, dict] = {}  # raw_lower → scenario dict

def _load_scenarios() -> None:
    """Load scenarios from construction_conflict_rules.json at startup."""
    global _scenario_index
    if not _RULES_FILE.exists():
        logger.warning(f"Conflict rules file not found: {_RULES_FILE}")
        return
    try:
        data = json.loads(_RULES_FILE.read_text(encoding="utf-8"))
        for scenario in data.get("scenarios", []):
            key = scenario["raw"].lower().strip()
            _scenario_index[key] = scenario
        logger.info(f"Loaded {len(_scenario_index)} construction conflict scenarios.")
    except Exception as e:
        logger.error(f"Failed to load conflict rules: {e}")


_load_scenarios()


# ── Token matching helpers ─────────────────────────────────────────────────────

def _match_any(text: str, token_list: List[str]) -> List[str]:
    """Return which tokens from token_list appear as whole-phrase matches in text."""
    matched = []
    for token in token_list:
        # Use word boundary for single words, phrase match for multi-word tokens
        if " " in token:
            if token in text:
                matched.append(token)
        else:
            if re.search(r"\b" + re.escape(token) + r"\b", text):
                matched.append(token)
    return matched


def _is_compound_description(text: str) -> bool:
    """Return True if the description appears to describe a hybrid / mixed system."""
    return any(conj in text for conj in CONJUNCTION_TOKENS)


def _has_explicit_unknown_frame(text: str) -> bool:
    return bool(_match_any(text, UNKNOWN_FRAME_TOKENS)) or (
        "unknown" in text and ("frame" in text or "interior" in text or "structure" in text)
    )


# ── Main resolver ──────────────────────────────────────────────────────────────

class ConflictResolver:
    """
    Rule-based structural conflict resolver for construction descriptions.

    Usage:
        resolver = ConflictResolver()
        result = resolver.resolve("Steel frame with masonry infill walls")
        if result:
            print(result.air_code, result.final_category)
    """

    def resolve(self, raw_description: str) -> Optional[ConflictResult]:
        """
        Resolve a raw construction description to an AIR code using structural conflict rules.

        Returns ConflictResult if the description triggers a known conflict pattern,
        or None if the description is simple/non-compound (let deterministic stage handle it).
        """
        if not raw_description:
            return None

        text = raw_description.lower().strip()

        # ── Step 0: Exact scenario lookup ──────────────────────────────────────
        if text in _scenario_index:
            s = _scenario_index[text]
            return ConflictResult(
                final_category=s["final_category"],
                air_code=s["air_code"],
                iso_class=s.get("iso_class", "0"),
                confidence=s["confidence"],
                reasoning=s["reasoning"],
                conflict_flag=True,
                rule_applied=s.get("rule", "exact_scenario_match"),
            )

        # ── Step 1: Only process compound descriptions ─────────────────────────
        if not _is_compound_description(text):
            return None  # simple description — let deterministic stage handle

        # Collect tokens present in this description
        urm_found = _match_any(text, URM_TOKENS)
        masonry_walls_found = _match_any(text, MASONRY_WALL_TOKENS)
        combustible_found = _match_any(text, COMBUSTIBLE_STRUCTURAL)
        frame_tokens_found = _match_any(text, FRAME_TOKENS)
        non_structural_found = _match_any(text, NON_STRUCTURAL_TOKENS)
        tilt_up_found = _match_any(text, TILT_UP_TOKENS)
        heavy_timber_found = _match_any(text, HEAVY_TIMBER_TOKENS)
        minor_qualifier_found = _match_any(text, MINOR_QUALIFIER)
        unknown_frame = _has_explicit_unknown_frame(text)

        # ── Rule 3.1: URM always governs (weakest system) ─────────────────────
        if urm_found:
            return ConflictResult(
                final_category="Joisted Masonry",
                air_code="119",
                iso_class="2",
                confidence=0.93,
                reasoning=f"URM detected ({', '.join(urm_found)}) — weakest structural system governs regardless of secondary elements",
                conflict_flag=True,
                rule_applied="urm_governs_weakest",
                alternatives=[{"code": "114", "confidence": 0.55}],
            )

        # ── Rule 3.2: Combustible roof/floor over masonry → Joisted Masonry ───
        if masonry_walls_found and combustible_found:
            # Filter out minor qualifiers that suggest non-load-bearing
            non_minor = [c for c in combustible_found if not any(
                q in text for q in ["partition", "interior office", "mezzanine"]
            )]
            if non_minor:
                return ConflictResult(
                    final_category="Joisted Masonry",
                    air_code="119",
                    iso_class="2",
                    confidence=0.93,
                    reasoning=f"Masonry walls ({', '.join(masonry_walls_found[:1])}) + combustible element ({', '.join(non_minor[:1])}) → Joisted Masonry",
                    conflict_flag=True,
                    rule_applied="combustible_downgrade",
                    alternatives=[{"code": "111", "confidence": 0.40}],
                )

        # ── Rule 3.3: Mixed construction — two distinct load-bearing systems ──
        result = self._check_mixed_construction(text, combustible_found, frame_tokens_found)
        if result:
            return result

        # ── Rule 3.4: Tilt-Up governs over any roof type ─────────────────────
        if tilt_up_found:
            return ConflictResult(
                final_category="Masonry Non-Combustible",
                air_code="111",
                iso_class="4",
                confidence=0.92,
                reasoning=f"Tilt-up concrete walls ({', '.join(tilt_up_found)}) govern structural classification",
                conflict_flag=True,
                rule_applied="tilt_up_governs",
                alternatives=[{"code": "136", "confidence": 0.60}],
            )

        # ── Rule 3.5: Heavy timber governs over masonry secondary ─────────────
        if heavy_timber_found and masonry_walls_found:
            return ConflictResult(
                final_category="Heavy Timber",
                air_code="104",
                iso_class="7",
                confidence=0.92,
                reasoning=f"Heavy timber ({', '.join(heavy_timber_found[:1])}) is primary structural system; masonry walls are secondary",
                conflict_flag=True,
                rule_applied="heavy_timber_governs",
                alternatives=[{"code": "119", "confidence": 0.40}],
            )

        # ── Rule 3.6: Frame governs over non-structural facade/infill ─────────
        if frame_tokens_found and non_structural_found:
            # Exclude if minor qualifiers suggest the frame element is the minor one
            primary_frame = self._identify_primary_frame(text, frame_tokens_found)
            if primary_frame:
                air_code, final_cat, iso_cls = self._frame_to_air(primary_frame, text)
                return ConflictResult(
                    final_category=final_cat,
                    air_code=air_code,
                    iso_class=iso_cls,
                    confidence=0.90,
                    reasoning=f"Structural frame ({primary_frame}) governs; {', '.join(non_structural_found[:1])} classified as non-structural",
                    conflict_flag=True,
                    rule_applied="frame_governs",
                    alternatives=[],
                )

        # ── Rule 3.7: Unknown frame → conservative Joisted Masonry ────────────
        if unknown_frame:
            return ConflictResult(
                final_category="Joisted Masonry",
                air_code="119",
                iso_class="2",
                confidence=0.70,
                reasoning="Structural frame unknown; only facade material identified → conservative Joisted Masonry downgrade",
                conflict_flag=True,
                rule_applied="unknown_frame_conservative_downgrade",
                alternatives=[{"code": "100", "confidence": 0.40}],
            )

        # ── No known conflict pattern matched ─────────────────────────────────
        return None

    def _check_mixed_construction(
        self,
        text: str,
        combustible_found: List[str],
        frame_tokens_found: List[str],
    ) -> Optional[ConflictResult]:
        """Detect mixed construction scenarios."""

        # Pattern A: vertical hybrid — wood upper floors over concrete/masonry
        vertical_hybrid_patterns = [
            (["wood upper", "wood upper floor", "wood upper level", "wood floors",
              "wood penthouse", "wood upper floors"], ["concrete podium", "rc", "concrete lower", "concrete"]),
            (["wood upper", "wood floors", "wood upper floors"], ["masonry", "cmu", "concrete block"]),
        ]
        for wood_patterns, base_patterns in vertical_hybrid_patterns:
            has_wood_upper = any(p in text for p in wood_patterns)
            has_concrete_base = any(p in text for p in base_patterns)
            if has_wood_upper and has_concrete_base:
                return ConflictResult(
                    final_category="Mixed Construction",
                    air_code="141",
                    iso_class="2",
                    confidence=0.92,
                    reasoning="Combustible wood structural upper floors over concrete/masonry base = Mixed Construction (vertical hybrid)",
                    conflict_flag=True,
                    rule_applied="mixed_construction",
                    alternatives=[{"code": "100", "confidence": 0.35}],
                )

        # Pattern B: adjacent distinct structural systems (horizontal)
        adjacent_patterns = [
            ("attached wood office", "steel", "141"),
            ("wood office", "metal warehouse", "152"),  # interior partition only → metal governs
            ("concrete block office", "steel warehouse", "141"),
            ("masonry addition", "metal building", "141"),
            ("concrete parking", "steel office", "100"),  # unresolvable separate systems
            ("wood office extension", "steel", "141"),
        ]
        for secondary, primary, result_code in adjacent_patterns:
            if secondary in text and primary in text:
                # Special: interior wood office inside metal building → primary governs
                if "interior" in text and "wood office" in secondary and "metal" in primary:
                    return ConflictResult(
                        final_category="Metal Building",
                        air_code="152",
                        iso_class="3",
                        confidence=0.90,
                        reasoning="Interior wood office is a non-structural partition; metal primary structure governs",
                        conflict_flag=True,
                        rule_applied="primary_governs_interior_partition",
                    )
                final_cat = "Mixed Construction" if result_code != "100" else "Unknown"
                return ConflictResult(
                    final_category=final_cat,
                    air_code=result_code,
                    iso_class="2" if result_code == "141" else "0",
                    confidence=0.86,
                    reasoning=f"Adjacent/connected distinct structural systems detected → {final_cat}",
                    conflict_flag=True,
                    rule_applied="mixed_construction",
                    alternatives=[],
                )

        # Pattern C: CMU lower + wood upper (vertical hybrid explicit)
        if ("cmu" in text or "concrete masonry" in text) and (
            "wood upper" in text or "wood level" in text or "wood floor" in text
        ):
            return ConflictResult(
                final_category="Mixed Construction",
                air_code="141",
                iso_class="2",
                confidence=0.91,
                reasoning="CMU lower structure + wood upper level/floors = Mixed Construction",
                conflict_flag=True,
                rule_applied="mixed_construction",
                alternatives=[{"code": "119", "confidence": 0.40}],
            )

        return None

    def _identify_primary_frame(
        self, text: str, frame_tokens: List[str]
    ) -> Optional[str]:
        """
        From a list of detected frame tokens, identify the primary (structural) one.
        Priority: concrete frame > steel frame > heavy timber > metal building > wood frame.
        """
        priority_order = [
            "concrete frame", "rc frame", "reinforced concrete frame",
            "steel moment frame", "steel braced frame", "steel frame",
            "heavy timber",
            "pre-engineered metal building", "metal building",
            "wood frame", "light wood frame",
            "tilt-up", "tilt up",
        ]
        for candidate in priority_order:
            if candidate in frame_tokens:
                return candidate
        return frame_tokens[0] if frame_tokens else None

    def _frame_to_air(
        self, frame_token: str, text: str
    ) -> Tuple[str, str, str]:
        """Map a primary frame token to (air_code, final_category, iso_class)."""
        mapping: Dict[str, Tuple[str, str, str]] = {
            "concrete frame":             ("131", "Concrete Frame",    "6"),
            "rc frame":                   ("131", "Concrete Frame",    "6"),
            "reinforced concrete frame":  ("131", "Concrete Frame",    "6"),
            "steel moment frame":         ("155", "Steel Frame",       "5"),
            "steel braced frame":         ("153", "Steel Frame",       "5"),
            "steel frame":                ("151", "Steel Frame",       "5"),
            "heavy timber":               ("104", "Heavy Timber",      "7"),
            "pre-engineered metal building": ("152", "Metal Building", "3"),
            "metal building":             ("152", "Metal Building",    "3"),
            "wood frame":                 ("101", "Frame",             "1"),
            "light wood frame":           ("102", "Frame",             "1"),
            "tilt-up":                    ("111", "Masonry Non-Combustible", "4"),
            "tilt up":                    ("111", "Masonry Non-Combustible", "4"),
        }
        return mapping.get(frame_token, ("100", "Unknown", "0"))
