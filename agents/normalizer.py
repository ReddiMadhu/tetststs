"""
normalizer.py â€” All row-level normalization for the CAT pipeline.

Functions:
  normalize_all_rows(rows, rules_config) â†’ (normalized_rows, new_flags)

Each sub-function focuses on one field group and returns
(updated_row, list_of_flag_dicts).
"""
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from rules import BusinessRulesConfig
from agents.cat.secondary_modifier_mapper import get_mapper

logger = logging.getLogger("normalizer")

# â”€â”€ Lookup tables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

SPRINKLER_TRUE_VALUES = {"yes", "y", "1", "true", "sprinklered", "wet pipe", "dry pipe", "wet", "dry"}
SPRINKLER_FALSE_VALUES = {"no", "n", "0", "false", "none", "no sprinkler", "unsprinklered"}

WOOD_FRAME_CODES_AIR = {str(c) for c in range(101, 106)}  # 101-105
MASONRY_CODES_AIR = {str(c) for c in range(111, 122)}     # 111-121

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "dual": 2, "double": 2, "triple": 3, "ground": 1,
}

# â”€â”€ Helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_flag(row_index: int, field: str, issue: str, current_value: Any,
               message: str, confidence: Optional[float] = None,
               alternatives: Optional[List] = None) -> dict:
    return {
        "row_index": row_index,
        "field": field,
        "issue": issue,
        "current_value": current_value,
        "confidence": confidence,
        "alternatives": alternatives or [],
        "message": message,
    }


# â”€â”€ Year built â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_years(row: dict, row_idx: int, rules: BusinessRulesConfig, target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    yb_key = "YearBuilt" if target == "AIR" else "YEARBUILT"
    upg_key = "YearRetrofitted" if target == "AIR" else "YEARUPGRAD"

    raw_built = str(row.get("YearBuilt") or row.get("YEARBUILT") or row.get(yb_key) or "").strip()
    raw_upgrad = str(row.get("YearRetrofitted") or row.get("YEARUPGRAD") or row.get(upg_key) or "").strip()

    combined_raw = f"{raw_built} {raw_upgrad}".strip()
    if not combined_raw:
        row[yb_key] = None
        row[upg_key] = None
        return row, flags

    s_lower = combined_raw.lower()

    # Alpha Year "198X" rule
    if re.search(r"\b19\d[a-zA-Z]\b", s_lower) or re.search(r"\b20\d[a-zA-Z]\b", s_lower):
        flags.append(_make_flag(row_idx, yb_key, "invalid_alpha_year", combined_raw, "Alpha year like 198X -> blank"))
        row[yb_key] = None
        row[upg_key] = None
        return row, flags

    # Find 4-digit years
    years = [int(m) for m in re.findall(r"\b(1\d{3}|2\d{3})\b", s_lower)]
    
    # Check 2-digit years if no 4-digits
    if not years:
        for m in re.findall(r"\b(\d{2})\b", s_lower):
            yy = int(m)
            years.append(1900 + yy if yy >= 25 else 2000 + yy)

    # Filter years (1700 to 2026 allowed)
    valid_years = sorted(list(set(y for y in years if 1700 <= y <= 2026)))

    if not valid_years:
        row[yb_key] = None
        row[upg_key] = None
        flags.append(_make_flag(row_idx, yb_key, "unparseable_year", combined_raw, "No valid year extracted."))
        return row, flags

    # EXCLUDE/INCLUDE checking
    exclude_pattern = re.compile(r"\b(roofs?|electrical|plumb(?:ing)?|hvac|mechanical|cosmetic|interior|tenant|appraisal)\b", re.IGNORECASE)
    include_pattern = re.compile(r"\b(upgrade[sd]?|renovat(?:e|ed|ion)?|retrofit(?:ted)?|rehab(?:bed)?|modernize[sd]?|code|reconstruct(?:ion|ed)?|rebuilt|reno|seismic(?:ally)?|update[sd]?)\b", re.IGNORECASE)
    bld_pattern = re.compile(r"\b(build|built|construct(?:ed|ion)?|\byob\b|original(?:ly)?|circa|approx|phase|rebuilt|rebuild)\b", re.IGNORECASE)

    has_include = include_pattern.search(s_lower)
    has_exclude = exclude_pattern.search(s_lower)
    has_bld = bld_pattern.search(s_lower)

    year_built = None
    upgrade_year = None

    if len(valid_years) == 1:
        y = valid_years[0]
        # Assign correctly based on which raw string contains the year
        if str(y) in str(raw_built):
            year_built = y
        elif str(y) in str(raw_upgrad):
            upgrade_year = y
            year_built = None
        else:
            year_built = y
    else:
        # Multiple years present
        year_built = min(valid_years)
        upgrade_candidate_years = [y for y in valid_years if y > year_built]
        
        if upgrade_candidate_years:
            if has_include:
                upgrade_year = max(upgrade_candidate_years)
            elif has_exclude:
                # roof replaced 2008 without structural word -> Ignore upgrade
                upgrade_year = None
            else:
                # No specific words, e.g. "1990/2012" -> Treat max as structural upgrade 
                upgrade_year = max(upgrade_candidate_years)

    row[yb_key] = year_built

    if upgrade_year:
        row[upg_key] = upgrade_year
    elif year_built and (has_exclude or has_include):
        row[upg_key] = 9999
    elif year_built and len(valid_years) == 1:
        row[upg_key] = 9999
    else:
        row[upg_key] = 9999 if year_built else None

    # Blank out if no year built
    if not year_built:
        row[upg_key] = None

    return row, flags


# â”€â”€ Number of stories â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


_WORD_TO_NUM_STORIES = {
    "single": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

def _normalize_stories(row: dict, row_idx: int, rules: BusinessRulesConfig,
                        target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    stories_key = "NumberOfStories" if target == "AIR" else "NUMSTORIES"
    raw = row.get("NumberOfStories") or row.get("NUMSTORIES") or row.get(stories_key)
    if raw is None or str(raw).strip() == "":
        row[stories_key] = None
        return row, flags

    s = str(raw).strip()

    # Exact negative check (invalid)
    if re.match(r"^\s*-\s*\d+(?:\.\d+)?\s*$", s):
        flags.append(_make_flag(row_idx, stories_key, "negative_stories",
                                raw, f"Negative stories '{raw}' â€” blanked."))
        row[stories_key] = None
        return row, flags

    s_lower = s.lower()
    
    # 0. Evaluate explicit summations like "3 + 2" -> "5"
    while re.search(r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', s_lower):
        def repl_add(m):
            return " " + str(float(m.group(1)) + float(m.group(2))) + " "
        s_lower = re.sub(r'(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)', repl_add, s_lower, count=1)
    
    # 1. Expand "G+X" / "Ground+X" -> (1+X)
    def repl_g(m):
        return " " + str(1 + int(m.group(1))) + " "
    if re.search(r'\bg(?:round)?\s*\+', s_lower):
        s_lower = re.sub(r'\bg(?:round)?\s*\+\s*(\d+)\b', repl_g, s_lower)
    
    # 2. Ignore Mezzanine / Basement
    if re.search(r'mezzanine|mezz|basement|bsmt', s_lower):
        s_lower = re.sub(r'(?:\+\s*)?\b(?:\d+\s+)?(?:mezzanine|mezz|basement|bsmt)\b', ' ', s_lower)
        flags.append(_make_flag(row_idx, stories_key, "ignored_mezzanine_basement",
                                raw, f"Ignored mezzanine/basement references in '{raw}'."))

    # 3. Ignore building counts (e.g. "5 Buildings - Various")
    if re.search(r'building|bldg|structure', s_lower):
        s_lower = re.sub(r'\b(\d+)\s+(?:buildings?|bldgs?|structures?)\b', ' ', s_lower)

    # 4. Convert words
    for w, n in _WORD_TO_NUM_STORIES.items():
        if w in s_lower:
            s_lower = re.sub(r'\b' + re.escape(w) + r'\b', str(n), s_lower)

    # 5. Extract all positive decimals
    s_clean = re.sub(r'[-\u2013/,(]+', ' ', s_lower)
    s_clean = re.sub(r'\bto\b', ' ', s_clean)
    
    nums = [float(m) for m in re.findall(r'\d+(?:\.\d+)?', s_clean)]
    
    if not nums:
        flags.append(_make_flag(row_idx, stories_key, "unparseable_stories",
                                raw, f"Cannot parse number of stories from '{raw}' â€” blanked."))
        row[stories_key] = None
        return row, flags
        
    import math
    stories = max(math.ceil(n) for n in nums)
    row[stories_key] = stories

    if '+' in str(raw) and len(nums) == 1 and str(raw).strip() != s_lower.strip():
        # Evaluated an addition
        flags.append(_make_flag(row_idx, stories_key, "summed_stories",
                                raw, f"Evaluated addition in '{raw}' yielding {stories}."))
    elif len(nums) > 1:
        flags.append(_make_flag(row_idx, stories_key, "multiple_story_values",
                                raw, f"Extracted maximum value {stories} from '{raw}'."))
    elif any("." in m for m in re.findall(r'\d+(?:\.\d+)?', s_clean)):
        flags.append(_make_flag(row_idx, stories_key, "decimal_stories",
                                raw, f"Decimal stories '{raw}' rounded up to {stories}."))

    # Business rule: wood frame story limit
    const_code = str(row.get("ConstructionCode", "") or row.get("BLDGCLASS", ""))
    if stories > rules.max_stories_wood_frame and const_code in WOOD_FRAME_CODES_AIR:
        action = rules.stories_exceeded_action
        const_field = "ConstructionCode" if target == "AIR" else "BLDGCLASS"
        if action == "reset_construction":
            row["Construction_Code_Original"] = const_code
            row[const_field] = "100"  # Unknown
            flags.append(_make_flag(row_idx, const_field, "wood_frame_stories_exceeded",
                                    const_code, f"{stories}-story building exceeds wood frame limit; construction reset to Unknown"))
        elif action == "reset_stories":
            row[stories_key] = None
            flags.append(_make_flag(row_idx, stories_key, "wood_frame_stories_exceeded",
                                    stories, f"Stories reset: {stories} exceeds max {rules.max_stories_wood_frame} for wood frame"))
        else:
            flags.append(_make_flag(row_idx, stories_key, "wood_frame_stories_exceeded",
                                    stories, f"Info: {stories} stories for wood frame (code {const_code}) exceeds typical limit of {rules.max_stories_wood_frame}"))

    return row, flags

# â”€â”€ Building count â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Named structures that count as +1 in additive strings if no number precedes them
_NAMED_STRUCTURES = frozenset({
    "clubhouse", "club house", "community center", "center", "office",
    "shed", "maintenance shed", "garage", "auxiliary", "aux",
    "main building", "reception", "gym", "pool house", "amenity center",
    "annex", "pavilion", "chapel", "library", "lobby",
})

_VAGUE_BLDG_PAT = re.compile(
    r"^(n/?a|various|multiple buildings?|multiple|tbd|unknown|"
    r"several|numerous|many|tba|none|nil|na|-)$",
    re.IGNORECASE,
)

_SINGLE_BLDG_PAT = re.compile(
    r"^(single building|single|one building|one)$", re.IGNORECASE
)

_WORD_TO_NUM: Dict[str, int] = {
    **WORD_NUMBERS,
    "a": 1, "an": 1,
    "twenty-one": 21, "twenty one": 21, "twenty-two": 22, "twenty two": 22,
    "twenty-three": 23, "twenty three": 23, "twenty-four": 24, "twenty four": 24,
    "twenty-five": 25, "twenty five": 25,
    "thirty": 30, "forty": 40, "fifty": 50,
}


def _word_to_int(token: str) -> Optional[int]:
    return _WORD_TO_NUM.get(token.strip().lower())


def _extract_all_ints(text: str) -> List[int]:
    return [int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", text)]


def _normalize_building_count(row: dict, row_idx: int,
                               target: str = "AIR") -> Tuple[dict, List[dict]]:
    """
    Normalize RiskCount (AIR) / NUMBLDGS (RMS) to a positive whole integer.

    Processing order:
      1  Blank/None -> None
      2  Vague/N/A text -> None + flag
      3  "Single Building" / "One" -> 1
      4  Word-number prefix (Eight Buildings -> 8)
      5  Units-vs-buildings ("X Units in Y Buildings" -> Y)
      6  Additive (+/&/and): sum all parts; named structure = +1 each
      7  Qualifier ("X (Including Y)") -> keep X
      8  Range -> take LARGEST
      9  Slash -> take LARGEST
     10  Decimal -> math.ceil
     11  Bare integer (or largest of multiple)
     12  Negative -> None + flag
     13  Unresolvable -> None + flag
    """
    flags: List[dict] = []
    bldg_key = "RiskCount" if target == "AIR" else "NUMBLDGS"

    raw = row.get(bldg_key) or row.get("RiskCount") or row.get("NUMBLDGS")
    if raw is None or str(raw).strip() == "":
        row[bldg_key] = None
        return row, flags

    s = str(raw).strip()

    # Negative value â€” detect BEFORE digit regex strips the minus sign
    if re.match(r"^-\s*\d", s):
        flags.append(_make_flag(row_idx, bldg_key, "negative_building_count",
                                raw, f"Negative building count '{raw}' â€” blanked."))
        row[bldg_key] = None
        return row, flags

    # 2. Vague / N/A
    if _VAGUE_BLDG_PAT.match(s):
        flags.append(_make_flag(row_idx, bldg_key, "vague_building_count",
                                raw, f"\'{raw}\' is vague/N/A â€” blanked."))
        row[bldg_key] = None
        return row, flags

    # 3. Single constant
    if _SINGLE_BLDG_PAT.match(s):
        row[bldg_key] = 1
        return row, flags

    s_lower = s.lower()

    # 4. Word-number prefix
    first_tok = s_lower.split()[0] if s_lower.split() else ""
    wn = _word_to_int(first_tok)
    if wn is not None:
        s_lower = s_lower.replace(first_tok, str(wn), 1)

    # 5a. "X Units in Y Buildings" -> Y
    m = re.search(r"(\d[\d,]*)\s+units?\s+in\s+(\d[\d,]*)\s+buildings?", s_lower)
    if m:
        result = int(m.group(2).replace(",", ""))
        flags.append(_make_flag(row_idx, bldg_key, "units_vs_buildings",
                                raw, f"Extracted building count ({result}) from 'units in buildings' pattern."))
        row[bldg_key] = result
        return row, flags

    # 5b. "X Buildings (Y Units Each)" -> X
    m = re.search(r"(\d[\d,]*)\s+buildings?\s*\(\s*\d+\s+units?", s_lower)
    if m:
        row[bldg_key] = int(m.group(1).replace(",", ""))
        return row, flags

    # 6. Additive
    if re.search(r"\+|&|\band\b", s_lower):
        parts = re.split(r"\+|&|\band\b", s_lower)
        total = 0
        valid_parse = True
        for part in parts:
            part = part.strip()
            if not part:
                continue
            nm = re.search(r"(\d[\d,]*)", part)
            if nm:
                total += int(nm.group(1).replace(",", ""))
            else:
                first = part.split()[0] if part.split() else ""
                wn2 = _word_to_int(first)
                if wn2 is not None:
                    total += wn2
                else:
                    part_clean = re.sub(
                        r"\b(buildings?|bldgs?|structures?)\b", "", part
                    ).strip()
                    if any(ns in part_clean for ns in _NAMED_STRUCTURES):
                        total += 1
                    elif part_clean:
                        valid_parse = False
                        break
        if valid_parse and total > 0:
            flags.append(_make_flag(row_idx, bldg_key, "additive_building_count",
                                    raw, f"Additive count: \'{raw}\' -> {total}."))
            row[bldg_key] = total
            return row, flags

    # 7. "X (Including Y)" qualifier -> keep X
    m = re.match(r"^(\d[\d,]*)\s*\(including\b", s_lower)
    if m:
        row[bldg_key] = int(m.group(1).replace(",", ""))
        return row, flags

    # Expand parenthetical content for remaining steps
    s_clean = re.sub(r"\(([^)]*)\)", lambda mo: " " + mo.group(1) + " ", s_lower)
    s_clean = s_clean.replace("(", " ").replace(")", " ").strip()

    # 8. Range -> take LARGEST
    m = re.search(r"(\d[\d,]*)\s*[-\u2013to]+\s*(\d[\d,]*)", s_clean)
    if m:
        a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        result = max(a, b)
        flags.append(_make_flag(row_idx, bldg_key, "building_count_range",
                                raw, f"Range \'{raw}\' â€” took largest ({result})."))
        row[bldg_key] = result
        return row, flags

    # 9. Slash -> take LARGEST
    m = re.search(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)", s_clean)
    if m:
        a, b = int(m.group(1).replace(",", "")), int(m.group(2).replace(",", ""))
        result = max(a, b)
        flags.append(_make_flag(row_idx, bldg_key, "building_count_slash",
                                raw, f"Slash \'{raw}\' â€” took largest ({result})."))
        row[bldg_key] = result
        return row, flags

    # 10. Decimal -> ceil
    float_m = re.search(r"(\d+\.\d+)", s_clean)
    if float_m:
        result = math.ceil(float(float_m.group(1)))
        flags.append(_make_flag(row_idx, bldg_key, "decimal_building_count",
                                raw, f"Decimal \'{raw}\' rounded up to {result}."))
        row[bldg_key] = result
        return row, flags

    # 11. Bare integers
    nums = _extract_all_ints(s_clean)
    if nums:
        result = max(nums)
        if len(nums) > 1:
            flags.append(_make_flag(row_idx, bldg_key, "multiple_integers_in_count",
                                    raw, f"Multiple integers in \'{raw}\' â€” kept largest ({result})."))
        # 12. Negative
        if result < 0:
            flags.append(_make_flag(row_idx, bldg_key, "negative_building_count",
                                    raw, f"Negative building count \'{raw}\' â€” blanked."))
            row[bldg_key] = None
            return row, flags
        row[bldg_key] = result
        return row, flags

    # 13. Unresolvable
    flags.append(_make_flag(row_idx, bldg_key, "unresolvable_building_count",
                            raw, f"Cannot parse building count from \'{raw}\' â€” blanked."))
    row[bldg_key] = None
    return row, flags


# â”€â”€ Gross area â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_VAGUE_AREA_PAT = re.compile(
    r"^(n/?a|various|varies|tbd|unknown|none|nil|included|-)$",
    re.IGNORECASE,
)

_INVALID_UNITS = re.compile(
    r"\b(acre|acres|ac|hectare|hectares|ha|sq\s*meters?|sqm|m2)\b",
    re.IGNORECASE,
)

def _normalize_area(row: dict, row_idx: int, rules: BusinessRulesConfig,
                    target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    area_key = "GrossArea" if target == "AIR" else "FLOORAREA"
    raw = row.get("GrossArea") or row.get("FLOORAREA") or row.get(area_key)
    if raw is None or str(raw).strip() == "":
        row[area_key] = None
        return row, flags

    s = str(raw).strip()

    if _INVALID_UNITS.search(s):
        flags.append(_make_flag(row_idx, area_key, "invalid_area_unit",
                                raw, f"Invalid unit in \'{raw}\' (Acres/Hectares/SqM) â€” blanked."))
        row[area_key] = None
        return row, flags

    if re.match(r"^\s*-\s*\d+(?:\.\d+)?\s*$", s):
        flags.append(_make_flag(row_idx, area_key, "negative_area",
                                raw, f"Negative area \'{raw}\' â€” blanked."))
        row[area_key] = None
        return row, flags

    s_lower = s.lower()

    if _VAGUE_AREA_PAT.match(s_lower):
        flags.append(_make_flag(row_idx, area_key, "vague_area",
                                raw, f"Vague/included area \'{raw}\' â€” blanked."))
        row[area_key] = None
        return row, flags

    def repl_km(m):
        val = float(m.group(1).replace(',', ''))
        mult = m.group(2)
        if mult == 'k': val *= 1000
        elif mult.startswith('m') or mult == 'mn': val *= 1000000
        return f" {int(val)} " if val.is_integer() else f" {val} "
    
    s_lower = re.sub(r'\b(\d[\d,\.]*)\s*(k|m|mn|million)[\bs]?\b', repl_km, s_lower)

    def repl_each(m):
        val1 = float(m.group(1).replace(',', ''))
        val2 = float(m.group(2).replace(',', ''))
        return f" {val1 * val2} "
        
    s_lower = re.sub(r'(\d[\d,\.]*)\s*\(each\)\s*(\d[\d,\.]*)', repl_each, s_lower)

    sqft_value = None

    if '+' in s_lower or '&' in s_lower:
        parts = re.split(r'\+|&', s_lower)
        total = 0.0
        for part in parts:
            nums = [float(n.replace(',', '')) for n in re.findall(r'(?<!-)\b\d[\d,\.]*', part.strip())]
            if nums:
                total += max(nums)
        if total > 0:
            sqft_value = total

    if sqft_value is None:
        s_clean = re.sub(r'[-\u2013/]', ' ', s_lower)
        nums = []
        for m in re.findall(r'\d[\d,\.]*', s_clean):
            val_str = m.replace(',', '')
            if val_str and val_str != '.':
                nums.append(float(val_str))
        
        if not nums:
            flags.append(_make_flag(row_idx, area_key, "unparseable_area",
                                    raw, f"Cannot parse area from \'{raw}\' â€” blanked."))
            row[area_key] = None
            return row, flags
        
        sqft_value = max(nums)

    row["Area_Converted"] = False

    if sqft_value < rules.min_area_sqft:
        if rules.invalid_area_action == "flag_review":
            flags.append(_make_flag(row_idx, area_key, "area_below_minimum",
                                    raw, f"Area {sqft_value:.0f} sqft below minimum {rules.min_area_sqft} sqft"))

    row[area_key] = round(sqft_value, 2) if not sqft_value.is_integer() else int(sqft_value)
    return row, flags


# â”€â”€ Financial values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_CURRENCY_SYMBOLS = {
    "$": "USD",
    "â‚¬": "EUR",
    "Â£": "GBP",
    "Â¥": "JPY",
    "â‚¹": "INR"
}

_CURRENCY_STRIP = re.compile(r"[$â‚¬Â£Â¥â‚¹,\s]")
_SHORTHAND = re.compile(r"^([\d.]+)\s*([KMBkmb])$")

def _parse_value(raw: Any) -> Tuple[Optional[float], Optional[str]]:
    if raw is None or str(raw).strip() == "":
        return None, None
    raw_str = str(raw).strip()
    
    cur_code = None
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in raw_str:
            cur_code = code
            break
            
    s = _CURRENCY_STRIP.sub("", raw_str)
    
    if s == '-' or s.lower() in ('n/a', 'na', 'none', 'nil'):
        return 0, cur_code
        
    m = _SHORTHAND.match(s)
    if m:
        num, suffix = float(m.group(1)), m.group(2).upper()
        multiplier = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
        val = num * multiplier
        return int(round(val)), cur_code
    try:
        val = float(s)
        return int(round(val)), cur_code
    except ValueError:
        return None, cur_code

VALUE_FIELDS_AIR = [
    ("BuildingValue",    "BuildingValue",    "max_building_value"),
    ("ContentsValue",    "ContentsValue",    "max_contents_value"),
    ("TimeElementValue", "TimeElementValue", "max_bi_value"),
]

VALUE_FIELDS_RMS = [
    # (src_field_candidates, dest_field, max_rule_attr)
    # src_field_candidates: list of keys to try in order; first non-None wins.
    # This handles both: column already remapped to canonical RMS key (e.g. EQCV1VAL)
    # AND: column still sitting under an intermediate AIR-style key (BuildingValue).
    ("BuildingValue",    "EQCV1VAL",  "max_building_value"),
    ("ContentsValue",    "EQCV2VAL",  "max_contents_value"),
    ("TimeElementValue", "EQCV3VAL",  "max_bi_value"),
]

BUILDING_VARIANTS = [f"BuildingValue{i}" for i in range(1, 6)]  # BuildingValue1..5


def _normalize_values(row: dict, row_idx: int, rules: BusinessRulesConfig,
                      target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    value_fields = VALUE_FIELDS_AIR if target == "AIR" else VALUE_FIELDS_RMS

    for src_field, dest_field, max_attr in value_fields:
        # For RMS: the source value may sit under the AIR intermediate key OR
        # already under the dest canonical key (if the column mapper wrote it there).
        # Try dest first (most specific), then fall back to src_field.
        if target == "RMS":
            raw = row.get(dest_field) or row.get(src_field)
        else:
            raw = row.get(src_field)
        
        val, cur_code = _parse_value(raw)
        
        if cur_code:
            if target == "RMS":
                lcur_key = dest_field.replace("VAL", "LCUR")
                if not row.get(lcur_key):
                    row[lcur_key] = cur_code
            else:
                if not row.get("Currency"):
                    row["Currency"] = cur_code

        if val is None:
            flags.append(_make_flag(row_idx, dest_field, "unparseable_value",
                                    raw, f"Cannot parse numeric value from '{raw}'"))
            row[dest_field] = None
            continue
        if val < 0:
            flags.append(_make_flag(row_idx, dest_field, "negative_value",
                                    val, f"{src_field} is negative: {val}"))
        max_val = getattr(rules, max_attr)
        if val > max_val:
            action = rules.invalid_value_action
            if action == "reset_value":
                row[dest_field] = None
                flags.append(_make_flag(row_idx, dest_field, "value_exceeds_max",
                                        val, f"{src_field}={val:,.0f} exceeds max {max_val:,.0f}; reset to None"))
            else:
                row[dest_field] = val
                if action == "flag_review":
                    flags.append(_make_flag(row_idx, dest_field, "value_exceeds_max",
                                            val, f"{src_field}={val:,.0f} exceeds configured max {max_val:,.0f}"))
        else:
            row[dest_field] = val

    return row, flags


# â”€â”€ Currency â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_currency(row: dict, row_idx: int, valid_currencies: set) -> Tuple[dict, List[dict]]:
    flags = []
    currency_cols = [
        "Currency", "CV1LCUR", "CV2LCUR", "CV3LCUR",
        "EQCV1LCUR", "EQCV2LCUR", "EQCV3LCUR", "WSCV3LCUR", "TOCV3LCUR",
        "HUCV3LCUR", "FPCV3LCUR", "TRCV3LCUR",
    ]
    found_currencies = set()

    for col in currency_cols:
        raw = row.get(col)
        if not raw:
            continue
        # Extract last 3 chars if embedded in value string
        s = str(raw).strip()[-3:].upper()
        if s in valid_currencies:
            found_currencies.add(s)
        else:
            # Try the whole value
            whole = str(raw).strip().upper()
            if whole in valid_currencies:
                found_currencies.add(whole)
            else:
                row[col] = None
                flags.append(_make_flag(row_idx, col, "unrecognized_currency",
                                        raw, f"Unrecognized currency code '{raw}' in {col}"))

    if len(found_currencies) > 1:
        flags.append(_make_flag(row_idx, "Currency", "currency_conflict",
                                list(found_currencies),
                                f"Currency conflict: multiple currencies found: {found_currencies}"))
        row["Currency_Conflicts"] = True
    elif found_currencies:
        resolved = list(found_currencies)[0]
        row["Currency"] = resolved
        row["EQCV1LCUR"] = resolved
        row["EQCV2LCUR"] = resolved
        row["EQCV3LCUR"] = resolved

    return row, flags


# â”€â”€ Sprinkler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_sprinkler(row: dict, row_idx: int,
                          target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    spk_key = "SprinklerSystem" if target == "AIR" else "SPRINKLER"
    raw = row.get("SprinklerSystem") or row.get("SPRINKLER") or row.get(spk_key)
    if raw is None:
        return row, flags
    s = str(raw).strip().lower()
    if s in SPRINKLER_TRUE_VALUES:
        row[spk_key] = 1
    elif s in SPRINKLER_FALSE_VALUES:
        row[spk_key] = 0
    else:
        row[spk_key] = None
        flags.append(_make_flag(row_idx, spk_key, "unrecognized_sprinkler",
                                raw, f"Unrecognized sprinkler value '{raw}'. Expected yes/no."))
    return row, flags


# â”€â”€ Roof / Wall / Foundation / Soft-Story (target-aware modifier mapping) â”€â”€â”€â”€â”€â”€

def _first_valid(row: dict, *keys: str) -> Any:
    for k in keys:
        v = row.get(k)
        if v is not None:
            s = str(v).strip().lower()
            if s != "nan" and s != "":
                return v
    return None

def _normalize_modifiers(row: dict, row_idx: int,
                          target: str = "AIR") -> Tuple[dict, List[dict]]:
    """
    Map secondary modifier fields using vendor-correct code schemas.

    AIR target : RoofGeometry â†’ roof_cover (0-11),  WallSiding/WallType â†’ wall_type (0-9)
    RMS target : ROOFGEOM    â†’ rms_roofsys (0-9),   CLADDING/WALLTYPE  â†’ rms_cladsys (0-10)
    Both targets: foundation_type (0-12), soft_story (0-2) â€” identical codes.
    """
    flags = []

    roof_key     = "RoofGeometry"   if target == "AIR" else "ROOFGEOM"
    wall_key     = "WallSiding"     if target == "AIR" else "CLADDING"
    found_key    = "FoundationType" if target == "AIR" else "FOUNDATION"
    soft_key     = "SoftStory"      if target == "AIR" else "SOFTSTORY"
    walltype_key = "WallType"       if target == "AIR" else "WALLTYPE"

    mapper = get_mapper()  # LLM enabled from API key, handled gracefully

    sub_row = {
        "roof_cover":      _first_valid(row, roof_key),
        "wall_type":       _first_valid(row, wall_key, walltype_key),
        "foundation_type": _first_valid(row, found_key),
        "soft_story":      _first_valid(row, soft_key),
    }

    # â”€â”€ Branch on target: use vendor-specific schemas for roof and wall â”€â”€â”€â”€â”€â”€â”€â”€
    if target == "AIR":
        result  = mapper.map_all(sub_row)
        roof_code_key  = "roof_cover"
        wall_code_key  = "wall_type"
    else:  # RMS
        result  = mapper.map_all_rms(sub_row)
        roof_code_key  = "rms_roofsys"
        wall_code_key  = "rms_cladsys"

    methods = result.get("_methods", {})

    # â”€â”€ Roof â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if row.get(roof_key) is not None:
        row[roof_key] = result[roof_code_key]
        method = methods.get(roof_code_key)
        if method not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, roof_key, "mapped_modifier",
                                    row.get(roof_key),
                                    f"Mapped roof ({roof_code_key}) via {method} "
                                    f"â†’ {result[roof_code_key + '_desc']}"))

    # â”€â”€ Wall / Cladding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if row.get(wall_key) is not None:
        row[wall_key] = result[wall_code_key]
        method = methods.get(wall_code_key)
        if method not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, wall_key, "mapped_modifier",
                                    row.get(wall_key),
                                    f"Mapped wall ({wall_code_key}) via {method} "
                                    f"â†’ {result[wall_code_key + '_desc']}"))

    # AIR WallType column (same AIR schema as WallSiding â€” both use wall_type codes)
    if target == "AIR" and row.get(walltype_key) is not None:
        row[walltype_key] = result["wall_type"]
        method = methods.get("wall_type")
        if method not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, walltype_key, "mapped_modifier",
                                    row.get(walltype_key),
                                    f"Mapped wall_type via {method} "
                                    f"â†’ {result['wall_type_desc']}"))

    # RMS also has a WALLTYPE column â€” write rms_cladsys code there too
    if target == "RMS" and row.get(walltype_key) is not None:
        row[walltype_key] = result["rms_cladsys"]
        method = methods.get("rms_cladsys")
        if method not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, walltype_key, "mapped_modifier",
                                    row.get(walltype_key),
                                    f"Mapped rms_cladsys via {method} "
                                    f"â†’ {result['rms_cladsys_desc']}"))

    # â”€â”€ Foundation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if row.get(found_key) is not None:
        row[found_key] = result["foundation_type"]
        method = methods.get("foundation_type")
        if method not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, found_key, "mapped_modifier",
                                    row.get(found_key),
                                    f"Mapped foundation via {method} "
                                    f"â†’ {result['foundation_type_desc']}"))

    # â”€â”€ Soft story â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if row.get(soft_key) is not None:
        row[soft_key] = result["soft_story"]
        method = methods.get("soft_story")
        if method not in ("empty", "integer"):
            flags.append(_make_flag(row_idx, soft_key, "mapped_modifier",
                                    row.get(soft_key),
                                    f"Mapped soft story via {method} "
                                    f"â†’ {result['soft_story_desc']}"))

    return row, flags



# â”€â”€ Location Name â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_location_name(row: dict, row_idx: int,
                              target: str = "AIR") -> Tuple[dict, List[dict]]:
    flags = []
    loc_key = "LocationName" if target == "AIR" else "LOCNAME"
    raw = row.get("LocationName") or row.get("LOCNAME") or row.get(loc_key)
    
    # Rule 2: Fallback if None
    if raw is None or str(raw).strip() == "":
        contract_key = "PolicyID" if target == "AIR" else "ACCNTNUM"
        fallback = (row.get(contract_key) or 
                    row.get("AccountName") or 
                    row.get("SubmissionName") or 
                    row.get("InsuredName"))
        if fallback and str(fallback).strip() != "":
            raw = fallback
            flags.append(_make_flag(row_idx, loc_key, "location_name_fallback",
                                    raw, f"Location name missing; used Account/Policy Name: '{raw}'"))
    
    # Rule 3: Missing
    if raw is None or str(raw).strip() == "":
        row[loc_key] = None
        flags.append(_make_flag(row_idx, loc_key, "missing_location_name",
                                None, "Location name missing and no fallback found."))
        return row, flags

    # Rule 4: Clean to varchar(40) (allowed: A-Za-z0-9 .,&'-()/)
    s = str(raw).strip()
    import re
    s = re.sub(r"[^A-Za-z0-9 .,&'\-()/]", "", s)
    s = s.strip()
    
    if len(s) > 40:
        s_truncated = s[:40].strip()
        flags.append(_make_flag(row_idx, loc_key, "location_name_truncated",
                                raw, f"Location name truncated to 40 chars: '{s_truncated}'"))
        s = s_truncated
    
    row[loc_key] = s
    return row, flags


# â”€â”€ Identity fields (bulk pass over all rows) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_identity_fields(
    rows: List[Dict],
    target_format: str,
    rules: BusinessRulesConfig
) -> Tuple[List[Dict], List[dict]]:
    """
    Bulk post-pass that enforces two rules:

    1. PolicyID (AIR) / ACCNTNUM (RMS) â€” uniform value across all rows.
       If provided globally in rules, that is applied. Else, fallback to first non-empty.

    2. LocationID (AIR) / LOCNUM (RMS) â€” serial 1-based integer per policy group.
       Rows are grouped by PolicyID/ACCNTNUM; within each group they receive
       sequential numbers 1, 2, 3 â€¦ in the order they appear.
       Single-policy SOV files get a simple 1-to-N across all rows.
    """
    from collections import defaultdict

    contract_key = "PolicyID" if target_format == "AIR" else "ACCNTNUM"
    location_key = "LocationID" if target_format == "AIR" else "LOCNUM"
    flags: List[dict] = []

    # â”€â”€ Step 1: resolve uniform policy ID â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    policy_id: Optional[str] = rules.policy_id if str(rules.policy_id).strip() else None

    if not policy_id:
        for row in rows:
            v = row.get(contract_key)
            if v and str(v).strip():
                policy_id = str(v).strip()
                break

    if not policy_id:
        for idx in range(len(rows)):
            flags.append(_make_flag(
                idx, contract_key, "missing_policy_id", None,
                f"{contract_key} is blank for all rows. "
                f"Map a source column to '{contract_key}' or provide it globally.",
            ))
    else:
        for row in rows:
            row[contract_key] = policy_id

    # â”€â”€ Step 1.5: apply global insured name (AIR only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if target_format == "AIR" and str(rules.insured_name).strip():
        for row in rows:
            if not row.get("InsuredName"):
                row["InsuredName"] = str(rules.insured_name).strip()

    # â”€â”€ Step 2: serial location numbering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Group row indexes by policy ID, then assign 1-based counters per group.
    groups: dict = defaultdict(list)
    for idx, row in enumerate(rows):
        group_key = str(row.get(contract_key) or "").strip() or "__no_policy__"
        groups[group_key].append(idx)

    for group_indices in groups.values():
        for seq_num, row_idx in enumerate(group_indices, start=1):
            rows[row_idx][location_key] = seq_num

    return rows, flags


# â”€â”€ Main entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def normalize_all_rows(
    rows: List[Dict[str, Any]],
    rules_config: BusinessRulesConfig,
    valid_currencies: Optional[set] = None,
    target_format: str = "AIR",
) -> Tuple[List[Dict[str, Any]], List[Dict]]:
    """
    Run all normalization steps over every row.
    Returns (normalized_rows, all_new_flags).
    """
    if valid_currencies is None:
        valid_currencies = set()

    all_flags: List[dict] = []
    normalized: List[Dict] = []

    for idx, row in enumerate(rows):
        row = dict(row)  # work on a copy

        row, f = _normalize_years(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_stories(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_building_count(row, idx, target_format)
        all_flags.extend(f)

        row, f = _normalize_area(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_values(row, idx, rules_config, target_format)
        all_flags.extend(f)

        row, f = _normalize_currency(row, idx, valid_currencies)
        all_flags.extend(f)

        row, f = _normalize_sprinkler(row, idx, target_format)
        all_flags.extend(f)

        row, f = _normalize_modifiers(row, idx, target_format)
        all_flags.extend(f)

        row, f = _normalize_location_name(row, idx, target_format)
        all_flags.extend(f)

        normalized.append(row)

    # â”€â”€ Bulk identity pass (runs after all per-row normalization) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    normalized, id_flags = _normalize_identity_fields(normalized, target_format, rules_config)
    all_flags.extend(id_flags)

    return normalized, all_flags

