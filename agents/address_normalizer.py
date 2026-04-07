"""
address_normalizer.py — Per-field address cleaning before geocoding.

Runs BEFORE assemble_address() in geocoder.py so Geoapify receives
clean, standardized input and returns high-confidence results.

Entry point: normalize_address_fields(row, target_format) → (row, flags)
"""
import re
import unicodedata
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("address_normalizer")

# ── Street abbreviation lookup ─────────────────────────────────────────────────
STREET_ABBREV: Dict[str, str] = {
    "aly": "Alley", "anx": "Annex", "arc": "Arcade", "ave": "Avenue",
    "blvd": "Boulevard", "bnd": "Bend", "br": "Branch", "brg": "Bridge",
    "brk": "Brook", "byp": "Bypass", "cswy": "Causeway", "ctr": "Center",
    "cir": "Circle", "clfs": "Cliffs", "cts": "Courts", "crse": "Course",
    "ct": "Court", "cv": "Cove", "crk": "Creek", "cres": "Crescent",
    "dr": "Drive", "est": "Estate", "expy": "Expressway", "ext": "Extension",
    "fls": "Falls", "fry": "Ferry", "fld": "Field", "frt": "Fort",
    "fwy": "Freeway", "gdn": "Garden", "gtwy": "Gateway", "gln": "Glen",
    "grn": "Green", "grv": "Grove", "hbr": "Harbor", "hvn": "Haven",
    "hts": "Heights", "hwy": "Highway", "hl": "Hill", "holw": "Hollow",
    "jct": "Junction", "ky": "Key", "lk": "Lake", "ln": "Lane",
    "lgt": "Light", "lf": "Loaf", "lck": "Lock", "ldg": "Lodge",
    "loop": "Loop", "mnr": "Manor", "mdw": "Meadow", "ml": "Mill",
    "mt": "Mount", "mtn": "Mountain", "orch": "Orchard", "oval": "Oval",
    "pkwy": "Parkway", "pass": "Pass", "path": "Path", "pike": "Pike",
    "pl": "Place", "pln": "Plain", "plz": "Plaza", "pt": "Point",
    "prt": "Port", "pr": "Prairie", "rpds": "Rapids", "rdg": "Ridge",
    "rd": "Road", "run": "Run", "shl": "Shoal", "shrs": "Shores",
    "skwy": "Skyway", "spgs": "Springs", "spur": "Spur", "sq": "Square",
    "sta": "Station", "st": "Street", "strm": "Stream", "trak": "Track",
    "trce": "Trace", "trl": "Trail", "tunl": "Tunnel", "tpke": "Turnpike",
    "un": "Union", "vly": "Valley", "via": "Viaduct", "vw": "View",
    "vlg": "Village", "vis": "Vista", "walk": "Walk", "xing": "Crossing",
    # common with dots
    "ave.": "Avenue", "blvd.": "Boulevard", "dr.": "Drive",
    "rd.": "Road", "st.": "Street", "ln.": "Lane", "ct.": "Court",
}

DIRECTIONALS: Dict[str, str] = {
    "n": "North", "s": "South", "e": "East", "w": "West",
    "ne": "Northeast", "nw": "Northwest", "se": "Southeast", "sw": "Southwest",
}

# Secondary info tokens to strip (trailing secondary info after street number)
_SECONDARY_PATTERN = re.compile(
    r"\b(suite|ste|apt|apartment|unit|fl|floor|bldg|building|rm|room|"
    r"wing|block|dept|department|attn|attention|c/o|care of|"
    r"near|opposite|opp|industrial park|ind park)\b[.:\s#]*[\w\-]*",
    re.IGNORECASE,
)

# Leading label pattern: strips "Attn: Something" or "C/O Someone" at START before the address number
_LEADING_LABEL_PATTERN = re.compile(
    r"^(attn|attention|c/o|care of|dept|department)\s*:?\s*[\w\s]+?(?=\d)",
    re.IGNORECASE,
)

_PO_BOX_PATTERN = re.compile(
    r"p\.?\s*o\.?\s*box\s+(\d+)", re.IGNORECASE
)

_HYPHEN_NUM_PATTERN = re.compile(r"^(\d+)-\d+\b")

_DUPE_WORD_PATTERN = re.compile(r"\b(\w+)( \1)+\b", re.IGNORECASE)


# ── City abbreviation lookup ───────────────────────────────────────────────────
CITY_ABBREV: Dict[str, str] = {
    "ft": "Fort", "st": "Saint", "mt": "Mount", "pt": "Point",
    "lk": "Lake", "pk": "Park", "spg": "Spring", "spgs": "Springs",
    "mtn": "Mountain", "vlg": "Village",
}


# ── US State name → 2-letter code ─────────────────────────────────────────────
STATE_NAME_TO_CODE: Dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT",
    "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    # DC + territories
    "district of columbia": "DC", "washington dc": "DC",
    "washington d.c.": "DC", "d.c.": "DC",
    "puerto rico": "PR", "guam": "GU", "virgin islands": "VI",
    "american samoa": "AS", "northern mariana islands": "MP",
    # Common abbreviations / misspellings
    "tex": "TX", "tex.": "TX", "calif": "CA", "cal": "CA",
    "colo": "CO", "fla": "FL", "mich": "MI", "minn": "MN",
    "miss": "MS", "n.y.": "NY", "penn": "PA", "tenn": "TN",
    "wis": "WI", "ore": "OR", "okla": "OK", "neb": "NE",
}

# Valid 2-letter US state codes
VALID_STATE_CODES = set(STATE_NAME_TO_CODE.values())


# ── Country lookup ─────────────────────────────────────────────────────────────
COUNTRY_NAME_TO_ISO2: Dict[str, str] = {
    "united states": "US", "united states of america": "US",
    "usa": "US", "u.s.a.": "US", "u.s.": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "u.k.": "GB",
    "england": "GB", "scotland": "GB", "wales": "GB",
    "great britain": "GB",
    "canada": "CA", "mexico": "MX", "australia": "AU",
    "germany": "DE", "france": "FR", "japan": "JP", "china": "CN",
    "india": "IN", "brazil": "BR", "italy": "IT", "spain": "ES",
    "netherlands": "NL", "switzerland": "CH", "sweden": "SE",
    "norway": "NO", "denmark": "DK", "finland": "FI",
    "new zealand": "NZ", "south africa": "ZA", "ireland": "IE",
    "singapore": "SG", "hong kong": "HK", "south korea": "KR",
    "korea": "KR", "taiwan": "TW", "israel": "IL", "turkey": "TR",
    "russia": "RU", "poland": "PL", "portugal": "PT", "greece": "GR",
    "austria": "AT", "belgium": "BE", "czech republic": "CZ",
    "hungary": "HU", "romania": "RO", "ukraine": "UA",
    "argentina": "AR", "chile": "CL", "colombia": "CO",
    "indonesia": "ID", "malaysia": "MY", "philippines": "PH",
    "thailand": "TH", "vietnam": "VN", "egypt": "EG",
    "nigeria": "NG", "kenya": "KE", "saudi arabia": "SA",
    "uae": "AE", "united arab emirates": "AE", "qatar": "QA",
    "kuwait": "KW", "bahrain": "BH",
}

ALPHA3_TO_ALPHA2: Dict[str, str] = {
    "USA": "US", "GBR": "GB", "CAN": "CA", "AUS": "AU",
    "DEU": "DE", "FRA": "FR", "JPN": "JP", "CHN": "CN",
    "IND": "IN", "BRA": "BR", "ITA": "IT", "ESP": "ES",
    "NLD": "NL", "CHE": "CH", "SWE": "SE", "NOR": "NO",
    "DNK": "DK", "FIN": "FI", "NZL": "NZ", "ZAF": "ZA",
    "IRL": "IE", "SGP": "SG", "HKG": "HK", "KOR": "KR",
    "TWN": "TW", "ISR": "IL", "TUR": "TR", "RUS": "RU",
    "POL": "PL", "PRT": "PT", "GRC": "GR", "AUT": "AT",
    "BEL": "BE", "CZE": "CZ", "HUN": "HU", "ROU": "RO",
    "UKR": "UA", "ARG": "AR", "CHL": "CL", "COL": "CO",
    "MEX": "MX", "SAU": "SA", "ARE": "AE", "QAT": "QA",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_flag(row_idx: int, field: str, issue: str,
               raw_value: Any, message: str) -> dict:
    return {
        "row_index": row_idx,
        "field": field,
        "issue": issue,
        "raw_value": raw_value,
        "message": message,
    }


def _to_ascii(s: str) -> str:
    """Normalize unicode characters to their closest ASCII equivalent."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


# ── 1. Street normalization ────────────────────────────────────────────────────

def normalize_street(raw: str, row_idx: int) -> Tuple[Optional[str], List[dict]]:
    """
    Clean and standardize a street address string.
    Rules: trim, remove junk chars, strip secondary info, reformat PO Box,
    expand abbreviations, expand directionals, fix hyphenated numbers,
    preserve fractions, deduplicate words.
    """
    flags: List[dict] = []
    if raw is None or str(raw).strip() == "":
        return None, flags

    s = str(raw).strip()

    # PO Box reformat: "P.O. Box 1234" → "1234 PO Box"
    po_match = _PO_BOX_PATTERN.search(s)
    if po_match:
        s = f"{po_match.group(1)} PO Box"
        return s, flags

    # Strip leading labels like "Attn: Warehouse" before a street number
    s = _LEADING_LABEL_PATTERN.sub("", s).strip()

    # Remove noise components (suite, apt, c/o, attn, near, etc.)
    s = _SECONDARY_PATTERN.sub("", s)

    # Remove special characters (keep alphanumeric, space, comma, hyphen, slash, period, fraction chars)
    s = re.sub(r"[#$%@!&*+=<>?^`~|]", "", s)

    # Hyphenated address numbers: "12-14 Main" → "12 Main"
    s = _HYPHEN_NUM_PATTERN.sub(r"\1", s)

    # Standardize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Directional expansion: only standalone tokens at word boundary
    # Process token by token to avoid replacing "North" built from "N" inside a word
    tokens = s.split()
    expanded = []
    for i, tok in enumerate(tokens):
        low = tok.rstrip(".").lower()
        # Expand directional only if it's NOT the first token (to avoid conflating with state)
        # and it looks like a standalone directional before a street name
        if low in DIRECTIONALS and i > 0:
            expanded.append(DIRECTIONALS[low])
        else:
            expanded.append(tok)
    s = " ".join(expanded)

    # Abbreviation expansion (last word or any word matching)
    tokens = s.split()
    expanded = []
    for tok in tokens:
        low = tok.rstrip(".").lower()
        if low in STREET_ABBREV:
            expanded.append(STREET_ABBREV[low])
        else:
            expanded.append(tok)
    s = " ".join(expanded)

    # Deduplicate consecutive words: "Road Road" → "Road"
    s = _DUPE_WORD_PATTERN.sub(r"\1", s)

    # Final cleanup
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r",+$", "", s).strip()

    if not s:
        return None, flags

    return s, flags


# ── 2. City normalization ──────────────────────────────────────────────────────

def normalize_city(raw: str, row_idx: int) -> Tuple[Optional[str], List[dict]]:
    """
    Clean and standardize a city name.
    Rules: trim, split from state/ZIP, expand Ft/St/Mt abbreviations,
    unicode normalize, remove duplicates, strip extra descriptions,
    preserve directional prefixes (West Palm Beach, North Chicago).
    """
    flags: List[dict] = []
    if raw is None or str(raw).strip() == "":
        return None, flags

    s = str(raw).strip()

    # If value is purely numeric → it's probably a ZIP, not a city
    if re.match(r"^\d+$", s):
        flags.append(_make_flag(row_idx, "City", "city_is_numeric",
                                raw, f"City value '{raw}' is numeric — likely a ZIP code. Field blanked."))
        return None, flags

    # Split "City, State" or "City, TX" or "City, TX 77002"
    s = re.split(r",\s*[A-Z]{2}(\s+\d{5})?$", s)[0].strip()

    # Split "City State" where state = trailing 2 uppercase letters
    s = re.sub(r"\s+[A-Z]{2}(\s+\d{5}(-\d{4})?)?$", "", s).strip()

    # Strip trailing ZIP codes
    s = re.sub(r"\s+\d{5}(-\d{4})?$", "", s).strip()

    # Strip extra descriptions after hyphen or dash: "Austin - Downtown" → "Austin"
    s = re.sub(r"\s*[-–—]\s*\w.*$", "", s).strip()

    # Strip parenthetical notes: "Houston (TX)"
    s = re.sub(r"\s*\(.*?\)", "", s).strip()

    # Detect county-level data (not a city name)
    if re.search(r"\bcounty\b", s, re.IGNORECASE):
        flags.append(_make_flag(row_idx, "City", "city_is_county",
                                raw, f"City field contains county: '{raw}'. Passing to geocoder for resolution."))
        # Keep as-is — geocoder can sometimes resolve county to city

    # Unicode normalization to handle accented chars (keep them — geocoder handles)
    # Only strip if they cause encoding issues
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = _to_ascii(s)
        flags.append(_make_flag(row_idx, "City", "city_unicode_normalized",
                                raw, "City name contained non-UTF-8 characters; converted to ASCII."))

    # Remove special characters (keep letters, numbers, spaces, hyphens, apostrophes, periods)
    s = re.sub(r"[^A-Za-z0-9 .'\-àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Expand city abbreviations (only as standalone first token)
    tokens = s.split()
    if tokens and tokens[0].rstrip(".").lower() in CITY_ABBREV:
        tokens[0] = CITY_ABBREV[tokens[0].rstrip(".").lower()]
    s = " ".join(tokens)

    # Remove duplicate trailing words: "New York City City" → "New York City"
    s = _DUPE_WORD_PATTERN.sub(r"\1", s)

    s = s.strip()
    if not s:
        return None, flags

    # Title-case
    s = s.title()

    return s, flags


# ── 3. State normalization ─────────────────────────────────────────────────────

def normalize_state(raw: str, row_idx: int,
                    country_iso: str = "US") -> Tuple[Optional[str], List[dict]]:
    """
    Normalize state to 2-letter USPS code for US, or pass through for international.
    """
    flags: List[dict] = []
    if raw is None or str(raw).strip() == "":
        return None, flags

    s = str(raw).strip()

    # If non-US, pass through as-is (international state/province)
    if country_iso.upper() not in ("US", "USA", ""):
        return s.strip(), flags

    # Remove non-letter/space characters
    s = re.sub(r"[^A-Za-z\s]", "", s)

    # Remove internal spaces between very short tokens: "T X" → "TX"
    if len(s.replace(" ", "")) <= 4:
        s = s.replace(" ", "")

    s = s.strip()

    # Already a valid 2-letter code
    if re.match(r"^[A-Za-z]{2}$", s):
        code = s.upper()
        if code in VALID_STATE_CODES:
            return code, flags
        else:
            flags.append(_make_flag(row_idx, "State", "invalid_state_code",
                                    raw, f"'{code}' is not a recognized US state code. Blanked."))
            return None, flags

    # Try full name / common abbreviation lookup
    key = s.lower().strip()
    if key in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[key], flags

    # Try "City, State" — extract state part after comma
    if "," in str(raw):
        parts = str(raw).split(",")
        state_part = re.sub(r"[^A-Za-z]", "", parts[-1]).strip().upper()
        if len(state_part) == 2 and state_part in VALID_STATE_CODES:
            return state_part, flags
        name_key = parts[-1].strip().lower()
        if name_key in STATE_NAME_TO_CODE:
            return STATE_NAME_TO_CODE[name_key], flags

    # Try extracting 2-letter code from "TX 77002" or "TX-77002"
    match = re.search(r"\b([A-Za-z]{2})\b", str(raw))
    if match:
        code = match.group(1).upper()
        if code in VALID_STATE_CODES:
            return code, flags

    # Give up
    flags.append(_make_flag(row_idx, "State", "unresolvable_state",
                            raw, f"Could not resolve '{raw}' to a valid US state code. Blanked."))
    return None, flags


# ── 4. Postal code normalization ───────────────────────────────────────────────

def normalize_postal(raw: str, row_idx: int,
                     country_iso: str = "US") -> Tuple[Optional[str], List[dict]]:
    """
    Normalize postal code.
    US: enforce 5-digit string (preserving leading zeros).
    International: trim and clean only.
    """
    flags: List[dict] = []
    if raw is None or str(raw).strip() == "":
        flags.append(_make_flag(row_idx, "PostalCode", "missing_postal",
                                None, "Postal code is blank. Will attempt geocoding with other address fields."))
        return None, flags

    s = str(raw).strip()

    # Non-US: just clean and return as string
    if country_iso.upper() not in ("US", "USA", ""):
        s = re.sub(r"[^A-Za-z0-9 \-]", "", s).strip()
        return s if s else None, flags

    # Letters in ZIP: standalone mixed token like "77A02" — detect BEFORE digit extraction
    # (stripping digits from 77A02 gives 7702 which would be wrongly padded to 07702)
    if re.match(r"^[0-9A-Za-z]+$", s) and re.search(r"[A-Za-z]", s) and re.search(r"\d", s):
        flags.append(_make_flag(row_idx, "PostalCode", "invalid_zip_has_letters",
                                raw, f"ZIP '{raw}' contains letters. Blanked; geocoder will use address."))
        return None, flags

    # For US: try to extract a 5-digit ZIP from combined fields
    # "Dallas, TX 77002" or "TX 77002" or "77002-1234"
    zip_match = re.search(r"\b(\d{5})\b", s)
    if zip_match:
        code = zip_match.group(1)
        return code, flags

    # Remove all non-numeric characters
    digits_only = re.sub(r"[^0-9]", "", s)

    if not digits_only:
        # Letters in zip with no numeric block → blank + flag
        flags.append(_make_flag(row_idx, "PostalCode", "invalid_zip_has_letters",
                                raw, f"ZIP '{raw}' contains letters with no extractable numeric ZIP. Blanked."))
        return None, flags

    # Handle ZIP+4 without dash: "770021234" → "77002"
    if len(digits_only) == 9:
        code = digits_only[:5]
        flags.append(_make_flag(row_idx, "PostalCode", "zip_plus4_stripped",
                                raw, f"ZIP+4 '{raw}' truncated to '{code}'."))
        return code, flags

    # Too long
    if len(digits_only) > 5:
        code = digits_only[:5]
        flags.append(_make_flag(row_idx, "PostalCode", "zip_truncated",
                                raw, f"ZIP '{raw}' ({len(digits_only)} digits) truncated to '{code}'."))
        return code, flags

    # Too short — pad with leading zeros
    if len(digits_only) < 5:
        code = digits_only.zfill(5)
        flags.append(_make_flag(row_idx, "PostalCode", "zip_padded",
                                raw, f"ZIP '{raw}' padded to '{code}' (leading zero restored)."))
        return code, flags

    return digits_only, flags


# ── 5. Country normalization ───────────────────────────────────────────────────

def normalize_country(raw: str, row_idx: int) -> Tuple[Optional[str], List[dict]]:
    """
    Normalize country to ISO 3166-1 alpha-2 (2-letter uppercase code).
    Handles: alpha-2, alpha-3, full country names, common aliases.
    """
    flags: List[dict] = []
    if raw is None or str(raw).strip() == "":
        return None, flags

    s = str(raw).strip()

    # Already alpha-2
    if re.match(r"^[A-Za-z]{2}$", s):
        code = s.upper()
        return code, flags

    # Alpha-3
    if re.match(r"^[A-Za-z]{3}$", s):
        code = s.upper()
        if code in ALPHA3_TO_ALPHA2:
            return ALPHA3_TO_ALPHA2[code], flags
        # Unknown alpha-3 — return as-is with flag
        flags.append(_make_flag(row_idx, "CountryISO", "unknown_alpha3_country",
                                raw, f"Alpha-3 code '{code}' not in lookup — passed through."))
        return code, flags

    # Full name or alias lookup
    key = s.lower().strip()
    if key in COUNTRY_NAME_TO_ISO2:
        return COUNTRY_NAME_TO_ISO2[key], flags

    # Partial match — try removing punctuation
    key_clean = re.sub(r"[^a-z ]", "", key).strip()
    if key_clean in COUNTRY_NAME_TO_ISO2:
        return COUNTRY_NAME_TO_ISO2[key_clean], flags

    flags.append(_make_flag(row_idx, "CountryISO", "unresolvable_country",
                            raw, f"Cannot resolve '{raw}' to an ISO alpha-2 country code."))
    return s.upper()[:2] if len(s) >= 2 else None, flags


# ── 6. Lat/Lon normalization ───────────────────────────────────────────────────

def _dms_to_decimal(dms_str: str) -> Optional[float]:
    """
    Convert DMS string to decimal degrees.
    Handles: "45-31-07.2 N", "45°31.120'N", "45°31'07.2\"N", "N45.5186"
    """
    s = str(dms_str).strip()

    # Strip degree/minute/second symbols
    s_clean = re.sub(r"[°′″\"']", " ", s)

    # Determine sign from compass direction
    direction_match = re.search(r"[NSEWnsew]", s_clean)
    direction = direction_match.group(0).upper() if direction_match else None
    s_clean = re.sub(r"[NSEWnsew]", "", s_clean).strip()

    # Parse numeric parts separated by spaces or hyphens
    parts = re.split(r"[\s\-]+", s_clean.strip())
    parts = [p for p in parts if p]

    try:
        if len(parts) == 1:
            decimal = float(parts[0])
        elif len(parts) == 2:
            decimal = float(parts[0]) + float(parts[1]) / 60.0
        elif len(parts) == 3:
            decimal = float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
        else:
            return None
    except ValueError:
        return None

    if direction in ("S", "W"):
        decimal = -decimal

    return decimal


def normalize_latlon(lat_raw: Any, lon_raw: Any,
                     row_idx: int,
                     country_iso: str = "") -> Tuple[Optional[float], Optional[float], List[dict]]:
    """
    Validate and clean latitude and longitude values.
    Handles DMS conversion, sign correction, reversal detection, zero coords.
    """
    flags: List[dict] = []

    def _parse(v: Any) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip().strip('"')
        if not s:
            return None
        # Try direct float
        try:
            return float(s.replace(",", "."))
        except ValueError:
            pass
        # Try DMS
        return _dms_to_decimal(s)

    lat = _parse(lat_raw)
    lon = _parse(lon_raw)

    # Check for non-convertible
    if lat_raw is not None and str(lat_raw).strip() and lat is None:
        flags.append(_make_flag(row_idx, "Latitude", "non_numeric_coordinates",
                                lat_raw, f"Cannot parse latitude '{lat_raw}'. Do NOT copy to template."))
        return None, None, flags

    if lon_raw is not None and str(lon_raw).strip() and lon is None:
        flags.append(_make_flag(row_idx, "Longitude", "non_numeric_coordinates",
                                lon_raw, f"Cannot parse longitude '{lon_raw}'. Do NOT copy to template."))
        return None, None, flags

    # Only one coordinate present
    if (lat is None) != (lon is None):
        flags.append(_make_flag(row_idx, "Latitude", "missing_lat_or_lon",
                                (lat_raw, lon_raw), "Only one coordinate provided. Both set to None."))
        return None, None, flags

    if lat is None and lon is None:
        return None, None, flags

    # (0, 0) check — Gulf of Guinea, almost certainly wrong
    if lat == 0.0 and lon == 0.0:
        flags.append(_make_flag(row_idx, "Latitude", "coordinates_zero",
                                (0, 0), "Coordinates are (0, 0). Likely invalid. Set to None."))
        return None, None, flags

    # Detect reversed columns: lat value looks like longitude and vice versa
    if abs(lat) > 90 and abs(lon) <= 90:
        lat, lon = lon, lat
        flags.append(_make_flag(row_idx, "Latitude", "coordinates_reversed",
                                (lat_raw, lon_raw), "Lat/Lon appear reversed — swapped automatically."))

    # Range validation
    if not (-90 <= lat <= 90):
        flags.append(_make_flag(row_idx, "Latitude", "latitude_out_of_range",
                                lat, f"Latitude {lat} outside [-90, 90]. Do NOT copy to template."))
        return None, None, flags

    if not (-180 <= lon <= 180):
        flags.append(_make_flag(row_idx, "Longitude", "longitude_out_of_range",
                                lon, f"Longitude {lon} outside [-180, 180]. Do NOT copy to template."))
        return None, None, flags

    # US longitude sign check: US longitudes should be negative (Western hemisphere)
    if country_iso.upper() in ("US", "USA") and lon > 0:
        lon = -lon
        flags.append(_make_flag(row_idx, "Longitude", "positive_longitude_in_us",
                                lon_raw, f"Positive longitude in US location — negated to {lon}."))

    # Low-precision warning
    lat_str = str(lat_raw) if lat_raw is not None else ""
    if "." in lat_str and len(lat_str.split(".")[-1]) < 3:
        flags.append(_make_flag(row_idx, "Latitude", "low_precision_coordinates",
                                lat_raw, "Latitude has fewer than 3 decimal places — may be low precision."))

    return round(lat, 7), round(lon, 7), flags


# ── Entry point ────────────────────────────────────────────────────────────────

def normalize_address_fields(row: dict, row_idx: int,
                              target_format: str = "AIR") -> Tuple[dict, List[dict]]:
    """
    Run all address field normalizers on a single row.
    Reads from canonical AIR or RMS key names (already resolved by column_mapper).
    Writes cleaned values back into the same canonical keys.
    Returns updated row + all flags raised.
    """
    all_flags: List[dict] = []
    row = dict(row)  # work on a copy

    # ── Key selection based on target format ──────────────────────────────────
    if target_format == "AIR":
        street_key   = "Street"
        city_key     = "City"
        state_key    = "Area"
        postal_key   = "PostalCode"
        country_key  = "CountryISO"
    else:
        street_key   = "STREETNAME"
        city_key     = "CITY"
        state_key    = "STATECODE"
        postal_key   = "POSTALCODE"
        country_key  = "CNTRYCODE"

    lat_key = "Latitude"
    lon_key = "Longitude"

    # ── Country first (needed by state and postal normalizers) ────────────────
    country_raw = row.get(country_key)
    country_iso, f = normalize_country(country_raw, row_idx)
    row[country_key] = country_iso
    all_flags.extend(f)
    effective_country = country_iso or ""

    # ── Street ────────────────────────────────────────────────────────────────
    street_val, f = normalize_street(row.get(street_key), row_idx)
    row[street_key] = street_val
    all_flags.extend(f)

    # ── City ──────────────────────────────────────────────────────────────────
    city_val, f = normalize_city(row.get(city_key), row_idx)
    row[city_key] = city_val
    all_flags.extend(f)

    # ── State ─────────────────────────────────────────────────────────────────
    state_val, f = normalize_state(row.get(state_key), row_idx, effective_country)
    row[state_key] = state_val
    all_flags.extend(f)

    # ── Postal code ───────────────────────────────────────────────────────────
    postal_val, f = normalize_postal(row.get(postal_key), row_idx, effective_country)
    row[postal_key] = postal_val
    all_flags.extend(f)

    # ── Lat / Lon ─────────────────────────────────────────────────────────────
    lat, lon, f = normalize_latlon(
        row.get(lat_key), row.get(lon_key),
        row_idx, effective_country
    )
    row[lat_key] = lat
    row[lon_key] = lon
    all_flags.extend(f)

    return row, all_flags


def normalize_addresses(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Normalizes addresses in raw rows before column mapping.
    Dynamically identifies address-related columns, cleans them,
    and builds a combined address for subsequent geocoding.
    """
    all_flags = []
    normalized = []
    
    if not rows:
        return [], []
        
    keys = list(rows[0].keys())
    
    def find_col(*substrings: str) -> Optional[str]:
        for k in keys:
            k_low = re.sub(r'[^a-z]', '', k.lower())
            if any(sub in k_low for sub in substrings):
                return k
        return None

    address_col = find_col("address", "fulladdress")
    street_col  = find_col("street", "address1", "address2")
    city_col    = find_col("city", "town")
    county_col  = find_col("county")
    state_col   = find_col("state", "area", "province")
    zip_col     = find_col("zip", "postal", "postcode")
    country_col = find_col("country", "nation")
    lat_col     = find_col("lat", "latitude")
    lon_col     = find_col("lon", "lng", "longitude")

    for idx, row in enumerate(rows):
        row_out = dict(row)
        
        addr_val    = str(row.get(address_col, "")).strip() if address_col else ""
        street_val  = str(row.get(street_col, "")).strip() if street_col else ""
        city_val    = str(row.get(city_col, "")).strip() if city_col else ""
        county_val  = str(row.get(county_col, "")).strip() if county_col else ""
        state_val   = str(row.get(state_col, "")).strip() if state_col else ""
        zip_val     = str(row.get(zip_col, "")).strip() if zip_col else ""
        country_val = str(row.get(country_col, "")).strip() if country_col else ""

        # Normalize in place
        c_iso, f = normalize_country(country_val, idx) if country_val else (None, [])
        if country_col and c_iso is not None: row_out[country_col] = c_iso
        all_flags.extend(f)

        s_val, f = normalize_street(street_val, idx) if street_val else (None, [])
        if street_col and s_val is not None: row_out[street_col] = s_val
        all_flags.extend(f)

        ct_val, f = normalize_city(city_val, idx) if city_val else (None, [])
        if city_col and ct_val is not None: row_out[city_col] = ct_val
        all_flags.extend(f)

        st_val, f = normalize_state(state_val, idx, c_iso or "") if state_val else (None, [])
        if state_col and st_val is not None: row_out[state_col] = st_val
        all_flags.extend(f)

        zp_val, f = normalize_postal(zip_val, idx, c_iso or "") if zip_val else (None, [])
        if zip_col and zp_val is not None: row_out[zip_col] = zp_val
        all_flags.extend(f)
        
        if lat_col and lon_col:
            lat_v, lon_v, f = normalize_latlon(row.get(lat_col), row.get(lon_col), idx, c_iso or "")
            if lat_col and lat_v is not None: row_out[lat_col] = lat_v
            if lon_col and lon_v is not None: row_out[lon_col] = lon_v
            all_flags.extend(f)

        # Build FullAddress properly (combining them)
        parts = []
        if addr_val:
            parts.append(addr_val)
        else:
            if s_val or street_val: parts.append(s_val or street_val)
            if ct_val or city_val: parts.append(ct_val or city_val)
            if county_val: parts.append(county_val)
            if st_val or state_val: parts.append(st_val or state_val)
            if zp_val or zip_val: parts.append(zp_val or zip_val)
            if c_iso or country_val: parts.append(c_iso or country_val)
            
        full_address = ", ".join(p for p in parts if p)
        row_out["_CombinedAddress"] = full_address

        normalized.append(row_out)
        
    return normalized, all_flags
