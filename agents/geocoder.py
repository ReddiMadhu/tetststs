"""
geocoder.py â€” Geoapify geocoding with LRU cache and ISO 3166 state validation.
"""
import httpx
import json
import logging
import os
import pathlib
import re
import unicodedata
from functools import lru_cache
from typing import Dict, Optional

from agents.address_normalizer import normalize_address_fields

logger = logging.getLogger("geocoder")

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "")
GEOAPIFY_URL = "https://api.geoapify.com/v1/geocode/search"
GEOCODE_TIMEOUT = 8.0

# â”€â”€ Reference data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_REF_DIR = pathlib.Path(__file__).parent.parent / "reference"
_iso3166: Dict[str, str] = {}   # state_name_lower â†’ state_code
_iso3166r: Dict[str, str] = {}  # state_code_upper â†’ state_name
_alpha3_to_alpha2: Dict[str, str] = {}


def load_reference_data() -> None:
    """Load ISO reference data from JSON files into module-level dicts."""
    global _iso3166, _iso3166r, _alpha3_to_alpha2

    states_path = _REF_DIR / "iso3166_states.json"
    if states_path.exists():
        data = json.loads(states_path.read_text(encoding="utf-8"))
        # Expected shape: {"US": {"CA": "California", ...}, ...}
        # or flat {"California": "CA", ...}
        for country_or_code, value in data.items():
            if isinstance(value, dict):
                # Nested: country_code â†’ {state_code: state_name}
                for code, name in value.items():
                    _iso3166[name.lower()] = code.upper()
                    _iso3166r[code.upper()] = name
            elif isinstance(value, str):
                # Flat: state_name â†’ code
                _iso3166[country_or_code.lower()] = value.upper()
                _iso3166r[value.upper()] = country_or_code
        # Also load the alpha-3 â†’ alpha-2 section if present
        alpha3 = data.get("_alpha3_to_alpha2", {})
        _alpha3_to_alpha2.update(alpha3)

    logger.info(f"Loaded {len(_iso3166)} ISO-3166 state entries")


# â”€â”€ Address normalisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize_address(raw: str) -> str:
    """Lowercase, remove double spaces, strip punctuation except commas/hyphens."""
    raw = raw.strip().lower()
    raw = re.sub(r"[^\w\s,\-]", " ", raw)
    raw = re.sub(r"\s{2,}", " ", raw)
    return raw


def assemble_address(row: dict, target_format: str = "AIR") -> Optional[str]:
    """
    Build an address string from a cleaned row dict for geocoding.
    Supports both AIR and RMS canonical key names.
    """
    def val(*keys: str) -> str:
        for k in keys:
            v = row.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return ""

    full = val("_CombinedAddress", "FullAddress", "Address")
    if full:
        return full

    if target_format == "AIR":
        parts = [
            val("Street"),
            val("City"),
            val("County"),
            val("Area", "State"),
            val("PostalCode"),
            val("CountryISO"),
        ]
    else:  # RMS
        parts = [
            val("STREETNAME"),
            val("CITY"),
            val("COUNTY"),
            val("STATECODE"),
            val("POSTALCODE"),
            val("CNTRYCODE"),
        ]

    assembled = ", ".join(p for p in parts if p)
    return assembled if assembled else None


# â”€â”€ Geocoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@lru_cache(maxsize=2000)
def geocode_address(normalized_address: str) -> dict:
    """
    Geocode a single normalized address string via Geoapify.
    Result is fully serialisable (all plain Python types) so it can be cached.
    """
    if not normalized_address or not normalized_address.strip():
        return {"status": "INSUFFICIENT_ADDRESS", "source": "Failed"}

    if not GEOAPIFY_API_KEY:
        return {"status": "NO_API_KEY", "source": "Failed"}

    params = {
        "text": normalized_address,
        "format": "json",
        "limit": 1,
        "lang": "en",
        "apiKey": GEOAPIFY_API_KEY,
    }

    for attempt in range(3):
        try:
            with httpx.Client(timeout=GEOCODE_TIMEOUT) as client:
                resp = client.get(GEOAPIFY_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            break
        except httpx.TimeoutException:
            if attempt == 2:
                return {"status": "TIMEOUT", "source": "Failed"}
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500 and attempt < 2:
                import time; time.sleep(1 * (attempt + 1))
                continue
            return {"status": f"HTTP_{exc.response.status_code}", "source": "Failed"}
        except Exception as exc:
            return {"status": f"ERROR:{exc}", "source": "Failed"}

    results = data.get("results", [])
    if not results:
        return {"status": "NO_RESULTS", "source": "Failed"}

    r = results[0]
    state_name = r.get("state", "")
    state_code = _resolve_state_code(state_name)

    # Fallback to ISO-3166-2 for state code (e.g. "DE-BE" -> "BE") if primary state_name missing
    iso2 = r.get("iso3166_2", "")
    if not state_code and iso2 and "-" in iso2:
        state_code = iso2.split("-")[-1].upper()

    country_raw = (r.get("country_code") or "").strip()
    country_iso = _resolve_country_alpha2(country_raw)

    return {
        "status": "OK",
        "source": "Geocoded",
        "latitude": r.get("lat"),
        "longitude": r.get("lon"),
        "street": _join_street(r),
        "city": r.get("city") or r.get("town") or r.get("village") or r.get("municipality") or "",
        "county": r.get("county", ""),
        "state": state_name,
        "statecode": state_code,
        "postcode": r.get("postcode", ""),
        "country_iso": country_iso,
        "formatted": r.get("formatted", ""),
        "confidence": r.get("rank", {}).get("confidence", 0),
        "state_code_validation": _validate_state_code(state_code),
    }


def _join_street(r: dict) -> str:
    housenumber = r.get("housenumber", "") or ""
    street = r.get("street", "") or ""
    return f"{housenumber} {street}".strip()


def _clean_street_fallback(raw_input, city, postcode):
    """
    Strip matched city and postcode from raw input to isolate the street part
    when the geocoder didn't return a specific street component.
    """
    if not raw_input:
        return ""
    s = str(raw_input).strip()

    # Remove trailing country if it exists
    s = re.sub(r",\s*[A-Za-z\s]+$", "", s)

    # Remove postcode
    if postcode:
        s = re.sub(rf"\b{re.escape(str(postcode))}\b", "", s, flags=re.IGNORECASE)

    # Remove city
    if city:
        s = re.sub(rf"\b{re.escape(str(city))}\b", "", s, flags=re.IGNORECASE)

    # Clean up double commas, trailing commas, and extra spaces
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,\s]+$", "", s)
    s = re.sub(r"^[,\s]+", "", s)

    return s.strip()


def _resolve_state_code(state_name: str) -> str:
    if not state_name:
        return ""
    key = state_name.strip().lower()
    return _iso3166.get(key, state_name[:2].upper() if len(state_name) >= 2 else state_name)


def _resolve_country_alpha2(raw: str) -> str:
    """Convert alpha-2 or alpha-3 country code â†’ alpha-2."""
    upper = raw.upper().strip()
    if len(upper) == 2:
        return upper
    if upper in _alpha3_to_alpha2:
        return _alpha3_to_alpha2[upper]
    return upper


def _validate_state_code(code: str) -> str:
    if not code:
        return "MISSING"
    if code.upper() in _iso3166r:
        return "VALID"
    if re.match(r"^[A-Z]{2}$", code.upper()):
        return "UNRECOGNIZED"
    return "INVALID_FORMAT"


# ── Row-level geocoding decision ───────────────────────────────────────────────

def process_row_geocoding(row: dict, column_map: dict,
                           target_format: str = "AIR") -> dict:
    """
    Apply the geocoding decision tree to one row.
    1. If valid coordinates already present, trust them and normalize.
    2. Otherwise assemble address string and geocode via Geoapify.
    3. Normalize the extracted Geoapify components via address_normalizer.
    Returns a dict of geo fields to merge into the row.
    """
    row_idx = row.get("_row_index", 0)

    lat = row.get("Latitude")
    lon = row.get("Longitude")

    # ── Step 1: Valid coordinates already present — trust them ────────────────
    if _is_valid_lat(lat) and _is_valid_lon(lon):
        res = {
            "Latitude":        float(lat),
            "Longitude":       float(lon),
            "Geosource":       "Provided",
            "GeocodingStatus": "PROVIDED",
        }
        final_res, _addr_flags = normalize_address_fields({**row, **res}, row_idx, target_format)
        output_keys = ["Latitude", "Longitude", "Street", "City", "Area", "PostalCode", "CountryISO", 
                       "STREETNAME", "CITY", "STATECODE", "POSTALCODE", "CNTRYCODE", 
                       "Geosource", "GeocodingStatus", "Geo_Confidence", "StateCodeValidation"]
        return {k: v for k, v in final_res.items() if k in output_keys}

    # ── Step 2: Assemble address and geocode ───────────────────────────
    address_raw = assemble_address(row, target_format)
    if not address_raw:
        return {
            "Geosource":       "Failed",
            "GeocodingStatus": "INSUFFICIENT_ADDRESS",
        }

    normalized = _normalize_address(address_raw)
    result = geocode_address(normalized)

    if result["status"] != "OK":
        res = {
            "Geosource":       "Failed",
            "GeocodingStatus": result["status"],
        }
        # In AIR, if only FullAddress was given and API fails, copy it to Street so the schema is not empty
        if target_format == "AIR" and not row.get("Street"):
            res["Street"] = row.get("FullAddress", "") or row.get("_CombinedAddress", "") or row.get("Address", "")
        return res

    # ── Step 3: Map Geoapify result back to canonical key names ───────────────
    if target_format == "AIR":
        # Determine isolated street if geocoder didn't return a specific one
        final_street = result["street"]
        if not final_street:
            # Fallback: clean the input string of city/postcode
            raw_in = row.get("Street") or row.get("FullAddress", "") or row.get("_CombinedAddress", "") or row.get("Address", "")
            final_street = _clean_street_fallback(raw_in, result["city"], result["postcode"])

        res = {
            "Latitude":          result["latitude"],
            "Longitude":         result["longitude"],
            "Street":            final_street,
            "City":              result["city"] or row.get("City", ""),
            "Area":              result["statecode"] or row.get("Area", ""),
            "PostalCode":        result["postcode"] or row.get("PostalCode", ""),
            "CountryISO":        result["country_iso"] or row.get("CountryISO", ""),
            "GeocodingStatus":   "OK",
            "Geosource":         "Geocoded",
            "Geo_Confidence":    result["confidence"],
            "StateCodeValidation": result["state_code_validation"],
        }
    else:  # RMS
        final_street = result["street"]
        if not final_street:
            raw_in = row.get("STREETNAME") or row.get("FullAddress", "") or row.get("_CombinedAddress", "") or row.get("Address", "")
            final_street = _clean_street_fallback(raw_in, result["city"], result["postcode"])

        res = {
            "Latitude":          result["latitude"],
            "Longitude":         result["longitude"],
            "STREETNAME":        final_street,
            "CITY":              result["city"] or row.get("CITY", ""),
            "STATECODE":         result["statecode"] or row.get("STATECODE", ""),
            "POSTALCODE":        result["postcode"] or row.get("POSTALCODE", ""),
            "CNTRYCODE":         result["country_iso"] or row.get("CNTRYCODE", ""),
            "GeocodingStatus":   "OK",
            "Geosource":         "Geocoded",
            "Geo_Confidence":    result["confidence"],
            "StateCodeValidation": result["state_code_validation"],
        }

    # ── Step 4: Run Address Normalization strictly on the cleanly extracted fields ───
    final_res, _addr_flags = normalize_address_fields({**row, **res}, row_idx, target_format)
    
    # We only want to return the newly generated geo fields and cleaned address fields.
    output_keys = ["Latitude", "Longitude", "Street", "City", "Area", "PostalCode", "CountryISO", 
                   "STREETNAME", "CITY", "STATECODE", "POSTALCODE", "CNTRYCODE", 
                   "Geosource", "GeocodingStatus", "Geo_Confidence", "StateCodeValidation"]
    
    return {k: v for k, v in final_res.items() if k in output_keys}


def _is_valid_lat(v) -> bool:
    try:
        f = float(v)
        return -90 <= f <= 90 and f != 0
    except (TypeError, ValueError):
        return False


def _is_valid_lon(v) -> bool:
    try:
        f = float(v)
        return -180 <= f <= 180 and f != 0
    except (TypeError, ValueError):
        return False

