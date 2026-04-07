"""
column_mapper.py â€” Fuzzy + Gemini LLM column mapping using a LangGraph pipeline.

Graph: fuzzy_matching â†’ extract_llm_candidates â†’ llm_matching â†’ merge_results

Memory layer: before the graph runs, mapping_memory.lookup_memory() short-circuits
any source column that was previously confirmed by a human â€” returning score=1.0
with method="memory" and skipping fuzzy/LLM entirely for those columns.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

import google.generativeai as genai
from langgraph.graph import END, StateGraph
from rapidfuzz import fuzz, process

import agents.cat.mapping_memory as mapping_memory

logger = logging.getLogger("column_mapper")

# â”€â”€ Gemini setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
_gemini_model = genai.GenerativeModel(
    "gemini-3.1-flash-lite-preview",
    generation_config={"response_mime_type": "application/json"},
)

# â”€â”€ AIR canonical fields + aliases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
AIR_CANONICAL_FIELDS: Dict[str, List[str]] = {
    "PolicyID":             ["contract id", "contract_id", "account id", "policy id", "policy number", "accntnum"],
    "InsuredName":          ["insured name", "insured", "client", "customer name", "company name", "insured client"],
    "LocationID":           ["location id", "loc id", "loc_id", "location number", "loc no", "locid", "locnum"],
    "LocationName":         [
        "location name", "location description", "location",
        "site name", "site description",
        "premises name", "premises",
        "warehouse name", "office location", "development name",
        "risk name", "risk description",
        "property name", "insured location", "named location",
        "facility name", "plant name", "terminal name",
        "hoa name", "condo association name",
        "branch name", "store name", "project name",
        "asset name", "campus name", "complex name",
        "apartment community", "community name",
        "building name", "loc name", "locname"
    ],
    "FullAddress":          ["full address", "address full", "complete address", "full_address",
                             "full location", "combined address"],
    "Street":               ["street", "street address", "address 1", "address1", "addr1", "street1",
                             "addr", "streetname", "address line 1", "addr line 1", "street name",
                             "risk address", "property address", "site address", "insured address",
                             "location address", "premise address"],
    "City":                 ["city", "town", "municipality", "locale", "city name", "town name",
                             "locality", "risk city", "site city", "property city", "insured city",
                             "branch city", "cty", "cty name", "loc city", "addr city"],
    "County":               ["county", "parish", "borough", "district"],
    "Area":                 ["area", "state", "province",
                             "region", "statecode", "state code", "state cd", "state abbr", "st",
                             "st code", "st cd", "prov cd", "province code", "region state",
                             "risk state", "loc state", "site state", "addr state",
                             "city state", "state zip"],
    "PostalCode":           ["postal code", "zip", "zip code", "postcode", "postal", "postalcode",
                             "zip5", "zip cd", "postal cd", "post code", "mail zip", "loc zip",
                             "risk zip", "addr zip", "zip plus4", "zipcode", "zip5"],
    "CountryISO":           ["country", "country code", "iso country", "cntry", "nation", "cntrycode",
                             "country name", "iso2", "iso 2", "iso code", "cntry cd", "cntry code",
                             "loc country", "risk country", "addr country", "nation code",
                             "country abbr", "iso country code"],
    "Latitude":             ["latitude", "lat", "y coordinate", "y_coord", "gps lat",
                             "latitude dd", "lat dd", "y coord", "decimal lat"],
    "Longitude":            ["longitude", "lon", "lng", "x coordinate", "x_coord", "gps lon",
                             "longitude dd", "lon dd", "long", "x coord", "decimal lon"],
    "OccupancyCodeType":    ["occ scheme", "occupancy scheme", "occ_scheme", "occscheme"],
    "OccupancyCode":        ["occ code", "occupancy code", "occ type", "occtype", "occ", "occupancy",
                             "building use", "use type", "property use",
                             "utility", "occupancy description", "building use description", "risk use"],
    "ConstructionCodeType": ["const scheme", "construction scheme", "bldgscheme", "constr scheme"],
    "ConstructionCode":     ["construction", "const code", "constr code", "bldgclass", "building class",
                             "building type", "construction type", "struct type", "class",
                             "structure type", "structural type", "building material",
                             "construction description", "structural system"],
    "RiskCount":            [
        "risk count",
        "num risks",
        "count",
        "number of risks",
        "no of risks",
        "numbldgs",
        "no of buildings",
        "num buildings",
        "# of buildings",
        "# structures on site",
        "asset count",
        "bldcnt",
        "bldg count",
        "bldg indicator",
        "bldg qty",
        "bldg total",
        "bldg_count",
        "bldg_count_loc",
        "bldg_no",
        "bldg_num",
        "bldg_per_loc",
        "bldg_qty",
        "bldg_qty_loc",
        "bldgcnt",
        "bldgcnt_loc",
        "bldgnum",
        "bldgs",
        "blocks",
        "building count",
        "building exposure count",
        "building indicator",
        "building quantity",
        "buildings",
        "buildings per asset",
        "campus building count",
        "campus structures",
        "classrooms",
        "complex building count",
        "complex buildings",
        "facility count",
        "insured buildings",
        "insured structure count",
        "loc_bldg_cnt",
        "loc_bldgs",
        "locbldg",
    ],
    "NumberOfStories":      [
        "# stories",
        "number of stories",
        "no. of stories",
        "stories",
        "story count",
        "total stories",
        "floor count",
        "no. floors",
        "total levels",
        "level count",
        "stry",
        "strys",
        "stry_cnt",
        "strynum",
        "no_stry",
        "nostry",
        "bldg_stry",
        "str_cnt",
        "story_no",
        "stry_qty",
        "above grade stories",
        "stories (above grade)",
        "bldg height (stories)",
        "structural height (stories)",
        "max story count",
        "no of stories",
        "num stories",
        "floors",
    ],
    "GrossArea":            [
        "gross area",
        "gross sq ft",
        "gross square footage",
        "gross sf",
        "total area",
        "building area",
        "total building area",
        "total sq ft",
        "total square feet",
        "bldg area",
        "building sq ft",
        "area (sq ft)",
        "grs_area",
        "grs_sqft",
        "gsf",
        "tla",
        "tba",
        "sqft",
        "square_footage",
        "area_sf",
        "bldg_sf",
        "bldgarea",
        "tot_area",
        "tot_sf",
        "exp_area",
        "area_value",
        "gla",
        "floor area",
        "floorarea",
    ],
    "YearBuilt":          [
        "year built",
        "yr built",
        "built year",
        "year of construction",
        "construction year",
        "year constructed",
        "yob",
        "yearbuilt",
        "yr_built",
        "const_year",
        "year_const",
        "build_year",
        "original year",
        "orig year",
        "year_built",
    ],
    "YearRetrofitted":    [
        "year upgrade",
        "yr upgrade",
        "upgrade year",
        "year renovated",
        "renovation year",
        "major renovation year",
        "full retrofit year",
        "rehab year",
        "year modernized",
        "yearupgrad",
        "yr_upgrade",
        "upgrade_year",
        "renovation_year",
        "year seismically retrofitted",
        "seismic retrofit year",
        "seismic upgrade year",
        "seismically upgraded",
        "year retrofitted",
        "structural retrofit year",
        "major reno",
        "fully rehabbed",
        "fully modernized",
        "brought up to code",
        "complete reconstruction",
        "building rebuilt",
        "full structural upgrade",
    ],
    "TIV":                     [
        "tiv",
        "total insured value",
        "total insurable value",
        "total exposure",
        "total value",
        "sum insured",
        "total sum insured",
        "tiv total",
    ],
    "BuildingValue":          [
        "bldg",
        "bldg value",
        "building limit",
        "building sum insured",
        "replacement cost",
        "rc",
        "rebuild",
        "reinstatement value",
        "real property",
        "structure",
        "real property value",
        "bldg incl contents",
        "pito",
        "hard cost",
        "land improvements",
        "building value",
        "building tiv",
        "bldg tiv",
    ],
    "ContentsValue":          [
        "content value",
        "limit",
        "contents coverage",
        "declared",
        "reported values",
        "personal property",
        "bpp",
        "movable assets",
        "tenant contents",
        "furniture & fixtures",
        "f&f",
        "machinery",
        "equipment",
        "inventory",
        "edp",
        "fine arts",
        "signs",
        "contents value",
        "contents tiv",
    ],
    "TimeElementValue":          [
        "bi",
        "time element",
        "time element tiv",
        "bi limit",
        "bi sum insured",
        "declared bi",
        "gross profit",
        "gross earnings",
        "net profit",
        "loss of income",
        "revenue",
        "sales",
        "turnover",
        "operating income",
        "rents",
        "loss",
        "profit",
        "payroll",
        "rental value",
        "extra expense",
        "accounts receivable",
        "business interruption",
        "bi value",
        "te value",
    ],
    "Currency":             ["currency", "curr", "currency code"],
    "LineOfBusiness":       ["line of business", "lob", "line", "business line"],
    "SprinklerSystem":      ["sprinkler", "sprinkler type", "fire suppression", "spk", "sprinklered"],
    "RoofGeometry":         ["roof type", "roof cover", "roof material", "roofing", "roof geometry", "roofgeom"],
    "FoundationType":       ["foundation", "foundation type"],
    "WallSiding":           ["wall siding", "cladding", "exterior wall", "wall material", "cladding type"],
    "SoftStory":            ["soft story", "soft storey", "soft_story", "softstory"],
    "WallType":             ["wall type", "wall"],
}

RMS_CANONICAL_FIELDS: Dict[str, List[str]] = {
    "ACCNTNUM":     ["account number", "account id", "accnt num", "accntnum", "contract id"],
    "LOCNUM":       ["loc id", "location id", "risk id", "loc no", "locnum", "locid"],
    "LOCNAME":              [
        "location name", "location description", "location",
        "site name", "site description",
        "premises name", "premises",
        "warehouse name", "office location", "development name",
        "risk name", "risk description",
        "property name", "insured location", "named location",
        "facility name", "plant name", "terminal name",
        "hoa name", "condo association name",
        "branch name", "store name", "project name",
        "asset name", "campus name", "complex name",
        "apartment community", "community name",
        "building name", "loc name", "locname"
    ],
    "STREETNAME":   ["street", "address", "street address", "streetname", "addr",
                     "street name", "addr line 1", "address 1", "address1"],
    "CITY":         ["city", "town", "municipality"],
    "COUNTY":       ["county", "parish", "borough", "district"],
    "STATECODE":    ["state", "province", "region", "statecode", "state code"],
    "POSTALCODE":   ["zip", "postal code", "postcode", "postalcode", "zip code"],
    "CNTRYCODE":    ["country", "country code", "cntry", "cntrycode", "iso country"],
    "Latitude":     ["latitude", "lat", "y coordinate"],
    "Longitude":    ["longitude", "lon", "lng", "x coordinate"],
    "BLDGSCHEME":   ["bldgscheme", "const scheme", "construction scheme"],
    "BLDGCLASS":    ["bldgclass", "construction", "const type", "building class", "construction type"],
    "OCCSCHEME":    ["occscheme", "occ scheme", "occupancy scheme"],
    "OCCTYPE":      ["occtype", "occ type", "occupancy", "occupancy type"],
    "NUMBLDGS":     [
        "risk count",
        "num risks",
        "count",
        "number of risks",
        "no of risks",
        "numbldgs",
        "no of buildings",
        "num buildings",
        "# of buildings",
        "# structures on site",
        "asset count",
        "bldcnt",
        "bldg count",
        "bldg indicator",
        "bldg qty",
        "bldg total",
        "bldg_count",
        "bldg_count_loc",
        "bldg_no",
        "bldg_num",
        "bldg_per_loc",
        "bldg_qty",
        "bldg_qty_loc",
        "bldgcnt",
        "bldgcnt_loc",
        "bldgnum",
        "bldgs",
        "blocks",
        "building count",
        "building exposure count",
        "building indicator",
        "building quantity",
        "buildings",
        "buildings per asset",
        "campus building count",
        "campus structures",
        "classrooms",
        "complex building count",
        "complex buildings",
        "facility count",
        "insured buildings",
        "insured structure count",
        "loc_bldg_cnt",
        "loc_bldgs",
        "locbldg",
    ],
    "NUMSTORIES":   [
        "# stories",
        "number of stories",
        "no. of stories",
        "stories",
        "story count",
        "total stories",
        "floor count",
        "no. floors",
        "total levels",
        "level count",
        "stry",
        "strys",
        "stry_cnt",
        "strynum",
        "no_stry",
        "nostry",
        "bldg_stry",
        "str_cnt",
        "story_no",
        "stry_qty",
        "above grade stories",
        "stories (above grade)",
        "bldg height (stories)",
        "structural height (stories)",
        "max story count",
        "no of stories",
        "num stories",
        "floors",
    ],
    "FLOORAREA":    [
        "gross area",
        "gross sq ft",
        "gross square footage",
        "gross sf",
        "total area",
        "building area",
        "total building area",
        "total sq ft",
        "total square feet",
        "bldg area",
        "building sq ft",
        "area (sq ft)",
        "grs_area",
        "grs_sqft",
        "gsf",
        "tla",
        "tba",
        "sqft",
        "square_footage",
        "area_sf",
        "bldg_sf",
        "bldgarea",
        "tot_area",
        "tot_sf",
        "exp_area",
        "area_value",
        "gla",
        "floor area",
        "floorarea",
    ],
    "YEARBUILT":          [
        "year built",
        "yr built",
        "built year",
        "year of construction",
        "construction year",
        "year constructed",
        "yob",
        "yearbuilt",
        "yr_built",
        "const_year",
        "year_const",
        "build_year",
        "original year",
        "orig year",
        "year_built",
    ],
    "YEARUPGRAD":    [
        "year upgrade",
        "yr upgrade",
        "upgrade year",
        "year renovated",
        "renovation year",
        "major renovation year",
        "full retrofit year",
        "rehab year",
        "year modernized",
        "yearupgrad",
        "yr_upgrade",
        "upgrade_year",
        "renovation_year",
        "year seismically retrofitted",
        "seismic retrofit year",
        "seismic upgrade year",
        "seismically upgraded",
        "year retrofitted",
        "structural retrofit year",
        "major reno",
        "fully rehabbed",
        "fully modernized",
        "brought up to code",
        "complete reconstruction",
        "building rebuilt",
        "full structural upgrade",
    ],
    "SPRINKLER":    ["sprinkler", "sprinkler type", "fire suppression"],
    "ROOFGEOM":     ["roof type", "roof cover", "roof material", "roofgeom", "roof geometry"],
    "FOUNDATION":   ["foundation", "foundation type"],
    "CLADDING":     ["cladding", "wall material", "exterior wall", "wall siding"],
    "SOFTSTORY":    ["soft story", "soft storey", "softstory"],
    "WALLTYPE":     ["wall type", "wall"],
    # Peril values â€” passed through directly from source
    "TIV":                [
        "tiv",
        "total insured value",
        "total insurable value",
        "total exposure",
        "total value",
        "sum insured",
        "total sum insured",
        "tiv total",
    ],
    "EQCV1VAL":          [
        "bldg",
        "bldg value",
        "building limit",
        "building sum insured",
        "replacement cost",
        "rc",
        "rebuild",
        "reinstatement value",
        "real property",
        "structure",
        "real property value",
        "bldg incl contents",
        "pito",
        "hard cost",
        "land improvements",
        "building value",
        "building tiv",
        "bldg tiv",
    ], "EQCV2VAL":          [
        "content value",
        "limit",
        "contents coverage",
        "declared",
        "reported values",
        "personal property",
        "bpp",
        "movable assets",
        "tenant contents",
        "furniture & fixtures",
        "f&f",
        "machinery",
        "equipment",
        "inventory",
        "edp",
        "fine arts",
        "signs",
        "contents value",
        "contents tiv",
    ], "EQCV3VAL":          [
        "bi",
        "time element",
        "time element tiv",
        "bi limit",
        "bi sum insured",
        "declared bi",
        "gross profit",
        "gross earnings",
        "net profit",
        "loss of income",
        "revenue",
        "sales",
        "turnover",
        "operating income",
        "rents",
        "loss",
        "profit",
        "payroll",
        "rental value",
        "extra expense",
        "accounts receivable",
        "business interruption",
        "bi value",
        "te value",
    ],
    "WSCV1VAL": ["wscv1val"], "WSCV2VAL": ["wscv2val"], "WSCV3VAL": ["wscv3val"],
    "TOCV1VAL": ["tocv1val"], "TOCV2VAL": ["tocv2val"], "TOCV3VAL": ["tocv3val"],
    "FLCV1VAL": ["flcv1val"], "FLCV2VAL": ["flcv2val"], "FLCV3VAL": ["flcv3val"],
    "TRCV1VAL": ["trcv1val"], "TRCV2VAL": ["trcv2val"], "TRCV3VAL": ["trcv3val"],
    "FRCV1VAL": ["frcv1val"], "FRCV2VAL": ["frcv2val"], "FRCV3VAL": ["frcv3val"],
    "EQCV1LCUR": ["eqcv1lcur"], "EQCV2LCUR": ["eqcv2lcur"], "EQCV3LCUR": ["eqcv3lcur"],
    "WSCV1LCUR": ["wscv1lcur"], "WSCV2LCUR": ["wscv2lcur"], "WSCV3LCUR": ["wscv3lcur"],
    "TOCV1LCUR": ["tocv1lcur"], "TOCV2LCUR": ["tocv2lcur"], "TOCV3LCUR": ["tocv3lcur"],
    "FLCV1LCUR": ["flcv1lcur"], "FLCV2LCUR": ["flcv2lcur"], "FLCV3LCUR": ["flcv3lcur"],
    "TRCV1LCUR": ["trcv1lcur"], "TRCV2LCUR": ["trcv2lcur"], "TRCV3LCUR": ["trcv3lcur"],
    "FRCV1LCUR": ["frcv1lcur"], "FRCV2LCUR": ["frcv2lcur"], "FRCV3LCUR": ["frcv3lcur"],
}


def _get_canonical_registry(target_format: str) -> Dict[str, List[str]]:
    return AIR_CANONICAL_FIELDS if target_format == "AIR" else RMS_CANONICAL_FIELDS


def _flat_alias_list(registry: Dict[str, List[str]]) -> List[str]:
    """Return a flat list of (canonical field) + all aliases for fuzzy search."""
    entries = []
    for canonical, aliases in registry.items():
        entries.append(canonical)
        entries.extend(aliases)
    return entries


def _alias_to_canonical(alias: str, registry: Dict[str, List[str]]) -> Optional[str]:
    """Reverse-lookup: given an alias string, return its canonical field name."""
    alias_lower = alias.lower()
    for canonical, aliases in registry.items():
        if alias_lower == canonical.lower() or alias_lower in [a.lower() for a in aliases]:
            return canonical
    return None


# â”€â”€ LangGraph state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ColumnMappingState(TypedDict):
    target_format: str
    source_columns: List[str]
    sample_values: Dict[str, List[Any]]
    fuzzy_results: Dict[str, List[Dict]]    # col â†’ [{canonical, score, method}]
    llm_candidates: List[str]              # cols needing LLM
    llm_results: Dict[str, Dict]           # col â†’ {match, confidence, reason}
    final_suggestions: Dict[str, List[Dict]]
    fuzzy_threshold: int
    cutoff: int


# â”€â”€ Graph nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _node_fuzzy_matching(state: ColumnMappingState) -> ColumnMappingState:
    """Run rapidfuzz token_sort_ratio against all canonical names + aliases."""
    registry = _get_canonical_registry(state["target_format"])
    alias_list = _flat_alias_list(registry)

    results: Dict[str, List[Dict]] = {}
    for col in state["source_columns"]:
        matches = process.extract(
            col,
            alias_list,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=state["cutoff"],
            limit=5,
        )
        mapped = []
        seen_canonicals = set()
        for match_str, score, _ in sorted(matches, key=lambda x: -x[1]):
            canonical = _alias_to_canonical(match_str, registry) or match_str
            if canonical not in seen_canonicals:
                mapped.append({
                    "canonical": canonical,
                    "score": round(score / 100, 4),
                    "method": "fuzzy",
                })
                seen_canonicals.add(canonical)
            if len(mapped) >= 3:
                break
        results[col] = mapped

    return {**state, "fuzzy_results": results}


def _node_extract_llm_candidates(state: ColumnMappingState) -> ColumnMappingState:
    """Identify columns whose best fuzzy score is below the LLM fallback threshold."""
    threshold = state["fuzzy_threshold"] / 100.0
    candidates = [
        col for col, suggestions in state["fuzzy_results"].items()
        if not suggestions or suggestions[0]["score"] < threshold
    ]
    return {**state, "llm_candidates": candidates}


def _node_llm_matching(state: ColumnMappingState) -> ColumnMappingState:
    """Batch-call Gemini for all LLM candidate columns in one request."""
    candidates = state["llm_candidates"]
    if not candidates:
        return {**state, "llm_results": {}}

    registry = _get_canonical_registry(state["target_format"])
    canonical_list = "\n".join(f"- {c}" for c in registry.keys())
    sample_vals = state["sample_values"]

    items_text = "\n".join(
        f'{i+1}. "{col}": [{", ".join(repr(v) for v in sample_vals.get(col, [])[:3])}]'
        for i, col in enumerate(candidates)
    )

    prompt = f"""You are mapping columns from an insurance property data file to standard CAT modeling fields.

Target format: {state["target_format"]}

Standard field list:
{canonical_list}

For each numbered source column below, identify the best matching standard field.
Each entry shows: column name, then up to 3 sample values.

{items_text}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "1": {{"match": "CanonicalField", "confidence": 0.0, "reason": "one sentence"}},
  "2": {{"match": "CanonicalField", "confidence": 0.0, "reason": "one sentence"}}
}}

Rules:
- If no reasonable match exists, set match to null and confidence to 0.0
- Confidence 0.0-1.0 (max 0.95)
- Give the exact canonical field name from the standard field list
"""

    llm_results: Dict[str, Dict] = {}
    try:
        response = _gemini_model.generate_content(prompt)
        parsed = json.loads(response.text)
        for i, col in enumerate(candidates):
            entry = parsed.get(str(i + 1), {})
            llm_results[col] = {
                "match": entry.get("match"),
                "confidence": float(entry.get("confidence", 0.0)),
                "reason": entry.get("reason", ""),
            }
    except Exception as exc:
        logger.warning(f"LLM column mapping failed: {exc}. Returning empty LLM results.")

    return {**state, "llm_results": llm_results}


def _node_merge_results(state: ColumnMappingState) -> ColumnMappingState:
    """Merge fuzzy and LLM results into final_suggestions per column."""
    final: Dict[str, List[Dict]] = {}

    for col in state["source_columns"]:
        fuzzy = state["fuzzy_results"].get(col, [])
        llm = state["llm_results"].get(col, {})

        suggestions: List[Dict] = []

        # LLM result (if column was an LLM candidate and returned a valid match)
        if col in state["llm_candidates"] and llm.get("match"):
            llm_score = llm["confidence"]
            fuzzy_best = fuzzy[0]["score"] if fuzzy else 0.0
            suggestions.append({
                "canonical": llm["match"],
                "score": llm_score,
                "method": "llm",
                "reason": llm.get("reason"),
            })
            # Keep high-scoring fuzzy results too
            for f in fuzzy:
                if f["score"] >= fuzzy_best * 0.85 and f["canonical"] != llm["match"]:
                    suggestions.append(f)
        else:
            suggestions = list(fuzzy)

        # Deduplicate by canonical name, keep highest score
        seen: Dict[str, Dict] = {}
        for s in suggestions:
            c = s["canonical"]
            if c not in seen or s["score"] > seen[c]["score"]:
                seen[c] = s
        suggestions = sorted(seen.values(), key=lambda x: -x["score"])[:3]
        final[col] = suggestions

    return {**state, "final_suggestions": final}


# â”€â”€ Build and compile the graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_column_mapping_graph():
    g = StateGraph(ColumnMappingState)
    g.add_node("fuzzy_matching", _node_fuzzy_matching)
    g.add_node("extract_llm_candidates", _node_extract_llm_candidates)
    g.add_node("llm_matching", _node_llm_matching)
    g.add_node("merge_results", _node_merge_results)
    g.set_entry_point("fuzzy_matching")
    g.add_edge("fuzzy_matching", "extract_llm_candidates")
    g.add_edge("extract_llm_candidates", "llm_matching")
    g.add_edge("llm_matching", "merge_results")
    g.add_edge("merge_results", END)
    return g.compile()


_column_mapping_graph = _build_column_mapping_graph()


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def suggest_columns(
    source_columns: List[str],
    sample_values: Dict[str, List[Any]],
    target_format: str = "AIR",
    fuzzy_threshold: int = 72,
    cutoff: int = 50,
) -> Dict[str, Any]:
    """
    Run the column mapping pipeline and return:
    {
      "suggestions": {col: [{canonical, score, method, reason?}]},
      "unmapped": [cols with no suggestion >= cutoff]
    }

    MEMORY LAYER (runs first):
      Any source column whose normalised name exists in mapping_memory.json
      is immediately assigned canonical=<remembered>, score=1.0, method="memory".
      Those columns are excluded from the fuzzy+LLM graph, reducing latency and
      LLM cost.  If ALL columns are covered by memory, the graph is skipped.
    """
    # â”€â”€ Step 0: Memory lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    memory_hits = mapping_memory.lookup_memory(source_columns, target_format)

    # Columns not found in memory still need fuzzy / LLM
    remaining_cols = [c for c in source_columns if c not in memory_hits]

    # â”€â”€ Step 1: Run fuzzy + LLM graph only for remaining columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    graph_final: Dict[str, List[Dict]] = {}
    if remaining_cols:
        initial_state: ColumnMappingState = {
            "target_format": target_format,
            "source_columns": remaining_cols,
            "sample_values": {k: v for k, v in sample_values.items() if k in remaining_cols},
            "fuzzy_results": {},
            "llm_candidates": [],
            "llm_results": {},
            "final_suggestions": {},
            "fuzzy_threshold": fuzzy_threshold,
            "cutoff": cutoff,
        }
        result = _column_mapping_graph.invoke(initial_state)
        graph_final = result["final_suggestions"]

    # â”€â”€ Step 2: Merge memory hits + graph results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    final: Dict[str, List[Dict]] = {}
    for col in source_columns:
        if col in memory_hits:
            hit = memory_hits[col]
            final[col] = [{
                "canonical": hit["canonical"],
                "score": 1.0,
                "method": "memory",
                "reason": hit.get("reason", ""),
                "count": hit.get("count", 1),
            }]
        else:
            final[col] = graph_final.get(col, [])

    memory_count = len(memory_hits)
    if memory_count:
        logger.info(
            f"suggest_columns: {memory_count}/{len(source_columns)} columns "
            f"resolved from memory (target={target_format})"
        )

    unmapped = [col for col, sug in final.items() if not sug]
    return {
        "suggestions": final,
        "unmapped": unmapped,
        "memory_count": memory_count,
    }


def validate_required_fields(column_map: Dict[str, Optional[str]], target_format: str) -> List[str]:
    """Return list of required canonical fields not covered by the column_map."""
    AIR_REQUIRED = {"OccupancyCode", "ConstructionCode", "LineOfBusiness"}
    RMS_REQUIRED = {"OCCTYPE", "BLDGCLASS"}

    required = AIR_REQUIRED if target_format == "AIR" else RMS_REQUIRED
    mapped_canonicals = set(v for v in column_map.values() if v)
    return sorted(required - mapped_canonicals)

