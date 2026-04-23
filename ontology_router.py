"""
ontology_router.py — FastAPI router for COPE ontology browsing,
versioned override uploads, template downloads, and business rules config.
"""
import csv
import io
import json
import logging
import os
import pathlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from rules import BusinessRulesConfig

logger = logging.getLogger("ontology_router")

router = APIRouter(tags=["Ontology & Rules"])

# ── Paths ────────────────────────────────────────────────────────────────────────
_REF_DIR = pathlib.Path(__file__).parent / "reference"
_OVERRIDES_DIR = _REF_DIR / "overrides"
_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)

# ── File registry (base name → JSON filename) ────────────────────────────────
_COPE_FILES = {
    "construction": {
        "AIR": "air_const_codes.json",
        "RMS": "rms_const_codes.json",
    },
    "occupancy": {
        "AIR": "air_occ_codes.json",
        "RMS": "rms_occ_codes.json",
    },
    "protection": {
        "ALL": "iso_fire_class_map.json",
    },
    "exposure": {
        "ALL": "secondary_modifiers.json",
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    p = _REF_DIR / filename
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _load_with_overrides(cope_type: str, fmt: str) -> dict:
    """Load the base reference JSON and merge any versioned overrides on top."""
    files = _COPE_FILES.get(cope_type, {})
    filename = files.get(fmt) or files.get("ALL")
    if not filename:
        return {}

    base = _load_json(filename)

    # Find override files for this type+format, sorted by timestamp (oldest first)
    prefix = f"{cope_type}_{fmt.lower()}_"
    overrides = sorted(
        [f for f in _OVERRIDES_DIR.iterdir() if f.name.startswith(prefix) and f.suffix == ".json"],
        key=lambda p: p.stat().st_mtime,
    )

    for override_path in overrides:
        try:
            override_data = json.loads(override_path.read_text(encoding="utf-8"))
            # Merge: for each code, extend keywords list (deduplicated)
            for code, meta in override_data.items():
                if code.startswith("_"):
                    continue
                if code in base:
                    existing_kw = base[code].get("keywords", [])
                    new_kw = meta.get("keywords", [])
                    merged = list(dict.fromkeys(existing_kw + new_kw))
                    base[code]["keywords"] = merged
                    if meta.get("description"):
                        base[code]["description"] = meta["description"]
                else:
                    base[code] = meta
        except Exception as e:
            logger.warning(f"Failed to load override {override_path}: {e}")

    return base


def _list_override_files(cope_type: str, fmt: str = "") -> List[dict]:
    """List all override files for a given cope_type, optionally filtered by format."""
    results = []
    for f in sorted(_OVERRIDES_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not f.suffix == ".json":
            continue
        name = f.stem  # e.g. construction_air_v1_20260422_153000
        if not name.startswith(cope_type):
            continue
        if fmt and f"_{fmt.lower()}_" not in name:
            continue
        stat = f.stat()
        results.append({
            "version_id": f.name,
            "filename": f.name,
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "cope_type": cope_type,
        })
    return results


# ── COPE Dictionary Endpoints ────────────────────────────────────────────────────

@router.get("/ontology/construction")
def get_construction(format: str = Query("AIR", pattern="^(AIR|RMS)$")):
    """Return the active construction code dictionary for the given format."""
    data = _load_with_overrides("construction", format)
    return {
        "cope_type": "construction",
        "format": format,
        "code_count": len([k for k in data if not k.startswith("_")]),
        "codes": data,
    }


@router.get("/ontology/occupancy")
def get_occupancy(format: str = Query("AIR", pattern="^(AIR|RMS)$")):
    """Return the active occupancy code dictionary for the given format."""
    data = _load_with_overrides("occupancy", format)
    return {
        "cope_type": "occupancy",
        "format": format,
        "code_count": len([k for k in data if not k.startswith("_")]),
        "codes": data,
    }


@router.get("/ontology/protection")
def get_protection():
    """Return the ISO Fire Class protection map."""
    data = _load_with_overrides("protection", "ALL")
    # The JSON has a nested structure with iso_to_air key
    iso_data = data.get("iso_to_air", data)
    return {
        "cope_type": "protection",
        "format": "ISO",
        "code_count": len([k for k in iso_data if not k.startswith("_")]),
        "codes": iso_data,
    }


@router.get("/ontology/exposure")
def get_exposure(format: str = Query("AIR", pattern="^(AIR|RMS)$")):
    """Return secondary modifiers (roof, wall, foundation, soft_story) for the given format."""
    data = _load_with_overrides("exposure", "ALL")

    # Filter to vendor-specific sections
    if format == "AIR":
        sections = ["roof_cover", "wall_type", "foundation_type", "soft_story"]
    else:
        sections = ["rms_roofsys", "rms_cladsys", "foundation_type", "soft_story"]

    filtered = {}
    for key in sections:
        if key in data:
            filtered[key] = data[key]

    return {
        "cope_type": "exposure",
        "format": format,
        "sections": list(filtered.keys()),
        "data": filtered,
    }


# ── Template Download ────────────────────────────────────────────────────────────

@router.get("/ontology/template/{cope_type}")
def download_template(
    cope_type: str,
    format: str = Query("AIR", pattern="^(AIR|RMS)$"),
):
    """Generate and return a downloadable CSV template for the requested COPE type."""
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}. Must be one of: {list(_COPE_FILES.keys())}")

    data = _load_with_overrides(cope_type, format)

    # Handle nested structures
    if cope_type == "protection":
        data = data.get("iso_to_air", data)

    if cope_type == "exposure":
        # Flatten all sections into one template
        rows = []
        for section_key, section_data in data.items():
            if section_key.startswith("_"):
                continue
            codes = section_data.get("codes", {})
            aliases = section_data.get("aliases", {})
            for code, desc in codes.items():
                matching_aliases = [alias for alias, c in aliases.items() if str(c) == str(code)]
                rows.append({
                    "Section": section_key,
                    "Code": code,
                    "Description": desc,
                    "Keywords": "; ".join(matching_aliases[:10]),
                })
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["Section", "Code", "Description", "Keywords"])
        writer.writeheader()
        writer.writerows(rows)
        # Add empty rows for new entries
        for _ in range(5):
            writer.writerow({"Section": "", "Code": "", "Description": "", "Keywords": ""})
    else:
        rows = []
        for code, meta in data.items():
            if code.startswith("_"):
                continue
            desc = meta.get("description", "") if isinstance(meta, dict) else str(meta)
            keywords = meta.get("keywords", []) if isinstance(meta, dict) else []
            # For protection, handle nested structure
            if isinstance(meta, dict) and "iso_label" in meta:
                desc = meta.get("description", meta.get("iso_label", ""))
                keywords = meta.get("aliases", [])
            rows.append({
                "Code": code,
                "Description": desc,
                "Keywords": "; ".join(keywords[:15]) if isinstance(keywords, list) else str(keywords),
            })
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["Code", "Description", "Keywords"])
        writer.writeheader()
        writer.writerows(rows)
        # Add empty rows for new entries
        for _ in range(5):
            writer.writerow({"Code": "", "Description": "", "Keywords": ""})

    buf.seek(0)
    filename = f"{cope_type}_{format.lower()}_template.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── Versioned Upload ─────────────────────────────────────────────────────────────

@router.post("/ontology/upload/{cope_type}")
async def upload_ontology(
    cope_type: str,
    format: str = Query("AIR", pattern="^(AIR|RMS)$"),
    file: UploadFile = File(...),
):
    """Upload a CSV/JSON file as a versioned override for the given COPE type."""
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")

    content = await file.read()
    fname = (file.filename or "").lower()

    parsed: Dict[str, Any] = {}

    if fname.endswith(".json"):
        try:
            parsed = json.loads(content.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    elif fname.endswith(".csv"):
        try:
            text = content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                code = (row.get("Code") or "").strip()
                if not code:
                    continue
                desc = (row.get("Description") or "").strip()
                keywords_raw = (row.get("Keywords") or "").strip()
                keywords = [kw.strip() for kw in re.split(r"[;,]", keywords_raw) if kw.strip()]
                parsed[code] = {
                    "description": desc,
                    "keywords": keywords,
                }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or JSON.")

    if not parsed:
        raise HTTPException(status_code=400, detail="File contains no valid entries.")

    # Save as versioned override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_versions = _list_override_files(cope_type, format)
    version_num = len(existing_versions) + 1
    override_filename = f"{cope_type}_{format.lower()}_v{version_num}_{ts}.json"
    override_path = _OVERRIDES_DIR / override_filename

    override_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved ontology override: {override_path} ({len(parsed)} entries)")

    # Also merge into the live in-memory registries if code_mapper is loaded
    merged_count = 0
    try:
        import agents.cat.code_mapper as code_mapper
        if cope_type == "construction" and format == "AIR":
            registry = code_mapper._const_codes
        elif cope_type == "construction" and format == "RMS":
            registry = code_mapper._rms_const_codes
        elif cope_type == "occupancy" and format == "AIR":
            registry = code_mapper._occ_codes
        elif cope_type == "occupancy" and format == "RMS":
            registry = code_mapper._rms_occ_codes
        else:
            registry = None

        if registry is not None:
            for code, meta in parsed.items():
                if code.startswith("_"):
                    continue
                if code in registry:
                    existing_kw = registry[code].get("keywords", [])
                    new_kw = meta.get("keywords", [])
                    registry[code]["keywords"] = list(dict.fromkeys(existing_kw + new_kw))
                    merged_count += 1
                else:
                    registry[code] = meta
                    merged_count += 1
    except Exception as e:
        logger.warning(f"Could not merge into live registry: {e}")

    return {
        "status": "ok",
        "version_id": override_filename,
        "entries_uploaded": len(parsed),
        "entries_merged_live": merged_count,
        "message": f"Saved as {override_filename}. {merged_count} entries merged into active registry.",
    }


# ── Version Management ───────────────────────────────────────────────────────────

@router.get("/ontology/versions/{cope_type}")
def list_versions(
    cope_type: str,
    format: str = Query("", description="Optional format filter (AIR/RMS)"),
):
    """List all uploaded override versions for a given COPE type."""
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")
    versions = _list_override_files(cope_type, format)
    return {"cope_type": cope_type, "versions": versions}


@router.delete("/ontology/versions/{cope_type}/{version_id}")
def delete_version(cope_type: str, version_id: str):
    """Remove a specific override version file."""
    target = _OVERRIDES_DIR / version_id
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Version file not found: {version_id}")
    if not target.name.startswith(cope_type):
        raise HTTPException(status_code=400, detail="Version ID does not match the cope_type.")
    target.unlink()
    logger.info(f"Deleted ontology override: {target}")
    return {"status": "ok", "deleted": version_id}


# ── Rules Config ─────────────────────────────────────────────────────────────────

# In-memory session-level override (per server instance)
_active_rules_override: Dict[str, Any] = {}


@router.get("/rules/config")
def get_rules_config():
    """Return the current BusinessRulesConfig (defaults + any session overrides)."""
    base = BusinessRulesConfig(**_active_rules_override)
    return base.model_dump()


@router.post("/rules/config")
def set_rules_config(body: dict):
    """Accept a partial config update. Merges into the active override."""
    global _active_rules_override
    # Validate by constructing the model
    merged = {**_active_rules_override, **body}
    try:
        validated = BusinessRulesConfig(**merged)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid config: {e}")
    _active_rules_override = validated.model_dump()
    return {"status": "ok", "config": _active_rules_override}


@router.get("/rules/presets")
def get_rules_presets():
    """Return named presets: Conservative, Balanced, Permissive."""
    return {
        "presets": [
            {
                "name": "Conservative",
                "description": "Tight limits, flag everything, low confidence thresholds. Best for high-value portfolios.",
                "config": {
                    "year_min": 1850,
                    "invalid_year_action": "flag_review",
                    "max_stories_wood_frame": 2,
                    "stories_exceeded_action": "reset_construction",
                    "min_area_sqft": 200.0,
                    "invalid_area_action": "flag_review",
                    "max_building_value": 50_000_000,
                    "max_contents_value": 25_000_000,
                    "max_bi_value": 10_000_000,
                    "invalid_value_action": "flag_review",
                    "occ_confidence_threshold": 0.80,
                    "const_confidence_threshold": 0.80,
                    "deterministic_score_threshold": 0.90,
                    "llm_confidence_threshold": 0.80,
                    "tfidf_confidence_threshold": 0.60,
                },
            },
            {
                "name": "Balanced",
                "description": "Default settings. Good balance between accuracy and throughput.",
                "config": BusinessRulesConfig().model_dump(),
            },
            {
                "name": "Permissive",
                "description": "High limits, minimal flagging. For bulk processing with manual review later.",
                "config": {
                    "year_min": 1700,
                    "invalid_year_action": "none",
                    "max_stories_wood_frame": 5,
                    "stories_exceeded_action": "none",
                    "min_area_sqft": 50.0,
                    "invalid_area_action": "none",
                    "max_building_value": 500_000_000,
                    "max_contents_value": 200_000_000,
                    "max_bi_value": 100_000_000,
                    "invalid_value_action": "none",
                    "occ_confidence_threshold": 0.50,
                    "const_confidence_threshold": 0.50,
                    "deterministic_score_threshold": 0.70,
                    "llm_confidence_threshold": 0.50,
                    "tfidf_confidence_threshold": 0.30,
                },
            },
        ]
    }
