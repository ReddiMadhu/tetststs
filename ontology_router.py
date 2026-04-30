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
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel as _BaseModel

from rules import BusinessRulesConfig
from duplicate_detector import find_duplicates, find_internal_duplicates
from models import (
    AddEntryRequest, AddEntryResponse, DuplicateMatchSchema,
    ExcelUploadResponse, CommitEntriesRequest,
)

logger = logging.getLogger("ontology_router")

router = APIRouter(tags=["Ontology & Rules"])

# ── Paths ────────────────────────────────────────────────────────────────────────
_REF_DIR = pathlib.Path(__file__).parent / "reference"
_OVERRIDES_DIR = _REF_DIR / "overrides"
_OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)

# Temporary storage for Excel upload previews (token -> parsed data)
_excel_preview_store: Dict[str, dict] = {}

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


# ── Manual Entry Addition ────────────────────────────────────────────────────────

def _save_override_and_merge(cope_type: str, fmt: str, parsed: dict) -> str:
    """Write entries to an override JSON file and merge into live registry. Returns version_id."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    existing_versions = _list_override_files(cope_type, fmt)
    version_num = len(existing_versions) + 1
    override_filename = f"{cope_type}_{fmt.lower()}_v{version_num}_{ts}.json"
    override_path = _OVERRIDES_DIR / override_filename
    override_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved ontology override: {override_path} ({len(parsed)} entries)")

    # Merge into live in-memory registries
    try:
        import agents.cat.code_mapper as code_mapper
        registry = None
        if cope_type == "construction" and fmt == "AIR":
            registry = code_mapper._const_codes
        elif cope_type == "construction" and fmt == "RMS":
            registry = code_mapper._rms_const_codes
        elif cope_type == "occupancy" and fmt == "AIR":
            registry = code_mapper._occ_codes
        elif cope_type == "occupancy" and fmt == "RMS":
            registry = code_mapper._rms_occ_codes

        if registry is not None:
            for code, meta in parsed.items():
                if code.startswith("_"):
                    continue
                if code in registry:
                    existing_kw = registry[code].get("keywords", [])
                    new_kw = meta.get("keywords", [])
                    registry[code]["keywords"] = list(dict.fromkeys(existing_kw + new_kw))
                else:
                    registry[code] = meta
    except Exception as e:
        logger.warning(f"Could not merge into live registry: {e}")

    return override_filename


_VALID_EXPOSURE_SECTIONS = [
    "roof_cover", "wall_type", "foundation_type", "soft_story",
    "rms_roofsys", "rms_cladsys",
]


@router.post("/ontology/entry/{cope_type}", response_model=AddEntryResponse)
def add_entry(
    cope_type: str,
    body: AddEntryRequest,
    format: str = Query("AIR", pattern="^(AIR|RMS)$"),
):
    """Add a single dictionary entry with duplicate detection."""
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")

    code = body.code.strip()
    if not code:
        raise HTTPException(status_code=422, detail="Code is required.")
    if code.startswith("_"):
        raise HTTPException(status_code=422, detail="Code cannot start with underscore.")
    if not body.description.strip():
        raise HTTPException(status_code=422, detail="Description is required.")
    if cope_type == "exposure" and not body.section:
        raise HTTPException(status_code=422, detail="Section is required for exposure entries.")
    if cope_type == "exposure" and body.section not in _VALID_EXPOSURE_SECTIONS:
        raise HTTPException(status_code=422, detail=f"Invalid section: {body.section}. Must be one of: {_VALID_EXPOSURE_SECTIONS}")

    new_entry = {
        code: {
            "description": body.description.strip(),
            "keywords": [kw.strip() for kw in body.keywords if kw.strip()],
        }
    }

    # Duplicate check (unless force=True)
    if not body.force:
        existing_data = _load_with_overrides(cope_type, format)
        dupes = find_duplicates(new_entry, existing_data, cope_type)
        if dupes:
            matches = dupes.get(code, [])
            return AddEntryResponse(
                status="duplicates_found",
                entry_saved=False,
                duplicates=[
                    DuplicateMatchSchema(**m.model_dump()) for m in matches
                ],
            )

    # Save as override
    version_id = _save_override_and_merge(cope_type, format, new_entry)

    return AddEntryResponse(
        status="ok",
        entry_saved=True,
        version_id=version_id,
    )


# ── Delete Entry ─────────────────────────────────────────────────────────────────

@router.delete("/ontology/entry/{cope_type}/{code}")
def delete_entry(
    cope_type: str,
    code: str,
    format: str = Query("AIR", pattern="^(AIR|RMS)$"),
):
    """Delete a single code entry from the dictionary.

    Removes the code from:
    1. All override files for this cope_type+format
    2. The base reference JSON file
    3. The live in-memory registry (if loaded)
    """
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")

    removed_from = []

    # 1. Remove from override files
    prefix = f"{cope_type}_{format.lower()}_"
    for f in _OVERRIDES_DIR.iterdir():
        if f.name.startswith(prefix) and f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if code in data:
                    del data[code]
                    f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    removed_from.append(f"override:{f.name}")
            except Exception as e:
                logger.warning(f"Failed to clean override {f}: {e}")

    # 2. Remove from base reference file
    files = _COPE_FILES.get(cope_type, {})
    filename = files.get(format) or files.get("ALL")
    if filename:
        base_path = _REF_DIR / filename
        if base_path.exists():
            try:
                base_data = json.loads(base_path.read_text(encoding="utf-8"))
                if code in base_data:
                    del base_data[code]
                    base_path.write_text(json.dumps(base_data, indent=2, ensure_ascii=False), encoding="utf-8")
                    removed_from.append(f"base:{filename}")
            except Exception as e:
                logger.warning(f"Failed to clean base file {base_path}: {e}")

    # 3. Remove from live in-memory registry
    try:
        import agents.cat.code_mapper as code_mapper
        registry = None
        if cope_type == "construction" and format == "AIR":
            registry = code_mapper._const_codes
        elif cope_type == "construction" and format == "RMS":
            registry = code_mapper._rms_const_codes
        elif cope_type == "occupancy" and format == "AIR":
            registry = code_mapper._occ_codes
        elif cope_type == "occupancy" and format == "RMS":
            registry = code_mapper._rms_occ_codes

        if registry is not None and code in registry:
            del registry[code]
            removed_from.append("live_registry")
    except Exception as e:
        logger.warning(f"Could not remove from live registry: {e}")

    if not removed_from:
        raise HTTPException(status_code=404, detail=f"Code '{code}' not found in {cope_type}/{format}.")

    logger.info(f"Deleted code '{code}' from {cope_type}/{format}: {removed_from}")
    return {
        "status": "ok",
        "deleted_code": code,
        "removed_from": removed_from,
    }


# ── Edit Entry ───────────────────────────────────────────────────────────────────

class _EditEntryBody(_BaseModel):
    description: str
    keywords: List[str] = []

@router.put("/ontology/entry/{cope_type}/{code}")
def edit_entry(
    cope_type: str,
    code: str,
    body: _EditEntryBody,
    format: str = Query("AIR", pattern="^(AIR|RMS)$"),
):
    """Update an existing dictionary entry's description and keywords in-place.

    Writes through to:
    1. Base reference JSON file
    2. All override files containing this code
    3. Live in-memory registry (if loaded)
    """
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")
    if not body.description.strip():
        raise HTTPException(status_code=422, detail="Description is required.")

    updated_in = []
    new_meta = {
        "description": body.description.strip(),
        "keywords": [kw.strip() for kw in body.keywords if kw.strip()],
    }

    # 1. Update base reference file
    files = _COPE_FILES.get(cope_type, {})
    filename = files.get(format) or files.get("ALL")
    if filename:
        base_path = _REF_DIR / filename
        if base_path.exists():
            try:
                base_data = json.loads(base_path.read_text(encoding="utf-8"))
                if code in base_data:
                    if isinstance(base_data[code], dict):
                        base_data[code]["description"] = new_meta["description"]
                        base_data[code]["keywords"] = new_meta["keywords"]
                    else:
                        base_data[code] = new_meta
                    base_path.write_text(json.dumps(base_data, indent=2, ensure_ascii=False), encoding="utf-8")
                    updated_in.append(f"base:{filename}")
            except Exception as e:
                logger.warning(f"Failed to update base file {base_path}: {e}")

    # 2. Update override files
    prefix = f"{cope_type}_{format.lower()}_"
    for f in _OVERRIDES_DIR.iterdir():
        if f.name.startswith(prefix) and f.suffix == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if code in data:
                    if isinstance(data[code], dict):
                        data[code]["description"] = new_meta["description"]
                        data[code]["keywords"] = new_meta["keywords"]
                    else:
                        data[code] = new_meta
                    f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    updated_in.append(f"override:{f.name}")
            except Exception as e:
                logger.warning(f"Failed to update override {f}: {e}")

    # 3. Update live in-memory registry
    try:
        import agents.cat.code_mapper as code_mapper
        registry = None
        if cope_type == "construction" and format == "AIR":
            registry = code_mapper._const_codes
        elif cope_type == "construction" and format == "RMS":
            registry = code_mapper._rms_const_codes
        elif cope_type == "occupancy" and format == "AIR":
            registry = code_mapper._occ_codes
        elif cope_type == "occupancy" and format == "RMS":
            registry = code_mapper._rms_occ_codes

        if registry is not None and code in registry:
            if isinstance(registry[code], dict):
                registry[code]["description"] = new_meta["description"]
                registry[code]["keywords"] = new_meta["keywords"]
            else:
                registry[code] = new_meta
            updated_in.append("live_registry")
    except Exception as e:
        logger.warning(f"Could not update live registry: {e}")

    if not updated_in:
        # Code not found anywhere — create as override
        version_id = _save_override_and_merge(cope_type, format, {code: new_meta})
        updated_in.append(f"new_override:{version_id}")

    logger.info(f"Edited code '{code}' in {cope_type}/{format}: {updated_in}")
    return {
        "status": "ok",
        "code": code,
        "updated_in": updated_in,
    }


# ── Excel Upload with Duplicate Handling ─────────────────────────────────────────

@router.post("/ontology/upload-excel/{cope_type}", response_model=ExcelUploadResponse)
async def upload_excel(
    cope_type: str,
    format: str = Query("AIR", pattern="^(AIR|RMS)$"),
    file: UploadFile = File(...),
):
    """Upload an Excel/CSV file, validate entries, detect duplicates, and return a preview."""
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")

    content = await file.read()
    fname = (file.filename or "").lower()

    # Parse file into DataFrame
    try:
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False, sheet_name=0)
        elif fname.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="latin-1")
        elif fname.endswith(".json"):
            parsed_json = json.loads(content.decode("utf-8"))
            # Convert JSON dict to DataFrame
            rows = []
            for c, meta in parsed_json.items():
                if c.startswith("_"):
                    continue
                rows.append({
                    "Code": c,
                    "Description": meta.get("description", "") if isinstance(meta, dict) else str(meta),
                    "Keywords": "; ".join(meta.get("keywords", [])) if isinstance(meta, dict) else "",
                })
            df = pd.DataFrame(rows)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload XLSX, CSV, or JSON.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File parsing error: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no data rows.")

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Validate required columns
    has_code = "Code" in df.columns
    has_desc = "Description" in df.columns
    if not has_code or not has_desc:
        missing = []
        if not has_code:
            missing.append("Code")
        if not has_desc:
            missing.append("Description")
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}. Expected columns: Code, Description, Keywords",
        )

    # Parse and validate each row
    parsed: Dict[str, dict] = {}
    validation_errors: List[dict] = []
    is_exposure = cope_type == "exposure"

    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-indexed + header row
        code = str(row.get("Code", "")).strip()
        desc = str(row.get("Description", "")).strip()
        keywords_raw = str(row.get("Keywords", "")).strip()
        section = str(row.get("Section", "")).strip() if is_exposure else None

        # Skip completely empty rows
        if not code and not desc:
            continue

        # Validate
        if not code:
            validation_errors.append({"row": row_num, "field": "Code", "error": "Code is required"})
            continue
        if code.startswith("_"):
            validation_errors.append({"row": row_num, "field": "Code", "error": "Code cannot start with underscore"})
            continue
        if not desc:
            validation_errors.append({"row": row_num, "field": "Description", "error": "Description is required"})
            continue
        if is_exposure and not section:
            validation_errors.append({"row": row_num, "field": "Section", "error": "Section is required for exposure"})
            continue
        if is_exposure and section not in _VALID_EXPOSURE_SECTIONS:
            validation_errors.append({"row": row_num, "field": "Section", "error": f"Invalid section: {section}"})
            continue

        keywords = [kw.strip() for kw in re.split(r"[;,]", keywords_raw) if kw.strip()]
        entry = {"description": desc, "keywords": keywords}
        if is_exposure and section:
            entry["section"] = section
        parsed[code] = entry

    if not parsed and validation_errors:
        return ExcelUploadResponse(
            status="validation_error",
            total_entries=len(df),
            valid_entries=0,
            validation_errors=validation_errors,
        )

    # Detect internal duplicates
    internal_dupes = find_internal_duplicates(parsed)

    # Detect duplicates against existing data
    existing_data = _load_with_overrides(cope_type, format)
    external_dupes = find_duplicates(parsed, existing_data, cope_type)

    # Separate clean entries from duplicates
    duplicate_codes = set(external_dupes.keys())
    clean_entries = {k: v for k, v in parsed.items() if k not in duplicate_codes}

    # Build duplicate detail for response
    duplicates_detail = {}
    for code, matches in external_dupes.items():
        duplicates_detail[code] = {
            "new_entry": parsed[code],
            "matches": [m.model_dump() for m in matches],
        }

    # Store for later commit
    preview_token = str(uuid.uuid4())[:12]
    _excel_preview_store[preview_token] = {
        "cope_type": cope_type,
        "format": format,
        "clean_entries": clean_entries,
        "all_parsed": parsed,
        "created_at": datetime.now().isoformat(),
    }

    has_dupes = bool(external_dupes) or bool(internal_dupes)

    return ExcelUploadResponse(
        status="duplicates_found" if has_dupes else "ok",
        total_entries=len(df),
        valid_entries=len(parsed),
        validation_errors=validation_errors,
        duplicates=duplicates_detail,
        internal_duplicates=internal_dupes,
        clean_entries=clean_entries,
        preview_token=preview_token,
    )


# ── Commit Resolved Entries ──────────────────────────────────────────────────────

@router.post("/ontology/commit/{cope_type}")
def commit_entries(
    cope_type: str,
    body: CommitEntriesRequest,
):
    """Commit a set of resolved entries (from manual add or Excel upload review)."""
    if cope_type not in _COPE_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid cope_type: {cope_type}")

    fmt = body.format
    entries = body.entries

    if not entries:
        raise HTTPException(status_code=422, detail="No entries to commit.")

    # Validate entries
    for code, meta in entries.items():
        if not code.strip():
            raise HTTPException(status_code=422, detail="Entry has empty code.")
        if not meta.get("description", "").strip():
            raise HTTPException(status_code=422, detail=f"Entry '{code}' has no description.")

    version_id = _save_override_and_merge(cope_type, fmt, entries)

    return {
        "status": "ok",
        "version_id": version_id,
        "entries_committed": len(entries),
        "source": body.source,
    }


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
