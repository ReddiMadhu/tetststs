"""
main.py â€” FastAPI application for the CAT Modeling Data Pipeline.
"""
import io
import json
import logging
import os
import pathlib
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

# â”€â”€ Load env early â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load_dotenv()

# â”€â”€ Local imports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import session as session_store
from rules import BusinessRulesConfig
from models import (
    UploadResponse, SuggestColumnsResponse, ColumnSuggestion,
    ConfirmColumnsRequest, ConfirmColumnsResponse,
    GeocodeResponse, MapCodesResponse, NormalizeResponse,
    ReviewResponse, FlagEntry, CorrectRequest, CorrectResponse,
    SessionInfoResponse,
)
from output_builder import build_xlsx, build_csv

# Agents
import agents.geocoder as geocoder
import agents.cat.code_mapper as code_mapper
import agents.cat.mapping_memory as mapping_memory
from agents.cat.column_mapper import suggest_columns, validate_required_fields
from agents.normalizer import normalize_all_rows

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s â€” %(message)s",
)
logger = logging.getLogger("main")

# Global event queues for SSE
_event_queues: Dict[str, List[asyncio.Queue]] = {}

def dispatch_event(upload_id: str, agent_id: str, event: str, message: str = "", result: dict = None):
    """Utility to push an event to all connected SSE clients for a session."""
    if upload_id in _event_queues:
        data = {
            "agent_id": agent_id,
            "event": event,
            "message": message,
            "result": result
        }
        for q in _event_queues[upload_id]:
            q.put_nowait(data)

# â”€â”€ Lifespan: pre-build TF-IDF and start TTL cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CAT pipeline â€” loading TF-IDF indexesâ€¦")
    code_mapper.build_tfidf_indexes()
    geocoder.load_reference_data()
    session_store.start_ttl_cleanup()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# â”€â”€ App â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="CAT Modeling Data Pipeline",
    description="Upload, map, geocode, classify, and normalise property exposure data for AIR/RMS CAT models.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _get_session_or_404(upload_id: str) -> dict:
    try:
        return session_store.require_session(upload_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{upload_id}' not found or expired.")


def _require_stage(session: dict, stage: str, endpoint: str) -> None:
    if not session.get("stages_complete", {}).get(stage):
        raise HTTPException(
            status_code=422,
            detail=f"Stage '{stage}' must be completed before calling '{endpoint}'.",
        )


def _load_iso4217() -> set:
    p = pathlib.Path(__file__).parent / "reference" / "iso4217_currency.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return set(data.keys())
    return set()


_VALID_CURRENCIES = _load_iso4217()


def _enrich_excel_formats(content: bytes, df: pd.DataFrame) -> pd.DataFrame:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
        ws = wb.active
        col_currencies = {}
        for col_idx in range(1, ws.max_column + 1):
            if col_idx - 1 >= len(df.columns): break
            col_name = df.columns[col_idx - 1]
            for r in range(2, min(ws.max_row + 1, 10)):
                cell = ws.cell(row=r, column=col_idx)
                if cell.value is not None:
                    fmt = str(cell.number_format or "")
                    for sym in ["$", "â‚¬", "Â£", "Â¥", "â‚¹"]:
                        if sym in fmt:
                            col_currencies[col_name] = sym
                            break
                    if col_name in col_currencies:
                        break
        for col, sym in col_currencies.items():
            df[col] = df[col].apply(lambda v: f"{sym}{v}" if pd.notnull(v) and str(v).strip() != "" else v)
    except Exception as e:
        logger.warning(f"Could not extract Excel currency formats: {e}")
    return df


# â”€â”€ Step 1: Upload â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/upload", response_model=UploadResponse, tags=["Pipeline"])
async def upload(
    file: UploadFile = File(...),
    target_format: str = Query("AIR", pattern="^(AIR|RMS)$"),
    rules_config: Optional[str] = Form(None),
):
    content = await file.read()
    fname = (file.filename or "").lower()

    try:
        rules_dict = json.loads(rules_config) if rules_config and rules_config.strip() else {}
        rules = BusinessRulesConfig(**rules_dict)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid rules_config: {exc}")

    try:
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="latin-1")
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False, sheet_name=0)
            df = _enrich_excel_formats(content, df)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or XLSX.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File parsing error: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File contains no data rows.")

    seen: Dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        col = str(col).strip()
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            new_cols.append(col)
    df.columns = new_cols

    df = df.map(lambda v: v.strip() if isinstance(v, str) and v.strip() != "" else None)

    headers = list(df.columns)
    raw_rows = df.to_dict(orient="records")
    sample = raw_rows[:5]

    upload_id = session_store.create_session({
        "target_format": target_format,
        "rules_config": rules.model_dump(),
        "raw_rows": raw_rows,
        "headers": headers,
        "sample": sample,
        "column_map": {},
        "unmapped_cols": [],
        "geo_rows": [],
        "code_map": {},
        "final_rows": [],
    })

    return UploadResponse(
        upload_id=upload_id,
        row_count=len(raw_rows),
        headers=headers,
        sample=sample,
        target_format=target_format,
    )


# â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _apply_column_map(raw_rows: List[Dict], column_map: Dict[str, Optional[str]]) -> List[Dict]:
    canonical_claimed_by: Dict[str, str] = {}
    for src_col, canonical in column_map.items():
        if canonical is None: continue
        if canonical not in canonical_claimed_by:
             canonical_claimed_by[canonical] = src_col

    result = []
    for row in raw_rows:
        new_row: Dict[str, Any] = {}
        for canonical, src_col in canonical_claimed_by.items():
            new_row[canonical] = row.get(src_col)
        result.append(new_row)
    return result


def _find_code_columns(column_map: Dict, field_type: str, target: str):
    if field_type == "occupancy":
        scheme_options = ["OccupancyCodeType"] if target == "AIR" else ["OCCSCHEME"]
        value_options  = ["OccupancyCode"] if target == "AIR" else ["OCCTYPE"]
    else:
        scheme_options = ["ConstructionCodeType"] if target == "AIR" else ["BLDGSCHEME"]
        value_options  = ["ConstructionCode"]  if target == "AIR" else ["BLDGCLASS"]

    mapped_vals = set(v for v in column_map.values() if v)
    scheme_col = next((c for c in scheme_options if c in mapped_vals), "")
    value_col  = next((c for c in value_options  if c in mapped_vals), "")
    return scheme_col, value_col


# â”€â”€ SSE Streaming Endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/stream/{upload_id}", tags=["Event Stream"])
async def message_stream(upload_id: str, request: Request):
    """SSE endpoint for agent event propagation to the frontend."""
    _get_session_or_404(upload_id) # Verify session exists

    q = asyncio.Queue()
    if upload_id not in _event_queues:
        _event_queues[upload_id] = []
    _event_queues[upload_id].append(q)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                data = await q.get()
                yield {"data": json.dumps(data)}
        finally:
            if upload_id in _event_queues:
                if q in _event_queues[upload_id]:
                    _event_queues[upload_id].remove(q)
                if not _event_queues[upload_id]:
                    del _event_queues[upload_id]

    return EventSourceResponse(event_generator())


# â”€â”€ Step 2: Address Normalization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/normalize/{upload_id}", tags=["Pipeline"])
async def run_address_normalization(upload_id: str):
    """Normalize raw addresses before geocoding."""
    session = _get_session_or_404(upload_id)
    from agents.address_normalizer import normalize_addresses
    
    dispatch_event(upload_id, "address_normalizer", "start")
    dispatch_event(upload_id, "address_normalizer", "log", "Agent starting... Loading rows.")
    
    rows = session["raw_rows"]
    
    # Run the normalizer (we adapt it directly here, skipping the SSE integration in the tool for brevity)
    normalized, flags = normalize_addresses(rows)
    
    dispatch_event(upload_id, "address_normalizer", "log", f"Normalized {len(rows)} addresses.")
    dispatch_event(upload_id, "address_normalizer", "done", result={"flags_added": len(flags)})

    headers = session.get("headers", [])
    if normalized and "_CombinedAddress" in normalized[0] and "_CombinedAddress" not in headers:
        headers.append("_CombinedAddress")

    session_store.update_session(upload_id, {"raw_rows": normalized, "headers": headers})
    session_store.append_flags(upload_id, flags)
    session_store.session_mark_stage(upload_id, "upload") # reuse stage marker
    
    return NormalizeResponse(
        upload_id=upload_id,
        total_rows=len(rows),
        flags_added=len(flags),
        sample=normalized[:10],
        headers=headers,
        normalization_summary={"flags": len(flags)}
    )


# â”€â”€ Step 2a: Suggest columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/suggest-columns/{upload_id}", response_model=SuggestColumnsResponse, tags=["Pipeline"])
def suggest_columns_endpoint(upload_id: str):
    session = _get_session_or_404(upload_id)
    raw_rows = session["raw_rows"]
    headers = session["headers"]

    sample_values: Dict[str, List[Any]] = {}
    for col in headers:
        vals = [r[col] for r in raw_rows[:20] if r.get(col) is not None][:3]
        sample_values[col] = vals

    result = suggest_columns(
        source_columns=headers,
        sample_values=sample_values,
        target_format=session["target_format"],
        fuzzy_threshold=session["rules_config"].get("fuzzy_llm_fallback_threshold", 72),
        cutoff=session["rules_config"].get("fuzzy_score_cutoff", 50),
    )

    suggestions_typed = {
        col: [ColumnSuggestion(**s) for s in sug_list]
        for col, sug_list in result["suggestions"].items()
    }
    return SuggestColumnsResponse(
        suggestions=suggestions_typed,
        unmapped=result["unmapped"],
        memory_count=result.get("memory_count", 0),
    )


# â”€â”€ Step 2b: Confirm columns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/confirm-columns/{upload_id}", response_model=ConfirmColumnsResponse, tags=["Pipeline"])
def confirm_columns(upload_id: str, body: ConfirmColumnsRequest):
    session = _get_session_or_404(upload_id)

    column_map = body.column_map
    canonical_to_sources: Dict[str, List[str]] = {}
    for src_col, canonical in column_map.items():
        if canonical is None: continue
        canonical_to_sources.setdefault(canonical, []).append(src_col)

    duplicate_violations = { c: s for c, s in canonical_to_sources.items() if len(s) > 1 }

    if duplicate_violations:
        details = "; ".join(f"'{c}' claimed by: {s}" for c, s in duplicate_violations.items())
        raise HTTPException(status_code=422, detail=f"Mapping violation: {details}")

    unmapped = [k for k, v in column_map.items() if v is None]
    mapped_count = len(column_map) - len(unmapped)

    missing_required = validate_required_fields(column_map, session["target_format"])
    warnings = [f"Required field '{f}' is not mapped." for f in missing_required]

    session_store.update_session(upload_id, {
        "column_map": column_map,
        "unmapped_cols": unmapped,
    })
    session_store.session_mark_stage(upload_id, "column_map")

    try:
        mapping_memory.record_confirmed(column_map, session["target_format"])
    except Exception as exc:
        pass

    return ConfirmColumnsResponse(
        upload_id=upload_id,
        mapped_count=mapped_count,
        unmapped_cols=unmapped,
        missing_required=missing_required,
        warnings=warnings,
    )


# â”€â”€ Step 3: Geocode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/geocode/{upload_id}", response_model=GeocodeResponse, tags=["Pipeline"])
async def geocode_endpoint(upload_id: str):
    session = _get_session_or_404(upload_id)
    # Geocoding in this flow does not require column mapping first.
    
    rules_config = BusinessRulesConfig(**session["rules_config"])
    raw_rows = session["raw_rows"]

    agent_id = "geocoder"
    dispatch_event(upload_id, agent_id, "start")
    dispatch_event(upload_id, agent_id, "log", f"Started geocoding {len(raw_rows)} rows via Geoapify API...")
    
    # Call geocoder
    geocoded_count = 0
    provided_count = 0
    failed_count = 0
    new_flags: List[dict] = []
    geo_rows: List[dict] = []
    
    target_format = session.get("target_format", "AIR")
    
    for idx, row in enumerate(raw_rows):
        if idx % 10 == 0:
            dispatch_event(upload_id, agent_id, "log", f"Processing {idx}/{len(raw_rows)} rows...")

        # In geocoder.py from catai, it expects a column map to find the address fields.
        # But wait! We do Address Normalization first in this MVP, which outputs standardized fields.
        # Let's pass a dummy map since Address Normalization standardized the fields.
        dummy_col_map = {
            "FullAddress": "FullAddress",
            "Street": "Street", "City": "City", "Area": "Area", 
            "PostalCode": "PostalCode", "CountryISO": "CountryISO",
            "STREETNAME": "STREETNAME", "STATECODE": "STATECODE",
            "CNTRYCODE": "CNTRYCODE", "CITY": "CITY"
        }
        
        geo_fields = geocoder.process_row_geocoding(row, dummy_col_map, target_format=target_format)
        row.update(geo_fields)
        geo_rows.append(row)

        status = geo_fields.get("GeocodingStatus", "")
        source = geo_fields.get("Geosource", "")

        if source == "Provided": provided_count += 1
        elif status == "OK": geocoded_count += 1
        else: failed_count += 1

    session_store.update_session(upload_id, {"geo_rows": geo_rows})
    session_store.append_flags(upload_id, new_flags)
    session_store.session_mark_stage(upload_id, "geocoding")

    result = {
        "geocoded": geocoded_count,
        "provided": provided_count,
        "failed": failed_count,
        "flags_added": len(new_flags)
    }
    
    dispatch_event(upload_id, agent_id, "done", result=result)

    return GeocodeResponse(
        upload_id=upload_id, 
        total_rows=len(raw_rows), 
        sample=geo_rows[:10],
        headers=session.get("headers", []),
        **result
    )


# â”€â”€ Step 4: Map codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/map-codes/{upload_id}", response_model=MapCodesResponse, tags=["Pipeline"])
async def map_codes_endpoint(upload_id: str):
    session = _get_session_or_404(upload_id)
    _require_stage(session, "geocoding", "/map-codes")

    geo_rows = session["geo_rows"]
    column_map = session["column_map"]
    rules_config = BusinessRulesConfig(**session["rules_config"])
    target = session["target_format"]
    
    agent_id = "cat_code_mapper"
    dispatch_event(upload_id, agent_id, "start")

    occ_scheme_col, occ_value_col = _find_code_columns(column_map, "occupancy", target)
    const_scheme_col, const_value_col = _find_code_columns(column_map, "construction", target)

    target_occ_scheme = "OccupancyCodeType" if target == "AIR" else "OCCSCHEME"
    target_const_scheme = "ConstructionCodeType" if target == "AIR" else "BLDGSCHEME"

    new_flags: List[dict] = []
    enriched_rows = []
    code_map: Dict[str, Dict] = {}

    if occ_value_col:
        dispatch_event(upload_id, agent_id, "log", "Identifying unique occupancy combinations...")
        occ_items = code_mapper.extract_unique_pairs(geo_rows, occ_scheme_col, occ_value_col)
        occ_results = code_mapper.map_codes(occ_items, target, "occupancy", rules_config)
        for item in occ_items:
            key = code_mapper.build_row_key(item["scheme"], item["value"])
            result = occ_results.get(str(item["index"]), {})
            if result: code_map[f"occ|{key}"] = result

    if const_value_col:
        dispatch_event(upload_id, agent_id, "log", "Identifying unique construction combinations...")
        const_items = code_mapper.extract_unique_pairs(geo_rows, const_scheme_col, const_value_col)
        const_results = code_mapper.map_codes(const_items, target, "construction", rules_config)
        for item in const_items:
            key = code_mapper.build_row_key(item["scheme"], item["value"])
            result = const_results.get(str(item["index"]), {})
            if result: code_map[f"const|{key}"] = result

    occ_by_method: Dict[str, int] = {}
    const_by_method: Dict[str, int] = {}

    dispatch_event(upload_id, agent_id, "log", "Enriching rows with mapped canonical codes...")

    for idx, row in enumerate(geo_rows):
        row = dict(row)

        if occ_value_col:
            scheme = str(row.get(occ_scheme_col) or "").strip()
            value = str(row.get(occ_value_col) or "").strip()
            key = f"occ|{code_mapper.build_row_key(scheme, value)}"
            result = code_map.get(key, {})
            if result:
                row[occ_value_col] = result["code"]
                if not row.get(target_occ_scheme): row[target_occ_scheme] = "ATC" if target == "RMS" else "AIR"
                row["Occupancy_Code"]        = result["code"]
                row["Occupancy_Description"] = result["description"]
                row["Occupancy_Confidence"]  = result["confidence"]
                row["Occupancy_Method"]      = result["method"]
                row["Occupancy_Original"]    = result["original"]
                occ_by_method[result["method"]] = occ_by_method.get(result["method"], 0) + 1

        if const_value_col:
            scheme = str(row.get(const_scheme_col) or "").strip()
            value = str(row.get(const_value_col) or "").strip()
            key = f"const|{code_mapper.build_row_key(scheme, value)}"
            result = code_map.get(key, {})
            if result:
                row[const_value_col] = result["code"]
                scheme_override = result.get("scheme_override")
                if scheme_override == "ISF": row[target_const_scheme] = "ISF"
                elif target == "RMS" and scheme_override: row[target_const_scheme] = scheme_override
                elif not row.get(target_const_scheme): row[target_const_scheme] = target
                
                row["Construction_Code"]        = result["code"]
                row["Construction_Description"] = result["description"]
                row["Construction_Confidence"]  = result["confidence"]
                row["Construction_Method"]      = result["method"]
                row["Construction_Original"]    = result["original"]
                row["Construction_Scheme"]      = row.get(target_const_scheme, target)
                const_by_method[result["method"]] = const_by_method.get(result["method"], 0) + 1

        enriched_rows.append(row)

    session_store.update_session(upload_id, {
        "code_map": code_map,
        "final_rows": enriched_rows,
    })
    session_store.append_flags(upload_id, new_flags)
    session_store.session_mark_stage(upload_id, "code_mapping")

    unique_occ = len([k for k in code_map if k.startswith("occ|")])
    unique_const = len([k for k in code_map if k.startswith("const|")])
    
    result = {
        "unique_occ_pairs": unique_occ,
        "unique_const_pairs": unique_const,
        "occ_by_method": occ_by_method,
        "const_by_method": const_by_method,
        "flags_added": len(new_flags),
    }

    dispatch_event(upload_id, agent_id, "done", result=result)

    return MapCodesResponse(upload_id=upload_id, **result)


# â”€â”€ Step 5: Normalize â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/normalize-values/{upload_id}", response_model=NormalizeResponse, tags=["Pipeline"])
async def normalize_endpoint(upload_id: str):
    session = _get_session_or_404(upload_id)
    _require_stage(session, "code_mapping", "/normalize")
    
    agent_id = "cat_normalizer"
    dispatch_event(upload_id, agent_id, "start")

    rules_config = BusinessRulesConfig(**session["rules_config"])
    # Apply column map to final_rows to ensure canonical names
    final_rows = session["final_rows"] or session["geo_rows"]
    column_map = session["column_map"]
    
    dispatch_event(upload_id, agent_id, "log", "Applying canonical column mappings...")
    final_rows_canonical = _apply_column_map(final_rows, column_map)

    valid_currencies = _VALID_CURRENCIES
    target_format = session.get("target_format", "AIR")

    dispatch_event(upload_id, agent_id, "log", "Normalizing years, floors, areas, and values...")
    normalized, new_flags = normalize_all_rows(final_rows_canonical, rules_config, valid_currencies, target_format=target_format)

    lob_col = "LineOfBusiness" if target_format == "AIR" else ""
    if lob_col and rules_config.line_of_business:
        for r in normalized:
            if not r.get(lob_col): r[lob_col] = rules_config.line_of_business

    session_store.update_session(upload_id, {"final_rows": normalized})
    session_store.append_flags(upload_id, new_flags)
    session_store.session_mark_stage(upload_id, "normalization")

    summary = {
        "year_flags": sum(1 for f in new_flags if "year" in f.get("issue", "")),
        "story_flags": sum(1 for f in new_flags if "story" in f.get("issue", "") or "stories" in f.get("issue", "")),
        "area_flags": sum(1 for f in new_flags if "area" in f.get("issue", "")),
        "value_flags": sum(1 for f in new_flags if "value" in f.get("issue", "")),
        "currency_flags": sum(1 for f in new_flags if "currency" in f.get("issue", "")),
    }

    result = {
        "total_rows": len(normalized),
        "flags_added": len(new_flags),
        "normalization_summary": summary
    }

    dispatch_event(upload_id, agent_id, "done", result=result)

    return NormalizeResponse(upload_id=upload_id, **result)


# â”€â”€ Slip Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _safe_float(v) -> float:
    if v is None: return 0.0
    s = str(v).strip().replace(",", "").replace("$", "").replace("Â£", "").replace("â‚¬", "").replace("Â¥", "").replace("â‚¹", "")
    try: return float(s)
    except (ValueError, TypeError): return 0.0

def _bucket_year(year_val) -> str:
    s = str(year_val or "").strip()
    try:
        y = int(float(s))
        if y <= 0: return "Unknown"
        if y < 1995: return "Pre 1995"
        if y <= 2001: return "1995 â€“ 2001"
        if y <= 2010: return "2002 â€“ 2010"
        if y <= 2017: return "2011 â€“ 2017"
        return "Post 2017"
    except (ValueError, TypeError): return "Unknown"

def _bucket_stories(stories_val) -> str:
    s = str(stories_val or "").strip()
    try:
        n = int(float(s))
        if n <= 0: return "Unknown"
        if n == 1: return "1"
        if n <= 3: return "2â€“3"
        if n <= 7: return "4â€“7"
        return "7+"
    except (ValueError, TypeError): return "Unknown"

@app.get("/summary/{upload_id}", tags=["Output"])
def slip_summary(upload_id: str):
    session = _get_session_or_404(upload_id)
    _require_stage(session, "normalization", "/summary")

    target = session.get("target_format", "AIR")
    final_rows = session.get("final_rows", [])

    if target == "AIR":
        bldg_col, cont_col, bi_col, tiv_col = "BuildingValue", "ContentsValue", "TimeElementValue", "TIV"
        occ_col, const_col = "Occupancy_Description", "Construction_Description"
        year_col, stories_col = "YearBuilt", "NumberOfStories"
        country_col, state_col, city_col, street_col, zip_col = "CountryISO", "Area", "City", "Street", "PostalCode"
        loc_id_col = "LocationID"
    else:
        bldg_col, cont_col, bi_col, tiv_col = "EQCV1VAL", "EQCV2VAL", "EQCV3VAL", "TIV"
        occ_col, const_col = "Occupancy_Description", "Construction_Description"
        year_col, stories_col = "YEARBUILT", "NUMSTORIES"
        country_col, state_col, city_col, street_col, zip_col = "CNTRYCODE", "STATECODE", "CITY", "STREETNAME", "POSTALCODE"
        loc_id_col = "LOCNUM"

    total_bldg = total_cont = total_bi = total_tiv_col = 0.0
    country_state_map: Dict[str, Dict] = {}
    loc_rows = []
    occ_map: Dict[str, float] = {}
    const_map: Dict[str, float] = {}
    year_map: Dict[str, float] = {}
    story_map: Dict[str, float] = {}

    for row in final_rows:
        bldg, cont, bi = _safe_float(row.get(bldg_col)), _safe_float(row.get(cont_col)), _safe_float(row.get(bi_col))
        raw_tiv = _safe_float(row.get(tiv_col))

        if raw_tiv > 0:
            row_tiv = raw_tiv
            total_bldg += bldg; total_cont += cont; total_bi += bi; total_tiv_col += raw_tiv
        else:
            row_tiv = bldg + cont + bi
            total_bldg += bldg; total_cont += cont; total_bi += bi

        country = str(row.get(country_col) or "Unknown").strip() or "Unknown"
        state   = str(row.get(state_col) or "NA").strip() or "NA"
        cs_key  = f"{country}||{state}"
        if cs_key not in country_state_map: country_state_map[cs_key] = {"country": country, "state": state, "count": 0, "tiv": 0.0}
        country_state_map[cs_key]["count"] += 1
        country_state_map[cs_key]["tiv"]   += row_tiv

        loc_rows.append({
            "loc_id":  str(row.get(loc_id_col) or ""), "address": str(row.get(street_col) or ""),
            "city":    str(row.get(city_col) or ""), "state":   str(row.get(state_col) or ""),
            "zip":     str(row.get(zip_col) or ""), "tiv":     row_tiv,
        })

        occ = str(row.get(occ_col) or row.get("OccupancyCode") or "Unknown").strip() or "Unknown"
        occ_map[occ] = occ_map.get(occ, 0.0) + row_tiv

        const = str(row.get(const_col) or row.get("ConstructionCode") or "Unknown").strip() or "Unknown"
        const_map[const] = const_map.get(const, 0.0) + row_tiv

        yb = _bucket_year(row.get(year_col))
        year_map[yb] = year_map.get(yb, 0.0) + row_tiv

        st = _bucket_stories(row.get(stories_col))
        story_map[st] = story_map.get(st, 0.0) + row_tiv

    grand_total = total_tiv_col if total_tiv_col > 0 else (total_bldg + total_cont + total_bi)
    top_locs = sorted(loc_rows, key=lambda r: r["tiv"], reverse=True)[:10]
    cs_list = sorted(country_state_map.values(), key=lambda r: r["tiv"], reverse=True)

    year_order = ["Unknown", "Pre 1995", "1995 â€“ 2001", "2002 â€“ 2010", "2011 â€“ 2017", "Post 2017"]
    year_dist  = [{"label": k, "tiv": year_map.get(k, 0.0)} for k in year_order if k in year_map]

    story_order = ["Unknown", "1", "2â€“3", "4â€“7", "7+"]
    story_dist  = [{"label": k, "tiv": story_map.get(k, 0.0)} for k in story_order if k in story_map]

    return {
        "total_risks": len(final_rows),
        "location_values": {"building": total_bldg, "contents": total_cont, "bi": total_bi, "total": grand_total},
        "country_state": cs_list,
        "top_locations": top_locs,
        "occupancy_dist": [{"label": k, "tiv": v} for k, v in sorted(occ_map.items(), key=lambda x: -x[1])],
        "construction_dist": [{"label": k, "tiv": v} for k, v in sorted(const_map.items(), key=lambda x: -x[1])],
        "year_built_dist": year_dist,
        "stories_dist": story_dist,
    }


# â”€â”€ Download â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/download/{upload_id}", tags=["Output"])
def download(upload_id: str, format: str = Query("xlsx", pattern="^(xlsx|csv)$")):
    session = _get_session_or_404(upload_id)
    _require_stage(session, "normalization", "/download")

    final_rows = session.get("final_rows", [])
    unmapped_cols = session.get("unmapped_cols", [])
    flags = session.get("flags", [])
    target = session.get("target_format", "AIR")
    short_id = upload_id[:8]

    if format == "csv":
        buf = build_csv(final_rows, unmapped_cols, target)
        filename = f"cat_output_{short_id}.csv"
        media_type = "text/csv"
    else:
        buf = build_xlsx(final_rows, unmapped_cols, flags, target, upload_id)
        filename = f"cat_output_{short_id}.xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    return StreamingResponse(buf, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

# â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

