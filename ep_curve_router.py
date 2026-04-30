"""
ep_curve_router.py — FastAPI router for EP Curve Generation pipeline.

Handles policy file upload, frequency configuration, sub-agent status,
and placeholder EP curve generation.
"""
import io
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

import session as session_store

logger = logging.getLogger("ep_curve")

router = APIRouter(prefix="/ep-curve", tags=["EP Curve"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class FrequencyConfig(BaseModel):
    num_simulations: int = Field(10000, gt=0, description="Number of stochastic simulations")
    time_horizon_years: int = Field(1, gt=0, le=100, description="Simulation time horizon in years")
    frequency_model: str = Field("poisson", pattern="^(poisson|negative_binomial)$")


class PolicyUploadResponse(BaseModel):
    upload_id: str
    row_count: int
    headers: List[str]
    sample: List[Dict[str, Any]]
    validation_warnings: List[str] = []


class FrequencyConfigResponse(BaseModel):
    upload_id: str
    config: Dict[str, Any]
    message: str = "Frequency configuration saved."


class RunEpHazardResponse(BaseModel):
    upload_id: str
    message: str
    peril_config: Dict[str, Any]


class EpCurveStatusResponse(BaseModel):
    upload_id: str
    location_ready: bool = False
    policy_ready: bool = False
    account_ready: bool = False
    peril_ready: bool = False
    frequency_ready: bool = False
    all_ready: bool = False
    ready_count: int = 0
    total_count: int = 5


class EpCurveGenerateResponse(BaseModel):
    upload_id: str
    status: str
    message: str
    oep_curve: List[Dict[str, Any]] = []
    aep_curve: List[Dict[str, Any]] = []


# ── Policy file schema ────────────────────────────────────────────────────────

POLICY_REQUIRED_COLUMNS = [
    "Policy_ID",
    "Account_ID",
    "Policy_Limit",
    "Policy_Deductible",
    "Coverage_Type",
    "Policy_Type",
]

POLICY_OPTIONAL_COLUMNS = [
    "CoinsuranceParticipation",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_session_or_404(upload_id: str) -> dict:
    try:
        return session_store.require_session(upload_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{upload_id}' not found or expired.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload-policy/{upload_id}", response_model=PolicyUploadResponse)
async def upload_policy_file(upload_id: str, file: UploadFile = File(...)):
    """
    Upload a policy file (CSV/XLSX) for the EP Curve pipeline.
    Validates against the expected schema and stores parsed rows in the session.
    """
    session = _get_session_or_404(upload_id)

    content = await file.read()
    fname = (file.filename or "").lower()
    warnings: List[str] = []

    # Parse file
    try:
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False, encoding="latin-1")
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False, sheet_name=0)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or XLSX.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"File parsing error: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Policy file contains no data rows.")

    # Clean whitespace
    df = df.map(lambda v: v.strip() if isinstance(v, str) and v.strip() != "" else None)

    # Validate schema — check for required columns
    actual_cols = set(df.columns)
    missing = [c for c in POLICY_REQUIRED_COLUMNS if c not in actual_cols]
    if missing:
        warnings.append(f"Missing required columns: {', '.join(missing)}. EP Curve generation may be incomplete.")

    headers = list(df.columns)
    rows = df.to_dict(orient="records")
    sample = rows[:10]

    # Store in session
    session_store.update_session(upload_id, {
        "ep_policy_rows": rows,
        "ep_policy_headers": headers,
        "ep_policy_file_name": file.filename,
    })

    logger.info(f"Session {upload_id}: policy file uploaded — {len(rows)} rows, {len(headers)} columns")

    return PolicyUploadResponse(
        upload_id=upload_id,
        row_count=len(rows),
        headers=headers,
        sample=sample,
        validation_warnings=warnings,
    )


@router.post("/configure-frequency/{upload_id}", response_model=FrequencyConfigResponse)
async def configure_frequency(upload_id: str, config: FrequencyConfig):
    """Save frequency/simulation configuration for the EP Curve pipeline."""
    session = _get_session_or_404(upload_id)

    config_dict = config.model_dump()

    session_store.update_session(upload_id, {
        "ep_frequency_config": config_dict,
    })

    logger.info(f"Session {upload_id}: frequency config saved — {config_dict}")

    return FrequencyConfigResponse(
        upload_id=upload_id,
        config=config_dict,
    )


@router.get("/status/{upload_id}", response_model=EpCurveStatusResponse)
async def ep_curve_status(upload_id: str):
    """
    Return readiness status of all 5 EP Curve sub-agents.
    Checks session state for each input's availability.
    """
    session = _get_session_or_404(upload_id)

    stages = session.get("stages_complete", {})

    # Location file: ready if SOV pipeline normalization is complete
    location_ready = stages.get("normalization", False)

    # Policy file: ready if user uploaded a policy file
    policy_ready = bool(session.get("ep_policy_rows"))

    # Account file: ready if SOV pipeline normalization is complete
    account_ready = stages.get("normalization", False)

    # Peril + Region: ready if EP hazard assessment has run (check session for ep_peril_config)
    peril_ready = bool(session.get("ep_peril_config"))

    # Frequency config: ready if user configured it
    frequency_ready = bool(session.get("ep_frequency_config"))

    readiness = [location_ready, policy_ready, account_ready, peril_ready, frequency_ready]
    ready_count = sum(readiness)

    return EpCurveStatusResponse(
        upload_id=upload_id,
        location_ready=location_ready,
        policy_ready=policy_ready,
        account_ready=account_ready,
        peril_ready=peril_ready,
        frequency_ready=frequency_ready,
        all_ready=all(readiness),
        ready_count=ready_count,
    )


@router.post("/run-hazard/{upload_id}", response_model=RunEpHazardResponse)
async def run_ep_hazard_assessment(upload_id: str):
    """
    Run the EP-specific Hazard Assessment.
    Reads normalized locations, runs peril models, and saves peril_config.
    """
    import asyncio
    session = _get_session_or_404(upload_id)
    
    # Simulate processing time for hazard assessment models
    await asyncio.sleep(2.5)

    # Placeholder result
    peril_config = {
        "earthquake_regions": ["US_WestCoast", "Japan"],
        "wind_regions": ["US_GulfCoast", "US_EastCoast"],
        "peril_count": 4,
        "locations_mapped": len(session.get("mapped_data", []))
    }

    session_store.update_session(upload_id, {
        "ep_peril_config": peril_config,
    })

    logger.info(f"Session {upload_id}: EP Hazard assessment complete.")

    return RunEpHazardResponse(
        upload_id=upload_id,
        message="EP Hazard assessment completed successfully.",
        peril_config=peril_config
    )


@router.post("/generate/{upload_id}", response_model=EpCurveGenerateResponse)
async def generate_ep_curve(upload_id: str):
    """
    Placeholder EP Curve generation endpoint.
    Validates that all inputs are ready and returns mock EP curve data.
    Future: actual Monte Carlo simulation.
    """
    session = _get_session_or_404(upload_id)

    # Check readiness
    stages = session.get("stages_complete", {})
    location_ready = stages.get("normalization", False)
    policy_ready = bool(session.get("ep_policy_rows"))
    account_ready = stages.get("normalization", False)
    peril_ready = bool(session.get("ep_peril_config"))
    frequency_ready = bool(session.get("ep_frequency_config"))

    not_ready = []
    if not location_ready:
        not_ready.append("Location File (SOV COPE not complete)")
    if not policy_ready:
        not_ready.append("Policy File (not uploaded)")
    if not account_ready:
        not_ready.append("Account File (SOV COPE not complete)")
    if not peril_ready:
        not_ready.append("Peril + Region (Hazard Assessment not complete)")
    if not frequency_ready:
        not_ready.append("Frequency Configuration (not set)")

    if not_ready:
        raise HTTPException(
            status_code=422,
            detail=f"EP Curve generation requires all inputs. Missing: {'; '.join(not_ready)}"
        )

    # ── Placeholder: generate mock EP curve data ──────────────────────────
    freq_config = session.get("ep_frequency_config", {})
    num_sims = freq_config.get("num_simulations", 10000)

    # Mock OEP curve (Occurrence Exceedance Probability)
    return_periods = [10, 25, 50, 100, 250, 500, 1000]
    oep_curve = [
        {
            "return_period": rp,
            "exceedance_probability": round(1.0 / rp, 6),
            "loss_amount": round(rp * 50000 * (1 + (rp / 1000)), 2),  # placeholder formula
        }
        for rp in return_periods
    ]

    # Mock AEP curve (Aggregate Exceedance Probability)
    aep_curve = [
        {
            "return_period": rp,
            "exceedance_probability": round(1.0 / rp, 6),
            "loss_amount": round(rp * 75000 * (1 + (rp / 800)), 2),  # placeholder formula
        }
        for rp in return_periods
    ]

    result = {
        "oep_curve": oep_curve,
        "aep_curve": aep_curve,
    }

    session_store.update_session(upload_id, {"ep_curve_result": result})
    session_store.session_mark_stage(upload_id, "ep_curve")

    logger.info(f"Session {upload_id}: EP Curve generated (placeholder) — {len(return_periods)} return periods")

    return EpCurveGenerateResponse(
        upload_id=upload_id,
        status="complete",
        message=f"EP Curve generated with {num_sims} simulations (placeholder).",
        oep_curve=oep_curve,
        aep_curve=aep_curve,
    )
