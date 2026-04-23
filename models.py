"""
models.py — All Pydantic request/response schemas for the CAT pipeline API.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
from rules import BusinessRulesConfig


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    upload_id: str
    row_count: int
    headers: List[str]
    sample: List[Dict[str, Any]]
    target_format: Literal["AIR", "RMS"]


# ── Column Mapping ─────────────────────────────────────────────────────────────

class ColumnSuggestion(BaseModel):
    canonical: str
    score: float
    method: Literal["fuzzy", "llm", "memory"]
    reason: Optional[str] = None
    count: Optional[int] = None   # number of confirmed sessions (memory only)


class SuggestColumnsResponse(BaseModel):
    suggestions: Dict[str, List[ColumnSuggestion]]
    unmapped: List[str]
    memory_count: int = 0   # number of columns resolved from learned memory


class ConfirmColumnsRequest(BaseModel):
    column_map: Dict[str, Optional[str]]  # source_col → canonical_field or null


class ConfirmColumnsResponse(BaseModel):
    upload_id: str
    mapped_count: int
    unmapped_cols: List[str]
    missing_required: List[str]    # required fields not covered
    warnings: List[str]


# ── Geocoding ──────────────────────────────────────────────────────────────────

class GeocodeResponse(BaseModel):
    upload_id: str
    total_rows: int
    geocoded: int
    provided: int
    failed: int
    flags_added: int
    sample: List[Dict[str, Any]] = []
    headers: List[str] = []
    diff_data: Optional[Dict[str, Any]] = None


# ── Code Mapping ───────────────────────────────────────────────────────────────

class CodeResult(BaseModel):
    code: str
    confidence: float
    description: str
    method: Literal["rule", "llm", "tfidf", "default", "user_override"]
    original: str
    alternatives: List[Dict[str, Any]] = []
    reasoning: Optional[str] = None


class MapCodesResponse(BaseModel):
    upload_id: str
    unique_occ_pairs: int
    unique_const_pairs: int
    occ_by_method: Dict[str, int]
    const_by_method: Dict[str, int]
    flags_added: int
    summary_text: Optional[str] = None
    diff_data: Optional[Dict[str, Any]] = None


# ── Normalization ──────────────────────────────────────────────────────────────

class NormalizeResponse(BaseModel):
    upload_id: str
    total_rows: int
    flags_added: int
    sample: List[Dict[str, Any]] = []
    headers: List[str] = []
    normalization_summary: Dict[str, Any] = {}
    summary_text: Optional[str] = None
    diff_data: Optional[Dict[str, Any]] = None


# ── Review & Corrections ───────────────────────────────────────────────────────

class FlagEntry(BaseModel):
    row_index: int
    field: str
    issue: str
    current_value: Any
    confidence: Optional[float] = None
    alternatives: List[Any] = []
    message: str


class ReviewResponse(BaseModel):
    upload_id: str
    flags: List[FlagEntry]
    stages_complete: Dict[str, bool]


class CorrectionItem(BaseModel):
    row_index: int
    field: str
    new_value: Any


class CorrectRequest(BaseModel):
    corrections: List[CorrectionItem]


class CorrectResponse(BaseModel):
    applied: int
    flags_removed: int


# ── Download ───────────────────────────────────────────────────────────────────

class DownloadQueryParams(BaseModel):
    format: Literal["xlsx", "csv"] = "xlsx"


# ── Session Info ───────────────────────────────────────────────────────────────

class SessionInfoResponse(BaseModel):
    upload_id: str
    created_at: float
    target_format: str
    row_count: int
    stages_complete: Dict[str, bool]
    flag_count: int
    rules_config: Dict[str, Any]
