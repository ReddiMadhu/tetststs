"""
rules.py — BusinessRulesConfig Pydantic model with sensible defaults.
Passed as an immutable parameter to every processing function.
"""
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime


class BusinessRulesConfig(BaseModel):
    # ── Year validation ────────────────────────────────────────────────────────
    year_min: int = 1800
    year_max: int = Field(default_factory=lambda: datetime.now().year)
    invalid_year_action: Literal["none", "reset_year", "set_default", "flag_review"] = "flag_review"
    year_default: int = 1980

    # ── Global overrides ───────────────────────────────────────────────
    line_of_business: str = ""
    policy_id: str = ""
    insured_name: str = ""


    # ── Stories ────────────────────────────────────────────────────────────────
    max_stories_wood_frame: int = 3
    stories_exceeded_action: Literal["none", "reset_construction", "reset_stories"] = "none"

    # ── Area ───────────────────────────────────────────────────────────────────
    min_area_sqft: float = 100.0
    invalid_area_action: Literal["none", "flag_review"] = "flag_review"

    # ── Financial values ───────────────────────────────────────────────────────
    max_building_value: float = 100_000_000
    max_contents_value: float = 50_000_000
    max_bi_value: float = 25_000_000
    invalid_value_action: Literal["none", "reset_value", "flag_review"] = "flag_review"

    # ── Code confidence thresholds ─────────────────────────────────────────────
    occ_confidence_threshold: float = 0.70
    const_confidence_threshold: float = 0.70

    # ── Deterministic → LLM threshold ─────────────────────────────────────────
    deterministic_score_threshold: float = 0.85   # below this → try LLM
    llm_confidence_threshold: float = 0.70        # below this → try TF-IDF
    tfidf_confidence_threshold: float = 0.50      # below this → use default

    # ── Fuzzy column mapping thresholds ───────────────────────────────────────
    fuzzy_score_cutoff: int = 50          # minimum score to keep a fuzzy result
    fuzzy_llm_fallback_threshold: int = 72  # below this → also try LLM

    # ── Default codes ─────────────────────────────────────────────────────────
    default_occ_code_air: str = "311"     # General Commercial
    default_const_code_air: str = "100"   # Unknown Construction
    default_occ_code_rms: str = "Com1"
    default_const_code_rms: str = "W1"

    class Config:
        validate_default = True
