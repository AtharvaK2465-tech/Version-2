"""
The shared state object that flows through every node in the LangGraph
pipeline. Each agent reads what it needs and writes its output back in.
"""
from typing import TypedDict, Optional, Any
from datetime import date


class PipelineState(TypedDict, total=False):
    # --- input from the UI ---
    polygon_geojson: dict
    date_start: date
    date_end: date

    # --- Geo Agent output ---
    raster: Optional[Any]          # clipped multi-band array (mock or real)
    ndvi: Optional[Any]
    ndmi: Optional[Any]
    ndre: Optional[Any]
    cloud_cover_pct: Optional[float]
    query_attempts: int            # tracks re-query loop count
    geo_fallback_note: Optional[str]  # set if a live Sentinel Hub call failed and fell back to mock

    # --- Input/Validation Agent output ---
    imagery_ok: Optional[bool]
    validation_notes: Optional[str]

    # --- Vision Agent output ---
    vision_classification: Optional[str]   # disease / water_stress / nutrient_deficiency / healthy
    vision_confidence: Optional[float]

    # --- weather fusion (Reasoning Agent input) ---
    weather: Optional[dict]

    # --- Reasoning Agent output ---
    diagnosis: Optional[str]
    diagnosis_reasoning: Optional[str]
    risk_level: Optional[str]      # low / medium / high

    # --- Report Agent output ---
    report_markdown: Optional[str]

    # --- control flow ---
    error: Optional[str]
