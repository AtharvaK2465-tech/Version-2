"""
Reasoning Agent
---------------
Combines the Vision Agent's classification with rainfall data to decide
between competing explanations — e.g. "water_stress" reads differently if
there's been no rain for two weeks vs. if rainfall was normal (pointing
more toward disease or nutrient issues instead).
"""
from shapely.geometry import shape

from agents.state import PipelineState
from utils.weather import get_recent_weather


def _polygon_centroid(geojson: dict) -> tuple[float, float]:
    geom = shape(geojson)
    c = geom.centroid
    return c.y, c.x  # lat, lon


def reasoning_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    lat, lon = _polygon_centroid(state["polygon_geojson"])
    weather = get_recent_weather(lat, lon, state["date_end"])

    classification = state["vision_classification"]
    confidence = state["vision_confidence"]
    rainfall = weather.get("total_rainfall_mm")

    reasoning_lines = [
        f"Vision Agent flagged '{classification}' with {confidence:.0%} confidence."
    ]

    diagnosis = classification
    risk_level = "low"

    if classification == "water_stress" and rainfall is not None:
        if rainfall < 10:
            reasoning_lines.append(
                f"Only {rainfall}mm rainfall in the past {weather['days']} days — "
                "low rainfall supports a genuine water-stress diagnosis."
            )
            risk_level = "high"
        else:
            reasoning_lines.append(
                f"{rainfall}mm rainfall recorded recently, which is enough that "
                "water stress alone is less likely — re-flagging as possible disease."
            )
            diagnosis = "disease (reclassified from water_stress)"
            risk_level = "high"
    elif classification == "disease":
        reasoning_lines.append("Spectral signature consistent with disease/pathogen stress.")
        risk_level = "high"
    elif classification == "nutrient_deficiency":
        reasoning_lines.append("Moderate NDVI depression without strong moisture signal.")
        risk_level = "medium"
    else:
        reasoning_lines.append("Indices within a healthy range for the selected area.")
        risk_level = "low"

    return {
        **state,
        "weather": weather,
        "diagnosis": diagnosis,
        "diagnosis_reasoning": " ".join(reasoning_lines),
        "risk_level": risk_level,
    }
