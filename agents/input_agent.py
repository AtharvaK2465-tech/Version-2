"""
Input/Validation Agent
-----------------------
Sanity-checks the imagery the Geo Agent fetched. If cloud cover is too
high, this agent doesn't just fail — it signals the graph to loop back to
the Geo Agent with a shifted date range (see graph.py's conditional edge).
"""
from utils.config import MAX_CLOUD_COVER_PCT
from agents.state import PipelineState

MAX_QUERY_ATTEMPTS = 3


def input_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    cloud_cover = state.get("cloud_cover_pct", 100)
    attempts = state.get("query_attempts", 1)

    if cloud_cover > MAX_CLOUD_COVER_PCT and attempts < MAX_QUERY_ATTEMPTS:
        return {
            **state,
            "imagery_ok": False,
            "validation_notes": (
                f"Cloud cover {cloud_cover:.1f}% exceeds {MAX_CLOUD_COVER_PCT}% "
                f"threshold (attempt {attempts}/{MAX_QUERY_ATTEMPTS}) — retrying "
                "with an earlier date range."
            ),
        }

    if cloud_cover > MAX_CLOUD_COVER_PCT:
        return {
            **state,
            "imagery_ok": True,  # proceed anyway after max retries, but flag it
            "validation_notes": (
                f"Proceeding with {cloud_cover:.1f}% cloud cover after "
                f"{attempts} attempts — results may be less reliable."
            ),
        }

    return {
        **state,
        "imagery_ok": True,
        "validation_notes": f"Imagery accepted ({cloud_cover:.1f}% cloud cover).",
    }


def should_retry(state: PipelineState) -> str:
    """Conditional edge function used by the LangGraph graph."""
    if state.get("error"):
        return "end"
    return "retry" if state.get("imagery_ok") is False else "continue"
