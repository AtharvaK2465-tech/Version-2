"""
Builds the LangGraph state graph for the full pipeline:

    geo_agent -> input_agent --(cloud cover too high, retries left)--> geo_agent (retry)
                              --(ok)--> vision_agent -> reasoning_agent -> report_agent -> END
                              --(error)--> END

The retry loop is a real conditional edge, not just a comment — this is
the "agentic" part worth highlighting in a project demo/viva: the system
makes a decision and changes its own next step based on data quality.
"""
from langgraph.graph import StateGraph, END

from agents.state import PipelineState
from agents.geo_agent import geo_agent, widen_date_range
from agents.input_agent import input_agent, should_retry
from agents.vision_agent import vision_agent
from agents.reasoning_agent import reasoning_agent
from agents.report_agent import report_agent


def _retry_and_refetch(state: PipelineState) -> PipelineState:
    state = widen_date_range(state)
    return geo_agent(state)


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("geo_agent", geo_agent)
    graph.add_node("input_agent", input_agent)
    graph.add_node("retry_geo_agent", _retry_and_refetch)
    graph.add_node("vision_agent", vision_agent)
    graph.add_node("reasoning_agent", reasoning_agent)
    graph.add_node("report_agent", report_agent)

    graph.set_entry_point("geo_agent")
    graph.add_edge("geo_agent", "input_agent")

    graph.add_conditional_edges(
        "input_agent",
        should_retry,
        {
            "retry": "retry_geo_agent",
            "continue": "vision_agent",
            "end": END,
        },
    )
    # after a retry, re-validate the newly fetched imagery
    graph.add_edge("retry_geo_agent", "input_agent")

    graph.add_edge("vision_agent", "reasoning_agent")
    graph.add_edge("reasoning_agent", "report_agent")
    graph.add_edge("report_agent", END)

    return graph.compile()


def run_pipeline(polygon_geojson: dict, date_start, date_end) -> PipelineState:
    app = build_graph()
    initial_state: PipelineState = {
        "polygon_geojson": polygon_geojson,
        "date_start": date_start,
        "date_end": date_end,
        "query_attempts": 0,
    }
    return app.invoke(initial_state)
