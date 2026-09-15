"""
Report Agent
------------
Turns the Reasoning Agent's structured diagnosis into a readable report.
Uses Gemini when GEMINI_API_KEY is set; otherwise falls back to a clean
template so the app still produces a full report in mock mode.
"""
from agents.state import PipelineState
from utils.config import GEMINI_MOCK, GEMINI_API_KEY


def _mock_report(state: PipelineState) -> str:
    w = state.get("weather", {})
    return f"""## Crop Health Report — Selected Area

**Diagnosis:** {state['diagnosis'].replace('_', ' ').title()}
**Risk level:** {state['risk_level'].upper()}
**Vision confidence:** {state['vision_confidence']:.0%}

### Reasoning
{state['diagnosis_reasoning']}

### Supporting data
- Mean NDVI-derived condition consistent with the vision classification above
- Recent rainfall: {w.get('total_rainfall_mm', 'N/A')} mm over {w.get('days', '?')} days
- Cloud cover at capture: {state.get('cloud_cover_pct', 0):.1f}%

### Recommendation
{"Immediate field inspection recommended." if state['risk_level'] == 'high'
 else "Continue routine monitoring; re-check in 1–2 weeks." if state['risk_level'] == 'medium'
 else "No action needed — area appears healthy."}

*Note: this report was generated in MOCK MODE (no GEMINI_API_KEY set). Add a real key
to .env to generate natural-language reports via the Gemini API.*
"""


def _real_report(state: PipelineState) -> str:
    """LIVE MODE — calls Gemini to turn the structured diagnosis into a
    short, plain-language report. Raises on failure so report_agent()
    below can fall back to the template rather than crash the pipeline."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-flash-latest")

    w = state.get("weather", {})
    prompt = f"""Write a short, clear crop health report for a farmer/agronomist
based on this structured diagnosis for a selected area in Maharashtra, India.

Diagnosis: {state['diagnosis']}
Risk level: {state['risk_level']}
Vision model confidence: {state['vision_confidence']:.0%}
Reasoning: {state['diagnosis_reasoning']}
Recent rainfall: {w.get('total_rainfall_mm', 'unknown')} mm over {w.get('days', '?')} days
Cloud cover at image capture: {state.get('cloud_cover_pct', 0):.1f}%

Structure it with a one-line diagnosis summary, a short "why" explanation
in plain language (avoid jargon like NDVI), and one concrete recommended
action. Keep the whole thing under 200 words. Format as Markdown with
headers matching: ## Crop Health Report, ### Diagnosis, ### Why,
### Recommendation."""

    response = model.generate_content(prompt)
    return response.text


def report_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return {**state, "report_markdown": f"⚠️ Could not generate report: {state['error']}"}

    if GEMINI_MOCK:
        report = _mock_report(state)
    else:
        try:
            report = _real_report(state)
        except Exception as e:
            # Live call failed (quota, network, bad key, etc.) — don't let
            # a report-generation hiccup take down the whole pipeline.
            report = _mock_report(state) + f"\n\n*(Live Gemini call failed, showed template instead: {e})*"

    if state.get("geo_fallback_note"):
        report += f"\n\n*(Note: {state['geo_fallback_note']})*"

    return {**state, "report_markdown": report}
