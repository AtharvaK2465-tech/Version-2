# Multi-Agent Crop Health Monitoring — Maharashtra

Satellite-imagery-based crop disease/stress detection for a user-selected
area of Maharashtra, built as a LangGraph multi-agent pipeline with a
Streamlit + map-based UI for area selection.

## How it works

1. You draw a polygon/rectangle on the map (Leaflet, via `streamlit-folium`).
2. **Geo/Area Agent** fetches Sentinel-2 imagery clipped to that polygon and
   computes NDVI / NDMI / NDRE.
3. **Input/Validation Agent** checks cloud cover. If it's too high, it loops
   back to the Geo Agent with a shifted date range (a real conditional edge
   in the graph — not just a linear pipeline).
4. **Vision Agent** classifies the area: healthy / water stress / nutrient
   deficiency / disease.
5. **Reasoning Agent** cross-checks that classification against recent
   rainfall (Open-Meteo) to distinguish, e.g., drought stress from disease.
6. **Report Agent** turns the structured diagnosis into a readable report
   via Gemini.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in real API keys (optional, see below)
streamlit run app.py
```

## Mock mode vs. live mode

**You don't need any API keys to run this.** If `SENTINEL_HUB_CLIENT_ID`/
`SENTINEL_HUB_CLIENT_SECRET` or `GEMINI_API_KEY` are missing from `.env`,
the app automatically runs those pieces in mock mode:

- Sentinel Hub → synthetic but realistic 4-band rasters with a seeded
  "stressed patch," so the vision/reasoning agents always have something
  meaningful to detect.
- Gemini → a clean template-based report instead of an LLM-generated one.

This means you can build, demo, and test the *entire* pipeline — including
the retry loop and the map UI — before you have real API access. Every
place that needs a real key is marked `# TODO: insert API key` in
`agents/geo_agent.py` and `agents/report_agent.py`, with the real
implementation sketched in a docstring right above the `raise
NotImplementedError`.

## Getting real API keys

- **Sentinel Hub**: free trial / student tier at https://www.sentinel-hub.com
- **Gemini**: free tier key at https://aistudio.google.com/apikey

## Project structure

```
agents/
  state.py           # shared state schema passed between all nodes
  geo_agent.py        # fetch + clip imagery, compute indices
  input_agent.py       # cloud-cover validation + retry decision
  vision_agent.py       # stress/disease classification
  reasoning_agent.py     # fuses vision output with weather data
  report_agent.py        # generates the final report
  graph.py                # wires all agents into a LangGraph StateGraph
utils/
  config.py           # env vars, mock-mode flags, Maharashtra bbox
  geo.py               # polygon validation (bounds, area cap)
  weather.py            # Open-Meteo rainfall lookup (no key needed)
tests/
  test_geo.py          # polygon validation tests
  test_vision.py         # index computation + classification heuristic tests
app.py                  # Streamlit UI (map + controls + results)
```

## Running tests

```bash
pytest tests/ -v
```

## Next steps / extensions

- Swap the NDVI-threshold heuristic in `vision_agent.py` for a trained
  CNN once you have labeled imagery (hook is at `_model_predict`).
- Wire real Sentinel Hub / Gemini calls in the two `NotImplementedError`
  stubs once you have API keys.
- Add the optional Alert/Monitoring Agent: re-run the pipeline on a saved
  polygon on a schedule, notify only when `risk_level` crosses a threshold.
- Overlay the risk zone back onto the Leaflet map (currently only a plain
  NDVI grayscale image is shown).
- Use real Sentinel-2 band-edge (B05/B8A/B11) values for NDMI/NDRE once
  live imagery is available — they're currently approximated from
  RGB+NIR only, which is a known simplification worth noting in your
  report/viva.
