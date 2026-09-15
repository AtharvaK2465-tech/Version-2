"""
Streamlit entry point. Draw a polygon on the map over Maharashtra, pick a
date range, and run the multi-agent pipeline against exactly that area.

Run with:  streamlit run app.py
"""
from datetime import date, timedelta

import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

from utils.config import MAHARASHTRA_BBOX, SENTINEL_HUB_MOCK, GEMINI_MOCK
from agents.graph import run_pipeline

st.set_page_config(page_title="Crop Health Monitor — Maharashtra", layout="wide")
st.title("🌾 Multi-Agent Crop Health Monitoring — Maharashtra")

if SENTINEL_HUB_MOCK or GEMINI_MOCK:
    st.info(
        "Running in **mock mode** for "
        + ", ".join(
            [s for s, flag in [("Sentinel Hub", SENTINEL_HUB_MOCK), ("Gemini", GEMINI_MOCK)] if flag]
        )
        + " — add real API keys to `.env` to switch to live data. "
        "The full pipeline still runs end-to-end with realistic synthetic data.",
        icon="🧪",
    )

col_map, col_controls = st.columns([2, 1])

with col_controls:
    st.subheader("1. Select date range")
    end_date = st.date_input("End date", value=date.today() - timedelta(days=3))
    start_date = st.date_input("Start date", value=end_date - timedelta(days=14))
    run_button = st.button("Run analysis on drawn area", type="primary")

with col_map:
    st.subheader("2. Draw your area of interest")
    minx, miny, maxx, maxy = MAHARASHTRA_BBOX
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    # Zoom in further before drawing — helps confirm you're on the right
    # spot before you commit to a polygon, especially in dense areas.
    tile_choice = st.radio(
        "Basemap",
        ["Satellite (Esri)", "Satellite + labels (Esri hybrid)", "Streets (OSM)"],
        horizontal=True,
        help=(
            "Esri's satellite imagery isn't updated on the same schedule "
            "everywhere — rural Maharashtra can lag behind what you'd see "
            "on Google Maps. If buildings/roads look outdated, switch to "
            "Streets to confirm your area against current OpenStreetMap "
            "data, then switch back to draw."
        ),
    )

    m = folium.Map(location=center, zoom_start=6, tiles=None)

    if tile_choice == "Streets (OSM)":
        folium.TileLayer(
            "OpenStreetMap", name="Streets", max_zoom=19
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri, Maxar, Earthstar Geographics",
            name="Satellite",
            max_zoom=19,
        ).add_to(m)
        if tile_choice == "Satellite + labels (Esri hybrid)":
            # Reference overlay: place/road labels and boundaries drawn on
            # top of the satellite tiles, so you can see place names and
            # roads without losing the imagery underneath.
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="Labels",
                overlay=True,
                control=False,
                max_zoom=19,
            ).add_to(m)

    Draw(
        export=False,
        draw_options={"polygon": True, "rectangle": True, "circle": False,
                      "circlemarker": False, "marker": False, "polyline": False},
        edit_options={"edit": True},
    ).add_to(m)

    map_data = st_folium(m, height=520, width=None, key="area_map")

st.divider()

polygon_geojson = None
if map_data and map_data.get("last_active_drawing"):
    polygon_geojson = map_data["last_active_drawing"]["geometry"]

if run_button:
    if polygon_geojson is None:
        st.error("Draw a polygon or rectangle on the map first.")
    else:
        with st.spinner("Running Geo → Input → Vision → Reasoning → Report agents..."):
            result = run_pipeline(polygon_geojson, start_date, end_date)

        if result.get("error"):
            st.error(result["error"])
        else:
            st.success(result.get("validation_notes", "Pipeline completed."))

            c1, c2, c3 = st.columns(3)
            c1.metric("Diagnosis", result["diagnosis"].replace("_", " ").title())
            c2.metric("Risk level", result["risk_level"].upper())
            c3.metric("Vision confidence", f"{result['vision_confidence']:.0%}")

            st.subheader("NDVI map (selected area)")
            ndvi = result["ndvi"]
            # Normalize to 0-255 for a quick visual; a proper colored heatmap
            # (e.g. matplotlib 'RdYlGn' colormap) is a natural next upgrade.
            ndvi_img = ((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-9) * 255).astype(np.uint8)
            st.image(ndvi_img, caption="Darker = lower NDVI (more stressed vegetation)", width=300)

            st.subheader("Report")
            st.markdown(result["report_markdown"])

st.caption(
    "Multi-agent pipeline: Geo/Area Agent → Input/Validation Agent "
    "(with cloud-cover retry loop) → Vision Agent → Reasoning Agent → Report Agent."
)
