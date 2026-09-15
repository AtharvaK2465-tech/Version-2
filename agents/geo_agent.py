"""
Geo/Area Agent
--------------
Takes the polygon the user drew on the map + a date range, fetches
Sentinel-2 imagery clipped to that polygon, and computes NDVI/NDMI/NDRE.

Runs in MOCK MODE (synthetic but realistic-looking rasters) when Sentinel
Hub credentials aren't set, so the rest of the pipeline can be built and
demoed without real API access.
"""
import numpy as np
from datetime import timedelta

from utils.config import SENTINEL_HUB_MOCK, DATE_RANGE_DAYS
from utils.geo import validate_polygon
from agents.state import PipelineState


def _mock_fetch_imagery(state: PipelineState) -> tuple[np.ndarray, float]:
    """Generates a fake 4-band (R,G,B,NIR) raster and a plausible cloud
    cover percentage, seeded by the polygon so results are stable across
    re-runs of the same area."""
    seed = abs(hash(str(state["polygon_geojson"]))) % (2**32)
    rng = np.random.default_rng(seed)

    h, w = 128, 128
    red = rng.uniform(0.05, 0.25, (h, w)).astype(np.float32)
    green = rng.uniform(0.05, 0.30, (h, w)).astype(np.float32)
    blue = rng.uniform(0.05, 0.20, (h, w)).astype(np.float32)
    nir = rng.uniform(0.20, 0.55, (h, w)).astype(np.float32)

    # Sprinkle in a "stressed patch" so downstream agents have something
    # interesting to detect in mock mode.
    cy, cx = rng.integers(30, 98, 2)
    yy, xx = np.ogrid[:h, :w]
    patch = (yy - cy) ** 2 + (xx - cx) ** 2 <= 20 ** 2
    nir[patch] *= 0.5   # stressed vegetation reflects less NIR
    red[patch] *= 1.3

    raster = np.stack([red, green, blue, nir], axis=-1)
    cloud_cover = float(rng.uniform(0, 45))
    return raster, cloud_cover


_EVALSCRIPT_6BAND = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B05", "B08", "B11"],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  // order: red, green, blue, nir, swir(B11), rededge(B05)
  return [sample.B04, sample.B03, sample.B02, sample.B08, sample.B11, sample.B05];
}
"""


def _bbox_from_polygon(polygon_geojson: dict):
    from shapely.geometry import shape
    return shape(polygon_geojson).bounds  # (minx, miny, maxx, maxy)


def _real_fetch_imagery(state: PipelineState) -> tuple[np.ndarray, float]:
    """LIVE MODE — queries Sentinel Hub's Catalog API for the least-cloudy
    Sentinel-2 L2A scene in the requested date range over the drawn
    polygon, then fetches a 6-band clip (RGB, NIR, SWIR, red-edge) sized
    to stay well under Sentinel Hub's per-request pixel limits regardless
    of how large the drawn area is."""
    from sentinelhub import (
        SHConfig, BBox, CRS, SentinelHubRequest, SentinelHubCatalog,
        DataCollection, MimeType, bbox_to_dimensions,
    )
    from utils.config import SENTINEL_HUB_CLIENT_ID, SENTINEL_HUB_CLIENT_SECRET

    config = SHConfig()
    config.sh_client_id = SENTINEL_HUB_CLIENT_ID
    config.sh_client_secret = SENTINEL_HUB_CLIENT_SECRET

    minx, miny, maxx, maxy = _bbox_from_polygon(state["polygon_geojson"])
    bbox = BBox(bbox=(minx, miny, maxx, maxy), crs=CRS.WGS84)

    # Pick a resolution that keeps the output around 512px on the long
    # side no matter how big the drawn polygon is — avoids "too many
    # pixels requested" errors on large areas.
    lon_span_km = (maxx - minx) * 105.0
    lat_span_km = (maxy - miny) * 111.0
    span_km = max(lon_span_km, lat_span_km, 0.5)
    resolution = max((span_km * 1000) / 512, 10)  # meters/pixel, floor at native 10m
    size = bbox_to_dimensions(bbox, resolution=resolution)
    size = (min(size[0], 1024), min(size[1], 1024))

    time_interval = (state["date_start"].isoformat(), state["date_end"].isoformat())

    catalog = SentinelHubCatalog(config=config)
    search_results = list(catalog.search(
        DataCollection.SENTINEL2_L2A,
        bbox=bbox,
        time=time_interval,
        fields={"include": ["properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
    ))
    if not search_results:
        raise RuntimeError(
            f"No Sentinel-2 scenes found for this area between "
            f"{state['date_start']} and {state['date_end']}. Try widening the date range."
        )
    best_scene = min(search_results, key=lambda r: r["properties"].get("eo:cloud_cover", 100))
    cloud_cover = float(best_scene["properties"].get("eo:cloud_cover", 100))

    request = SentinelHubRequest(
        evalscript=_EVALSCRIPT_6BAND,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=time_interval,
            mosaicking_order="leastCC",
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    raster = request.get_data()[0].astype(np.float32)
    return raster, cloud_cover


def _compute_indices(raster: np.ndarray) -> dict:
    """raster channels: [red, green, blue, nir, (swir, rededge)].
    Mock mode only supplies 4 bands, so NDMI/NDRE fall back to a red-based
    approximation there; live mode supplies real SWIR/red-edge bands for
    accurate NDMI/NDRE."""
    red = raster[..., 0]
    nir = raster[..., 3]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)

    if raster.shape[-1] >= 6:
        swir = raster[..., 4]
        rededge = raster[..., 5]
        ndmi = (nir - swir) / (nir + swir + eps)
        ndre = (nir - rededge) / (nir + rededge + eps)
    else:
        ndmi = (nir - red * 0.8) / (nir + red * 0.8 + eps)
        ndre = (nir - red * 0.6) / (nir + red * 0.6 + eps)

    return {"ndvi": ndvi, "ndmi": ndmi, "ndre": ndre}


def geo_agent(state: PipelineState) -> PipelineState:
    is_valid, message = validate_polygon(state["polygon_geojson"])
    if not is_valid:
        return {**state, "error": message}

    attempts = state.get("query_attempts", 0) + 1
    live_fetch_note = None

    if SENTINEL_HUB_MOCK:
        raster, cloud_cover = _mock_fetch_imagery(state)
    else:
        try:
            raster, cloud_cover = _real_fetch_imagery(state)
        except Exception as e:
            # Live Sentinel Hub call failed (quota, auth, no scenes for the
            # date range, network, etc.) — fall back to mock imagery rather
            # than crash the whole pipeline mid-demo, but say so clearly.
            raster, cloud_cover = _mock_fetch_imagery(state)
            live_fetch_note = f"Live Sentinel Hub fetch failed, used synthetic imagery instead: {e}"

    indices = _compute_indices(raster)

    result = {
        **state,
        "raster": raster,
        "ndvi": indices["ndvi"],
        "ndmi": indices["ndmi"],
        "ndre": indices["ndre"],
        "cloud_cover_pct": cloud_cover,
        "query_attempts": attempts,
    }
    if live_fetch_note:
        result["geo_fallback_note"] = live_fetch_note
    return result


def widen_date_range(state: PipelineState) -> PipelineState:
    """Called on the retry edge when cloud cover is too high — shifts the
    search window back to look for a clearer scene."""
    new_start = state["date_start"] - timedelta(days=DATE_RANGE_DAYS)
    return {**state, "date_start": new_start}
