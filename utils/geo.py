"""
Geometry helpers for validating a user-drawn polygon before it's sent to
the satellite imagery API.
"""
from shapely.geometry import shape, box
from shapely.geometry.base import BaseGeometry

from utils.config import MAHARASHTRA_BBOX, MAX_AREA_KM2

# Rough degrees-to-km conversion at Maharashtra's latitude (~19°N).
# Good enough for a sanity-check cap, not for precise area billing.
_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LON = 105.0  # cos(19°) * 111


def polygon_from_geojson(geojson: dict) -> BaseGeometry:
    """Convert a GeoJSON geometry dict (as returned by streamlit-folium's
    draw tool) into a shapely geometry."""
    return shape(geojson)


def approx_area_km2(geom: BaseGeometry) -> float:
    """Rough polygon area in km^2 using a flat-earth approximation.
    Fine for a UI sanity check; not for scientific area measurement."""
    minx, miny, maxx, maxy = geom.bounds
    # Approximate by scaling the polygon's degree-area by local km/deg factors
    deg_area = geom.area  # in square degrees
    km2 = deg_area * _KM_PER_DEG_LAT * _KM_PER_DEG_LON
    return abs(km2)


def validate_polygon(geojson: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, message). Checks:
      1. Geometry parses correctly
      2. It intersects Maharashtra's bounding box
      3. It's under the max area threshold
    """
    try:
        geom = polygon_from_geojson(geojson)
    except Exception as e:
        return False, f"Could not parse the drawn shape: {e}"

    if not geom.is_valid or geom.is_empty:
        return False, "The drawn shape is invalid or empty. Try redrawing it."

    maha_box = box(*MAHARASHTRA_BBOX)
    if not geom.intersects(maha_box):
        return False, "Selected area is outside Maharashtra. Please draw within the state."

    area = approx_area_km2(geom)
    if area > MAX_AREA_KM2:
        return False, (
            f"Selected area (~{area:,.0f} km²) exceeds the {MAX_AREA_KM2:,} km² "
            "limit for a single query. Draw a smaller region."
        )

    return True, f"Valid selection (~{area:,.1f} km²)."
