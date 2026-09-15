import pytest
from utils.geo import validate_polygon

MUMBAI_SMALL_SQUARE = {
    "type": "Polygon",
    "coordinates": [[
        [72.80, 19.00], [72.90, 19.00], [72.90, 19.10], [72.80, 19.10], [72.80, 19.00]
    ]],
}

OUTSIDE_MAHARASHTRA = {
    "type": "Polygon",
    "coordinates": [[
        [77.0, 28.0], [77.1, 28.0], [77.1, 28.1], [77.0, 28.1], [77.0, 28.0]  # Delhi
    ]],
}

HUGE_AREA = {
    "type": "Polygon",
    "coordinates": [[
        [72.7, 16.0], [80.8, 16.0], [80.8, 22.0], [72.7, 22.0], [72.7, 16.0]
    ]],
}


def test_valid_polygon_within_maharashtra():
    ok, msg = validate_polygon(MUMBAI_SMALL_SQUARE)
    assert ok is True
    assert "Valid" in msg


def test_polygon_outside_maharashtra_rejected():
    ok, msg = validate_polygon(OUTSIDE_MAHARASHTRA)
    assert ok is False
    assert "outside" in msg.lower()


def test_oversized_polygon_rejected():
    ok, msg = validate_polygon(HUGE_AREA)
    assert ok is False
    assert "exceeds" in msg.lower()


def test_malformed_geojson_rejected():
    ok, msg = validate_polygon({"type": "Polygon", "coordinates": []})
    assert ok is False
