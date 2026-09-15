import numpy as np
from agents.geo_agent import _compute_indices
from agents.vision_agent import _heuristic_classify, LABELS


def test_compute_indices_shapes():
    raster = np.random.uniform(0, 0.5, (64, 64, 4)).astype(np.float32)
    indices = _compute_indices(raster)
    assert set(indices.keys()) == {"ndvi", "ndmi", "ndre"}
    for arr in indices.values():
        assert arr.shape == (64, 64)
        assert np.all(np.isfinite(arr))


def test_healthy_classification_high_ndvi():
    ndvi = np.full((32, 32), 0.7)
    ndmi = np.full((32, 32), 0.4)
    label, confidence = _heuristic_classify(ndvi, ndmi)
    assert label == "healthy"
    assert label in LABELS
    assert 0 <= confidence <= 1


def test_water_stress_classification_low_ndmi():
    ndvi = np.full((32, 32), 0.15)
    ndmi = np.full((32, 32), -0.2)
    label, confidence = _heuristic_classify(ndvi, ndmi)
    assert label == "water_stress"
    assert label in LABELS
