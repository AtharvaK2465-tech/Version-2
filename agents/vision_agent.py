"""
Vision Agent
------------
Classifies the clipped imagery into a stress category. Ships with a
transparent heuristic over the NDVI/NDMI indices so the pipeline is fully
runnable and explainable out of the box; swap in a trained PyTorch model
(see `_model_predict`) once you have labeled training data.
"""
import numpy as np

from agents.state import PipelineState

LABELS = ["healthy", "water_stress", "nutrient_deficiency", "disease"]


def _heuristic_classify(ndvi: np.ndarray, ndmi: np.ndarray) -> tuple[str, float]:
    """A simple, explainable stand-in classifier based on mean index values
    and how much of the scene falls below a stress threshold. Replace with
    `_model_predict` once a trained CNN is available."""
    mean_ndvi = float(np.mean(ndvi))
    stressed_fraction = float(np.mean(ndvi < 0.3))
    mean_ndmi = float(np.mean(ndmi))

    if stressed_fraction < 0.05:
        return "healthy", 0.9 - stressed_fraction
    if mean_ndmi < 0.1:
        return "water_stress", min(0.6 + stressed_fraction, 0.95)
    if mean_ndvi < 0.2:
        return "disease", min(0.55 + stressed_fraction, 0.9)
    return "nutrient_deficiency", min(0.5 + stressed_fraction, 0.85)


def _model_predict(raster: np.ndarray) -> tuple[str, float]:
    """
    LIVE MODE placeholder — plug in your trained PyTorch/TensorFlow model.

        model = load_model("models/crop_stress_cnn.pt")
        tensor = preprocess(raster)  # normalize, resize, to CHW tensor
        with torch.no_grad():
            logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)
        label = LABELS[probs.argmax()]
        confidence = float(probs.max())
        return label, confidence
    """
    raise NotImplementedError(
        "TODO: load a trained model checkpoint and run inference here."
    )


def vision_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state

    label, confidence = _heuristic_classify(state["ndvi"], state["ndmi"])

    return {
        **state,
        "vision_classification": label,
        "vision_confidence": round(confidence, 3),
    }
