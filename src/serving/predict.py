"""Prediction logic: image preprocessing and inference."""
from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.data.transforms import get_eval_transforms

# Food-101 class names (populated at startup)
CLASS_NAMES: list[str] = []


def load_class_names(path: str = "class_names.txt") -> list[str]:
    """Load class names from file."""
    global CLASS_NAMES
    with open(path) as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
    return CLASS_NAMES


class Predictor:
    """ONNX-based image classifier."""

    def __init__(self, model_path: str, class_names_path: str = "class_names.txt") -> None:
        self.session = ort.InferenceSession(model_path)
        self.transform = get_eval_transforms(image_size=224)
        load_class_names(class_names_path)

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """Classify a single image. Returns predictions and latency."""
        t0 = time.perf_counter()

        tensor = self.transform(image.convert("RGB")).unsqueeze(0).numpy()
        logits = self.session.run(None, {"image": tensor})[0][0]

        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        top_indices = probs.argsort()[::-1][:top_k]
        predictions = [
            {"class": CLASS_NAMES[i], "confidence": round(float(probs[i]), 4)}
            for i in top_indices
        ]

        latency_ms = (time.perf_counter() - t0) * 1000
        return {"predictions": predictions, "latency_ms": round(latency_ms, 1)}
