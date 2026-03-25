"""FastAPI application for food image classification."""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.serving.predict import Predictor

predictor: Predictor | None = None

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    model_path = os.getenv("MODEL_PATH", "model.onnx")
    class_names_path = os.getenv("CLASS_NAMES_PATH", "class_names.txt")
    if os.path.exists(model_path) and os.path.exists(class_names_path):
        predictor = Predictor(model_path, class_names_path)
    yield


app = FastAPI(title="Food Vision API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "efficientnet_b2",
        "version": "1.0.0",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(422, f"Invalid file type: {file.content_type}. Use JPEG/PNG/WebP.")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(422, f"File too large. Max {MAX_SIZE_MB}MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(422, "Could not decode image.")

    if predictor is None:
        raise HTTPException(503, "Model not loaded.")

    return predictor.predict(image, top_k=5)


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 16:
        raise HTTPException(422, "Max 16 images per batch.")
    if predictor is None:
        raise HTTPException(503, "Model not loaded.")

    results = []
    for f in files:
        if f.content_type not in ALLOWED_TYPES:
            results.append({"error": f"Invalid file type: {f.content_type}"})
            continue
        contents = await f.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results.append(predictor.predict(image, top_k=5))
    return {"results": results}
