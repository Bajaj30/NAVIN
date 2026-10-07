"""ML router — live activity classification (1D-CNN via ONNX Runtime) + model card."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

from app.services import ml_service

router = APIRouter()


class PredictPayload(BaseModel):
    # 20 samples @ 10 Hz, each [accel_x, accel_y, accel_z, rotation_alpha, rotation_beta, rotation_gamma]
    samples: list[list[float]]
    session_id: str | None = None


@router.post("/predict")
def predict(payload: PredictPayload):
    if not ml_service.is_loaded():
        raise HTTPException(status_code=503, detail="ML model not loaded")

    window = np.asarray(payload.samples, dtype=np.float32)
    expected = ml_service.window_shape()
    if window.shape != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected[0]} samples × {expected[1]} values, got shape {list(window.shape)}",
        )
    if not np.isfinite(window).all():
        raise HTTPException(status_code=422, detail="Samples must be finite numbers")

    result = ml_service.predict(window)
    if payload.session_id:
        ml_service.record(payload.session_id, result)
    return result


@router.get("/model")
def model_info():
    """Model card (architecture, data, honest evaluation) + live serving stats for the admin panel."""
    return {
        "loaded": ml_service.is_loaded(),
        "card": ml_service.card(),
        "serving_latency": ml_service.latency_stats(),
    }
