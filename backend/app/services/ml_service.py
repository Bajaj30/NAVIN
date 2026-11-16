"""Activity classifier serving — ONNX Runtime, model + model card loaded once at startup.

The ONNX graph (exported by ml/scripts/train_cnn.py) takes a RAW sensor window and
does the posture-invariant feature transform + normalisation itself, so there is no
preprocessing here that could drift from training.
"""
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml"
MODEL_PATH = MODEL_DIR / "activity_cnn.onnx"
CARD_PATH = MODEL_DIR / "model_card.json"

_session = None
_card: dict = {}
_latencies: deque = deque(maxlen=500)   # recent server-side inference times (ms)
_latest: dict = {}                      # session_id → latest prediction (in-memory, demo-scale)

# Temporal smoothing: a session's prediction is the mean of its last few windows' probabilities.
# Measured leave-one-session-out, this lifts weighted F1 from 0.941 to 0.960 (ml/scripts/train_cnn.py)
# because single-window walking ↔ stairs flips get outvoted by their neighbours.
SMOOTH_WINDOWS = 3
SMOOTH_MAX_AGE_S = 3.0                  # windows arrive ~1/s; anything older belongs to a previous episode
_history: dict[str, deque] = {}         # session_id → deque[(monotonic time, probabilities)]
_prediction_count = 0                   # served since boot (admin dashboard)


def load_model() -> None:
    """Load the model; on failure the API reports it unavailable and clients fall back to the heuristic."""
    global _session, _card
    try:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # tiny model — threads only add overhead on a shared CPU
        _session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
        _card = json.loads(CARD_PATH.read_text())
        logger.info(f"ML model loaded: {MODEL_PATH.name} v{_card.get('version')} ({_card.get('params')} params)")
    except Exception as e:  # noqa: BLE001 — any failure means "serve without ML"
        _session = None
        logger.error(f"ML model failed to load: {e}")


def is_loaded() -> bool:
    return _session is not None


def card() -> dict:
    return _card


def window_shape() -> tuple[int, int]:
    spec = _card.get("input", {})
    return spec.get("window_samples", 20), len(spec.get("sample_layout", [])) or 6


def _smoothed(session_id: str, probs: np.ndarray) -> tuple[np.ndarray, int]:
    """Mean of this session's recent window probabilities (including this one) → (probs, windows used)."""
    now = time.monotonic()
    history = _history.setdefault(session_id, deque(maxlen=SMOOTH_WINDOWS))
    while history and now - history[0][0] > SMOOTH_MAX_AGE_S:
        history.popleft()
    history.append((now, probs))
    return np.mean([p for _, p in history], axis=0), len(history)


def predict(window: np.ndarray, session_id: str | None = None) -> dict:
    """window: (samples, 6) raw accel x/y/z + rotationRate alpha/beta/gamma → class probabilities.

    With a session_id the result is smoothed over that session's last SMOOTH_WINDOWS windows;
    `raw_activity` is always the single-window answer.
    """
    global _prediction_count
    t0 = time.perf_counter()
    raw = _session.run(None, {"window": window[None].astype(np.float32)})[0][0]
    ms = (time.perf_counter() - t0) * 1000
    _latencies.append(ms)
    _prediction_count += 1

    classes = _card["classes"]
    probs, used = _smoothed(session_id, raw) if session_id else (raw, 1)
    idx = int(np.argmax(probs))
    return {
        "activity": classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
        "raw_activity": classes[int(np.argmax(raw))],
        "smoothed_over": used,
        "inference_ms": round(ms, 3),
    }


def prediction_count() -> int:
    return _prediction_count


def record(session_id: str, prediction: dict) -> None:
    """Remember the latest prediction per session (for the admin view); log activity changes."""
    previous = _latest.get(session_id)
    _latest[session_id] = {**prediction, "updated_at": datetime.utcnow().isoformat()}
    if previous is None or previous["activity"] != prediction["activity"]:
        before = previous["activity"] if previous else "—"
        logger.info(
            f"ML | session {session_id[:8]} | {before} → {prediction['activity']} "
            f"({prediction['confidence']:.2f})"
        )


def latest(session_id: str) -> dict | None:
    return _latest.get(session_id)


def latency_stats() -> dict:
    if not _latencies:
        return {"count": 0, "p50_ms": None, "p95_ms": None}
    arr = np.fromiter(_latencies, dtype=float)
    return {
        "count": len(arr),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
    }
