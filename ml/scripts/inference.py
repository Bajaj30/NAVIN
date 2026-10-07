#!/usr/bin/env python3
"""
NAVIN ML — Inference on new CSV files (uses the exported ONNX model, same as production)
Usage: python3 ml/scripts/inference.py path/to/new_session.csv [more.csv ...]
"""

import os
import sys
from datetime import datetime

import numpy as np
import onnxruntime as ort
import pandas as pd

from navin_ml import CLASSES, MODEL_DIR, RAW_COLS, STRIDE, WINDOW_SIZE


def run_inference(csv_path, session):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"Loaded: {csv_path} ({len(df)} rows)")

    values = df[RAW_COLS].to_numpy(np.float32)
    timestamps = df['timestamp'].to_numpy()
    starts = list(range(0, len(values) - WINDOW_SIZE + 1, STRIDE))
    if not starts:
        print("  Not enough samples for one 2-second window.")
        return []

    windows = np.stack([values[s:s + WINDOW_SIZE] for s in starts])
    probs = session.run(None, {'window': windows})[0]
    preds = probs.argmax(1)

    # Merge consecutive windows with the same prediction into segments
    segments = []
    for i, s in enumerate(starts):
        label, end_ts = CLASSES[preds[i]], int(timestamps[s + WINDOW_SIZE - 1])
        if segments and segments[-1]['activity'] == label:
            segments[-1]['end_ms'] = end_ts
            segments[-1]['conf'].append(probs[i, preds[i]])
        else:
            segments.append({'activity': label, 'start_ms': int(timestamps[s]), 'end_ms': end_ts,
                             'conf': [probs[i, preds[i]]]})

    print(f"\n{'=' * 65}")
    print(f"{'Activity':<15} {'Duration':>10} {'Confidence':>12} {'Time Range':>25}")
    print(f"{'=' * 65}")
    totals = {c: 0.0 for c in CLASSES}
    for seg in segments:
        duration = (seg['end_ms'] - seg['start_ms']) / 1000
        totals[seg['activity']] += duration
        start = datetime.fromtimestamp(seg['start_ms'] / 1000).strftime('%H:%M:%S')
        end = datetime.fromtimestamp(seg['end_ms'] / 1000).strftime('%H:%M:%S')
        print(f"{seg['activity']:<15} {duration:>8.1f}s {np.mean(seg['conf']) * 100:>10.1f}% {start:>12} → {end}")

    total = sum(totals.values())
    print(f"\n{'=' * 65}\nSUMMARY:")
    for activity, t in totals.items():
        print(f"  {activity:<15} {t:>6.1f}s  ({(t / total * 100) if total else 0:>5.1f}%)")
    print(f"{'=' * 65}")
    return segments


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 ml/scripts/inference.py <csv_file> [csv_file2 ...]")
        sys.exit(1)
    onnx_session = ort.InferenceSession(os.path.join(MODEL_DIR, 'activity_cnn.onnx'),
                                        providers=['CPUExecutionProvider'])
    for path in sys.argv[1:]:
        run_inference(path, onnx_session)
        print()
