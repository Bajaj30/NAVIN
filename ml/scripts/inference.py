#!/usr/bin/env python3
"""
NAVIN ML — Inference on new CSV files
Usage: python3 ml/scripts/inference.py path/to/new_session.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import pickle
from collections import Counter
from datetime import datetime

# Add parent for imports
SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

WINDOW_SIZE = 20
STRIDE = 10
SENSOR_COLS = [
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'mag_x', 'mag_y', 'mag_z'
]
CLASS_NAMES = {0: 'STAIRS', 1: 'STATIONARY', 2: 'WALKING'}


# Same model definition as train_cnn.py
class ActivityCNN(torch.nn.Module):
    def __init__(self, in_channels=9, num_classes=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.gap = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def run_inference(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"Loaded: {csv_path} ({len(df)} rows)")

    # Load scaler
    with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Load model
    model = ActivityCNN()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'cnn_activity.pth'), weights_only=True))
    model.eval()

    # Create windows
    sensor_data = df[SENSOR_COLS].values
    timestamps = df['timestamp'].values

    windows = []
    window_times = []

    for start in range(0, len(sensor_data) - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window = sensor_data[start:end]
        windows.append(window)
        window_times.append({
            'start_ts': int(timestamps[start]),
            'end_ts': int(timestamps[end - 1]),
            'start_idx': start,
            'end_idx': end - 1
        })

    X = np.array(windows)

    # Normalize using training scaler
    n, w, c = X.shape
    X_flat = X.reshape(-1, c)
    X_norm = scaler.transform(X_flat).reshape(n, w, c)

    # Transpose for PyTorch: (N, 20, 9) → (N, 9, 20)
    X_tensor = torch.FloatTensor(np.transpose(X_norm, (0, 2, 1)))

    # Predict
    with torch.no_grad():
        logits = model(X_tensor)
        preds = logits.argmax(1).numpy()
        probs = torch.softmax(logits, dim=1).numpy()

    # Build timeline: merge consecutive windows with same prediction
    segments = []
    current_label = CLASS_NAMES[preds[0]]
    current_start = window_times[0]['start_ts']
    current_conf = [probs[0][preds[0]]]

    for i in range(1, len(preds)):
        label = CLASS_NAMES[preds[i]]
        if label != current_label:
            segments.append({
                'activity': current_label,
                'start_ms': current_start,
                'end_ms': window_times[i - 1]['end_ts'],
                'duration_s': round((window_times[i - 1]['end_ts'] - current_start) / 1000, 1),
                'confidence': round(float(np.mean(current_conf)) * 100, 1)
            })
            current_label = label
            current_start = window_times[i]['start_ts']
            current_conf = [probs[i][preds[i]]]
        else:
            current_conf.append(probs[i][preds[i]])

    # Last segment
    segments.append({
        'activity': current_label,
        'start_ms': current_start,
        'end_ms': window_times[-1]['end_ts'],
        'duration_s': round((window_times[-1]['end_ts'] - current_start) / 1000, 1),
        'confidence': round(float(np.mean(current_conf)) * 100, 1)
    })

    # Print results
    print(f"\n{'='*65}")
    print(f"{'Activity':<15} {'Duration':>10} {'Confidence':>12} {'Time Range':>25}")
    print(f"{'='*65}")

    total_time = {name: 0.0 for name in CLASS_NAMES.values()}

    for seg in segments:
        start_str = datetime.fromtimestamp(seg['start_ms'] / 1000).strftime('%H:%M:%S')
        end_str = datetime.fromtimestamp(seg['end_ms'] / 1000).strftime('%H:%M:%S')
        print(f"{seg['activity']:<15} {seg['duration_s']:>8.1f}s {seg['confidence']:>10.1f}% {start_str:>12} → {end_str}")
        total_time[seg['activity']] += seg['duration_s']

    print(f"\n{'='*65}")
    print("SUMMARY:")
    total = sum(total_time.values())
    for activity, time in sorted(total_time.items()):
        pct = (time / total * 100) if total > 0 else 0
        print(f"  {activity:<15} {time:>6.1f}s  ({pct:>5.1f}%)")
    print(f"  {'TOTAL':<15} {total:>6.1f}s")
    print(f"{'='*65}")

    return segments


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 ml/scripts/inference.py <csv_file> [csv_file2 ...]")
        sys.exit(1)

    for csv_path in sys.argv[1:]:
        run_inference(csv_path)
        print()
