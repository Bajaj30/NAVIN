#!/usr/bin/env python3
"""
NAVIN ML Preprocessing Pipeline
================================
1. Load all CSVs from ml/data/raw/
2. Clean: drop rows with NaN
3. Window: 2-second sliding windows (20 samples), 50% overlap
4. Split: stratified random 70/15/15
5. Normalize: z-score, fit on train only
6. Save: .npy arrays + scaler.pkl
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

# ============================================================================
# CONFIG
# ============================================================================
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

WINDOW_SIZE = 20        # 2 seconds at 10 Hz
STRIDE = 10             # 50% overlap
SENSOR_COLS = [
    'accel_x', 'accel_y', 'accel_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'mag_x', 'mag_y', 'mag_z'
]
LABEL_COL = 'label'
LABEL_MAP = {'STAIRS': 0, 'STATIONARY': 1, 'WALKING': 2}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def load_all_csvs(raw_dir):
    csv_files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    sessions = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        sessions.append((os.path.basename(f), df))
        print(f"  Loaded {os.path.basename(f)}: {len(df)} rows")

    return sessions


def clean_session(df):
    for col in SENSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    before = len(df)
    df = df.dropna(subset=SENSOR_COLS)
    dropped = before - len(df)
    if dropped > 0:
        print(f"    Dropped {dropped} rows")
    return df


def create_windows(df, window_size, stride):
    sensor_data = df[SENSOR_COLS].values
    labels = df[LABEL_COL].values

    X_windows = []
    y_windows = []

    for start in range(0, len(sensor_data) - window_size + 1, stride):
        end = start + window_size
        window = sensor_data[start:end]
        window_labels = labels[start:end]

        label_counts = Counter(window_labels)
        majority_label = label_counts.most_common(1)[0][0]
        majority_pct = label_counts.most_common(1)[0][1] / window_size

        if majority_pct < 0.6:
            continue

        X_windows.append(window)
        y_windows.append(LABEL_MAP[majority_label])

    return np.array(X_windows), np.array(y_windows)


def normalize_data(X_train, X_val, X_test):
    n_train, w, c = X_train.shape

    X_train_flat = X_train.reshape(-1, c)
    X_val_flat = X_val.reshape(-1, c)
    X_test_flat = X_test.reshape(-1, c)

    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    X_train_norm = scaler.transform(X_train_flat).reshape(n_train, w, c)
    X_val_norm = scaler.transform(X_val_flat).reshape(X_val.shape[0], w, c)
    X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape[0], w, c)

    return X_train_norm, X_val_norm, X_test_norm, scaler


def main():
    print("=" * 60)
    print("NAVIN ML — Preprocessing Pipeline")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    print("\n[1/5] Loading raw CSVs...")
    sessions = load_all_csvs(RAW_DIR)

    # Clean + Window
    print("\n[2/5] Cleaning data...")
    print(f"\n[3/5] Creating sliding windows (size={WINDOW_SIZE}, stride={STRIDE}, overlap=50%)...")

    all_X = []
    all_y = []

    for filename, df in sessions:
        df = clean_session(df)
        X, y = create_windows(df, WINDOW_SIZE, STRIDE)
        all_X.append(X)
        all_y.append(y)
        print(f"  {filename}: {len(X)} windows")

    X_all = np.concatenate(all_X)
    y_all = np.concatenate(all_y)
    print(f"\n  Total: {len(X_all)} windows")

    # Stratified split
    print("\n[4/5] Stratified random split (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED, stratify=y_temp
    )

    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    for name, y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        dist = {inv_label_map[k]: v for k, v in sorted(Counter(y).items())}
        print(f"  {name}: {len(y)} windows — {dist}")

    # Normalize
    print("\n[5/5] Normalizing (z-score, fit on train only)...")
    X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)
    print(f"  Mean: {np.round(scaler.mean_, 2)}")
    print(f"  Std:  {np.round(scaler.scale_, 2)}")

    # Save
    print(f"\nSaving...")
    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train.astype(np.int64))
    np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val.astype(np.int64))
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test.astype(np.int64))

    with open(os.path.join(OUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT_DIR, 'class_names.json'), 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)

    metadata = {
        'window_size': WINDOW_SIZE, 'stride': STRIDE,
        'num_channels': len(SENSOR_COLS), 'label_map': LABEL_MAP,
        'split_method': 'stratified_random', 'random_seed': RANDOM_SEED,
        'splits': {
            'train': {'windows': int(X_train.shape[0])},
            'val': {'windows': int(X_val.shape[0])},
            'test': {'windows': int(X_test.shape[0])},
        },
    }
    with open(os.path.join(OUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()
