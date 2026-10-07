#!/usr/bin/env python3
"""
NAVIN ML Preprocessing Pipeline
================================
1. Load all CSVs from ml/data/raw/
2. Clean: drop NaN rows + rows duplicated across files (session_004 contains session_003)
3. Segment: split each recording at sampling gaps > 500 ms
4. Window: 2-second sliding windows (20 samples @ 10 Hz), 50% overlap, ≥80% label purity
5. Save: raw windows + labels + session group ids (no split here — evaluation is
   leave-one-session-out, and normalisation lives inside the model)

Usage: python3 ml/scripts/preprocess.py
"""

import json
import os
from collections import Counter

import numpy as np

from navin_ml import (CLASSES, FEATURE_NAMES, LABEL_PURITY, MAX_GAP_MS, PROCESSED_DIR, RATE_HZ,
                      RAW_COLS, STRIDE, WINDOW_SIZE, build_dataset)


def main():
    print("=" * 60)
    print("NAVIN ML — Preprocessing Pipeline")
    print("=" * 60)

    X, y, groups, sessions, stats = build_dataset()
    print(f"\n  Raw rows:               {stats['raw_rows']}")
    print(f"  Duplicate rows dropped: {stats['duplicate_rows_dropped']}")
    print(f"  Continuous segments:    {stats['segments']}")
    print(f"  Windows:                {len(X)}  shape={X.shape}")

    print("\n  Windows per session:")
    for g, name in enumerate(sessions):
        counts = Counter(y[groups == g].tolist())
        print(f"    {name:<40} " + ", ".join(f"{CLASSES[k]} {v}" for k, v in sorted(counts.items())))

    class_counts = {c: int(n) for c, n in zip(CLASSES, np.bincount(y, minlength=len(CLASSES)))}
    print(f"\n  Class balance: {class_counts}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, 'windows_raw.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'labels.npy'), y)
    np.save(os.path.join(PROCESSED_DIR, 'groups.npy'), groups)

    unique_rows = stats['raw_rows'] - stats['duplicate_rows_dropped']
    metadata = {
        'window_size': WINDOW_SIZE, 'stride': STRIDE, 'sampling_rate_hz': RATE_HZ,
        'label_purity': LABEL_PURITY, 'max_gap_ms': MAX_GAP_MS,
        'raw_channels': RAW_COLS, 'model_channels': FEATURE_NAMES, 'classes': CLASSES,
        'sessions': sessions,
        'raw_rows': stats['raw_rows'],
        'duplicate_rows_dropped': stats['duplicate_rows_dropped'],
        'unique_minutes': round(unique_rows / RATE_HZ / 60, 1),
        'segments': stats['segments'],
        'windows': int(len(X)),
        'class_windows': class_counts,
        'split_method': 'leave-one-session-out (see train_*.py)',
    }
    with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved → {PROCESSED_DIR}")


if __name__ == '__main__':
    main()
