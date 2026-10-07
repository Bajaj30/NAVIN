#!/usr/bin/env python3
"""
NAVIN ML — Baseline Model (Random Forest)
==========================================
PURPOSE: Establish a benchmark F1 score using a simple model.
         The CNN must beat this number, or it's not worth the complexity.

HOW IT WORKS:
1. Load raw windows (N, 20, 6) — accel x/y/z + rotation rate alpha/beta/gamma.
2. Convert each window to 5 posture-invariant channels (|a|, vertical accel,
   horizontal accel, turn rate, tilt rate) — same transform the CNN uses.
3. FEATURE EXTRACTION (this is what makes it a "baseline"): a Random Forest
   needs one flat vector per window, so we summarise each channel by hand:
     mean, std, min, max, RMS            → 5 stats × 5 channels = 25
     zero-crossings of vertical accel    → step count proxy
     FFT dominant frequency of |a|       → step cadence (stairs are slower)
   The CNN skips this step and learns its own features from the raw signal.
4. Evaluate with LEAVE-ONE-SESSION-OUT: train on every recording but one, test
   on the held-out recording, repeat for each. A random window split would leak
   (overlapping windows + same posture on both sides) and inflate the score.

Usage: python3 ml/scripts/train_baseline.py
"""

import json
import os
import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from navin_ml import (CLASSES, MODEL_DIR, RESULTS_DIR, handcrafted_features, leave_one_session_out,
                      load_processed, summarize)

warnings.filterwarnings('ignore')


def make_rf():
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def main():
    print("=" * 60)
    print("NAVIN ML — Baseline Model (Random Forest)")
    print("=" * 60)

    X, y, groups = load_processed()
    F = handcrafted_features(X)
    print(f"  Windows: {len(X)}, features per window: {F.shape[1]}, sessions: {len(set(groups))}")

    print("\n  Leave-one-session-out evaluation...")
    preds = leave_one_session_out(F, y, groups, lambda Ftr, ytr, Fte: make_rf().fit(Ftr, ytr).predict(Fte))
    metrics = summarize(y, preds)
    print(classification_report(y, preds, target_names=CLASSES, zero_division=0))
    print(f"  Confusion matrix (rows = true {CLASSES}):")
    for name, row in zip(CLASSES, metrics['confusion_matrix']):
        print(f"    {name:>10} {row}")

    # Final model on all data (kept for comparison / offline analysis)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'baseline_rf.pkl'), 'wb') as f:
        pickle.dump(make_rf().fit(F, y), f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        'model': 'RandomForest',
        'n_estimators': 200,
        'num_features': int(F.shape[1]),
        'feature_type': 'mean, std, min, max, RMS of 5 posture-invariant channels + zero-crossings + FFT dominant frequency',
        'evaluation': 'leave-one-session-out',
        **metrics,
    }
    with open(os.path.join(RESULTS_DIR, 'baseline_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print(f"BASELINE weighted F1 (leave-one-session-out): {metrics['weighted_f1']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
