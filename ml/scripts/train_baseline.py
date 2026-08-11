#!/usr/bin/env python3
"""
NAVIN ML — Baseline Model (Random Forest)
==========================================
PURPOSE: Establish a benchmark F1 score using a simple model.
         The CNN must beat this number, or it's not worth the complexity.

HOW IT WORKS (the full story):

1. Load the preprocessed windows: shape (N, 20, 9)
   - N windows, each 20 samples long, each sample has 9 sensor channels

2. FEATURE EXTRACTION (this is what makes it a "baseline")
   - A Random Forest can't understand a 2D array (20×9).
   - It needs a flat vector of numbers — one row per window.
   - So WE manually compute statistics that summarize each window:
     - mean: average value (is the phone mostly tilted? is accel high?)
     - std:  how much variation (walking has high std, stationary has low)
     - min/max: extreme values (stairs have bigger peaks)
     - rms: root mean square — captures signal energy
   - We compute these 5 stats for each of the 9 channels = 45 features per window
   - The CNN skips this step entirely — it learns its own features

3. Train a Random Forest classifier on these 45 features
   - Random Forest = many decision trees, each trained on a random subset
   - Each tree votes, majority wins
   - Why RF? It's fast, needs no tuning, handles small datasets well

4. Evaluate on test set → get F1 score = the number to beat
"""

import os
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_features(X):
    """
    Convert raw windows into a flat feature vector.

    Input:  X of shape (num_windows, window_size, num_channels) = (N, 20, 9)
    Output: features of shape (num_windows, num_channels * 5) = (N, 45)

    For each channel in each window, we compute:
      1. mean  — average value over the 20 samples
      2. std   — standard deviation (how spread out the values are)
      3. min   — minimum value in the window
      4. max   — maximum value in the window
      5. rms   — root mean square = sqrt(mean(x²)) — captures signal energy

    WHY THESE?
    - mean: captures phone orientation (gravity shows up as a constant offset)
    - std: KEY discriminator — walking has high std (oscillation), stationary has ~0
    - min/max: stairs have larger peaks than walking (higher vertical displacement)
    - rms: combines magnitude and variation — useful for distinguishing activity intensity

    INTERVIEW NOTE: These are called "handcrafted features" because YOU chose them.
    A CNN would learn equivalent (or better) features automatically from the raw data.
    """
    num_windows, window_size, num_channels = X.shape
    features = []

    for ch in range(num_channels):
        channel_data = X[:, :, ch]  # shape: (num_windows, window_size)

        features.append(np.mean(channel_data, axis=1))                          # mean
        features.append(np.std(channel_data, axis=1))                           # std
        features.append(np.min(channel_data, axis=1))                           # min
        features.append(np.max(channel_data, axis=1))                           # max
        features.append(np.sqrt(np.mean(channel_data ** 2, axis=1)))            # rms

    # Stack: each feature is (N,), we have 45 of them → (N, 45)
    return np.column_stack(features)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("NAVIN ML — Baseline Model (Random Forest)")
    print("=" * 60)

    # Load preprocessed data
    print("\n[1/4] Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Extract features
    print("\n[2/4] Extracting handcrafted features (5 stats × 9 channels = 45 features)...")
    F_train = extract_features(X_train)
    F_val = extract_features(X_val)
    F_test = extract_features(X_test)
    print(f"  Feature vectors: train={F_train.shape}, val={F_val.shape}, test={F_test.shape}")

    # Train Random Forest
    print("\n[3/4] Training Random Forest...")
    # n_estimators=100: use 100 trees (more trees = more stable, diminishing returns past ~100)
    # random_state=42: fixed seed so results are reproducible
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    rf.fit(F_train, y_train)

    # Evaluate
    print("\n[4/4] Evaluating...")

    # Load class names
    with open(os.path.join(DATA_DIR, 'class_names.json')) as f:
        label_map = json.load(f)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    # Validation set
    y_val_pred = rf.predict(F_val)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Test set
    y_test_pred = rf.predict(F_test)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"\n  Validation F1 (weighted): {val_f1:.4f}")
    print(f"  Test F1 (weighted):       {test_f1:.4f}")

    print(f"\n--- Test Set Classification Report ---")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    print(f"--- Test Set Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_test_pred)
    # Print with labels
    print(f"{'':>12} {'STAIRS':>10} {'STATION':>10} {'WALKING':>10}  ← predicted")
    for i, name in enumerate(class_names):
        row = '  '.join(f'{cm[i,j]:>8}' for j in range(len(class_names)))
        print(f"  {name:>10} {row}")
    print(f"  ↑ actual")

    # Save model
    model_path = os.path.join(MODEL_DIR, 'baseline_rf.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n  Model saved: {model_path}")

    # Save metrics
    metrics = {
        'model': 'RandomForest',
        'n_estimators': 100,
        'num_features': int(F_train.shape[1]),
        'feature_type': 'handcrafted (mean, std, min, max, rms) × 9 channels',
        'val_f1_weighted': round(float(val_f1), 4),
        'test_f1_weighted': round(float(test_f1), 4),
        'test_classification_report': classification_report(
            y_test, y_test_pred, target_names=class_names, output_dict=True
        ),
        'test_confusion_matrix': cm.tolist(),
    }
    metrics_path = os.path.join(RESULTS_DIR, 'baseline_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"BASELINE F1: {test_f1:.4f}")
    print("This is the number your CNN needs to beat.")
    print("=" * 60)


if __name__ == '__main__':
    main()
