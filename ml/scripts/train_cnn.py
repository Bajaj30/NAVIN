#!/usr/bin/env python3
"""
NAVIN ML — 1D-CNN Activity Classifier (PyTorch → ONNX)
=======================================================
Input:  raw window (20 samples @ 10 Hz × [accel x/y/z, rotationRate alpha/beta/gamma])
Model:  posture-invariant features → z-score → Conv1D/BN/ReLU/Dropout ×2 → GAP → Linear
Output: STAIRS(0), STATIONARY(1), WALKING(2)

Steps:
1. Leave-one-session-out evaluation, 3 seeds → honest weighted F1 (mean ± std)
2. Random window split, for comparison only — shows how much leakage inflates scores
3. Train the final model on every session
4. Export to ONNX (preprocessing baked in), verify it matches PyTorch, time it
5. Write the model card and copy ONNX + card into backend/app/ml/ for serving

Usage: python3 ml/scripts/train_cnn.py
"""

import json
import os
import shutil
import time
import warnings
from datetime import date

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from navin_ml import (BACKEND_MODEL_DIR, CLASSES, FEATURE_NAMES, MODEL_DIR, PROCESSED_DIR, RATE_HZ,
                      RESULTS_DIR, WINDOW_SIZE, ActivityNetProbs, fit_cnn,
                      leave_one_session_out, load_processed, predict_proba, summarize)

warnings.filterwarnings('ignore')

EPOCHS = 40
SEEDS = [0, 1, 2]
ONNX_NAME = 'activity_cnn.onnx'


def loso(X, y, groups, seed):
    fp = lambda Xtr, ytr, Xte: predict_proba(fit_cnn(Xtr, ytr, EPOCHS, seed)[0], Xte).argmax(1)
    return leave_one_session_out(X, y, groups, fp)


def export_onnx(model, path):
    wrapper = ActivityNetProbs(model).eval()
    dummy = torch.zeros(1, WINDOW_SIZE, 6)
    torch.onnx.export(
        wrapper, dummy, path,
        input_names=['window'], output_names=['probabilities'],
        dynamic_axes={'window': {0: 'batch'}, 'probabilities': {0: 'batch'}},
        opset_version=17, dynamo=False,
    )


def check_onnx(path, model, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    onnx_probs = sess.run(None, {'window': X})[0]
    torch_probs = predict_proba(model, X)
    single = X[:1]
    for _ in range(20):                      # warm-up
        sess.run(None, {'window': single})
    times = []
    for i in range(500):
        t0 = time.perf_counter()
        sess.run(None, {'window': X[i % len(X):i % len(X) + 1]})
        times.append((time.perf_counter() - t0) * 1000)
    return {
        'max_abs_prob_diff': float(np.abs(onnx_probs - torch_probs).max()),
        'argmax_agreement': float((onnx_probs.argmax(1) == torch_probs.argmax(1)).mean()),
        'latency_ms_p50': round(float(np.percentile(times, 50)), 3),
        'latency_ms_p95': round(float(np.percentile(times, 95)), 3),
    }


def save_plots(cm, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap='Blues')
    ax.set_title('1D-CNN — leave-one-session-out (seed 0)')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history)
    ax.set_title('Final model — training loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Weighted cross-entropy')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'training_curve.png'), dpi=150); plt.close(fig)


def main():
    print("=" * 60)
    print("NAVIN ML — 1D-CNN Training (PyTorch → ONNX)")
    print("=" * 60)
    X, y, groups = load_processed()
    with open(os.path.join(PROCESSED_DIR, 'metadata.json')) as f:
        meta = json.load(f)
    print(f"  Windows: {X.shape}, sessions: {len(meta['sessions'])}")

    # 1. Honest evaluation
    per_seed = []
    for seed in SEEDS:
        preds = loso(X, y, groups, seed)
        per_seed.append(summarize(y, preds))
        print(f"  LOSO seed {seed}: weighted F1 {per_seed[-1]['weighted_f1']:.4f}")
        if seed == SEEDS[0]:
            print(classification_report(y, preds, target_names=CLASSES, zero_division=0))
    wf1 = np.array([m['weighted_f1'] for m in per_seed])
    mf1 = np.array([m['macro_f1'] for m in per_seed])

    # 2. Leaky random split — reported only to show the inflation
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    random_split = summarize(yte, predict_proba(fit_cnn(Xtr, ytr, EPOCHS, 0)[0], Xte).argmax(1))
    print(f"  Random window split (leaky): weighted F1 {random_split['weighted_f1']:.4f}")

    # 3. Final model
    model, history = fit_cnn(X, y, EPOCHS, seed=0)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'cnn_activity.pth'))
    params = sum(p.numel() for p in model.parameters())

    # 4. ONNX export + parity
    onnx_path = os.path.join(MODEL_DIR, ONNX_NAME)
    export_onnx(model, onnx_path)
    onnx_check = check_onnx(onnx_path, model, X)
    print(f"  ONNX parity: max |Δp| = {onnx_check['max_abs_prob_diff']:.2e}, "
          f"argmax agreement {onnx_check['argmax_agreement']:.1%}, "
          f"p50 {onnx_check['latency_ms_p50']} ms / p95 {onnx_check['latency_ms_p95']} ms")
    assert onnx_check['argmax_agreement'] == 1.0 and onnx_check['max_abs_prob_diff'] < 1e-4, "ONNX mismatch"

    # 5. Metrics, model card, plots
    baseline_path = os.path.join(RESULTS_DIR, 'baseline_metrics.json')
    baseline = json.load(open(baseline_path)) if os.path.exists(baseline_path) else None
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cnn_metrics = {
        'model': '1D-CNN (2 conv blocks + GAP)', 'framework': 'PyTorch → ONNX',
        'evaluation': 'leave-one-session-out', 'seeds': SEEDS, 'epochs': EPOCHS,
        'weighted_f1_mean': round(float(wf1.mean()), 4), 'weighted_f1_std': round(float(wf1.std()), 4),
        'macro_f1_mean': round(float(mf1.mean()), 4),
        'per_seed': per_seed,
        'random_split_weighted_f1': random_split['weighted_f1'],
        'total_params': params, 'final_train_loss': history, 'onnx': onnx_check,
    }
    with open(os.path.join(RESULTS_DIR, 'cnn_metrics.json'), 'w') as f:
        json.dump(cnn_metrics, f, indent=2)

    card = {
        'name': 'NAVIN activity classifier',
        'version': date.today().isoformat(),
        'architecture': 'Posture-invariant features → Conv1D(32)/BN/ReLU/Dropout → Conv1D(64)/BN/ReLU/Dropout → GlobalAvgPool → Linear',
        'framework': 'PyTorch → ONNX Runtime',
        'params': params,
        'classes': CLASSES,
        'input': {
            'window_samples': WINDOW_SIZE, 'rate_hz': RATE_HZ,
            'sample_layout': ['accel_x', 'accel_y', 'accel_z', 'rotation_alpha', 'rotation_beta', 'rotation_gamma'],
            'units': 'accelerationIncludingGravity m/s², rotationRate °/s',
        },
        'model_channels': FEATURE_NAMES,
        'data': {k: meta[k] for k in ('sessions', 'unique_minutes', 'windows', 'class_windows', 'duplicate_rows_dropped')},
        'evaluation': {
            'method': 'leave-one-session-out', 'seeds': len(SEEDS),
            'cnn': {
                'weighted_f1_mean': cnn_metrics['weighted_f1_mean'], 'weighted_f1_std': cnn_metrics['weighted_f1_std'],
                'macro_f1_mean': cnn_metrics['macro_f1_mean'],
                'per_class_f1': per_seed[0]['per_class_f1'], 'confusion_matrix': per_seed[0]['confusion_matrix'],
            },
            'random_forest': None if baseline is None else {
                'weighted_f1': baseline['weighted_f1'], 'macro_f1': baseline['macro_f1'],
                'per_class_f1': baseline['per_class_f1'],
            },
            'random_split_weighted_f1': random_split['weighted_f1'],
        },
        'onnx': onnx_check,
    }
    with open(os.path.join(MODEL_DIR, 'model_card.json'), 'w') as f:
        json.dump(card, f, indent=2)
    save_plots(per_seed[0]['confusion_matrix'], history)

    os.makedirs(BACKEND_MODEL_DIR, exist_ok=True)
    shutil.copy(onnx_path, os.path.join(BACKEND_MODEL_DIR, ONNX_NAME))
    shutil.copy(os.path.join(MODEL_DIR, 'model_card.json'), os.path.join(BACKEND_MODEL_DIR, 'model_card.json'))

    print("\n" + "=" * 60)
    print(f"CNN weighted F1 (leave-one-session-out): {wf1.mean():.4f} ± {wf1.std():.4f}")
    if baseline:
        print(f"RF  weighted F1 (leave-one-session-out): {baseline['weighted_f1']:.4f}")
    print(f"Random split (leaky, for reference):     {random_split['weighted_f1']:.4f}")
    print(f"Exported → {onnx_path}\n         → {BACKEND_MODEL_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
