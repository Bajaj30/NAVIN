"""
NAVIN ML — shared code for preprocessing, training, evaluation and export.

Everything the deployed model needs at inference time lives inside `ActivityNet`
(posture-invariant feature transform → z-score normalisation → 1D-CNN), so the
exported ONNX graph takes RAW sensor windows. Training and serving cannot drift
apart because there is only one implementation of the preprocessing.
"""

import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# ============================================================================
# PATHS
# ============================================================================
ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ML_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(ML_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(ML_DIR, 'models')
RESULTS_DIR = os.path.join(ML_DIR, 'results')
BACKEND_MODEL_DIR = os.path.abspath(os.path.join(ML_DIR, '..', 'backend', 'app', 'ml'))

# ============================================================================
# DATA CONFIG
# ============================================================================
RATE_HZ = 10            # collector samples every 100 ms
WINDOW_SIZE = 20        # 2 seconds
STRIDE = 10             # 50% overlap
LABEL_PURITY = 0.8      # drop windows that straddle an activity change
MAX_GAP_MS = 500        # split a recording wherever sampling paused

# Raw inputs, in the order the browser reports them:
#   accelerationIncludingGravity x/y/z (m/s²), rotationRate alpha/beta/gamma (°/s)
# The mag_* columns are DeviceOrientation ANGLES (heading/tilt), not magnetometer
# readings — they encode posture and walking direction, not activity, so we drop them.
RAW_COLS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
FEATURE_NAMES = ['accel_magnitude', 'vertical_accel', 'horizontal_accel', 'turn_rate', 'tilt_rate']
CLASSES = ['STAIRS', 'STATIONARY', 'WALKING']
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}


# ============================================================================
# LOADING + WINDOWING
# ============================================================================
def load_sessions(raw_dir=RAW_DIR):
    """Load raw CSVs → ({session_name: [segment DataFrames]}, stats).

    - Rows whose timestamp already appeared in an earlier file are dropped
      (session_004 was downloaded without clearing session_003 first, so its
      first 904 rows are an exact copy).
    - Each file is split at sampling gaps > MAX_GAP_MS so no window spans a pause.
    """
    seen = set()
    sessions, stats = {}, {'raw_rows': 0, 'duplicate_rows_dropped': 0, 'segments': 0}
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.csv'))):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        stats['raw_rows'] += len(df)
        df = df.dropna(subset=RAW_COLS + ['label'])

        duplicate = df['timestamp'].isin(seen)
        seen.update(df['timestamp'])
        stats['duplicate_rows_dropped'] += int(duplicate.sum())
        df = df[~duplicate].reset_index(drop=True)
        if df.empty:
            continue

        breaks = np.flatnonzero(df['timestamp'].diff().fillna(0).to_numpy() > MAX_GAP_MS)
        bounds = [0, *breaks.tolist(), len(df)]
        segments = [df.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        sessions[os.path.splitext(os.path.basename(path))[0]] = segments
        stats['segments'] += len(segments)
    return sessions, stats


def make_windows(segment):
    """Slide a WINDOW_SIZE window (STRIDE step) over one continuous segment."""
    values = segment[RAW_COLS].to_numpy(np.float32)
    labels = segment['label'].to_numpy()
    X, y = [], []
    for start in range(0, len(values) - WINDOW_SIZE + 1, STRIDE):
        label, count = Counter(labels[start:start + WINDOW_SIZE]).most_common(1)[0]
        if count / WINDOW_SIZE >= LABEL_PURITY:
            X.append(values[start:start + WINDOW_SIZE])
            y.append(LABEL_MAP[label])
    return X, y


def build_dataset(raw_dir=RAW_DIR):
    """→ X (N, 20, 6) raw windows, y (N,), groups (N,) session index, session names, stats."""
    sessions, stats = load_sessions(raw_dir)
    X, y, groups = [], [], []
    for g, (name, segments) in enumerate(sessions.items()):
        for seg in segments:
            Xs, ys = make_windows(seg)
            X += Xs
            y += ys
            groups += [g] * len(ys)
    return (np.asarray(X, np.float32), np.asarray(y, np.int64),
            np.asarray(groups, np.int64), list(sessions), stats)


def load_processed():
    X = np.load(os.path.join(PROCESSED_DIR, 'windows_raw.npy'))
    y = np.load(os.path.join(PROCESSED_DIR, 'labels.npy'))
    groups = np.load(os.path.join(PROCESSED_DIR, 'groups.npy'))
    return X, y, groups


# ============================================================================
# MODEL
# ============================================================================
class InvariantFeatures(nn.Module):
    """(B, 20, 6) raw window → (B, 20, 5) channels that don't depend on how the phone is held.

    Gravity ≈ mean acceleration over the window. Dynamic acceleration and rotation
    are split into the component along gravity (vertical) and the remainder
    (horizontal), so hand-at-side, pocket and held-in-front postures produce the
    same signals. Magnitudes also make it immune to iOS/Android axis sign flips.
    """

    def forward(self, x):
        accel = x[:, :, 0:3]
        rot = x[:, :, [4, 5, 3]]  # rotationRate alpha/beta/gamma are about device z/x/y → (x, y, z)
        g = accel.mean(dim=1, keepdim=True)
        g_unit = g / (g.norm(dim=2, keepdim=True) + 1e-6)
        dyn = accel - g
        a_v = (dyn * g_unit).sum(dim=2)
        a_h = (dyn - a_v.unsqueeze(2) * g_unit).norm(dim=2)
        w_v = (rot * g_unit).sum(dim=2)
        w_h = (rot - w_v.unsqueeze(2) * g_unit).norm(dim=2)
        return torch.stack([accel.norm(dim=2), a_v, a_h, w_v.abs(), w_h], dim=2)


class ActivityCNN(nn.Module):
    """Conv1D → BN → ReLU → Dropout (×2) → GlobalAvgPool → Linear. Input (B, C, T)."""

    def __init__(self, in_channels=len(FEATURE_NAMES), num_classes=len(CLASSES), dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))   # (B, 32, T)
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))   # (B, 64, T)
        return self.fc(self.gap(x).squeeze(-1))                # (B, classes)


class ActivityNet(nn.Module):
    """Raw (B, 20, 6) window → class logits: features + normalisation + CNN in one graph."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        n = len(FEATURE_NAMES)
        self.features = InvariantFeatures()
        self.register_buffer('mean', torch.zeros(n) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.ones(n) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.cnn = ActivityCNN()

    def forward(self, raw):
        z = (self.features(raw) - self.mean) / self.std
        return self.cnn(z.transpose(1, 2))


class ActivityNetProbs(nn.Module):
    """Export wrapper: same as ActivityNet but returns softmax probabilities."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, raw):
        return torch.softmax(self.net(raw), dim=1)


def invariant_features(X_raw):
    with torch.no_grad():
        return InvariantFeatures()(torch.from_numpy(X_raw)).numpy()


def fit_cnn(X_raw, y, epochs=40, seed=0, batch_size=32, lr=1e-3):
    """Train ActivityNet on raw windows. Returns (model in eval mode, per-epoch train loss)."""
    torch.manual_seed(seed)
    X = torch.from_numpy(X_raw)
    Y = torch.from_numpy(y)

    feats = torch.from_numpy(invariant_features(X_raw))
    model = ActivityNet(feats.mean(dim=(0, 1)), feats.std(dim=(0, 1)) + 1e-6)

    # Inverse-frequency class weights — STATIONARY / STAIRS have fewer windows than WALKING
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    weights = torch.from_numpy(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    gen = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm needs > 1 sample
                continue
            optimizer.zero_grad()
            loss = criterion(model(X[idx]), Y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append(total / len(X))
    return model.eval(), history


def predict_proba(model, X_raw):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X_raw)), dim=1).numpy()


# ============================================================================
# BASELINE FEATURES (Random Forest)
# ============================================================================
def handcrafted_features(X_raw):
    """Per window: mean/std/min/max/RMS of each invariant channel (25)
    + zero-crossings of vertical accel + FFT dominant frequency of |a| (step cadence)."""
    Z = invariant_features(X_raw)                                   # (N, 20, 5)
    stats = [Z.mean(1), Z.std(1), Z.min(1), Z.max(1), np.sqrt((Z ** 2).mean(1))]

    a_v = Z[:, :, 1]
    zero_crossings = (np.diff(np.sign(a_v - a_v.mean(1, keepdims=True)), axis=1) != 0).sum(1)

    mag = Z[:, :, 0] - Z[:, :, 0].mean(1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(mag, axis=1))
    freqs = np.fft.rfftfreq(WINDOW_SIZE, d=1 / RATE_HZ)
    dominant_freq = freqs[1:][spectrum[:, 1:].argmax(1)]          # skip the DC bin

    return np.column_stack(stats + [zero_crossings, dominant_freq])


# ============================================================================
# EVALUATION
# ============================================================================
def leave_one_session_out(X, y, groups, fit_predict):
    """Hold out each recording session in turn → pooled out-of-session predictions.

    A random window split leaks: neighbouring windows share 50% of their samples
    and every session's posture/route appears on both sides. Grouping by session
    measures what we actually care about — a recording the model has never seen.
    """
    preds = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        preds[test] = fit_predict(X[~test], y[~test], X[test])
    return preds


def summarize(y_true, y_pred):
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    return {
        'weighted_f1': round(float(f1_score(y_true, y_pred, average='weighted')), 4),
        'macro_f1': round(float(f1_score(y_true, y_pred, average='macro')), 4),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'per_class_f1': {c: round(float(f), 4) for c, f in zip(CLASSES, per_class)},
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
    }
