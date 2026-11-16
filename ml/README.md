# NAVIN activity classifier — model card

Classifies a 2-second window of phone motion as **WALKING**, **STAIRS** or **STATIONARY**. The live app uses it to detect parking (sustained STATIONARY) and to show what the user is doing.

## Data

- Collected with [`collector/index.html`](collector/index.html) (also served at `/collector.html`): 10 Hz, labels tapped live.
- 9 sessions in a two-floor building (corridors, one staircase, rooms), phone in hand or on a table.
- 9,480 raw rows → **904 duplicate rows removed**: `session_004` was downloaded without clearing `session_003`, so it began with an exact copy. That leaves **14.3 unique minutes**.
- Recordings are split wherever sampling paused for more than 500 ms. 2 s windows (20 samples), 50% overlap, ≥ 80% label purity → **835 windows** (Stairs 242 · Stationary 204 · Walking 389).
- The `mag_x/y/z` columns are DeviceOrientation *angles* (heading/tilt), not magnetometer µT. They encode posture and walking direction rather than activity, so they are **not used**.

## Model

```
raw window (20 × [accel x/y/z, rotationRate α/β/γ])
  → posture-invariant channels: |a|, vertical accel, horizontal accel, |turn rate|, tilt rate
     (gravity = window-mean acceleration; components taken along / across it)
  → z-score (training statistics)
  → Conv1D(32,k3) → BN → ReLU → Dropout(0.2)
  → Conv1D(64,k3) → BN → ReLU → Dropout(0.2)
  → GlobalAvgPool → Linear(3) → softmax          7,107 parameters
```

The whole chain, including features and normalisation, is exported as **one ONNX graph** ([`models/activity_cnn.onnx`](models/activity_cnn.onnx)). The server feeds it raw windows, so preprocessing cannot drift between training and serving. The export matches PyTorch to within 1e-6 and runs in about 0.03 ms per window on a laptop CPU.

**Why posture-invariant inputs?** The first model (raw 9 channels) learned how the phone was held during collection. Standing still with the phone held in front of you, the way you look at the screen, it predicted WALKING 100% of the time. The gravity-aligned channels give the same signals in a hand, a pocket or held in front, and are also immune to iOS/Android axis sign differences.

## Evaluation

**Leave-one-session-out:** each recording is predicted by a model trained on the other eight.

| | Weighted F1 | Stairs | Stationary | Walking |
|---|---|---|---|---|
| **1D-CNN** (3 seeds) | **0.941 ± 0.005** | 0.92 | 0.98 | 0.95 |
| 1D-CNN + 3-window smoothing *(as served)* | 0.960 ± 0.004 | 0.95 | 0.98 | 0.97 |
| Random Forest (27 handcrafted features: mean/std/min/max/RMS per channel, zero-crossings, FFT dominant frequency) | 0.801 | 0.67 | 0.98 | 0.79 |
| 1D-CNN, random window split *(leaky, for reference only)* | 0.972 | | | |

The random split scores higher because overlapping windows from the same recording land on both sides. That gap is the leakage. The earlier pipeline reported 0.986 for both models this way.

**Smoothing.** Live, the server averages each session's last 3 window probabilities before answering, so a single walking ↔ stairs flip is outvoted by its neighbours. `train_cnn.py` applies the same causal average to the out-of-session predictions. Windows that straddle an activity change are not in the evaluation set (label purity < 80%), so the live gain is smaller right at transitions, where smoothing also adds about a second of lag.

The confusion matrix and training curve are in [`results/`](results/); full numbers are in `results/*.json` and `models/model_card.json`.

## Limitations

- One person, one building, one phone, mostly hand-held. Expect lower accuracy for other gaits or pockets until more varied data is collected.
- Stairs vs walking is the hard pair (most of the remaining errors).
- 14 minutes of data. The collector page plus Admin → ML Model uploads make it easy to add more.

## Reproduce

```bash
python3 ml/scripts/preprocess.py
python3 ml/scripts/train_baseline.py
python3 ml/scripts/train_cnn.py          # also copies the ONNX + card into backend/app/ml/
python3 ml/scripts/inference.py ml/data/raw/session_007_mixed_full_hand.csv   # timeline for one CSV
```
