# 🧭 NAVIN — Indoor Parking Navigation PWA

NAVIN is a Progressive Web App that helps drivers navigate inside parking structures, where GPS doesn't reach, using only the sensors already in their phone. A QR sticker at the gate starts a session (no app install). A 1D-CNN classifies what you're doing (walking, taking the stairs, standing still) from the phone's motion sensors, detects when you've parked, and the occupancy grid updates for everyone.

**Live demo:** https://navin-black.vercel.app · **Admin:** https://navin-black.vercel.app/admin

## 🏗️ Architecture

```
┌──────────────────────────┐   /api/* (Vercel rewrite)   ┌──────────────────────────────┐
│  React PWA  (Vercel)     │ ──────────────────────────▶ │  FastAPI  (Render, free)     │
│                          │                             │                              │
│  • QR scanner / entry    │  20×6 sensor window / 1 s   │  • Sessions · occupancy · QR │
│  • Live sensors @ 60 Hz  │ ──────────────────────────▶ │  • /api/ml/predict           │
│  • 10 Hz ML windows      │ ◀────────────────────────── │    1D-CNN on ONNX Runtime    │
│  • Heuristic fallback    │   activity + probabilities  │  • Admin: logs, stats, model │
│  • Dead-reckoning map    │                             │  • SQLite (re-seeded on boot)│
│  • Admin + PSO simulator │                             └──────────────────────────────┘
└──────────────────────────┘
          ▲
          │ trained offline: ml/scripts  (PyTorch → ONNX, preprocessing baked into the graph)
```

## ✨ Features

- **QR Entry/Exit**: scan at the gate to start or end a session; zero install.
- **Live activity AI**: a 1D-CNN classifies WALKING / STAIRS / STATIONARY from 2-second motion windows. It's served by the backend, smoothed over the last 3 windows, falls back to an on-device heuristic, and detects parking after ~8 s of stillness.
- **Stairs → floor change**: a sustained stairs episode switches the map to the other floor (with undo). A browser has no barometer, so with more than two floors it asks which one.
- **Posture-invariant model**: works with the phone in a hand, in a pocket, or held in front of you (it uses gravity-aligned features, not raw axes).
- **Dead-reckoning map**: floor map drawn in metres. A peak-detecting step counter (gated by the CNN, so standing still doesn't move you) plus a wrap-safe heading (true compass where the phone exposes one, gyro-relative otherwise) move the dot 0.7 m per step. It drifts, as dead reckoning does: tap the map to correct it.
- **Occupancy grid**: free/taken bays, updating live.
- **Admin dashboard**: sessions with live CNN activity, server logs, QR codes, floor-plan upload, a **model card** (honest metrics, confusion matrix) and a **PSO positioning simulator**.
- **Data collector**: `/collector.html` records labelled sensor CSVs for training (download or upload to the server).
- **Cold-start aware**: the UI waits for the free-tier backend to wake (~30–60 s) instead of failing.

## 🤖 Activity classifier

Self-collected data: 9 recording sessions, 14.3 unique minutes at 10 Hz, 835 two-second windows.

| Model | Weighted F1, **leave-one-session-out** |
|---|---|
| **1D-CNN** (7.1k params), single window | **94.1% ± 0.5** (3 seeds) |
| 1D-CNN + 3-window smoothing (what the server returns live) | 96.0% ± 0.4 |
| Random Forest on handcrafted features | 80.1% |
| *1D-CNN on a random window split (leaky — not reported)* | *97.2%* |

Why leave-one-session-out: neighbouring windows overlap by 50%, so a random split puts near-copies of test windows in training. Holding out whole recordings measures what matters, which is a walk the model has never seen. Details: [ml/README.md](ml/README.md).

## 🚀 Quick Start

### Backend
```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest httpx
uvicorn app.main:app --reload --port 8000     # seeds the demo car park on first start
```
(`poetry install` also works.)

### Frontend
```bash
cd frontend
npm install
npm run dev                                    # http://localhost:5173, proxies /api → :8000
# Port 8000 busy? VITE_PROXY_TARGET=http://localhost:8001 npm run dev
```

### ML pipeline (optional — trained artifacts are committed)
```bash
pip install numpy pandas scikit-learn torch onnx onnxruntime matplotlib
python3 ml/scripts/preprocess.py      # CSVs → windows (dedup, gap-split)
python3 ml/scripts/train_baseline.py  # Random Forest, leave-one-session-out
python3 ml/scripts/train_cnn.py       # CNN eval + final model → ONNX → backend/app/ml/
```

### Test on a phone
Sensors need HTTPS (or `localhost`), so the deployed Vercel URL is easiest. iOS and recent Chrome ask for sensor permission: tap **Enable Sensors** on the Navigate screen.

## ☁️ Deployment (free tier)

| | Render (backend) | Vercel (frontend) |
|---|---|---|
| Root directory | `backend` | `frontend` |
| Build | `pip install -r requirements.txt` | `npm run build` → `dist` |
| Start | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` | — |
| Env | `PYTHON_VERSION=3.11.11` | rewrites in `vercel.json` proxy `/api/*` to Render |

Render's free tier sleeps after 15 minutes idle and wipes the disk on restart. The backend therefore re-seeds `struct_demo_001` (20 bays) on every boot, and the frontend shows a "waking up" state until `/api/health` answers. Sessions, uploads and floor plans last until the next restart; point `DATABASE_URL` at a free Postgres (e.g. Neon) if you need persistence.

**Demo day:** open the site ~2 minutes early so the backend is awake.

## 🧪 Tests

```bash
cd backend && pytest tests/ -v      # 38 tests: sessions, occupancy, admin, ML serving, seeding, uploads
cd frontend && npm run lint && npm run build
```

## 🔧 API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Readiness (used to ride out cold starts) |
| POST | `/api/sessions/entry` | Create session |
| GET | `/api/sessions/{id}` | Session details |
| POST | `/api/sessions/{id}/park` | Park in a bay |
| POST | `/api/sessions/{id}/exit` | End session, free bay |
| GET | `/api/occupancy/{struct_id}` | Bay status list |
| GET | `/api/occupancy/{struct_id}/summary` | Free/occupied counts |
| POST | `/api/ml/predict` | Classify a 20×6 sensor window; with a `session_id` the answer is smoothed over that session's last 3 windows |
| GET | `/api/ml/model` | Model card + live serving latency |
| POST | `/api/sensors/batch` | Ingest raw sensor batch (the app no longer sends these) |
| GET | `/api/sensors/position/{session_id}` | Zone-based position (stub) |
| GET | `/api/qr/{struct_id}` · `/exit` | Entry / exit QR PNG |
| GET | `/api/admin/stats` · `/sessions` · `/logs` | Dashboard data |
| POST | `/api/admin/seed` | Seed demo data (idempotent) |
| POST/GET | `/api/admin/training-data` | Collector CSV upload / list |
| POST/GET | `/api/admin/floorplan…` | Floor plan upload / serve |

## 🛣️ Roadmap

- Magnetic-fingerprint positioning on real phones: the PSO correction layer is prototyped in Admin → Positioning on a synthetic map. It needs a fingerprint survey and raw magnetometer access.
- Native app for background tracking; BLE anchors for turn-by-turn guidance.
