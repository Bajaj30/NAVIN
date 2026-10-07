# 🎤 NAVIN — 10-Minute Demo Script

> **Total time: 10 minutes**
> **Format: 5 min slides + 4 min live demo + 1 min Q&A**

---

## Pre-Demo Checklist (do 5 minutes before)

- [ ] **Wake the backend:** open `https://navin-black.vercel.app/` on the laptop and wait for **● Server online** under the Demo Entry button (free hosting sleeps; waking takes up to a minute).
- [ ] Phone: Chrome open on the same URL, screen timeout ≥ 10 min, Do Not Disturb on.
- [ ] Admin dashboard open in a laptop tab: `/admin` (bookmark `/admin?tab=ml` and `/admin?tab=pso` too).
- [ ] Do one practice run: Demo Entry on the phone → **Enable Sensors** → check that the Activity AI card shows **CNN LIVE**.
- [ ] PPT loaded and ready.

---

## PART 1: Slides (5 minutes)

### Slide 1 — Title (30 sec)
> **Say:** "Good [morning/afternoon]. We're presenting NAVIN — an Indoor Parking Navigation System. The core idea is simple: what happens when GPS dies the moment you drive into a parking garage? NAVIN solves this using the sensors already in your phone."

- Point to the GitHub URL: `github.com/Bajaj30/NAVIN`

---

### Slide 2 — The Problem (45 sec)
> **Say:** "Every GPS-based app — Google Maps, Waze, Apple Maps — loses signal inside parking structures. Concrete and steel rebar block satellite signals. Drivers waste 5 to 15 minutes circling for spots, and later struggle to find their car."

**Key point to emphasize:** *This isn't a niche problem — every multi-level parking structure has this.*

---

### Slide 3 — NAVIN Solution (45 sec)
1. **"Zero-Install PWA"** — "Open a URL from a QR sticker. No app store, no account."
2. **"Sensor-based intelligence"** — "Accelerometer and gyroscope tell us what you're doing — walking, on the stairs, or parked — and the compass + step count track where you are."
3. **"QR-Gate Management"** — "Sessions start and end with QR codes at the gates — no extra hardware."

---

### Slide 4 — Core Features (45 sec)
1. **QR Entry** — "Scan at the gate, session starts instantly"
2. **Activity AI** — "A 1D-CNN classifies your movement every second and detects when you've parked"
3. **Occupancy Grid** — "Bays go red/green live for every driver"
4. **Admin Dashboard** — "Live sessions with each driver's activity, logs, QR codes, floor plans, and the model card"

> "I'll demo all four of these live in a moment."

---

### Slide 5 — Tech Stack (30 sec)
> **Say:** "React PWA on Vercel, FastAPI on Render — all free tier. The phone samples motion at 10 Hz and sends a 2-second window every second to the backend, where a PyTorch-trained CNN runs on ONNX Runtime in well under a millisecond. If the model is unreachable, an on-device heuristic takes over automatically."

---

### Slide 6 — User Flow (45 sec)
1. **Entry** → "Scan QR at the gate"
2. **Navigate** → "Walk with the phone; the system tracks steps, heading and activity"
3. **Park** → "The CNN sees sustained stillness for about eight seconds and suggests parking"
4. **Exit** → "Exit closes the session and frees the bay"

---

### Slide 7 — The ML Model (45 sec)
> **Say:** "We collected our own labelled data with a web collector page — walking, stairs, standing still — then trained a 1D-CNN. Two engineering decisions mattered most:"
1. "**Posture-invariant inputs** — we split acceleration and rotation along gravity, so it works whether the phone is in your hand, your pocket, or in front of you. Our first model failed exactly there."
2. "**Honest evaluation** — we test on whole recordings the model never saw. That gives 94% weighted F1, versus 80% for a Random Forest baseline. A random split would have claimed 97%, but that number leaks."

---

### Slide 8 — Admin Dashboard (20 sec)
> **Say:** "Facility managers see live sessions — including what each driver is doing right now — plus logs, QR codes and floor plans."

---

### Slide 9 — Future Work (15 sec)
> **Say:** "Next is metre-level positioning from magnetic fingerprints of the steel in the building. We've prototyped the particle-swarm correction layer in simulation — I'll show it. On real phones it needs a fingerprint survey and raw magnetometer access."

---

### Slide 10 — Closing (10 sec)
> **Say:** "Let me show you NAVIN working live."

---

## PART 2: Live Demo (4 minutes)

### Demo Step 1: Entry Gate (30 sec)
1. **On the phone**, open `https://navin-black.vercel.app/`
2. > **Say:** "This is the entry gate. A driver scans the QR posted at the entrance — for the demo I'll use the demo button."
3. Tap **Demo Entry (No QR)** → session created, zone assigned.

### Demo Step 2: Navigate + Activity AI (1 min 30 sec)
1. Tap **Enable Sensors** (iOS and recent Chrome ask for permission).
2. Hold the phone still in front of you, then point to the **Activity AI** card:
   > **Say:** "Every second the phone sends the last two seconds of motion to our CNN. Right now it says Stationary with high confidence — that's the model on the server, 'CNN LIVE', with the round-trip time here."
3. **Walk** a few steps (keep the phone in hand):
   > **Say:** "Now it flips to Walking — notice the probabilities shift. Stairs is a separate class; telling stairs from walking is exactly what a simple threshold can't do."
4. **Stop and stand still ~8 seconds** → the **Vehicle appears parked** banner appears.
   > **Say:** "Sustained stillness — the system suggests I've parked."
5. Point to the **Floor Map** and **Parking Bays**: "Dead reckoning moves the dot with steps and heading; green bays are free."

### Demo Step 3: Park + Exit (30 sec)
1. Tap a **free bay** (e.g. C1) → Parked page shows bay, zone, duration counting up.
2. Tap **Exit Parking** → "Session closed, bay freed for the next driver."

### Demo Step 4: Admin Dashboard (1 min 30 sec)
1. Laptop → `/admin` → **Sessions**:
   > **Say:** "Every session, with the CNN's latest activity per driver."
2. **🤖 ML Model** tab:
   > **Say:** "The model card: 94% weighted F1 on held-out recordings versus 80% for the Random Forest. The random-split number is crossed out on purpose — it leaks. Here's the confusion matrix; most remaining errors are stairs versus walking."
3. **🛰️ Positioning** tab, set **4×**:
   > **Say:** "This is the next layer, in simulation. White is the true walk, orange is dead reckoning alone — it drifts metres off. Blue is particle swarm optimisation: 80 particles matching magnetometer readings against a fingerprint map. It stays within about a metre."
   - Tap **Scatter swarm**: "Even if it loses track completely, the swarm relocks within a few seconds."

---

## PART 3: Wrap-Up (1 minute)

> **Say:** "To summarize — NAVIN gives you indoor parking navigation with nothing but your phone: QR gates for sessions, a CNN for activity and parking detection, and a clear path to metre-level positioning. No GPS, no app download, no extra hardware. The repo is public at github.com/Bajaj30/NAVIN. Happy to take questions."

---

## 🚨 Fallback Plans

| Problem | Solution |
|---------|----------|
| "Waking up the server" banner on entry | Normal for free hosting — wait (≤ 1 min). Avoid it by opening the site 2 min early. |
| Activity AI shows **SENSORS OFF** | Tap **Enable Sensors** at the top of the Navigate screen. |
| Activity AI shows **HEURISTIC FALLBACK** | The model API is unreachable; the app keeps working on the on-device heuristic. Say so — it's the designed fallback. |
| Sensors show 0.0 on the laptop | Laptops have no motion sensors — demo on the phone; use the laptop for admin. |
| Camera access denied on entry | Use **Demo Entry (No QR)**. |
| Backend down entirely | Run locally: `uvicorn app.main:app --port 8000` + `npm run dev`, phone via the laptop's LAN IP. |

---

## ⏱️ Timing Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| Slides 1–3 (Problem + Solution) | 2:00 | 2:00 |
| Slides 4–8 (Features, Stack, ML, Admin) | 2:40 | 4:40 |
| Slides 9–10 (Future + Transition) | 0:25 | 5:05 |
| Live Demo (Entry → AI → Park/Exit → Admin) | 4:00 | 9:05 |
| Wrap-up + Q&A | 0:55 | **10:00** |
