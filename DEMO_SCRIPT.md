# 🎤 NAVIN — 10-Minute Demo Script

> **Total time: 10 minutes**
> **Format: 5 min slides + 4 min live demo + 1 min Q&A**

---

## Pre-Demo Checklist (do 5 minutes before)

- [ ] Backend running locally OR Render URL is live
- [ ] Frontend open on laptop: `https://navin-black.vercel.app/` (or localhost)
- [ ] Phone ready on same WiFi with frontend URL loaded
- [ ] Admin dashboard open in a separate tab: `/admin`
- [ ] PPT loaded and ready

---

## PART 1: Slides (5 minutes)

### Slide 1 — Title (30 sec)
> **Say:** "Good [morning/afternoon]. We presenting NAVIN — an Indoor Parking Navigation System. The core idea is simple: what happens when GPS dies the moment you drive into a parking garage? NAVIN solves this using the sensors already in your phone."

- Point to the GitHub URL: `github.com/Bajaj30/NAVIN`

---

### Slide 2 — The Problem (45 sec)
> **Say:** "Every GPS-based app — Google Maps, Waze, Apple Maps — completely loses signal inside parking structures. Concrete walls and steel rebar block satellite signals. The result? Drivers waste 5 to 15 minutes circling for spots, and later struggle to find their car."

**Key point to emphasize:** *This isn't a niche problem — every multi-level parking structure has this.*

---

### Slide 3 — NAVIN Solution (45 sec)
> **Say:** "NAVIN takes three approaches to solve this:"

1. **"Zero-Install PWA"** — "It's a Progressive Web App. No app store download, no account creation. Open a URL and you're in."
2. **"Sensor-Based Positioning"** — "We use the magnetometer, accelerometer, and gyroscope already in your phone to track position and detect walking vs stationary."
3. **"QR-Gate Management"** — "Sessions are managed via QR codes at entry and exit gates — simple, clean, no extra hardware."

---

### Slide 4 — Core Features (45 sec)
> **Say:** "Let me walk through the four core features we've built:"

1. **QR Entry** — "Scan at the gate, session starts instantly"
2. **Live Sensors** — "Real-time accelerometer, gyroscope, compass displayed on screen"
3. **Occupancy Grid** — "Backend tracks which bays are free or taken, updates live"
4. **Admin Dashboard** — "Facility managers see everything — sessions, logs, stats, floor plans"

> "I'll demo all four of these live in a moment."

---

### Slide 5 — Tech Stack (30 sec)
> **Say:** "Quick tech overview — FastAPI backend with SQLAlchemy ORM, React 18 frontend with Vite and Tailwind CSS, deployed on Render and Vercel. The sensor pipeline collects data at 60Hz, throttles UI to 15fps, and batch-sends to the backend."

*Don't linger here — move on quickly.*

---

### Slide 6 — User Flow (45 sec)
> **Say:** "The user journey is four steps:"

Draw a visual flow with your hand:
1. **Entry** → "Scan QR at the gate"
2. **Navigate** → "Walk with phone, system tracks steps and heading"
3. **Park** → "Activity classifier detects you've stopped moving — auto-transitions to PARKED"
4. **Exit** → "Scan exit QR, session closes, bay freed"

> "The activity classifier uses a state machine — not simple thresholds. It requires sustained stillness before confirming parking, avoiding false triggers."

---

### Slide 7 — Sensor Tracking (30 sec)
> **Say:** "Under the hood, we're reading three sensor streams: the magnetometer for compass heading, the accelerometer for step detection and movement magnitude, and the gyroscope for turn detection — left turn, right turn. These combine to give us dead reckoning positioning without any GPS."

---

### Slide 8 — Admin Dashboard (30 sec)
> **Say:** "The admin side gives full visibility — live session table, occupancy heatmap, server logs, QR code generation, and floor plan upload. Admins can upload their own parking structure blueprint and the system renders it as the navigation background."

---

### Slide 9 — Future Work (15 sec)
> **Say:** "For future phases: native app for background tracking, magnetic fingerprint matching for sub-meter accuracy, and BLE beacon anchoring for turn-by-turn guidance."

*Keep this brief — this is aspirational, not built.*

---

### Slide 10 — Closing (15 sec)
> **Say:** "That's the overview. Let me now show you NAVIN working live."

**Transition to live demo →**

---

## PART 2: Live Demo (4 minutes)

### Demo Step 1: Entry Gate (45 sec)

1. **On laptop**, open `https://navin-black.vercel.app/`
2. > **Say:** "This is the entry gate screen. In production, a driver would scan the QR code posted at the parking entrance. For this demo, I'll use the demo button."
3. Click **"Demo Entry (No QR)"**
4. > **Say:** "Session created instantly — I've been assigned zone D, and the system starts collecting sensor data."

---

### Demo Step 2: Navigate Page (1 min 15 sec)

1. Point to the **Sensor Data** card:
   > **Say:** "Here you can see three columns of live sensor data — Orientation from the compass, Accelerometer measuring movement force, and Gyroscope tracking rotation speed."

2. **Pick up the phone/laptop and tilt it** to show values changing:
   > **Say:** "Watch the values change as I move — the compass heading updates, accelerometer shows movement, and the activity classifier in the header switches between WALKING and STATIONARY."

3. Point to the **Floor Map**:
   > **Say:** "This canvas-based map shows the parking structure layout. The blue dot is my estimated position using dead reckoning — each step I take moves the dot based on my compass heading. In production, the admin uploads their actual floor plan as the background."

4. Point to the **Occupancy Grid**:
   > **Say:** "Below the map, every parking bay is shown — green means free, red means taken. The lightning bolt indicates EV/premium spots."

---

### Demo Step 3: Park (30 sec)

1. Click a **free bay** (e.g., C1):
   > **Say:** "I'll park in bay C1."

2. Show the **Parked page**:
   > **Say:** "The system confirms parking — Bay C1, Zone B, Duration 0 minutes. This duration counts up in real-time. Notice there's no timezone offset bug — it correctly shows 0 minutes."

---

### Demo Step 4: Exit (15 sec)

1. Click **"Exit Parking"**:
   > **Say:** "Exiting frees the bay and ends the session. Bay C1 is now available for the next driver."

---

### Demo Step 5: Admin Dashboard (1 min)

1. Navigate to `/admin`:
   > **Say:** "Now let me show the admin side."

2. Point to **stat cards**:
   > **Say:** "Live stats — active sessions, parked vehicles, exits today, and total sensor batches collected."

3. Point to **sessions table**:
   > **Say:** "Every session is tracked — you can see the one I just completed, marked as EXITED with the correct duration."

4. Click **"Logs"** tab:
   > **Say:** "Server logs stream in real-time — useful for debugging and monitoring."

5. Click **"QR Codes"** tab:
   > **Say:** "Auto-generated QR codes that you'd print and post at physical gates."

6. Click **"Map"** tab:
   > **Say:** "And finally, admins can upload their parking structure floor plan here — set the physical dimensions and it becomes the navigation background."

---

## PART 3: Wrap-Up (1 minute)

> **Say:** "To summarize — NAVIN provides a complete indoor navigation solution using only the sensors in your phone. No GPS needed, no app download, no extra hardware. The system handles entry, navigation, parking detection, exit, and facility management through a clean PWA interface."

> **Say:** "The repo is public at github.com/Bajaj30/NAVIN. Happy to take any questions."

---

## 🚨 Fallback Plans

| Problem | Solution |
|---------|----------|
| Render backend is sleeping | Use local servers: `localhost:5173` + `localhost:8000` |
| Sensors show 0.0 on laptop | Normal — sensors only work on mobile. Explain this and show the architecture |
| Camera access denied on entry | Use the "Demo Entry (No QR)" button |
| Page loads but API fails | Switch to localhost demo, show network tab to explain the architecture |

---

## ⏱️ Timing Summary

| Section | Duration | Cumulative |
|---------|----------|------------|
| Slides 1-3 (Problem + Solution) | 2:00 | 2:00 |
| Slides 4-8 (Features + Tech) | 2:45 | 4:45 |
| Slides 9-10 (Future + Transition) | 0:30 | 5:15 |
| Live Demo (Entry → Park → Exit → Admin) | 3:45 | 9:00 |
| Wrap-up + Q&A | 1:00 | **10:00** |
