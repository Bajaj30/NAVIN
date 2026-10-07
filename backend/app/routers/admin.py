"""Admin router — sessions overview, log tail, aggregate stats, floor plan management."""
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.session import Session as SessionModel, SensorBatch
from app.models.structure import Bay
from app.services import ml_service
from app.services.seed_service import ensure_demo_structure, DEMO_STRUCTURE_ID
from pathlib import Path
from collections import Counter
from loguru import logger
from datetime import datetime, timedelta
import re
import time
import shutil

UPLOAD_DIR = Path("uploads/floorplans")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR = Path("uploads/training")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

CSV_HEADER = "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,label"
MAX_TRAINING_BYTES = 20 * 1024 * 1024

router = APIRouter()


@router.post("/seed")
def seed_demo_data(db: Session = Depends(get_db)):
    """Seed demo structure + bays (idempotent)."""
    if not ensure_demo_structure(db):
        bay_count = db.query(func.count(Bay.id)).filter(Bay.structure_id == DEMO_STRUCTURE_ID).scalar()
        return {"seeded": False, "message": "Already exists", "bays": bay_count}
    return {"seeded": True, "message": f"Created {DEMO_STRUCTURE_ID} with 20 bays"}


@router.post("/training-data")
async def upload_training_data(request: Request, filename: str = "collector.csv"):
    """Receive a labelled CSV from the data collector page (raw text/csv body)."""
    body = await request.body()
    if len(body) > MAX_TRAINING_BYTES:
        raise HTTPException(status_code=413, detail="CSV too large (max 20 MB)")

    text = body.decode("utf-8", errors="replace").strip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != CSV_HEADER:
        raise HTTPException(status_code=400, detail="Not a NAVIN collector CSV (unexpected header)")

    labels = Counter(line.rsplit(",", 1)[-1].strip() for line in lines[1:] if line.strip())
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name) or "collector.csv"
    if not safe_name.endswith(".csv"):
        safe_name += ".csv"
    stored_name = f"{int(time.time())}_{safe_name}"
    (TRAINING_DIR / stored_name).write_text(text + "\n")

    logger.info(f"POST /admin/training-data | {stored_name} | {len(lines) - 1} samples | {dict(labels)}")
    return {"uploaded": True, "filename": stored_name, "rows": len(lines) - 1, "labels": dict(labels)}


@router.get("/training-data")
def list_training_data():
    """List CSVs uploaded from the collector (ephemeral on Render's free tier)."""
    files = []
    for f in sorted(TRAINING_DIR.glob("*.csv"), reverse=True):
        rows = sum(1 for _ in f.open()) - 1
        files.append({
            "filename": f.name,
            "rows": rows,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "uploaded_at": datetime.utcfromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return {"files": files}


@router.get("/training-data/{filename}")
def download_training_data(filename: str):
    path = TRAINING_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.post("/floorplan")
async def upload_floorplan(
    file: UploadFile = File(...),
    structure_id: str = Form(...),
    floor: int = Form(0),
    width_meters: float = Form(50.0),
    height_meters: float = Form(30.0),
):
    """Upload a floor plan image for a structure + floor."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    filename = f"{structure_id}_floor{floor}.{ext}"
    filepath = UPLOAD_DIR / filename

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Save metadata alongside
    meta_path = UPLOAD_DIR / f"{structure_id}_floor{floor}.meta"
    meta_path.write_text(f"{width_meters},{height_meters},{filename}")

    logger.info(f"Floor plan uploaded: {filename} ({width_meters}x{height_meters}m)")
    return {
        "uploaded": True,
        "filename": filename,
        "structure_id": structure_id,
        "floor": floor,
        "width_meters": width_meters,
        "height_meters": height_meters,
    }


@router.get("/floorplan/{structure_id}")
def get_floorplan(structure_id: str, floor: int = 0):
    """Serve the floor plan image."""
    meta_path = UPLOAD_DIR / f"{structure_id}_floor{floor}.meta"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No floor plan uploaded for this structure/floor")

    parts = meta_path.read_text().strip().split(",")
    filename = parts[2]
    filepath = UPLOAD_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Floor plan file missing")

    return FileResponse(filepath, media_type="image/png")


@router.get("/floorplan/{structure_id}/meta")
def get_floorplan_meta(structure_id: str, floor: int = 0):
    """Get floor plan metadata (dimensions)."""
    meta_path = UPLOAD_DIR / f"{structure_id}_floor{floor}.meta"
    if not meta_path.exists():
        return {"exists": False}

    parts = meta_path.read_text().strip().split(",")
    return {
        "exists": True,
        "width_meters": float(parts[0]),
        "height_meters": float(parts[1]),
        "filename": parts[2],
        "url": f"/api/admin/floorplan/{structure_id}?floor={floor}",
    }


@router.get("/sessions")
def get_sessions(db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    sessions = db.query(SessionModel).order_by(SessionModel.entry_time.desc()).limit(100).all()
    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"GET /admin/sessions | {len(sessions)} results | {ms}ms")
    return {
        "sessions": [
            {
                "id": s.id,
                "structure_id": s.structure_id,
                "status": s.status,
                "assigned_zone": s.assigned_zone,
                "entry_time": s.entry_time.isoformat(),
                "exit_time": s.exit_time.isoformat() if s.exit_time else None,
                "duration_mins": s.duration_mins,
                "parked_bay_id": s.parked_bay_id,
                "activity": ml_service.latest(s.id),
            }
            for s in sessions
        ],
        "total": len(sessions),
    }


@router.get("/logs")
def get_logs():
    log_path = Path("logs/navin.log")
    if not log_path.exists():
        return {"lines": ["No logs yet"]}
    lines = log_path.read_text().strip().split("\n")
    return {"lines": lines[-50:]}


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    t0 = time.perf_counter()

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    active = db.query(func.count(SessionModel.id)).filter(
        SessionModel.status == "active"
    ).scalar()

    parked = db.query(func.count(SessionModel.id)).filter(
        SessionModel.status == "parked"
    ).scalar()

    exited_today = db.query(func.count(SessionModel.id)).filter(
        SessionModel.status == "exited",
        SessionModel.exit_time >= today_start,
    ).scalar()

    batches_today = db.query(func.count(SensorBatch.id)).filter(
        SensorBatch.recorded_at >= today_start,
    ).scalar()

    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"GET /admin/stats | {ms}ms")
    return {
        "active": active,
        "parked": parked,
        "exited_today": exited_today,
        "batches_today": batches_today,
    }
