from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.database import Base, engine, SessionLocal
from app.core.logger import setup_logging
from app.routers import sessions, occupancy, sensors, structures, qr, admin, ml
from app.services import ml_service
from app.services.seed_service import ensure_demo_structure


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()  # start logger before anything else
    Base.metadata.create_all(bind=engine)  # safety net if alembic skipped
    if settings.seed_demo_data:
        with SessionLocal() as db:
            ensure_demo_structure(db)  # Render free tier wipes SQLite on restart
    ml_service.load_model()
    yield


import os

app = FastAPI(title="NAVIN API", lifespan=lifespan)

# CORS — allow frontend origins
cors_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
cors_origins += [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for prototype — restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(occupancy.router, prefix="/api/occupancy", tags=["occupancy"])
app.include_router(sensors.router, prefix="/api/sensors", tags=["sensors"])
app.include_router(structures.router, prefix="/api/structures", tags=["structures"])
app.include_router(qr.router, prefix="/api/qr", tags=["qr"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(ml.router, prefix="/api/ml", tags=["ml"])


@app.get("/")
def root():
    return {"app": "NAVIN", "version": "1.0.0", "status": "running"}


@app.get("/api/health")
def health():
    """Cheap readiness probe — the frontend polls this to ride out Render cold starts."""
    return {"status": "ok", "model_loaded": ml_service.is_loaded()}
