"""Reset the database to a fresh demo parking structure with 20 free bays."""
from app.database import SessionLocal, Base, engine
import app.models  # noqa — registers all models
from app.models.structure import Structure, Bay
from app.services.seed_service import ensure_demo_structure, DEMO_STRUCTURE_ID

Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Clear existing data
db.query(Bay).delete()
db.query(Structure).delete()
db.commit()

ensure_demo_structure(db)
print(f"✅ Seeded: {DEMO_STRUCTURE_ID} with 20 bays (2 floors × 10 bays, 4 EV)")
db.close()
