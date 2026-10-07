"""Demo data seeding — shared by app startup, POST /api/admin/seed and seed.py."""
from sqlalchemy.orm import Session
from loguru import logger
from app.models.structure import Structure, Bay

DEMO_STRUCTURE_ID = "struct_demo_001"
ZONES = "ABCDEFGHIJ"
FLOORS = 2


def ensure_demo_structure(db: Session) -> bool:
    """Create the demo structure + 20 bays if missing. Returns True if it seeded.

    Render's free tier wipes the SQLite file on every restart, so this runs at
    startup — the demo always comes back with a fresh, fully free car park.
    """
    if db.get(Structure, DEMO_STRUCTURE_ID):
        return False

    structure = Structure(
        id=DEMO_STRUCTURE_ID,
        name="NAVIN Demo Parking",
        address="Demo Campus, Block A",
        total_bays=len(ZONES) * FLOORS,
        floor_count=FLOORS,
    )
    db.add(structure)
    db.flush()

    # 10 bays per floor, first 2 on each floor are EV
    for floor in range(FLOORS):
        for i, zone in enumerate(ZONES):
            db.add(Bay(
                structure_id=structure.id,
                bay_number=f"{zone}{floor + 1}",
                floor=floor,
                bay_type="ev" if i < 2 else "regular",
            ))
    db.commit()
    logger.info(f"Demo data seeded: {structure.total_bays} bays across {FLOORS} floors")
    return True
