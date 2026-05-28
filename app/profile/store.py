"""SQLite-backed profile store via TinyDB for MVP simplicity."""
from tinydb import TinyDB, Query
from app.schemas import SessionBlueprint
from pathlib import Path
import os
import datetime as _datetime

DB_PATH = os.getenv("PROFILE_DB_PATH", "./data/profiles.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
db = TinyDB(DB_PATH)
sessions_table = db.table("sessions")
U = Query()


async def save_rating(user_id: str, session_id: str,
                      bp: SessionBlueprint, rating: int,
                      notes: str | None = None):
    sessions_table.insert({
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": _datetime.datetime.now(_datetime.UTC).isoformat(),
        "blueprint": bp.model_dump(),
        "rating": rating,
        "notes": notes or "",
    })


def get_user_sessions(user_id: str, limit: int = 20) -> list[dict]:
    rows = sessions_table.search(U.user_id == user_id)
    return sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:limit]
