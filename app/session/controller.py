import asyncio
import uuid
from enum import Enum
from app.schemas import SessionBlueprint
from app.audio.mixer import AudioMixer
from app.audio.subliminal import build_subliminal_loop
from app.light.mobile_generator import generate_mobile_frames
from app.light.vr_generator import generate_vr_commands
from app.safety.validator import validate_blueprint


class State(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


# In-memory session store (replace with Redis for multi-instance)
SESSIONS: dict[str, dict] = {}


async def create_session(bp: SessionBlueprint,
                         epilepsy: bool = False) -> str:
    bp, warnings = validate_blueprint(bp, epilepsy)
    sid = str(uuid.uuid4())

    # Pre-render subliminal audio (async, non-blocking)
    sub_loop = None
    if bp.subliminal.enabled and bp.subliminal.messages:
        sub_loop = await build_subliminal_loop(bp.subliminal.messages)

    SESSIONS[sid] = {
        "blueprint": bp,
        "state": State.IDLE,
        "t": 0.0,
        "mixer": AudioMixer(bp, sub_loop),
        "epilepsy": epilepsy,
        "warnings": warnings,
        "audio_q": asyncio.Queue(maxsize=4),
        "light_q": asyncio.Queue(maxsize=120),
        "stop_evt": asyncio.Event(),
        "pause_evt": asyncio.Event(),
    }
    return sid


async def run_session(sid: str):
    s = SESSIONS[sid]
    bp = s["blueprint"]
    s["state"] = State.RUNNING
    total = bp.duration_seconds

    while s["t"] < total:
        if s["stop_evt"].is_set():
            break
        if s["pause_evt"].is_set():
            await asyncio.sleep(0.05)
            continue

        # Audio chunk
        chunk = s["mixer"].next_chunk()
        await s["audio_q"].put(chunk)

        # Light data
        t_start = s["t"]
        t_end = min(s["t"] + 1.0, total)
        if bp.client_type == "mobile":
            frames = generate_mobile_frames(bp, t_start, t_end,
                                            epilepsy=s["epilepsy"])
            await s["light_q"].put(frames)
        else:
            cmds = generate_vr_commands(bp, t_start, t_end,
                                        epilepsy=s["epilepsy"])
            await s["light_q"].put(cmds)

        s["t"] += 1.0
        await asyncio.sleep(0)  # yield to event loop

    s["state"] = State.COMPLETED if not s["stop_evt"].is_set() else State.STOPPED


def pause_session(sid: str):
    SESSIONS[sid]["pause_evt"].set()
    SESSIONS[sid]["state"] = State.PAUSED


def resume_session(sid: str):
    SESSIONS[sid]["pause_evt"].clear()
    SESSIONS[sid]["state"] = State.RUNNING


def stop_session(sid: str):
    SESSIONS[sid]["stop_evt"].set()
