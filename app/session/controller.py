import asyncio,uuid
from enum import Enum
from app.schemas import SessionBlueprint
from app.audio.mixer import AudioMixer
from app.light.mobile import generate_mobile_frames
from app.light.vr import generate_vr_commands
from app.audio.subliminal import build_subliminal_loop
from app.voice.tts_renderer import build_meditation_loop
from app.voice.script_generator import generate_script_segment
from app.ai.architect import call_llm

class State(str,Enum): IDLE="idle";RUNNING="running";PAUSED="paused";COMPLETED="completed";STOPPED="stopped"
SESSIONS:dict[str,dict]={}

async def create_session(bp:SessionBlueprint,epilepsy:bool=False)->str:
    sid=str(uuid.uuid4())
    sub_loop=None
    if bp.subliminal.enabled and bp.subliminal.messages:
        sub_loop=await build_subliminal_loop(bp.subliminal.messages)
    med_loop=None
    if bp.meditation.enabled and bp.meditation.style!="silent":
        script=await generate_script_segment(bp.meditation,bp.meditation.script_outline,
            t_norm=0.0,segment_secs=bp.duration_seconds,llm_caller=lambda s,u:call_llm(s,u))
        med_loop=await build_meditation_loop(script,bp.meditation.voice_tone,bp.duration_seconds)
    SESSIONS[sid]={"blueprint":bp,"state":State.IDLE,"t":0.0,"epilepsy":epilepsy,
        "mixer":AudioMixer(bp,sub_loop,med_loop),"audio_q":asyncio.Queue(maxsize=4),
        "light_q":asyncio.Queue(maxsize=120),"stop_evt":asyncio.Event(),"pause_evt":asyncio.Event()}
    return sid

async def run_session(sid:str):
    s=SESSIONS[sid];bp=s["blueprint"];s["state"]=State.RUNNING;total=bp.duration_seconds
    while s["t"]<total:
        if s["stop_evt"].is_set():break
        if s["pause_evt"].is_set():await asyncio.sleep(0.05);continue
        await s["audio_q"].put(s["mixer"].next_chunk())
        t0=s["t"];t1=min(t0+1.0,total)
        if bp.client_type=="mobile": frames=generate_mobile_frames(bp,t0,t1,epilepsy=s["epilepsy"])
        else: frames=generate_vr_commands(bp,t0,t1,epilepsy=s["epilepsy"])
        await s["light_q"].put(frames);s["t"]=t1;await asyncio.sleep(0)
    s["state"]=State.COMPLETED if not s["stop_evt"].is_set() else State.STOPPED

def pause_session(sid): SESSIONS[sid]["pause_evt"].set();SESSIONS[sid]["state"]=State.PAUSED
def resume_session(sid): SESSIONS[sid]["pause_evt"].clear();SESSIONS[sid]["state"]=State.RUNNING
def stop_session(sid): SESSIONS[sid]["stop_evt"].set()
