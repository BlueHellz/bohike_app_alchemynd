from pathlib import Path
import os
from app.schemas import MeditationConfig
PHASE_MAP={0.0:"opening",0.25:"deepening",0.5:"peak",0.75:"integration",0.9:"closing"}
def get_phase(t_norm:float)->str:
    for th in sorted(PHASE_MAP,reverse=True):
        if t_norm>=th: return PHASE_MAP[th]
    return "opening"
async def generate_script_segment(med:MeditationConfig,outline:str,t_norm:float,segment_secs:int,llm_caller):
    _THIS_DIR = Path(__file__).resolve().parent.parent
    with open(_THIS_DIR / "ai/prompts/system_meditation.txt") as f: template=f.read()
    system=template.format(style=med.style,voice_tone=med.voice_tone,pacing=med.pacing,
        phase=get_phase(t_norm),segment_secs=segment_secs,outline=outline)
    return await llm_caller(system,"Generate the script now.")
