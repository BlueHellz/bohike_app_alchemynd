import math
from app.schemas import SessionBlueprint
def hex_to_rgb(h):
    h=h.lstrip("#");return int(h[:2],16),int(h[2:4],16),int(h[4:],16)
def generate_mobile_frames(bp,t_start,t_end,fps=60.0,epilepsy=False):
    mv=bp.mobile_visual;e=bp.entrainment;total=bp.duration_seconds
    r,g,b=hex_to_rgb(mv.color_hex);frames=[];t=t_start;dt=1.0/fps
    while t<t_end:
        p=t/total;k=p*p*(3-2*p);bf=e.initial_hz+(e.target_hz-e.initial_hz)*k
        if epilepsy: bf=min(bf,3.0)
        if mv.pulse_sync and bf<=25.0: alpha=0.3+0.7*math.sin(2*math.pi*bf*t)
        elif mv.pulse_sync: alpha=0.3+0.7*math.sin(2*math.pi*25.0*t)
        else: alpha=1.0
        fade=min(t/2.0,1.0,(total-t)/2.0)
        frames.append({"t_ms":int(t*1000),"r":r,"g":g,"b":b,"a":round(abs(alpha)*fade,4)})
        t+=dt
    return frames
