import math
from app.schemas import SessionBlueprint
def generate_vr_commands(bp,t_start,t_end,epilepsy=False):
    if bp.vr_visual is None: return[]
    vr=bp.vr_visual;e=bp.entrainment;total=bp.duration_seconds;cmds=[]
    if t_start==0.0: cmds.append({"cmd":"load_scene","t_ms":0,"scene_type":vr.scene_type,"color_palette":vr.color_palette})
    t=t_start
    while t<t_end:
        p=t/total;k=p*p*(3-2*p);bf=e.initial_hz+(e.target_hz-e.initial_hz)*k
        if epilepsy: bf=min(bf,3.0)
        intensity=0.5+0.5*math.sin(2*math.pi*bf*t)
        if vr.global_light_rhythm: cmds.append({"cmd":"set_global_light","t_ms":int(t*1000),"intensity":round(intensity,4),"color":vr.color_palette[0]})
        for el in vr.dynamic_elements: cmds.append({"cmd":"update_element","t_ms":int(t*1000),"type":el.type,"sync_val":round(intensity,4),"params":el.params})
        t+=1.0
    return cmds
