import asyncio, os, numpy as np
from scipy.signal import butter, sosfilt
SR=44100; ATTEN_DB=-40.0

def bandpass(audio,lo=300,hi=3000,sr=SR):
    sos=butter(4,[lo,hi],'band',fs=sr,output='sos')
    return sosfilt(sos,audio).astype(np.float32)

def _norm(audio,target_rms):
    rms=np.sqrt(np.mean(audio**2))+1e-9; return audio*(target_rms/rms)

async def build_subliminal_loop(messages,ambient_rms=0.08,tts_engine=None):
    from app.voice.tts_renderer import render_script
    trms=ambient_rms*(10**(ATTEN_DB/20.0))
    sil=np.zeros(SR*1,np.float32); parts=[]
    for msg in messages:
        wav=await render_script(msg,tone="whispering")
        wav=bandpass(wav); wav=_norm(wav,trms)
        pl=SR*5; wav=wav[:pl] if len(wav)>=pl else np.concatenate([wav,np.zeros(pl-len(wav),np.float32)])
        parts.extend([wav,sil])
    return np.concatenate(parts)
