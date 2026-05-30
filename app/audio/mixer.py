import numpy as np
from app.schemas import SessionBlueprint
from app.audio.binaural import BinauralGenerator, CHUNK_SAMP, SR
from app.audio.noise import generate_noise_chunk

GAINS={"binaural":1.0,"noise":0.7,"meditation":0.9,"subliminal":1.0}

class AudioMixer:
    def __init__(self,bp,subliminal_loop=None,meditation_loop=None):
        e=bp.entrainment
        self.beat=BinauralGenerator(initial_hz=e.initial_hz,target_hz=e.target_hz,
            total_secs=bp.duration_seconds,carrier_hz=bp.audio.binaural_carrier_hz,
            iso_intensity=bp.audio.isochronic_intensity,curve=e.curve)
        self.bp=bp; self.subliminal_loop=subliminal_loop; self.meditation_loop=meditation_loop
        self._sc=0; self._mc=0; self._t=0.0; self.total=bp.duration_seconds

    def _slice(self,buf,cursor,n):
        if buf is None: return np.zeros(n,np.float32),cursor
        end=cursor+n
        if end<=len(buf): return buf[cursor:end].copy(),end
        p1=buf[cursor:]; p2=buf[:end-len(buf)]
        return np.concatenate([p1,p2]),end-len(buf)

    def next_chunk(self):
        N=CHUNK_SAMP
        binn=self.beat.next_chunk(self._t)
        noise_m=generate_noise_chunk(self.bp.audio.noise_types,self.bp.audio.noise_blend,N)
        noise_l=noise_m*GAINS["noise"]; noise_r=noise_m*GAINS["noise"]
        med_m,self._mc=self._slice(self.meditation_loop,self._mc,N)
        sub_m,self._sc=self._slice(self.subliminal_loop,self._sc,N)
        fade=np.ones(N,np.float32)
        if self._t<2.0: ri=min(int(2.0*SR),N); fade[:ri]*=np.linspace(0,1,ri)
        rem=self.total-self._t
        if rem<3.0: ri=min(int(rem*SR),N); fade[-ri:]*=np.linspace(1,0,ri) if ri>0 else 1
        left=(binn[:,0]*GAINS["binaural"]+noise_l+med_m*GAINS["meditation"]+sub_m*GAINS["subliminal"])*fade
        right=(binn[:,1]*GAINS["binaural"]+noise_r+med_m*GAINS["meditation"]+sub_m*GAINS["subliminal"])*fade
        left=np.clip(left,-1.0,1.0); right=np.clip(right,-1.0,1.0)
        self._t+=1.0
        return (np.stack([left,right],axis=1)*32767).astype(np.int16).tobytes()
