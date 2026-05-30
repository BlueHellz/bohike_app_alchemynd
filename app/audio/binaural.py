import os, numpy as np
from dataclasses import dataclass, field

SR = int(os.getenv("SAMPLE_RATE","44100"))
CHUNK_SECS = float(os.getenv("CHUNK_SECS","1.0"))
CHUNK_SAMP = int(SR * CHUNK_SECS)

@dataclass
class BinauralGenerator:
    initial_hz: float; target_hz: float; total_secs: float
    carrier_hz: float = 200.0; iso_intensity: float = 0.4; curve: str = "ease_in_out"
    _phase_L: float = field(default=0.0, init=False)
    _phase_R: float = field(default=0.0, init=False)
    _phase_iso: float = field(default=0.0, init=False)

    def _beat_at(self, t: float) -> float:
        p = max(0.0, min(1.0, t / self.total_secs))
        k = {"linear":p,"ease_in":p*p,"ease_out":1-(1-p)**2,"ease_in_out":p*p*(3-2*p)}.get(self.curve,p)
        return self.initial_hz + (self.target_hz - self.initial_hz) * k

    def next_chunk(self, t_start: float) -> np.ndarray:
        N = CHUNK_SAMP
        t = np.linspace(t_start, t_start+CHUNK_SECS, N, endpoint=False)
        beat = np.array([self._beat_at(ti) for ti in t], dtype=np.float32)
        dt = 1.0/SR
        dL = np.full(N, 2*np.pi*self.carrier_hz*dt, dtype=np.float64)
        dR = 2*np.pi*(self.carrier_hz+beat)*dt
        dI = 2*np.pi*beat*dt
        pL = self._phase_L + np.cumsum(dL)
        pR = self._phase_R + np.cumsum(dR)
        pI = self._phase_iso + np.cumsum(dI)
        self._phase_L = float(pL[-1]) % (2*np.pi)
        self._phase_R = float(pR[-1]) % (2*np.pi)
        self._phase_iso = float(pI[-1]) % (2*np.pi)
        left = np.sin(pL).astype(np.float32)
        right = np.sin(pR).astype(np.float32)
        iso = (np.sin(pI)*self.iso_intensity).astype(np.float32)
        left += iso*0.5; right += iso*0.5
        return np.stack([left,right],axis=1)
