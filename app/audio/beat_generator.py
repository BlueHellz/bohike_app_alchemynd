import numpy as np
from dataclasses import dataclass, field

SAMPLE_RATE = 44100
CHUNK_SECS = 1.0
CHUNK_SAMP = int(SAMPLE_RATE * CHUNK_SECS)


@dataclass
class BeatGenerator:
    """Phase-continuous binaural + isochronic generator."""

    initial_hz: float
    target_hz: float
    total_secs: float
    carrier_hz: float = 200.0
    iso_intensity: float = 0.4
    curve: str = "ease_in_out"
    _phase_L: float = field(default=0.0, init=False)
    _phase_R: float = field(default=0.0, init=False)
    _phase_iso: float = field(default=0.0, init=False)

    def _f(self, t: float) -> float:
        """Instantaneous beat frequency at time t."""
        p = t / self.total_secs
        if self.curve == "linear":
            k = p
        elif self.curve == "ease_in":
            k = p * p
        elif self.curve == "ease_out":
            k = 1 - (1 - p) ** 2
        elif self.curve == "ease_in_out":
            k = p * p * (3 - 2 * p)  # smoothstep
        else:
            k = p
        return self.initial_hz + (self.target_hz - self.initial_hz) * k

    def next_chunk(self, t_start: float, sr: int = 44100) -> np.ndarray:
        """
        Returns stereo float32 array shape (N, 2).
        Maintains phase across calls -- no clicks at buffer boundaries.
        """
        N = int(CHUNK_SECS * sr)
        t = np.linspace(t_start, t_start + CHUNK_SECS, N, endpoint=False)
        beat = np.array([self._f(ti) for ti in t])  # shape (N,)

        # Phase accumulation (2*pi * freq * dt per sample)
        dt = 1.0 / sr
        d_phase_L = 2 * np.pi * self.carrier_hz * dt
        d_phase_R = 2 * np.pi * (self.carrier_hz + beat) * dt
        d_phase_iso = 2 * np.pi * beat * dt

        phases_L = self._phase_L + np.cumsum(np.full(N, d_phase_L))
        phases_R = self._phase_R + np.cumsum(d_phase_R)
        phases_iso = self._phase_iso + np.cumsum(d_phase_iso)

        self._phase_L = phases_L[-1] % (2 * np.pi)
        self._phase_R = phases_R[-1] % (2 * np.pi)
        self._phase_iso = phases_iso[-1] % (2 * np.pi)

        left = np.sin(phases_L).astype(np.float32)
        right = np.sin(phases_R).astype(np.float32)
        iso = (np.sin(phases_iso) * self.iso_intensity).astype(np.float32)

        # Isochronic is mono-centre: add to both channels at lower gain
        left += iso * 0.5
        right += iso * 0.5
        return np.stack([left, right], axis=1)
