"""
Stateful per-session audio mixer.
Combines binaural beats + soundscape + subliminal -> int16 stereo chunks.
"""
import numpy as np
from app.audio.beat_generator import BeatGenerator, CHUNK_SECS
from app.audio.soundscape import SoundscapeGenerator
from app.schemas import SessionBlueprint

BINAURAL_GAIN = 1.0
SOUNDSCAPE_GAIN = 0.8
SUBLIMINAL_GAIN = 1.0  # subliminal is pre-attenuated in subliminal.py


class AudioMixer:
    def __init__(self, bp: SessionBlueprint,
                 subliminal_loop: np.ndarray | None = None):
        e = bp.entrainment
        self.beat_gen = BeatGenerator(
            initial_hz=e.initial_freq_hz,
            target_hz=e.target_freq_hz,
            total_secs=bp.duration_seconds,
            carrier_hz=bp.audio_scape.binaural_base_freq_hz,
            iso_intensity=bp.audio_scape.isochronic_intensity,
            curve=e.transition_curve,
        )
        self.soundscape = SoundscapeGenerator(
            bp.audio_scape.ambient_type,
            bp.audio_scape.intensity,
        )
        self.subliminal_loop = subliminal_loop
        self._sub_cursor = 0
        self._t = 0.0
        self.total_secs = bp.duration_seconds
        self.sr = 44100

    def next_chunk(self) -> bytes:
        N = self.sr  # 1 second of samples

        binaural = self.beat_gen.next_chunk(self._t, self.sr)  # (N, 2)
        noise_m = self.soundscape.next_chunk(N)  # (N,) mono

        # Build stereo noise
        noise_l = noise_m * SOUNDSCAPE_GAIN
        noise_r = noise_m * SOUNDSCAPE_GAIN

        # Subliminal slice (mono, centre)
        if self.subliminal_loop is not None:
            loop = self.subliminal_loop
            end = self._sub_cursor + N
            if end <= len(loop):
                sub = loop[self._sub_cursor:end]
                self._sub_cursor = end
            else:
                part1 = loop[self._sub_cursor:]
                part2 = loop[:end - len(loop)]
                sub = np.concatenate([part1, part2])
                self._sub_cursor = end - len(loop)
        else:
            sub = np.zeros(N, dtype=np.float32)

        left = binaural[:, 0] * BINAURAL_GAIN + noise_l + sub * SUBLIMINAL_GAIN
        right = binaural[:, 1] * BINAURAL_GAIN + noise_r + sub * SUBLIMINAL_GAIN

        # Apply fade-in (first 2s) and fade-out (last 2s)
        remaining = self.total_secs - self._t
        fade = np.ones(N, dtype=np.float32)
        if self._t < 2.0:
            ramp_len = min(int(2.0 * self.sr), N)
            fade[:ramp_len] = np.linspace(0, 1, ramp_len)
        if remaining < 2.0:
            ramp_len = min(int(remaining * self.sr), N)
            fade[-ramp_len:] = np.linspace(1, 0, ramp_len)

        left = np.clip(left * fade, -1.0, 1.0)
        right = np.clip(right * fade, -1.0, 1.0)
        stereo = np.stack([left, right], axis=1)

        self._t += CHUNK_SECS
        return (stereo * 32767).astype(np.int16).tobytes()
