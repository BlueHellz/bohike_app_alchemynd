"""
Ambient soundscape generator.
Priority: load pre-licensed audio sample -> fallback to DSP synthesis.
Sample files live in assets/soundscapes/{ambient_type}.wav (16-bit 44100 stereo).
If file absent: synthesize from filtered noise.
"""
import numpy as np
from scipy.signal import lfilter, butter
from pathlib import Path

ASSETS = Path("assets/soundscapes")


def _pink_noise(n: int) -> np.ndarray:
    white = np.random.randn(n).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, white)


def _brown_noise(n: int) -> np.ndarray:
    white = np.random.randn(n).astype(np.float32)
    b = np.cumsum(white)
    b -= b.mean()
    return (b / (np.abs(b).max() + 1e-9)).astype(np.float32)


def _synth_bowls(n: int, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False)
    fundamental = 432.0
    harmonics = [1, 2, 3, 5, 7]
    wave = sum(np.sin(2 * np.pi * fundamental * h * t) * (1 / h) for h in harmonics)
    envelope = np.exp(-t * 0.3)
    return (wave * envelope / (np.abs(wave).max() + 1e-9)).astype(np.float32)


def _synth_waves(n: int, sr: int = 44100) -> np.ndarray:
    base = _pink_noise(n)
    t = np.linspace(0, n / sr, n, endpoint=False)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)  # 0.15 Hz wave rhythm
    return (base * lfo).astype(np.float32)


SYNTH_MAP = {
    "dawn_forest": lambda n: _pink_noise(n),
    "cosmic_drone": lambda n: _brown_noise(n),
    "tibetan_bowls": lambda n: _synth_bowls(n),
    "night_rain": lambda n: _pink_noise(n) * 0.9,
    "ocean_waves": lambda n: _synth_waves(n),
    "default": lambda n: _pink_noise(n),
}


class SoundscapeGenerator:
    def __init__(self, ambient_type: str, intensity: float):
        self.ambient_type = ambient_type
        self.intensity = intensity
        self._buffer: np.ndarray | None = None
        self._cursor = 0
        self._load()

    def _load(self):
        path = ASSETS / f"{self.ambient_type}.wav"
        if path.exists():
            import soundfile as sf
            data, _ = sf.read(str(path), dtype="float32", always_2d=True)
            self._buffer = data[:, 0]  # mono; stereo handled in mixer
        else:
            synth = SYNTH_MAP.get(self.ambient_type, SYNTH_MAP["default"])
            self._buffer = synth(44100 * 30)  # 30-sec synth buffer, will loop

    def next_chunk(self, n: int = 44100) -> np.ndarray:
        buf = self._buffer
        end = self._cursor + n
        if end <= len(buf):
            chunk = buf[self._cursor:end]
            self._cursor = end
        else:
            # Wrap around (seamless loop)
            part1 = buf[self._cursor:]
            part2 = buf[:end - len(buf)]
            chunk = np.concatenate([part1, part2])
            self._cursor = end - len(buf)
        # Apply crossfade only on wrap -- not needed here since we use cumsum
        target_rms = 0.1 * self.intensity  # -20 dBFS baseline * intensity
        rms = np.sqrt(np.mean(chunk ** 2)) + 1e-9
        return (chunk * (target_rms / rms)).astype(np.float32)
