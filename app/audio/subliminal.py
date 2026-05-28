"""
Generates and processes subliminal affirmation audio.
Pre-rendered at session start -- never blocks the streaming loop.
"""
import asyncio
import numpy as np
from scipy.signal import butter, sosfilt
import tempfile
import os

TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")


async def tts_to_wav(text: str) -> np.ndarray:
    """Convert text -> float32 mono numpy array via Piper (offline) or edge-tts."""
    if TTS_ENGINE == "piper":
        proc = await asyncio.create_subprocess_exec(
            "piper", "--model", "en_US-lessac-medium.onnx",
            "--output_raw",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        raw, _ = await proc.communicate(input=text.encode())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        import edge_tts
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        await edge_tts.Communicate(text, "en-US-JennyNeural").save(path)
        import soundfile as sf
        audio, _ = sf.read(path, dtype="float32")
        os.unlink(path)
        if audio.ndim > 1:
            audio = audio[:, 0]
    return audio


def bandpass(audio: np.ndarray, lo: int = 300, hi: int = 3000, sr: int = 44100) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def normalise_to_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    return (audio * (target_rms / rms)).astype(np.float32)


async def build_subliminal_loop(messages: list[str],
                                 ambient_rms: float = 0.1) -> np.ndarray:
    """
    Returns a single loopable float32 array of all affirmations.
    Attenuation: -40 dBFS below ambient.
    """
    target_rms = ambient_rms * (10 ** (-40.0 / 20.0))
    silence = np.zeros(44100 * 1, dtype=np.float32)  # 1-sec gap
    parts = []
    for msg in messages:
        wav = await tts_to_wav(msg)
        wav = bandpass(wav)
        wav = normalise_to_rms(wav, target_rms)
        # Pad/trim to 5 seconds
        target_len = 44100 * 5
        if len(wav) < target_len:
            wav = np.concatenate([wav, np.zeros(target_len - len(wav), dtype=np.float32)])
        else:
            wav = wav[:target_len]
        parts.extend([wav, silence])
    return np.concatenate(parts)
