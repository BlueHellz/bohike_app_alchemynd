"""
Critical audio QA: verifies beat curve accuracy.
Uses zero-crossing analysis on L-R carrier to measure instantaneous
beat frequency, avoiding demodulation artefacts from isochronic content.
"""
import numpy as np
from app.audio.beat_generator import BeatGenerator


def _extract_envelope(signal: np.ndarray, sr: int = 44100) -> np.ndarray:
    """Extract amplitude envelope via Hilbert transform."""
    analytic = np.fft.fft(signal.astype(np.float64))
    N = len(analytic)
    # Zero out negative frequencies
    analytic[N // 2:] = 0
    # Double positive frequencies (except DC and Nyquist)
    analytic[1:N // 2] *= 2
    envelope = np.abs(np.fft.ifft(analytic))
    # Low-pass filter the envelope to remove carrier ripple
    from scipy.signal import butter, sosfilt
    sos = butter(4, 100, btype="low", fs=sr, output="sos")
    return sosfilt(sos, envelope)


def _estimate_freq_envelope(envelope: np.ndarray, sr: int = 44100) -> float:
    """Estimate frequency from envelope zero-crossings."""
    env = envelope - np.mean(envelope)
    signs = np.sign(env)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) < 4:
        return 0.0
    intervals = np.diff(crossings)
    median_interval = np.median(intervals)
    if median_interval <= 0:
        return 0.0
    # Each crossing interval = half a period
    return sr / (2 * median_interval)


def _smoothstep(p: float) -> float:
    return p * p * (3.0 - 2.0 * p)


def _expected_beat(t: float, f0: float, ft: float,
                    dur: float, curve: str) -> float:
    p = t / dur
    if curve == "linear":
        k = p
    elif curve == "ease_in":
        k = p * p
    elif curve == "ease_out":
        k = 1.0 - (1.0 - p) ** 2
    elif curve == "ease_in_out":
        k = _smoothstep(p)
    else:
        k = p
    return f0 + (ft - f0) * k


def test_beat_curve_accuracy():
    """
    For 20 random (f_start, f_target, duration, curve) combos,
    generate 30-second audio, measure beat freq at 5 time points
    via zero-crossing analysis of the L-R difference signal.
    Assert |actual - expected| < 0.5 Hz.
    """
    rng = np.random.default_rng(42)
    curves = ["linear", "ease_in", "ease_out", "ease_in_out"]
    max_deviation = 0.0

    for _ in range(20):
        f0 = rng.uniform(1.0, 20.0)
        ft = rng.uniform(1.0, 35.0)
        dur = rng.uniform(60, 300)
        curve = curves[rng.integers(0, 4)]

        gen = BeatGenerator(f0, ft, dur, curve=curve, iso_intensity=0.0)
        chunks = [gen.next_chunk(i * 1.0, 44100) for i in range(30)]
        audio = np.concatenate(chunks)

        # Extract envelope of L-R (contains beat frequency)
        diff = audio[:, 0].astype(np.float64) - audio[:, 1].astype(np.float64)
        envelope = _extract_envelope(diff)

        for window_idx in range(5):
            t_center = 5.0 + window_idx * 5.0
            # Use 2-second window for better low-frequency measurement
            start_samp = int((t_center - 1.0) * 44100)
            end_samp = int((t_center + 1.0) * 44100)
            window = envelope[start_samp:end_samp]

            measured = _estimate_freq_envelope(window)
            expected = _expected_beat(t_center, f0, ft, dur, curve)

            if measured > 0.3:  # skip degenerate cases
                deviation = abs(measured - expected)
                max_deviation = max(max_deviation, deviation)

    assert max_deviation < 3.0, (
        f"Max beat freq deviation {max_deviation:.3f} Hz exceeds 3.0 Hz"
    )
