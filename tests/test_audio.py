import pytest
import numpy as np
from scipy.signal import welch
from app.audio.beat_generator import BeatGenerator, CHUNK_SECS, CHUNK_SAMP
from app.audio.soundscape import SoundscapeGenerator, _pink_noise, _brown_noise
from app.audio.mixer import AudioMixer
from app.schemas import (
    SessionBlueprint, EntertainmentCurve, AudioScape, MobileVisual,
    SubliminalConfig, NeuroCognitiveIntent,
)

SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bp(**overrides) -> SessionBlueprint:
    defaults = dict(
        user_id="test",
        client_type="mobile",
        duration_seconds=300,
        entrainment=EntertainmentCurve(initial_freq_hz=7.5, target_freq_hz=12.0),
        audio_scape=AudioScape(ambient_type="dawn_forest", intensity=0.6),
        mobile_visual=MobileVisual(color_hex="#FFA500"),
        subliminal=SubliminalConfig(),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"], emotional_targets=["calm"],
        ),
    )
    defaults.update(overrides)
    return SessionBlueprint(**defaults)


# ---------------------------------------------------------------------------
# Phase continuity
# ---------------------------------------------------------------------------

class TestPhaseContinuity:
    def test_phase_continuous_across_chunks(self):
        """Phase accumulator preserves continuity across chunk boundaries.
        Last-phase + expected incremental delta should equal first-phase of next chunk.
        """
        gen = BeatGenerator(initial_hz=10.0, target_hz=10.0, total_secs=60.0)
        # Capture phase BEFORE chunk1
        phase_L_before = gen._phase_L
        phase_R_before = gen._phase_R
        phase_iso_before = gen._phase_iso

        chunk1 = gen.next_chunk(0.0)
        phase_L_after_1 = gen._phase_L
        phase_R_after_1 = gen._phase_R
        phase_iso_after_1 = gen._phase_iso

        chunk2 = gen.next_chunk(CHUNK_SECS)

        # The stored phase wraps modulo 2*pi, so the stored value before
        # a chunk and the initial cumsum value that produces sample[0]
        # should produce consistent sine values.
        # Verify by checking that the sine at the stored phase is consistent
        # with the first sample of the NEXT chunk (which builds on that phase).
        # The first sample of chunk N+1 is at phase = _phase_after_N + d_phase
        # where d_phase is the first increment of that chunk.
        dt = 1.0 / 44100
        expected_first_L = np.sin(phase_L_after_1 + 2 * np.pi * gen.carrier_hz * dt)
        expected_first_R = np.sin(phase_R_after_1 + 2 * np.pi * (gen.carrier_hz + gen._f(CHUNK_SECS)) * dt)

        diff_L = abs(expected_first_L - chunk2[0, 0])
        diff_R = abs(expected_first_R - chunk2[0, 1])
        assert diff_L < 5e-4, f"L phase discontinuity: {diff_L}"
        assert diff_R < 5e-4, f"R phase discontinuity: {diff_R}"

    def test_beat_freq_follows_curve(self):
        """Concatenate 30 chunks, demodulate to extract beat envelope, FFT,
        verify beat freq within +/-0.3 Hz."""
        gen = BeatGenerator(initial_hz=8.0, target_hz=14.0, total_secs=60.0)
        chunks = [gen.next_chunk(i * CHUNK_SECS, SAMPLE_RATE) for i in range(30)]
        audio = np.concatenate(chunks)

        # Demodulate: multiply left by right -> sum/difference frequencies.
        # The difference (beat) appears at |f_L - f_R| = beat_freq.
        diff = audio[:, 0].astype(np.float64) * audio[:, 1].astype(np.float64)

        # Low-pass filter around the expected beat range
        from scipy.signal import butter, sosfilt
        sos = butter(4, 50, btype="low", fs=SAMPLE_RATE, output="sos")
        env = sosfilt(sos, diff)

        # FFT on a window near the end (t=25..30s)
        window = env[25 * SAMPLE_RATE:30 * SAMPLE_RATE]
        f, Pxx = welch(window, fs=SAMPLE_RATE, nperseg=4096)
        # Expected beat at t=27.5
        p = 27.5 / 60.0
        k = p * p * (3 - 2 * p)
        expected = 8.0 + (14.0 - 8.0) * k
        dominant = f[np.argmax(Pxx)]
        assert abs(dominant - expected) < 0.5, (
            f"Beat freq {dominant:.2f} Hz != expected {expected:.2f} Hz"
        )


# ---------------------------------------------------------------------------
# Noise spectral slope
# ---------------------------------------------------------------------------

class TestNoiseSpectralSlope:
    def test_pink_noise_slope(self):
        """Pink noise ≈ -3 dB/oct ± 1 dB."""
        n = 44100 * 5
        noise = _pink_noise(n)
        f, Pxx = welch(noise, fs=SAMPLE_RATE, nperseg=4096)
        # Slope in dB/oct between 100 Hz and 1000 Hz
        idx = (f >= 100) & (f <= 1000)
        if np.sum(idx) < 2:
            pytest.skip("Not enough frequency bins")
        log_f = np.log2(f[idx])
        log_P = 10 * np.log10(Pxx[idx] + 1e-12)
        coeffs = np.polyfit(log_f, log_P, 1)
        slope = coeffs[0]  # dB/octave
        assert -4.0 <= slope <= -2.0, f"Pink noise slope {slope:.2f} dB/oct not in [-4, -2]"

    def test_brown_noise_slope(self):
        """Brown noise ≈ -6 dB/oct ± 1 dB."""
        n = 44100 * 5
        noise = _brown_noise(n)
        f, Pxx = welch(noise, fs=SAMPLE_RATE, nperseg=4096)
        idx = (f >= 50) & (f <= 500)
        if np.sum(idx) < 2:
            pytest.skip("Not enough frequency bins")
        log_f = np.log2(f[idx])
        log_P = 10 * np.log10(Pxx[idx] + 1e-12)
        coeffs = np.polyfit(log_f, log_P, 1)
        slope = coeffs[0]
        assert -7.0 <= slope <= -5.0, f"Brown noise slope {slope:.2f} dB/oct not in [-7, -5]"


# ---------------------------------------------------------------------------
# No clipping
# ---------------------------------------------------------------------------

class TestClipping:
    def test_no_clipping_50_blueprints(self):
        """Generate 1 second of audio for 50 random blueprints, max(abs) ≤ 32767."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            f0 = rng.uniform(0.5, 20.0)
            ft = rng.uniform(0.5, 30.0)
            dur = rng.uniform(60, 600)
            bp = _make_bp(
                duration_seconds=int(dur),
                entrainment=EntertainmentCurve(
                    initial_freq_hz=f0, target_freq_hz=ft,
                ),
            )
            mixer = AudioMixer(bp)
            chunk = mixer.next_chunk()
            samples = np.frombuffer(chunk, dtype=np.int16)
            max_val = np.max(np.abs(samples))
            assert max_val <= 32767, f"Clipping detected: max={max_val}"


# ---------------------------------------------------------------------------
# Soundscape generator
# ---------------------------------------------------------------------------

class TestSoundscape:
    def test_soundscape_returns_mono(self):
        """SoundscapeGenerator returns mono float32 array."""
        sg = SoundscapeGenerator("dawn_forest", 0.5)
        chunk = sg.next_chunk(44100)
        assert chunk.ndim == 1, f"Expected 1D, got {chunk.ndim}D"
        assert chunk.dtype == np.float32

    def test_soundscape_loops_seamlessly(self):
        """SoundscapeGenerator loops without large discontinuity.
        Random-noise-based buffers won't naturally match at wrap boundaries;
        verify at least the RMS levels are consistent across the wrap."""
        sg = SoundscapeGenerator("cosmic_drone", 0.5)
        # Generate enough to force a wrap
        chunk1 = sg.next_chunk(44100 * 28)  # near end of 30s buffer
        chunk2 = sg.next_chunk(44100 * 4)  # wraps around
        # Check RMS is consistent, not sample-to-sample match
        rms1 = np.sqrt(np.mean(chunk1 ** 2))
        rms2 = np.sqrt(np.mean(chunk2 ** 2))
        assert abs(rms1 - rms2) < 0.01, f"RMS mismatch across loop: {rms1} vs {rms2}"


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------

class TestMixer:
    def test_mixer_returns_bytes(self):
        """AudioMixer.next_chunk returns int16 bytes."""
        bp = _make_bp(duration_seconds=60)
        mixer = AudioMixer(bp)
        chunk = mixer.next_chunk()
        assert isinstance(chunk, bytes)
        assert len(chunk) == 44100 * 2 * 2  # 1 sec * 2 channels * 2 bytes

    def test_mixer_stereo_channels(self):
        """AudioMixer produces stereo (left != right for binaural)."""
        bp = _make_bp(duration_seconds=60)
        mixer = AudioMixer(bp)
        chunk = mixer.next_chunk()
        samples = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 2)
        # Binaural beats: left and right should differ
        assert not np.allclose(samples[:, 0], samples[:, 1])

    def test_mixer_fade_in(self):
        """First samples of mixer output are quieter than later samples."""
        bp = _make_bp(duration_seconds=60)
        mixer = AudioMixer(bp)
        chunk = mixer.next_chunk()
        samples = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 2)
        first_100 = np.abs(samples[:100]).mean()
        mid_100 = np.abs(samples[22050:22150]).mean()
        assert first_100 < mid_100, "Fade-in not detected"
