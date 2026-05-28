import numpy as np
import pytest
import math
from app.schemas import (
    SessionBlueprint, EntertainmentCurve, AudioScape, MobileVisual,
    VRVisual, VRElement, SubliminalConfig, NeuroCognitiveIntent,
)
from app.light.mobile_generator import generate_mobile_frames, hex_to_rgb
from app.light.vr_generator import generate_vr_commands


def _make_mobile_bp(**overrides) -> SessionBlueprint:
    defaults = dict(
        user_id="test",
        client_type="mobile",
        duration_seconds=120,
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


def _make_vr_bp(**overrides) -> SessionBlueprint:
    defaults = dict(
        user_id="test",
        client_type="vr",
        duration_seconds=120,
        entrainment=EntertainmentCurve(initial_freq_hz=7.5, target_freq_hz=12.0),
        audio_scape=AudioScape(ambient_type="cosmic_drone", intensity=0.6),
        mobile_visual=MobileVisual(color_hex="#FFA500"),
        vr_visual=VRVisual(
            scene_type="cosmic",
            color_palette=["#FF0000", "#00FF00", "#0000FF"],
            dynamic_elements=[
                VRElement(type="sphere", sync_to="beat", params={"size": 2.0}),
                VRElement(type="particles", sync_to="beat", params={"count": 100}),
            ],
            global_light_rhythm=True,
        ),
        subliminal=SubliminalConfig(),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"], emotional_targets=["calm"],
        ),
    )
    defaults.update(overrides)
    return SessionBlueprint(**defaults)


# ---------------------------------------------------------------------------
# hex_to_rgb
# ---------------------------------------------------------------------------

class TestHexToRgb:
    def test_hex_with_hash(self):
        assert hex_to_rgb("#FFA500") == (255, 165, 0)

    def test_hex_without_hash(self):
        assert hex_to_rgb("FF00FF") == (255, 0, 255)

    def test_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert hex_to_rgb("#FFFFFF") == (255, 255, 255)


# ---------------------------------------------------------------------------
# Mobile frames
# ---------------------------------------------------------------------------

class TestMobileFrames:
    def test_tms_monotonically_increasing(self):
        """t_ms values must be strictly increasing."""
        bp = _make_mobile_bp(duration_seconds=120)
        frames = generate_mobile_frames(bp, 0, 10)
        t_ms = [f["t_ms"] for f in frames]
        assert all(t_ms[i] < t_ms[i + 1] for i in range(len(t_ms) - 1)), \
            "t_ms not monotonically increasing"

    def test_alpha_in_range(self):
        """Alpha values must be in [0, 1]."""
        bp = _make_mobile_bp(duration_seconds=120)
        frames = generate_mobile_frames(bp, 0, 10)
        for f in frames:
            assert 0.0 <= f["a"] <= 1.0, f"Alpha {f['a']} out of range"

    def test_zero_crossings_match_beat_freq(self):
        """10 sec at 10 Hz -> FFT dominant freq close to 10 Hz."""
        from scipy.signal import welch
        bp = _make_mobile_bp(
            duration_seconds=120,
            entrainment=EntertainmentCurve(
                initial_freq_hz=10.0, target_freq_hz=10.0, transition_curve="linear",
            ),
        )
        frames = generate_mobile_frames(bp, 0, 10, fps=60.0)
        alphas = np.array([f["a"] for f in frames], dtype=np.float64)
        f, Pxx = welch(alphas, fs=60.0, nperseg=128)
        dominant = f[np.argmax(Pxx)]
        assert abs(dominant - 10.0) < 2.0, f"Dominant freq {dominant:.2f} Hz != expected ~10 Hz"

    def test_high_freq_capped_at_25hz(self):
        """Beat > 25 Hz should cap alpha oscillation at 25 Hz."""
        from scipy.signal import welch
        bp = _make_mobile_bp(
            duration_seconds=120,
            entrainment=EntertainmentCurve(
                initial_freq_hz=40.0, target_freq_hz=40.0, transition_curve="linear",
            ),
        )
        frames = generate_mobile_frames(bp, 0, 10, fps=60.0)
        alphas = np.array([f["a"] for f in frames], dtype=np.float64)
        f, Pxx = welch(alphas, fs=60.0, nperseg=128)
        dominant = f[np.argmax(Pxx)]
        assert abs(dominant - 25.0) < 2.0, f"Dominant freq {dominant:.2f} Hz != expected ~25 Hz (cap)"

    def test_epilepsy_caps_at_3hz(self):
        """Epilepsy mode: alpha oscillation must not exceed ~3 Hz."""
        from scipy.signal import welch
        bp = _make_mobile_bp(
            duration_seconds=120,
            entrainment=EntertainmentCurve(
                initial_freq_hz=20.0, target_freq_hz=20.0, transition_curve="linear",
            ),
        )
        frames = generate_mobile_frames(bp, 0, 10, fps=60.0, epilepsy=True)
        alphas = np.array([f["a"] for f in frames], dtype=np.float64)
        f, Pxx = welch(alphas, fs=60.0, nperseg=128)
        dominant = f[np.argmax(Pxx)]
        assert dominant <= 4.0, f"Epilepsy dominant freq {dominant:.2f} Hz exceeds 3 Hz cap"

    def test_no_pulse_returns_constant_alpha(self):
        """When pulse_sync=False, alpha should be 1.0 (minus fade)."""
        bp = _make_mobile_bp(
            mobile_visual=MobileVisual(color_hex="#FFA500", pulse_sync=False, pulse_shape="none"),
        )
        frames = generate_mobile_frames(bp, 2, 8, fps=10.0)  # skip fade zone
        for f in frames:
            assert abs(f["a"] - 1.0) < 0.01, f"Alpha {f['a']} != 1.0"

    def test_rgb_consistency(self):
        """All frames share the same r, g, b values."""
        bp = _make_mobile_bp(duration_seconds=120)
        frames = generate_mobile_frames(bp, 0, 10)
        r, g, b = frames[0]["r"], frames[0]["g"], frames[0]["b"]
        for f in frames:
            assert f["r"] == r
            assert f["g"] == g
            assert f["b"] == b


# ---------------------------------------------------------------------------
# VR commands
# ---------------------------------------------------------------------------

class TestVRCommands:
    def test_load_scene_at_t0(self):
        """load_scene command appears exactly once at t=0."""
        bp = _make_vr_bp()
        cmds = generate_vr_commands(bp, 0, 10)
        load_scenes = [c for c in cmds if c["cmd"] == "load_scene"]
        assert len(load_scenes) == 1
        assert load_scenes[0]["t_ms"] == 0

    def test_update_element_commands(self):
        """update_element commands follow load_scene."""
        bp = _make_vr_bp()
        cmds = generate_vr_commands(bp, 0, 5)
        assert any(c["cmd"] == "update_element" for c in cmds)
        for c in cmds:
            if c["cmd"] == "update_element":
                assert "type" in c
                assert "sync_val" in c
                assert "params" in c

    def test_vr_epilepsy_limits_frequency(self):
        """Epilepsy mode: all update_element sync_val oscillations <= 3 Hz."""
        bp = _make_vr_bp(
            entrainment=EntertainmentCurve(
                initial_freq_hz=20.0, target_freq_hz=20.0, transition_curve="linear",
            ),
        )
        cmds = generate_vr_commands(bp, 0, 10, epilepsy=True)
        # Extract sync_val sequence for the first element type
        sphere_vals = [c["sync_val"] for c in cmds if c.get("type") == "sphere"]
        if len(sphere_vals) > 2:
            crossings = 0
            for i in range(1, len(sphere_vals)):
                if (sphere_vals[i - 1] - 0.5) * (sphere_vals[i] - 0.5) < 0:
                    crossings += 1
            # At 3 Hz cap -> ~6 crossings per 10 sec
            assert crossings <= 10, f"VR epilepsy cap exceeded: {crossings}"

    def test_vr_no_vr_visual_returns_empty(self):
        """Mobile blueprint with no VR visual returns empty list."""
        bp = _make_mobile_bp()
        cmds = generate_vr_commands(bp, 0, 10)
        assert cmds == []

    def test_set_global_light_present(self):
        """VR with global_light_rhythm generates set_global_light commands."""
        bp = _make_vr_bp()
        cmds = generate_vr_commands(bp, 0, 5)
        assert any(c["cmd"] == "set_global_light" for c in cmds)
