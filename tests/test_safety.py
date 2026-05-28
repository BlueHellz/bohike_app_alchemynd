import pytest
from app.schemas import (
    SessionBlueprint, EntertainmentCurve, AudioScape, MobileVisual,
    VRVisual, VRElement, SubliminalConfig, NeuroCognitiveIntent
)
from app.safety.validator import validate_blueprint, safety_summary
from app.safety.validator import FREQ_MIN, FREQ_MAX, DURATION_MAX


def _make_bp(**overrides) -> SessionBlueprint:
    defaults = dict(
        user_id="test",
        client_type="mobile",
        duration_seconds=600,
        entrainment=EntertainmentCurve(initial_freq_hz=7.5, target_freq_hz=12.0),
        audio_scape=AudioScape(ambient_type="dawn_forest", intensity=0.6),
        mobile_visual=MobileVisual(color_hex="#FFA500"),
        subliminal=SubliminalConfig(),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"],
            emotional_targets=["calm"],
        ),
    )
    defaults.update(overrides)
    return SessionBlueprint(**defaults)


def _make_unvalidated_bp(**overrides) -> SessionBlueprint:
    """Bypass Pydantic validation to test safety clamp behavior."""
    defaults = dict(
        user_id="test",
        client_type="mobile",
        duration_seconds=600,
        entrainment=EntertainmentCurve(initial_freq_hz=7.5, target_freq_hz=12.0),
        audio_scape=AudioScape(ambient_type="dawn_forest", intensity=0.6),
        mobile_visual=MobileVisual(color_hex="#FFA500"),
        subliminal=SubliminalConfig(),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"],
            emotional_targets=["calm"],
        ),
    )
    defaults.update(overrides)
    return SessionBlueprint.model_construct(**defaults)


class TestFrequencyClamping:
    def test_freq_below_min_is_clamped(self):
        bp = _make_unvalidated_bp(entrainment=EntertainmentCurve.model_construct(
            initial_freq_hz=0.1, target_freq_hz=12.0
        ))
        safe, warnings = validate_blueprint(bp)
        assert safe.entrainment.initial_freq_hz == FREQ_MIN
        assert any("clamped" in w for w in warnings)

    def test_freq_above_max_is_clamped(self):
        bp = _make_unvalidated_bp(entrainment=EntertainmentCurve.model_construct(
            initial_freq_hz=7.5, target_freq_hz=45.0
        ))
        safe, warnings = validate_blueprint(bp)
        assert safe.entrainment.target_freq_hz == FREQ_MAX
        assert any("clamped" in w for w in warnings)

    def test_freq_in_range_unchanged(self):
        bp = _make_bp()
        safe, warnings = validate_blueprint(bp)
        assert safe.entrainment.initial_freq_hz == 7.5
        assert safe.entrainment.target_freq_hz == 12.0
        assert not any("clamped" in w for w in warnings)


class TestDurationCapping:
    def test_duration_over_max_capped(self):
        bp = _make_unvalidated_bp(duration_seconds=DURATION_MAX + 3600)
        safe, warnings = validate_blueprint(bp)
        assert safe.duration_seconds == DURATION_MAX
        assert any("capped" in w.lower() for w in warnings)

    def test_duration_under_max_unchanged(self):
        bp = _make_bp(duration_seconds=1800)
        safe, warnings = validate_blueprint(bp)
        assert safe.duration_seconds == 1800


class TestEpilepsyOverrides:
    def test_epilepsy_disables_mobile_pulse(self):
        bp = _make_bp()
        safe, _ = validate_blueprint(bp, epilepsy=True)
        assert safe.mobile_visual.pulse_sync is False
        assert safe.mobile_visual.pulse_shape == "none"

    def test_epilepsy_disables_vr_rhythm(self):
        bp = _make_bp(
            client_type="vr",
            vr_visual=VRVisual(
                scene_type="cosmic",
                color_palette=["#FF0000", "#00FF00"],
                dynamic_elements=[
                    VRElement(type="sphere", sync_to="beat", params={"size": 2.0})
                ],
            ),
        )
        safe, _ = validate_blueprint(bp, epilepsy=True)
        assert safe.vr_visual is not None
        assert safe.vr_visual.global_light_rhythm is False
        for el in safe.vr_visual.dynamic_elements:
            assert el.sync_to == "slow_breath"

    def test_epilepsy_warning_added(self):
        bp = _make_bp()
        _, warnings = validate_blueprint(bp, epilepsy=True)
        assert any("epilepsy" in w.lower() for w in warnings)

    def test_no_epilepsy_leaves_pulse_unchanged(self):
        bp = _make_bp()
        safe, _ = validate_blueprint(bp, epilepsy=False)
        assert safe.mobile_visual.pulse_sync is True
        assert safe.mobile_visual.pulse_shape == "sine"


class TestSafetySummary:
    def test_summary_without_subliminal(self):
        bp = _make_bp()
        summary = safety_summary(bp)
        assert "Duration:" in summary
        assert "Soundscape:" in summary
        assert "Brainwave transition:" in summary
        assert "Subliminal messages: none" in summary

    def test_summary_with_subliminal(self):
        bp = _make_bp(
            subliminal=SubliminalConfig(
                enabled=True,
                messages=["I am calm", "I am focused"],
            )
        )
        summary = safety_summary(bp)
        assert '"I am calm"' in summary
        assert '"I am focused"' in summary
        assert "Subliminal messages:" in summary

    def test_summary_empty_subliminal(self):
        bp = _make_bp(subliminal=SubliminalConfig())
        summary = safety_summary(bp)
        assert "Subliminal messages: none" in summary
