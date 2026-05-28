from app.schemas import SessionBlueprint
from typing import Tuple

FREQ_MIN = 0.5
FREQ_MAX = 40.0
DURATION_MAX = 10800        # 3 hours
FLICKER_MAX_HZ = 25.0
EPILEPSY_FLICKER_CAP = 3.0


def validate_blueprint(bp: SessionBlueprint, epilepsy: bool = False) -> Tuple[SessionBlueprint, list]:
    """
    Mutates blueprint to enforce safety constraints.
    Returns (safe_blueprint, list_of_warnings).
    Never raises -- returns warnings instead so frontend can display them.
    """
    warnings = []

    # 1. Frequency clamp
    e = bp.entrainment
    if not (FREQ_MIN <= e.initial_freq_hz <= FREQ_MAX):
        e.initial_freq_hz = max(FREQ_MIN, min(e.initial_freq_hz, FREQ_MAX))
        warnings.append(f"initial_freq_hz clamped to {e.initial_freq_hz}")
    if not (FREQ_MIN <= e.target_freq_hz <= FREQ_MAX):
        e.target_freq_hz = max(FREQ_MIN, min(e.target_freq_hz, FREQ_MAX))
        warnings.append(f"target_freq_hz clamped to {e.target_freq_hz}")

    # 2. Duration cap
    if bp.duration_seconds > DURATION_MAX:
        bp.duration_seconds = DURATION_MAX
        warnings.append("Duration capped at 3 hours.")

    # 3. Epilepsy overrides
    if epilepsy:
        bp.mobile_visual.pulse_sync = False
        bp.mobile_visual.pulse_shape = "none"
        if bp.vr_visual:
            # Force all VR elements to max 3 Hz smooth gradient
            for el in bp.vr_visual.dynamic_elements:
                el.sync_to = "slow_breath"   # frontend interprets as <=3 Hz
            bp.vr_visual.global_light_rhythm = False
        warnings.append("Epilepsy mode: all rhythmic flicker disabled.")

    return bp, warnings


def safety_summary(bp: SessionBlueprint) -> str:
    """Human-readable disclosure string shown to user before session start."""
    lines = [
        f"Duration: {bp.duration_seconds // 60} min {bp.duration_seconds % 60} sec",
        f"Soundscape: {bp.audio_scape.ambient_type}",
        f"Brainwave transition: {bp.entrainment.initial_freq_hz} Hz -> {bp.entrainment.target_freq_hz} Hz",
    ]
    if bp.subliminal.enabled:
        lines.append("Subliminal messages: " + " | ".join(f'"{m}"' for m in bp.subliminal.messages))
    else:
        lines.append("Subliminal messages: none")
    return "\n".join(lines)
