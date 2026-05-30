import os
from app.schemas import SessionBlueprint

FREQ_MIN = float(os.getenv("MIN_FREQ_HZ", "0.5"))
FREQ_MAX = float(os.getenv("MAX_FREQ_HZ", "40.0"))
SESSION_MAX = int(os.getenv("MAX_SESSION_MINUTES", "180")) * 60


def validate(bp: SessionBlueprint, epilepsy: bool = False) -> tuple[SessionBlueprint, list[str]]:
    warnings = []

    bp.entrainment.initial_hz = _clamp(bp.entrainment.initial_hz, FREQ_MIN, FREQ_MAX, "initial_hz", warnings)
    bp.entrainment.target_hz = _clamp(bp.entrainment.target_hz, FREQ_MIN, FREQ_MAX, "target_hz", warnings)
    if bp.duration_seconds > SESSION_MAX:
        bp.duration_seconds = SESSION_MAX
        warnings.append(f"Duration capped at {SESSION_MAX // 60} minutes.")

    if epilepsy:
        bp.mobile_visual.pulse_sync = False
        bp.mobile_visual.pulse_shape = "none"
        if bp.vr_visual:
            for el in bp.vr_visual.dynamic_elements:
                el.sync_to = "slow_breath"
            bp.vr_visual.global_light_rhythm = False
        warnings.append("Epilepsy mode: all flicker disabled.")

    if bp.subliminal.enabled and not bp.subliminal.messages:
        bp.subliminal.enabled = False
        warnings.append("Subliminal disabled: no user-provided messages.")

    noise_sum = sum(bp.audio.noise_blend)
    if abs(noise_sum - 1.0) > 0.01:
        bp.audio.noise_blend = [w / noise_sum for w in bp.audio.noise_blend]
        warnings.append("Noise blend weights normalised to sum 1.0.")

    return bp, warnings


def safety_summary(bp: SessionBlueprint) -> str:
    lines = [
        f"Duration: {bp.duration_seconds // 60} min {bp.duration_seconds % 60} sec",
        f"Goal: {bp.intent_summary}",
        f"Entrainment: {bp.entrainment.initial_hz} Hz -> {bp.entrainment.target_hz} Hz",
        f"Active modalities: {', '.join(bp.active_modalities)}",
    ]
    if bp.subliminal.enabled:
        lines.append("Subliminal: " + " | ".join(f'"{m}"' for m in bp.subliminal.messages))
    else:
        lines.append("Subliminal: none")
    return "\n".join(lines)


def _clamp(val, lo, hi, name, warnings):
    if not (lo <= val <= hi):
        clamped = max(lo, min(val, hi))
        warnings.append(f"{name} clamped from {val} to {clamped}")
        return clamped
    return val
