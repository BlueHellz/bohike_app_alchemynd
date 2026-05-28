import math
from app.schemas import SessionBlueprint


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def generate_mobile_frames(bp: SessionBlueprint,
                            t_start: float, t_end: float,
                            fps: float = 60.0,
                            epilepsy: bool = False) -> list[dict]:
    """
    Returns list of {t_ms, r, g, b, a} for the [t_start, t_end] window.
    Frequencies > 25 Hz are capped. Epilepsy mode: max 3 Hz smooth cosine.
    """
    mv = bp.mobile_visual
    r, g, b = hex_to_rgb(mv.color_hex)
    total = bp.duration_seconds
    e = bp.entrainment
    frames = []

    t = t_start
    dt = 1.0 / fps
    while t < t_end:
        # Compute instantaneous beat freq
        p = t / total
        k = p * p * (3 - 2 * p)  # smoothstep (mirrors ease_in_out)
        bf = e.initial_freq_hz + (e.target_freq_hz - e.initial_freq_hz) * k

        if epilepsy:
            bf = min(bf, 3.0)
            alpha = 0.3 + 0.7 * math.sin(2 * math.pi * bf * t)
        elif mv.pulse_sync and bf <= 25.0:
            alpha = 0.3 + 0.7 * math.sin(2 * math.pi * bf * t)
        elif mv.pulse_sync and bf > 25.0:
            alpha = 0.3 + 0.7 * math.sin(2 * math.pi * 25.0 * t)  # cap
        else:
            alpha = 1.0

        # Fade in/out
        fade = min(t / 2.0, 1.0, (total - t) / 2.0)
        alpha = round(abs(alpha) * fade, 4)

        frames.append({"t_ms": int(t * 1000), "r": r, "g": g, "b": b, "a": alpha})
        t += dt

    return frames
