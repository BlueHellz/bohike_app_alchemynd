#!/usr/bin/env python3
"""
NeuroSync Session Simulator
============================
Second-by-second terminal log of the full session experience for
a list of user intents loaded from a JSON file.

Usage:
    python scripts/simulate_sessions.py intents.json
    python scripts/simulate_sessions.py intents.json --live-llm   # needs DEEPSEEK_API_KEY or OPENAI_API_KEY set
"""
import asyncio
import json
import math
import sys
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import patch, AsyncMock

# ---------------------------------------------------------------------------
# Backend imports (assumes PYTHONPATH includes project root)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import SessionBlueprint
from app.audio.beat_generator import BeatGenerator
from app.audio.soundscape import SoundscapeGenerator
from app.light.mobile_generator import generate_mobile_frames, hex_to_rgb
from app.light.vr_generator import generate_vr_commands
from app.safety.validator import validate_blueprint, safety_summary


# ---------------------------------------------------------------------------
# Mock blueprint database – deterministic blueprints keyed by intent keywords
# to enable simulation without an LLM API key.
# ---------------------------------------------------------------------------

# Mapping: intent keywords -> (initial_freq, target_freq, curve, ambient_type,
#           color_hex, subliminal_msgs, is_vr, scene_type, palette, vr_elements)
# Each entry is a pre-designed blueprint that the mock LLM returns.

MOCK_BLUEPRINTS = [
    {  # 0: Anxious -> calm, 5 min
        "match": ["anxious", "calm"],
        "initial_freq": 24.0,
        "target_freq": 8.0,
        "curve": "ease_in_out",
        "ambient_type": "night_rain",
        "intensity": 0.55,
        "color_hex": "#4A7FB5",
        "isochronic": 0.35,
        "subliminal_msgs": [],
        "is_vr": False,
        "cognitive": ["calm", "serenity"],
        "emotional": ["relaxed", "peaceful"],
        "goal": "Transition from anxiety to deep calm through alpha entrainment",
    },
    {  # 1: Exhausted -> energised, 8 min
        "match": ["exhausted", "energised", "sluggish", "sharp"],
        "initial_freq": 6.0,
        "target_freq": 18.0,
        "curve": "ease_in",
        "ambient_type": "mountain_wind",
        "intensity": 0.7,
        "color_hex": "#FF8C00",
        "isochronic": 0.5,
        "subliminal_msgs": [],
        "is_vr": False,
        "cognitive": ["energy", "alertness"],
        "emotional": ["invigorated", "awake"],
        "goal": "Rise from theta drowsiness to beta alertness",
    },
    {  # 2: Procrastination -> focus, subliminals, 6 min
        "match": ["procrastinating", "focus", "unstoppable"],
        "initial_freq": 4.0,
        "target_freq": 22.0,
        "curve": "ease_in_out",
        "ambient_type": "crystal_cave",
        "intensity": 0.6,
        "color_hex": "#9B59B6",
        "isochronic": 0.45,
        "subliminal_msgs": ["I am unstoppable.", "I finish what I start."],
        "is_vr": False,
        "cognitive": ["focus", "concentration"],
        "emotional": ["determined", "empowered"],
        "goal": "Deep theta-to-beta transition for focused work",
    },
    {  # 3: VR forest relax, 10 min
        "match": ["forest", "vr", "tranquil", "relax"],
        "initial_freq": 20.0,
        "target_freq": 8.0,
        "curve": "ease_out",
        "ambient_type": "dawn_forest",
        "intensity": 0.5,
        "color_hex": "#2ECC71",
        "isochronic": 0.3,
        "subliminal_msgs": [],
        "is_vr": True,
        "scene_type": "tranquil_forest_clearing",
        "palette": ["#2ECC71", "#8B4513", "#87CEEB", "#F5DEB3"],
        "vr_elements": [
            {"type": "fireflies", "sync_to": "slow_breath", "params": {"count": 60, "radius": 8.0}},
            {"type": "leaves", "sync_to": "gentle_wind", "params": {"sway_speed": 0.3}},
            {"type": "sunbeams", "sync_to": "beat", "params": {"opacity_range": [0.2, 0.6]}},
        ],
        "cognitive": ["relaxation", "presence"],
        "emotional": ["tranquil", "grounded"],
        "goal": "Immersive forest relaxation with beta-to-alpha wind-down",
    },
    {  # 4: Quit smoking hypnosis, 15 min
        "match": ["quit smoking", "hypnotic", "subconscious", "cigarettes"],
        "initial_freq": 8.0,
        "target_freq": 3.0,
        "curve": "ease_out",
        "ambient_type": "tibetan_bowls",
        "intensity": 0.45,
        "color_hex": "#8E44AD",
        "isochronic": 0.3,
        "subliminal_msgs": [
            "I am completely free from cigarettes.",
            "I breathe pure air.",
        ],
        "is_vr": False,
        "cognitive": ["subconscious_reprogramming"],
        "emotional": ["peaceful", "liberated"],
        "goal": "Deep theta/delta hypnosis for addiction release",
    },
]


def _find_mock_blueprint(user_text: str) -> dict:
    """Find the best matching mock blueprint by keyword overlap."""
    text_lower = user_text.lower()
    best = MOCK_BLUEPRINTS[0]  # default
    best_score = 0
    for bp in MOCK_BLUEPRINTS:
        score = sum(1 for kw in bp["match"] if kw in text_lower)
        if score > best_score:
            best_score = score
            best = bp
    return best


def _build_mock_llm_response(user_text: str, client_type: str) -> str:
    """Build deterministic blueprint JSON like the LLM would return."""
    mock = _find_mock_blueprint(user_text)

    # Extract subliminal messages from user text if present
    subliminal_msgs = list(mock["subliminal_msgs"])
    # If user provided their own quoted words, use those instead
    import re
    quoted = re.findall(r'"([^"]+)"', user_text)
    if quoted:
        subliminal_msgs = quoted

    duration = _extract_duration(user_text)

    bp = {
        "client_type": client_type,
        "duration_seconds": duration,
        "entrainment": {
            "initial_freq_hz": mock["initial_freq"],
            "target_freq_hz": mock["target_freq"],
            "transition_curve": mock["curve"],
            "custom_points": None,
        },
        "audio_scape": {
            "ambient_type": mock["ambient_type"],
            "intensity": mock["intensity"],
            "binaural_base_freq_hz": 200.0,
            "isochronic_intensity": mock["isochronic"],
            "spatial_audio_model": "ambisonics" if mock["is_vr"] else "stereo",
        },
        "mobile_visual": {
            "color_hex": mock["color_hex"],
            "pulse_sync": True,
            "pulse_shape": "sine",
        },
        "subliminal": {
            "enabled": len(subliminal_msgs) > 0,
            "messages": subliminal_msgs,
        },
        "neuro_cognitive_intent": {
            "cognitive_targets": mock["cognitive"],
            "emotional_targets": mock["emotional"],
            "subconscious_goal": mock["goal"],
        },
    }

    if mock["is_vr"]:
        bp["vr_visual"] = {
            "scene_type": mock["scene_type"],
            "color_palette": mock["palette"],
            "dynamic_elements": mock["vr_elements"],
            "global_light_rhythm": True,
        }
    else:
        bp["vr_visual"] = None

    return json.dumps(bp)


def _extract_duration(text: str) -> int:
    """Extract duration in seconds from natural language."""
    import re
    # Pattern: "N minutes" or "N min"
    m = re.search(r"(\d+)\s*(?:minutes?|min|mins)", text, re.IGNORECASE)
    if m:
        return int(m.group(1)) * 60
    # Pattern: "N seconds"
    m = re.search(r"(\d+)\s*(?:seconds?|sec|secs)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Default
    return 360


# ---------------------------------------------------------------------------
# Subjective description generator (template-based, no LLM)
# ---------------------------------------------------------------------------

SOUNDSCAPE_DESCRIPTIONS = {
    "night_rain": "a steady, soft rainfall on leaves",
    "dawn_forest": "birdsong filtered through morning mist",
    "ocean_waves": "rhythmic surf on a distant shore",
    "tibetan_bowls": "resonant singing bowls vibrating through the air",
    "cosmic_drone": "a deep, interstellar hum",
    "theta_hum": "a low, meditative drone at the edge of hearing",
    "crystal_cave": "crystalline chimes echoing through a vast cavern",
    "mountain_wind": "wind sweeping across high alpine ridges",
    "deep_space": "the profound silence of the cosmos",
    "campfire": "crackling logs and warm embers",
}

PULSE_SHAPES = {
    "sine": "smooth sinusoidal pulse",
    "none": "steady, non-pulsing glow",
}


def describe_soundscape(ambient_type: str, intensity: float) -> str:
    desc = SOUNDSCAPE_DESCRIPTIONS.get(ambient_type, f"a {ambient_type.replace('_', ' ')} ambiance")
    if intensity < 0.3:
        desc = f"barely audible, {desc}"
    elif intensity > 0.8:
        desc = f"richly present, {desc}"
    return desc


def describe_beat(freq: float, initial: float, target: float) -> str:
    bands = {
        (0.5, 4.0): "delta (deep sleep / hypnosis)",
        (4.0, 8.0): "theta (meditation / hypnagogic)",
        (8.0, 12.0): "alpha (relaxed focus / calm)",
        (12.0, 30.0): "beta (active thinking / alertness)",
        (30.0, 100.0): "gamma (peak performance / insight)",
    }
    band = "unknown"
    for (lo, hi), label in sorted(bands.items()):
        if lo <= freq < hi:
            band = label
            break
    progress = (freq - initial) / (target - initial + 1e-9)
    if progress < 0.3:
        phase = "starting phase"
    elif progress < 0.6:
        phase = "mid-transition"
    else:
        phase = "arriving at target state"
    return f"{freq:.1f} Hz ({band}) — {phase}"


def describe_light_alpha(alpha: float, color_hex: str, pulse_freq: float) -> str:
    r, g, b = hex_to_rgb(color_hex)
    if alpha < 0.1:
        return "practically dark"
    br = "bright" if alpha > 0.7 else "soft" if alpha > 0.4 else "dim"
    return f"{br} #{color_hex[1:]} glow ({r},{g},{b}) pulsing at {pulse_freq:.1f} Hz"


def describe_subliminal(message: Optional[str]) -> str:
    if message:
        return f"whispered: \"{message}\""
    return ""


def build_subjective_line(t: float, beat_freq: float, initial_hz: float,
                           target_hz: float, ambient_type: str, intensity: float,
                           color_hex: str, pulse_freq: float, alpha: float,
                           subliminal_msg: Optional[str] = None) -> str:
    parts = []

    # Sound
    sound = describe_soundscape(ambient_type, intensity)
    parts.append(sound)

    # Beat
    beat = describe_beat(beat_freq, initial_hz, target_hz)
    parts.append(beat)

    # Light
    light = describe_light_alpha(alpha, color_hex, pulse_freq)
    parts.append(light)

    base = "; ".join(parts) + "."

    if subliminal_msg:
        base += f" At the threshold of hearing, a voice {subliminal_msg}"

    return base


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

async def simulate_intent(intent: dict, use_live_llm: bool = False):
    """Run a single session simulation for one user intent."""
    user_text = intent["user_text"]
    user_id = intent["user_id"]
    client_type = intent.get("client_type", "mobile")

    # ── Step A: Design session (mocked or live) ──────────────────────────
    if use_live_llm:
        # Auto-detect provider: prefer DeepSeek, fallback to OpenAI
        if os.getenv("DEEPSEEK_API_KEY"):
            os.environ["LLM_PROVIDER"] = "deepseek"
        elif os.getenv("OPENAI_API_KEY"):
            os.environ["LLM_PROVIDER"] = "openai"
        else:
            print("  [SKIP]  No DEEPSEEK_API_KEY or OPENAI_API_KEY set. Use mock mode or set a key.")
            return
        from app.ai.session_designer import design_session
        resp = await design_session(user_text, user_id, client_type)
    else:
        from app.ai.session_designer import design_session
        mock_response = _build_mock_llm_response(user_text, client_type)
        mock_llm = AsyncMock(return_value=mock_response)
        with patch("app.ai.session_designer.call_llm", mock_llm):
            resp = await design_session(user_text, user_id, client_type)

    if resp.requires_duration:
        print("  [SKIP]  LLM could not determine duration. Skipping intent.")
        return

    bp = resp.blueprint
    if bp is None:
        print("  [SKIP]  No blueprint returned. Skipping intent.")
        return

    # ── Step B: Safety validation ────────────────────────────────────────
    bp, warnings = validate_blueprint(bp)
    summary = safety_summary(bp)

    # ── Step C: Instantiate generators ───────────────────────────────────
    e = bp.entrainment
    beat_gen = BeatGenerator(
        initial_hz=e.initial_freq_hz,
        target_hz=e.target_freq_hz,
        total_secs=bp.duration_seconds,
        curve=e.transition_curve,
        iso_intensity=bp.audio_scape.isochronic_intensity,
    )
    soundscape = SoundscapeGenerator(
        bp.audio_scape.ambient_type,
        bp.audio_scape.intensity,
    )

    total = bp.duration_seconds
    subliminal_msgs = bp.subliminal.messages if bp.subliminal.enabled else []
    sub_idx = 0

    # ── Step D: Print session header ─────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print()
    print("=" * 78)
    print(f"  SESSION: {user_id}")
    print(f"  Intent:  {user_text}")
    print(f"  Time:    {now}")
    print(f"  Client:  {client_type.upper()}")
    print(f"  Duration: {total // 60} min {total % 60} sec")
    print("-" * 78)
    print(f"  {summary.replace(chr(10), chr(10) + '  ')}")
    if warnings:
        print(f"  [SAFETY] {'; '.join(warnings)}")
    print("-" * 78)
    print(f"  {'T':>5}  {'BEAT':>7}  {'SOUNDSCAPE':<28}  {'LIGHT':<28}  {'SUBLIMINAL':<22}")
    print(f"  {'(s)':>5}  {'(Hz)':>7}  {'':28}  {'':28}  {'':22}")
    print("-" * 78)

    # ── Step E: Second-by-second simulation loop ─────────────────────────
    for t_sec in range(total):
        beat_freq = beat_gen._f(t_sec)

        # Mobile light state at this second
        if client_type == "mobile":
            frames = generate_mobile_frames(bp, float(t_sec), float(t_sec + 1), fps=1)
            if frames:
                f0 = frames[0]
                alpha = f0["a"]
                r, g, b = f0["r"], f0["g"], f0["b"]
            else:
                alpha, r, g, b = 0.0, 0, 0, 0

            # Compute effective pulse frequency (capped at 25 Hz)
            p = t_sec / total
            k = p * p * (3 - 2 * p)
            pulse_freq = e.initial_freq_hz + (e.target_freq_hz - e.initial_freq_hz) * k
            pulse_freq = min(pulse_freq, 25.0)

            light_str = f"#{bp.mobile_visual.color_hex.lstrip('#'):<6} a={alpha:.3f} @{pulse_freq:.1f}Hz"
        else:
            # VR: get commands for this second
            cmds = generate_vr_commands(bp, float(t_sec), float(t_sec + 1))
            active = [c["cmd"] for c in cmds if c["t_ms"] >= t_sec * 1000]
            scene = ""
            for c in cmds:
                if c["cmd"] == "load_scene":
                    scene = c.get("scene_type", "")
            light_str = f"VR scene: {scene}" if scene else f"VR cmds: {len(active)}"
            alpha = 0.5

        # Subliminal message rotation (one message per ~5 secs)
        sub_msg = None
        if subliminal_msgs:
            if t_sec % 5 == 0:
                sub_idx = (t_sec // 5) % len(subliminal_msgs)
                sub_msg = subliminal_msgs[sub_idx]
        sub_str = sub_msg[:40] + "..." if sub_msg and len(sub_msg) > 40 else (sub_msg or "")
        sub_str = sub_str.replace('"', "'")

        # Soundscape intensity varies with time
        scape_intensity = soundscape.intensity

        # Build attentive description every 5 seconds
        if t_sec % 5 == 0:
            subjective = build_subjective_line(
                t_sec, beat_freq, e.initial_freq_hz, e.target_freq_hz,
                bp.audio_scape.ambient_type, scape_intensity,
                bp.mobile_visual.color_hex, pulse_freq if client_type == "mobile" else 0,
                alpha, sub_msg,
            )
        else:
            subjective = ""

        # Print log line
        beat_str = f"{beat_freq:>7.2f}"
        amb_str = f"{bp.audio_scape.ambient_type:<14} {scape_intensity:.2f}"
        print(f"  {t_sec:>5}  {beat_str}  {amb_str:<28}  {light_str:<28}  {sub_str:<22}")
        if subjective:
            print(f"          {chr(10038)} {subjective}")

        # Small delay for readable output (only if running interactively)
        if sys.stdout.isatty() and t_sec < 10:
            time.sleep(0.02)

    # ── Step F: Session complete ────────────────────────────────────────
    print(f"  {total:>5}  {'─'*7}  {'─'*28}  {'─'*28}  {'─'*22}")
    print(f"  {'END':>5}  Session complete — fade-out applied.")
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <intents.json> [--live-llm]")
        sys.exit(1)

    intents_path = Path(sys.argv[1])
    if not intents_path.exists():
        print(f"Error: intents file not found: {intents_path}")
        sys.exit(1)

    use_live_llm = "--live-llm" in sys.argv

    with open(intents_path) as f:
        intents = json.load(f)

    print()
    print("  ╔══════════════════════════════════════════════════════════════════════╗")
    print("  ║              NEUROSYNC v3 — Session Experience Simulator            ║")
    print("  ║  Second-by-second log of brainwave entrainment, soundscape, light,  ║")
    print("  ║  and subliminal affirmation delivery for each user intent.          ║")
    print("  ╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Loaded {len(intents)} intents from {intents_path.name}")
    if use_live_llm:
        print("  Mode: LIVE LLM (requires valid OPENAI_API_KEY)")
    else:
        print("  Mode: deterministic mock (no API key needed)")
    print()

    for i, intent in enumerate(intents):
        print(f"[{i + 1}/{len(intents)}] Processing intent: {intent['user_id']}")
        await simulate_intent(intent, use_live_llm)

    print("  All sessions simulated.")


if __name__ == "__main__":
    asyncio.run(main())
