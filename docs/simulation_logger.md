# Session Experience Logger

The experiment logger (`scripts/simulate_sessions.py`) provides a
second-by-second terminal readout of what the user would experience during
a NeuroSync session — brainwave frequency, soundscape, light pulsing,
subliminal affirmations, and a subjective narrative description.

## Usage

```bash
# Default mode: deterministic mock (no API key needed)
python scripts/simulate_sessions.py intents.json

# Live LLM mode (requires valid OPENAI_API_KEY)
OPENAI_API_KEY=sk-... python scripts/simulate_sessions.py intents.json --live-llm
```

## Input Format

The script accepts a JSON file containing an array of intent objects:

```json
[
  {
    "user_text": "I feel anxious and I want to feel calm. Make it 5 minutes.",
    "user_id": "user_a",
    "client_type": "mobile"
  },
  {
    "user_text": "Take me to a tranquil forest in VR to deeply relax for 10 minutes.",
    "user_id": "user_d",
    "client_type": "vr"
  }
]
```

Each intent is processed through the full NeuroSync pipeline:
1. `design_session()` — generates a SessionBlueprint via LLM or mock
2. `validate_blueprint()` — enforces safety constraints
3. Component instantiation — BeatGenerator, SoundscapeGenerator, light/VR generators
4. Second-by-second simulation — logs all parameters with subjective descriptions

## Output Columns

| Column | Description |
|--------|-------------|
| `T (s)` | Elapsed time in seconds |
| `BEAT (Hz)` | Instantaneous brainwave entrainment frequency |
| `SOUNDSCAPE` | Ambient type and intensity level (0.0–1.0) |
| `LIGHT` | Colour hex, alpha brightness, pulse frequency (mobile) or VR scene/commands |
| `SUBLIMINAL` | Active affirmation message being whispered |
| `✦ ...` | Subjective narrative of the perceived experience (every 5s) |

## Mock Blueprint System

When no LLM API key is available, the script uses a keyword-matching system
to select pre-designed blueprints corresponding to common intent patterns:

| Intent Keywords | Initial→Target | Curve | Soundscape | Colour |
|----------------|---------------|-------|------------|--------|
| anxious, calm | 24→8 Hz (beta→alpha) | ease_in_out | night_rain | Blue (#4A7FB5) |
| exhausted, energised, sluggish | 6→18 Hz (theta→beta) | ease_in | mountain_wind | Orange (#FF8C00) |
| procrastinating, focus | 4→22 Hz (theta→beta) | ease_in_out | crystal_cave | Purple (#9B59B6) |
| forest, vr, relax | 20→8 Hz (beta→alpha) | ease_out | dawn_forest | Green (#2ECC71) |
| quit smoking, hypnotic | 8→3 Hz (alpha→delta) | ease_out | tibetan_bowls | Deep Purple (#8E44AD) |
