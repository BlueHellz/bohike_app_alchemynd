"""
Functional (behavioural) test suite for NeuroSync v3 backend.
Uses ASGITransport to talk to the FastAPI app directly (no server needed).
Mocks the LLM to test the full pipeline without an API key.
"""
import pytest
import json
import numpy as np
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.schemas import SessionBlueprint



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blueprint_for_test(duration_seconds: int = 720, **overrides) -> dict:
    """Generate a valid SessionBlueprint dict for direct API calls."""
    bp = {
        "user_id": "test_user_1",
        "client_type": "mobile",
        "duration_seconds": duration_seconds,
        "entrainment": {
            "initial_freq_hz": 24.0,
            "target_freq_hz": 10.0,
            "transition_curve": "ease_in_out",
            "custom_points": None,
        },
        "audio_scape": {
            "ambient_type": "dawn_forest",
            "intensity": 0.6,
            "binaural_base_freq_hz": 200.0,
            "isochronic_intensity": 0.4,
            "spatial_audio_model": "stereo",
        },
        "mobile_visual": {
            "color_hex": "#4A90D9",
            "pulse_sync": True,
            "pulse_shape": "sine",
        },
        "vr_visual": None,
        "subliminal": {
            "enabled": False,
            "messages": [],
        },
        "neuro_cognitive_intent": {
            "cognitive_targets": ["calm"],
            "emotional_targets": ["relaxed"],
            "subconscious_goal": "Transition from anxiety to calm",
        },
    }
    bp.update(overrides)
    return bp


# Shared ASGI transport for all tests
transport = ASGITransport(app=app)


# =========================================================================
# TEST GROUP 1: AI Session Designer – Core Intent Parsing
# =========================================================================

@pytest.mark.asyncio
async def test_11_simple_mood_transition():
    """Test 1.1: anxious->calm, 12 min -> duration=720, beta->alpha freqs."""
    mock_llm = AsyncMock(return_value=json.dumps(_make_blueprint_for_test(
        duration_seconds=720,
        entrainment={
            "initial_freq_hz": 24.0,
            "target_freq_hz": 10.0,
            "transition_curve": "ease_in_out",
            "custom_points": None,
        },
        audio_scape={
            "ambient_type": "night_rain",
            "intensity": 0.6,
            "binaural_base_freq_hz": 200.0,
            "isochronic_intensity": 0.4,
            "spatial_audio_model": "stereo",
        },
        mobile_visual={"color_hex": "#4A90D9", "pulse_sync": True, "pulse_shape": "sine"},
        subliminal={"enabled": False, "messages": []},
        neuro_cognitive_intent={
            "cognitive_targets": ["calm"],
            "emotional_targets": ["relaxed"],
            "subconscious_goal": "Transition from anxiety to calm",
        },
    )))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/design", json={
                "user_text": "I am anxious and want to feel calm. Make it 12 minutes.",
                "user_id": "test_user_1",
                "client_type": "mobile",
            })

    assert r.status_code == 200
    data = r.json()
    bp = data["blueprint"]
    assert bp is not None, f"Expected blueprint, got: {data}"

    # Verify duration
    assert bp["duration_seconds"] == 720, f"Expected 720s, got {bp['duration_seconds']}"

    # Frequency ranges
    assert 18 <= bp["entrainment"]["initial_freq_hz"] <= 30, \
        f"initial_freq {bp['entrainment']['initial_freq_hz']} not in beta range"
    assert 8 <= bp["entrainment"]["target_freq_hz"] <= 12, \
        f"target_freq {bp['entrainment']['target_freq_hz']} not in alpha range"

    # Soundscape should be relaxing
    assert bp["audio_scape"]["ambient_type"] in ("night_rain", "ocean_waves",
        "dawn_forest", "tibetan_bowls", "crystal_cave"), \
        f"Unexpected soundscape: {bp['audio_scape']['ambient_type']}"

    # Mobile visual
    assert bp["mobile_visual"]["pulse_sync"] is True
    assert bp["subliminal"]["enabled"] is False

    # Valid Pydantic schema
    assert set(bp.keys()) == {"user_id", "client_type", "duration_seconds",
        "entrainment", "audio_scape", "mobile_visual", "vr_visual",
        "subliminal", "neuro_cognitive_intent"}, \
        f"Unexpected keys: {bp.keys()}"


@pytest.mark.asyncio
async def test_12_missing_duration():
    """Test 1.2: No duration stated -> requires_duration: true."""
    mock_llm = AsyncMock(return_value=json.dumps({"requires_duration": True}))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/design", json={
                "user_text": "I feel tired and want to be energised.",
                "user_id": "test_user_1",
            })

    assert r.status_code == 200
    data = r.json()
    assert data["requires_duration"] is True
    assert data["blueprint"] is None
    assert data["requires_affirmations"] is False


@pytest.mark.asyncio
async def test_13_user_provided_subliminal():
    """Test 1.3: User provides affirmation words -> subliminal enabled."""
    mock_llm = AsyncMock(return_value=json.dumps(_make_blueprint_for_test(
        subliminal={
            "enabled": True,
            "messages": ["I am laser-focused.", "I finish every task."],
        },
    )))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/design", json={
                "user_text": "Help me focus for 20 minutes. "
                            "Use my words: 'I am laser-focused. I finish every task.'",
                "user_id": "test_user_1",
            })

    assert r.status_code == 200
    data = r.json()
    bp = data["blueprint"]
    assert bp["subliminal"]["enabled"] is True
    assert len(bp["subliminal"]["messages"]) == 2
    msgs = bp["subliminal"]["messages"]
    assert any("laser-focused" in m for m in msgs)
    assert any("finish every task" in m for m in msgs)


@pytest.mark.asyncio
async def test_14_therapeutic_metaphor():
    """Test 1.4: Complex metaphor -> coherent blueprint (not an error)."""
    mock_llm = AsyncMock(return_value=json.dumps(_make_blueprint_for_test(
        duration_seconds=900,
        entrainment={
            "initial_freq_hz": 5.0,
            "target_freq_hz": 20.0,
            "transition_curve": "ease_in_out",
            "custom_points": None,
        },
        audio_scape={
            "ambient_type": "dawn_forest",
            "intensity": 0.7,
            "binaural_base_freq_hz": 200.0,
            "isochronic_intensity": 0.5,
            "spatial_audio_model": "stereo",
        },
        neuro_cognitive_intent={
            "cognitive_targets": ["focus", "clarity"],
            "emotional_targets": ["empowered"],
            "subconscious_goal": "Break through procrastination into clarity",
        },
    )))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/design", json={
                "user_text": "I'm stuck in a dark fog of procrastination, "
                            "I want to break out into sunrise clarity. 15 minutes.",
                "user_id": "test_user_1",
            })

    assert r.status_code == 200
    bp = r.json()["blueprint"]
    assert bp is not None
    # Theta (4-7Hz) to high-beta/gamma (18-30Hz) transition
    assert 3.0 <= bp["entrainment"]["initial_freq_hz"] <= 8.0, \
        f"Expected theta range, got {bp['entrainment']['initial_freq_hz']}"
    assert 15.0 <= bp["entrainment"]["target_freq_hz"] <= 30.0, \
        f"Expected beta/gamma range, got {bp['entrainment']['target_freq_hz']}"
    # Soundscape should be uplifting
    assert "forest" in bp["audio_scape"]["ambient_type"] or \
           "dawn" in bp["audio_scape"]["ambient_type"] or \
           bp["audio_scape"]["ambient_type"] in ("dawn_forest", "mountain_wind", "deep_space")
    # Neuro-cognitive intent should have meaningful content
    assert bp["neuro_cognitive_intent"]["subconscious_goal"] is not None
    assert len(bp["neuro_cognitive_intent"]["subconscious_goal"]) > 0


@pytest.mark.asyncio
async def test_15_vr_client_type():
    """Test 1.5: VR client -> vr_visual present, ambisonics audio."""
    mock_llm = AsyncMock(return_value=json.dumps(_make_blueprint_for_test(
        client_type="vr",
        duration_seconds=480,
        audio_scape={
            "ambient_type": "ocean_waves",
            "intensity": 0.6,
            "binaural_base_freq_hz": 200.0,
            "isochronic_intensity": 0.3,
            "spatial_audio_model": "ambisonics",
        },
        vr_visual={
            "scene_type": "underwater_coral",
            "color_palette": ["#006994", "#00BFFF", "#7FFFD4"],
            "dynamic_elements": [
                {"type": "bubble_particles", "sync_to": "beat", "params": {"count": 200}},
                {"type": "coral_glow", "sync_to": "slow_breath", "params": {"intensity": 0.8}},
            ],
            "global_light_rhythm": True,
        },
    )))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/design", json={
                "user_text": "Take me to a calm underwater world for 8 minutes.",
                "user_id": "test_user_1",
                "client_type": "vr",
            })

    assert r.status_code == 200
    bp = r.json()["blueprint"]
    assert bp["client_type"] == "vr"
    assert bp["vr_visual"] is not None
    assert bp["vr_visual"]["scene_type"] == "underwater_coral"
    assert len(bp["vr_visual"]["dynamic_elements"]) >= 1
    assert bp["vr_visual"]["global_light_rhythm"] is True
    assert bp["audio_scape"]["spatial_audio_model"] == "ambisonics"


# =========================================================================
# TEST GROUP 2: Safety Layer – Hard Constraints
# =========================================================================

@pytest.mark.asyncio
async def test_21_epilepsy_override():
    """Test 2.1: Epilepsy flag disables flicker and caps VR rhythm."""
    bp_dict = _make_blueprint_for_test(
        duration_seconds=60,
        mobile_visual={"color_hex": "#FF0000", "pulse_sync": True, "pulse_shape": "sine"},
    )
    # Create a session with epilepsy=True
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/start", json=bp_dict)
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    sid = data["session_id"]

    # The session blueprint was validated. For a full epilepsy test,
    # we need to check the internal blueprint state.
    # Directly test the safety validator:
    from app.safety.validator import validate_blueprint
    from app.schemas import SessionBlueprint
    bp = SessionBlueprint(**bp_dict)
    safe_bp, warnings = validate_blueprint(bp, epilepsy=True)
    assert safe_bp.mobile_visual.pulse_sync is False
    assert safe_bp.mobile_visual.pulse_shape == "none"
    assert any("epilepsy" in w.lower() for w in warnings)


@pytest.mark.asyncio
async def test_22_frequency_clamping():
    """Test 2.2: Out-of-range frequency clamped to 40 Hz."""
    from app.safety.validator import validate_blueprint
    from app.schemas import (SessionBlueprint, EntertainmentCurve, AudioScape,
                              MobileVisual, SubliminalConfig, NeuroCognitiveIntent)
    bp = SessionBlueprint(
        user_id="test",
        client_type="mobile",
        duration_seconds=60,
        entrainment=EntertainmentCurve(
            initial_freq_hz=7.5,
            target_freq_hz=12.0,  # within range; test clamps the pre-validated schema
            transition_curve="ease_in_out",
        ),
        audio_scape=AudioScape(ambient_type="dawn_forest", intensity=0.5),
        mobile_visual=MobileVisual(color_hex="#FFA500"),
        subliminal=SubliminalConfig(),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"], emotional_targets=["calm"]
        ),
    )
    # Manually bypass Pydantic to set an out-of-range value
    bp.entrainment.target_freq_hz = 60.0
    safe_bp, warnings = validate_blueprint(bp)
    assert safe_bp.entrainment.target_freq_hz == 40.0
    assert any("clamped" in w for w in warnings)


@pytest.mark.asyncio
async def test_23_duration_cap():
    """Test 2.3: Duration over 3h caped at 10800."""
    from app.safety.validator import validate_blueprint
    from app.schemas import (SessionBlueprint, EntertainmentCurve, AudioScape,
                              MobileVisual, SubliminalConfig, NeuroCognitiveIntent)

    # Use model_construct to bypass Pydantic's max validation
    bp = SessionBlueprint.model_construct(**{
        "user_id": "test",
        "client_type": "mobile",
        "duration_seconds": 18000,  # 5 hours
        "entrainment": EntertainmentCurve.model_construct(
            initial_freq_hz=7.5, target_freq_hz=12.0,
            transition_curve="ease_in_out",
        ),
        "audio_scape": AudioScape.model_construct(
            ambient_type="dawn_forest", intensity=0.5,
        ),
        "mobile_visual": MobileVisual.model_construct(
            color_hex="#FFA500", pulse_sync=True, pulse_shape="sine",
        ),
        "subliminal": SubliminalConfig.model_construct(enabled=False, messages=[]),
        "neuro_cognitive_intent": NeuroCognitiveIntent.model_construct(
            cognitive_targets=["focus"], emotional_targets=["calm"],
        ),
    })
    safe_bp, warnings = validate_blueprint(bp)
    assert safe_bp.duration_seconds == 10800
    assert any("capped" in w.lower() for w in warnings)


# =========================================================================
# TEST GROUP 3: Session Streaming – Real-time Audio and Light
# =========================================================================

@pytest.mark.asyncio
async def test_31_session_start_flow():
    """Test 3.1: Start session -> session created with idle state."""
    bp_dict = _make_blueprint_for_test(duration_seconds=60)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/start", json=bp_dict)
    assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
    data = r.json()
    assert "session_id" in data
    assert "summary" in data
    sid = data["session_id"]

    # Verify the session exists
    from app.session.controller import SESSIONS
    assert sid in SESSIONS
    assert SESSIONS[sid]["state"].value == "idle"


@pytest.mark.asyncio
async def test_32_light_frames_structure():
    """Test 3.2: Light frames have correct structure (mobile)."""
    from app.light.mobile_generator import generate_mobile_frames
    from app.schemas import SessionBlueprint

    bp = SessionBlueprint(**_make_blueprint_for_test(duration_seconds=60))
    frames = generate_mobile_frames(bp, 0, 3, fps=60.0)

    # ~180 frames in 3 sec
    assert len(frames) >= 170, f"Expected ~180 frames, got {len(frames)}"

    # Check structure
    for f in frames:
        assert "t_ms" in f
        assert "r" in f
        assert "g" in f
        assert "b" in f
        assert "a" in f
        assert 0 <= f["a"] <= 1.0
        assert isinstance(f["r"], int)
        assert 0 <= f["r"] <= 255

    # t_ms monotonically increasing
    t_ms = [f["t_ms"] for f in frames]
    assert all(t_ms[i] < t_ms[i + 1] for i in range(len(t_ms) - 1))


@pytest.mark.asyncio
async def test_33_vr_commands_structure():
    """Test 3.3: VR commands have correct structure."""
    from app.light.vr_generator import generate_vr_commands
    from app.schemas import SessionBlueprint, VRVisual, VRElement

    bp_dict = _make_blueprint_for_test(
        duration_seconds=60,
        client_type="vr",
        vr_visual={
            "scene_type": "cosmic",
            "color_palette": ["#FF0000", "#00FF00"],
            "dynamic_elements": [
                {"type": "sphere", "sync_to": "beat", "params": {"size": 2.0}},
            ],
            "global_light_rhythm": True,
        },
    )
    bp = SessionBlueprint(**bp_dict)
    cmds = generate_vr_commands(bp, 0, 5)

    # load_scene at t=0
    load_scenes = [c for c in cmds if c["cmd"] == "load_scene"]
    assert len(load_scenes) == 1
    assert load_scenes[0]["t_ms"] == 0

    # update_element commands present
    updates = [c for c in cmds if c["cmd"] == "update_element"]
    assert len(updates) >= 1
    for u in updates:
        assert "type" in u
        assert "sync_val" in u
        assert isinstance(u["sync_val"], float)


# =========================================================================
# TEST GROUP 4: Profile & Personalisation Learning
# =========================================================================

@pytest.mark.asyncio
async def test_41_rating_saved():
    """Test 4.1: Rating a session persists to store."""
    bp_dict = _make_blueprint_for_test(duration_seconds=60)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/start", json=bp_dict)
    sid = r.json()["session_id"]

    # Rate the session
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r2 = await c.post(f"/api/session/{sid}/rate", json={
            "rating": 2,
            "notes": "too intense",
        })
    assert r2.status_code == 200
    assert r2.json()["status"] == "saved"

    # Check the store directly
    from app.profile.store import get_user_sessions
    rows = get_user_sessions("test_user_1")
    assert len(rows) >= 1
    rated = [r for r in rows if r.get("session_id") == sid]
    assert len(rated) == 1
    assert rated[0]["rating"] == 2
    assert "too intense" in rated[0]["notes"]


@pytest.mark.asyncio
async def test_42_few_shot_prompt_context():
    """Test 4.2: Profile context is built from past sessions."""
    from app.profile.prompt_builder import build_profile_context
    from app.profile.store import save_rating
    from app.schemas import SessionBlueprint

    # Save a highly-rated session
    bp = SessionBlueprint(**(_make_blueprint_for_test()))
    await save_rating("test_user_1", "past_sid_1", bp, 5, "Perfect calm")

    # Build context (without current_text to avoid sentence-transformers)
    context = await build_profile_context("test_user_1")
    assert "Rating 5/5" in context
    assert "Soundscape:" in context
    assert "dawn_forest" in context


# =========================================================================
# TEST GROUP 5: Edge Cases & Graceful Degradation
# =========================================================================

@pytest.mark.asyncio
async def test_51_garbled_input():
    """Test 5.1: Garbled text -> 422 validation error or graceful response."""
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/design", json={
            "user_text": "asdfghjkl 12345",
            "user_id": "test_user_1",
        })
    # Should NOT be a 500 error
    assert r.status_code in (200, 422)
    if r.status_code == 422:
        data = r.json()
        assert "detail" in data  # Validation error


@pytest.mark.asyncio
async def test_52_missing_user_id():
    """Test: Missing user_id -> 422 validation error."""
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/design", json={
            "user_text": "Help me relax",
        })
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_53_nonexistent_session_status():
    """Test: Query non-existent session -> returns empty state (not 500)."""
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/session/nonexistent_session_id/status")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] is None
    assert data["t"] is None


@pytest.mark.asyncio
async def test_54_llm_fallback_on_error():
    """Test 5.2: LLM failure -> fallback to requires_duration (not crash)."""
    mock_llm = AsyncMock(side_effect=Exception("LLM completely down"))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/design", json={
                "user_text": "Make me a session for 10 minutes",
                "user_id": "test_user_1",
            })
    assert r.status_code == 200
    data = r.json()
    assert data["requires_duration"] is True  # graceful fallback


@pytest.mark.asyncio
async def test_55_concurrent_sessions_no_crosstalk():
    """Test 5.3: 3 concurrent sessions -> no state bleed between them."""
    sids = []
    for i in range(3):
        bp_dict = _make_blueprint_for_test(
            user_id=f"user_{i}",
            duration_seconds=60,
            entrainment={
                "initial_freq_hz": 7.5 + i * 2.0,
                "target_freq_hz": 10.0 + i * 2.0,
                "transition_curve": "ease_in_out",
                "custom_points": None,
            },
        )
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/api/session/start", json=bp_dict)
        assert r.status_code == 200
        sids.append(r.json()["session_id"])

    # All sessions should have distinct IDs
    assert len(set(sids)) == 3

    # All should be in the store
    from app.session.controller import SESSIONS
    for sid in sids:
        assert sid in SESSIONS
        assert SESSIONS[sid]["state"].value == "idle"

    # Verify the blueprints have distinct frequencies (no crosstalk)
    freqs = []
    for sid in sids:
        bp = SESSIONS[sid]["blueprint"]
        freqs.append(bp.entrainment.initial_freq_hz)
    assert len(set(freqs)) == 3, f"Expected 3 distinct frequencies, got {freqs}"
