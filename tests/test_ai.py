import pytest
import json
from unittest.mock import patch, AsyncMock
from app.schemas import SessionBlueprint
from app.ai.session_designer import design_session, extract_json


# ---------------------------------------------------------------------------
# Mock helpers -- return valid SessionBlueprint JSON for any LLM call
# ---------------------------------------------------------------------------

def _mock_blueprint_json(**overrides) -> str:
    """Generate a valid SessionBlueprint JSON payload as the LLM would return."""
    base = {
        "client_type": "mobile",
        "duration_seconds": 600,
        "entrainment": {
            "initial_freq_hz": 7.5,
            "target_freq_hz": 12.0,
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
            "color_hex": "#FFA500",
            "pulse_sync": True,
            "pulse_shape": "sine",
        },
        "subliminal": {
            "enabled": False,
            "messages": [],
        },
        "neuro_cognitive_intent": {
            "cognitive_targets": ["focus"],
            "emotional_targets": ["calm"],
            "subconscious_goal": "Achieve relaxed alertness",
        },
    }
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# extract_json unit tests
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced_json(self):
        raw = "```json\n{\"a\": 1}\n```"
        assert extract_json(raw) == {"a": 1}

    def test_markdown_fenced_no_lang(self):
        raw = "```\n{\"a\": 1}\n```"
        assert extract_json(raw) == {"a": 1}

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            extract_json("not json at all")


# ---------------------------------------------------------------------------
# design_session integration tests (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_design_session_returns_blueprint():
    """Happy path: LLM returns valid blueprint JSON."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json())
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I feel anxious, I want calm for 20 minutes",
            user_id="test_user",
        )
    assert resp.blueprint is not None
    bp = resp.blueprint
    assert bp.duration_seconds == 600
    assert 0.5 <= bp.entrainment.initial_freq_hz <= 40.0
    assert bp.subliminal.enabled is False


@pytest.mark.asyncio
async def test_requires_duration_when_missing():
    """No duration in LLM response -> requires_duration=True."""
    mock_llm = AsyncMock(return_value=json.dumps({"requires_duration": True}))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I feel anxious I want calm",
            user_id="test_user",
        )
    assert resp.requires_duration is True
    assert resp.blueprint is None


@pytest.mark.asyncio
async def test_subliminal_with_user_provided_words():
    """User provides quoted affirmations -> subliminal enabled with those messages."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json(
        subliminal={"enabled": True, "messages": ["I am unstoppable"]}
    ))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text='use my words: "I am unstoppable"',
            user_id="test_user",
        )
    assert resp.blueprint is not None
    assert resp.blueprint.subliminal.enabled is True
    assert resp.blueprint.subliminal.messages == ["I am unstoppable"]


@pytest.mark.asyncio
async def test_no_subliminal_without_user_words():
    """No quoted words in user text -> subliminal disabled."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json())
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I feel tired, energize me",
            user_id="test_user",
        )
    assert resp.blueprint is not None
    assert resp.blueprint.subliminal.enabled is False


@pytest.mark.asyncio
async def test_exact_duration_extraction():
    """User says 'exactly 10 minutes' -> duration_seconds=600."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json(duration_seconds=600))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I am exhausted, boost me for exactly 10 minutes",
            user_id="test_user",
        )
    assert resp.blueprint is not None
    assert resp.blueprint.duration_seconds == 600


@pytest.mark.asyncio
async def test_retry_on_failure_then_fallback():
    """After all retries fail, return requires_duration fallback."""
    mock_llm = AsyncMock(side_effect=Exception("LLM down"))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I feel anxious",
            user_id="test_user",
            retries=1,
        )
    assert resp.requires_duration is True
    assert mock_llm.call_count == 2  # initial + 1 retry


@pytest.mark.asyncio
async def test_frequencies_in_valid_range():
    """All generated frequencies must be in [0.5, 40.0] Hz."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json(
        entrainment={
            "initial_freq_hz": 18.5,
            "target_freq_hz": 10.2,
            "transition_curve": "ease_in_out",
            "custom_points": None,
        }
    ))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I feel anxious I want calm for 20 minutes",
            user_id="test_user",
        )
    assert resp.blueprint is not None
    assert 0.5 <= resp.blueprint.entrainment.initial_freq_hz <= 40.0
    assert 0.5 <= resp.blueprint.entrainment.target_freq_hz <= 40.0


@pytest.mark.asyncio
async def test_vr_client_type():
    """VR client type produces VR visual blueprint."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json(
        client_type="vr",
        vr_visual={
            "scene_type": "cosmic",
            "color_palette": ["#FF0000", "#00FF00", "#0000FF"],
            "dynamic_elements": [
                {"type": "sphere", "sync_to": "beat", "params": {"size": 2.0}}
            ],
            "global_light_rhythm": True,
        },
    ))
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="I want a VR relaxation session for 15 minutes",
            user_id="test_user",
            client_type="vr",
        )
    assert resp.blueprint is not None
    assert resp.blueprint.client_type == "vr"
    assert resp.blueprint.vr_visual is not None
    assert resp.blueprint.vr_visual.scene_type == "cosmic"


@pytest.mark.asyncio
async def test_blueprint_user_id_is_set():
    """The user_id from design_session is written into the blueprint."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json())
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="Relax me for 5 minutes",
            user_id="user_42",
        )
    assert resp.blueprint is not None
    assert resp.blueprint.user_id == "user_42"


@pytest.mark.asyncio
async def test_blueprint_client_type_default():
    """Default client_type is 'mobile'."""
    mock_llm = AsyncMock(return_value=_mock_blueprint_json())
    with patch("app.ai.session_designer.call_llm", mock_llm):
        resp = await design_session(
            user_text="Relax me for 5 minutes",
            user_id="test_user",
        )
    assert resp.blueprint is not None
    assert resp.blueprint.client_type == "mobile"
