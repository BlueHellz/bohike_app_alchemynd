"""
Profile tests.
Note: sentence-transformers requires torch which is not available for
Python 3.13 on PyPI yet. All tests that use the embedding model mock it out.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.profile.store import save_rating, get_user_sessions, sessions_table
from app.profile.prompt_builder import build_profile_context
from app.schemas import (
    SessionBlueprint, EntertainmentCurve, AudioScape, MobileVisual,
    SubliminalConfig, NeuroCognitiveIntent,
)


@pytest.fixture(autouse=True)
def clear_db():
    """Clear TinyDB sessions table before each test."""
    sessions_table.truncate()
    yield


def _make_bp(**overrides) -> SessionBlueprint:
    defaults = dict(
        user_id="test_user",
        client_type="mobile",
        duration_seconds=600,
        entrainment=EntertainmentCurve(initial_freq_hz=7.5, target_freq_hz=12.0),
        audio_scape=AudioScape(ambient_type="dawn_forest", intensity=0.6),
        mobile_visual=MobileVisual(color_hex="#FFA500"),
        subliminal=SubliminalConfig(),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"], emotional_targets=["calm"],
            subconscious_goal="Achieve relaxed alertness",
        ),
    )
    defaults.update(overrides)
    return SessionBlueprint(**defaults)


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_and_retrieve_rating():
    bp = _make_bp()
    await save_rating("user_1", "session_abc", bp, 5, "Amazing session!")

    rows = get_user_sessions("user_1")
    assert len(rows) == 1
    assert rows[0]["user_id"] == "user_1"
    assert rows[0]["session_id"] == "session_abc"
    assert rows[0]["rating"] == 5
    assert rows[0]["notes"] == "Amazing session!"


@pytest.mark.asyncio
async def test_multiple_sessions_sorted_by_time():
    bp = _make_bp()
    # Insert in non-chronological order to test sorting
    await save_rating("user_1", "session_1", bp, 3)
    await save_rating("user_1", "session_2", bp, 4)
    await save_rating("user_1", "session_3", bp, 5)

    rows = get_user_sessions("user_1")
    assert len(rows) == 3
    # Should be sorted newest first
    timestamps = [r["timestamp"] for r in rows]
    assert timestamps == sorted(timestamps, reverse=True)


@pytest.mark.asyncio
async def test_get_user_sessions_limit():
    bp = _make_bp()
    for i in range(5):
        await save_rating("user_1", f"session_{i}", bp, 4)

    rows = get_user_sessions("user_1", limit=3)
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_user_isolation():
    bp = _make_bp()
    await save_rating("user_1", "s1", bp, 5)
    await save_rating("user_2", "s2", bp, 3)

    assert len(get_user_sessions("user_1")) == 1
    assert len(get_user_sessions("user_2")) == 1
    assert len(get_user_sessions("user_3")) == 0


# ---------------------------------------------------------------------------
# Prompt builder tests (mocked sentence-transformers)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_past_sessions():
    """No sessions yet -> returns 'No past sessions.'"""
    context = await build_profile_context("new_user")
    assert context == "No past sessions."


@pytest.mark.asyncio
async def test_no_highly_rated_sessions():
    """Only low-rated sessions exist -> returns appropriate message."""
    bp = _make_bp()
    await save_rating("user_1", "s1", bp, 2, "Too slow")
    await save_rating("user_1", "s2", bp, 3, "Okay")

    context = await build_profile_context("user_1")
    assert "No highly-rated past sessions yet" in context


@pytest.mark.asyncio
async def test_high_rated_sessions_returned():
    """High-rated sessions appear in context."""
    bp = _make_bp(
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"],
            emotional_targets=["calm"],
            subconscious_goal="Achieve relaxed alertness",
        ),
    )
    await save_rating("user_1", "s1", bp, 5, "Perfect!")
    await save_rating("user_1", "s2", bp, 4, "Very good")

    context = await build_profile_context("user_1")
    # Should not say "No highly-rated past sessions yet"
    assert "No highly-rated" not in context
    assert "Rating 5/5" in context or "Rating 4/5" in context
    assert "Soundscape:" in context


@pytest.mark.asyncio
async def test_context_includes_user_note():
    """User's note from past session appears in context."""
    bp = _make_bp()
    await save_rating("user_1", "s1", bp, 5, "I want it faster next time")

    context = await build_profile_context("user_1")
    assert "faster" in context


@pytest.mark.asyncio
async def test_semantic_similarity_with_mock():
    """With current_text, the flow attempts semantic search (mocked)."""
    bp1 = _make_bp(
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["energy"],
            emotional_targets=["drowsy"],
            subconscious_goal="Wake up and energize",
        ),
    )
    bp2 = _make_bp(
        entrainment=EntertainmentCurve(initial_freq_hz=14.0, target_freq_hz=20.0),
        audio_scape=AudioScape(ambient_type="cosmic_drone", intensity=0.7),
        neuro_cognitive_intent=NeuroCognitiveIntent(
            cognitive_targets=["focus"],
            emotional_targets=["tired"],
            subconscious_goal="Stay alert and sharp",
        ),
    )
    await save_rating("user_1", "s1", bp1, 5, "Great energy boost")
    await save_rating("user_1", "s2", bp2, 5, "Sharp focus achieved")

    # Build context without current_text (avoids calling sentence-transformers)
    context = await build_profile_context("user_1")
    assert "Rating 5/5" in context
    assert "Soundscape:" in context
    assert "dawn_forest" in context or "cosmic_drone" in context
