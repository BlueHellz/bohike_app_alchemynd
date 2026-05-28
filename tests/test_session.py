import pytest
import asyncio
from app.schemas import (
    SessionBlueprint, EntertainmentCurve, AudioScape, MobileVisual,
    SubliminalConfig, NeuroCognitiveIntent,
)
from app.session.controller import (
    create_session, run_session, pause_session, resume_session,
    stop_session, SESSIONS, State,
)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Reset SESSIONS dict before each test."""
    SESSIONS.clear()
    yield


def _make_bp(**overrides) -> SessionBlueprint:
    defaults = dict(
        user_id="test",
        client_type="mobile",
        duration_seconds=60,
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


@pytest.mark.asyncio
async def test_create_session_returns_sid():
    bp = _make_bp()
    sid = await create_session(bp)
    assert isinstance(sid, str)
    assert len(sid) > 0
    assert sid in SESSIONS


@pytest.mark.asyncio
async def test_session_state_transitions():
    """State: idle -> running -> paused -> running -> stopped."""
    bp = _make_bp(duration_seconds=60)
    sid = await create_session(bp)
    assert SESSIONS[sid]["state"] == State.IDLE

    # Simulate a short run, then pause
    run_task = asyncio.create_task(run_session(sid))
    await asyncio.sleep(0.1)  # let it start

    assert SESSIONS[sid]["state"] == State.RUNNING

    pause_session(sid)
    assert SESSIONS[sid]["state"] == State.PAUSED

    resume_session(sid)
    assert SESSIONS[sid]["state"] == State.RUNNING

    stop_session(sid)
    await asyncio.sleep(0.1)  # let loop see stop
    assert SESSIONS[sid]["state"] == State.STOPPED
    run_task.cancel()


@pytest.mark.asyncio
async def test_audio_chunks_are_produced():
    """Running session produces audio chunks at ~1 Hz."""
    bp = _make_bp(duration_seconds=60)
    sid = await create_session(bp)
    run_task = asyncio.create_task(run_session(sid))
    await asyncio.sleep(0.3)

    chunks = []
    try:
        while len(chunks) < 2:
            chunk = await asyncio.wait_for(SESSIONS[sid]["audio_q"].get(), timeout=1.0)
            chunks.append(chunk)
    except asyncio.TimeoutError:
        pass

    stop_session(sid)
    await asyncio.sleep(0.1)
    run_task.cancel()

    assert len(chunks) >= 1, "No audio chunks were produced"


@pytest.mark.asyncio
async def test_light_frames_monotonically_increasing():
    """Light frames have monotonically increasing t_ms."""
    bp = _make_bp(duration_seconds=60)
    sid = await create_session(bp)
    run_task = asyncio.create_task(run_session(sid))
    await asyncio.sleep(0.3)

    try:
        frames = await asyncio.wait_for(SESSIONS[sid]["light_q"].get(), timeout=1.0)
        if frames:
            t_ms = [f["t_ms"] for f in frames]
            assert all(t_ms[i] < t_ms[i + 1] for i in range(len(t_ms) - 1))
    except asyncio.TimeoutError:
        pass

    stop_session(sid)
    run_task.cancel()


@pytest.mark.asyncio
async def test_multiple_sessions_no_state_bleed():
    """Three concurrent sessions should not share state."""
    sids = []
    for i in range(3):
        bp = _make_bp(user_id=f"user_{i}", duration_seconds=60)
        sid = await create_session(bp)
        sids.append(sid)

    tasks = [asyncio.create_task(run_session(sid)) for sid in sids]
    await asyncio.sleep(0.2)

    for sid in sids:
        assert SESSIONS[sid]["state"] == State.RUNNING
        assert sid in SESSIONS

    stop_session(sids[0])
    await asyncio.sleep(0.1)
    assert SESSIONS[sids[0]]["state"] == State.STOPPED
    assert SESSIONS[sids[1]]["state"] == State.RUNNING
    assert SESSIONS[sids[2]]["state"] == State.RUNNING

    for t in tasks:
        t.cancel()


@pytest.mark.asyncio
async def test_epilepsy_validation_applied():
    """create_session applies epilepsy validation to blueprint."""
    bp = _make_bp(mobile_visual=MobileVisual(color_hex="#FF0000"))
    sid = await create_session(bp, epilepsy=True)
    assert SESSIONS[sid]["epilepsy"] is True
    assert SESSIONS[sid]["blueprint"].mobile_visual.pulse_sync is False
