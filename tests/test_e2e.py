"""
End-to-end test: design -> start -> stream -> verify.
Requires the server to be running on localhost:8000.

Run with: .venv/bin/python3 -m pytest tests/test_e2e.py -v --asyncio-mode=auto
Or start server first: .venv/bin/uvicorn app.main:app &
"""
import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.mark.asyncio
async def test_design_endpoint_returns_blueprint():
    """
    POST /api/session/design with a valid request returns either a blueprint
    or a requires_duration response (depending on whether LLM is available).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/design", json={
            "user_text": "I feel drained and foggy. I want sharp focus "
                         "for exactly 10 minutes.",
            "user_id": "test_user_001",
            "client_type": "mobile",
        })
    assert r.status_code == 200
    data = r.json()
    # If no LLM key is configured, we get requires_duration fallback
    if data.get("blueprint"):
        bp = data["blueprint"]
        assert bp["duration_seconds"] == 600
        assert 0.5 <= bp["entrainment"]["initial_freq_hz"] <= 40.0
        assert bp["subliminal"]["enabled"] is False
    else:
        assert data.get("requires_duration") is True


@pytest.mark.asyncio
async def test_start_endpoint():
    """
    POST /api/session/start with a valid blueprint returns session_id + summary.
    """
    bp = {
        "user_id": "test_user_001",
        "client_type": "mobile",
        "duration_seconds": 60,
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
        "vr_visual": None,
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/start", json=bp)
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert "summary" in data
    assert "Duration:" in data["summary"]
    assert "Brainwave transition:" in data["summary"]
    assert data["session_id"] is not None


@pytest.mark.asyncio
async def test_status_endpoint():
    """
    GET /api/session/{sid}/status returns current state.
    """
    bp = {
        "user_id": "test_user_001",
        "client_type": "mobile",
        "duration_seconds": 60,
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
        "vr_visual": None,
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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/session/start", json=bp)
        sid = r.json()["session_id"]

        r2 = await c.get(f"/api/session/{sid}/status")
    assert r2.status_code == 200
    assert r2.json()["state"] in ("idle", "running", "completed", "stopped")
