"""
Builds the few-shot context string injected into the LLM prompt.
Uses semantic similarity to find relevant past sessions.

Note: sentence-transformers (and its torch dependency) are imported lazily
so the module can be imported without them being installed.
"""
from app.profile.store import get_user_sessions
import json

_model = None


def _embed_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer, util as st_util
        _model = {
            "model": SentenceTransformer("all-MiniLM-L6-v2"),
            "util": st_util,
        }
    return _model


async def build_profile_context(user_id: str, current_text: str = "",
                                 n_shots: int = 3) -> str:
    rows = get_user_sessions(user_id, limit=50)
    if not rows:
        return "No past sessions."

    high_rated = [r for r in rows if r.get("rating", 0) >= 4]
    if not high_rated:
        return "No highly-rated past sessions yet."

    if current_text:
        st = _embed_model()
        q_emb = st["model"].encode(current_text, convert_to_tensor=True)
        c_embs = st["model"].encode(
            [
                r["blueprint"].get("neuro_cognitive_intent", {})
                .get("subconscious_goal", "")
                for r in high_rated
            ],
            convert_to_tensor=True,
        )
        scores = st["util"].cos_sim(q_emb, c_embs)[0].tolist()
        ranked = sorted(zip(scores, high_rated), key=lambda x: -x[0])
        shots = [r for _, r in ranked[:n_shots]]
    else:
        shots = high_rated[:n_shots]

    lines = [
        "Here are highly-rated past sessions from this user (use as style guidance):"
    ]
    for r in shots:
        bp = r["blueprint"]
        notes = r.get("notes", "")
        lines.append(
            f"- Rating {r['rating']}/5. Soundscape: {bp['audio_scape']['ambient_type']}. "
            f"Curve: {bp['entrainment']['transition_curve']}. "
            f"Freq: {bp['entrainment']['initial_freq_hz']}->{bp['entrainment']['target_freq_hz']} Hz. "
            f"User note: '{notes}'"
        )
    return "\n".join(lines)
