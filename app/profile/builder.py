from app.profile.store import get_user_sessions
async def build_profile_context(user_id:str,current_text:str="")->str:
    rows=get_user_sessions(user_id,50)
    if not rows: return "No past sessions yet."
    try:
        from sentence_transformers import SentenceTransformer,util
        high=[r for r in rows if r.get("rating",0)>=4]
        if not high: return "No highly-rated past sessions yet."
        if current_text:
            m=SentenceTransformer("all-MiniLM-L6-v2")
            qe=m.encode(current_text,convert_to_tensor=True)
            ce=m.encode([r["blueprint"].get("intent_summary","") for r in high],convert_to_tensor=True)
            scores=util.cos_sim(qe,ce)[0].tolist()
            ranked=[r for _,r in sorted(zip(scores,high),key=lambda x:-x[0])[:3]]
        else: ranked=high[:3]
        lines=["Top-rated past sessions (calibrate style to these):"]
        for r in ranked:
            bp=r["blueprint"];m=r.get("blueprint",{}).get("meditation",{})
            lines.append(f"- Rating {r['rating']}/5 | Entrainment: {bp['entrainment']['initial_hz']}->{bp['entrainment']['target_hz']} Hz | Style: {m.get('style','?')} | Noise: {bp['audio']['noise_types']} | Note: {r.get('notes','')}")
        return "\n".join(lines)
    except ImportError: return "No highly-rated past sessions yet."
