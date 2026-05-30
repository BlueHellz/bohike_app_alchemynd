import os, json, re
from pathlib import Path
from app.schemas import SessionBlueprint, DesignRequest, DesignResponse
from app.safety.validator import validate, safety_summary
from app.profile.builder import build_profile_context

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
SCHEMA = SessionBlueprint.model_json_schema()

async def call_llm(system: str, user: str) -> str:
    if LLM_PROVIDER == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        r = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.4, response_format={"type":"json_object"},
        )
        return r.choices[0].message.content
    else:
        import httpx
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(
                os.getenv("LOCAL_LLM_URL","http://localhost:11434/v1")+"/chat/completions",
                json={"model":os.getenv("LOCAL_LLM_MODEL","mistral"),
                      "messages":[{"role":"system","content":system},{"role":"user","content":user}],
                      "temperature":0.4},
            )
        return r.json()["choices"][0]["message"]["content"]

def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```","",raw).strip()
    return json.loads(cleaned)

async def design_session(req: DesignRequest, epilepsy: bool = False) -> DesignResponse:
    profile_ctx = await build_profile_context(req.user_id, req.user_text)
    _THIS_DIR = Path(__file__).parent
    with open(_THIS_DIR / "prompts/system_architect.txt") as f:
        template = f.read()
    system = (
        template
        .replace("{profile_context}", profile_ctx)
        .replace("{client_type}", req.client_type or "mobile")
        .replace("{preferred_modalities}", str(req.preferred_modalities or "AI selects best"))
        .replace("{schema_json}", json.dumps(SCHEMA, indent=2))
        .replace("{user_text}", "")
    )
    for attempt in range(3):
        try:
            raw = await call_llm(system, req.user_text)
            data = _extract_json(raw)
            if data.get("requires_duration"):
                return DesignResponse(requires_duration=True)
            bp = SessionBlueprint(**data)
            bp.user_id = req.user_id
            bp.client_type = req.client_type or "mobile"
            bp, warnings = validate(bp, epilepsy)
            return DesignResponse(blueprint=bp, safety_summary=safety_summary(bp))
        except Exception:
            if attempt == 2: return DesignResponse(requires_duration=True)
            continue

_warmed = False
async def warm_up():
    global _warmed
    if not _warmed:
        try: await call_llm("Respond with: ready.", "ping")
        except: pass
        _warmed = True
