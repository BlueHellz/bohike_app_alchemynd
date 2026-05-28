import os
import json
import re
from pathlib import Path
from typing import Optional, Tuple
from app.schemas import SessionBlueprint, DesignResponse
from app.profile.prompt_builder import build_profile_context

# Resolve paths relative to this file, not CWD
_THIS_DIR = Path(__file__).parent

SCHEMA_JSON = SessionBlueprint.model_json_schema()


def _get_llm_provider() -> str:
    """Read LLM provider at call time so .env loading (dotenv) takes effect."""
    return os.getenv("LLM_PROVIDER", "openai")


# -- LLM call abstraction ----------------------------------------------------

async def call_llm(system_prompt: str, user_message: str) -> str:
    if _get_llm_provider() == "deepseek":
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    elif _get_llm_provider() == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    else:
        # Local LLM (Ollama / llama.cpp)
        import httpx
        url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        model = os.getenv("LOCAL_LLM_MODEL", "mistral")
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{url}/chat/completions", json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.4,
            })
        return r.json()["choices"][0]["message"]["content"]


# -- JSON extraction with retry -----------------------------------------------

def extract_json(raw: str) -> dict:
    """Strip markdown fences, parse JSON. Raises ValueError on failure."""
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(cleaned)


# -- Main designer ------------------------------------------------------------

async def design_session(user_text: str, user_id: str,
                         client_type: str = "mobile",
                         retries: int = 2) -> DesignResponse:
    profile_ctx = await build_profile_context(user_id)
    with open(_THIS_DIR / "prompt_template.txt") as f:
        template = f.read()

    # Use replace() instead of format() to avoid conflicts with
    # curly braces in the schema JSON string.
    system_prompt = (
        template
        .replace("{profile_context}", profile_ctx)
        .replace("{client_type}", client_type)
        .replace("{schema_json}", json.dumps(SCHEMA_JSON, indent=2))
        .replace("{user_text}", "")
    )

    for attempt in range(retries + 1):
        try:
            raw = await call_llm(system_prompt, user_text)
            data = extract_json(raw)

            # Handle requires_duration shortcircuit
            if data.get("requires_duration"):
                return DesignResponse(requires_duration=True)

            data["user_id"] = user_id
            data["client_type"] = client_type
            bp = SessionBlueprint(**data)
            return DesignResponse(blueprint=bp)

        except Exception:
            if attempt == retries:
                # Final fallback: ask user to rephrase
                return DesignResponse(requires_duration=True)
            continue  # retry with same prompt


# -- Warm-up (call on FastAPI startup) ---------------------------------------

_warmed = False


async def warm_up():
    """Pre-warm LLM connection on startup. Never blocks or crashes the server."""
    global _warmed
    if not _warmed:
        try:
            await call_llm("You are NeuroSync.", "Say OK if you receive this.")
            _warmed = True
        except Exception:
            import logging
            logging.info("LLM warm-up deferred (will warm on first request).")
