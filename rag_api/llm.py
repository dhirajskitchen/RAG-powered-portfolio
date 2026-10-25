"""Free-tier LLM calls: Gemini first, optional Groq fallback (both have free tiers)."""
import os

from google import genai
from google.genai import types
from openai import AsyncOpenAI

GEMINI_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = 400

_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"]) if os.getenv("GEMINI_API_KEY") else None
_groq = (
    AsyncOpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
    if os.getenv("GROQ_API_KEY")
    else None
)


async def generate(system: str, user: str) -> str:
    """One LLM call. Raises RuntimeError if no provider succeeds."""
    errors = []
    if _gemini:
        try:
            r = await _gemini.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.1,
                    max_output_tokens=MAX_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            if r.text:
                return r.text
        except Exception as e:  # rate limit, quota, network -> try next provider
            errors.append(f"gemini: {e}")
    if _groq:
        try:
            r = await _groq.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.1,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            if r.choices[0].message.content:
                return r.choices[0].message.content
        except Exception as e:
            errors.append(f"groq: {e}")
    raise RuntimeError("; ".join(errors) or "no LLM API key configured")
