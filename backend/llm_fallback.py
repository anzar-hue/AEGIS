import os
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Prioritized model chain: tries fastest/newest first, then lighter models
MODEL_PRIORITY = [
    m for m in [
        os.getenv("GEMINI_MODEL"),  # Optional override in Render env vars
        "gemini-3.8-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ] if m
]


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if GENAI_AVAILABLE and api_key:
        return genai.Client(api_key=api_key)
    return None


def generate_with_fallback(prompt: str, json_mode: bool = False) -> Tuple[Optional[str], str]:
    """
    Tries each model in MODEL_PRIORITY in order.
    Returns (response_text, mode_label).
    If all models fail (503 high demand / 429 / 404), returns (None, "LOCAL_FALLBACK_MODE").
    """
    client = _get_client()
    if not client:
        return None, "LOCAL_FALLBACK_MODE"

    config = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None

    for model_name in MODEL_PRIORITY:
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (resp.text or "").strip()
            if text:
                return text, f"LIVE_API ({model_name})"
        except Exception as e:
            print(f"[LLM Fallback] {model_name} failed ({type(e).__name__}): {e}")
            continue

    return None, "LOCAL_FALLBACK_MODE"