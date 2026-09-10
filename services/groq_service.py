"""
Groq LLM Service for LawWise
"""
import os
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_groq_client():
    """Initialize and return the Groq client."""
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file."
        )
    return Groq(api_key=api_key)


def get_model() -> str:
    return os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    expect_json: bool = False,
) -> str:
    """
    Call the Groq LLM and return the response text.
    Handles rate limits and model errors gracefully.
    """
    try:
        client = get_groq_client()
        model = get_model()
        logger.info(f"Calling Groq model: {model}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        err = str(e)
        err_lower = err.lower()
        if "rate_limit" in err_lower or "429" in err:
            logger.warning("Groq rate limit hit.")
            return "Rate limit reached. Please wait a moment and try again."
        if (
            "model_not_found" in err_lower
            or ("model" in err_lower and "not found" in err_lower)
            or "404" in err
        ):
            model_name = get_model()
            logger.error(f"Model not found: {model_name} — {err}")
            return (
                f"The configured AI model '{model_name}' is not available on your Groq account. "
                "Please update GROQ_MODEL in your .env to a model your account has access to. "
                "Check available models at console.groq.com."
            )
        if (
            "invalid_api_key" in err_lower
            or "api_key" in err_lower
            or "authentication" in err_lower
            or "401" in err
        ):
            logger.error(f"Groq auth error: {err}")
            return (
                "Invalid or missing Groq API key. "
                "Please check GROQ_API_KEY in your .env file. "
                "The key should start with 'gsk_'."
            )
        logger.error(f"Groq LLM error: {e}")
        return f"AI service error: {err}"


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> Optional[Dict[str, Any]]:
    """
    Call LLM expecting JSON output.
    Returns parsed dict or None on failure.
    """
    raw = call_llm(system_prompt, user_prompt, temperature, max_tokens, expect_json=True)
    if not raw or raw.startswith("Rate limit") or raw.startswith("AI service"):
        return None
    # Try to extract JSON from the response
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON block in markdown-fenced response
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try finding first { ... } block
        match2 = re.search(r"(\{.*\})", raw, re.DOTALL)
        if match2:
            try:
                return json.loads(match2.group(1))
            except json.JSONDecodeError:
                pass
        logger.warning("LLM returned non-JSON response, falling back to text.")
        return None
