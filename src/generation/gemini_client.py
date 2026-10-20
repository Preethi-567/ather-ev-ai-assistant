# src/generation/gemini_client.py
"""
Shared Gemini client with:
- Robust response text extraction (handles preview model response shapes)
- Model fallback chain
- Retry with backoff
- Per-call rate limiter
"""

import os
import re
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from loguru import logger

from src.generation.rate_limiter import SlidingWindowLimiter, max_calls_from_env

load_dotenv()

# ── Working models confirmed on your account ──────────────────────────────────
MODEL_FALLBACK_CHAIN = [
    "gemini-3.1-flash-lite",     # primary: stable (non-preview), ~1 s per call
    "gemini-3.5-flash-lite",     # fallback: fastest in the probe, ~1.2 s per call
    "gemini-3.1-flash-lite-preview"   # best for short queries, but often unavailable
]

# ── Rate limiting ─────────────────────────────────────────────────────────────
MIN_CALL_GAP_SECONDS = 6.0
MAX_RETRIES          = 3

# ── State ─────────────────────────────────────────────────────────────────────
_client:       genai.Client | None = None
_active_model: str | None          = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        _client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialised")
    return _client


def _extract_text(response) -> str:
    """
    Robustly extract text from a Gemini response object.

    Preview models sometimes return a different response shape than
    stable models. We try every known access pattern in order:
      1. response.text                     — standard stable models
      2. response.candidates[0].content.parts[0].text  — preview models
      3. str(response)                     — absolute last resort

    Returns empty string only if genuinely nothing is there.
    """
    # ── Attempt 1: .text attribute (most common) ──────────────────────────────
    try:
        text = response.text
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    # ── Attempt 2: dig into candidates → content → parts ─────────────────────
    try:
        candidates = response.candidates
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", [])
                for part in parts:
                    text = getattr(part, "text", None)
                    if text and text.strip():
                        return text.strip()
    except Exception:
        pass

    # ── Attempt 3: check for .result or .output (some preview SDKs) ──────────
    for attr in ("result", "output", "generated_text"):
        try:
            text = getattr(response, attr, None)
            if text and str(text).strip():
                return str(text).strip()
        except Exception:
            pass

    # ── Attempt 4: iterate response as iterable (streaming shape) ────────────
    try:
        collected = []
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                collected.append(text)
        if collected:
            return "".join(collected).strip()
    except Exception:
        pass

    logger.warning("Could not extract text from Gemini response — all methods failed")
    return ""


def _parse_retry_delay(error_message: str) -> float:
    match = re.search(r'retry.*?(\d+(?:\.\d+)?)\s*s', str(error_message), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return MIN_CALL_GAP_SECONDS


# Sliding window (see rate_limiter.py): a burst is fine, the Nth+1 call in any
# 60 s waits. N = GEMINI_MAX_CALLS_PER_MIN, default 10 (the old 6 s gap's rate).
_limiter = SlidingWindowLimiter(max_calls_from_env())


def _rate_limit_wait() -> None:
    """Block until the window allows one more call, and count that call."""
    waited = _limiter.acquire()
    if waited > 0.05:
        logger.debug(f"Rate limit: waited {waited:.1f}s for the per-minute window")


def gemini_calls_made() -> int:
    """
    Gemini attempts since startup (retries after a 429 included, because they
    consume quota too). Callers take a before/after difference to learn how many
    calls one request cost.
    """
    return _limiter.total_calls


def _try_call(
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int
) -> str:
    """Single Gemini call attempt. Raises on any error."""
    client = get_client()

    # Every attempt (first try or retry) takes a slot in the window.
    _rate_limit_wait()

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )

    text = _extract_text(response)
    logger.debug(f"Model: {model} | extracted length: {len(text)}")
    return text


def call_gemini(
    prompt: str,
    temperature: float = 0.1,
    max_output_tokens: int = 1000
) -> str:
    """
    Call Gemini with automatic model fallback and retry-with-backoff.
    Retries both 429 (quota) and 503 (server overload).
    Returns "" only if every model and every retry fails.
    """
    global _active_model

    models_to_try = (
        [_active_model] + [m for m in MODEL_FALLBACK_CHAIN if m != _active_model]
        if _active_model
        else MODEL_FALLBACK_CHAIN
    )

    for model in models_to_try:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = _try_call(model, prompt, temperature, max_output_tokens)

                if _active_model != model:
                    logger.info(f"Locked to model: {model}")
                    _active_model = model

                return result

            except Exception as e:
                error_str    = str(e)
                is_quota     = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                is_overload  = "503" in error_str or "UNAVAILABLE" in error_str
                is_not_found = "404" in error_str or "NOT_FOUND" in error_str

                if is_not_found:
                    logger.warning(f"Model {model} not found — skipping")
                    break   # try next model immediately

                if is_quota and attempt == 1:
                    # The 429 body names the exhausted quota (per-minute vs
                    # per-day, per model). Log it once so the real free-tier
                    # limit is visible instead of guessed.
                    logger.warning(f"429 detail from {model}: {error_str[:400]}")

                if (is_quota or is_overload) and attempt < MAX_RETRIES:
                    # For 503, use a fixed backoff since there's no retryDelay in the error
                    wait = _parse_retry_delay(error_str) if is_quota else (attempt * 5.0)
                    logger.warning(
                        f"{'429' if is_quota else '503'} on {model} "
                        f"attempt {attempt}/{MAX_RETRIES} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"Failed on {model} attempt {attempt}: {error_str[:120]}")
                break   # try next model

    logger.error("All models exhausted — returning empty string")
    return ""


# ── Probe utility ─────────────────────────────────────────────────────────────
def find_working_model() -> None:
    """
    Probe each model with a concrete multi-word prompt.
    Using a longer prompt catches truncation bugs that single-word probes miss.
    """
    client = get_client()
    probe  = (
        "You are a helpful assistant. "
        "Complete this sentence in exactly 10 words: "
        "The Ather 450X electric scooter has a maximum range of"
    )

    print("\n" + "="*60)
    print("Gemini Model Availability Test")
    print("="*60)

    for model in MODEL_FALLBACK_CHAIN:
        try:
            response = client.models.generate_content(
                model=model,
                contents=probe,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=50    # enough for 10 words
                )
            )
            text = _extract_text(response)
            status = "OK  " if text else "EMPTY"
            print(f"  {status} → {model}  (response: '{text[:60]}')")
            time.sleep(MIN_CALL_GAP_SECONDS)

        except Exception as e:
            print(f"  FAIL → {model}  ({str(e)[:80]})")
            time.sleep(2)


if __name__ == "__main__":
    find_working_model()