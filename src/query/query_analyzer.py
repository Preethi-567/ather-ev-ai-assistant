# src/query/query_analyzer.py
"""
One entry point for "what is this query, and how should it be searched?".

THE COST IT REMOVES
-------------------
Before this module the API called classify_intent() and then rewrite_query().
When no keyword matched AND the query needed rewriting (typical for short
follow-ups like "how long does it take?"), that was TWO sequential Gemini calls
for two small jobs that read the same sentence. Every Gemini call is a quota unit
on a free-tier key and a round trip of latency.

DECISION TABLE (Gemini calls spent before retrieval)
----------------------------------------------------
    keyword intent found   rewrite needed    calls
    ---------------------  ----------------  -----
    out_of_scope keyword   (not attempted)     0     skip RAG immediately
    yes                    fast / passthrough  0
    yes                    gemini              1     rewrite only
    no                     fast / passthrough  1     classify only
    no                     gemini              1     ONE merged call (was 2)

The merged call returns a single JSON object {"intent": ..., "rewritten": ...}.
If the model answers but the JSON cannot be used, the code falls back to the two
original single-purpose calls, so a formatting slip costs calls, never correctness.
If the model answers with nothing at all (Gemini down), the safe defaults are used
and no further calls are spent into an outage.

Return shape matches the old pair so callers change one line:
    intent_result  = {"intent", "method", "skip_rag"}
    rewrite_result = {"original", "rewritten", "method", "changed"}  or None when
    the query is out of scope (there is nothing to retrieve, so nothing to rewrite).
"""

import json
import re

from loguru import logger

from src.generation.gemini_client import call_gemini
from src.query.intent_classifier import (
    VALID_INTENTS,
    classify_by_keywords,
    classify_intent,
)
from src.query.query_rewriter import plan_rewrite, rewrite_query

DEFAULT_INTENT = "spec_lookup"      # same default the single-purpose classifier uses
MAX_REWRITE_WORDS = 40              # same bound the single-purpose rewriter uses

MERGED_PROMPT = """You are the query analyser for an Ather electric scooter assistant.

Do TWO jobs on the user query and reply with ONE JSON object and nothing else:
{"intent": "<label>", "rewritten": "<question>"}

JOB 1 - "intent": choose EXACTLY ONE label.
- spec_lookup      : specifications, numbers, range, speed, weight, tyre pressure
- troubleshooting  : a problem, error, warning light, noise, malfunction
- charging         : the charging process, time, cable, port
- safety           : safety risks, fire, electric shock, emergency
- out_of_scope     : completely unrelated to Ather scooters

JOB 2 - "rewritten": a clear, complete, standalone question for searching a manual.
- Fix typos silently. Keep it under 25 words.
- Expand model shorthand only as typed: "450X" -> "Ather 450X".
- Never add a scooter model the user did not mention{followup_model_rule}.
- Do not invent information; only rephrase what the user said.
- If the query is already clear, return it unchanged.
- If the intent is out_of_scope, return the query unchanged.
{followup_block}
Examples:
"red light blinking" -> {"intent": "troubleshooting", "rewritten": "What does a red warning indicator light on the Ather dashboard mean?"}
"range 450x" -> {"intent": "spec_lookup", "rewritten": "What is the maximum riding range of the Ather 450X on a full charge?"}
"who won the IPL?" -> {"intent": "out_of_scope", "rewritten": "who won the IPL?"}

User query: {query}
JSON:"""

# The previous query is ALWAYS offered when there is one, and the model decides
# whether the current query needs it. The old keyword heuristic
# (_is_followup_query) misses ordinary follow-ups: "how long does it take?" has
# five words and no pronoun at the start, so it was rewritten with no context at
# all and the topic was guessed. Deciding in the same call removes that fragile
# rule; the explicit "ignore it if self-contained" line stops a standalone
# question from absorbing the previous topic.
_FOLLOWUP_BLOCK = """
CONVERSATION CONTEXT - previous query: {previous}
If the current query is NOT self-contained (it leans on "it", "that", "this
model", "the other one", or has no subject of its own), resolve it using the
previous query so the rewritten question stands alone.
If the current query is already self-contained, IGNORE the previous query.
"""


def _build_prompt(query: str, previous_query: str | None) -> str:
    # str.replace, not str.format: user text may contain braces.
    has_previous = bool(previous_query)
    block = _FOLLOWUP_BLOCK.replace("{previous}", previous_query or "") if has_previous else ""
    rule = " (or that the previous query named, when resolving a follow-up)" if has_previous else ""
    return (
        MERGED_PROMPT
        .replace("{followup_model_rule}", rule)
        .replace("{followup_block}", block)
        .replace("{query}", query)
    )


def _parse_merged(raw: str) -> tuple[str, str] | None:
    """
    Pull (intent, rewritten) out of the model's reply.
    Tolerates ```json fences and stray prose around the object.
    Returns None when the reply is unusable (no object, bad JSON, unknown intent).
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    intent = str(data.get("intent", "")).strip().lower()
    if intent not in VALID_INTENTS:
        return None
    rewritten = str(data.get("rewritten", "")).strip().strip('"').strip()
    return intent, rewritten


def _result(intent: str, method: str, query: str, rewritten: str):
    intent_result = {"intent": intent, "method": method, "skip_rag": intent == "out_of_scope"}
    if intent_result["skip_rag"]:
        return intent_result, None
    rewrite_result = {
        "original": query,
        "rewritten": rewritten,
        "method": method,
        "changed": rewritten != query,
    }
    return intent_result, rewrite_result


def _merged_call(query: str, previous_query: str | None):
    """
    One Gemini call for intent + rewrite.

    Returns (intent_result, rewrite_result) on success or on a model outage
    (safe defaults), and None when the model replied but the reply was unusable
    (the caller then falls back to the two single-purpose calls).
    """
    raw = call_gemini(_build_prompt(query, previous_query), temperature=0.1, max_output_tokens=250)

    if not raw:
        logger.warning("Merged analysis: Gemini unavailable — using defaults, no further calls")
        return _result(DEFAULT_INTENT, "default", query, query)

    parsed = _parse_merged(raw)
    if parsed is None:
        logger.warning(f"Merged analysis: unusable reply {raw[:80]!r} — falling back to two calls")
        return None

    intent, rewritten = parsed
    if not rewritten or len(rewritten.split()) > MAX_REWRITE_WORDS:
        logger.warning("Merged analysis: bad rewrite — keeping the original query")
        rewritten = query

    logger.info(f"Intent: '{query[:60]}' → {intent} (via gemini_merged)")
    logger.debug(f"Rewritten: '{query[:50]}' → '{rewritten[:80]}'")
    return _result(intent, "gemini_merged", query, rewritten)


def analyze_query(query: str, previous_query: str | None = None):
    """
    Classify and rewrite a query with the fewest Gemini calls (see the table above).

    Returns (intent_result, rewrite_result). rewrite_result is None exactly when
    intent_result["skip_rag"] is True.
    """
    query = query.strip()

    # Free path first: keywords.
    intent_result = classify_by_keywords(query)
    if intent_result is not None and intent_result["skip_rag"]:
        return intent_result, None

    method, _ = plan_rewrite(query)

    # Both jobs need Gemini → one merged call.
    if intent_result is None and method == "gemini":
        merged = _merged_call(query, previous_query)
        if merged is not None:
            return merged
        # Unusable reply: fall through to the two single-purpose calls.

    if intent_result is None:
        intent_result = classify_intent(query)        # at most one call
        if intent_result["skip_rag"]:
            return intent_result, None

    return intent_result, rewrite_query(query, previous_query=previous_query)   # at most one call
