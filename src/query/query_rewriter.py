# src/query/query_rewriter.py
"""
Query Rewriter — converts informal queries to retrieval-optimised ones.

Why rewrite at all?
  BGE-large-en-v1.5 is trained on well-formed English sentences.
  Users type like they're texting: "red light blinking wtf",
  "range 450x", "charging not working help".

  The embedding distance between "red light blinking" and
  "dashboard warning indicator illuminated red" is much larger
  than it should be semantically — rewriting bridges that gap.

Two modes:
  1. Standalone query  → rewrite assuming no prior context
  2. Follow-up query   → inject last query for coreference resolution
     e.g. "what about the S model?" needs to know the previous
     topic was "charging time" to become a good rewritten query

Also handles:
  - Typo tolerance (Gemini corrects "scootr" → "scooter")
  - Abbreviation expansion ("IP67" → "IP67 water resistance rating")
  - Unit normalisation ("0-40" → "0 to 40 km/h acceleration")
"""

# src/query/intent_classifier.py

"""
Query Rewriter — converts informal queries to retrieval-optimised ones.
Uses the new google-genai SDK via the shared gemini_client module.
"""

from loguru import logger
from src.generation.gemini_client import call_gemini

# ── Fast lookup table — zero API cost ────────────────────────────────────────
FAST_REWRITES: dict[str, str] = {
    "range":          "What is the maximum riding range of the Ather scooter on a full charge?",
    "speed":          "What is the top speed of the Ather scooter?",
    "charge time":    "How long does it take to fully charge the Ather scooter battery?",
    "charging time":  "How long does it take to fully charge the Ather scooter battery?",
    "battery":        "What are the battery specifications and capacity of the Ather scooter?",
    "weight":         "What is the kerb weight of the Ather scooter?",
    "service":        "When should I service my Ather scooter and what does it include?",
    "warranty":       "What is the warranty period for the Ather scooter and battery?",
    "ip67":           "What does the IP67 water resistance rating mean for Ather scooters?",
    "features":       "What are the main features of the Ather scooter?",
    "modes":          "What are the different riding modes available on the Ather scooter?",
    "warp mode":      "What is Warp mode on the Ather scooter and how do I activate it?",
    "eco mode":       "What is Eco mode on the Ather scooter and when should I use it?",
    "atherstack":     "What is AtherStack and what features does it provide?",
    "ather app":      "What can I do with the Ather mobile app?",
    "smart eco":      "What is Smart Eco mode on the Ather scooter?",
    "ride mode":      "What riding modes are available on the Ather scooter?",
}

STANDALONE_REWRITE_PROMPT = """You are a query rewriter for an Ather electric scooter knowledge base.

Convert the user's informal query into a clear, specific, retrieval-optimised question.

Rules:
1. Output ONLY the rewritten query. No explanation, no preamble.
2. Make it a complete, grammatically correct question.
3. Expand abbreviations: "450X" → "Ather 450X"
4. Fix typos silently.
5. Keep it under 25 words.
6. Do NOT invent information — only rephrase what the user said.
7. If already clear and specific, return it unchanged.

Examples:
"red light blinking" → "What does a red warning indicator light on the Ather dashboard mean?"
"charge time" → "How long does it take to fully charge an Ather scooter battery?"
"range 450x" → "What is the maximum riding range of the Ather 450X on a full charge?"
"water ok?" → "Is the Ather scooter safe to ride in rain or waterproof?"

User query: {query}
Rewritten query:"""

FOLLOWUP_REWRITE_PROMPT = """You are a query rewriter for an Ather electric scooter knowledge base.

The user is asking a follow-up. Use conversation context to resolve references.

Previous query: {previous_query}
Current query:  {query}

Rules:
1. Output ONLY the rewritten standalone query.
2. Resolve all references: "it", "that", "this model", "the other one".
3. The rewritten query must make sense WITHOUT conversation context.
4. Keep it under 25 words.

Examples:
Previous: "how do I charge my 450X?"
Current: "how long does it take?"
Rewritten: "How long does it take to fully charge the Ather 450X battery?"

Previous: "what is the range of Rizta S?"
Current: "what about the Z model?"
Rewritten: "What is the maximum riding range of the Ather Rizta Z?"

Rewritten query:"""


def _fast_rewrite(query: str) -> str | None:
    return FAST_REWRITES.get(query.lower().strip())


def _is_followup_query(query: str) -> bool:
    """Heuristic: does this query need prior context to make sense?"""
    q = query.lower().strip()

    # Explicit follow-up starters
    followup_starters = {
        "what about", "how about", "and the", "also the",
        "but what", "what if", "does it", "can it",
        "same for", "what is its"
    }
    if any(q.startswith(s) for s in followup_starters):
        return True

    # Pronouns without a clear referent in the query itself
    if any(q.startswith(p) for p in {"it ", "its ", "they ", "that ", "this "}):
        return True

    # ── NOT a follow-up if query has its own clear subject ───────────────────
    # "is it safe in rain?" has "rain" as its own topic — not a follow-up
    # "how long does it take?" has no topic — genuine follow-up
    # Rule: if query is <= 4 words AND starts with a pronoun → follow-up
    #        if query is <= 4 words but has a concrete noun → standalone
    words = q.split()
    if len(words) <= 4:
        # Check if there's a concrete noun (not just pronouns/articles)
        filler = {"how", "what", "is", "it", "the", "a", "an", "do", "does", "can"}
        content_words = [w for w in words if w not in filler]
        if not content_words:
            return True   # pure function words — definitely a follow-up
        # Has a content word → treat as standalone
        return False

    return False


def _gemini_rewrite(query: str, previous_query: str | None) -> str:
    """Call Gemini to rewrite the query."""
    is_followup = previous_query is not None and _is_followup_query(query)

    if is_followup:
        prompt = FOLLOWUP_REWRITE_PROMPT.format(
            previous_query=previous_query,
            query=query
        )
    else:
        prompt = STANDALONE_REWRITE_PROMPT.format(query=query)

    # temperature=0.2 for natural phrasing variety
    # max 60 tokens — rewritten query must stay short
    rewritten = call_gemini(prompt, temperature=0.2, max_output_tokens=150)

    if not rewritten or len(rewritten.split()) > 40:
        logger.warning(f"Rewriter bad output — using original query")
        return query

    logger.debug(f"Rewritten: '{query[:50]}' → '{rewritten[:80]}'")
    return rewritten


def plan_rewrite(query: str) -> tuple[str, str | None]:
    """
    Decide HOW a query would be rewritten, without calling Gemini.

    Returns (method, text):
        ("fast", "<canned rewrite>")   instant lookup, zero API cost
        ("gemini", None)               needs a Gemini rewrite
        ("passthrough", None)          already clear, keep as typed

    Exposed so the query analyser can tell, before spending any call, whether a
    rewrite call is coming and can fold the intent label into it.
    """
    query = query.strip()
    if not query:
        return "passthrough", None

    fast = _fast_rewrite(query)
    if fast:
        return "fast", fast

    needs_rewriting = (
        len(query.split()) <= 6
        or _is_followup_query(query)
        or query[-1] not in ".?"
    )
    return ("gemini", None) if needs_rewriting else ("passthrough", None)


def rewrite_query(query: str, previous_query: str | None = None) -> dict:
    """
    Rewrite a user query for better retrieval.

    Returns:
        {
          "original":  str,
          "rewritten": str,
          "method":    "fast" | "gemini" | "passthrough",
          "changed":   bool
        }
    """
    query = query.strip()

    # Layers 1 and 2: see plan_rewrite
    method, fast = plan_rewrite(query)
    if method == "fast":
        return {"original": query, "rewritten": fast, "method": "fast", "changed": True}

    if method == "gemini":
        rewritten = _gemini_rewrite(query, previous_query)
    else:
        rewritten = query
        logger.debug(f"Passthrough: '{query[:60]}'")

    return {
        "original":  query,
        "rewritten": rewritten,
        "method":    method,
        "changed":   rewritten != query
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("range",                       None),
        ("red light blinking",          None),
        ("how long does it take?",      "how do I charge at home?"),
        ("what about the Z model?",     "what is the range of Rizta S?"),
        ("warp mode",                   None),
        ("is it waterproof?",           None),
    ]
    for query, prev in tests:
        r = rewrite_query(query, previous_query=prev)
        print(f"\nOriginal  : {r['original']}")
        if prev:
            print(f"Previous  : {prev}")
        print(f"Rewritten : {r['rewritten']}")
        print(f"Method    : {r['method']}")