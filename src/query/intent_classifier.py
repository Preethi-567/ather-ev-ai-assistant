# src/query/intent_classifier.py
"""
Intent Classifier — classifies every query before RAG.

Intent classes:
  spec_lookup     → specs, range, speed, weight, voltage
  troubleshooting → something wrong: noise, error, warning light
  charging        → charging port, time, cable, fast charge
  safety          → fire, electric shock, flood, emergency
  out_of_scope    → nothing to do with Ather scooters
"""

import re
from loguru import logger
from src.generation.gemini_client import call_gemini

VALID_INTENTS = {
    "spec_lookup",
    "troubleshooting",
    "charging",
    "safety",
    "out_of_scope"
}

# ── Keyword heuristic tables ──────────────────────────────────────────────────
KEYWORD_RULES: list[tuple[str, set[str]]] = [
    (
        "out_of_scope", {
            "weather", "recipe", "cricket", "movie", "stock", "news",
            "politics", "iphone", "samsung", "pizza", "hotel", "flight",
            "joke", "poem", "capital of", "who is the president",
            "chatgpt", "openai", "google", "amazon", "ipl"
        }
    ),
    (
        "safety", {
            "fire", "smoke", "explosion", "electric shock", "burn",
            "sparks", "dangerous", "emergency", "accident", "crash",
            "flood", "water damage", "overheating", "hot battery",
            "burning smell", "catch fire"
        }
    ),
    (
        "charging", {
            "charge", "charging", "charger", "plug", "cable", "port",
            "fast charge", "charging time", "battery percentage",
            "charging point", "home charging", "public charging",
            "charging station", "charging cable", "charging adapter"
        }
    ),
    (
        "troubleshooting", {
            "not working", "problem", "issue", "error", "warning",
            "red light", "blinking", "noise", "vibration", "stuck",
            "won't start", "wont start", "dead", "no power",
            "display off", "screen off", "strange", "broken", "fault"
        }
    ),
    (
        "spec_lookup", {
            "range", "speed", "top speed", "weight", "dimension",
            "horsepower", "torque", "motor", "battery capacity",
            "specifications", "specs", "kilowatt", "watt", "voltage",
            "wheelbase", "ground clearance", "seat height", "tyre",
            "0 to 60", "acceleration", "ip67", "waterproof"
        }
    ),
]

CLASSIFIER_PROMPT = """You are an intent classifier for an Ather electric scooter assistant.

Classify the user query into EXACTLY ONE of these intents:
- spec_lookup      : asking about specifications, numbers, range, speed, weight
- troubleshooting  : reporting a problem, error, warning light, noise, malfunction
- charging         : asking about charging process, time, cable, port
- safety           : asking about safety risks, fire, electric shock, emergency
- out_of_scope     : completely unrelated to Ather scooters

Rules:
- Reply with ONLY the intent label. Nothing else.
- "out_of_scope" only for queries completely unrelated to Ather scooters.

Examples:
"what is the range of 450X?" → spec_lookup
"my scooter wont start" → troubleshooting
"how do I charge at home?" → charging
"is it safe to ride in rain?" → safety
"who won the IPL?" → out_of_scope
"red light blinking" → troubleshooting

User query: {query}
Intent:"""


def _keyword_classify(query: str) -> str | None:
    """Fast keyword match — zero API cost."""
    query_lower = query.lower().strip()
    for intent, keywords in KEYWORD_RULES:
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                logger.debug(f"Keyword match: '{keyword}' → {intent}")
                return intent
    return None


def _gemini_classify(query: str) -> str:
    """Gemini fallback for ambiguous queries."""
    prompt = CLASSIFIER_PROMPT.format(query=query)

    # temperature=0.0 for deterministic classification
    # max 10 tokens — we only need one word back
    raw = call_gemini(prompt, temperature=0.0, max_output_tokens=20).lower()

    for intent in VALID_INTENTS:
        if intent in raw:
            logger.debug(f"Gemini classified: '{query[:50]}' → {intent}")
            return intent

    logger.warning(f"Gemini returned unexpected: '{raw}' — defaulting to spec_lookup")
    return "spec_lookup"


def classify_intent(query: str) -> dict:
    """
    Classify user query intent.

    Returns:
        {
          "intent":   str,
          "method":   "keyword" | "gemini",
          "skip_rag": bool   ← True if out_of_scope
        }
    """
    query = query.strip()
    if not query:
        return {"intent": "out_of_scope", "method": "empty", "skip_rag": True}

    intent = _keyword_classify(query)
    method = "keyword"

    if intent is None:
        intent = _gemini_classify(query)
        method = "gemini"

    skip_rag = (intent == "out_of_scope")
    logger.info(f"Intent: '{query[:60]}' → {intent} (via {method})")

    return {"intent": intent, "method": method, "skip_rag": skip_rag}


def intent_to_content_filter(intent: str) -> str | None:
    """Map intent → ChromaDB content_type filter for hybrid_search."""
    return {
        "charging":        "procedure",
        "troubleshooting": None,
        "spec_lookup":     "specification",
        "safety":          "warning",
        "out_of_scope":    None,
    }.get(intent)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        "how do I charge my Ather at home?",
        "what is the range of Rizta S?",
        "red light blinking on dashboard",
        "my scooter is making a strange noise",
        "is it safe to ride in heavy rain?",
        "who won IPL yesterday?",
        "warp mode battery capacity",
    ]
    for q in tests:
        r = classify_intent(q)
        print(f"{r['intent']:>15} ({r['method']:>7}) | {q}")