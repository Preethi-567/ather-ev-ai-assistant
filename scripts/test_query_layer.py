# scripts/test_query_layer.py
"""
End-to-end test of the full query layer:
memory → intent classifier → query rewriter
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.memory           import ConversationMemory
from src.query.intent_classifier import classify_intent, intent_to_content_filter
from src.query.query_rewriter   import rewrite_query

OUT_OF_SCOPE_RESPONSE = (
    "I'm only able to help with Ather scooter questions. "
    "For other topics, please use a general search engine."
)

def process_query(query: str, memory: ConversationMemory, session_id: str) -> dict:
    """Simulate the full query processing pipeline."""

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Step 1: Classify intent
    intent_result = classify_intent(query)
    print(f"Intent   : {intent_result['intent']} (via {intent_result['method']})")

    # Step 2: Short-circuit if out_of_scope
    if intent_result["skip_rag"]:
        print(f"Action   : SKIP RAG → return polite redirect")
        return {
            "query":    query,
            "intent":   "out_of_scope",
            "response": OUT_OF_SCOPE_RESPONSE,
            "skip_rag": True
        }

    # Step 3: Rewrite query (pass memory context for follow-ups)
    previous_query = memory.get_last_query(session_id)
    rewrite_result = rewrite_query(query, previous_query=previous_query)
    print(f"Rewritten: {rewrite_result['rewritten']}")
    print(f"Method   : {rewrite_result['method']}")

    # Step 4: Get content_type filter for retrieval
    content_filter = intent_to_content_filter(intent_result["intent"])
    print(f"Filter   : {content_filter}")
    print(f"→ Ready for hybrid_search('{rewrite_result['rewritten']}', filter='{content_filter}')")

    return {
        "query":          query,
        "rewritten":      rewrite_result["rewritten"],
        "intent":         intent_result["intent"],
        "content_filter": content_filter,
        "skip_rag":       False
    }


if __name__ == "__main__":
    memory     = ConversationMemory()
    session_id = "test-session"

    queries = [
        "how do I charge at home?",         # charging → procedure filter
        "how long does it take?",            # follow-up → needs memory
        "what is the range of 450X?",        # spec_lookup → specification filter
        "red light blinking on dashboard",   # troubleshooting
        "who won IPL yesterday?",            # out_of_scope → skip RAG
        "warp mode",                         # fast rewrite
        "is it safe in rain?",               # safety → warning filter
    ]

    for q in queries:
        result = process_query(q, memory, session_id)
        # Simulate adding a turn to memory (in real system, added after generation)
        if not result["skip_rag"]:
            memory.add_turn(session_id, q, "[simulated response]")