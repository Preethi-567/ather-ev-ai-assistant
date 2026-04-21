# scripts/test_generation.py
"""
End-to-end generation test — queries matched to actual manual content.

Confirmed sections in your 5 PDFs:
  - Charging: Ather Duo, Ather Dot procedures
  - Riding modes: Eco, Ride, Sport, Warp (MULTI-MODE section)
  - Switchgear and controls
  - Regenerative braking
  - Precautions and warnings
  - Troubleshooting
  - Specifications: range, speed, weight (450X spec sheet)
  - Accelerator: Magic Twist feature
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

from src.query.intent_classifier  import classify_intent, intent_to_content_filter
from src.query.query_rewriter     import rewrite_query
from src.query.memory             import get_memory
from src.retrieval.hybrid_search  import hybrid_search
from src.generation.prompt_chain  import get_prompt_chain
from src.generation.prompts       import OUT_OF_SCOPE_RESPONSE, FALLBACK_RESPONSE

DIVIDER = "=" * 65


def run_query(raw_query: str, session_id: str, memory, chain) -> None:
    print(f"\n{DIVIDER}")
    print(f"Query: {raw_query}")
    print(DIVIDER)

    # ── Intent ────────────────────────────────────────────────────────────
    intent_result = classify_intent(raw_query)
    intent        = intent_result["intent"]
    print(f"Intent   : {intent} (via {intent_result['method']})")

    if intent_result["skip_rag"]:
        print(f"Action   : SKIP RAG")
        print(f"\nAnswer:\n{OUT_OF_SCOPE_RESPONSE}")
        return

    # ── Rewrite ───────────────────────────────────────────────────────────
    previous_query = memory.get_last_query(session_id)
    rewrite_result = rewrite_query(raw_query, previous_query=previous_query)
    rewritten      = rewrite_result["rewritten"]
    print(f"Rewritten: {rewritten}")

    # ── Retrieve ──────────────────────────────────────────────────────────
    content_filter   = intent_to_content_filter(intent)
    print(f"Filter   : {content_filter}")

    retrieval_result = hybrid_search(
        query               = rewritten,
        content_type_filter = content_filter
    )

    chunks    = retrieval_result["chunks"]
    top_score = retrieval_result["top_score"]
    low_conf  = retrieval_result["low_confidence"]

    print(f"Retrieved: {len(chunks)} chunks | top_score={top_score:.3f} | low_conf={low_conf}")
    if chunks:
        for i, c in enumerate(chunks, 1):
            meta = c.get("metadata", {})
            print(
                f"  #{i} Page {str(meta.get('page_number', '?')):>3} | "
                f"{str(meta.get('content_type', '')):>13} | "
                f"{str(meta.get('section', ''))[:45]}"
            )

    # ── Generate ──────────────────────────────────────────────────────────
    result = chain.run(
        query            = rewritten,
        retrieval_result = retrieval_result,
        session_id       = session_id
    )

    # Detect if the model itself wrote the fallback phrase
    # (generation succeeded technically but context was insufficient)
    answer_is_fallback = FALLBACK_RESPONSE.strip() in result["answer"]

    print(f"\nLatency  : {result['latency_ms']}ms")
    print(f"Fallback : {result['fallback_used'] or answer_is_fallback}")
    print(f"Pages    : {result['cited_pages']}")
    print(f"\n{'─'*65}")
    print("ANSWER:")
    print(f"{'─'*65}")
    print(result["answer"])


def main():
    memory     = get_memory()
    chain      = get_prompt_chain()
    session_id = "test-e2e-session"

    test_queries = [

        # ── Charging (Duo / Dot procedures — confirmed in chunks) ─────────
        "how do I use the Ather Duo charger?",
        "how do I use the Ather Dot charger?",
        "what should I check before charging?",       # CAUTION/WARNING sections

        # ── Riding modes (MULTI-MODE section confirmed) ───────────────────
        "what riding modes are available on the Ather 450X?",
        "what is warp mode and when should I use it?",
        "what is eco mode on my Ather scooter?",

        # ── Controls (SWITCHGEAR / MAGIC TWIST confirmed) ─────────────────
        "how does the Magic Twist accelerator work?",
        "what are the switchgear controls on the Ather?",

        # ── Regenerative braking (REGENERATIVE section confirmed) ─────────
        "what is regenerative braking on the Ather scooter?",

        # ── Specifications (450X spec sheet confirmed) ────────────────────
        "what is the top speed of the Ather 450X?",
        "what is the battery capacity of Ather 450X?",
        "what is the kerb weight of the Ather 450X?",

        # ── Safety / Precautions (PRECAUTIONS FOR A confirmed) ────────────
        "what precautions should I follow while riding?",
        "what are the safety warnings for the Ather scooter?",

        # ── Troubleshooting (TROUBLESHOOTING section confirmed) ───────────
        "what should I do if my scooter does not start?",

        # ── Follow-up (tests conversation memory) ─────────────────────────
        "how do I switch between them?",               # follows riding modes query

        # ── Out of scope (should skip RAG entirely) ───────────────────────
        "what is the price of Ather 450X?",            # not in manuals
        "who won IPL yesterday?",                      # completely unrelated
    ]

    for query in test_queries:
        run_query(query, session_id, memory, chain)
        print()


if __name__ == "__main__":
    main()