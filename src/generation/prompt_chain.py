# src/generation/prompt_chain.py
"""
3-Step Anti-Hallucination Generation Chain

Step 1 — Sufficiency Check
    Send chunk summaries to Gemini.
    Ask: can you answer this question from this context alone?
    If NO → return FALLBACK_RESPONSE immediately. Do not proceed.
    This is the primary hallucination gate.

Step 2 — Answer Generation
    Build full prompt: system prompt + memory + context chunks + query.
    Call Gemini at temperature=0.1 for near-deterministic answers.
    Citations are written by the model following Rule 3 in SYSTEM_PROMPT.

Step 3 — Citation Rendering (deterministic)
    The model sees chunk TAGS ([C1]..) and the scooter model, never page numbers.
    It cites by tag; code maps each tag to (model, page) from chunk metadata.
    A tag with no retrieved chunk behind it is stripped. As a second layer, any
    free-text "page N" the model writes anyway is removed unless N was retrieved.
    If reranker confidence is low → prepend LOW_CONFIDENCE_PREFIX.
"""

import time
from loguru import logger

from src.generation.gemini_client import call_gemini
from src.generation.prompts import (
    SYSTEM_PROMPT,
    SUFFICIENCY_PROMPT,
    GENERATION_PROMPT,
    FALLBACK_RESPONSE,
    LOW_CONFIDENCE_PREFIX,
)
from src.generation.citations import (
    render_citations,
    strip_invalid_free_text_pages,
)
from src.query.memory import ConversationMemory, get_memory

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIDENCE_FLOOR = 0.4     # reranker score below this → prepend low-confidence warning


# ── Context builder ───────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    """
    Convert retrieved chunks into the context string for the generation prompt.

    Each block shows the model:
      - a citation TAG  [C1], [C2], ...   (position in `chunks`; see citations.py)
      - the scooter MODEL the chunk comes from
      - section, content type, summary, component names, full text

    Deliberately NOT shown: page number and file name. The model cannot cite or
    misquote what it cannot see; the page is attached by code after generation.
    """
    blocks = []

    for i, chunk in enumerate(chunks, 1):
        meta            = chunk.get("metadata", {})
        summary         = chunk.get("summary", "").strip()
        component_names = chunk.get("component_names", [])

        lines = [f"--- [C{i}] ---"]
        lines.append(f"Model     : {meta.get('vehicle_model', 'Unknown')}")
        lines.append(f"Section   : {meta.get('section', 'N/A')}")
        lines.append(f"Type      : {meta.get('content_type', 'general')}")

        if summary:
            lines.append(f"Summary   : {summary}")

        if component_names:
            lines.append(f"Components: {', '.join(component_names)}")

        lines.append(f"Text:\n{chunk['text']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _build_summaries_block(chunks: list[dict]) -> str:
    """
    Compact summaries-only block for the sufficiency check (Step 1).

    Why summaries instead of full text? The check only needs to know IF the topic
    is covered; full text wastes tokens on a YES/NO question.
    Tags and model are included so the check sees the same scope as generation.
    """
    lines = []
    for i, chunk in enumerate(chunks, 1):
        summary = chunk.get("summary", "").strip()
        model   = chunk.get("metadata", {}).get("vehicle_model", "Unknown")

        # Fall back to first 120 chars of text if no summary
        if not summary:
            summary = chunk["text"][:120].strip()

        lines.append(f"{i}. [C{i}] ({model}) {summary}")

    return "\n".join(lines)


# ── Main chain ────────────────────────────────────────────────────────────────

class PromptChain:
    """
    Executes the 3-step generation pipeline.
    Instantiate once at app startup — reuse across all requests.
    """

    def __init__(self):
        self.memory = get_memory()
        logger.info("PromptChain initialised")

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def _check_sufficiency(self, query: str, chunks: list[dict]) -> bool:
        """
        Ask Gemini: can this question be answered from these summaries?

        Failure policy:
        - If Gemini returns YES → proceed
        - If Gemini returns NO  → fallback (honest "not found" response)
        - If Gemini returns ""  (503/timeout) → PROCEED anyway
        Reasoning: we already retrieved relevant chunks via the reranker.
        A network hiccup on the sufficiency check should not block a
        valid answer. The generation step will naturally produce the
        fallback phrase if context truly doesn't support an answer.
        """
        summaries = _build_summaries_block(chunks)

        prompt = SUFFICIENCY_PROMPT.format(
            context=summaries,
            query=query
        )

        response = call_gemini(prompt, temperature=0.0, max_output_tokens=5)

        # Empty response = Gemini unavailable = give benefit of the doubt
        if not response:
            logger.warning(
                "Sufficiency check: Gemini unavailable — "
                "proceeding with generation (benefit of the doubt)"
            )
            return True

        can_answer = "YES" in response.upper()
        logger.info(
            f"Sufficiency check: '{response.strip()}' → "
            f"{'PROCEED' if can_answer else 'FALLBACK'}"
        )
        return can_answer

    # ── Step 2 ────────────────────────────────────────────────────────────────

    def _generate_answer(
        self,
        query: str,
        chunks: list[dict],
        session_id: str
    ) -> str:
        """
        Generate the answer using full context chunks + conversation memory.
        """
        context_block  = _build_context_block(chunks)
        memory_context = self.memory.get_context(session_id)

        prompt = GENERATION_PROMPT.format(
            system_prompt = SYSTEM_PROMPT,
            memory        = memory_context if memory_context else "(No previous conversation)",
            context       = context_block,
            query         = query
        )

        # temperature=0.1 → near-deterministic, slight variation for natural phrasing
        # max 800 tokens → enough for detailed answers with steps
        answer = call_gemini(prompt, temperature=0.1, max_output_tokens=800)

        if not answer:
            logger.warning("Generation returned empty — using fallback")
            return FALLBACK_RESPONSE

        return answer

    # ── Step 3 ────────────────────────────────────────────────────────────────

    def _apply_confidence_prefix(
        self,
        answer: str,
        low_confidence: bool
    ) -> str:
        """Prepend uncertainty warning if reranker score was below threshold."""
        if low_confidence:
            logger.warning("Low confidence flag set — prepending warning prefix")
            return LOW_CONFIDENCE_PREFIX + answer
        return answer

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        retrieval_result: dict,
        session_id: str = "default"
    ) -> dict:
        """
        Execute the full 3-step generation pipeline.

        Args:
            query:            Rewritten user query from QueryRewriter
            retrieval_result: Output from hybrid_search() containing:
                              - chunks:         list of top-5 chunk dicts
                              - top_score:      float reranker confidence
                              - low_confidence: bool
            session_id:       For conversation memory

        Returns:
            {
              "answer":         str,   final answer to show the user
              "fallback_used":  bool,  True if sufficiency check failed
              "cited_pages":    list,  page numbers of verified citations
              "sources":        list[Source]  verified (model, page, file) citations
              "low_confidence": bool,
              "latency_ms":     int
            }
        """
        start_time = time.time()

        chunks       = retrieval_result.get("chunks", [])
        low_conf     = retrieval_result.get("low_confidence", False)
        top_score    = retrieval_result.get("top_score", 0.0)

        logger.info(
            f"PromptChain.run | query='{query[:60]}' | "
            f"chunks={len(chunks)} | top_score={top_score:.3f} | "
            f"low_conf={low_conf} | session={session_id}"
        )

        # ── Guard: no chunks retrieved ────────────────────────────────────────
        if not chunks:
            logger.warning("No chunks passed to PromptChain — returning fallback")
            return {
                "answer":         FALLBACK_RESPONSE,
                "fallback_used":  True,
                "cited_pages":    [],
                "sources":        [],
                "low_confidence": True,
                "latency_ms":     int((time.time() - start_time) * 1000),
            }

        # ── Step 1: Sufficiency check ─────────────────────────────────────────
        can_answer = self._check_sufficiency(query, chunks)

        if not can_answer:
            return {
                "answer":         FALLBACK_RESPONSE,
                "fallback_used":  True,
                "cited_pages":    [],
                "sources":        [],
                "low_confidence": False,
                "latency_ms":     int((time.time() - start_time) * 1000),
            }

        # ── Step 2: Generate answer ───────────────────────────────────────────
        raw_answer = self._generate_answer(query, chunks, session_id)

        # ── Step 3: Render citations ──────────────────────────────────────────
        # Layer 2 first (on the model's raw text, so our own rendered
        # "page N" strings are never re-validated), then tags → citations.
        raw_answer, removed_pages = strip_invalid_free_text_pages(raw_answer, chunks)
        clean_answer, sources, unknown_tags = render_citations(raw_answer, chunks)
        cited_pages = sorted({s.page for s in sources})

        if removed_pages:
            logger.warning(f"Citation layer 2: removed free-text pages {removed_pages}")
        if unknown_tags:
            logger.warning(
                f"Citation render: model used unknown tags {unknown_tags} "
                f"(retrieved {len(chunks)} chunks) — stripped"
            )

        # ── Apply confidence prefix if needed ─────────────────────────────────
        final_answer = self._apply_confidence_prefix(clean_answer, low_conf)

        # ── Detect self-generated fallback ────────────────────────────────
        # The model may write the fallback phrase itself when context is
        # insufficient — this counts as a fallback even though generation
        # technically succeeded
        is_self_fallback = FALLBACK_RESPONSE.strip() in final_answer

        # Only update memory if we gave a real answer
        if not is_self_fallback:
            self.memory.add_turn(session_id, query, final_answer)
            if not sources:
                # Rule 3 says every answer is cited. Not enforced here (P1 only
                # reports it); the eval run should count these before we decide
                # whether to reject uncited answers.
                logger.warning("Answer has NO verified citation — Rule 3 violated")

        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"PromptChain complete | latency={latency_ms}ms | "
            f"cited_pages={cited_pages} |"
            f"fallback= {is_self_fallback} | low_conf={low_conf}"
        )

        return {
            "answer":         final_answer,
            "fallback_used":  is_self_fallback,
            "cited_pages":    cited_pages,
            "sources":        sources,
            "low_confidence": low_conf,
            "latency_ms":     latency_ms,
        }
    



# ── Module-level singleton ────────────────────────────────────────────────────
_chain: PromptChain | None = None

def get_prompt_chain() -> PromptChain:
    global _chain
    if _chain is None:
        _chain = PromptChain()
    return _chain