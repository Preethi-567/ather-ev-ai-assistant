# src/retrieval/hybrid_search.py
"""
Hybrid Retrieval: Dense + BM25 → RRF Fusion → Cross-Encoder Reranker → Top 5

V2 upgrades:
  - Reranker input enriched with summary + component_names from V2 chunks
  - Component-boosted BM25 results treated with higher initial RRF weight
  - Final results carry all V2 fields for the generation layer
  - Confidence score exposed for prompt_chain's threshold check
"""

from typing import Sequence

from loguru import logger
from sentence_transformers import CrossEncoder

from src.retrieval.dense_search import dense_search
from src.retrieval.bm25_search  import bm25_search
from src.retrieval.group_merge  import merge_group_results

# ── Constants ────────────────────────────────────────────────────────────────
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K            = 60       # RRF constant — higher = smoother rank fusion
DENSE_TOP_K      = 10       # candidates from dense search
BM25_TOP_K       = 10       # candidates from BM25 search
FINAL_TOP_K      = 5        # results after reranking
# NEW — cross-encoder/ms-marco-MiniLM-L-6-v2 logit scale:
#   > 3.0  = high confidence (good match)
#   0 to 3 = moderate confidence
#   < 0    = poor match (chunks probably not relevant)
CONFIDENCE_FLOOR = 0.0      # below this → prepend uncertainty warning

# ── Multi-scope retrieval (several scooters / both series) ───────────────────
# Same logit scale as CONFIDENCE_FLOOR. Non-best groups must clear this to
# contribute chunks (see group_merge.py). When P3 moves to sigmoid scores,
# re-base this together with CONFIDENCE_FLOOR.
GROUP_MIN_RERANK    = 0.0
MAX_CONTEXT_CHUNKS  = 6     # total chunks sent to the generator across groups
MIN_PER_GROUP       = 2


class HybridSearcher:
    """
    Orchestrates the full hybrid retrieval pipeline.
    Instantiate once at startup — the CrossEncoder is ~80MB.
    """

    def __init__(self):
        logger.info(f"Loading reranker: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("HybridSearcher ready")

    def _rrf_score(self, rank: int, k: int = RRF_K) -> float:
        """
        Reciprocal Rank Fusion score.
        rank is 0-indexed.  RRF(rank) = 1 / (k + rank + 1)
        Lower rank (better position) → higher score.
        """
        return 1.0 / (k + rank + 1)

    def _fuse(
        self,
        dense_results: list[dict],
        bm25_results: list[dict]
    ) -> list[dict]:
        """
        Merge dense and BM25 result lists using Reciprocal Rank Fusion.

        Why RRF instead of score normalisation?
        - Dense scores (cosine similarity) and BM25 scores (TF-IDF variant)
          are on completely different scales — you can't add them directly.
        - RRF only cares about rank position, not raw score magnitude.
        - Proven effective in hybrid retrieval literature (Cormack et al. 2009).

        V2 twist: component-boosted BM25 results get a 10% RRF bonus
        to reflect the high-confidence component name match.
        """
        scores: dict[str, float]   = {}  # chunk_id → fused RRF score
        chunks: dict[str, dict]    = {}  # chunk_id → chunk data

        # ── Dense RRF ────────────────────────────────────────────────────────
        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + self._rrf_score(rank)
            chunks[cid] = result

        # ── BM25 RRF ─────────────────────────────────────────────────────────
        for rank, result in enumerate(bm25_results):
            cid       = result["chunk_id"]
            rrf_score = self._rrf_score(rank)

            # V2: component-boosted results get a 10% bonus
            if result.get("component_boosted", False):
                rrf_score *= 1.10

            scores[cid] = scores.get(cid, 0.0) + rrf_score

            # Prefer the dense version if it exists (has more metadata)
            if cid not in chunks:
                chunks[cid] = result

        # ── Sort by fused RRF score ───────────────────────────────────────────
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Return top 20 for reranking (reranker narrows to FINAL_TOP_K)
        return [chunks[cid] for cid, _ in fused[:20]]

    def _build_reranker_input(self, query: str, chunk: dict) -> str:
        """
        Build the text fed to the cross-encoder for reranking.

        V2 upgrade: we prepend the summary and any component names.
        The cross-encoder sees a richer document representation:
          [summary] | [component names] | [full text]
        This helps it score component-page chunks more accurately
        when the query mentions a specific part.
        """
        parts = []

        summary = chunk.get("summary", "").strip()
        if summary:
            parts.append(summary)

        component_names = chunk.get("component_names", [])
        if component_names:
            parts.append("Components: " + ", ".join(component_names))

        parts.append(chunk["text"])

        # Truncate to 400 chars — cross-encoder has max_length=512 tokens
        doc_text = " | ".join(parts)[:400]
        return doc_text

    def search(
        self,
        query: str,
        vehicle_models: Sequence[str] | None = None,
        content_type_filter: str | None = None
    ) -> dict:
        """
        Full hybrid search pipeline for ONE scope.

        Args:
            query:               User query (already rewritten by QueryRewriter)
            vehicle_models:      Restrict to these model tags (None = no restriction)
            content_type_filter: Optional — 'warning'|'procedure'|'specification'

        Returns:
            {
              "chunks":     list of top-5 chunk dicts,
              "top_score":  float — reranker score of best chunk,
              "low_confidence": bool — True if top_score < CONFIDENCE_FLOOR
            }
        """
        logger.info(f"Hybrid search: '{query[:70]}'")

        # ── Step 1: Parallel retrieval ────────────────────────────────────────
        dense_results = dense_search(
            query, top_k=DENSE_TOP_K,
            vehicle_models=vehicle_models,
            content_type_filter=content_type_filter
        )
        bm25_results = bm25_search(
            query, top_k=BM25_TOP_K,
            vehicle_models=vehicle_models
        )

        logger.debug(
            f"Retrieved: {len(dense_results)} dense, {len(bm25_results)} BM25"
        )

        # ── Step 2: RRF Fusion ───────────────────────────────────────────────
        fused_candidates = self._fuse(dense_results, bm25_results)

        if not fused_candidates:
            logger.warning("Hybrid search: no candidates after fusion")
            return {"chunks": [], "top_score": 0.0, "low_confidence": True}

        # ── Step 3: Cross-encoder Reranking ──────────────────────────────────
        # Build (query, document) pairs for the cross-encoder
        reranker_inputs = [
            (query, self._build_reranker_input(query, chunk))
            for chunk in fused_candidates
        ]

        reranker_scores = self.reranker.predict(reranker_inputs)

        # Attach scores and sort
        for chunk, score in zip(fused_candidates, reranker_scores):
            chunk["reranker_score"] = float(score)

        reranked = sorted(
            fused_candidates,
            key=lambda c: c["reranker_score"],
            reverse=True
        )

        top_results  = reranked[:FINAL_TOP_K]
        top_score    = top_results[0]["reranker_score"] if top_results else 0.0
        low_conf     = top_score < CONFIDENCE_FLOOR

        if low_conf:
            logger.warning(
                f"Low confidence: top reranker score {top_score:.3f} "
                f"< threshold {CONFIDENCE_FLOOR}"
            )

        logger.info(
            f"Hybrid search complete: {len(top_results)} final chunks, "
            f"top score: {top_score:.3f} (logit scale), "
            f"low_confidence: {low_conf}"
)

        return {
            "chunks":         top_results,
            "top_score":      round(top_score, 4),
            "low_confidence": low_conf,
        }


    def search_grouped(
        self,
        query: str,
        groups: Sequence[Sequence[str]],
        content_type_filter: str | None = None
    ) -> dict:
        """
        One hybrid search per scope group, merged into a single context set.

        groups:
            []                     → unrestricted search
            [[450S]]               → plain single-scope search (5 chunks)
            [[450X],[Rizta Z]]     → one search per group, merged (see group_merge)
            [[450X,450S],[Rizta…]] → the "no model named" series fallback

        COST: n_groups × (embed + dense + BM25 + rerank). With the series
        fallback that is 2×; measure against the 3 s target in the eval run.
        An obvious optimisation (embed the query once and pass it down) is left
        out on purpose — correctness first, then profile.
        """
        if not groups:
            return self.search(query, None, content_type_filter)

        if len(groups) == 1:
            res = self.search(query, list(groups[0]), content_type_filter)
            res["groups_searched"] = 1
            return res

        per_group = max(MIN_PER_GROUP, MAX_CONTEXT_CHUNKS // len(groups))
        per_group_results = [
            self.search(query, list(g), content_type_filter) for g in groups
        ]

        merged = merge_group_results(
            per_group_results, per_group=per_group, min_score=GROUP_MIN_RERANK
        )

        top_score = merged["top_score"]
        low_conf  = (not merged["chunks"]) or top_score < CONFIDENCE_FLOOR

        logger.info(
            f"Grouped search: {len(groups)} groups → {len(merged['chunks'])} chunks "
            f"(best group #{merged['best_group']}, top score {top_score:.3f})"
        )

        return {
            "chunks":          merged["chunks"],
            "top_score":       round(top_score, 4),
            "low_confidence":  low_conf,
            "groups_searched": len(groups),
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_searcher: HybridSearcher | None = None

def get_hybrid_searcher() -> HybridSearcher:
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher


def hybrid_search(
    query: str,
    vehicle_models: Sequence[str] | None = None,
    content_type_filter: str | None = None
) -> dict:
    """Convenience function — single scope, uses module-level singleton."""
    return get_hybrid_searcher().search(query, vehicle_models, content_type_filter)


def hybrid_search_grouped(
    query: str,
    groups: Sequence[Sequence[str]],
    content_type_filter: str | None = None
) -> dict:
    """Convenience function — one search per scope group, merged."""
    return get_hybrid_searcher().search_grouped(query, groups, content_type_filter)


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = hybrid_search("what does the red warning light on dashboard mean?")

    print(f"\nTop score: {result['top_score']}")
    print(f"Low confidence: {result['low_confidence']}")

    for i, chunk in enumerate(result["chunks"], 1):
        print(
            f"\n#{i} [reranker={chunk['reranker_score']:.3f}] "
            f"Page {chunk['metadata'].get('page_number')} | "
            f"{chunk['metadata'].get('section')}"
        )
        print(f"   keywords:   {chunk.get('keywords', [])}")
        print(f"   components: {chunk.get('component_names', [])}")
        print(f"   text: {chunk['text'][:120]}...")