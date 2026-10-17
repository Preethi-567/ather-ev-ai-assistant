# src/retrieval/group_merge.py
"""
Merge per-group retrieval results into one context set.

Used when a query maps to several scopes — either because the user named more
than one scooter ("450X vs Rizta Z") or named none, in which case we retrieve
once per series so the answer can cover both product lines.

Kept as a pure function (no model / DB imports) so the policy is unit-testable.

POLICY
------
1. Every group contributes at most `per_group` chunks (its own top-k).
2. The group that owns the globally best chunk is always kept as-is.
3. Every OTHER group keeps only chunks scoring >= `min_score`.
   Why: with a series fallback, a question about the Joey Pocket (Rizta only)
   would otherwise drag five irrelevant 450 chunks into the prompt, and the
   generator would then write "for the 450X I couldn't find that" — which also
   trips the exact-fallback detector downstream and suppresses conversation
   memory for a perfectly good answer.
4. Duplicate chunk_ids across overlapping groups are kept once (highest score).
5. Output is ordered by score, so citation tags [C1].. follow relevance.

UNITS: scores are the raw cross-encoder logits used by hybrid_search today.
If hybrid_search switches to sigmoid output (planned in P3), `min_score` must
be re-based with it — it lives next to CONFIDENCE_FLOOR for that reason.
"""

from typing import Any


def merge_group_results(
    results: list[dict[str, Any]],
    per_group: int,
    min_score: float,
) -> dict[str, Any]:
    """
    Args:
        results:   one hybrid_search() result per group, each shaped
                   {"chunks": [...with "reranker_score"...], "top_score": float, ...}
        per_group: max chunks to keep from each group
        min_score: floor for non-best groups (see policy 3)

    Returns:
        {"chunks": [...], "top_score": float, "best_group": int | None}
    """
    non_empty = [(i, r) for i, r in enumerate(results) if r.get("chunks")]
    if not non_empty:
        return {"chunks": [], "top_score": 0.0, "best_group": None}

    best_group, best = max(non_empty, key=lambda ir: ir[1].get("top_score", float("-inf")))

    merged: dict[str, dict[str, Any]] = {}

    for i, res in non_empty:
        kept = res["chunks"][:per_group]
        if i != best_group:
            kept = [c for c in kept if c.get("reranker_score", float("-inf")) >= min_score]

        for chunk in kept:
            cid = chunk["chunk_id"]
            prev = merged.get(cid)
            if prev is None or chunk.get("reranker_score", 0.0) > prev.get("reranker_score", 0.0):
                merged[cid] = chunk

    ordered = sorted(
        merged.values(),
        key=lambda c: c.get("reranker_score", float("-inf")),
        reverse=True,
    )

    return {
        "chunks": ordered,
        "top_score": best.get("top_score", 0.0),
        "best_group": best_group,
    }
