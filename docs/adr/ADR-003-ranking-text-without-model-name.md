# ADR-003: Rank on text without the model name; rerank on chunk text only

- **Status:** Accepted
- **Date:** 2026-10-19
- **Builds on:** ADR-001 (model-scoped retrieval)

## Context

After ADR-001, a question such as *"how do I charge my Rizta Z?"* returned the
exact fallback sentence even though the Rizta Z guide has a whole "Charge it up"
section (pages 35-39).

A probe over every candidate (`dense_search` and `bm25_search`, top 40 each, then
the production reranker) showed that retrieval was **not** the failure:

| Page | What it is | Dense rank | Final rank |
|---|---|---|---|
| 35 | Plugging it in | 10 | outside top 5 |
| 36 | Charger vents / charging starts | 5 | outside top 5 |
| 37 | Unplugging | 4 | 2 |
| 39 | What does the charger LED say | 7 | outside top 5 |
| 16 | Joey Pocket (storage) | 1 | **1** |

All four charging pages were candidates. The *ordering* was wrong, and the
generator, handed Joey Pocket first, correctly refused to answer from it.

Two causes were identified and then measured, not assumed.

1. **The model name pollutes ranking once the scope is already pinned.** The
   metadata filter restricts every candidate to Rizta Z. The words "Rizta Z" in
   the query therefore cannot tell candidates apart, but they attract pages that
   merely *mention* the name (intro, trademarks, storage).
2. **The reranker input was noisy and wasteful.** It was
   `summary | components | text` cut at 400 characters. The summary is the first
   sentence of the text (shown twice), and "components" are sentence fragments.

## Decision

1. When a scope is set, replace model mentions with the neutral word "scooter"
   in the text used to **rank** (dense query, BM25 query, reranker query).
   Scope detection and the question given to the generator are unchanged.
2. Feed the reranker the chunk text only (up to 1500 characters).

## Evidence (real index, real models, four queries)

| Query | Current: first correct page | Name stripped + text-only |
|---|---|---|
| how do I charge my Rizta Z? | rank 2 (p37 unplugging); plugging-in page not in top 5 | **rank 1** (p35), top score -2.5 -> +7.9 |
| what do the charger light colours mean on my Rizta Z? | rank 1 | rank 1, score 1.6 -> 6.7 |
| where is the Joey Pocket on the Rizta Z? | rank 1 | rank 1 |
| what is the tyre pressure on my 450S? | rank 2 | **rank 1**, score -4.9 -> +5.1 |

No regression on the two queries that already worked; two failures fixed.

## Consequences

- Good matches now score clearly positive on the cross-encoder logit scale, so
  the low-confidence prefix stops firing on correct answers. The floor still
  needs calibrating on the evaluation set (P5); this ADR does not set it.
- BM25 also loses the model token for scoped queries. That is intentional (the
  filter carries the scope) but **one tension is known**: the spec sheet mixes
  450X and 450 Plus columns in one chunk (see TODO(P2) in `model_detector.py`),
  so a query naming "450 Plus" loses that token until P2 splits the table.
- `search()` returns `ranking_query` so the text actually ranked is observable.

## Alternatives considered

- **Keep the name, tune the reranker threshold.** Rejected: the problem is the
  order, not the cut-off.
- **Replace the reranker.** Rejected for now: with clean input it ranked the
  right page first on all four cases. Revisit only if the P5 evaluation shows
  otherwise.
- **Boost by section title.** Deferred: more machinery for a gap the simpler
  change already closed.

## Known limits

Four queries is a small sample; it is enough to justify the change and to pin
it as a regression test, not enough to claim a general quality number. That
claim belongs to the P5 evaluation set (30 questions, RAGAS).
`tests/test_ranking_real_models.py` re-runs these cases against the real
models (opt-in: `RUN_REAL_MODEL_TESTS=1`).

## Interview framing

"Retrieval looked broken, so I measured each stage before touching anything. All
the right pages were retrieved; the ranking was wrong. Two cheap, explainable
changes fixed it, I proved the fix on the real index with a before/after table,
and I pinned it with tests that fail if either change is reverted."
