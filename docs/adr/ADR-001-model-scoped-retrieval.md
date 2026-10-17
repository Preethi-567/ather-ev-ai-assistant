# ADR-001: Scope retrieval to the rider's scooter model

**Status:** Proposed (P1) · **Date:** 2026-10-17

## Context
The index holds five documents for four scooters (450X, 450S, Rizta S, Rizta Z).
`/query` never passed a model filter, and the generator was never told which
scooter a chunk came from. Four of the five documents contain near-identical
tyre-pressure pages, so a 450S question could be answered, and cited, from a
Rizta chunk with nothing in the pipeline able to notice.

## Decision
Resolve a **model scope** for every query, then retrieve inside it:

1. **Explicit** mention in the query (original text first, rewritten text second).
2. Else the **session's** last explicitly named model ("sticky").
3. Else **no scope**: search each product series separately and merge, so an
   unqualified question ("how much boot space?") is answered for both series.

Each mention becomes its own retrieval *group* (`450X vs Rizta Z` → two searches),
so one scooter cannot crowd the other out of the top-k. Non-best groups must clear a
relevance floor to contribute, so an irrelevant series does not pad the prompt.
Detection is deterministic regex, not an LLM call.

## Consequences
- (+) A named model can no longer be answered from another model's manual. The
  filter is enforced twice: in ChromaDB, and again in Python if the DB-side
  filter errors (the previous code silently retried *unfiltered*).
- (+) No extra LLM call, no extra latency for detection.
- (−) Unscoped queries cost two retrievals + two rerank passes. To be measured
  against the 3 s target in the evaluation run.
- (−) Sticky scope can carry a model across a topic change. An explicit mention
  always overrides it, and `sticky=False` disables it.
- (−) "450 Plus" maps to the 450X document set until the spec sheet is split
  per model (P2).

## Interview framing
"I treated model ambiguity as a retrieval-scoping problem rather than a prompting
problem. The prompt can't be trusted to keep four near-duplicate manuals apart, so
the boundary is enforced before generation, deterministically, and tested against
a real vector store."
