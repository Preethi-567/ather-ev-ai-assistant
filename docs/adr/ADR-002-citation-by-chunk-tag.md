# ADR-002: Cite by chunk tag; render page numbers in code

**Status:** Proposed (P1) · **Date:** 2026-10-17

## Context
Rule 3 requires every answer to cite its source. v1.0 asked the LLM to write
"page X" and regex-validated the number afterwards. Two defects:

- Page numbers are not unique across five documents. "Page 20" is the tyre page in
  two manuals and neighbours it in two more, so "the page exists in the retrieved
  set" does not mean "the page belongs to the scooter being discussed".
- The validator ran after generation and could not read the format the prompt
  itself requested ("pages 12, 14").

## Decision
The model never sees page numbers. Each retrieved chunk is shown as `[C1]..[Cn]`
with its scooter model. The model cites by tag; code maps tag → (model, page,
file). Unknown tags are stripped. A second layer removes any free-text "page N"
(including lists) that was not retrieved.

## Consequences
- (+) A fabricated citation cannot be rendered: only retrieved chunks have tags.
- (+) Citations carry the scooter model, and the API returns structured `sources`.
- (−) The model can still attach a *real* tag to the wrong sentence. This ADR makes
  citations traceable, not semantically correct; faithfulness is measured by RAGAS.
- (−) Rule 3's literal "According to your … manual, page …" opener is rendered as a
  parenthetical per claim. One constant (`CITATION_TEMPLATE`) controls it.
- (−) Uncited answers are logged, not rejected, until the eval run shows how often
  they occur.

## Interview framing
"I moved citation from a generation problem to a rendering problem. The LLM
decides *which evidence it used*; code decides *how that evidence is displayed*."
