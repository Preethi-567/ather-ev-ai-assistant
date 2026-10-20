# ADR-004: Spend fewer Gemini calls, and stop sleeping between them

- **Status:** Accepted
- **Date:** 2026-10-19
- **Builds on:** ADR-003 (ranking text), which made retrieval scores meaningful

## Context

End-to-end runs on the real stack took 9.5 s, 13.4 s and 29 s per question
against a 3 s target. The server log showed where the time went:

| Component | Time |
|---|---|
| Retrieval + rerank + SQLite | under 0.5 s |
| One Gemini call (the network round trip) | 1.4-1.9 s |
| **Sleeping in the client before each call** | **5.6-6.0 s, every call** |

The client enforced "6 s since the previous call finished" before every call,
so wall time was about 7.5 s x (number of Gemini calls). A follow-up question
spent four calls (classify, rewrite, sufficiency check, generation), roughly 30 s.

The API key is **free tier**, so the number of calls is also the scarce resource:
requests-per-minute and requests-per-day limits apply, and they differ by model
and account. The design therefore has to cut calls, not only cut waiting.

## Decision

Four independent changes, each pinned by tests:

1. **Sliding-window limiter** instead of a fixed gap. A short burst is allowed;
   the call after N in 60 s waits. N comes from `GEMINI_MAX_CALLS_PER_MIN`
   (default 10, the old gap's average rate). Every attempt, retries after a 429
   included, takes a slot. The first 429 logs the error body, which names the
   exhausted quota, so the real limit is observed rather than assumed.
2. **One merged intent + rewrite call** when neither a keyword nor a fast
   rewrite can answer (was two sequential calls). On a malformed reply the code
   falls back to the two original calls; on an outage it uses safe defaults and
   spends nothing further. The previous query is always offered and the model
   decides whether the question needs it (see "Bug found on the way").
3. **Fast reject on retrieval score.** Best reranker score below `NO_MATCH_FLOOR`
   (-3.0) returns the exact fallback with no Gemini call.
4. **Sufficiency check behind a switch**, `ATHER_SUFFICIENCY_CHECK` (default on,
   as specified). It is measured, not removed: see below.

## Evidence for the score floor

Best-chunk reranker score per question, real index, after ADR-003:

| Question | Expected | Score |
|---|---|---|
| regenerative braking (Rizta Z) | answerable | 8.85 |
| charge (Rizta Z) | answerable | 7.95 |
| charger LED colours | answerable | 6.69 |
| Joey Pocket | answerable | 6.16 |
| parking | answerable | 5.96 |
| tyre pressure (450S) | answerable | 5.05 |
| hibernation | answerable | 3.60 |
| charge **time** (Rizta Z) | absent, on-topic | 4.85 |
| replace the motor | absent, on-topic | 2.09 |
| sunroof (450S) | absent, off-topic | -5.68 |
| dashcam (Rizta Z) | absent, off-topic | -6.16 |
| price (450S) | absent, off-topic | -6.39 |

Two groups. Off-topic questions score near -6, every answerable one scores 3.6
or more, and the gap between them is empty. A floor at -3.0 sits in the gap and
rejects all three off-topic questions without touching any answerable one.

It is set low deliberately. A wrongly rejected real question is worse than one
extra Gemini call, so the floor only removes the unambiguous cases.

**What the score cannot do:** the two on-topic-but-absent questions (charge time
4.85, motor 2.09) score like answerable ones. Retrieval knows the page is about
charging; it cannot know the page lacks a duration. The grounded generator (Rule
2, exact fallback) is what refuses those, and it did (charge time returned the
fallback). The score is therefore a cheap first filter, never a substitute for
grounded generation.

## Calls per question

| Question | Before | After (defaults) | After (sufficiency off) |
|---|---|---|---|
| "who won the IPL?" | 0 | 0 | 0 |
| "how do I charge my Rizta Z?" | 2 | 2 | 1 |
| "how long does it take?" (follow-up) | 4 | 3 | 2 |
| "does the 450S have a sunroof?" | 4 | **1** (merged call, then fast reject) | 1 |

Time per call is about 1.5-2 s, so the follow-up goes from ~30 s to an expected
~5-6 s. Those are estimates derived from the logged call times; the end-to-end
run reports the actual numbers (`gemini_calls` is now in the API response).

## The sufficiency check

In the logged runs it answered YES every time, including for the question the
generator then correctly refused. It sees only chunk summaries, so it is a weak
judge, and it costs a call and about 7 s. It stays on because it is part of the
specified three-step chain and four observations are not enough to remove a
safety gate. Setting `ATHER_SUFFICIENCY_CHECK=0` lets the P5 evaluation (30
questions, RAGAS faithfulness) compare both settings and decide on data.

## Bug found on the way

`_is_followup_query` decided whether the rewriter saw the previous turn, using
keyword rules. "how long does it take?" (five words, no leading pronoun) was
classed as standalone, so the rewriter had no context and guessed "charge" from
its own prompt examples. The merged prompt no longer uses that rule: it always
provides the previous query and tells the model to ignore it when the new
question is self-contained. The old single-purpose rewriter keeps the heuristic
for its fallback path; replacing it is a P3 clean-up.

## Consequences

- Free-tier headroom: a typical question costs 1-3 calls instead of 2-4.
- A burst above the window waits (about 3 questions inside one minute at the
  default of 10). That is the limiter doing its job, not a regression.
- `NO_MATCH_FLOOR` is on the raw logit scale like `CONFIDENCE_FLOOR`. P3 moves
  both to a 0-1 scale together; they are listed side by side so that cannot be
  forgotten.
- RAGAS evaluation is LLM-heavy. On a free tier it will have to run in batches
  (P5 concern, recorded here so it is not a surprise).

## Alternatives considered

- **Raise the limit and keep per-call gaps.** Rejected: the cost was the gap
  itself, and the real limit is unknown.
- **Run the intent classifier and rewriter locally.** Still the plan for later
  (it removes the last pre-retrieval call), but it needs a model download and
  its own evaluation; the merged call is the cheap first step.
- **Drop the sufficiency check now.** Rejected: measure first (see above).

## Interview framing

"The pipeline took 30 seconds and I assumed Gemini was slow. The log showed each
call took under 2 seconds and the client was sleeping 6 seconds before every
one. I replaced the fixed gap with a rate window, then attacked the real
constraint on a free tier, which is the number of calls: merged two calls into
one, and rejected obvious non-matches from a calibrated retrieval score. I set
the threshold from a 12-question table where the two groups did not overlap,
and I kept the one check I could not yet justify removing behind a switch to be
measured on the evaluation set."
