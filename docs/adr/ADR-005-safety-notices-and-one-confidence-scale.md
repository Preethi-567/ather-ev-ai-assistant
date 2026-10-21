# ADR-005: Safety notices come from code; one confidence scale

- **Status:** Accepted
- **Date:** 2026-10-21
- **Builds on:** ADR-004 (which listed the scale change as owed to P3)

## Context

Two defects of the same kind: a rule that must always hold was left to chance.

1. **Rules 4 and 6** ("always append the technician warning", "always end with the
   escalation line") were instructions to the model inside the prompt. Whether they
   appear is then a coin the model flips, the wording drifts, and a test can only
   check one sample. The repo prompt also did not match the project rules: it said
   "For electrical or battery-related issues, always consult an Ather-certified
   technician." and "...book a service appointment via the Ather app."
2. **Three confidence floors in three files:** `hybrid_search` 0.0 (a raw logit,
   the only one with any effect), `prompt_chain` 0.4 and `config` 0.4 (both unused).
   Tuning "the threshold" meant guessing which constant was real.

## Decision

**Notices (`src/generation/safety.py`).** Whether a notice is needed is a property of
the *question*, so code decides it from the intent and the original and rewritten
query, and appends the exact project-rule sentences. Technician notice first,
escalation notice last. The model is told (prompt v1.2) not to write them; any copy
it writes anyway is stripped before ours is added, so there is exactly one.
The strip is narrow on purpose: a manual sentence such as "Battery replacement
must be done by an Ather-certified technician [C2]" is an answer and stays.
Not applied to the exact fallback, the fast reject or the out-of-scope reply: those
are fixed strings, and the fallback already sends the user to Ather Support.
Where Rule 6 and the exact fallback collide (a fault question the manual cannot
answer), the fallback wins.

**One scale.** Thresholds are probabilities and live only in `src/config.py`:
`CONFIDENCE_FLOOR = 0.4`, `NO_MATCH_CONFIDENCE = 0.05` (logit -2.94, in the empty gap
of the ADR-004 table), `GROUP_MIN_CONFIDENCE = 0.5`. `src/retrieval/scoring.py`
converts once and is the single decision point (`is_low_confidence`, `is_no_match`).
Chunks keep the raw logit for ranking and group merging, because a sigmoid saturates
at 1.0 and hides the difference between a good and a great match. Comparisons are done
in the logit domain against a pre-converted threshold, so the boundary is exact.
Behaviour change: the low-confidence line moves from logit 0.0 (0.50) to 0.4.
Every answerable question in the table scored 3.6+ (0.97+), so none changes.

**Also:** `/query` returns `confidence` and `notices`; one JSON `REQUEST` line per
request goes to `logs/ather_assistant.log` (set up in the lifespan, not at import);
`ConversationMemory` is guarded by an `RLock`, with a test that forces the
check-then-set interleaving.

## Consequences

- Safety wording is testable and identical on every answer.
- Over-triggering is possible (a keyword match adds a notice to a harmless question);
  the symptom list omits everyday words ("sound", "hot", "slow") for that reason, and
  P5's safety questions will measure both misses and false alarms.
- Memory stores the answer body without notices.

## Interview framing

"A rule that must always hold should not depend on a model remembering it. I moved
the two safety sentences into code, decided from the question, and left the model
only the part that needs language. I also found three confidence thresholds on two
different scales and made it one: probabilities in config, one conversion module,
raw logits kept where ranking needs resolution."
