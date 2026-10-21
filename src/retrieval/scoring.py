# src/retrieval/scoring.py
"""
The one place that turns a reranker score into a decision.

WHY THIS EXISTS
---------------
Before P3 there were three "confidence floors" in three files: 0.0 on the raw
logit scale in hybrid_search (the only one that did anything), 0.4 in
prompt_chain (unused) and 0.4 in config (unused). The 0.4 looked like a
probability and the 0.0 was a logit, so "tune the threshold" meant editing the
wrong constant. Now:

  * thresholds are probabilities and live in src/config.py only;
  * this module converts once and answers "low confidence?" / "no match?";
  * chunks keep their raw logit as `reranker_score` (group merging compares
    logits, and logits keep their resolution where a sigmoid saturates at 1.0).

The comparison is done in the logit domain against a threshold converted ONCE,
so `score < threshold` is exact at the boundary (no sigmoid round trip).
"""

import math

from src.config import (
    CONFIDENCE_FLOOR,
    GROUP_MIN_CONFIDENCE,
    NO_MATCH_CONFIDENCE,
)


def sigmoid(logit: float) -> float:
    """Logit -> probability. Stable for large |logit| (no OverflowError)."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


def logit(probability: float) -> float:
    """Probability -> logit. Only defined on the open interval (0, 1)."""
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    return math.log(probability / (1.0 - probability))


# Thresholds converted once, at import.
LOW_CONFIDENCE_LOGIT = logit(CONFIDENCE_FLOOR)
NO_MATCH_LOGIT       = logit(NO_MATCH_CONFIDENCE)
GROUP_MIN_LOGIT      = logit(GROUP_MIN_CONFIDENCE)


def confidence(top_logit: float) -> float:
    """What the API reports: the top reranker score as a 0-1 probability."""
    return round(sigmoid(top_logit), 4)


def is_low_confidence(top_logit: float) -> bool:
    """True -> prefix the answer with the uncertainty warning."""
    return top_logit < LOW_CONFIDENCE_LOGIT


def is_no_match(top_logit: float) -> bool:
    """True -> nothing relevant: return the exact fallback, spend no Gemini call."""
    return top_logit < NO_MATCH_LOGIT
