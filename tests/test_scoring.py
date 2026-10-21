# tests/test_scoring.py
"""One confidence scale: probabilities in config, logits in chunks (scoring.py)."""
import math

import pytest

from src import config
from src.retrieval import scoring
from src.retrieval.scoring import (
    LOW_CONFIDENCE_LOGIT, NO_MATCH_LOGIT, confidence, is_low_confidence, is_no_match, logit, sigmoid,
)


def test_sigmoid_and_logit_are_inverses():
    for p in (0.01, 0.05, 0.4, 0.5, 0.97):
        assert sigmoid(logit(p)) == pytest.approx(p)


def test_sigmoid_is_stable_at_extremes():
    assert sigmoid(1000) == 1.0 and sigmoid(-1000) == 0.0       # no OverflowError


def test_logit_rejects_non_probabilities():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            logit(bad)


def test_thresholds_are_probabilities_defined_only_in_config():
    for name in ("CONFIDENCE_FLOOR", "NO_MATCH_CONFIDENCE", "GROUP_MIN_CONFIDENCE"):
        assert 0.0 < getattr(config, name) < 1.0
    assert config.NO_MATCH_CONFIDENCE < config.CONFIDENCE_FLOOR     # reject below, warn below that


def test_boundaries_are_strict_and_exact():
    assert is_low_confidence(LOW_CONFIDENCE_LOGIT) is False
    assert is_low_confidence(LOW_CONFIDENCE_LOGIT - 1e-9) is True
    assert is_no_match(NO_MATCH_LOGIT) is False
    assert is_no_match(NO_MATCH_LOGIT - 1e-9) is True


def test_calibration_table_from_adr_004_still_holds():
    """The measured scores must land on the right side of both floors."""
    answerable = [8.85, 7.95, 6.69, 6.16, 5.96, 5.05, 3.60]
    off_topic  = [-5.68, -6.16, -6.39]
    assert not any(is_no_match(s) or is_low_confidence(s) for s in answerable)
    assert all(is_no_match(s) for s in off_topic)


def test_confidence_is_reported_as_zero_to_one():
    assert confidence(0.0) == 0.5 and 0.0 <= confidence(-6.0) < 0.01 and confidence(8.0) > 0.99


def test_no_duplicate_floor_constants_remain():
    import src.generation.prompt_chain as pc
    import src.retrieval.hybrid_search as hs
    for mod in (pc, hs):
        for name in ("CONFIDENCE_FLOOR", "NO_MATCH_FLOOR", "GROUP_MIN_RERANK"):
            assert not hasattr(mod, name), f"{mod.__name__}.{name} must live in config only"
