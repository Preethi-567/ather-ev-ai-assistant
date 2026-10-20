# tests/test_gemini_client_quota.py
"""
Quota accounting inside gemini_client: every ATTEMPT takes a slot in the rate
window (a 429'd request still consumes quota), and gemini_calls_made() reports
the same count the API logs per request.
"""
import pytest

from src.generation import gemini_client as gc
from src.generation.rate_limiter import SlidingWindowLimiter


class Resp:
    text = "ok"


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kw):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, models):
        self.models = models


@pytest.fixture()
def install(monkeypatch):
    def _install(outcomes):
        models = FakeModels(outcomes)
        limiter = SlidingWindowLimiter(1000)             # generous: never blocks the test
        monkeypatch.setattr(gc, "get_client", lambda: FakeClient(models))
        monkeypatch.setattr(gc, "_limiter", limiter)
        monkeypatch.setattr(gc, "_active_model", None)
        monkeypatch.setattr(gc.time, "sleep", lambda s: None)   # skip retry back-off
        return models, limiter
    return _install


def test_one_successful_call_takes_one_slot(install):
    models, limiter = install([Resp()])
    before = gc.gemini_calls_made()
    assert gc.call_gemini("hi") == "ok"
    assert limiter.total_calls == 1 and gc.gemini_calls_made() - before == 1


def test_a_retry_after_429_takes_a_second_slot(install):
    models, limiter = install([Exception("429 RESOURCE_EXHAUSTED retry in 2s"), Resp()])
    assert gc.call_gemini("hi") == "ok"
    assert models.calls == 2 and limiter.total_calls == 2


def test_slots_taken_always_equal_requests_sent(install):
    """Whatever the retry/fallback pattern, the count must match what reached Google."""
    models, limiter = install([Exception("503 UNAVAILABLE")] * 20)
    assert gc.call_gemini("hi") == ""                    # everything failed → empty string, no crash
    assert models.calls >= 1 and limiter.total_calls == models.calls
