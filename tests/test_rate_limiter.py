# tests/test_rate_limiter.py
"""The sliding window, with a fake clock so nothing really sleeps."""
import pytest

from src.generation.rate_limiter import (
    DEFAULT_MAX_CALLS_PER_MIN, ENV_VAR, SlidingWindowLimiter, max_calls_from_env,
)


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self):
        return self.t

    def sleep(self, d):
        self.sleeps.append(d)
        self.t += d


def limiter(max_calls, clock):
    return SlidingWindowLimiter(max_calls, 60.0, clock=clock.now, sleep=clock.sleep)


def test_a_burst_up_to_the_limit_never_waits():
    c = FakeClock()
    lim = limiter(4, c)
    # One question's worth of calls (2-4) must not be charged a gap per call.
    assert [lim.acquire() for _ in range(4)] == [0.0] * 4
    assert c.sleeps == []


def test_the_call_over_the_limit_waits_for_the_oldest_to_age_out():
    c = FakeClock()
    lim = limiter(3, c)
    for _ in range(3):
        lim.acquire()                      # three calls at t=1000
    waited = lim.acquire()                 # the 4th must wait the full window
    assert waited == pytest.approx(60.0) and c.sleeps == [pytest.approx(60.0)]


def test_the_window_slides_instead_of_resetting():
    c = FakeClock()
    lim = limiter(2, c)
    lim.acquire()                          # t=1000
    c.t += 30
    lim.acquire()                          # t=1030
    waited = lim.acquire()                 # window holds both; oldest ages out at t=1060
    assert waited == pytest.approx(30.0)


def test_old_calls_stop_counting():
    c = FakeClock()
    lim = limiter(2, c)
    lim.acquire(); lim.acquire()
    c.t += 61                              # both aged out
    assert lim.acquire() == 0.0 and c.sleeps == []


def test_total_calls_counts_every_attempt():
    c = FakeClock()
    lim = limiter(10, c)
    for _ in range(5):
        lim.acquire()
    assert lim.total_calls == 5


def test_rejects_a_non_positive_limit():
    with pytest.raises(ValueError):
        SlidingWindowLimiter(0)


@pytest.mark.parametrize("value, expected", [
    (None, DEFAULT_MAX_CALLS_PER_MIN),
    ("", DEFAULT_MAX_CALLS_PER_MIN),
    ("15", 15),
    ("  5 ", 5),
    ("0", DEFAULT_MAX_CALLS_PER_MIN),        # nonsense → safe default, not a crash
    ("-3", DEFAULT_MAX_CALLS_PER_MIN),
    ("ten", DEFAULT_MAX_CALLS_PER_MIN),
])
def test_env_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, value)
    assert max_calls_from_env() == expected
