# src/generation/rate_limiter.py
"""
Sliding-window rate limiter for Gemini calls.

WHY THIS REPLACES THE FIXED GAP
-------------------------------
The old client slept until 6.0 s had passed since the PREVIOUS call finished,
before EVERY call. A Gemini call itself takes 1.4-1.9 s, so each call cost about
7.5 s of wall time and a query with four calls took ~30 s, almost all of it
sleeping. The limit it was protecting is "N requests per minute", which is a
property of a WINDOW, not of the gap between two calls.

A sliding window allows a short burst (the 2-4 calls of one question) and then
makes the NEXT call wait only if the last 60 seconds already hold N calls. Same
average throughput as the old 6 s gap when N = 10, but a single question is no
longer charged 6 s per call.

THE LIMIT IS CONFIGURABLE, NOT ASSUMED
--------------------------------------
Free-tier limits depend on the model and the account and change over time, so
nothing here claims to know the real number. Set GEMINI_MAX_CALLS_PER_MIN (in
.env or the environment); the default of 10 equals the old behaviour. If the
real limit is lower, the 429 retry in gemini_client.py is still the safety net.

Every attempt is counted, including retries after a 429, because a rejected
request still consumes quota.

Pure Python. The clock and the sleep function are injectable so the window logic
is tested without real waiting.
"""

import os
import threading
import time
from collections import deque
from typing import Callable

from loguru import logger

DEFAULT_MAX_CALLS_PER_MIN = 10
ENV_VAR = "GEMINI_MAX_CALLS_PER_MIN"


class SlidingWindowLimiter:
    """At most `max_calls` acquisitions in any `window_seconds` span."""

    def __init__(
        self,
        max_calls: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self._max = max_calls
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()
        self.total_calls = 0          # attempts since start; used for per-request accounting

    def acquire(self) -> float:
        """
        Block until a call is allowed, record it, and return the seconds spent
        waiting (0.0 when no wait was needed).

        The sleep happens OUTSIDE the lock so one waiting thread never blocks
        another thread's bookkeeping. Two threads that wake together re-check
        the window, so the limit holds even then.
        """
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= self._window:
                    self._stamps.popleft()
                if len(self._stamps) < self._max:
                    self._stamps.append(now)
                    self.total_calls += 1
                    return waited
                delay = self._window - (now - self._stamps[0])
            self._sleep(delay)
            waited += delay


def max_calls_from_env() -> int:
    """GEMINI_MAX_CALLS_PER_MIN, falling back to the default on a bad value."""
    raw = os.getenv(ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_MAX_CALLS_PER_MIN
    try:
        value = int(raw)
        if value < 1:
            raise ValueError
        return value
    except ValueError:
        logger.warning(
            f"{ENV_VAR}={raw!r} is not a positive integer — "
            f"using {DEFAULT_MAX_CALLS_PER_MIN}"
        )
        return DEFAULT_MAX_CALLS_PER_MIN
