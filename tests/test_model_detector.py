# tests/test_model_detector.py
import threading

import pytest

from src.query.model_detector import (
    M_450S, M_450X, M_RIZTA_S, M_RIZTA_Z,
    SessionModelStore, detect_models, resolve_scope,
)


@pytest.mark.parametrize("text, expected", [
    ("what is the tyre pressure on my 450S?",        [[M_450S]]),
    ("450X range",                                   [[M_450X]]),
    ("450 X range",                                  [[M_450X]]),
    ("Ather450S manual",                             [[M_450S]]),     # no space
    ("ATHER 450x",                                   [[M_450X]]),     # case
    ("how do I charge the Rizta Z",                  [[M_RIZTA_Z]]),
    ("rizta-s boot space",                           [[M_RIZTA_S]]),
    ("boot space on Rizta",                          [[M_RIZTA_S, M_RIZTA_Z]]),
    ("Rizta Zip mode",                               [[M_RIZTA_S, M_RIZTA_Z]]),  # Zip ≠ Z
    ("Ather 450",                                    [[M_450X, M_450S]]),
    ("450 Plus top speed",                           [[M_450X]]),
    ("450+ range",                                   [[M_450X]]),
    ("450X vs Rizta Z boot space",                   [[M_450X], [M_RIZTA_Z]]),
    ("Rizta Z and Rizta S",                          [[M_RIZTA_Z], [M_RIZTA_S]]),
    ("Rizta Z vs Rizta",                             [[M_RIZTA_Z], [M_RIZTA_S, M_RIZTA_Z]]),
    ("what about the Z model?",                      []),
    ("how do I reset the dashboard",                 []),
    ("",                                             []),
])
def test_detect_models(text, expected):
    assert detect_models(text) == expected


def test_450_seconds_is_not_450s():
    assert detect_models("it took 450 seconds") != [[M_450S]]


def test_numbers_containing_450_do_not_match():
    assert detect_models("part number 14500") == []
    assert detect_models("1450X") == []


# ── resolve_scope ─────────────────────────────────────────────────────────────

def test_explicit_in_original():
    store = SessionModelStore()
    scope = resolve_scope("tyre pressure 450S", "What is the tyre pressure?", "s1", store)
    assert scope.source == "explicit" and scope.models == [M_450S]


def test_explicit_found_only_in_rewrite_counts_as_explicit():
    store = SessionModelStore()
    scope = resolve_scope("what about the Z model?",
                          "What is the range of the Ather Rizta Z?", "s1", store)
    assert scope.source == "explicit" and scope.models == [M_RIZTA_Z]


def test_session_is_sticky_until_a_new_model_is_named():
    store = SessionModelStore()
    resolve_scope("charging on rizta z", "How do I charge the Ather Rizta Z?", "s1", store)

    follow_up = resolve_scope("how long does it take?", "How long does charging take?", "s1", store)
    assert follow_up.source == "session" and follow_up.models == [M_RIZTA_Z]

    switched = resolve_scope("and on the 450X?", "And on the Ather 450X?", "s1", store)
    assert switched.source == "explicit" and switched.models == [M_450X]

    after = resolve_scope("how long?", "How long does it take?", "s1", store)
    assert after.models == [M_450X]


def test_sessions_do_not_leak_into_each_other():
    store = SessionModelStore()
    resolve_scope("450S tyre", "tyre", "a", store)
    other = resolve_scope("boot space?", "boot space?", "b", store)
    assert other.source == "none" and not other.is_scoped


def test_sticky_can_be_disabled():
    store = SessionModelStore()
    resolve_scope("450S tyre", "tyre", "s1", store, sticky=False)
    assert not resolve_scope("again?", "again?", "s1", store, sticky=False).is_scoped


def test_store_is_safe_under_concurrent_writes():
    store = SessionModelStore()
    errors: list[Exception] = []

    def hammer(i: int):
        try:
            for j in range(200):
                store.set("shared", ((M_450X,),) if (i + j) % 2 else ((M_450S,),))
                assert store.get("shared") in (((M_450X,),), ((M_450S,),))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(16)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert not errors
