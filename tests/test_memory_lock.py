# tests/test_memory_lock.py
"""
ConversationMemory under concurrency.

RACE under test (check-then-set in _get_session): two threads ask for a session
that does not exist yet. Both see it missing, both create a deque, the second
assignment replaces the first, and the first thread's turn is appended to a deque
nobody holds. A lost turn, no exception.

Real thread timing would make this flaky, so the interleaving is FORCED: the dict
holds each thread at the "is it there?" step until both have looked (Barrier with
a timeout, so a correct lock - which keeps the second thread out - does not
deadlock; the barrier simply times out and the threads run one after the other).
"""
import threading

from src.query.memory import ConversationMemory


class SlowDict(dict):
    def __init__(self):
        super().__init__()
        self.barrier = threading.Barrier(2)

    def __contains__(self, key):
        found = super().__contains__(key)
        try:
            self.barrier.wait(timeout=0.3)     # both threads have now looked
        except threading.BrokenBarrierError:
            pass
        return found


def test_two_first_turns_on_a_new_session_are_both_kept():
    mem = ConversationMemory()
    mem._sessions = SlowDict()

    threads = [threading.Thread(target=mem.add_turn, args=("s", f"q{i}", f"a{i}")) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert sorted(t.query for t in mem.get_turns("s")) == ["q0", "q1"]


def test_reads_while_writing_never_raise():
    mem = ConversationMemory()
    errors, stop = [], threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            mem.add_turn("s", f"q{i}", "a"); i += 1

    def reader():
        for _ in range(3000):
            try:
                mem.get_context("s")
            except Exception as e:           # e.g. "deque mutated during iteration"
                errors.append(e)

    w = threading.Thread(target=writer); w.start()
    r = threading.Thread(target=reader); r.start(); r.join()
    stop.set(); w.join()
    assert errors == []


def test_reset_and_session_count_still_work():
    mem = ConversationMemory()
    mem.add_turn("a", "q", "r")
    assert mem.session_count() == 1
    mem.reset("a"); mem.reset("a")            # second reset is a no-op, not a KeyError
    assert mem.session_count() == 0
