"""serve/test_yield_wait.py - #656: a waiter held past YIELD_WAIT_S also earns a yield.

Measured 2026-10-18: two prompts of similar length never yielded to each other (only a prompt under half
qualified), so the second request waited for the whole first read - 52-70 s of control_occupied_wait.

    python3 -m unittest serve.test_yield_wait -v
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine  # noqa: E402


def engine_without_process(n=2):
    """A StrataEngine with the admission state of a started one and no process: alive() is `e.up`."""
    e = StrataEngine("missing-executable", [], lazy=True)
    e.batch = n
    e.slot_busy = [False] * n
    e.slot_claims, e.slot_releasing, e._orphan_seen = set(), set(), {}
    e.slot_cv = threading.Condition()
    e.waiting, e.admitting, e.wait_lens, e.ctl_epoch = 0, 0, [], 0
    e.ctl = threading.Lock()
    e.gen = 1
    setattr(e, "up", True)   # the lazy constructor leaves it unset
    e.alive = lambda: getattr(e, "up", False)
    return e


class YieldWaitTest(unittest.TestCase):
    def setUp(self):
        self.e = engine_without_process()

    def waiter(self, plen, age_s):
        now = time.monotonic()
        entry = [plen, object(), "foreground", now, now - age_s, True]
        with self.e.slot_cv:
            self.e.wait_lens.append(entry)
            self.e.waiting += 1
        return entry

    def test_half_length_waiter_still_qualifies_immediately(self):
        self.waiter(1000, 0.0)
        self.assertTrue(self.e._shorter_waiting(2000), "a half-length waiter keeps the #656 rule")
        self.assertFalse(self.e._shorter_waiting(1900), "just over half is not half")

    def test_equal_length_waiter_qualifies_only_after_the_wait(self):
        self.waiter(2000, self.e.YIELD_WAIT_S - 1.0)
        self.assertFalse(self.e._shorter_waiting(2000), "a fresh equal-length waiter must not preempt a read")
        self.waiter(2000, self.e.YIELD_WAIT_S + 1.0)
        self.assertTrue(self.e._shorter_waiting(2000), "a waiter held past YIELD_WAIT_S must preempt")

    def test_a_foreground_read_does_not_yield_to_aged_background_work(self):
        now = time.monotonic()
        entry = [2000, object(), "background", now, now - self.e.YIELD_WAIT_S - 1.0, True]
        with self.e.slot_cv:
            self.e.wait_lens.append(entry)
            self.e.waiting += 1
        self.assertFalse(self.e._shorter_waiting(2000, "foreground"),
                         "aged background work must not preempt a foreground read")
        self.assertTrue(self.e._shorter_waiting(2000, "background"),
                        "background work may yield to aged background work (FIFO fairness)")
        self.assertTrue(self.e._shorter_waiting(5000, "foreground"),
                        "a much shorter background waiter still preempts by the #656 rule")

    def test_a_foreground_read_yields_to_aged_foreground_work(self):
        now = time.monotonic()
        entry = [2000, object(), "foreground", now, now - self.e.YIELD_WAIT_S - 1.0, True]
        with self.e.slot_cv:
            self.e.wait_lens.append(entry)
            self.e.waiting += 1
        self.assertTrue(self.e._shorter_waiting(2000, "foreground"))

    def test_no_waiter_never_yields(self):
        self.assertFalse(self.e._shorter_waiting(4000))


if __name__ == "__main__":
    unittest.main()
