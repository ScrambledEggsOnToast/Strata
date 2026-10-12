"""#465: "parallel": N - several requests decode together in the engine's batch slots (--batch N).

A fake engine (a Python script speaking the engine's GEN / BGEN / BT / BDONE / BADM / BSTOP lines) runs behind the
real StrataEngine and Service, so the server's side is tested without a GPU: requests up to N run at once, more
wait for a slot, /metrics shows the slots, each request's history row is its own, and a conversation's next turn
goes back to the slot that holds it."""
import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, Service, SpeculationPolicy, StrataEngine, engine_args, parallel_args, serve)

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257).
FAKE_BATCH = r'''import os, queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args
groups = int(args[args.index("--says-groups") + 1]) if "--says-groups" in args else 0   # INFO batch_groups=G (--batch-groups auto)        # #997: the first window over two slots fails
STEP = 0.02
hold = args[args.index("--hold-replies") + 1] if "--hold-replies" in args else None
CH = 32
lines, stop = queue.Queue(), threading.Event()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 else "") +
      (" slot_cache=1" if "--slotcache" in args else "") + (f" batch_groups={groups}" if groups > 1 else ""), flush=True)
if "--says-budget" in args:
    print("INFO foreground_budget_ms=100 background_share=0.25", flush=True)
if "--says-spec" in args:
    print("INFO spec=4", flush=True)
if "--says-residency" in args:
    for info in args[args.index("--says-residency") + 1].splitlines():
        print("INFO " + info, flush=True)
ctx = int(args[args.index("--says-context") + 1]) if "--says-context" in args else 4096
print(f"READY {ctx} stop", flush=True)
LONG = list(b"LONGREPLY")
def continued(ids):        # (the whole reply, how much of it the prompt already ends with: a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids)))
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    return R, max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
def reply(ids):            # the rest of the reply after what the prompt already ends with
    R, k = continued(ids)
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
stopped = set()      # BSTOPped slots: they end "cancel"
def window():        # one batch window: every active slot one token
    if hold and os.path.exists(hold):
        return
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        left, max_new, produced = active[b]
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if log and "--says-budget" in args:
            log.write(f"COMMIT {b} {produced}\n"); log.flush()
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            print(f"BDONE {b} {produced} {fin} 1.0", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
REUSE = "--reuse" in args     # DONE's reused field as the engine gives it: a continued prompt is all held but its last
def reused(ids):              # token (the request's own output included), a fresh one the first 3 tokens
    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3
parked = {}
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        parked.pop(b, None)
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = f[-1].split(",")
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # #1059: refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        if bytes(int(x) & 255 for x in ids).find(b"ADMISSIONREFUSE") >= 0:
            print("REJECT bounded=1 prompt + max_new exceeds the context", flush=True)
            if slot is not None:
                boundary = slot + 1 if "--bad-refusal-boundary" in args else slot
                print(f"BADM {boundary} 0", flush=True)
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        toks = reply(ids)
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        saved = parked.pop(slot, None) if slot is not None else None
        start = saved[1] if saved and saved[0] == ids else 0
        read_reused = max(reused(ids), start)   # Resumed legs include this request's own parked work.
        for q in range(start + CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            if "--quiet-prefill" not in args:
                print(f"PP {q} {len(ids)} 1 1", flush=True)
            if log and "--says-budget" in args:
                log.write(f"PREFILL {slot} {q}\n"); log.flush()
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            parked[given[0]] = (ids, given[1])
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {read_reused}", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if slot is not None else max_new
        out = 0
        for t in toks[:n]:
            while slot is None and hold and os.path.exists(hold) and not stop.is_set():
                time.sleep(0.005)
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {read_reused}", flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            print(f"BADM {slot} {1 if cont else 0}", flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
'''


class ParallelArgs(unittest.TestCase):
    def test_parallel_config_key(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"parallel": 1}, []), [])
        self.assertEqual(parallel_args({}, []), [])
        self.assertEqual(parallel_args({"parallel": 2}, ["--batch", "3"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({"parallel": "4"}, []), [])                   # said and ignored
        self.assertEqual(parallel_args({"parallel": True}, []), [])
        self.assertEqual(parallel_args({"parallel": 12}, []), ["--batch", "12"])     # the engine warns and caps
    def test_parallel_above_the_window_says_so_unless_batch_mtp(self):
        import contextlib, io, os
        def said(args, env):
            buf = io.StringIO()
            old = os.environ.get("STRATA_BATCH_MTP")
            if env is None:
                os.environ.pop("STRATA_BATCH_MTP", None)
            else:
                os.environ["STRATA_BATCH_MTP"] = env
            try:
                with contextlib.redirect_stdout(buf):
                    r = parallel_args({"parallel": 12}, args)
            finally:
                if old is None:
                    os.environ.pop("STRATA_BATCH_MTP", None)
                else:
                    os.environ["STRATA_BATCH_MTP"] = old
            self.assertEqual(r, ["--batch", "12"])
            return "at most" in buf.getvalue()
        self.assertTrue(said([], None))                  # 0.1.39: the window holds 8 rows
        self.assertTrue(said([], "0"))
        self.assertFalse(said(["--batch-mtp"], None))    # --batch-mtp waves more through the window
        self.assertFalse(said([], "1"))

        args = engine_args({"args": ["--pack", "p"], "parallel": 2})
        self.assertEqual(args[-2:], ["--batch", "2"])

    def test_resident_multiplex_is_explicit_and_demands_slots(self):
        self.assertEqual(engine_args({"args": []}), [])
        self.assertEqual(engine_args({"args": [], "parallel": 2, "resident_multiplex": True}),
                         ["--batch", "2", "--resident-multiplex"])
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            engine_args({"args": [], "resident_multiplex": True})
        with self.assertRaises(ValueError):
            engine_args({"args": [], "resident_multiplex": "true"})


class AdaptiveSpeculation(unittest.TestCase):
    @staticmethod
    def waiter(context=100, due=2.0, priority="foreground"):
        return [context, object(), priority, due]

    def test_cold_policy_never_withholds_waiting_capacity(self):
        policy = SpeculationPolicy(2.0, 4)
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter()], now=0))
        self.assertTrue(policy.prefer_solo(100, 64, now=0))
        self.assertFalse(policy.yield_solo(100, 64, [], now=0))

    def test_wire_counts_commits_not_proposals_or_terminal_totals(self):
        policy = SpeculationPolicy(2.0, 4)
        policy.begin("solo", 100, transition=True, now=0)
        policy.wire("REUSED 99", now=1)
        policy.wire("DRAFT 999 rejected 999", now=1.05)
        for i in range(8):
            policy.wire(f"T {10 + i}", now=1.1 + i / 10)
        key = policy._key("solo", 1, 100)
        before = policy.costs[key][0] / policy.costs[key][1]
        policy.wire("DONE 8 100 1 1 length 999 999 99", now=2)
        after = policy.costs[key][0] / policy.costs[key][1]
        self.assertEqual(policy.costs[key][2], 8)
        self.assertLess(after, before, "zero-output tail drafting/drain still costs wall time")
        self.assertAlmostEqual(policy.handoffs[key], 1.1)
        policy.begin("batch", 100, slot=0, now=3)
        policy.wire("T 20", now=3.2)  # BGEN's first committed target token
        policy.wire("DONE 1 100 1 1 length 0 0 99", now=3.3)
        policy.wire("BADM 0 1", now=3.4)
        policy.begin("batch", 200, slot=1, now=3.5)
        policy.wire("T 21", now=3.6)
        policy.wire("DONE 1 200 1 1 length 0 0 0", now=3.7)
        policy.wire("BADM 1 1", now=3.8)
        for i in range(8):
            policy.wire(f"BT {i % 2} {30 + i}", now=4 + i / 10)
        policy.wire("BDONE 0 5 cancel 1700", now=5)
        policy.wire("BDONE 1 5 length 1900", now=5.2)
        self.assertEqual(sum(value[2] for k, value in policy.costs.items() if k[0] == "batch"), 10)
        self.assertEqual(policy.slots, {})
        self.assertIsNone(policy.epoch)

    def test_context_load_and_fixed_slack_change_actual_choice(self):
        policy = SpeculationPolicy(2.0, 4)
        policy.observe("solo", 1, 100, 80, 1)
        policy.observe("batch", 2, 100, 20, 1)
        waiters = [self.waiter()]
        self.assertFalse(policy.yield_solo(100, 64, waiters, now=0))
        self.assertFalse(policy.yield_solo(100, 64, waiters, now=.49))
        self.assertTrue(policy.yield_solo(100, 64, waiters, now=.5),
                        "a profitable solo burst cannot keep extending its own deadline")
        policy.yield_solo(100, 64, [], now=.6)
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter(due=.7)], now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter(due=0, priority="background")], now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [], foreground_pending=True, now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter(priority="background")],
                                         foreground_pending=True, now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter() for _ in range(3)], now=.6),
                        "unmeasured higher load retains immediate ordinary-batch fallback")
        policy.observe("solo", 1, 4096, 10, 1)
        policy.observe("batch", 2, 4096, 50, 1)
        self.assertTrue(policy.yield_solo(4096, 64, [self.waiter(4096)], now=0))

    def test_gain_must_repay_catchup_and_clear_hysteresis(self):
        policy = SpeculationPolicy(2.0, 2)
        policy.observe("solo", 1, 100, 80, 1)
        policy.observe("batch", 1, 100, 20, 1)
        key = policy._key("solo", 1, 100)
        policy.probed[key] = 0
        policy.handoffs[key] = 3.0
        self.assertFalse(policy.prefer_solo(100, 64, now=1))
        self.assertTrue(policy.prefer_solo(100, 120, now=1))
        policy.observe("solo", 1, 200, 21, 1)
        policy.observe("batch", 1, 200, 20, 1)
        self.assertFalse(policy.prefer_solo(200, 1000, now=1), "a 5% noisy gain is not a switch")
        policy.observe("solo", 1, 300, 10, 1)
        policy.observe("batch", 1, 300, 40, 1)
        key = policy._key("solo", 1, 300)
        policy.probed[key] = 0
        self.assertFalse(policy.prefer_solo(300, 64, now=15.99))
        self.assertTrue(policy.prefer_solo(300, 64, now=16), "stale acceptance is remeasured, not frozen")
        policy.begin("solo", 300, now=16)
        self.assertFalse(policy.prefer_solo(300, 64, now=16.01))

    def test_new_process_has_no_predecessor_cost_or_deadline_state(self):
        before = SpeculationPolicy(2.0, 2)
        before.observe("solo", 1, 100, 100, 1)
        before.observe("batch", 2, 100, 20, 1)
        self.assertFalse(before.yield_solo(100, 64, [self.waiter()], now=0))
        after = SpeculationPolicy(2.0, 2)
        self.assertTrue(after.yield_solo(100, 64, [self.waiter()], now=.1))
        self.assertIsNone(after.defer_until)


class ControlBudget(unittest.TestCase):
    def test_arrival_trace_bounds_background_wait_without_reordering_foreground(self):
        import collections
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        engine.background_share = .25
        background = [3000, object(), "background", 1.0]
        foreground = [[20, object(), "foreground", float("inf")] for _ in range(100)]
        engine.control_waiters = collections.deque([background, *foreground])
        self.assertIs(engine._control_turn(0), foreground[0])
        engine.control_waiters.remove(foreground[0])
        self.assertIs(engine._control_turn(.99), foreground[1])
        self.assertIs(engine._control_turn(1.0), background,
                      "new foreground arrivals cannot overtake a due background continuation")
        with mock.patch("serve.server.time.monotonic", return_value=2.0):
            self.assertAlmostEqual(engine._background_next(1.8), 2.6)
            self.assertEqual(engine._background_next(0), 8.0,
                             "background requeue must repay measured service, not force a chunk every budget")
        self.assertTrue(engine._can_yield(100), "the legacy two-yield cap bypasses the configured policy")
        engine.foreground_budget_s = 0
        self.assertIs(engine._control_turn(1.0), background)
        self.assertFalse(engine._can_yield(2), "rollback preserves the legacy policy")

    def test_foreground_waiting_for_a_physical_slot_stays_in_the_control_queue(self):
        import collections
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        engine.background_share = .25
        engine.slot_cv = threading.Condition()
        engine.ctl = threading.Lock()
        engine.ctl_epoch = 0
        engine.waiting = 0
        engine.wait_lens = []
        engine.control_waiters = collections.deque()
        engine.alive = lambda: True
        ready, acquired, cancel = threading.Event(), threading.Event(), threading.Event()
        results = []
        def request():
            def wait():
                results.append((yield from engine._take_control(cancel, 20, lambda _: acquired.set(),
                                                               slot_ready=ready.is_set)))
            list(wait())
        thread = threading.Thread(target=request)
        thread.start()
        with engine.slot_cv:
            self.assertTrue(engine.slot_cv.wait_for(lambda: engine.waiting == 1, timeout=2))
            foreground = engine.control_waiters[0]
            background = [3000, object(), "background", time.monotonic() + 10]
            engine.control_waiters.append(background)
            self.assertIs(engine._control_turn(time.monotonic()), foreground,
                          "a foreground slot waiter must not disappear and make background appear alone")
            self.assertFalse(engine.ctl.locked())
            background[3] = 0
            self.assertIs(engine._control_turn(time.monotonic()), background,
                          "keeping a slot waiter visible must not starve due background work")
            engine.control_waiters.remove(background)
            ready.set()
            engine.slot_cv.notify_all()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [True])
        self.assertTrue(acquired.is_set())
        self.assertEqual(engine.waiting, 0)
        self.assertEqual(list(engine.control_waiters), [])
        engine.ctl.release()

    def test_http_foreground_waiter_is_visible_but_cannot_starve_due_background(self):
        import collections
        service = object.__new__(Service)
        foreground = {"priority": "foreground"}
        service.admission_waiters = [foreground]
        service.admission_active = {}
        service._refresh_foreground_pending()
        self.assertTrue(service._foreground_pending)
        service.admission_waiters.remove(foreground)
        service.admission_active["foreground"] = foreground
        service._refresh_foreground_pending()
        self.assertTrue(service._foreground_pending, "admitted but unsubmitted requests are still waiting")
        foreground["_executing"] = True
        service._refresh_foreground_pending()
        self.assertFalse(service._foreground_pending, "an executing decoder is not an admission waiter")
        foreground["_executing"] = False
        service._refresh_foreground_pending()
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        engine.background_share = .25
        engine.slot_cv = threading.Condition()
        engine.ctl = threading.Lock()
        engine.ctl_epoch = 0
        engine.waiting = 0
        engine.wait_lens = []
        engine.control_waiters = collections.deque()
        engine.alive = lambda: True
        acquired, cancel = threading.Event(), threading.Event()
        results = []
        due = time.monotonic() + 1.0
        def request():
            def wait():
                results.append((yield from engine._take_control(
                    cancel, 3000, lambda _: acquired.set(), priority="background", background_due=due,
                    foreground_pending=lambda: service._foreground_pending)))
            list(wait())
        thread = threading.Thread(target=request)
        thread.start()
        with engine.slot_cv:
            self.assertTrue(engine.slot_cv.wait_for(lambda: engine.waiting == 1, timeout=.5))
            self.assertFalse(engine.ctl.locked(), "HTTP foreground demand forbids early background reacquisition")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set(), "a due background request advances even under persistent demand")
        self.assertEqual(results, [True])
        engine.ctl.release()



class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_scope = ["a"] * n
        e.slot_groups = 1
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        e.slot_group = [0] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6], 'a'), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9], 'a'), 0)
        self.assertEqual(e.pick_slot([7, 8], 'a'), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8], 'a'), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8], 'a'))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3], 'a'), 1)                 # a held prompt is never the WHOLE prompt

    def test_batch_groups_spread_requests_over_the_groups(self):
        # #1249: with --batch-groups 2 over 4 slots (groups {0,1} and {2,3}) a second request goes to the other group,
        # and a short held prefix does not pull it back into the busy one
        e = self.engine(4)
        e.slot_groups = 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], 'a'), 2)
        e.slot_held[1] = list(range(1, 601))                       # a long prefix is worth the unbalanced group
        self.assertEqual(e.pick_slot(list(range(1, 700)), 'a'), 1)
        e.slot_groups, e.slot_group = 1, [0] * 4                   # without groups: any held prefix, as before
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], 'a'), 1)

    def test_pipelined_prefix_affinity_is_scoped_and_precedes_hole_filling(self):
        e = self.engine(4)
        e.slot_groups, e.slot_gs = 2, 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[2] = list(range(600))
        e.slot_scope[2] = "a"
        prompt = list(range(601))
        self.assertEqual(e.pick_slot(prompt, "a"), 2)
        self.assertEqual(e.pick_slot(prompt, "other"), 1)
        self.assertEqual(e.pick_slot(prompt, ""), 1)

    def test_slots_view(self):
        e = self.engine(2)
        e.slot_held[1] = [1, 2]
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual(v[0]["state"], "decoding")
        self.assertEqual(v[0]["generated"], 4)
        self.assertEqual(v[1], {"slot": 1, "state": "idle", "held_tokens": 2})

    def test_slots_view_keeps_reported_residency_while_decoding(self):
        e = self.engine(2)
        e.info.update(slot_0_kv_mode="streamed", slot_0_kv_window=65536,
                      slot_0_draft_kv_mode="resident", slot_0_draft_kv_window=262144,
                      slot_1_kv_mode="resident", slot_1_kv_window=262144,
                      slot_1_draft_kv_mode="off", slot_1_draft_kv_window=0)
        e.slot_live[0] = {"state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        view = e.slots_view()
        self.assertEqual((view[0]["kv_mode"], view[0]["kv_window"],
                          view[0]["draft_kv_mode"], view[0]["draft_kv_window"]),
                         ("streamed", 65536, "resident", 262144))
        self.assertEqual(view[1], {"slot": 1, "state": "idle", "held_tokens": 0,
                                  "kv_mode": "resident", "kv_window": 262144,
                                  "draft_kv_mode": "off", "draft_kv_window": 0})


class ParallelService(unittest.TestCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, reuse=False, says_groups=0, more=()):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--reuse"] if reuse else []
        extra += ["--says-groups", str(says_groups)] if says_groups else []
        extra += list(more)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def chat(self, text, max_tokens=64, priority="foreground"):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "strata_priority": priority}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def message(self, text, max_tokens=64):
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_native_info_reports_each_effective_slot_and_its_draft(self):
        """Scripted INFO -> HTTP reporting only, not proof of native K/V allocation or execution."""
        info = ("working_kv_mode=resident working_kv_window=262144 "
                "working_draft_kv_mode=resident working_draft_kv_window=262144\n"
                "slot_0_kv_mode=streamed slot_0_kv_window=65536 "
                "slot_0_draft_kv_mode=streamed slot_0_draft_kv_window=65536\n"
                "slot_1_kv_mode=streamed slot_1_kv_window=65536 "
                "slot_1_draft_kv_mode=resident slot_1_draft_kv_window=262144")
        self.start(4, fit=2, more=("--slot-kv-window", "65533", "--says-context", "262144",
                                   "--says-residency", info))
        expected = [{"slot": 0, "state": "idle", "held_tokens": 0,
                     "kv_mode": "streamed", "kv_window": 65536,
                     "draft_kv_mode": "streamed", "draft_kv_window": 65536},
                    {"slot": 1, "state": "idle", "held_tokens": 0,
                     "kv_mode": "streamed", "kv_window": 65536,
                     "draft_kv_mode": "resident", "draft_kv_window": 262144}]
        metrics = self.get("/metrics")
        self.assertEqual(metrics["live"]["slots"], expected)
        self.assertEqual(metrics["engine"]["working_kv_mode"], "resident")
        self.assertEqual(metrics["engine"]["working_draft_kv_mode"], "resident")
        self.assertEqual(metrics["engine"]["slot_0_kv_window"], 65536)
        self.assertEqual(self.get("/v1/status")["slots"], expected)
        self.assertEqual(self.get("/slots"),
                         [{"id": 0, "n_ctx": 262144, "is_processing": False, "n_prompt_tokens": 0,
                           "kv_mode": "streamed", "kv_window": 65536,
                           "draft_kv_mode": "streamed", "draft_kv_window": 65536},
                          {"id": 1, "n_ctx": 262144, "is_processing": False, "n_prompt_tokens": 0,
                           "kv_mode": "streamed", "kv_window": 65536,
                           "draft_kv_mode": "resident", "draft_kv_window": 262144}])

    def test_reported_resident_context_and_absent_draft_override_requested_window(self):
        info = ("slot_0_kv_mode=resident slot_0_kv_window=4096 "
                "slot_0_draft_kv_mode=off slot_0_draft_kv_window=0\n"
                "slot_1_kv_mode=resident slot_1_kv_window=4096 "
                "slot_1_draft_kv_mode=off slot_1_draft_kv_window=0")
        self.start(2, more=("--slot-kv-window", "65536", "--says-residency", info))
        for path in ("/slots", "/metrics", "/v1/status"):
            with self.subTest(path=path):
                reply = self.get(path)
                slots = reply if path == "/slots" else reply["live"]["slots"] if path == "/metrics" else reply["slots"]
                for slot in slots:
                    self.assertEqual((slot["kv_mode"], slot["kv_window"],
                                      slot["draft_kv_mode"], slot["draft_kv_window"]),
                                     ("resident", 4096, "off", 0))

    def test_legacy_info_leaves_status_unchanged(self):
        self.start(2)
        self.assertEqual(self.get("/slots"),
                         [{"id": 0, "n_ctx": 4096, "is_processing": False, "n_prompt_tokens": 0},
                          {"id": 1, "n_ctx": 4096, "is_processing": False, "n_prompt_tokens": 0}])
        self.assertEqual(self.get("/metrics")["live"]["slots"],
                         [{"slot": 0, "state": "idle", "held_tokens": 0},
                          {"slot": 1, "state": "idle", "held_tokens": 0}])
        self.assertNotIn("slots", self.get("/v1/status"))

    def test_multiplex_requires_native_acknowledgement(self):
        self.start(2)
        self.engine.close()
        self.engine.spawn[1].append("--resident-multiplex")
        import serve.server as server
        real = server.subprocess.Popen
        script = Path(self.tmp.name) / "fake_strata.py"
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            with self.assertRaisesRegex(RuntimeError, "INFO resident_multiplex=1"):
                self.engine.restart(tries=1)
        self.assertFalse(self.engine.alive())
        self.assertIsNone(self.engine.proc)


    def test_early_close_does_not_release_successor_reservation(self):
        self.start(2)
        engine = self.engine
        engine.slot_busy[1] = True  # peer reservation forces the real BGEN/BDONE path
        release = engine._release_slot_when_done
        successor = []
        def reserve_after_drain(slot, stream=None):
            release(slot, stream)
            with engine.slot_cv:
                chosen = engine.pick_slot([], "")
                self.assertEqual(chosen, slot)
                engine.slot_busy[chosen] = True
                successor.append(chosen)
        output = engine.generate_batched(list(b'LONGREPLY'), 64, {}, threading.Event())
        try:
            with mock.patch.object(engine, '_release_slot_when_done', side_effect=reserve_after_drain):
                self.assertIsInstance(next(output), int)
                self.assertIsInstance(next(output), int)  # first BT after BADM, not admission T
                output.close()
            self.assertEqual(len(successor), 1)
            self.assertTrue(engine.slot_busy[successor[0]], 'old consumer released the successor reservation')
        finally:
            for slot in successor:
                engine.slot_busy[slot] = False
            engine.slot_busy[1] = False
            output.close()

    def test_batch_groups_auto_follows_the_engine(self):
        """--batch-groups auto: the engine picks the groups and says so (INFO batch_groups=G); the server spreads the
        requests over those groups, and a plain number is read as before."""
        self.start(8, says_groups=4, more=["--batch-groups", "auto"])
        e = self.engine
        self.assertEqual((e.batch, e.slot_groups), (8, 4))
        self.assertEqual(e.slot_group, [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(e.slot_order[:4], [0, 2, 4, 6])
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, more=["--batch-groups", "auto"])          # an engine that reports none: one group
        self.assertEqual(self.engine.slot_groups, 1)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, more=["--batch-groups", "2"])
        self.assertEqual(self.engine.slot_groups, 2)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, says_groups=4)                           # 0.1.41: no flag on a layer split = the engine's choice
        self.assertEqual(self.engine.slot_groups, 4)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, says_groups=4, more=["--batch-groups", "1"])   # the opt-out
        self.assertEqual(self.engine.slot_groups, 1)

    def test_a_burst_of_connections_is_not_reset(self):
        """30-40 clients at once got "connection reset by peer" with the listen backlog of 5 (4 x R9700 burst): the
        server listens with a deep backlog (STRATA_HTTP_BACKLOG, 256), so 60 simultaneous requests all get answers.
        This fork admits a bounded number of requests at once, so an answer may be a 429 refusal: what the deep
        backlog has to deliver is that every connection is answered rather than reset or left hanging."""
        import serve.server as server
        self.assertGreaterEqual(server.Server.request_queue_size, 64)
        self.start(2)
        errs, ok, refused = [], [], []
        def one(i):
            try:
                self.chat(f"burst {i}", max_tokens=8)
                ok.append(i)
            except urllib.error.HTTPError as e:
                refused.append(e.code)
            except Exception as e:                      # noqa: BLE001
                errs.append(repr(e))
        th = [threading.Thread(target=one, args=(i,)) for i in range(60)]
        for t in th: t.start()
        for t in th: t.join()
        self.assertEqual(errs[:2], [])
        self.assertEqual(len(ok) + len(refused), 60)
        self.assertLessEqual(set(refused), {429}, refused)   # only the fork's bounded admission may refuse
        self.assertGreater(len(ok), 0, 'a burst is still served, not only refused')

    def test_stop_strings_in_a_batch_slot(self):
        """#454: a stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def queue_boundary(self, depth):
        """Effective HTTP reservations: two slots plus a configured waiting depth."""
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            gate.touch()
            self.start(4, fit=2, more=("--hold-replies", str(gate)))
            self.svc.configure_admission({"request_queue_limit": depth})
            replies, errors = [], []

            def go(index):
                try:
                    replies.append(self.chat(f"queued question {index}"))
                except Exception as error:  # reported in the test thread
                    errors.append(error)

            threads = [threading.Thread(target=go, args=(i,)) for i in range(2 + depth)]
            try:
                for thread in threads:
                    thread.start()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    snapshot = self.get("/status")["admission"]
                    if (snapshot["active"], snapshot["waiting"], snapshot.get("executing"),
                            snapshot.get("native_waiting")) == (2, depth, 1, 1):
                        break
                    time.sleep(0.005)
                self.assertEqual((snapshot["slots"], snapshot["request_queue_limit"],
                                  snapshot["active"], snapshot["waiting"]), (2, depth, 2, depth))
                self.assertEqual((snapshot["executing"], snapshot["native_waiting"]), (1, 1))
                if depth:
                    self.assertGreater(snapshot["oldest_wait_s"], 0)
                else:
                    self.assertEqual(snapshot["oldest_wait_s"], 0)
                for path in ("/metrics", "/health", "/v1/status"):
                    observed = self.get(path)["admission"]
                    self.assertEqual((observed["active"], observed["waiting"],
                                      observed["request_queue_limit"]), (2, depth, depth))
                    self.assertEqual((observed["executing"], observed["native_waiting"]), (1, 1))
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.chat("beyond the bounded capacity")
                with refused.exception as response:
                    self.assertEqual(response.code, 429)
                    self.assertEqual(response.headers.get("Retry-After"), "1")
                    self.assertIn("queue is full", json.load(response)["error"]["message"])
            finally:
                gate.unlink(missing_ok=True)
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(replies), 2 + depth)
            self.assertTrue(all(reply["choices"][0]["message"]["content"] == "ok, done." for reply in replies))
            snapshot = self.get("/status")["admission"]
            self.assertEqual((snapshot["active"], snapshot["waiting"], snapshot["oldest_wait_s"]), (0, 0, 0))

    def test_configured_depth_refuses_at_effective_slots_plus_depth(self):
        self.queue_boundary(3)

    def test_zero_depth_refuses_at_effective_slots(self):
        self.queue_boundary(0)

    def test_native_budget_refusal_drains_without_poisoning_a_survivor(self):
        self.start(2)
        replies, errors = [], []

        def survivor():
            try:
                replies.append(self.chat("survivor LONGREPLY"))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=survivor)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not self.log.exists():
                time.sleep(0.005)
            self.assertTrue(self.log.exists(), "the survivor submitted work before the refusal")
            with self.assertRaises(urllib.error.HTTPError) as refused:
                self.chat("ADMISSIONREFUSE")
            with refused.exception as response:
                self.assertEqual(response.code, 400)
                self.assertIn("exceeds the context", json.load(response)["error"]["message"])
        finally:
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(replies[0]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertTrue(self.engine.alive(), "an explicit bounded refusal must not kill coexisting work")
        self.assertEqual(self.chat("replacement")["choices"][0]["message"]["content"], "ok, done.")
        refused_record = next(row for row in self.svc.lifecycle_records() if row["state"] == "error")
        self.assertEqual(refused_record.get("output_tokens", 0), 0)
        self.assertFalse(any(self.engine.slot_busy))
        snapshot = self.get("/metrics")["admission"]
        self.assertEqual((snapshot["active"], snapshot["waiting"], snapshot["executing"],
                          snapshot["native_waiting"]), (0, 0, 0, 0))

    def test_mixed_context_refusal_never_submits_or_truncates_work(self):
        self.start(2, more=("--says-context", "256"))
        self.assertEqual(self.chat("short", max_tokens=4)["usage"]["completion_tokens"], 4)
        before = self.log.read_text()
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.chat("long context " * 30, max_tokens=4)
        with refused.exception as response:
            self.assertEqual(response.code, 400)
            self.assertIn("never truncated", json.load(response)["error"]["message"])
        self.assertEqual(self.log.read_text(), before, "an over-context request must not reach GEN/BGEN")
        self.assertEqual(self.chat("short successor", max_tokens=4)["usage"]["completion_tokens"], 4)
        records = self.svc.lifecycle_records()
        self.assertEqual([row.get("output_tokens", 0) for row in records], [4, 0, 4])
        self.assertEqual([row["state"] for row in records], ["completed", "error", "completed"])

    def test_bounded_refusal_with_wrong_slot_boundary_still_fails_closed(self):
        self.start(2, more=("--bad-refusal-boundary",))
        # A one-token request enters BGEN even when alone.
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.chat("ADMISSIONREFUSE", max_tokens=1)
        with refused.exception as response:
            self.assertEqual(response.code, 503)
            self.assertIn("missing its slot boundary", json.load(response)["error"]["message"])
        self.assertFalse(self.engine.alive())

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # the slots were reported, and a request was admitted while another decoded in a slot (the fake logs how many
        # slots were active at each admission)
        self.assertTrue(any(m.get("parallel") == 2 and len(m.get("slots", [])) == 2 for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)
        identities = {row["id"] for row in self.svc.lifecycle_records()}
        seen_owners = {}
        for snapshot in seen:
            active_slots = [slot for slot in snapshot.get("slots", []) if slot["state"] != "idle"]
            request_ids = [slot["request_id"] for slot in active_slots]
            self.assertEqual(len(request_ids), len(set(request_ids)))
            self.assertLessEqual(set(request_ids), identities)
            for slot in active_slots:
                seen_owners.setdefault(slot["slot"], set()).add(slot["request_id"])
        self.assertTrue(any(len(owners) > 1 for owners in seen_owners.values()),
                        "a departing slot takes a new identity while a peer still runs")

    def test_wire_trace_reconciles_committed_tokens_not_scheduled_rows(self):
        import serve.server as server
        self.start(2)
        trace = io.StringIO()
        with mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            replies = self.race("trace owner LONGREPLY", "trace peer", delay=0.05)
        frames = [line.split(" wire< ", 1)[1] for line in trace.getvalue().splitlines() if " wire< " in line]
        emitted = sum(frame.startswith(("T ", "BT ")) for frame in frames)
        self.assertTrue(any(frame.startswith("BT ") for frame in frames), "trace exposes resident commits")
        self.assertEqual(emitted, sum(row["output_tokens"] for row in self.get("/metrics")["requests"]))
        self.assertEqual(replies["trace owner LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertEqual(replies["trace peer"][0], "ok, done.")
        self.assertTrue(any(frame.startswith("BADM ") for frame in frames))
        self.assertTrue(any(frame.startswith("BDONE ") for frame in frames))

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """#656: a long solo prompt read gives way at a chunk boundary; the short request is answered first, then the
        long one goes on (in the slot its part waited in) and is answered too."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_three_requests_long_long_short_do_not_deadlock(self):
        """ENGINE_REVIEW finding 1 (0.1.41 check): two long prompts fill both slots, a short one arrives a moment later
        and wants one of them (the yield path).  The reviewer's repro: long, long, short at 0.25 s steps.  Every
        request must be answered; none may wait on the control lock that the yielding read holds."""
        self.start(2)
        texts = ["long " * 600, "lung " * 600, "short"]
        res, errors = {}, []

        def go(t):
            t0 = time.time()
            try:
                r = self.chat(t, max_tokens=64)
                res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
            except Exception as e:      # noqa: BLE001 - a timeout is the failure this test looks for
                errors.append((t[:6], repr(e)))
        threads = []
        for t in texts:
            th = threading.Thread(target=go, args=(t,))
            th.start()
            threads.append(th)
            time.sleep(0.25)
        for th in threads:
            th.join(45)
        self.assertFalse(any(th.is_alive() for th in threads), "a request never finished (deadlock)")
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 3)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_time_budget_silent_prefill_yields_before_the_ten_second_heartbeat(self):
        self.start(2, says_groups=2, more=("--says-budget", "--quiet-prefill"))
        replies, failures = {}, []
        def request(name, text):
            try:
                replies[name] = self.chat(text, max_tokens=12,
                                          priority="background" if name == "long" else "foreground")
            except Exception as error:
                failures.append(str(error))
        long = threading.Thread(target=request, args=("long", "long " * 600))
        long.start()
        time.sleep(.1)
        short = threading.Thread(target=request, args=("short", "short chat"))
        short.start()
        for thread in (short, long):
            thread.join(5)
        self.assertFalse(long.is_alive() or short.is_alive(), "silent prefill bypassed cooperative yield polling")
        self.assertEqual(failures, [])
        self.assertEqual(set(replies), {"long", "short"})
        log = self.log.read_text().splitlines()
        yielded = next(index for index, line in enumerate(log) if line.startswith("YIELD "))
        continued = next(index for index, line in enumerate(log) if line.startswith("PREFILL ") and
                         int(line.split()[2]) > 2500)
        self.assertLess(yielded, continued, "the short request waited for the whole silent prefill")
        self.assertFalse(any(self.engine.slot_busy))

    def test_time_budget_background_progresses_under_foreground_arrivals(self):
        self.start(3, says_groups=3, more=("--says-budget",))
        failures, replies = [], {}
        def request(name, text, priority="foreground"):
            try:
                replies[name] = self.chat(text, max_tokens=12, priority=priority)
            except Exception as error:
                failures.append(str(error))
        background = threading.Thread(target=request, args=("background", "long " * 600, "background"))
        background.start()
        arrivals = []
        for index in range(8):
            time.sleep(.08)
            thread = threading.Thread(target=request, args=(f"short-{index}", f"short arrival {index}"))
            thread.start()
            arrivals.append(thread)
        for thread in [background, *arrivals]:
            thread.join(15)
        self.assertFalse(any(thread.is_alive() for thread in [background, *arrivals]), "an arrival starved")
        self.assertEqual(failures, [])
        self.assertEqual(len(replies), 9)
        self.assertTrue(all(reply["choices"][0]["message"]["content"] == "ok, done." for reply in replies.values()))
        self.assertEqual(replies["background"]["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
                         "a cold request's own parked prefill is not reused input")
        log = self.log.read_text().splitlines()
        progress = [int(line.split()[2]) for line in log if line.startswith("PREFILL ") and int(line.split()[2]) > 64]
        self.assertGreater(max(progress), 2500, "background prefill did not keep advancing")
        yields = [int(line.split()[2]) for line in log if line.startswith("YIELD ")]
        self.assertGreater(len(yields), 2, "sustained arrivals must exercise beyond the legacy yield cap")
        self.assertEqual(yields, sorted(set(yields)), "a resumed read lost its arithmetic position")
        self.assertTrue(any(line.startswith("COMMIT ") for line in log), "decoders ran during the read")
        self.assertFalse(any(self.engine.slot_busy))

    def test_time_budget_several_decoders_and_prefill_complete_in_isolation(self):
        self.start(4, says_groups=2, more=("--says-budget",))
        replies, failures = {}, []
        def request(name, text, priority="foreground"):
            try:
                replies[name] = self.chat(text, max_tokens=60, priority=priority)
            except Exception as error:
                failures.append(str(error))
        threads = [threading.Thread(target=request, args=(str(i), f"decoder {i} LONGREPLY")) for i in range(3)]
        for thread in threads:
            thread.start()
            time.sleep(.03)
        prefill = threading.Thread(target=request, args=("prefill", "long " * 500, "background"))
        prefill.start()
        time.sleep(.15)
        short = threading.Thread(target=request, args=("short", "arriving short chat"))
        short.start()
        for thread in [*threads, prefill, short]:
            thread.join(15)
        self.assertFalse(any(thread.is_alive() for thread in [*threads, prefill, short]))
        self.assertEqual(failures, [])
        self.assertEqual(replies["prefill"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(replies["short"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(replies["prefill"]["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
                         "a cold BGEN's first parked leg must not receive internal-prefix credit")
        for index in range(3):
            self.assertEqual(replies[str(index)]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        frames = self.log.read_text().splitlines()
        first_read = next(i for i, frame in enumerate(frames) if frame.startswith("PREFILL ") and int(frame.split()[2]) > 64)
        self.assertTrue(any(frame.startswith("COMMIT ") for frame in frames[first_read + 1:]),
                        "active decoders must commit while long prefill advances")
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_adaptive_policy_preserves_solo_concurrent_solo_identity_and_wire_totals(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            self.start(2, slot_cache=True, reuse=True,
                       more=("--says-budget", "--says-spec", "--mtp", "fake", "--spec", "4"))
        policy = self.engine.speculation_policy
        self.assertIsNotNone(policy)
        for exponent in range(1, 13):
            policy.observe("solo", 1, 1 << exponent, 200, 1)
            policy.observe("batch", 1, 1 << exponent, 20, 1)
            policy.observe("batch", 2, 1 << exponent, 20, 1)
        trace = io.StringIO()
        with mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            replies = self.race("adaptive owner LONGREPLY", "adaptive peer", delay=.05, max_tokens=200)
        self.assertEqual(replies["adaptive owner LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertEqual(replies["adaptive peer"][0], "ok, done.")
        kinds = [line.split()[0] for line in self.log.read_text().splitlines()
                 if line.startswith(("GEN ", "BGEN "))]
        self.assertEqual(kinds[0], "GEN")
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN")
        frames = [line.split(" wire< ", 1)[1] for line in trace.getvalue().splitlines() if " wire< " in line]
        owners = [line.split() for line in trace.getvalue().splitlines() if " owner " in line]
        requests = self.get("/metrics")["requests"]
        self.assertEqual(sum(frame.startswith(("T ", "BT ")) for frame in frames),
                         sum(request["output_tokens"] for request in requests))
        self.assertEqual(len({line[4] for line in owners}), 2, "one stable identity per request across returns")
        self.assertTrue(any(line[6] != "None" for line in owners), "the cohort was admitted through a physical slot")
        self.assertEqual(kinds.count("GEN"), 2, "the owner's own solo leg, and its return after the peer left")
        self.assertFalse(any(self.engine.slot_busy))
        self.assertEqual(policy.slots, {})

    def test_adaptive_unprofitable_solo_uses_real_ordinary_singleton(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            self.start(2, slot_cache=True,
                       more=("--says-budget", "--says-spec", "--mtp", "fake", "--spec", "4"))
        policy = self.engine.speculation_policy
        for exponent in range(1, 13):
            policy.observe("solo", 1, 1 << exponent, 8, 2)
            policy.observe("batch", 1, 1 << exponent, 100, 1)
        reply = self.chat("ordinary fallback", max_tokens=200)
        self.assertEqual(reply["choices"][0]["message"]["content"], "ok, done.")
        kinds = [line.split()[0] for line in self.log.read_text().splitlines()
                 if line.startswith(("GEN ", "BGEN "))]
        self.assertEqual(kinds, ["BGEN"], "no invented target-only command or spurious return to MTP")
        self.assertFalse(any(self.engine.slot_busy))

    def test_adaptive_without_mtp_reports_inactive_and_retains_static_paths(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.start(2, more=("--says-budget",))
        self.assertIn("no loaded solo MTP", output.getvalue())
        self.assertIsNone(self.engine.speculation_policy)
        self.chat("ordinary no drafter")
        self.assertEqual(self.log.read_text().split()[0], "GEN")

    def test_adaptive_refuses_a_different_batch_mechanism(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            with self.assertRaisesRegex(ValueError, "ordinary batches"):
                self.start(2, slot_cache=True,
                           more=("--says-budget", "--says-spec", "--mtp", "fake", "--batch-mtp"))

    def test_adaptive_requires_the_fixed_foreground_budget(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            with self.assertRaisesRegex(ValueError, "foreground-budget-ms"):
                self.start(2, slot_cache=True, more=("--says-spec", "--mtp", "fake", "--spec", "4"))

    def test_static_default_does_not_construct_or_train_a_policy(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ):
            server.os.environ.pop("STRATA_SPEC_POLICY", None)
            self.start(2, slot_cache=True,
                       more=("--says-budget", "--says-spec", "--mtp", "fake", "--spec", "4"))
        self.assertIsNone(self.engine.speculation_policy)
        self.chat("static unchanged")
        self.assertEqual(self.log.read_text().split()[0], "GEN")

    def test_invalid_policy_is_refused_before_any_native_process(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "invalid"}), \
                mock.patch.object(server.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "static or adaptive"):
                StrataEngine("not-launched", [])
            spawn.assert_not_called()

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request decoding in a slot whose neighbour finished goes back to the solo path (MTP drafts), continued
        from its slot - only with an engine that keeps the slots' conversations (INFO slot_cache=1)."""
        self.start(2, slot_cache=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(res["long"]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11], self.engine.slot_scope[slot]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_a_failed_window_names_its_error(self):
        # #997 #890: a batch window fails while every request reads its own slot's lines, so no request reads the
        # engine's ERR line before it exits; the server's note on the death gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_a_continued_request_reports_its_own_prompt_reuse(self):
        """A request STOPped on the solo path and continued in a slot (or back on the solo path) is sent again as its
        prompt + what it generated, and the engine holds all of that: the usage must still split the request's OWN
        prompt (input_tokens + cache_read_input_tokens = the prompt, input_tokens >= 1).  input_tokens 0 makes Claude
        Code count the prompt twice (message_start's input_tokens + cache_read_input_tokens): "Prompt is too long"
        at half the context."""
        self.start(2, slot_cache=True, reuse=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.message("LONGREPLY please", 200)))
        long.start()
        time.sleep(0.05)
        self.message("short", 200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertIn("BGEN", kinds)                    # it was continued: the case under test
        u = res["long"]["usage"]
        prompt = self.svc.history[[h["output_tokens"] for h in self.svc.history].index(
            max(h["output_tokens"] for h in self.svc.history))]["prompt_total"]
        self.assertGreaterEqual(u["input_tokens"], 1, u)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], prompt, u)
        self.assertEqual(u["cache_read_input_tokens"], 3, u)    # the first read's reuse, not the continuation's


class ParkedCancellation(unittest.TestCase):
    def test_cancelled_park_is_not_continued_by_an_identical_request(self):
        # Stateful protocol peer: the server must end parked ownership, not merely
        # recycle its frontend reservation. This is not CUDA isolation evidence.
        peer = r'''import queue,sys,threading
lines=queue.Queue()
def reader():
    for line in sys.stdin: lines.put(line.strip())
threading.Thread(target=reader,daemon=True).start()
print('INFO batch_slots=2 slot_cache=1',flush=True)
print('READY 4096 stop',flush=True)
parked=False
yielded=False
while True:
    line=lines.get(timeout=10)
    if line=='QUIT': break
    if line.startswith('BSTOP '):
        parked=False
        continue
    if not line.startswith(('GEN ','BGEN ')): continue
    fields=line.split();slot=int(fields[1]) if fields[0]=='BGEN' else None
    long=len(fields[-1].split(','))>100
    if long and not yielded:
        while True:
            print('PP 64 481 1 1',flush=True)
            try: command=lines.get(timeout=.02)
            except queue.Empty: continue
            if command.startswith('BYIELD '):
                destination=int(command.split()[1]);break
        yielded=parked=True
        print(f'YIELDED {destination} 64',flush=True)
        print('DONE 0 481 1 0 cancel 0 0 0',flush=True)
        if slot is not None: print(f'BADM {slot} 0',flush=True)
        continue
    # A later identical request gets the parked identity unless cancelled.
    print('T '+str(71 if long and parked else 72 if long else 73),flush=True)
    print('DONE 1 481 1 1 stop 0 0 0',flush=True)
    if slot is not None: print(f'BADM {slot} 0',flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'peer.py'
            script.write_text(peer)
            import serve.server as server
            popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, 'Popen',
                                   lambda cmd, **kw: popen([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = StrataEngine('strata', [])
            long_cancel, short_release, long_started = threading.Event(), threading.Event(), threading.Event()
            errors, output = [], {}

            def consume(name, prompt, cancel, ready):
                try:
                    output[name] = [t for t in engine.generate(prompt, 2, {'_strata_control_ready': ready}, cancel)
                                    if t is not None]
                except Exception as error:
                    errors.append(error)

            def short_holds_control(_):
                long_cancel.set()
                if not short_release.wait(5):
                    raise TimeoutError('parked cancellation did not release its frontend slot')

            long = threading.Thread(target=consume, args=('long', [1] * 481, long_cancel,
                                                          lambda _: long_started.set()))
            short = threading.Thread(target=consume, args=('short', [2] * 4, threading.Event(), short_holds_control))
            try:
                long.start()
                self.assertTrue(long_started.wait(3))
                short.start()
                long.join(4)
                self.assertFalse(long.is_alive(), 'cancelled parked request did not terminate')
                short_release.set()
                short.join(4)
                self.assertFalse(short.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(output, {'long': [], 'short': [73]})
                again = [t for t in engine.generate([1] * 481, 2, {}, threading.Event()) if t is not None]
                self.assertEqual(again, [72], 'new request inherited cancelled parked ownership')
                self.assertFalse(any(engine.slot_busy))
            finally:
                short_release.set()
                long_cancel.set()
                engine.unload()
                long.join(5)
                if short.ident is not None:
                    short.join(5)


if __name__ == "__main__":
    unittest.main()
