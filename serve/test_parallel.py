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
FAKE_BATCH = r'''import json, os, queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args
groups = int(args[args.index("--says-groups") + 1]) if "--says-groups" in args else 0   # INFO batch_groups=G (--batch-groups auto)        # #997: the first window over two slots fails
STEP = 0.02
hold = args[args.index("--hold-replies") + 1] if "--hold-replies" in args else None
hold_badm = args[args.index("--hold-badm") + 1] if "--hold-badm" in args else None
drafts = [pair.split(",") for pair in args[args.index("--draft-counters") + 1].split(":")] if "--draft-counters" in args else None
CH = 32
lines, stop = queue.Queue(), threading.Event()
score_cancel = set()
score_lock = threading.Lock()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        elif l.startswith("SCANCEL "):
            with score_lock:
                score_cancel.add(l.split()[1])
            if "--score-cancel-file" in args:
                open(args[args.index("--score-cancel-file") + 1], "w").close()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 else "") +
      (" slot_cache=1" if "--slotcache" in args else "") + (f" batch_groups={groups}" if groups > 1 else ""), flush=True)
if "--says-multiplex" in args:
    print("INFO resident_multiplex=1", flush=True)
print("INFO score=1 score_version=1 score_max_prompt=4096 score_max_selected=256 score_max_pending=8", flush=True)
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
# #879: a prompt with NFSOLO / NFADMIT ends its first GEN / BGEN with DONE ... nonfinite (nothing emitted); one with
# NFSLOT ends its slot with BDONE ... nonfinite after its second token
fired, nf_slots = set(), set()
def window():        # one batch window: every active slot one token
    if hold and os.path.exists(hold):
        return
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        left, max_new, produced = active[b]
        if b in nf_slots and produced >= 2:
            nf_slots.discard(b)
            print(f"BDONE {b} {produced} nonfinite 1.0", flush=True)
            del active[b]
            continue
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if log and "--says-budget" in args:
            log.write(f"COMMIT {b} {produced}\n"); log.flush()
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            accepted, offered = drafts[b % len(drafts)] if drafts else (None, None)
            counts = f" draft_accepted={accepted} draft_offered={offered}" if drafts else ""
            print(f"BDONE {b} {produced} {fin} 1.0{counts}", flush=True)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
READF = "--readfield" in args   # DONE's 15th field: the prompt tokens the engine read (a continued prompt: the 1 it copies)
REUSE = "--reuse" in args     # DONE's reused field as the engine gives it: a continued prompt is all held but its last
def reused(ids):              # token (the request's own output included), a fresh one the first 3 tokens
    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3
def score_done(request_id, status, result):
    print(f"SDONE {request_id} {status} " + json.dumps(result, separators=(",", ":")), flush=True)
parked = {}
while True:
    try:
        line = lines.get(timeout=STEP if active else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if line.startswith("BPRIORITY "):
        continue
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        b = int(line.split()[1])
        parked.pop(b, None)
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith("SCORE "):
        _, request_id, payload = line.split(" ", 2)
        fields = payload.split()
        prompt = [int(t) for t in fields[1].removeprefix("prompt=").split(",")]
        selected = [int(t) for t in fields[2].removeprefix("selected=").split(",")]
        result = {"version": 1, "request_id": request_id, "generated_tokens": 0}
        if log:
            log.write(f"SCORE {request_id} {len(prompt)} {len(active)}\n"); log.flush()
        if any(t >= 258 for t in prompt + selected):
            result["error"] = "a token id is outside the vocabulary"
            score_done(request_id, "REJECT", result)
            continue
        admission_n = len(prompt) + (1 if "--score-wrong-admission" in args else 0)
        print(f"SADMIT {request_id} v1 prompt_tokens={admission_n} selected_tokens={len(selected)}", flush=True)
        if "--score-flood" in args:
            for _ in range(50):
                print(f"SADMIT {request_id} v1 prompt_tokens={len(prompt)} selected_tokens={len(selected)}", flush=True)
            continue
        for q in range(0, len(prompt), CH):
            if "--score-hold-file" in args:
                gate = args[args.index("--score-hold-file") + 1]
                while os.path.exists(gate):
                    with score_lock:
                        if request_id in score_cancel:
                            break
                    time.sleep(.005)
            time.sleep(STEP / 2)
            if active:
                window()
            with score_lock:
                cancelled = request_id in score_cancel
            if cancelled:
                if "--score-drain-gate" in args:
                    gate = args[args.index("--score-drain-gate") + 1]
                    while os.path.exists(gate):
                        time.sleep(.005)
                result["error"] = "cancelled"
                score_done(request_id, "CANCELLED", result)
                break
        else:
            # Independent FP32 row fixture: these are not sampling probabilities.
            row = {0: .125, 1: -17.5, 2: 2.25, 3: .3333333432674408}
            result.update(prompt_tokens=len(prompt), selected_token_ids=selected,
                          logits=[row.get(t, -3.0) for t in selected],
                          receipt={"prompt_tokens": len(prompt), "selected_tokens": len(selected),
                                   "logit_passes": 1, "dtype": "float32", "scale": 1})
            if "--score-missing-terminal" in args:
                continue
            if "--score-wrong-terminal" in args:
                print("DONE 0 0 0 0 stop", flush=True)
                continue
            if "--score-wrong-batch-terminal" in args:
                print(f"BDONE {request_id} 0 stop 0", flush=True)
                continue
            if "--score-wrong-id" in args:
                request_id += "x"
            if "--score-wrong-boundary" in args:
                result["prompt_tokens"] += 1
            if "--score-partial" in args:
                result["logits"] = result["logits"][:-1]
            if "--score-wrong-pass" in args:
                result["receipt"]["logit_passes"] = 2
            if "--score-generated" in args:
                result["generated_tokens"] = 1
            if "--score-reordered" in args:
                result["selected_token_ids"] = sorted(selected)
            if "--score-nonfinite" in args:
                result["logits"][0] = float("nan")
            if "--score-error" in args:
                result = {"version": 1, "request_id": request_id, "generated_tokens": 0,
                          "error": "device state corrupted"}
            score_done(request_id, "ERROR" if "--score-error" in args else "OK", result)
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
        # #1994: the engine's own view of its slots differs: BUSYSLOT is refused once, BUSYALL always
        pbx = bytes(int(x) & 255 for x in ids)
        if slot is not None and (b"BUSYALL" in pbx or (b"BUSYSLOT" in pbx and "busyslot" not in fired)):
            fired.add("busyslot")
            print(f"ERR BGEN: no such free slot (--batch {slots}) or a bad max_new", flush=True)
            if log:
                log.write(f"REFUSED {slot}\n"); log.flush()
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        pb = bytes(int(x) & 255 for x in ids)
        mark = b"NFSOLO" if f[0] == "GEN" else b"NFADMIT"
        if mark in pb and mark not in fired:
            fired.add(mark)
            print(f"DONE 0 {len(ids)} 5.0 0.0 nonfinite 0 0 0", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        if slot is not None and b"NFSLOT" in pb:
            nf_slots.add(slot)
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
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {read_reused}" +
                  (f" 0 0 0 0 0.0 {given[1] - read_reused}" if READF else ""), flush=True)
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
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {read_reused}" +
              (f" 0 0 0 0 0.0 {len(ids) - read_reused}" if READF else ""), flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > 1
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            while hold_badm and os.path.exists(hold_badm):
                time.sleep(.005)
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
    def test_parallel_above_the_window_says_so_even_with_batch_mtp(self):
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
        self.assertTrue(said(["--batch-mtp"], None))     # native admission also normalizes MTP slots to eight
        self.assertTrue(said([], "1"))

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

    def test_releasing_control_wakes_a_blocked_owner_without_a_poll_timeout(self):
        import collections
        waiting, wakeups = threading.Event(), []
        class WaitingCondition(threading.Condition):
            def wait(self, timeout=None):
                waiting.set()
                notified = super().wait(timeout=2)
                wakeups.append(notified)
                return notified
        engine = object.__new__(StrataEngine)
        engine.slot_cv = WaitingCondition()
        engine.ctl = threading.Lock()
        engine.ctl.acquire()
        engine.ctl_epoch, engine.waiting = 0, 0
        engine.wait_lens, engine.control_waiters = [], collections.deque()
        engine.alive = lambda: True
        results = []
        def request():
            def turn():
                results.append((yield from engine._take_control(threading.Event(), 77)))
            list(turn())
        thread = threading.Thread(target=request)
        thread.start()
        try:
            self.assertTrue(waiting.wait(2))
            engine._release_control()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results, [True])
            self.assertEqual(wakeups, [True], "ownership must wake on release, not the polling timeout")
            self.assertEqual(list(engine.control_waiters), [])
        finally:
            with engine.slot_cv:
                engine.slot_cv.notify_all()
            thread.join(3)
            if engine.ctl.locked():
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

    def _two_at_once(self, text_b):
        """A long request first, then one that has to go to a batch slot (the engine is busy with the first)."""
        out = {}
        def go(k, text):
            try:
                out[k] = ("ok", self.chat(text, max_tokens=64))
            except urllib.error.HTTPError as e:
                out[k] = ("http", e.code, e.read().decode(), dict(e.headers))
        ta = threading.Thread(target=go, args=("a", "LONGREPLY please"))
        ta.start()
        time.sleep(0.15)
        tb = threading.Thread(target=go, args=("b", text_b))
        tb.start()
        ta.join(30); tb.join(30)
        return out

    def attributed_chat(self, *, max_tokens=4, stream=False, stop=None, text="hello", priority="foreground"):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "stream": stream, "strata_priority": priority}
        if stop is not None:
            body["stop"] = stop
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            identity = response.headers["X-Strata-Request-Id"]
            raw = response.read().decode()
        result = ([json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")][-1]
                  if stream else json.loads(raw))
        return identity, result

    def test_bdone_draft_counters_reach_the_http_response(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "79,107"))
        identity, result = self.attributed_chat()
        self.assertTrue(identity)
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (79, 107))
        self.assertEqual(result["usage"]["completion_tokens"], 4)

    def recorded_chat(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            identity, result = self.attributed_chat(**kwargs)
            deadline = time.monotonic() + 2
            prefix = "[strata] response_attribution: "
            matching = []
            while time.monotonic() < deadline:
                records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                           if line.startswith(prefix)]
                matching = [record for record in records if record["request_id"] == identity]
                if matching:
                    break
                time.sleep(0.005)
        self.assertEqual(len(matching), 1, log.getvalue())
        return result, matching[0]

    def _yielded_accounting(self, *, reuse=False, yields=1):
        self.start(2, reuse=reuse, more=("--resident-multiplex", "--says-multiplex", "--says-budget",
                                        "--readfield", "--draft-counters", "0,0"))
        replies, failures = [], []
        def long_request():
            try:
                replies.append(self.recorded_chat(text="long " * 600, priority="background"))
            except Exception as error:
                failures.append(error)
        worker = threading.Thread(target=long_request)
        worker.start()
        def wait_read(admissions):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                lines = self.log.read_text().splitlines() if self.log.exists() else []
                long_reads = [i for i, line in enumerate(lines) if line.startswith("BGEN ") and int(line.split()[2]) > 500]
                if len(long_reads) >= admissions and any(line.startswith("PREFILL ") for line in lines[long_reads[-1] + 1:]):
                    return
                time.sleep(.005)
            self.fail("long admission did not reach its prefill barrier")
        try:
            for wave in range(yields):
                wait_read(wave + 1)
                self.attributed_chat(text=f"short {wave}")
        finally:
            worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        result, record = replies[0]
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 3 if reuse else 0)
        self.assertEqual(result["timings"]["prompt_ms"], 5 * (yields + 1))
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (4, 4))
        self.assertEqual(len(record["legs"]), yields + 1)
        self.assertEqual([leg["generated"] for leg in record["legs"]], [0] * yields + [4])
        self.assertEqual(record["legs"][-1]["admission"]["generated"], 1)
        self.assertEqual(sum(leg["prompt_read"] for leg in record["legs"]) + record["legs"][0]["reused"],
                         result["usage"]["prompt_tokens"])
        self.assertEqual(sum(leg["prompt_ms"] for leg in record["legs"]), result["timings"]["prompt_ms"])

    def test_cold_yielded_prefill_is_read_work_not_an_external_cache_hit(self):
        self._yielded_accounting()

    def test_warm_yield_preserves_original_cache_and_sums_remaining_reads(self):
        self._yielded_accounting(reuse=True)

    def test_two_yields_keep_disjoint_prompt_work_and_one_final_batch_commit(self):
        self._yielded_accounting(yields=2)

    def test_bdone_response_record_binds_identity_interval_and_slot_work(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "79,107"))
        result, record = self.recorded_chat()
        self.assertEqual(record["path"], "/v1/chat/completions")
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (4, 4))
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (79, 107))
        interval = record["interval"]
        self.assertLessEqual(interval["started_at"], interval["finished_at"])
        self.assertGreater(interval["duration_s"], 0)
        self.assertEqual(len(record["legs"]), 1)
        leg = record["legs"][0]
        self.assertEqual((leg["kind"], leg["slot"], leg["generated"], leg["decode_ms"]), ("batch", 0, 4, 1.0))
        self.assertEqual((leg["drafts_accepted"], leg["drafts_offered"]), (79, 107))
        self.assertEqual(leg["admission"]["generated"], 1)
        self.assertEqual(result["timings"]["draft_n"], 107)

    def test_close_after_admission_done_drains_badm_and_one_batch_terminal(self):
        import http.client
        import socket
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold-badm"
            gate.touch()
            self.start(2, more=("--resident-multiplex", "--says-multiplex", "--says-budget",
                                "--hold-badm", str(gate), "--draft-counters", "1,2"))
            parsed = threading.Event()
            parse_done = self.engine._parse_done
            def note_done(*args, **kwargs):
                parse_done(*args, **kwargs)
                parsed.set()
            connection = http.client.HTTPConnection(*self.httpd.server_address, timeout=5)
            response, transport = None, None
            with contextlib.redirect_stdout(io.StringIO()) as log, \
                    mock.patch.object(self.engine, "_parse_done", side_effect=note_done):
                try:
                    connection.request("POST", "/v1/chat/completions", json.dumps({
                        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4,
                        "reasoning_effort": "none", "stream": True}), {"Content-Type": "application/json"})
                    transport = connection.sock.dup()
                    response = connection.getresponse()
                    identity = response.getheader("X-Strata-Request-Id")
                    self.assertTrue(parsed.wait(3), "admission DONE was not ingested before BADM gate")
                    transport.shutdown(socket.SHUT_RDWR)
                    response.close()
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline and not any(
                            row["id"] == identity and row["state"] == "draining" for row in self.svc.lifecycle_records()):
                        time.sleep(.005)
                    self.assertTrue(any(row["id"] == identity and row["state"] == "draining"
                                        for row in self.svc.lifecycle_records()))
                    self.assertEqual(len(self.engine.slot_claims), 1)
                    claimed = next(iter(self.engine.slot_claims))
                    self.assertTrue(self.engine.slot_busy[claimed], "claim released before BADM/BDONE")
                    gate.unlink()
                    prefix, records = "[strata] response_attribution: ", []
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                                   if line.startswith(prefix)]
                        records = [record for record in records if record["request_id"] == identity]
                        if records: break
                        time.sleep(.005)
                    self.assertEqual(len(records), 1, log.getvalue())
                    record = records[0]
                    self.assertEqual(len(record["legs"]), 1)
                    terminal = record["legs"][0]
                    self.assertEqual((terminal["kind"], terminal["admission"]["generated"]), ("batch", 1))
                    self.assertEqual(record["committed_tokens"], terminal["generated"])
                    self.assertEqual(record["drafts_offered"], 2)
                    self.assertFalse(self.engine.slot_claims)
                    self.assertFalse(any(self.engine.slot_busy))
                finally:
                    gate.unlink(missing_ok=True)
                    if transport: transport.close()
                    if response: response.close()
                    connection.close()

    def test_bdone_counters_survive_early_stop_and_synchronous_drain(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "1,2"))
        result, record = self.recorded_chat(max_tokens=64, stop="ok")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (1, 2))
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (3, 2))
        self.assertEqual(record["legs"][0]["kind"], "batch")
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (1, 2))

    def test_bdone_counters_survive_streaming_eos_drain(self):
        import serve.server as server
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "7,9"))
        with mock.patch.object(server, "EOS_IDS", {257}):
            result, record = self.recorded_chat(max_tokens=64, stream=True)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (7, 9))
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (10, 10))
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (7, 9))
        self.assertEqual(record["legs"][0]["generated"], 10)

    def test_bdone_without_drafts_reports_explicit_zeros(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "0,0"))
        result, record = self.recorded_chat()
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (0, 0))
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (0, 0))
        self.assertEqual((record["legs"][0]["drafts_accepted"], record["legs"][0]["drafts_offered"]), (0, 0))

    def test_bdone_without_counters_does_not_borrow_the_admission_zeros(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex"))
        result, record = self.recorded_chat()
        self.assertNotIn("draft_n", result["timings"])
        self.assertNotIn("draft_n_accepted", result["timings"])
        self.assertIsNone(record["drafts_accepted"])
        self.assertIsNone(record["drafts_offered"])
        self.assertIsNone(record["legs"][0]["drafts_offered"])
        self.assertEqual(record["legs"][0]["admission"]["drafts_offered"], 0)

    def test_equal_count_responses_keep_their_slot_counters_by_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            gate.touch()
            self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "1,2:2,3",
                                "--hold-replies", str(gate)))
            responses, errors = [], []

            def go():
                try:
                    responses.append(self.attributed_chat())
                except Exception as error:
                    errors.append(error)

            prefix = "[strata] response_attribution: "
            with contextlib.redirect_stdout(io.StringIO()) as log:
                threads = [threading.Thread(target=go) for _ in range(2)]
                for thread in threads:
                    thread.start()
                try:
                    deadline = time.monotonic() + 3
                    ready = False
                    while not ready and time.monotonic() < deadline:
                        with self.engine.slot_cv:
                            ready = sum(bool(slot) and slot["state"] == "decoding"
                                        for slot in self.engine.slot_live) == 2
                        if not ready:
                            time.sleep(0.005)
                    self.assertTrue(ready, "both requests must enter different native slots")
                finally:
                    gate.unlink(missing_ok=True)
                    for thread in threads:
                        thread.join(10)
                deadline = time.monotonic() + 2
                while log.getvalue().count(prefix) < 2 and time.monotonic() < deadline:
                    time.sleep(0.005)
            self.assertEqual(errors, [])
            self.assertEqual(len(responses), 2)
            records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                       if line.startswith(prefix)]
            self.assertEqual(len(records), 2, log.getvalue())
            by_id = {record["request_id"]: record for record in records}
            self.assertEqual(len(by_id), 2)
            self.assertEqual({record["legs"][0]["slot"] for record in records}, {0, 1})
            for identity, response in responses:
                record = by_id[identity]
                slot = record["legs"][0]["slot"]
                expected = (1, 2) if slot == 0 else (2, 3)
                self.assertEqual((response["timings"]["draft_n_accepted"], response["timings"]["draft_n"]), expected)
                self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), expected)
                self.assertEqual((record["committed_tokens"], record["output_tokens"]), (4, 4))

    def test_bgen_refused_by_the_engine_is_retried_not_a_400(self):
        """#1994: the engine refuses a BGEN ("no such free slot") although the server saw the slot free: the request
        goes to another slot (or waits) and is answered; the same conversation's next turn works too."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.2
        out = self._two_at_once("BUSYSLOT hello")
        self.assertEqual(out["b"][0], "ok", out["b"])
        self.assertIn("REFUSED", self.log.read_text())
        again = self.chat("BUSYSLOT hello")                     # the conversation is not poisoned
        self.assertEqual(again["choices"][0]["message"]["content"], "ok, done.")
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])

    def test_bgen_refused_for_good_is_a_retryable_503_and_frees_the_slots(self):
        """#1994: when the engine keeps refusing, the answer is a 503 (retryable), never a 400, and no slot stays busy."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.05
        out = self._two_at_once("BUSYALL hello")
        self.assertEqual(out["b"][0], "http", out["b"])
        self.assertEqual(out["b"][1], 503, out["b"])
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])
        self.assertEqual(self.chat("fine again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_requested_nine_slots_report_effective_eight(self):
        """Frontend uses native INFO after normalization; the CPU planner test pins native 9 -> 8."""
        self.start(9, fit=8, says_groups=2, more=("--batch-groups", "2"))
        self.assertEqual(self.engine.batch, 8)
        self.assertEqual(self.engine.slot_groups, 2)
        self.assertEqual(len(self.engine.slot_busy), 8)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 8)

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


    def test_quiet_resident_cancel_does_not_wait_for_another_output_token(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            gate.touch()
            self.start(2, more=("--hold-replies", str(gate)))
            engine = self.engine
            engine.slot_busy[1] = True
            engine.slot_claims.add(1)  # another owner prevents solo promotion
            cancel, admitted, stop_written = threading.Event(), threading.Event(), threading.Event()
            errors, commands = [], []
            send = engine._send

            def observe(command):
                send(command)
                commands.append(command)
                if command.startswith("BSTOP "):
                    stop_written.set()

            output = engine.generate_batched(list(b"LONGREPLY"), 64, {}, cancel)
            def consume():
                try:
                    for token in output:
                        if isinstance(token, int):
                            cancel.set()  # admission token; the peer withholds every resident BT
                            admitted.set()
                except Exception as error:
                    errors.append(error)

            worker = threading.Thread(target=consume, daemon=True)
            with mock.patch.object(engine, "_send", side_effect=observe):
                worker.start()
                try:
                    self.assertTrue(admitted.wait(2), "the request never reached its resident phase")
                    self.assertTrue(stop_written.wait(2), "quiet cancellation waited for output or the old ten-second poll")
                    self.assertTrue(worker.is_alive(), "ownership was released before the peer's terminal")
                    self.assertIn(0, engine.slot_claims)
                finally:
                    gate.unlink(missing_ok=True)  # permit the peer's matching cancellation terminal
                    worker.join(3)
                    engine.slot_busy[1] = False
                    engine.slot_claims.discard(1)
                    if not worker.is_alive():
                        output.close()
            self.assertFalse(worker.is_alive(), "the cancellation lifecycle did not drain")
            self.assertEqual(errors, [])
            self.assertNotIn("STOP", commands, "resident cancellation must not stop another owner's prompt read")
            self.assertNotIn(0, engine.slot_claims)
            self.assertEqual(self.chat("successor")["choices"][0]["message"]["content"], "ok, done.")

    def test_early_close_does_not_release_successor_reservation(self):
        self.start(2)
        engine = self.engine
        engine.slot_busy[1] = True  # peer reservation forces the real BGEN/BDONE path
        engine.slot_claims.add(1)
        release = engine._release_slot_when_done
        successor = []
        def reserve_after_drain(slot, stream, owner):
            release(slot, stream, owner)
            with engine.slot_cv:
                chosen = engine.pick_slot([], "")
                self.assertEqual(chosen, slot)
                self.assertNotIn(slot, engine.slot_claims, "drain must retire the old claim before publishing availability")
                engine.slot_busy[chosen] = True
                engine.slot_claims.add(chosen)
                successor.append(chosen)
        output = engine.generate_batched(list(b'LONGREPLY'), 64, {}, threading.Event())
        try:
            with mock.patch.object(engine, '_release_slot_when_done', side_effect=reserve_after_drain):
                self.assertIsInstance(next(output), int)
                self.assertIsInstance(next(output), int)  # first BT after BADM, not admission T
                output.close()
            self.assertEqual(len(successor), 1)
            self.assertTrue(engine.slot_busy[successor[0]], 'old consumer released the successor reservation')
            self.assertIn(successor[0], engine.slot_claims, "old finalizer erased the successor claim")
            with engine.slot_cv:
                engine.ORPHAN_SLOT_S = 0
                self.assertEqual(engine._reap_orphan_slots(), [])
                self.assertTrue(engine.slot_busy[successor[0]], "orphan scan stole an owned successor")
        finally:
            for slot in successor:
                engine.slot_busy[slot] = False
                engine.slot_claims.discard(slot)
            engine.slot_busy[1] = False
            engine.slot_claims.discard(1)
            output.close()

    def test_old_generation_finalizer_keeps_new_generation_claim(self):
        self.start(2)
        engine = self.engine
        engine.slot_busy[1] = True  # force slot admission rather than solo generation
        output = engine.generate_batched(list(b"LONGREPLY"), 64, {}, threading.Event())
        try:
            self.assertIsInstance(next(output), int)
            self.assertIsInstance(next(output), int)
            old_busy, old_claims = engine.slot_busy, engine.slot_claims
            self.assertEqual(old_claims, {0})
            import serve.server as server
            real = server.subprocess.Popen
            script = Path(self.tmp.name) / "fake_strata.py"
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine.restart(tries=1)
            with engine.slot_cv:
                chosen = engine.pick_slot([], "")
                self.assertEqual(chosen, 0)
                engine.slot_busy[chosen] = True
                engine.slot_claims.add(chosen)
            output.close()  # old-generation finally executes only now
            self.assertFalse(old_busy[0])
            self.assertEqual(old_claims, set(), "old generation's claim was not retired")
            self.assertTrue(engine.slot_busy[chosen])
            self.assertEqual(engine.slot_claims, {chosen}, "old generation discarded the new generation's claim")
            with engine.slot_cv:
                engine.ORPHAN_SLOT_S = 0
                self.assertEqual(engine._reap_orphan_slots(), [])
        finally:
            output.close()
            with engine.slot_cv:
                engine.slot_busy[:] = [False] * engine.batch
                engine.slot_claims.clear()

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
                    self.assertGreaterEqual(observed["retry_after_s"], 1)
                    self.assertLessEqual(observed["retry_after_s"], observed["retry_after_ceiling_s"])
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.chat("beyond the bounded capacity")
                with refused.exception as response:
                    self.assertEqual(response.code, 429)
                    self.assertIn(int(response.headers["Retry-After"]), range(1, 31))
                    self.assertIn("queue is full", json.load(response)["error"]["message"])
                with self.svc.status_lock:
                    if depth:
                        self.svc.admission_waiters[0]["_clock"] -= 40
                    else:
                        next(iter(self.svc.admission_active.values()))["_clock"] -= 40
                for path in ("/status", "/metrics", "/health", "/v1/status"):
                    old = self.get(path)["admission"]
                    self.assertEqual((old["active"], old["waiting"]), (2, depth))
                    self.assertEqual((old["retry_after_s"], old["retry_after_ceiling_s"]), (30, 30))
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.chat("retry while the queue is still full")
                with refused.exception as response:
                    self.assertEqual(response.code, 429)
                    self.assertEqual(response.headers["Retry-After"], "30")
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
            self.assertEqual(snapshot["retry_after_s"], 1)

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
        self.start(2, slot_cache=True, reuse=True, more=("--readfield", "--draft-counters", "7,11"))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.recorded_chat(text="LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        response, record = res["long"]
        self.assertEqual(response["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertEqual([leg["kind"] for leg in record["legs"]], ["solo", "batch", "solo"])
        self.assertTrue(all(leg["generated"] > 0 for leg in record["legs"]))
        self.assertEqual(record["committed_tokens"], response["usage"]["completion_tokens"])
        self.assertEqual(record["committed_tokens"], sum(leg["generated"] for leg in record["legs"]))
        self.assertEqual((response["timings"]["draft_n_accepted"], response["timings"]["draft_n"]), (7, 11))
        self.assertEqual(response["timings"]["prompt_ms"], sum(leg["prompt_ms"] for leg in record["legs"]))
        self.assertEqual(response["timings"]["prompt_ms"], 15.0)
        self.assertEqual(response["usage"]["prompt_tokens_details"]["cached_tokens"], 3)
        self.assertEqual(sum(leg["prompt_read"] for leg in record["legs"]), response["usage"]["prompt_tokens"] - 3 + 2)
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

    def test_nonfinite_in_batch_mode(self):
        """#879 in --batch mode: DONE ... nonfinite on the solo path or at an admission (nothing emitted yet) sends the
        request again once; BDONE ... nonfinite in a slot ends the reply that had started, as "length"."""
        self.start(2)
        self.assertEqual(self.chat("NFSOLO alone")["choices"][0]["message"]["content"], "ok, done.")
        gens = [x for x in self.log.read_text().splitlines() if x.startswith("GEN")]
        self.assertEqual(len(gens), 2, gens)                              # sent again once
        res = self.race("first question LONGREPLY", "NFADMIT second")
        self.assertEqual(res["NFADMIT second"][0], "ok, done.")
        self.assertEqual(res["first question LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        out = {}

        def go(t):
            body = {"messages": [{"role": "user", "content": t}], "max_tokens": 64, "reasoning_effort": "none"}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out[t] = json.loads(r.read().decode())["choices"][0]
        a = threading.Thread(target=go, args=("third question LONGREPLY",))
        a.start()
        time.sleep(0.15)
        go("NFSLOT fourth")
        a.join(30)
        c = out["NFSLOT fourth"]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok", "length"))
        self.assertEqual(out["third question LONGREPLY"]["message"]["content"], "ok, " + "la " * 15 + "done.")
        # the engine dropped that slot's state: the server does not route the conversation's next turn to it
        self.assertFalse([h for h in self.engine.slot_held if bytes(t & 255 for t in h).find(b"NFSLOT") >= 0])
        with self.svc.status_lock:
            self.assertEqual(self.svc.live_reqs, {})

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

    def test_a_slot_request_ended_by_its_stop_token_reports_the_slot_leg(self):
        """#1869: the consumer leaves at the stop token, before the BDONE: the request kept the admission's one token and
        its few ms.  Its figures are the slot leg's now (finish=stop was the broken case, finish=length read the BDONE)."""
        import serve.server as server
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):          # the fake engine's stop token
            a = threading.Thread(target=lambda: res.setdefault("a", self.chat("List three primes please", 200)))
            b = threading.Thread(target=lambda: res.setdefault("b", self.chat("Name two rivers in Germany", 200)))
            a.start(); b.start(); a.join(30); b.join(30)
        self.assertEqual(len(self.svc.history), 2)
        for h in self.svc.history:
            self.assertEqual(h["engine_generated"], h["output_tokens"], h)
            self.assertGreaterEqual(h["decode_ms"], 1.0, h)
        for r in res.values():
            t = r["timings"]
            self.assertEqual(t["prompt_n"] + t["cache_n"], r["usage"]["prompt_tokens"], t)
            self.assertEqual(t["predicted_n"], 10, t)
        self.assertFalse(any(self.engine.slot_busy))                 # the slots are free (no BDONE left for a wait)

    def test_a_request_moved_between_solo_and_slot_keeps_its_totals(self):
        """#1869/#1870: a request that read its prompt and decoded on the solo path, was STOPped when another arrived and
        went on in a slot (its admission copies the held prefix: prompt_read 1, one token) reports the prompt it read and
        the decode time and tokens of ALL its legs, not the last one's."""
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.25)                                   # a few solo tokens first
        self.chat("short", max_tokens=200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)           # solo, then continued in a slot: the case under test
        n = res["long"]["usage"]["completion_tokens"]
        row = max(self.svc.history, key=lambda h: h["output_tokens"])
        self.assertEqual(row["engine_generated"], n, row)
        self.assertGreater(row["decode_ms"], 20.0 * 3, row)          # the solo leg's STEP-long tokens are in it
        self.assertGreater(row["prompt_read"], 1, row)                # the first read, not the copy leg's 1
        t = res["long"]["timings"]
        self.assertEqual(t["prompt_n"] + t["cache_n"], res["long"]["usage"]["prompt_tokens"], t)
        self.assertEqual(t["cache_n"], 3, t)                          # the first read's reuse only
        self.assertGreater(t["predicted_ms"], 60.0, t)

    def test_a_leaked_busy_slot_does_not_hang_the_next_request(self):
        """#1603: a slot whose busy flag has no owner (no request holds it, no BDONE wait is running) kept every later
        request waiting for a slot nothing would free: running 0, queued 0, and the frozen-engine check never looked.
        The waiting request is now counted (admitting), the slot is given back after ORPHAN_SLOT_S, and it is
        answered."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.4
        with e.slot_cv:
            e.slot_busy[0] = e.slot_busy[1] = True       # the leak: busy, claimed by nobody, nothing ending them
        # the solo path needs a quiet engine; a busy flag sends the request to the slot path, which waits
        res, errs = {}, []
        def go():
            try:
                res["r"] = self.chat("after the leak", max_tokens=16)
            except Exception as ex:                       # noqa: BLE001
                errs.append(ex)
        t = threading.Thread(target=go)
        t0 = time.time()
        t.start()
        seen = []
        while t.is_alive() and time.time() - t0 < 20:
            live = self.get("/metrics")["live"]
            seen.append((live.get("running"), live.get("queued"), live.get("admitting")))
            time.sleep(0.05)
        t.join(5)
        self.assertEqual(errs, [])
        self.assertFalse(t.is_alive(), "the request hung on a slot nobody owns")
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        # while it waited it was visible: running as a request, and counted as waiting for a slot - never 0 / 0 / 0
        self.assertTrue(any(r == 1 and a == 1 for r, q, a in seen), seen)
        first = next(i for i, x in enumerate(seen) if sum(v or 0 for v in x))
        self.assertTrue(all(sum(v or 0 for v in x) for x in seen[first:-1]), seen)   # never invisible once it arrived
        self.assertEqual(e.admitting, 0)
        self.assertEqual(e.slot_claims, set())

    def test_an_owned_busy_slot_is_never_taken(self):
        """The leak check frees only slots with no owner: a slot a request holds stays busy however long it waits."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.0
        with e.slot_cv:
            e.slot_busy[0] = True
            e.slot_claims.add(0)
            e.slot_busy[1] = True
            e.slot_releasing.add(1)
            self.assertEqual(e._reap_orphan_slots(), [])
            self.assertEqual(e.slot_busy, [True, True])
            e.slot_claims.discard(0)
            self.assertEqual(e._reap_orphan_slots(), [0])
            self.assertEqual(e.slot_busy, [False, True])

    def test_a_parallel_request_does_not_wait_for_the_fifo_on_a_loaded_engine(self):
        """#1603: parallel requests used to take the FIFO (held by an image encode, /v1/vram, load) just to find the
        engine loaded; between 'queued' and 'running' they were neither.  Loaded: no fifo.  Unloaded: they wait for it
        and are counted as queued the whole time."""
        self.start(2)
        self.assertEqual(self.chat("warm")["choices"][0]["message"]["content"], "ok, done.")
        self.svc.fifo.acquire()
        try:
            r = self.chat("with the fifo held")             # would block here before
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, done.")
        finally:
            self.svc.fifo.release()
        import serve.server as server
        script, real = Path(self.tmp.name) / "fake_strata.py", server.subprocess.Popen
        patch = mock.patch.object(server.subprocess, "Popen",      # the reload starts the fake again
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine.unload()
        self.svc.fifo.acquire()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", self.chat("reloads", max_tokens=16)))
        t.start()
        try:
            deadline = time.time() + 10
            live = {}
            while time.time() < deadline:
                live = self.get("/metrics")["live"]
                if live.get("queued"):
                    break
                time.sleep(0.05)
            self.assertEqual(live.get("queued"), 1, live)    # waiting for the fifo: queued, not nothing
            self.assertEqual(live.get("running"), 0, live)
        finally:
            self.svc.fifo.release()
        t.join(60)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/metrics")["live"]["queued"], 0)


class ScoreTransport(unittest.TestCase):
    """#233: the public bounded adapter over the existing scripted engine pipe."""
    start = ParallelService.start
    tearDown = ParallelService.tearDown
    chat = ParallelService.chat
    get = ParallelService.get

    def score(self, prompt=None, selected=None, **extra):
        body = {"version": 1, "prompt_token_ids": prompt if prompt is not None else [5, 6],
                "selected_token_ids": selected if selected is not None else [3, 1, 0, 3]}
        request = urllib.request.Request(self.base + "/v1/score", data=json.dumps(body | extra).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def test_complete_score_has_ordered_raw_fp32_and_identity_receipt(self):
        self.start(2)
        proc = self.engine.proc
        result = self.score()
        self.assertEqual(result["logits"], [.3333333432674408, -17.5, .125, .3333333432674408])
        self.assertEqual(result["selected_token_ids"], [3, 1, 0, 3])
        self.assertEqual(result["generated_tokens"], 0)
        self.assertEqual(result["receipt"], {"prompt_tokens": 2, "selected_tokens": 4,
                                           "logit_passes": 1, "dtype": "float32", "scale": 1})
        record = self.svc.lifecycle_records(result["request_id"])
        self.assertEqual((record["state"], record["output_tokens"]), ("completed", 0))
        self.assertIs(self.engine.proc, proc)
        self.assertFalse(self.engine.score_q)

    def test_rejected_score_is_nonfatal_while_generation_interleaves_and_successor_scores(self):
        self.start(2)
        result, errors = [], []
        def generate():
            try:
                result.append(self.chat("survivor LONGREPLY"))
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=generate)
        thread.start()
        try:
            deadline = time.monotonic() + 3
            while not self.log.exists() and time.monotonic() < deadline:
                time.sleep(.005)
            proc = self.engine.proc
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.score(selected=[999])
            with rejected.exception as response:
                self.assertEqual(response.code, 400)
            self.assertEqual(self.score(prompt=[5] * 96)["generated_tokens"], 0)
            self.assertIs(self.engine.proc, proc)
            self.assertTrue(self.engine.alive())
        finally:
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result[0]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(line.startswith("SCORE ") and int(line.split()[3]) > 0
                            for line in self.log.read_text().splitlines()))
        self.assertEqual(self.chat("successor")["choices"][0]["message"]["content"], "ok, done.")

    def assert_protocol_failure(self, flag, text):
        self.start(2, more=(flag,))
        self.engine.SCORE_TIMEOUT_S = .2
        self.engine.SCORE_DRAIN_S = .2
        with self.assertRaises(urllib.error.HTTPError) as failed:
            self.score()
        with failed.exception as response:
            self.assertEqual(response.code, 503)
            body = json.load(response)
            self.assertIn(text, body["error"]["message"])
            self.assertNotIn("logits", body)
        self.assertFalse(self.engine.alive(), "ambiguous or corrupt score state must be retired")
        self.assertFalse(self.engine.score_q)
        self.assertFalse(self.engine.ctl.locked())

    def test_wrong_generation_terminal_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-terminal", "wrong score terminal")

    def test_wrong_score_id_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-id", "score terminal ID")

    def test_missing_terminal_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-missing-terminal", "missing matching SDONE")

    def test_mismatched_prompt_boundary_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-boundary", "receipt boundary mismatch")

    def test_mismatched_admission_boundary_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-admission", "admission boundary")

    def test_score_cannot_hide_a_wrong_terminal_in_generation_slot_routing(self):
        self.assert_protocol_failure("--score-wrong-batch-terminal", "wrong score terminal")

    def test_partial_option_set_is_never_a_successful_answer(self):
        self.assert_protocol_failure("--score-partial", "incomplete option set")

    def test_wrong_pass_receipt_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-pass", "receipt boundary mismatch")

    def test_generated_token_count_cannot_be_disguised_as_scoring(self):
        self.assert_protocol_failure("--score-generated", "generated-token boundary mismatch")

    def test_selected_token_order_receipt_cannot_be_changed(self):
        self.assert_protocol_failure("--score-reordered", "receipt boundary mismatch")

    def test_nonfinite_logit_never_publishes_an_option_set(self):
        self.assert_protocol_failure("--score-nonfinite", "non-finite")

    def test_device_or_state_error_is_not_an_ordinary_rejection(self):
        self.assert_protocol_failure("--score-error", "fatal score engine/state error")

    def test_duplicate_nonterminal_cannot_flood_the_reply_queue(self):
        self.assert_protocol_failure("--score-flood", "duplicate score admission")

    def wait_for(self, predicate, text):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate(), text)

    def test_disconnect_during_prompt_read_keeps_admission_until_matching_terminal_drains(self):
        import socket
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gate, cancelled = Path(directory.name) / "drain", Path(directory.name) / "cancelled"
        gate.touch()
        self.start(2, more=("--score-drain-gate", str(gate), "--score-cancel-file", str(cancelled)))
        body = json.dumps({"version": 1, "prompt_token_ids": [5] * 4096, "selected_token_ids": [3, 1]}).encode()
        client = socket.create_connection(self.httpd.server_address, timeout=3)
        self.addCleanup(client.close)
        client.sendall(b"POST /v1/score HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: " +
                       str(len(body)).encode() + b"\r\n\r\n" + body)
        self.wait_for(lambda: self.log.exists() and "SCORE " in self.log.read_text(), "score submitted")
        request_id = self.log.read_text().split()[1]
        client.close()
        self.wait_for(cancelled.exists, "disconnect sends SCANCEL during prompt read")
        self.assertIn(request_id, self.engine.score_q)
        self.assertTrue(self.engine.ctl.locked())
        self.assertIn(request_id, self.svc.admission_active)
        self.assertFalse(self.engine.score_q[request_id]["terminal"])
        self.assertEqual(self.svc.unload(), "busy", "disconnect cannot unload before score terminal drain")
        results, errors = [], []
        def successor():
            try:
                results.append(self.score())
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=successor)
        thread.start()
        try:
            self.wait_for(lambda: len(self.svc.admission_waiters) == 1, "successor waits behind the draining score owner")
            self.assertEqual(list(self.svc.admission_active), [request_id])
            self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 1)
        finally:
            gate.unlink(missing_ok=True)
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results[0]["generated_tokens"], 0)
        self.assertNotEqual(results[0]["request_id"], request_id)
        self.wait_for(lambda: not self.svc.admission_active, "both admission owners released after drain")
        self.assertFalse(self.engine.score_q)
        self.assertTrue(self.engine.alive())

    def test_http_score_uses_its_own_bounded_admission_queue(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gate = Path(directory.name) / "hold"
        gate.touch()
        self.start(2, more=("--score-hold-file", str(gate)))
        self.svc.request_queue_limit = 0
        self.engine.info["score_max_pending"] = 2
        results, errors = [], []
        def request():
            try:
                results.append(self.score())
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        try:
            self.wait_for(lambda: len(self.svc.admission_active) == 1 and len(self.svc.admission_waiters) == 1,
                          "one score owns control and the other occupies its bounded queue")
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.score()
            with rejected.exception as response:
                self.assertEqual(response.code, 429)
                self.assertIn("queue is full", json.load(response)["error"]["message"])
            self.assertEqual(len(self.engine.score_q), 1)
            self.assertEqual(next(iter(self.engine.score_q.values()))["queue"].maxsize, 2)
        finally:
            gate.unlink(missing_ok=True)
            for thread in threads:
                thread.join(10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertTrue(self.engine.alive())

    def test_direct_native_waiter_count_is_bounded(self):
        from serve.server import AdmissionError
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gate = Path(directory.name) / "hold"
        gate.touch()
        self.start(2, more=("--score-hold-file", str(gate)))
        self.engine.info["score_max_pending"] = 2
        request = {"version": 1, "prompt_token_ids": [5], "selected_token_ids": [0]}
        cancels, errors = [threading.Event(), threading.Event()], []
        def native(cancel):
            try:
                self.engine.score(request, cancel)
            except AdmissionError as error:
                errors.append(error)
        threads = [threading.Thread(target=native, args=(cancel,)) for cancel in cancels]
        for thread in threads:
            thread.start()
        try:
            self.wait_for(lambda: self.engine.score_waiters == 2, "bounded direct callers own the queue")
            with self.assertRaisesRegex(AdmissionError, "score queue is full"):
                self.engine.score(request, threading.Event())
        finally:
            for cancel in cancels:
                cancel.set()
            gate.unlink(missing_ok=True)
            for thread in threads:
                thread.join(10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), 2)
        self.assertEqual(self.engine.score_waiters, 0)
        self.assertFalse(self.engine.score_q)

    def test_work_and_payload_shape_are_bounded_before_submission(self):
        self.start(2)
        invalid = [({"prompt": [5] * 4097}, "prompt_token_ids"),
                   ({"selected": [0] * 257}, "selected_token_ids"),
                   ({"selected": [True]}, "int32"), ({"selected": [1.0]}, "int32"),
                   ({"prompt": []}, "prompt_token_ids"), ({"selected": []}, "selected_token_ids"),
                   ({"version": 2}, "version"), ({"temperature": .5}, "unsupported score fields"),
                   ({"max_tokens": 0}, "unsupported score fields")]
        for body, reason in invalid:
            with self.subTest(body_keys=list(body)):
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.score(**body)
                with refused.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn(reason, json.load(response)["error"]["message"])
        self.assertEqual(self.log.read_text(), "", "refused work never reaches any native command")
        self.assertEqual(self.score()["generated_tokens"], 0)
        self.assertEqual(self.get("/health")["scoring"],
                         {"available": True, "version": 1, "max_prompt_tokens": 4096,
                          "max_selected_tokens": 256, "max_pending": 8})

    def test_serial_generation_and_score_keep_their_protocols_separate(self):
        import serve.server as server
        self.start(0)
        trace = io.StringIO()
        with mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            self.assertEqual(self.score()["generated_tokens"], 0)
        frames = [line.split(" wire< ", 1)[1] for line in trace.getvalue().splitlines() if " wire< " in line]
        self.assertEqual(sum(frame.startswith("SDONE ") for frame in frames), 1)
        self.assertFalse(any(frame.startswith(("T ", "BT ", "DONE ", "BDONE ")) for frame in frames))
        self.assertEqual(self.chat("generation")["choices"][0]["message"]["content"], "ok, done.")

    def test_failed_score_write_never_reuses_ambiguous_ownership(self):
        from serve.server import EngineDied
        self.start(2)
        request = {"version": 1, "prompt_token_ids": [5], "selected_token_ids": [0]}
        with mock.patch.object(self.engine, "_send", side_effect=EngineDied("partial score write")):
            with self.assertRaises(EngineDied):
                self.engine.score(request, threading.Event())
        self.assertFalse(self.engine.alive(), "a failed write and failed cancel drain cannot release live ownership")
        self.assertFalse(self.engine.score_q)
        self.assertFalse(self.engine.ctl.locked())



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
        print('DONE 0 481 1 0 cancel 0 0 0 0 0 0 0 0.0 64 0',flush=True)
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
            errors, output, native_work = [], {}, {}

            def consume(name, prompt, cancel, ready):
                try:
                    output[name] = [t for t in engine.generate(prompt, 2, {'_strata_control_ready': ready}, cancel)
                                    if t is not None]
                    native_work[name] = dict(engine.last)
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
                self.assertEqual(len(native_work["long"]["legs"]), 1)
                self.assertEqual((native_work["long"]["prompt_read"], native_work["long"]["prompt_ms"]), (64, 1.0))
                self.assertEqual(native_work["long"]["generated"], 0)
                cancelled = threading.Event()
                cancelled.set()
                engine.last = {"generated": 999, "prompt_ms": 123, "legs": [{"kind": "solo", "generated": 999}]}
                self.assertEqual([t for t in engine.generate([3] * 4, 2, {}, cancelled) if t is not None], [])
                self.assertEqual(engine.last["legs"], [])
                self.assertIsNone(engine.last.get("generated"))
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
