"""#879: a Stop that arrives while the engine reads the prompt (PP lines, no tokens yet) is sent at once, not after the
first token."""
import os
import queue
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import server  # noqa: E402


class ControlCancelTest(unittest.TestCase):
    def _engine(self, lines):
        eng = object.__new__(server.StrataEngine)
        eng.lines = queue.Queue()
        for l in lines:
            eng.lines.put(l)
        eng.sent = []
        eng._send = eng.sent.append
        eng._ctl_mode = "solo"
        eng._ctl_result = None
        eng.progress = None
        eng.prefill_tok_s_mean = None
        eng._last_done = None
        eng._parse_done = lambda line, slot=None: None
        return eng

    def _resident_engine(self, identity):
        eng = server.StrataEngine("not-started", [], lazy=True)
        eng.gen, eng.batch = 1, 1
        eng.slot_q, eng.slot_busy, eng.slot_held = [queue.Queue()], [True], [[]]
        eng.slot_claims, eng.slot_releasing = {0}, set()
        eng._slot_identity = [identity]
        eng.sent = []
        eng._send = eng.sent.append
        eng.alive = lambda: True
        eng.last = {"generated": 1, "finish": "length", "prompt_ms": 2.0}
        owner = (eng.gen, eng.slot_q, eng.slot_busy, eng.slot_held, eng.slot_claims,
                 eng.slot_releasing, eng._slot_identity)
        return eng, owner

    def test_badm_records_identity_on_control_and_early_close_drain(self):
        for boundary, expected in (("BADM 0 1 request=17 incarnation=2", (17, 2)),
                                   ("BADM 0 0 request=17 incarnation=3", (17, 3)),
                                   ("BADM 0 1", None), ("BADM 0 0", None)):
            for draining in (False, True):
                with self.subTest(boundary=boundary, draining=draining):
                    eng = self._engine(["DONE 1", boundary])
                    eng._slot_identity = [(9, 1)]  # never inherit a previous admission's identity
                    eng._ctl_mode = "batch"
                    if draining:
                        eng._drain_control("BADM", slot=0)
                    else:
                        list(eng._control(threading.Event(), lambda _: None, slot=0))
                    self.assertEqual(eng._slot_identity[0], expected)

    def test_identity_drain_ignores_stale_terminals_before_releasing_the_slot(self):
        eng, owner = self._resident_engine((18, 3))
        eng.slot_q[0].put("BDONE 0 99 nonfinite 500 request=17 incarnation=3")
        eng.slot_q[0].put("BDONE 0 99 cancel 500 request=18 incarnation=2")
        eng.slot_q[0].put("BT 0 4")
        eng.slot_q[0].put("BDONE 0 2 cancel 5 draft_accepted=1 draft_offered=2 request=18 incarnation=3")
        read = eng.slot_q[0].get
        def observe(*args, **kwargs):
            if eng.slot_q[0].qsize() == 2:  # both stale terminals were consumed, but neither ended ownership
                self.assertTrue(eng.slot_busy[0])
                self.assertEqual(eng.slot_claims, {0})
                self.assertEqual(eng.slot_releasing, {0})
                self.assertEqual(eng.last["generated"], 1)
            return read(*args, **kwargs)
        with mock.patch.object(eng.slot_q[0], "get", side_effect=observe):
            eng._release_slot_when_done(0, [1, 2], owner)
        self.assertEqual(eng.sent, ["BSTOP 0 18 3"])
        self.assertEqual((eng.last["generated"], eng.last["finish"], eng.last["decode_ms"]), (2, "cancel", 5))
        self.assertEqual((eng.last["drafts_accepted"], eng.last["drafts_offered"]), (1, 2))
        self.assertEqual(len(eng._ignored_bdone), 2)
        self.assertEqual(eng.slot_held[0], [1, 2])
        self.assertFalse(eng.slot_busy[0])
        self.assertEqual(eng.slot_claims | eng.slot_releasing, set())

    def test_eos_drain_skips_stale_terminals_with_a_bounded_diagnostic(self):
        eng, owner = self._resident_engine((18, 3))
        for incarnation in range(40, 80):
            eng.slot_q[0].put(f"BDONE 0 99 nonfinite 500 request=18 incarnation={incarnation}")
        terminal = "BDONE 0 2 stop 5 request=18 incarnation=3"
        eng.slot_q[0].put(terminal)
        self.assertEqual(eng._bdone_after_eos(0, owner), terminal)
        self.assertEqual(eng.last["generated"], 1, "EOS lookahead must not apply a stale terminal")
        self.assertEqual(len(eng._ignored_bdone), 32)
        self.assertEqual(eng._ignored_bdone[-1], {"slot": 0, "expected": (18, 3), "received": (18, 79)})
        self.assertTrue(eng._apply_bdone(terminal.split(), (18, 3)))
        self.assertEqual((eng.last["generated"], eng.last["finish"]), (2, "stop"))

    def test_identity_optional_terminals_keep_legacy_completion(self):
        for expected, suffix in ((None, ""), ((18, 3), ""), (None, " request=18 incarnation=3")):
            with self.subTest(expected=expected, suffix=suffix):
                eng, _ = self._resident_engine(expected)
                self.assertTrue(eng._apply_bdone(("BDONE 0 2 stop 5" + suffix).split(), expected))
                self.assertEqual((eng.last["generated"], eng.last["finish"]), (2, "stop"))
                self.assertEqual(list(eng._ignored_bdone), [])

    def test_stop_during_prompt_read(self):
        cancel = threading.Event()
        eng = self._engine(["PP 256 4096 1 100.0", "PP 512 4096 2 100.0", "PP 768 4096 3 100.0", "DONE 0"])
        gen = eng._control(cancel, lambda t: None)
        self.assertIsNone(next(gen))               # first PP line, nothing cancelled yet
        self.assertEqual(eng.sent, [])
        cancel.set()
        self.assertIsNone(next(gen))               # the next PP line: STOP goes out now
        self.assertEqual(eng.sent, ["STOP"])
        self.assertIsNone(next(gen))
        self.assertEqual(eng.sent, ["STOP"])       # once

    def test_no_stop_without_cancel(self):
        eng = self._engine(["PP 256 4096 1 100.0", "DONE 0"])
        for _ in eng._control(threading.Event(), lambda t: None):
            pass
        self.assertEqual(eng.sent, [])

    def test_admission_frame_does_not_end_control_read(self):
        eng = self._engine(["ADMIT cells=10 state_demand_bytes=64 context_limit=4096",
                            "T 90", "DONE 1", "BADM 0 0"])
        eng._ctl_mode = "batch"
        tokens = []
        list(eng._control(threading.Event(), tokens.append))
        self.assertEqual(tokens, [90])
        self.assertEqual(eng._ctl_result, ("badm", False))

    def test_refusal_without_boundary_never_reuses_control_lines(self):
        eng = self._engine(["REJECT request memory exceeds admitted envelope"])
        # GEN's REJECT is terminal; BGEN must prove its bounded BADM boundary.
        eng._ctl_mode = "batch"
        eng._silent = mock.Mock(return_value=server.EngineSilent("request memory refused"))
        tokens = []
        with self.assertRaises(server.EngineSilent):
            list(eng._control(threading.Event(), tokens.append, slot=0))
        self.assertEqual(tokens, [])
        eng._silent.assert_called_once()

    def test_active_slot_is_not_reusable_until_bdone(self):
        eng = self._engine([])
        eng.gen = 1
        eng.slot_cv = threading.Condition()
        eng.slot_busy, eng.slot_held = [True], [[]]
        eng.slot_claims = {0}
        eng.slot_releasing = set()
        eng._slot_identity = [None]
        arrived, drained = threading.Event(), threading.Event()
        class SlotQueue(queue.Queue):
            def get(self, *args, **kwargs):
                arrived.set()
                return super().get(*args, **kwargs)
        eng.slot_q = [SlotQueue()]
        eng.alive = lambda: True
        owner = (eng.gen, eng.slot_q, eng.slot_busy, eng.slot_held, eng.slot_claims, eng.slot_releasing,
                 eng._slot_identity)
        errors = []
        def drain():
            try:
                eng._release_slot_when_done(0, [1, 2], owner)
            except Exception as error:
                errors.append(error)
            finally:
                drained.set()
        thread = threading.Thread(target=drain)
        thread.start()
        try:
            self.assertTrue(arrived.wait(2))
            self.assertEqual(eng.sent, ["BSTOP 0"])
            self.assertTrue(eng.slot_busy[0])
            self.assertEqual(eng.slot_releasing, {0})
            self.assertEqual(eng.slot_claims, {0})
            self.assertFalse(drained.is_set())
            eng.slot_q[0].put("BT 0 3")
            eng.slot_q[0].put("BDONE 0 3 cancel 1")
            self.assertTrue(drained.wait(2))
            self.assertEqual(errors, [])
            self.assertFalse(eng.slot_busy[0])
            self.assertEqual(eng.slot_releasing, set())
            self.assertEqual(eng.slot_claims, set())
            self.assertEqual(eng.slot_held[0], [1, 2])
        finally:
            eng.slot_q[0].put(None)
            thread.join(2)

    def test_drain_of_dead_process_never_consumes_successor_lines(self):
        eng = self._engine(["DONE successor"])
        eng.gen = 2
        eng.alive = lambda: True
        self.assertIsNone(eng._drain_control("DONE", born=1))
        self.assertEqual(eng.lines.get_nowait(), "DONE successor")

    def test_serial_admission_is_observable_and_refusal_has_zero_tokens(self):
        import io
        from types import SimpleNamespace
        eng = server.StrataEngine("not-started", [], lazy=True)
        eng.proc = SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        eng.ended = False
        eng.lines = queue.Queue()
        eng.lines.put("ADMIT cells=10 state_demand_bytes=64 context_limit=4096")
        eng.lines.put("T 90")
        eng.lines.put("DONE 1 1 0 1 stop")
        self.assertEqual(list(eng.generate([1], 8, {}, threading.Event())), [90])
        eng.lines.put("REJECT memory envelope exceeded")
        output = []
        with self.assertRaisesRegex(ValueError, "memory envelope exceeded"):
            output.extend(eng.generate([1], 8, {}, threading.Event()))
        self.assertEqual(output, [])
        self.assertTrue(eng.alive())

    def test_checkpoint_cache_opt_out_leaves_all_sampling_controls_unchanged(self):
        default = server.StrataEngine.sampling_keys({"temperature": .5, "seed": 9})
        disabled = server.StrataEngine.sampling_keys({"temperature": .5, "seed": 9, "strata_checkpoint": False})
        self.assertEqual(disabled, default + " ckpt=0")
        self.assertNotIn("ckpt=", server.StrataEngine.sampling_keys({"strata_checkpoint": True}))


if __name__ == "__main__":
    unittest.main()
