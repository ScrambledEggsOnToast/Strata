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
        eng._parse_done = lambda line: None
        return eng

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
        eng._silent = mock.Mock(return_value=server.EngineSilent("request memory refused"))
        tokens = []
        with self.assertRaises(server.EngineSilent):
            list(eng._control(threading.Event(), tokens.append))
        self.assertEqual(tokens, [])
        eng._silent.assert_called_once()

    def test_bounded_batch_refusal_drains_its_boundary_without_consuming_next_request(self):
        eng = self._engine(["REJECT bounded=1 prompt exceeds context", "BADM 0 0", "T 91", "DONE 1"])
        eng._ctl_mode = "batch"
        eng._silent = mock.Mock(return_value=server.EngineSilent("unexpected poisoning"))
        with self.assertRaises(server.AdmissionError):
            list(eng._control(threading.Event(), lambda token: self.fail("refused request emitted a token")))
        eng._silent.assert_not_called()
        eng._ctl_mode = "solo"
        tokens = []
        list(eng._control(threading.Event(), tokens.append))
        self.assertEqual(tokens, [91])

    def test_active_slot_is_not_reusable_until_bdone(self):
        eng = self._engine([])
        eng.gen = 1
        eng.slot_cv = threading.Condition()
        eng.slot_busy, eng.slot_held = [True], [[]]
        arrived, drained = threading.Event(), threading.Event()
        class SlotQueue(queue.Queue):
            def get(self, *args, **kwargs):
                arrived.set()
                return super().get(*args, **kwargs)
        eng.slot_q = [SlotQueue()]
        eng.alive = lambda: True
        errors = []
        def drain():
            try:
                eng._release_slot_when_done(0, [1, 2])
            except Exception as error:
                errors.append(error)
            finally:
                drained.set()
        thread = threading.Thread(target=drain)
        thread.start()
        try:
            self.assertTrue(arrived.wait(2))
            self.assertTrue(eng.slot_busy[0])
            self.assertFalse(drained.is_set())
            eng.slot_q[0].put("BT 0 3")
            eng.slot_q[0].put("BDONE 0 3 cancel 1")
            self.assertTrue(drained.wait(2))
            self.assertEqual(errors, [])
            self.assertFalse(eng.slot_busy[0])
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
