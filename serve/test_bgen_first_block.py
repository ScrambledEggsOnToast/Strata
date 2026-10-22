"""Actual frontend BGEN state machine with a deterministic CPU wire peer; no model/GPU."""
import os
import queue
import threading
import unittest
from unittest import mock
from serve import server, test_parallel


def block_peer():
    text = test_parallel.FAKE_BATCH
    text = text.replace('print(f"READY {ctx} stop", flush=True)',
                        'print("INFO bgen_first_block=1 bgen_first_block_version=1", flush=True)\nprint(f"READY {ctx} stop", flush=True)')
    text = text.replace('n = 1 if slot is not None else max_new',
                        'block = slot is not None and "first_block=1" in f\n        n = min(3, max_new) if block else (1 if slot is not None else max_new)')
    # Committed first-block output is already determined; STOP observed after emission.
    text = text.replace('            if stop.is_set():\n                break\n            print(f"T {t}"',
                        '            if stop.is_set() and not block:\n                break\n            print(f"T {t}"')
    text = text.replace('fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")',
                        'fin = "stop" if out and toks[out - 1] == 257 else ("admit" if block and out < max_new else ("cancel" if stop.is_set() else "length"))')
    text = text.replace('cont = fin == "length" and max_new > 1',
                        'cont = fin in ("length", "admit") and max_new > out')
    text = text.replace('active[slot] = [toks[1:], max_new, 1]', 'active[slot] = [toks[out:], max_new, out]')
    return text


class BlockAdmission(unittest.TestCase):
    start = test_parallel.ParallelService.start
    tearDown = test_parallel.ParallelService.tearDown

    def setUp(self):
        patch = mock.patch.object(test_parallel, 'FAKE_BATCH', block_peer())
        patch.start(); self.addCleanup(patch.stop)
        self.addCleanup(self.tearDown)
        with mock.patch.object(test_parallel, "serve", return_value=mock.Mock(server_address=("127.0.0.1", 0))), mock.patch.dict(os.environ, {'STRATA_DFLASH2_BGEN_FIRST_BLOCK': '1'}):
            self.start(2, more=('--legacy-identity', '--draft-counters', '2,3'))
        # Real BGEN path without timing races: another request owns the peer seat.
        self.engine.slot_busy[1] = True
        self.engine.slot_claims.add(1)

    def stream(self, prompt=b'hello?', cap=6):
        return self.engine.generate_batched(list(prompt), cap, {}, threading.Event())

    def test_multiple_admission_tokens_counts_timing_and_last_anchor(self):
        out = list(self.stream())
        tokens = [x for x in out if x is not None]
        self.assertEqual(tokens, list(b'ok, do'))
        self.assertEqual(self.engine.slot_held[0], list(b'hello?') + tokens[:-1])
        self.assertEqual(self.engine.last['generated'], 6)
        self.assertEqual(self.engine.last['drafts_accepted'], 2)
        self.assertEqual(self.engine.last['drafts_offered'], 3)
        self.assertEqual(self.engine.last['decode_ms'], 61.0)
        self.assertEqual(self.engine.last['legs'][0]['admission']['generated'], 3)
        self.assertTrue(self.engine.slot_busy[1])
        self.assertIn(1, self.engine.slot_claims)

    def test_close_after_middle_admission_token_collects_hidden_tail(self):
        out = self.stream()
        self.assertEqual(next(out), ord('o'))
        self.assertEqual(next(out), ord('k'))
        out.close()  # third admission T was committed but not consumed by the caller
        committed = self.engine.last['generated']
        self.assertGreaterEqual(committed, 3)  # resident BT may win the race with BSTOP
        self.assertLessEqual(committed, 6)
        self.assertEqual(self.engine.last['decode_ms'], 61.0)
        self.assertEqual(self.engine.slot_held[0], list(b'hello?') + list(b'ok, do')[:committed - 1])
        self.assertFalse(self.engine.slot_busy[0])
        self.assertNotIn(0, self.engine.slot_claims)
        self.assertTrue(self.engine.slot_busy[1])

    def test_real_cap_ends_without_resident_and_eos_ends_without_resident(self):
        for prompt, cap, expected, finish in [(b'hello?', 3, list(b'ok,'), 'length'),
                                            (b'hello ok, done', 8, [ord('.'), 257], 'stop')]:
            with self.subTest(finish=finish):
                output = [x for x in self.stream(prompt, cap) if x is not None]
                self.assertEqual(output, expected)
                self.assertEqual(self.engine.last['generated'], len(expected))
                self.assertEqual(self.engine.last['finish'], finish)
                self.assertEqual(self.engine.slot_held[0], [])
                self.assertFalse(self.engine.slot_busy[0])

    def test_missing_capability_refuses_before_submission_and_client_key_is_not_forwarded(self):
        self.assertNotIn('first_block', self.engine.sampling_keys({'first_block': 1}))
        self.assertEqual(self.engine._bgen_block_key('image-path'), '')
        self.engine.info.pop('bgen_first_block')
        with self.assertRaises(server.EngineDied):
            self.engine._bgen_block_key(None)


class BlockSeams(unittest.TestCase):
    def test_admission_count_and_owner_refuse_malformed_boundary(self):
        e = server.StrataEngine('not-started', [], lazy=True)
        e.batch = 1; e._slot_identity = [(4, 2)]
        e._silent = lambda message: server.EngineDied(message)
        e.last = {'generated': 3, 'finish': 'admit'}
        e._check_block_admission(0, 3, True)
        with self.assertRaises(server.EngineDied): e._check_block_admission(0, 2, True)
        e._slot_identity[0] = None
        with self.assertRaises(server.EngineDied): e._check_block_admission(0, 3, True)
        e.bgen_first_block = True
        with self.assertRaises(server.EngineDied): e._record_badm(['BADM', '1', '1'], 0)

    def test_policy_uses_control_slot_not_token_id_and_clears_terminal_only_for_owner(self):
        p = server.SpeculationPolicy(1.0, 2)
        p.begin('batch', 100, 0, now=1)
        for i in range(3): p.wire('T 999', now=1.1 + i / 10)
        p.wire('BADM 0 1', now=1.5)
        self.assertEqual(p.slots, {0: 102})
        p.begin('batch', 200, 1, now=2)
        p.wire('T 888', now=2.1)
        p.wire('BDONE 0 3 stop 1', now=2.2)
        self.assertEqual(p.control[1], 1)
        p.wire('BADM 1 0', now=2.3)
        self.assertIsNone(p.control)
        self.assertEqual(p.admission_tokens, 0)

    def test_legacy_bdone_duration_unchanged(self):
        e = server.StrataEngine('not-started', [], lazy=True)
        e.last = {'generated': 1, 'decode_ms': 20.0, 'finish': 'length'}
        self.assertTrue(e._apply_bdone('BDONE 0 4 length 7'.split(), None))
        self.assertEqual(e.last['decode_ms'], 7.0)
