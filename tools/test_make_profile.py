"""Tests for tools/make_profile.py, without a model: synthetic profiles and routing traces.

    python -m unittest tools.test_make_profile
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as MP  # noqa: E402
import bench_eviction as BE  # noqa: E402

NE = 4   # a tiny "experts per layer"; the layer count is fixed at MP.N_LAYER


def write_trace(path, records):
    """One record per (layer, [expert ids]); the reader skips k weights after the k ids."""
    with open(path, "wb") as f:
        for layer, experts in records:
            f.write(struct.pack("<ii", layer, len(experts)))
            f.write(struct.pack("<%di" % len(experts), *experts))
            f.write(struct.pack("<%df" % len(experts), *([1.0] * len(experts))))


def full_base():
    """Every pair, in an order that differs from rank_profile's fill (e-major then layer)."""
    return [(layer, e) for layer in range(MP.N_LAYER) for e in range(NE)]


class TraceStructure(unittest.TestCase):
    def test_both_consumers_refuse_every_partial_terminal_record(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "routing.bin"
            write_trace(path, [(0, [1, 2]), (1, [2, 3])])
            complete = path.read_bytes()
            first_record_bytes = 24
            # A complete prefix is not a substitute for the damaged final record.
            for end in range(first_record_bytes + 1, len(complete)):
                path.write_bytes(complete[:end])
                for reader in (MP.read_trace, BE.read_trace):
                    with self.subTest(end=end, reader=reader.__module__):
                        with self.assertRaisesRegex(ValueError, "truncated routing"):
                            reader(path, n_expert=NE)

    def test_invalid_counts_and_weights_are_refused(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "routing.bin"
            for k in (-1, 0, 4097):
                path.write_bytes(struct.pack("<ii", 0, k))
                for reader in (MP.read_trace, BE.read_trace):
                    with self.subTest(k=k, reader=reader.__module__):
                        with self.assertRaisesRegex(ValueError, "implausible expert count"):
                            reader(path, n_expert=NE)
            for weight in (float("nan"), float("inf"), -0.5):
                path.write_bytes(struct.pack("<iiif", 0, 1, 2, weight))
                for reader in (MP.read_trace, BE.read_trace):
                    with self.subTest(weight=weight, reader=reader.__module__):
                        with self.assertRaisesRegex(ValueError, "invalid routing weights"):
                            reader(path, n_expert=NE)

    def test_zero_weight_placeholders_and_frequency_order_remain_valid(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "routing.bin"
            path.write_bytes(struct.pack("<iiif", 0, 1, 2, 0.0) * 2)
            self.assertEqual(MP.read_trace(path, NE), {(0, 2): 2})
            self.assertEqual(list(BE.read_trace(path, NE)), [BE.encode(0, 2)] * 2)


class RankProfile(unittest.TestCase):
    def test_a_complete_base_hides_the_trace(self):
        # the shipped profile's situation: it already ranks every pair, so the trace cannot move anything
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked, base)
        self.assertEqual(counts["trace"], 0)
        self.assertEqual(counts["base"], MP.N_LAYER * NE)

    def test_reorder_puts_the_traces_first(self):
        base = full_base()
        freq = defaultdict(int, {(5, 1): 10, (6, 2): 7})
        ranked, counts = MP.rank_profile(base, freq, NE, reorder=True)
        self.assertEqual(ranked[:2], [(5, 1), (6, 2)])
        self.assertEqual(set(ranked), set(base))                 # still every pair, once
        self.assertEqual(len(ranked), len(base))
        self.assertEqual(counts["trace"], 2)
        self.assertEqual(counts["base"], MP.N_LAYER * NE - 2)    # the base fills the rest, minus the two

    def test_frequency_and_tie_break_order_the_trace(self):
        freq = defaultdict(int, {(1, 0): 1, (2, 0): 3, (3, 0): 3})
        ranked, _ = MP.rank_profile([], freq, NE, no_base=True)  # (2,0)/(3,0) tie -> the smaller pair first
        self.assertEqual(ranked[:3], [(2, 0), (3, 0), (1, 0)])

    def test_default_appends_only(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(0, 0): 5, (1, 3): 9})
        ranked, _ = MP.rank_profile(base, freq, NE)
        self.assertEqual(ranked[:3], [(0, 0), (0, 1), (1, 3)])   # base first; (0,0) already there, not moved

    def test_no_base_uses_the_traces_then_the_fill(self):
        base = [(0, 0), (0, 1)]
        freq = defaultdict(int, {(1, 3): 9, (0, 0): 5})
        ranked, counts = MP.rank_profile(base, freq, NE, no_base=True)
        self.assertEqual(ranked[:2], [(1, 3), (0, 0)])
        self.assertEqual(counts["base"], 0)
        self.assertEqual(len(ranked), MP.N_LAYER * NE)

    def test_reorder_without_traces_is_the_default(self):
        base = full_base()
        self.assertEqual(MP.rank_profile(base, defaultdict(int), NE, reorder=True)[0], base)


class Main(unittest.TestCase):
    def run_main(self, argv):
        old, sys.argv = sys.argv, ["make_profile.py"] + argv
        try:
            MP.main()
        finally:
            sys.argv = old

    def test_malformed_trace_never_rewrites_the_output_profile(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trace, out = root / "routing.bin", root / "profile.bin"
            write_trace(trace, [(0, [1]), (1, [2])])
            # Both IDs exist, but the terminal weight is missing.
            trace.write_bytes(trace.read_bytes()[:-4])
            out.write_bytes(b"existing profile must survive")
            with self.assertRaisesRegex(ValueError, "truncated routing body"):
                self.run_main([str(trace), "--no-base", "--out", str(out)])
            self.assertEqual(out.read_bytes(), b"existing profile must survive")

    def test_end_to_end_reorder_over_a_complete_base(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "base.bin"
            out = Path(d) / "out.bin"
            trace = Path(d) / "t.bin"
            MP.write_profile(str(base), full_base(), NE)
            write_trace(str(trace), [(5, [1]), (6, [2]), (6, [2]), (6, [2])])
            self.run_main([str(trace), "--base", str(base), "--reorder", "--n-expert", str(NE), "--out", str(out)])
            ranked = MP.read_profile(str(out), NE)
            self.assertEqual(ranked[:2], [(6, 2), (5, 1)])       # most frequent first
            self.assertEqual(set(ranked), set(full_base()))
            # and without --reorder it is the base's order, untouched
            self.run_main([str(trace), "--base", str(base), "--n-expert", str(NE), "--out", str(out)])
            self.assertEqual(MP.read_profile(str(out), NE), full_base())


if __name__ == "__main__":
    unittest.main()
