"""The tokenizer's piece cache (0.1.41): the ids are the ones the merge loop gives, cache cold, warm or switched off.

    python -m unittest tools.test_strata_tokenizer
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def toy() -> ST.Tokenizer:
    base = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = ["t h", "th e", "i n", "a n", "an d", "Ġ t", "Ġt h", "Ġth e", "e r", "o n"]
    merges = [m.replace("Ġ", ST.BYTE_TO_UNICODE[32]) for m in merges]
    tokens = list(base)
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b)
    return ST.Tokenizer(tokens, merges)


class PieceCache(unittest.TestCase):
    def test_same_ids_cold_warm_and_off(self):
        rng = random.Random(7)
        words = ["the", "and", "in", "on", "there", "other", "x" * 70, "你好", "def", "return", "\n\n", "  "]
        text = "".join(rng.choice(words) + rng.choice([" ", "", "\n", "\t"]) for _ in range(3000))
        a = toy()
        a.PIECE_CACHE_MAX = 0
        want = a.encode(text)
        b = toy()
        cold = b.encode(text)
        warm = b.encode(text)
        self.assertEqual(cold, want)
        self.assertEqual(warm, want)
        self.assertGreater(len(b._piece_ids), 0)
        self.assertEqual(b.decode(warm), text)

    def test_the_cache_is_bounded(self):
        t = toy()
        t.PIECE_CACHE_MAX = 5
        t.encode(" ".join("w%d" % i for i in range(100)))
        self.assertLessEqual(len(t._piece_ids), 5)


class ByteCache(unittest.TestCase):
    def test_decode_before_encode_cold_and_warm(self):
        t = toy()
        text = "café 日本語 🙂\r\n\x00"
        ids = list(text.encode("utf-8"))  # toy's first 256 token IDs are the raw bytes
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode([]), "")
        for i in range(256):
            self.assertEqual(t.token_bytes(i), bytes([i]))

    def test_invalid_ids_before_and_after_cache_use(self):
        t = toy()
        for _ in range(2):
            for i in (-1, len(t.tokens), len(t.tokens) + 100):
                with self.assertRaises(IndexError):
                    t.token_bytes(i)
            self.assertEqual(t.token_bytes(65), b"A")

    def test_instances_with_different_vocabularies_stay_independent(self):
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        a, b = ST.Tokenizer(tokens, []), ST.Tokenizer(tokens[::-1], [])
        text = "the café 日本語 🙂\n"
        for t in (a, b, a, b):
            expected = [t.ids[ST.BYTE_TO_UNICODE[byte]] for byte in text.encode("utf-8")]
            self.assertEqual(t.encode(text), expected)
            self.assertEqual(t.decode(expected, errors="strict"), text)
        self.assertEqual(a.token_bytes(0), b"\x00")
        self.assertEqual(b.token_bytes(0), b"\xff")

    def test_shared_tokenizer_concurrent_first_decode(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        t = toy()
        texts = ["café", "日本語", "🙂\r\n", "\x00plain"]
        barrier = threading.Barrier(len(texts))
        def decode(text):
            ids = list(text.encode("utf-8"))
            barrier.wait(timeout=5)
            for _ in range(100):
                if t.decode(ids, errors="strict") != text:
                    raise AssertionError("concurrent decode changed the text")
            return text
        with ThreadPoolExecutor(len(texts)) as pool:
            self.assertEqual(list(pool.map(decode, texts)), texts)


class BoundedEncode(unittest.TestCase):
    """`encode_within`: the preparation-only encode.  It returns `encode`'s ids or nothing - never a prefix - and it
    declines a single pre-tokenizer piece too big to merge inside the preparation allowance before merging it."""

    TEXT = "the cat sat on the mat, and 日本語 the cat sat again\r\n\r\n<think>quoted</think>"

    def test_the_ids_are_the_ones_encode_gives(self):
        for parse_special in (False, True):
            for plain in ((), [(0, 4)]):
                with self.subTest(parse_special=parse_special, plain=plain):
                    t = toy()
                    want = t.encode(self.TEXT, parse_special=parse_special, plain=plain)
                    self.assertEqual(t.encode_within(self.TEXT, len(want), parse_special=parse_special, plain=plain),
                                     want)

    def test_a_bound_that_does_not_fit_produces_nothing_at_all(self):
        t = toy()
        want = t.encode(self.TEXT, parse_special=True)
        self.assertGreater(len(want), 8)
        for limit in range(0, len(want)):
            with self.subTest(limit=limit):
                got = t.encode_within(self.TEXT, limit, parse_special=True)
                self.assertIsNone(got, "a bounded encode returned the first tokens of a prompt it could not fit")
        self.assertEqual(t.encode_within(self.TEXT, len(want) - 1, parse_special=True), None)
        self.assertEqual(t.encode_within(self.TEXT, len(want) + 100, parse_special=True), want)

    def test_a_piece_over_the_cap_is_never_merged(self):
        t = toy()
        t.PREPARATION_PIECE_MAX_BYTES = 4
        self.assertEqual(t.encode_within("the cat sat", 100), t.encode("the cat sat"))   # pieces of four or less
        long_word = "wonderful"                                                          # nine characters: over the cap
        with mock.patch.object(t, "_bpe", wraps=t._bpe) as bpe:
            self.assertIsNone(t.encode_within(f"{long_word} the cat", 100))
        self.assertEqual(bpe.call_args_list, [], "a bounded encode merged a piece it had already refused")
        self.assertIsNotNone(t.encode(f"the {long_word} cat"))        # the unbounded path is untouched

    def test_the_cache_never_holds_more_than_its_share(self):
        t = toy()
        t.CACHE_SHARE_BYTES = 3 * (t.CACHED_TOKEN_BYTES + t.CACHED_PIECE_BYTES)
        for i in range(50):
            t.encode(" ".join("w%d" % (i * 10 + k) for k in range(10)))
        self.assertLessEqual(t.retained_bytes(), t.CACHE_SHARE_BYTES,
                             "the piece cache kept more than the allowance pays it")
        self.assertGreater(len(t._piece_ids), 0)
        payload = sum(sys.getsizeof(key) + sys.getsizeof(value) for key, value in t._piece_ids.items())
        self.assertGreaterEqual(t.retained_bytes(), payload, "cache accounting omits retained string or list storage")
        off = toy()
        off.CACHE_SHARE_BYTES = 0
        off.encode("the cat sat on the mat")
        self.assertEqual(off.retained_bytes(), 0)



if __name__ == "__main__":
    unittest.main()
