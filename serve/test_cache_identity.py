"""serve/test_cache_identity.py - #42 cache identity at the server seams: the per-principal namespace, the
tokenizer/template frontend digest, and tenant-gated slot reuse (no GPU, no pack).

    python -m unittest serve.test_cache_identity -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class RecordingEngine(MockEngine):
    """Remembers the sampling dict of every request (the server's cache-identity choice is in it)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_calls: list[dict] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling_calls.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def identities(engine) -> list[tuple[str, str] | None]:
    return [s.get("_strata_cache") for s in engine.sampling_calls]


class PickSlotTenancy(unittest.TestCase):
    """#42: a slot's held prefix is offered only to its own principal; the engine re-checks it natively."""

    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_tenant = [None] * n
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        return e

    def test_a_held_prefix_is_its_owners_only(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], []]
        e.slot_tenant = ["aaa", None, None]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], "aaa"), 0)      # the owner continues its conversation
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], "bbb"), 1)      # a canary tenant: no prefix, an empty slot
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], None), 1)       # unidentified: nothing held is offered
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([1, 2, 3, 4, 5], "aaa"))        # every slot busy: none is offered
        e.slot_busy = [False, False, False]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], None), 1)       # unidentified: nothing held is offered

    def test_stale_tags_only_block_matching(self):
        e = self.engine(2)
        e.slot_held = [[1, 2, 3], []]
        e.slot_tenant = ["old", None]
        self.assertEqual(e.pick_slot([7, 8], "new"), 1)               # no match: the truly empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8], "new"), 0)               # taking the slot over is allowed...


class FrontendDigest(unittest.TestCase):
    """The digest hashes the tokenizer's and template's actual contents - the same directory with other
    contents digests differently - so a changed frontend never selects the old one's entries."""

    def svc(self, template_source="<|im_start|>x"):
        tok = ByteTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.jinja"
            path.write_text(template_source)
            return Service(MockEngine(tok, "ok"), tok, ChatTemplate(path))

    def digest(self, svc, files, effort_end=False):
        svc.effort_end = effort_end
        with tempfile.TemporaryDirectory() as d:
            for name, body in files.items():
                (Path(d) / name).write_text(body)
            return svc.compute_frontend_digest(Path(d))

    def test_contents_decide_not_paths(self):
        svc = self.svc()
        base = {"vocab.json": '{"a": 0}', "merges.txt": "a b\n", "token_type.json": "[0]"}
        first = self.digest(svc, base)
        self.assertEqual(first, self.digest(svc, dict(base)), "the same contents digest the same")
        self.assertNotEqual(first, self.digest(svc, {**base, "vocab.json": '{"a": 0, "b": 1}'}),
                            "another vocab is another frontend")
        self.assertNotEqual(first, self.digest(svc, {**base, "merges.txt": "a b\nb a\n"}),
                            "other merges are another frontend")
        self.assertNotEqual(first, self.digest(svc, {**base, "token_type.json": "[0, 1]"}),
                            "other token types are another frontend")
        self.assertNotEqual(first, self.digest(svc, base, effort_end=True),
                            "the effort turn is part of the frontend identity")
        other = self.svc("<|im_start|>y")
        self.assertNotEqual(first, self.digest(other, base), "another template is another frontend")

    def test_a_missing_tokenizer_file_is_stable(self):
        svc = self.svc()
        with tempfile.TemporaryDirectory() as d:
            svc.effort_end = False
            a = svc.compute_frontend_digest(Path(d))
            svc.effort_end = False
            self.assertEqual(a, svc.compute_frontend_digest(Path(d)), "absent files hash as absent, twice")


class PrincipalNamespace(unittest.TestCase):
    """The namespace is the authenticated principal's, decided by the auth gate; a canary principal (here:
    a rotated key) never shares it, and a failed request leaves nothing behind."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "ok")
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.compute_frontend_digest(Path(tempfile.mkdtemp()))   # no tokenizer files: a stable digest
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def chat(self, key=None):
        headers = {"Content-Type": "application/json"}
        if key is not None:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_two_authorizations_do_not_share_a_namespace(self):
        self.svc.api_key = "key-a"
        before = len(self.engine.sampling_calls)
        self.assertEqual(self.chat("key-a")[0], 200)
        self.svc.api_key = "key-b"                      # the canary principal (a rotated deployment)
        self.assertEqual(self.chat("key-b")[0], 200)
        ns = [x for x in identities(self.engine)[before:] if x]
        self.assertEqual(len(ns), 2)
        a, b = ns
        self.assertEqual(a[1], b[1], "one frontend: the tokenizer/template digest")
        self.assertNotEqual(a[0], b[0], "the canary principal never shares the first principal's namespace")
        for name, _ in ns:
            self.assertEqual(len(name), 32)
            self.assertNotIn("key-", name, "the namespace is a digest, never the credential")

    def test_no_identity_without_a_principal(self):
        self.svc.api_key = "secret"
        before = len(self.engine.sampling_calls)
        self.assertEqual(self.chat("wrong")[0], 401)    # denied: the engine is not called at all
        self.assertEqual(len(self.engine.sampling_calls), before)
        self.assertEqual(self.chat()[0], 401)           # no key at all: the same
        self.svc.api_key = ""
        before = len(self.engine.sampling_calls)
        self.assertEqual(self.chat()[0], 200)           # no key required: the one anonymous namespace
        anon = identities(self.engine)[before]
        self.assertIsNotNone(anon)
        self.assertEqual(self.chat()[0], 200)
        self.assertEqual(anon, identities(self.engine)[before + 1], "the anonymous namespace is stable")



if __name__ == "__main__":
    unittest.main()
