"""The frontend's half of the private conversation-offer channel: stable scopes, the queued frontier's preparation
and the transport it is offered on.

    python -m unittest serve.test_conversation_offers -v

Nothing here needs a GPU or the native engine: the native side of the channel is played by this test's own socket
peer (a HELLO, then whatever the frontend sends), and the engine by a scripted stand-in that can hold its first
requests open so a real queue forms.
"""
import array
import fcntl
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import select
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import conversation_offers                                              # noqa: E402
from serve import server                                                           # noqa: E402
from serve.conversation_offers import HEADER, Hello, OfferProtocolError, hello_payload  # noqa: E402
from serve.frontend import ChatTemplate                                            # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "serve/chat_template.jinja"
KEY = bytes(range(32))            # the issuer key this test's native side hands out


def prompt_text(tokens) -> str:
    """The text a byte-tokenized prompt was made of (a special token is one placeholder byte): a test can then name
    the request an offered payload belongs to without re-rendering it."""
    out = bytearray()
    for t in tokens:
        t = int(t)
        out.extend(t.to_bytes(1, "little") if 0 <= t < 256 else b"\x00")
    return out.decode("utf-8", "replace")


class OtherTokenizer(ByteTokenizer):
    """The byte tokenizer under another name: a different tokenizer identity (Service.render_identity)."""


class NativePeer:
    """The native side of one link, as these tests speak it: it answers the HELLO and reads what is offered."""

    def __init__(self, link, *, generation=41, frontier=1, metadata_bytes=1 << 20, enabled=True, key=KEY):
        assert link.child_socket is not None
        self.link = link
        self.sock: socket.socket = link.child_socket
        self.hello = Hello(generation=generation, frontier=frontier, metadata_bytes=metadata_bytes, enabled=enabled,
                           issuer_key=key)
        self.offers = []                  # one dict per OFFER: ids, deadline, tokens, seals, the descriptor's mode
        self.cancels = []                 # (request_id, generation) per CANCEL
        self.other = []                   # anything else (a malformed message would land here)

    def start(self) -> None:
        """Answer with the HELLO (what the engine does when its channel starts)."""
        self.sock.send(hello_payload(self.hello))
        self.link.wait_hello(5.0)

    def next_offer(self, timeout=5.0):
        """The next OFFER, or None; messages of another kind are recorded and skipped."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.read_one(deadline - time.monotonic()) == "offer":
                return self.offers[-1]
        return None

    def next_cancel(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.read_one(deadline - time.monotonic()) == "cancel":
                return self.cancels[-1]
        return None

    def drain(self, timeout=0.5) -> None:
        """Read whatever is already there (used to prove that nothing else arrives)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.read_one(deadline - time.monotonic()) is None:
                return

    def read_one(self, timeout):
        """One datagram, if one arrives in `timeout` seconds; -> "offer" / "cancel" / "other" / None."""
        if timeout <= 0 or not select.select([self.sock], [], [], timeout)[0]:
            return None
        payload, ancillary, _flags, _address = self.sock.recvmsg(HEADER.size, socket.CMSG_SPACE(8 * 4))
        if not payload:
            return None
        magic, version, kind, request_id, generation, deadline_ns, count, flags, scope = HEADER.unpack(payload)
        if (magic, version) != (conversation_offers.MAGIC, conversation_offers.VERSION):
            raise AssertionError(f"unexpected message header {magic!r} v{version}")
        fds = []
        for level, sort, data in ancillary:
            if level == socket.SOL_SOCKET and sort == socket.SCM_RIGHTS:
                fds.extend(array.array("i", data[:len(data) - len(data) % 4]).tolist())
        if kind == conversation_offers.TYPE_OFFER:
            if len(fds) != 1:
                raise AssertionError(f"an offer carries exactly one descriptor, got {len(fds)}")
            fd = fds[0]
            raw = os.read(fd, 4 * count)
            self.offers.append({"request_id": request_id, "generation": generation, "deadline_ns": deadline_ns,
                                "tokens": list(struct.unpack(f"<{count}i", raw)), "scope": scope,
                                "want_cvec": bool(flags & conversation_offers.FLAG_WANT_CVEC),
                                "seals": fcntl.fcntl(fd, conversation_offers.F_GET_SEALS),
                                "writable": fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY})
            os.close(fd)
            return "offer"
        for fd in fds:
            os.close(fd)
        if kind == conversation_offers.TYPE_CANCEL:
            self.cancels.append((request_id, generation))
            return "cancel"
        self.other.append((kind, request_id))
        return "other"


class PlainEngine(MockEngine):
    """A scripted engine carrying this test's offer channel (MockEngine itself has none)."""

    def __init__(self, tok, link=None, script="ok", max_context=4096, batch=0):
        super().__init__(tok, script, max_context=max_context)
        self.offers = link
        self.batch = batch


class QueuedEngine(PlainEngine):
    """A scripted engine that holds its first `hold` requests open until the test releases them (so the queue really
    queues, behind a full engine) and keeps the ids and sampling of every request it ran."""

    def __init__(self, tok, link=None, script="ok", max_context=4096, batch=0, hold=1):
        super().__init__(tok, link=link, script=script, max_context=max_context, batch=batch)
        self.hold = hold
        self.entered = threading.Event()
        self.release = threading.Event()
        self.prompts: list[list[int]] = []
        self.samplings: list[dict] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.samplings.append(dict(sampling))
        if len(self.prompts) <= self.hold:
            if len(self.prompts) == self.hold:
                self.entered.set()
            self.release.wait(timeout=10)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ChannelTestCase(unittest.TestCase):
    """A service with a private channel whose native side this test speaks, and an HTTP server in front of it."""

    frontier = 1
    generation = 41
    metadata_bytes = 1 << 20
    enabled = True
    key = KEY
    batch = 1
    hold = 1

    def setUp(self):
        self.tok = ByteTokenizer()
        self.link = conversation_offers.ConversationOfferLink()
        self.addCleanup(self.link.close)
        self.peer = NativePeer(self.link, generation=self.generation, frontier=self.frontier,
                               metadata_bytes=self.metadata_bytes, enabled=self.enabled, key=self.key)
        self.link.start()
        self.peer.start()
        self.engine = self.engine_for()
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def engine_for(self):
        """The scripted engine this case's service runs on (one that holds its first request open by default)."""
        return QueuedEngine(self.tok, link=self.link, batch=self.batch, hold=self.hold)

    def start_request(self, text="hello", body=None, path="/v1/chat/completions", headers=None):
        """Send one request on its own thread; -> (thread, outcome dict).  A queued request never returns here."""
        outcome: dict = {}

        def run():
            conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=20)
            try:
                payload = body if body is not None else {"messages": [{"role": "user", "content": text}]}
                conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json", **(headers or {})})
                response = conn.getresponse()
                outcome["status"] = response.status
                outcome["body"] = response.read()
            except Exception as e:                      # noqa: BLE001 - reported as the outcome, not swallowed
                outcome["error"] = e
            finally:
                conn.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, outcome

    def join(self, *sends, timeout=20):
        for thread, _outcome in sends:
            thread.join(timeout)

    def wait_queued(self, count, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.svc.status["queued"] >= count:
                return
            time.sleep(0.01)
        self.fail(f"never saw {count} queued requests (queued={self.svc.status['queued']})")

    def wait_prompts(self, count, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.engine.prompts) >= count:
                return
            time.sleep(0.01)
        self.fail(f"the engine never ran {count} requests (ran {len(self.engine.prompts)})")

    def keys_of(self, sampling: dict) -> str:
        return StrataEngine.sampling_keys(sampling)


class HelloWire(unittest.TestCase):
    def test_a_hello_is_read_strictly(self):
        hello = Hello(generation=3, frontier=2, metadata_bytes=1 << 20, enabled=True, issuer_key=KEY)
        self.assertEqual(conversation_offers.parse_hello(hello_payload(hello)), hello)
        self.assertTrue(hello.store_enabled)
        disabled = Hello(generation=3, frontier=1, metadata_bytes=0, enabled=False, issuer_key=b"\0" * 32)
        self.assertFalse(disabled.store_enabled)
        self.assertEqual(conversation_offers.parse_hello(hello_payload(disabled)), disabled)

    def test_a_malformed_hello_is_refused(self):
        good = hello_payload(Hello(generation=3, frontier=1, metadata_bytes=10, enabled=True, issuer_key=KEY))
        cases = {"short": good[:-1], "magic": b"XCC1" + good[4:],
                 "version": good[:4] + struct.pack("<H", 2) + good[6:],
                 "type": good[:6] + struct.pack("<H", conversation_offers.TYPE_OFFER) + good[8:],
                 "request id": good[:8] + struct.pack("<Q", 5) + good[16:],
                 "epoch": good[:16] + struct.pack("<Q", 0) + good[24:],
                 "frontier": good[:32] + struct.pack("<I", 3) + good[36:],
                 "flags": good[:36] + struct.pack("<I", 1 << 5) + good[40:]}
        for name, payload in cases.items():
            with self.subTest(name):
                with self.assertRaises(OfferProtocolError):
                    conversation_offers.parse_hello(payload)


class StableScope(unittest.TestCase):
    """With a persistent store the namespace survives a restart; without one it stays what it always was."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tokenizer_dir = Path(directory.name)
        (self.tokenizer_dir / "vocab.json").write_text('{"a":0}')
        (self.tokenizer_dir / "merges.txt").write_text("#version: 0.2\n")
        (self.tokenizer_dir / "token_type.json").write_text("[1]")

    def service(self, *, key: bytes | None = KEY, tok=None, template=None, model="m", tokenizer_dir=None,
                model_id="ab" * 32):
        link = None
        if key is not None:
            link = conversation_offers.ConversationOfferLink()
            self.addCleanup(link.close)
            peer = NativePeer(link, key=key)
            link.start()
            peer.start()
        engine = PlainEngine(tok or ByteTokenizer(), link=link)
        engine.cache_model_id = model_id
        return Service(engine, engine.tok, template or ChatTemplate(TEMPLATE), model_name=model,
                       tokenizer_dir=self.tokenizer_dir if tokenizer_dir is None else tokenizer_dir)

    def other_template(self):
        """The same template with different source (a re-exported template is a different namespace)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_template.jinja"
            path.write_text(TEMPLATE.read_text(encoding="utf-8") + "\n{# other #}\n", encoding="utf-8")
            return ChatTemplate(path)

    def test_the_same_store_keeps_the_same_scope_across_a_restart(self):
        first, second = self.service(), self.service()   # a fresh process: a fresh secret, the same issuer key
        self.assertNotEqual(first.cache_secret, second.cache_secret)
        request = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(first.cache_scope(request, "tenant-a"), second.cache_scope(request, "tenant-a"))
        self.assertIsNotNone(first.scope_key())

    def test_without_a_store_the_scope_stays_process_local(self):
        first, second = self.service(key=None), self.service(key=None)
        request = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertIsNone(first.scope_key())
        self.assertNotEqual(first.cache_scope(request, "tenant-a"), second.cache_scope(request, "tenant-a"))
        self.assertEqual(first.cache_scope(request, "tenant-a"), first.cache_scope(request, "tenant-a"))

    def test_a_disabled_store_hands_out_no_key(self):
        service = self.service(key=b"\0" * 32)
        self.assertFalse(service.offers_ready())
        self.assertIsNone(service.scope_key())

    def test_rotation_and_identity_changes_isolate_records(self):
        request = {"messages": [{"role": "user", "content": "hi"}]}
        baseline = self.service().cache_scope(request, "tenant-a")
        self.assertNotEqual(baseline, self.service(key=bytes(reversed(KEY))).cache_scope(request, "tenant-a"))
        self.assertNotEqual(baseline, self.service().cache_scope(request, "tenant-b"))       # another principal
        self.assertNotEqual(baseline, self.service().cache_scope(request, "tenant-a", credential="other-key"))
        self.assertNotEqual(baseline, self.service().cache_scope(request, "tenant-a", gateway=True))  # a tagged kind
        self.assertEqual(baseline, self.service(model="renamed-public-alias").cache_scope(request, "tenant-a"))
        self.assertNotEqual(baseline, self.service(model_id="cd" * 32).cache_scope(request, "tenant-a"))
        self.assertNotEqual(baseline, self.service(template=self.other_template()).cache_scope(request, "tenant-a"))
        self.assertNotEqual(baseline, self.service(tok=OtherTokenizer()).cache_scope(request, "tenant-a"))

    def test_the_tokenizer_artifacts_are_part_of_the_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text('{"a": 0}', encoding="utf-8")
            (root / "merges.txt").write_text("#version: 0.2\n")
            (root / "token_type.json").write_text("[1]")
            first = self.service(tokenizer_dir=root).cache_scope({"messages": []}, "tenant-a")
            (root / "vocab.json").write_text('{"a": 0, "b": 1}', encoding="utf-8")
            second = self.service(tokenizer_dir=root).cache_scope({"messages": []}, "tenant-a")
        self.assertNotEqual(first, second)

    def test_missing_identity_refuses_cross_restart_reuse(self):
        request = {"messages": []}
        first, second = self.service(model_id=""), self.service(model_id="")
        self.assertNotEqual(first.cache_scope(request, "t"), second.cache_scope(request, "t"))
        # ... while the process that has no verified identity keeps serving its own parked conversations
        self.assertEqual(first.cache_scope(request, "t"), first.cache_scope(request, "t"))
        (self.tokenizer_dir / "merges.txt").unlink()
        first, second = self.service(), self.service()
        self.assertIsNone(first.render_identity())
        self.assertNotEqual(first.cache_scope(request, "t"), second.cache_scope(request, "t"))
        self.assertEqual(first.cache_scope(request, "t"), first.cache_scope(request, "t"))

    def test_a_partial_tokenizer_identity_is_never_invented(self):
        """The artifacts' content names the namespace: what is missing must decline reuse, not be guessed at."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text('{"a": 0}', encoding="utf-8")
            partial = self.service(tokenizer_dir=root)
            self.assertIsNone(partial.render_identity(), "a partial tokenizer identity was invented from what was there")
            self.assertEqual(partial.cache_scope({"messages": []}, "t"),
                             partial.cache_scope({"messages": []}, "t"))     # the process-local namespace still works
            self.assertNotEqual(partial.cache_scope({"messages": []}, "t"),
                                self.service().cache_scope({"messages": []}, "t"))

    def test_a_client_cannot_name_its_own_scope(self):
        service = self.service()
        claimed = {"messages": [], "_strata_cache_scope": "de" * 32, "_strata_cache_offer": 7}
        self.assertEqual(service.cache_scope(claimed, "tenant-a"), service.cache_scope({"messages": []}, "tenant-a"))
        self.assertEqual(service.cache_scope(claimed, "tenant-a", credential="k"),
                         service.cache_scope({"messages": []}, "tenant-a", credential="k"))

    def shared_service(self, **kwargs):
        service = self.service(**kwargs)
        service.api_key = "client-key,gateway-token"
        return service

    def test_shared_scope_binds_the_canonical_exact_token_prefix(self):
        service = self.shared_service()
        principal = ("gateway", "tenant-a", server.credential_fingerprint("gateway-token"))
        request = {"strata_shared_prefix": {"tokens": 2}}
        ids = [1, 0x102, 3, 4]
        payload = ["scc-shared1", principal, "ab" * 32, service.render_identity(), 2,
                   hashlib.sha256(b"\x01\x00\x00\x00\x02\x01\x00\x00").hexdigest()]
        expected = hmac.new(KEY, json.dumps(payload, separators=(",", ":")).encode(), "sha256").hexdigest()
        scope = service.shared_cache_scope(request, principal, ids)
        self.assertEqual(scope, expected)
        self.assertRegex(scope, r"^[0-9a-f]{64}$")
        self.assertNotEqual(scope, service.cache_scope(request, "tenant-a", gateway=True, credential="gateway-token"))
        self.assertNotEqual(service.shared_cache_scope(request, principal, [1, 23, 3, 4]),
                            service.shared_cache_scope(request, principal, [12, 3, 3, 4]))

    def test_shared_scope_survives_restart_without_adopting_a_conversation_identity(self):
        first, second = self.shared_service(), self.shared_service()
        principal = ("gateway", "tenant-a", server.credential_fingerprint("gateway-token"))
        a = {"strata_shared_prefix": {"tokens": 2}, "strata_conversation_id": "A"}
        b = {**a, "strata_conversation_id": "B", "seed": 99, "_strata_request_id": "other",
             "_strata_cache_deadline_ns": 42}
        scope = first.shared_cache_scope(a, principal, [1, 2, 3, 4])
        self.assertIsNotNone(scope)
        self.assertNotEqual(first.cache_secret, second.cache_secret)
        self.assertEqual(scope, second.shared_cache_scope(b, principal, [1, 2, 99, 100]))
        self.assertNotEqual(first.cache_scope(a, "tenant-a"), first.cache_scope(b, "tenant-a"))

    def test_every_shared_scope_binding_isolates_records(self):
        service = self.shared_service()
        principal = ("gateway", "tenant-a", server.credential_fingerprint("gateway-token"))
        request = {"strata_shared_prefix": {"tokens": 2}}
        ids = [1, 2, 3, 4, 5]
        baseline = service.shared_cache_scope(request, principal, ids)
        self.assertIsNotNone(baseline)
        cases = {
            "issuer": (self.shared_service(key=bytes(reversed(KEY))), principal, request, ids),
            "kind": (service, ("principal", *principal[1:]), request, ids),
            "tenant": (service, ("gateway", "tenant-b", principal[2]), request, ids),
            "credential": (service, ("gateway", "tenant-a", server.credential_fingerprint("rotated")), request, ids),
            "model": (self.shared_service(model_id="cd" * 32), principal, request, ids),
            "template": (self.shared_service(template=self.other_template()), principal, request, ids),
            "tokenizer": (self.shared_service(tok=OtherTokenizer()), principal, request, ids),
            "count": (service, principal, {"strata_shared_prefix": {"tokens": 3}}, ids),
            "tokens": (service, principal, request, [1, 99, 3, 4, 5]),
        }
        for dimension, (other, authorization, req, tokens) in cases.items():
            with self.subTest(dimension=dimension):
                scope = other.shared_cache_scope(req, authorization, tokens)
                self.assertIsNotNone(scope)
                self.assertNotEqual(baseline, scope)
        alias = self.shared_service(model="renamed-public-alias")
        self.assertEqual(baseline, alias.shared_cache_scope(request, principal, ids))

    def test_shared_scope_binds_each_complete_tokenizer_artifact(self):
        principal = ("principal", "client-key", server.credential_fingerprint("client-key"))
        request = {"strata_shared_prefix": {"tokens": 2}}
        baseline = self.shared_service().shared_cache_scope(request, principal, [1, 2, 3, 4])
        self.assertIsNotNone(baseline)
        for name in server.TOKENIZER_FILES:
            with self.subTest(artifact=name):
                path = self.tokenizer_dir / name
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n")
                    changed = self.shared_service().shared_cache_scope(request, principal, [1, 2, 3, 4])
                    self.assertIsNotNone(changed)
                    self.assertNotEqual(baseline, changed)
                finally:
                    path.write_bytes(original)

    def test_sharing_never_invents_authorization_or_identity(self):
        principal = ("gateway", "tenant-a", server.credential_fingerprint("gateway-token"))
        request = {"strata_shared_prefix": {"tokens": 2}}
        ids = [1, 2, 3, 4]
        service = self.shared_service()
        for flags in ({"has_images": True}, {"fallback": True}, {"score": True}):
            with self.subTest(flags=flags):
                self.assertIsNone(service.shared_cache_scope(request, principal, ids, **flags))
        for unauthorized in (None, ("gateway", "", principal[2]), ("gateway", "tenant-a", ""),
                             ("anonymous", "tenant-a", principal[2]), list(principal)):
            with self.subTest(principal=unauthorized):
                self.assertIsNone(service.shared_cache_scope(request, unauthorized, ids))
        service.api_key = ""
        self.assertIsNotNone(service.shared_cache_scope(request, principal, ids),
                             "a trusted principal is the authorization; the endpoint need not have a general gate")
        service.api_key = "client-key"
        service.template_fallback = True
        self.assertIsNone(service.shared_cache_scope(request, principal, ids))
        for key in (None, b"\0" * 32):
            with self.subTest(key=key):
                self.assertIsNone(self.shared_service(key=key).shared_cache_scope(request, principal, ids))
        for model_id in ("", None, "AB" * 32):
            with self.subTest(model_id=model_id):
                self.assertIsNone(self.shared_service(model_id=model_id).shared_cache_scope(request, principal, ids))
        for render_id in (None, "", "AB" * 32):
            with self.subTest(render_id=render_id):
                service = self.shared_service()
                with mock.patch.object(service, "render_identity", return_value=render_id):
                    self.assertIsNone(service.shared_cache_scope(request, principal, ids))
        (self.tokenizer_dir / "merges.txt").unlink()
        self.assertIsNone(self.shared_service().shared_cache_scope(request, principal, ids))

    def test_shared_scope_refuses_unresolved_or_unusable_boundaries_and_noncanonical_tokens(self):
        service = self.shared_service()
        principal = ("principal", "client-key", server.credential_fingerprint("client-key"))
        for prefix in (None, {}, {"messages": 1}, {"tokens": 0}, {"tokens": True}, {"tokens": 1.0},
                       {"tokens": 3}, {"tokens": 4}, {"tokens": 2, "message": 1}):
            with self.subTest(prefix=prefix):
                self.assertIsNone(service.shared_cache_scope({"strata_shared_prefix": prefix}, principal, [1, 2, 3, 4]))
        for invalid in (-1, True, 1.0, 1 << 31):
            with self.subTest(token=invalid):
                self.assertIsNone(service.shared_cache_scope({"strata_shared_prefix": {"tokens": 2}},
                                                             principal, [1, invalid, 3, 4]))

    def test_a_request_without_an_id_keeps_one_namespace_per_tenant(self):
        service = self.service()
        first = {"messages": [{"role": "user", "content": "one"}]}
        second = {"messages": [{"role": "user", "content": "two"}]}
        declared = {**first, "strata_conversation_id": "a"}
        self.assertEqual(service.cache_scope(first, "t"), service.cache_scope(second, "t"))   # one content space
        self.assertNotEqual(service.cache_scope(first, "t"), service.cache_scope(first, "u"))  # never another's
        self.assertNotEqual(service.cache_scope(first, "t"), service.cache_scope(declared, "t"))


class LinkTransport(unittest.TestCase):
    """The protocol itself: a sealed read-only payload, bounded outstanding offers, a bounded failure surface."""

    def setUp(self):
        self.link = conversation_offers.ConversationOfferLink()
        self.addCleanup(self.link.close)

    def test_an_offer_reaches_the_store_sealed_and_read_only(self):
        peer = NativePeer(self.link)
        self.link.start()
        peer.start()
        tokens = [7, 8, 9, 1 << 20]
        self.assertTrue(self.link.offer(1, self.link.epoch(), 123456789, "ab" * 32, tokens))
        offer = peer.next_offer()
        self.assertIsNotNone(offer)
        assert offer is not None
        self.assertEqual(offer["tokens"], tokens)
        self.assertEqual(offer["request_id"], 1)
        self.assertEqual(offer["generation"], peer.hello.generation)
        self.assertEqual(offer["deadline_ns"], 123456789)
        self.assertEqual(offer["scope"], bytes.fromhex("ab" * 32))
        self.assertTrue(offer["want_cvec"])
        self.assertFalse(offer["writable"])
        for seal in (0x0001, 0x0002, 0x0004, 0x0008):
            self.assertTrue(offer["seals"] & seal, f"seal {seal} missing from {offer['seals']}")
        self.assertEqual(self.link.stats()["accepted"], 1)
        self.assertEqual(self.link.stats()["outstanding"], 1)

    def test_the_byte_budget_and_a_duplicate_id_bound_what_is_sent(self):
        peer = NativePeer(self.link, metadata_bytes=4 * 3)
        self.link.start()
        peer.start()
        tokens = [1, 2, 3]
        self.assertTrue(self.link.offer(1, self.link.epoch(), 0, "cd" * 32, tokens, token_limit=100))
        self.assertFalse(self.link.offer(2, self.link.epoch(), 0, "cd" * 32, [4, 5, 6, 7], token_limit=100))
        self.assertFalse(self.link.offer(1, self.link.epoch(), 0, "cd" * 32, tokens, token_limit=100))
        self.assertFalse(self.link.offer(3, self.link.epoch(), 0, "cd" * 32, []))
        self.assertFalse(self.link.offer(4, self.link.epoch(), 0, "cd" * 32, [1, 2], token_limit=1))
        self.assertFalse(self.link.offer(5, self.link.epoch() + 1, 0, "cd" * 32, [1]))
        self.assertFalse(self.link.offer(6, self.link.epoch(), 0, "not-hex", [1]))
        self.assertEqual(self.link.stats()["accepted"], 1)
        self.assertEqual(self.link.stats()["rejected"], 6)
        self.assertIsNotNone(peer.next_offer())
        peer.drain(0.4)
        self.assertEqual(len(peer.offers), 1)
        self.assertEqual(peer.other, [])

    def test_without_a_store_nothing_is_offered(self):
        peer = NativePeer(self.link)
        self.assertFalse(self.link.store_enabled)
        self.assertIsNone(self.link.scope_key())
        self.assertFalse(self.link.offer(1, 1, 0, "ab" * 32, [1, 2, 3]))
        self.assertIsNone(peer.read_one(0.2), "nothing at all may be sent before the HELLO")
        self.link.start()
        peer.start()
        self.assertTrue(self.link.store_enabled)
        self.assertIsNotNone(self.link.scope_key())

    def test_cancelling_retires_exactly_one_offer(self):
        peer = NativePeer(self.link, frontier=2)
        self.link.start()
        peer.start()
        for request_id in (1, 2):
            self.assertTrue(self.link.offer(request_id, self.link.epoch(), 0, "ef" * 32, [request_id] * 3))
        self.assertFalse(self.link.cancel(1, self.link.epoch() + 5))     # never another generation's
        self.assertTrue(self.link.cancel(1, self.link.epoch()))
        self.assertFalse(self.link.cancel(1, self.link.epoch()))         # the pair is retired once
        self.assertEqual(peer.next_cancel(), (1, self.link.epoch()))
        peer.drain(0.3)
        self.assertEqual(peer.cancels, [(1, self.link.epoch())])
        self.assertEqual(self.link.stats()["cancelled"], 1)
        self.assertEqual(self.link.stats()["outstanding"], 1)             # the second offer is untouched

    def test_a_broken_transport_is_a_refusal_and_leaks_nothing(self):
        peer = NativePeer(self.link)
        self.link.start()
        peer.start()
        before = len(os.listdir("/proc/self/fd"))
        peer.sock.close()                                                # the engine's end is gone
        for i in range(20):
            self.assertFalse(self.link.offer(100 + i, self.link.epoch(), 0, "ab" * 32, [1, 2, 3]))
        self.assertEqual(self.link.stats()["rejected"], 20)
        self.assertEqual(self.link.stats()["outstanding"], 0)             # no reservation is left behind
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), before)

    def test_close_releases_both_ends_and_is_idempotent(self):
        peer = NativePeer(self.link)
        self.link.start()
        peer.start()
        self.link.close()
        self.link.close()
        self.assertTrue(self.link.closed)
        self.assertIsNone(self.link.child_socket)
        self.assertEqual(self.link.stats()["closed"], True)
        self.assertFalse(self.link.offer(1, 1, 0, "ab" * 32, [1]))
        self.assertFalse(self.link.cancel(1, 1))

    def test_a_payload_that_cannot_be_built_is_a_refusal(self):
        peer = NativePeer(self.link)
        self.link.start()
        peer.start()
        self.assertFalse(self.link.offer(-1, self.link.epoch(), 0, "ab" * 32, [1]))
        self.assertFalse(self.link.offer(0, self.link.epoch(), 0, "ab" * 32, [1]))
        self.assertFalse(self.link.offer(1, 0, 0, "ab" * 32, [1]))
        self.assertIsNone(peer.read_one(0.3))

    def test_offer_ids_never_enter_the_native_demand_range(self):
        """Native-only demand reads own the high half of the id space: an offer stays strictly below it."""
        peer = NativePeer(self.link)
        self.link.start()
        peer.start()
        with self.link._lock:
            self.link._next_id = conversation_offers.OFFER_ID_MAX - 1
        self.assertEqual(self.link.next_request_id(), conversation_offers.OFFER_ID_MAX)
        self.assertEqual(self.link.next_request_id(), 0, "an exhausted incarnation kept handing out ids")
        self.assertFalse(self.link.offer(conversation_offers.OFFER_ID_MAX + 1, self.link.epoch(), 0, "ab" * 32, [1]))
        self.assertFalse(self.link.offer(1 << 63, self.link.epoch(), 0, "ab" * 32, [1]))
        self.assertFalse(self.link.offer(conversation_offers.OFFER_ID_MAX, self.link.epoch() + 1, 0, "ab" * 32, [1]))
        self.assertIsNone(peer.read_one(0.3))


class QueuedFrontier(ChannelTestCase):
    """What the real production path does with queued, varied prompts."""

    def test_the_frontier_is_prepared_once_and_the_demand_binds_that_offer(self):
        rendered = mock.patch.object(self.svc, "encode_prompt", wraps=self.svc.encode_prompt)
        spy = rendered.start()
        self.addCleanup(rendered.stop)
        resolved = mock.patch.object(self.svc, "resolve_openai", wraps=self.svc.resolve_openai)
        resolutions = resolved.start()
        self.addCleanup(resolved.stop)
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5), "the first request never reached the engine")
        second = self.start_request("second")
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered, "the queue head was never prepared and offered")
        self.assertGreater(self.svc.offer_stats()["prepared_bytes"], 0,
                           "the frontier's preparation was not reserved before it rendered")
        third = self.start_request("three")
        self.wait_queued(2)
        self.peer.drain(0.6)
        self.assertEqual(len(self.peer.offers), 1, "a request behind the frontier was prepared too")
        # Only the frontier is touched while it waits: the admitted request resolved itself, the one behind the
        # frontier was neither resolved nor rendered.
        self.assertEqual(resolutions.call_count, 2, "a request behind the frontier was resolved ahead of its turn")
        self.assertEqual(spy.call_count, 2, "a request behind the frontier was rendered ahead of its turn")
        self.assertEqual(self.peer.cancels, [])
        self.assertEqual(self.peer.other, [])
        self.engine.release.set()
        self.join(first, second, third)
        self.peer.drain(0.2)
        self.assertEqual([outcome.get("status") for _t, outcome in (first, second, third)], [200, 200, 200])
        self.assertEqual(self.engine.prompts[1], list(self.peer.offers[0]["tokens"]))   # consumed, not re-rendered
        self.assertEqual(spy.call_count, 3, "a prepared prompt was rendered again")
        # five resolutions: each request's own demand, plus the two preparations - and never a second one for a
        # result that was already prepared
        self.assertEqual(resolutions.call_count, 5, "a prepared prompt was resolved again")
        self.assertEqual([c[0] for c in self.peer.cancels], [o["request_id"] for o in self.peer.offers])
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 0,
                         "a finished request kept its preparation reserved")

    def test_each_queued_request_keeps_its_own_prompt_and_settings(self):
        """Varied queued requests: every offer is its own request's tokens, not a neighbour's or a re-render."""
        bodies = [{"messages": [{"role": "user", "content": f"turn {i} " + "x" * (10 * i)}],
                   "max_tokens": 3 + i, "temperature": 0.25 * i} for i in range(1, 4)]
        first = self.start_request(body=bodies[0])
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request(body=bodies[1])
        self.wait_queued(1)
        third = self.start_request(body=bodies[2])
        self.wait_queued(2)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered)
        assert offered is not None
        # the queue head is the second of the three (the first is already running): its own tokens, not the others'
        self.assertIn(bodies[1]["messages"][0]["content"], prompt_text(offered["tokens"]))
        self.assertNotIn(bodies[2]["messages"][0]["content"], prompt_text(offered["tokens"]))
        self.engine.release.set()
        self.join(first, second, third)
        self.assertEqual([o.get("status") for _t, o in (first, second, third)], [200, 200, 200])
        for i, sampling in enumerate(self.engine.samplings):
            self.assertIn(bodies[i]["messages"][0]["content"], prompt_text(self.engine.prompts[i]))
            self.assertEqual(sampling["max_tokens"], bodies[i]["max_tokens"])
        consumed = [p for p in self.engine.prompts if any(p == list(o["tokens"]) for o in self.peer.offers)]
        self.assertEqual(len(consumed), len(self.peer.offers),
                         "a prepared prompt did not reach the engine exactly as it was offered")

    def test_only_text_requests_are_prepared_early(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        image = {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"},
                                                           {"type": "image_url", "image_url": {"url": "data:x"}}]}]}
        queued = self.start_request(body=image)
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(self.peer.offers, [])
        self.engine.release.set()
        self.join(first, queued)
        self.assertEqual(queued[1].get("status"), 400)           # the demand path still owns its refusal

    def test_a_cancelled_queued_request_retires_its_own_offer(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request("second")
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered)
        assert offered is not None
        queued = [r for r in self.svc.lifecycle_records() if r["state"] == "queued"]
        self.assertEqual(len(queued), 1)
        record = next(r for r in self.svc.lifecycles if r["id"] == queued[0]["id"])
        self.assertIsNotNone(record.get("_prepared"))
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 1)     # its bytes really were reserved
        self.assertTrue(self.svc.cancel_request(queued[0]["id"]))
        self.join(second)
        self.assertEqual(second[1].get("status"), 409)
        self.assertEqual(self.peer.next_cancel(5), (offered["request_id"], offered["generation"]))
        self.peer.drain(0.4)
        self.assertEqual(len(self.peer.cancels), 1, "a retired request left another offer outstanding")
        # A cancelled request frees everything its preparation held: the reservation, the byte budget and its own
        # references to the rendered prompt.
        self.assertNotIn("_prepared", record)
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 0)
        self.assertEqual(self.svc.offer_stats()["prepared_bytes"], 0)
        self.assertEqual(self.svc.offer_stats()["outstanding"], 0)
        self.assertEqual(self.svc.offer_stats()["outstanding_bytes"], 0)
        self.engine.release.set()
        self.join(first)
        self.assertEqual(first[1].get("status"), 200)

    def test_a_queued_request_runs_the_pin_and_steering_the_store_was_offered(self):
        self.engine.info = {"cvec": 1}        # a loaded control vector: this request's own cvec= is what runs
        body = {"messages": [{"role": "user", "content": "a shared document " * 20},
                             {"role": "user", "content": "the question"}],
                "strata_prefix": {"messages": 1}, "max_tokens": 4,
                "experimental_speed_projection": False}
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request(body=body)
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered, "the pinned request was not prepared")
        assert offered is not None
        self.assertFalse(offered["want_cvec"], "the offer did not declare the request's own steering")
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)
        keys = self.keys_of(self.engine.samplings[1])
        self.assertIn("cvec=0", keys)                      # the steering the offer declared
        self.assertRegex(keys, r" pin=\d+")                # the boundary the freeze resolved, not the client's spec
        pinned = self.engine.samplings[1]["strata_prefix"]
        self.assertIsInstance(pinned, dict)
        self.assertIsInstance(pinned.get("tokens"), int)   # {"messages": 1} can never become this by itself
        self.assertGreater(pinned["tokens"], 0)
        self.assertEqual(self.engine.prompts[1], list(offered["tokens"]))

    def test_a_request_with_no_control_vector_is_offered_as_unsteered(self):
        """An engine without a loaded control vector prices nothing, so the offer must not ask for a vector."""
        body = {"messages": [{"role": "user", "content": "a turn"}], "experimental_speed_projection": False}
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request(body=body)
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered)
        assert offered is not None
        self.assertTrue(offered["want_cvec"], "an unsteered engine's offer claimed steering the demand will not run")
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)

    def test_a_prepared_result_is_consumed_once_and_only_for_its_own_prompt(self):
        generation = conversation_offers.generation_digest([], None, {}, 4, None)
        prepared = conversation_offers.PreparedGeneration(ids=(1, 2, 3), thinking=True, max_new=4, digest=generation,
                                                          ctx=4096, link=self.link, offer_id=9,
                                                          generation=self.link.epoch(),
                                                          prefix={"tokens": 2})
        self.assertIsNone(self.svc.take_prepared(None, [], None, {}, 4, None))
        record = {"id": "prepared-consumption", "_prepared": prepared}
        self.assertIs(self.svc.take_prepared(record, [], None, {}, 4, None), prepared)     # this exact prompt
        self.assertIsNone(self.svc.take_prepared(record, [{"role": "user"}], None, {}, 4, None))   # once only
        request: dict = {}
        self.svc.bind_offer(request, prepared)
        prepared.apply_to(request)
        self.assertEqual(request.get("_strata_cache_offer"), 9)
        self.assertEqual(request["strata_prefix"], {"tokens": 2})    # the frozen pin boundary, not the client's spec
        request["strata_prefix"] = {"messages": 1}                   # a second resolution must not be able to win
        prepared.apply_to(request)
        self.assertEqual(request["strata_prefix"], {"tokens": 2})
        prepared.release()
        self.assertEqual(prepared.ids, (), "a consumed result kept the whole prompt")
        self.assertIsNone(prepared.prefix)
        self.svc.bind_offer(request, prepared)
        self.assertEqual(request.get("_strata_cache_offer"), 9)
        stale = dict(request)
        strange = conversation_offers.ConversationOfferLink()
        self.addCleanup(strange.close)
        self.svc.bind_offer(stale, conversation_offers.PreparedGeneration(ids=(1,), thinking=False, max_new=1,
                                                                          digest="", ctx=0, link=strange, offer_id=3,
                                                                          generation=1))
        self.assertNotIn("_strata_cache_offer", stale)      # another incarnation's offer is never bound
        self.svc.bind_offer(stale, prepared)
        self.assertEqual(stale.get("_strata_cache_offer"), 9)
        self.svc.withdraw_offer({"id": record["id"], "_prepared": prepared})
        self.assertTrue(prepared.withdrawn)
        self.assertIsNone(self.svc.withdraw_offer({"id": record["id"], "_prepared": prepared}))
        self.assertIsNone(self.svc.withdraw_offer(None))

    def test_a_prepared_result_that_is_not_this_prompt_retires_its_offer(self):
        """A generation that re-resolved to another prompt never consumes the offer: it is cancelled, not held."""
        self.assertTrue(self.link.offer(4, self.link.epoch(), 0, "ab" * 32, [1, 2, 3]))
        prepared = conversation_offers.PreparedGeneration(ids=(1, 2, 3), thinking=False, max_new=4, digest="other",
                                                          ctx=self.svc.context_size(), link=self.link, offer_id=4,
                                                          generation=self.link.epoch())
        record = {"id": "prepared-mismatch", "_prepared": prepared}
        self.assertIsNone(self.svc.take_prepared(record, [], None, {}, 4, None))
        self.assertTrue(prepared.consumed)
        self.assertTrue(prepared.withdrawn)
        self.assertNotIn("_prepared", record)
        self.assertEqual(self.peer.next_cancel(5), (4, self.link.epoch()))
        self.assertEqual(self.link.stats()["outstanding"], 0)

    def test_a_shutdown_stops_admission_and_cancels_the_offer(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request("second")
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered)
        self.svc.stop_admission()
        third = self.start_request("third")
        self.join(second, third)
        self.assertEqual(second[1].get("status"), 409)
        self.assertEqual(third[1].get("status"), 503)
        self.assertIsNotNone(self.peer.next_cancel(5))
        self.engine.release.set()
        self.join(first)


class OfferDiagnosticCase(ChannelTestCase):
    timing = "1"

    def setUp(self):
        capture = mock.patch.object(server, "print", create=True)
        self.printed = capture.start()
        self.addCleanup(capture.stop)
        with mock.patch.dict(os.environ, {"STRATA_STORE_TIMING": self.timing}):
            super().setUp()
        self.addCleanup(self.engine.release.set)

    def diagnostic_records(self, kind):
        prefix = f"[strata] {kind}: "
        return [json.loads(call.args[0][len(prefix):]) for call in self.printed.call_args_list
                if call.args and isinstance(call.args[0], str) and call.args[0].startswith(prefix)]

    def wait_attributions(self, count):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            records = self.diagnostic_records("response_attribution")
            if len(records) >= count:
                return records
            time.sleep(.01)
        self.fail(f"never saw {count} completed response attributions")


class StoreOfferTiming(OfferDiagnosticCase):
    def test_consumed_offer_survives_withdrawal_and_never_aliases_the_next_request(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request("second")
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered)
        self.engine.release.set()
        self.join(first, second)
        self.wait_attributions(2)
        third = self.start_request("third")
        self.join(third)
        attributions = self.wait_attributions(3)
        self.assertEqual([outcome.get("status") for _, outcome in (first, second, third)], [200, 200, 200])
        mapped, = [r for r in attributions if "cache_offer" in r]
        correlation = mapped["cache_offer"]
        self.assertEqual((correlation["offer_id"], correlation["generation"]),
                         (offered["request_id"], offered["generation"]))
        self.assertEqual(correlation["native_acceptance"], "unknown")
        for field in ("sent", "prepared_consumed", "prepared_reusable", "demand_bound", "withdrawn", "cancel_sent"):
            self.assertTrue(correlation[field], field)
        events = [r for r in self.diagnostic_records("store_offer") if r["request_id"] == mapped["request_id"]]
        self.assertEqual([r["last_event"] for r in events],
                         ["offer_sent", "prepared_consumed", "demand_bound", "offer_withdrawn"])
        self.assertEqual([r["last_event_monotonic_ns"] for r in events],
                         sorted(r["last_event_monotonic_ns"] for r in events))
        self.assertTrue(all(r["native_acceptance"] == "unknown" for r in events))
        record = next(r for r in self.svc.lifecycles if r["id"] == mapped["request_id"])
        self.assertNotIn("_prepared", record)
        public = self.svc.lifecycle_records(mapped["request_id"])
        self.assertNotIn("_store_offer", public)
        self.assertNotIn("cache_offer", public["attribution"], "diagnostics must not extend the lifecycle API")
        for _, outcome in (first, second, third):
            self.assertNotIn("cache_offer", json.loads(outcome["body"]), "no new response API fields")

    def test_cancelled_queued_offer_retains_its_identity_without_claiming_native_acceptance(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request("cancel this")
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered)
        queued, = [r for r in self.svc.lifecycle_records() if r["state"] == "queued"]
        self.assertTrue(self.svc.cancel_request(queued["id"]))
        self.join(second)
        cancelled, = self.wait_attributions(1)
        self.assertEqual(second[1].get("status"), 409)
        self.assertEqual(cancelled["request_id"], queued["id"])
        self.assertEqual(cancelled["outcome"], "cancelled")
        correlation = cancelled["cache_offer"]
        self.assertEqual((correlation["offer_id"], correlation["generation"]),
                         (offered["request_id"], offered["generation"]))
        self.assertFalse(correlation["prepared_consumed"])
        self.assertFalse(correlation["demand_bound"])
        self.assertTrue(correlation["withdrawn"])
        self.assertTrue(correlation["cancel_sent"])
        self.assertEqual(correlation["native_acceptance"], "unknown")
        self.engine.release.set()
        self.join(first)
        first_record, = [r for r in self.wait_attributions(2) if r["request_id"] != queued["id"]]
        self.assertNotIn("cache_offer", first_record)

    def test_prompt_mismatch_and_reoffer_keep_distinct_bounded_correlations(self):
        request = {"messages": [{"role": "user", "content": "prepare this"}],
                   "_strata_cache_scope": "ab" * 32, "max_tokens": 4}
        record = self.svc.register_lifecycle(request, threading.Event())
        self.svc.prefetch_offer(record, "/v1/chat/completions", request)
        first = self.peer.next_offer(5)
        self.assertIsNotNone(first)
        self.assertIsNone(self.svc.take_prepared(record, [], None, {}, 4, None))
        self.assertNotIn("_prepared", record)
        self.svc.prefetch_offer(record, "/v1/chat/completions", request)
        second = self.peer.next_offer(5)
        self.assertIsNotNone(second)
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.svc.withdraw_offer(record)
        self.svc.finish_lifecycle(record)
        attribution = server._response_attribution(record, "/v1/chat/completions")
        correlation = attribution["cache_offer"]
        self.assertEqual(correlation["offer_sequence"], 2)
        self.assertEqual(correlation["offer_id"], second["request_id"])
        self.assertFalse(correlation["prepared_consumed"], "a prior offer's consumption must not leak")
        self.assertFalse(correlation["demand_bound"])
        self.assertNotIn("_prepared", record)
        events = self.diagnostic_records("store_offer")
        first_events = [r for r in events if r["offer_id"] == first["request_id"]]
        self.assertEqual([r["last_event"] for r in first_events],
                         ["offer_sent", "prepared_consumed", "offer_withdrawn"])
        self.assertFalse(first_events[1]["prepared_reusable"])
        self.assertTrue(all(r["offer_sequence"] == 1 for r in first_events))
        self.assertTrue(all(r["request_id"] == record["id"] for r in events))
        self.assertIsInstance(record["_store_offer"], dict)   # one snapshot, not an accumulating trace

    def test_unsent_preparation_never_claims_a_native_offer(self):
        record = self.svc.register_lifecycle({}, threading.Event())
        prepared = conversation_offers.PreparedGeneration(ids=(1, 2), thinking=False, max_new=4,
                                                          digest="", ctx=self.svc.context_size())
        self.svc.offer_prepared(record, self.link, prepared, {})  # no scope: tokens stay local
        request = {}
        self.svc.bind_offer(request, prepared, record=record)
        self.svc.withdraw_offer(record)
        self.svc.finish_lifecycle(record)
        correlation = server._response_attribution(record, "/v1/chat/completions")["cache_offer"]
        self.assertIsNone(correlation["offer_id"])
        self.assertFalse(correlation["sent"])
        self.assertEqual(correlation["native_acceptance"], "not_sent")
        self.assertFalse(correlation["demand_bound"])
        self.assertFalse(correlation["cancel_sent"])
        self.assertNotIn("_strata_cache_offer", request)
        self.assertEqual(self.peer.offers, [])


class StoreOfferTimingDisabled(OfferDiagnosticCase):
    timing = "0"

    def test_disabled_mode_keeps_normal_response_and_attribution_without_diagnostic_work(self):
        self.assertFalse(self.svc.store_timing)
        with mock.patch.object(self.svc, "_store_offer_event") as diagnostic:
            first = self.start_request("first")
            self.assertTrue(self.engine.entered.wait(5))
            second = self.start_request("second")
            self.assertIsNotNone(self.peer.next_offer(5))
            self.engine.release.set()
            self.join(first, second)
            attributions = self.wait_attributions(2)
        diagnostic.assert_not_called()
        self.assertEqual(self.diagnostic_records("store_offer"), [])
        self.assertTrue(all("cache_offer" not in r for r in attributions))
        self.assertTrue(all("_store_offer" not in r for r in self.svc.lifecycles))
        self.assertEqual([outcome.get("status") for _, outcome in (first, second)], [200, 200])
        for _, outcome in (first, second):
            body = json.loads(outcome["body"])
            message = body["choices"][0]["message"]
            self.assertEqual(message.get("content") or message.get("reasoning_content"), "ok")
            self.assertNotIn("cache_offer", body)


class WiderFrontier(ChannelTestCase):
    """The wider engine (two pipeline groups) prepares two queued candidates."""

    frontier = 2
    batch = 2
    hold = 2

    def test_both_candidates_of_the_frontier_are_prepared(self):
        first = self.start_request("first")
        second = self.start_request("second")
        self.assertTrue(self.engine.entered.wait(5), "the two engine slots were never both busy")
        third = self.start_request("third")
        fourth = self.start_request("fourth")
        self.wait_queued(2)
        self.assertIsNotNone(self.peer.next_offer(5))
        self.assertIsNotNone(self.peer.next_offer(5))
        self.assertEqual(len(self.peer.offers), 2)
        self.assertNotEqual(self.peer.offers[0]["tokens"], self.peer.offers[1]["tokens"])
        self.assertEqual(len({o["request_id"] for o in self.peer.offers}), 2)
        self.engine.release.set()
        self.join(first, second, third, fourth)
        self.assertEqual([outcome.get("status") for _t, outcome in (first, second, third, fourth)],
                         [200, 200, 200, 200])


class FrontierArithmetic(ChannelTestCase):
    """The allowance admits what the frontier actually settles: two whole-context reservations fit inside it beside
    the tokenizer's own cache share, so a frontier of two prepares both of its candidates at 262144 tokens instead of
    silently falling back to one.  The arithmetic decides that, not whether one prompt happens to tokenize."""

    frontier = 2
    batch = 2
    hold = 2
    context = 262144
    metadata_bytes = 64 << 20        # the HELLO's descriptor share (one whole payload per slot fits inside it)

    def engine_for(self):
        return QueuedEngine(self.tok, link=self.link, batch=self.batch, hold=self.hold, max_context=self.context)

    def test_two_full_context_slots_fit_the_allowance_and_both_are_prepared(self):
        slot, descriptor = self.svc.preparation_slot()
        self.assertEqual(descriptor, 4 * self.context)
        self.assertLessEqual(self.svc.preparation_cache_bytes() + self.frontier * slot,
                             self.svc.PREPARATION_ALLOWANCE_BYTES,
                             "the allowance cannot hold the frontier's own reservations and the cache share")
        first = self.start_request("first")
        second = self.start_request("second")
        self.assertTrue(self.engine.entered.wait(5), "the two engine slots were never both busy")
        third = self.start_request("third")
        fourth = self.start_request("fourth")
        self.wait_queued(2)
        self.assertIsNotNone(self.peer.next_offer(5), "the frontier's own two candidates were not both prepared")
        self.assertIsNotNone(self.peer.next_offer(5))
        self.assertEqual(len(self.peer.offers), 2, "the second full-context reservation was refused")
        stats = self.svc.offer_stats()
        self.assertEqual(stats["preparation_refused"], 0)
        self.assertEqual(stats["prepared_requests"], 2)
        self.assertEqual(stats["prepared_bytes"], 2 * slot)
        self.engine.release.set()
        self.join(first, second, third, fourth)
        self.assertEqual([outcome.get("status") for _t, outcome in (first, second, third, fourth)],
                         [200, 200, 200, 200])


class SpeculativeValidation(ChannelTestCase):
    """A `json_schema` request is prepared at demand only: its validation is the installed jsonschema's own walk of a
    structure this thread has not admitted, so preparation never runs it - while the demand validates and answers it
    exactly as it always did."""

    SCHEMA = {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"],
                   "additionalProperties": False}}}

    def engine_for(self):
        """The demand's answer is the object its schema asks for, so the whole path can be observed end to end."""
        return QueuedEngine(self.tok, link=self.link, script='{"answer": 4}', batch=self.batch, hold=self.hold)

    def test_the_demand_validates_its_own_schema_and_preparation_never_does(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        validating = mock.patch.object(server, "prepare_format", wraps=server.prepare_format)
        validate = validating.start()
        self.addCleanup(validating.stop)
        second = self.start_request(body={"messages": [{"role": "user", "content": "a turn"}],
                                          "response_format": self.SCHEMA, "reasoning_effort": "none",
                                          "max_tokens": 32})
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(validate.call_count, 0, "preparation validated a schema speculatively")
        self.assertEqual(self.peer.offers, [], "a json_schema request was prepared early")
        record = next(r for r in self.svc.lifecycles if r["state"] == "queued")
        self.assertTrue(record.get("_offer_declined"))
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(validate.call_count, 1, "the demand did not validate the schema it answers under")
        self.assertEqual(second[1].get("status"), 200, second[1].get("body"))
        reply = json.loads(second[1]["body"])
        self.assertEqual(json.loads(reply["choices"][0]["message"]["content"]), {"answer": 4})


class DigestBounds(unittest.TestCase):
    """The generation digest is taken incrementally, over inputs the same admission admitted, and a digest that
    cannot be taken is never reusable (two failures must not match each other)."""

    VALUES = ([{"role": "user", "content": "hi"}], [{"name": "t"}], {"reasoning_effort": "low"}, 4, None)

    def test_a_digest_is_stable_for_its_own_inputs_and_changes_with_them(self):
        first = conversation_offers.generation_digest(*self.VALUES)
        self.assertIsInstance(first, str)
        self.assertTrue(first)
        self.assertEqual(first, conversation_offers.generation_digest(*self.VALUES))
        self.assertNotEqual(first, conversation_offers.generation_digest(*([{"role": "user", "content": "ho"}],)
                                                                        + self.VALUES[1:]))
        self.assertNotEqual(first, conversation_offers.generation_digest(*self.VALUES[:2] + ({"reasoning_effort": "xhigh"},)
                                                                        + self.VALUES[3:]))

    def test_inputs_that_cannot_be_hashed_are_refused_rather_than_serialized(self):
        self.assertIsNone(conversation_offers.generation_digest([{1: "a"}], None, {}, 4, None))
        self.assertIsNone(conversation_offers.generation_digest([{"content": {1, 2}}], None, {}, 4, None))
        self.assertIsNone(conversation_offers.generation_digest(
            ["x" * (conversation_offers.PREPARATION_INPUT_CHARS + 1)], None, {}, 4, None))
        refused = conversation_offers.PreparationWalk(1 << 20, 1 << 20, 2)
        self.assertFalse(refused.add([[[[[[1]]]]]]))
        self.assertIsNone(conversation_offers.generation_digest([[[[[[1]]]]]], None, {}, 4, None, walk=refused))

    def test_a_digest_that_cannot_be_taken_is_never_reusable(self):
        prepared = conversation_offers.PreparedGeneration(ids=(1,), thinking=False, max_new=1, digest="", ctx=16)
        self.assertFalse(prepared.reusable(None, 16))
        self.assertFalse(prepared.reusable("", 16))
        prepared.digest = "abc"
        self.assertTrue(prepared.reusable("abc", 16))
        self.assertFalse(prepared.reusable("abc", 17))
        self.assertFalse(prepared.reusable(None, 16))

    def test_the_walk_counts_nodes_depth_and_what_a_decoded_string_hides(self):
        walk = conversation_offers.PreparationWalk(1 << 20, 1 << 20, 64)
        self.assertTrue(walk.add({"a": [1, 2, 3]}))
        self.assertEqual((walk.nodes, walk.depth), (5, 2))          # the root, the list, three numbers
        self.assertGreater(walk.chars, 0)
        self.assertIsNone(walk.refused)
        decoded = conversation_offers.PreparationWalk(1 << 20, 1 << 20, 64, ("messages",))
        self.assertTrue(decoded.add({"messages": "x" * 100}))
        self.assertEqual(decoded.nodes, 3 + 100)                    # ... and the nodes that string can decode to
        deep = conversation_offers.PreparationWalk(1 << 20, 1 << 20, 2)
        self.assertFalse(deep.add({"a": {"b": {"c": 1}}}))
        self.assertIn("nesting", deep.refused)
        wide = conversation_offers.PreparationWalk(1 << 20, 10, 64)
        self.assertFalse(wide.add(list(range(20))))
        self.assertIn("nodes", wide.refused)
        expensive = conversation_offers.PreparationWalk(16, 1 << 20, 64)
        self.assertFalse(expensive.add({"a": "b" * 100}))
        self.assertIn("characters", expensive.refused)


class ChangedDemandInput(ChannelTestCase):
    """A demand whose own inputs changed (or are past the allowance) never consumes the prepared result: the offer is
    retired first, without building the digest of a replacement nothing has admitted."""

    def test_a_demand_past_the_allowance_withdraws_the_offer_before_digesting_it(self):
        generation = conversation_offers.generation_digest([], None, {}, 4, None)
        self.assertTrue(self.link.offer(7, self.link.epoch(), 0, "ab" * 32, [1, 2, 3]))
        prepared = conversation_offers.PreparedGeneration(ids=(1, 2, 3), thinking=False, max_new=4, digest=generation,
                                                          ctx=self.svc.context_size(), link=self.link, offer_id=7,
                                                          generation=self.link.epoch())
        record = {"id": "changed-demand", "_prepared": prepared}
        huge = [{"role": "user", "content": "x" * (conversation_offers.PREPARATION_INPUT_CHARS + 1)}]
        hashing = mock.patch.object(conversation_offers.json.JSONEncoder, "iterencode",
                                    side_effect=AssertionError("an unadmitted input was digested"))
        hashing.start()
        self.addCleanup(hashing.stop)
        self.assertIsNone(self.svc.take_prepared(record, huge, None, {}, 4, None))
        self.assertTrue(prepared.consumed)
        self.assertTrue(prepared.withdrawn)
        self.assertNotIn("_prepared", record)
        self.assertEqual(self.peer.next_cancel(5), (7, self.link.epoch()))
        self.assertEqual(self.link.stats()["outstanding"], 0)


class ResolutionExpansion(ChannelTestCase):
    """A namespace tool's own description is written again in front of each of its members, so the admission charges
    that expansion *before* the resolution builds it: a request whose expansion is past the allowance is prepared at
    demand only, and the demand still expands it and runs it."""

    def engine_for(self):
        """The expanded prompt is many times the request's own text: the engine's context must hold what it makes."""
        return QueuedEngine(self.tok, link=self.link, batch=self.batch, hold=self.hold, max_context=131072)

    def test_a_namespace_expansion_past_the_allowance_is_declined_before_it_is_built(self):
        namespace = "namespace instructions for every member tool " * 10
        body = {"input": "a turn", "max_output_tokens": 4, "tools": [
            {"type": "namespace", "name": "mcp_fs", "description": namespace,
             "tools": [{"type": "function", "name": f"read{i}", "parameters": {"type": "object"}} for i in range(40)]}]}
        walk = self.svc.preparation_walk()
        self.assertTrue(walk.add(body))
        expansion = 40 * (conversation_offers.json_string_charge(namespace) + 2)
        self.assertGreater(expansion, 0)
        bound = walk.chars + expansion // 2          # between the request's own JSON and its expansion
        self.assertGreater(walk.chars, 0)
        with mock.patch.object(self.svc, "preparation_input_chars", return_value=bound):
            resolving = mock.patch.object(self.svc, "resolve_for_offer", wraps=self.svc.resolve_for_offer)
            resolve = resolving.start()
            self.addCleanup(resolving.stop)
            first = self.start_request("first")
            self.assertTrue(self.engine.entered.wait(5))
            second = self.start_request(body=body, path="/v1/responses")
            self.wait_queued(1)
            self.peer.drain(0.6)
            self.assertEqual(resolve.call_count, 0, "the expansion was built before it was paid for")
            self.assertEqual(self.peer.offers, [])
            record = next(r for r in self.svc.lifecycles if r["state"] == "queued")
            self.assertTrue(record.get("_offer_declined"))
            self.engine.release.set()
            self.join(first, second)
        self.assertEqual(second[1].get("status"), 200, second[1].get("body"))
        prompt = prompt_text(self.engine.prompts[1])
        self.assertIn(namespace, prompt, "the demand did not expand the namespace it had to render")
        self.assertIn("mcp_fs.read0", prompt)


class NoStoreFrontier(ChannelTestCase):
    """A store that keeps nothing (a disabled HELLO) offers nothing: the demand path stays exactly as it was."""

    enabled = False
    generation = 0
    key = b"\0" * 32

    def test_no_offer_is_ever_published(self):
        self.assertTrue(self.link.wait_hello(0), "a disabled native store must still complete its handshake")
        self.assertEqual(self.svc.offer_stats()["malformed"], 0)
        self.assertFalse(self.svc.offers_ready())
        self.assertIsNone(self.svc.scope_key())
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request("second")
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(self.peer.offers, [])
        self.assertEqual(self.peer.other, [])
        self.assertEqual(self.svc.offer_stats()["store"], False)
        self.engine.release.set()
        self.join(first, second)
        for response in (first, second):
            self.assertEqual(response[1].get("status"), 200, response[1])


class FrontierOrder(unittest.TestCase):
    """Preparation follows the admissions the selector would make, in its own order - not arrival order."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.link = conversation_offers.ConversationOfferLink()
        self.addCleanup(self.link.close)
        self.peer = NativePeer(self.link)
        self.link.start()
        self.peer.start()
        self.svc = Service(PlainEngine(self.tok, link=self.link), self.tok, ChatTemplate(TEMPLATE))

    def queue(self, text, priority="foreground"):
        """One request accepted and waiting exactly as the POST dispatch leaves it: registered and scoped, no work."""
        req = {"messages": [{"role": "user", "content": text}], "strata_priority": priority}
        req["_strata_cache_scope"] = self.svc.cache_scope(req, "tenant")
        return self.svc.register_lifecycle(req, threading.Event()), req

    def prepare(self, record, req) -> None:
        self.svc.prefetch_offer(record, "/v1/chat/completions", req)

    def test_only_the_selector_frontier_is_prepared_and_in_its_order(self):
        for i in range(3):                      # three real foreground admissions: no background one yet
            record, _req = self.queue(f"an earlier foreground turn {i}")
            self.svc.acquire_request(record)    # admitted at once (the slot is free)
            self.svc.finish_lifecycle(record)
        self.assertEqual(self.svc.admission_foreground[False], 3)
        head, head_req = self.queue("the foreground turn", priority="foreground")
        behind, behind_req = self.queue("a later foreground turn", priority="foreground")
        background, background_req = self.queue("the background turn", priority="background")
        self.assertEqual([r["priority"] for r in self.svc.selector_order(False, self.svc.admission_waiters, 3, 3)],
                         ["background", "foreground", "foreground"])
        self.prepare(head, head_req)            # the oldest waiter, and not the selector's next admission
        self.prepare(behind, behind_req)        # a later one: behind the frontier either way
        self.prepare(background, background_req)
        self.peer.drain(0.4)
        offered = self.peer.offers
        self.assertEqual(len(offered), 1, "the selector's frontier was not the only request prepared")
        self.assertIn("the background turn", prompt_text(offered[0]["tokens"]))
        self.assertIsNone(head.get("_prepared"))
        self.assertIsNone(behind.get("_prepared"))
        self.assertIsNotNone(background.get("_prepared"))
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 1)
        self.assertEqual([r["priority"] for r in self.svc.selector_order(True, self.svc.admission_waiters, 1, 1)], [])

    def test_later_background_frontier_revokes_an_earlier_foreground_offer(self):
        for i in range(3):
            record, _req = self.queue(f"priming foreground {i}")
            self.svc.acquire_request(record)
            self.svc.finish_lifecycle(record)
        foreground, foreground_req = self.queue("initial foreground")
        self.prepare(foreground, foreground_req)
        first = self.peer.next_offer(5)
        self.assertIsNotNone(first)
        background, background_req = self.queue("new background", priority="background")
        self.prepare(foreground, foreground_req)
        self.assertEqual(self.peer.next_cancel(5), (first["request_id"], first["generation"]))
        self.assertNotIn("_prepared", foreground)
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 0)
        self.prepare(background, background_req)
        next_offer = self.peer.next_offer(5)
        self.assertIsNotNone(next_offer)
        self.assertIn("new background", prompt_text(next_offer["tokens"]))
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 1)


class BoundedPreparation(ChannelTestCase):
    """What the frontier will and will not prepare: a prompt it cannot build inside the allowance is declined (its
    demand still renders, reports and runs it), the request it declined is not prepared again while it waits, a
    changed steering withdraws what was prepared, and the preparation never rewrites the request's own pin."""

    def test_an_over_context_prompt_is_never_prepared_or_offered(self):
        self.engine.max_context = 512                   # fits the active request, but not the queued prompt
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        prepared_patch = mock.patch.object(self.svc, "freeze_generation", wraps=self.svc.freeze_generation)
        frozen = prepared_patch.start()
        self.addCleanup(prepared_patch.stop)
        waited = mock.patch.object(self.svc, "prefetch_offer", wraps=self.svc.prefetch_offer)
        attempts = waited.start()
        self.addCleanup(waited.stop)
        second = self.start_request(body={"messages": [{"role": "user", "content": "x" * 1024}], "max_tokens": 4})
        self.wait_queued(1)
        self.peer.drain(0.8)                            # several admission waits, each with its own preparation
        self.assertEqual(self.peer.offers, [], "a prompt that cannot be prepared was offered anyway")
        self.assertGreaterEqual(attempts.call_count, 2, "the queued request was never given another wait")
        self.assertEqual(frozen.call_count, 1, "a refusal that cannot change was prepared again on every wait")
        record = next(r for r in self.svc.lifecycles if r["state"] == "queued")
        self.assertTrue(record.get("_offer_declined"))
        self.assertIsNone(record.get("_prepared"))
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 0, "a declined preparation kept its bytes")
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 400)   # the demand path still reports it in its own words
        self.assertIn("exceeds the context window", json.loads(second[1]["body"])["error"]["message"])

    def test_a_request_past_the_input_bound_is_declined_before_it_is_even_resolved(self):
        """The raw request's own JSON is admitted first: over the allowance nothing resolves, renders or is offered
        for it, and its demand still resolves, renders, runs and answers it (a preparation refusal only)."""
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        bound = mock.patch.object(self.svc, "preparation_input_chars", return_value=8)
        bound.start()
        self.addCleanup(bound.stop)
        resolving = mock.patch.object(self.svc, "resolve_for_offer", wraps=self.svc.resolve_for_offer)
        resolve = resolving.start()
        self.addCleanup(resolving.stop)
        rendered = mock.patch.object(self.svc, "render_prompt", wraps=self.svc.render_prompt)
        render = rendered.start()
        self.addCleanup(rendered.stop)
        second = self.start_request(body={"messages": [{"role": "user", "content": "y" * 400}], "max_tokens": 4})
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(resolve.call_count, 0, "a request over the input bound was resolved for preparation")
        self.assertEqual(render.call_count, 0, "the template rendered a request the preparation had not admitted")
        self.assertEqual(self.peer.offers, [])
        record = next(r for r in self.svc.lifecycles if r["state"] == "queued")
        self.assertTrue(record.get("_offer_declined"))
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)   # preparation is the only thing that was declined
        self.assertEqual(render.call_count, 1, "the demand did not render the request it had to run")
        self.assertIn("y" * 400, prompt_text(self.engine.prompts[1]))
        self.assertNotIn("cache_offer", self.keys_of(self.engine.samplings[1]))

    def test_a_prompt_the_bounded_render_cannot_produce_is_declined_and_its_demand_still_runs_it(self):
        """The render itself is bounded: a prompt the collector cannot hold is declined as preparation (never
        truncated), and the demand renders it whole and runs it."""
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        bound = mock.patch.object(self.svc, "preparation_render_chars", return_value=64)
        bound.start()
        self.addCleanup(bound.stop)
        collecting = mock.patch.object(self.svc.template, "render_within",
                                       wraps=self.svc.template.render_within)
        collector = collecting.start()
        self.addCleanup(collecting.stop)
        second = self.start_request(body={"messages": [{"role": "user", "content": "z" * 400}], "max_tokens": 4})
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(collector.call_count, 1, "the preparation did not render through the bounded collector")
        self.assertEqual(self.peer.offers, [], "a prompt the bound could not hold was offered anyway")
        record = next(r for r in self.svc.lifecycles if r["state"] == "queued")
        self.assertTrue(record.get("_offer_declined"))
        self.assertEqual(self.svc.offer_stats()["prepared_requests"], 0)
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)
        self.assertEqual(collector.call_count, 1, "the demand rendered through the preparation's own bound")
        self.assertIn("z" * 400, prompt_text(self.engine.prompts[1]), "the demand ran a shortened prompt")

    def test_a_changed_steering_withdraws_what_was_prepared(self):
        self.engine.info = {"cvec": 1}                   # a loaded control vector: its flag is what native prices
        self.svc.set_shared({"experimental_speed_projection": False})
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request(body={"messages": [{"role": "user", "content": "a turn"}], "max_tokens": 4})
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered, "the queued request was not prepared")
        assert offered is not None
        self.assertFalse(offered["want_cvec"], "the offer did not declare the steering in force when it was made")
        self.svc.set_shared({})                          # the settings change while the request waits
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)
        self.assertEqual(self.peer.next_cancel(5), (offered["request_id"], offered["generation"]),
                         "the demand kept an offer validated under another steering")
        keys = self.keys_of(self.engine.samplings[1])
        self.assertNotIn("cache_offer", keys)            # the store's candidate was validated under the old flag
        self.assertNotIn("cvec=", keys)                  # and the demand runs its own steering, not the frozen one

    def test_the_preparation_never_rewrites_the_requests_own_pin_specification(self):
        """The pin is resolved on a request-local copy: after the demand re-resolves to another prompt (the shared
        settings changed), it resolves the client's own specification, not the boundary preparation computed."""
        body = {"messages": [{"role": "user", "content": "a shared document " * 20},
                             {"role": "user", "content": "the question"}],
                "strata_prefix": {"messages": 1}}
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request(body=body)
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered, "the pinned request was not prepared")
        queued = next(r for r in self.svc.lifecycles if r["state"] == "queued")
        offered_pin = dict(queued["_prepared"].prefix)
        self.svc.set_shared({"reasoning_effort": "low", "max_tokens": 6})
        fresh = self.svc.resolve_openai(json.loads(json.dumps(body)), load=False)
        self.svc.prepare(fresh.messages, fresh.tools, fresh.kw, fresh.max_new_requested,
                         force=fresh.force, req=fresh.req)
        expected_pin = fresh.req["strata_prefix"]
        self.assertNotEqual(expected_pin, offered_pin, "the changed prompt must move the pin boundary")
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)
        pinned = self.engine.samplings[1].get("strata_prefix")
        self.assertEqual(pinned, expected_pin, "demand reused the old prompt's pin after preparation was invalidated")


class PreparedDialects(ChannelTestCase):
    """Both other text endpoints prepare exactly as the chat route does: the queued request's own tokens are
    offered, and its demand runs those tokens with the offer bound to the generation."""

    def prepared(self, body, path):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request(body=body, path=path)
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered, f"a queued {path} request was never prepared")
        assert offered is not None
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)
        self.assertEqual(self.engine.prompts[1], list(offered["tokens"]), "a prepared prompt was not the one run")
        self.assertIn(f"cache_offer={offered['request_id']}", self.keys_of(self.engine.samplings[1]))
        return offered

    def test_a_queued_messages_request_is_prepared_and_its_demand_binds_the_offer(self):
        offered = self.prepared({"model": "m", "messages": [{"role": "user", "content": "an anthropic turn"}],
                                 "max_tokens": 4}, "/v1/messages")
        self.assertIn("an anthropic turn", prompt_text(offered["tokens"]))

    def test_a_queued_responses_request_is_prepared_and_its_demand_binds_the_offer(self):
        offered = self.prepared({"input": "a responses turn", "max_output_tokens": 4}, "/v1/responses")
        self.assertIn("a responses turn", prompt_text(offered["tokens"]))


class UnofferedDeadline(ChannelTestCase):
    """A request the store is never offered still carries its original lifecycle deadline to the engine."""
    metadata_bytes = 4          # the store states a budget no token payload fits: nothing is prepared or offered

    def test_an_unoffered_request_reaches_the_engine_with_its_original_deadline(self):
        first = self.start_request("first")                 # admitted at once
        self.assertTrue(self.engine.entered.wait(5))
        second = self.start_request("second")               # queued behind it: reserved, refused, unoffered
        self.wait_queued(1)
        self.peer.drain(0.5)
        self.assertEqual(self.peer.offers, [])
        self.assertGreaterEqual(self.svc.offer_stats()["preparation_refused"], 1,
                                "the store's budget never refused the reservation")
        record = next(r for r in self.svc.lifecycles if r["state"] == "queued")
        self.assertIsNone(record.get("_prepared"))
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual([o.get("status") for _t, o in (first, second)], [200, 200])
        for i, sampling in enumerate(self.engine.samplings):
            keys = self.keys_of(sampling)
            found = re.search(r" cache_deadline_ns=(\d+)", keys)
            self.assertIsNotNone(found, f"request {i} reached the engine without its deadline")
            self.assertGreater(int(found.group(1)), 0)
        queued = self.keys_of(self.engine.samplings[1])
        original = int(record["_deadline"] * 1_000_000_000)
        self.assertEqual(int(re.search(r" cache_deadline_ns=(\d+)", queued).group(1)), original,
                         "the deadline the engine saw was not the one the request was accepted with")
        self.assertNotIn("cache_offer", self.keys_of(self.engine.samplings[1]))


class SharedQueuedPrefix(ChannelTestCase):
    """Shared-prefix derivation is identical on the bounded frontier and on an unoffered direct demand."""

    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tokenizer_dir = Path(directory.name)
        for name in server.TOKENIZER_FILES:
            (self.tokenizer_dir / name).write_text("test identity", encoding="utf-8")
        self.svc.tokenizer_dir = self.tokenizer_dir
        self.engine.cache_model_id = "ab" * 32
        self.svc.api_key = "client-key"
        self.headers = {"Authorization": "Bearer client-key"}
        self.principal = ("principal", "client-key", server.credential_fingerprint("client-key"))

    def test_prepared_and_direct_demands_keep_the_same_shared_scope_and_distinct_private_owners(self):
        body = {"messages": [{"role": "user", "content": "a shared document " * 20},
                             {"role": "user", "content": "the question"}],
                "strata_conversation_id": "A", "strata_shared_prefix": {"messages": 1},
                "strata_prefix": {"messages": 1}, "max_tokens": 4}
        resolved = self.svc.resolve_openai(dict(body), load=False)
        self.svc.lifecycle_local.record = {"_cache_principal": self.principal}
        direct_ids, _, _ = self.svc.prepare(resolved.messages, resolved.tools, resolved.kw, resolved.max_new_requested,
                                           force=resolved.force, req=resolved.req)
        expected_scope = resolved.req["_strata_cache_shared_scope"]
        expected_pin = resolved.req["strata_prefix"]
        self.svc.lifecycle_local.record = None
        with mock.patch.object(self.svc, "encode_prompt", wraps=self.svc.encode_prompt) as encoded:
            first = self.start_request("first", headers=self.headers)
            self.assertTrue(self.engine.entered.wait(5))
            second = self.start_request(body=body, headers=self.headers)
            self.wait_queued(1)
            offered = self.peer.next_offer(5)
            self.assertIsNotNone(offered)
            self.engine.release.set()
            self.join(first, second)
            self.assertEqual(second[1].get("status"), 200, second[1])
            self.assertEqual(encoded.call_count, 2, "a prepared shared demand rendered its full prompt again")
        sampling = self.engine.samplings[1]
        self.assertEqual(self.engine.prompts[1], direct_ids)
        self.assertEqual(sampling["_strata_cache_shared_scope"], expected_scope)
        self.assertEqual(sampling["strata_prefix"], expected_pin)
        self.assertEqual(sampling["strata_shared_prefix"], expected_pin)
        self.assertEqual(offered["tokens"], direct_ids)
        self.assertEqual(offered["scope"].hex(), sampling["_strata_cache_scope"])
        self.assertEqual(offered["request_id"], sampling["_strata_cache_offer"])
        self.assertEqual(offered["deadline_ns"], sampling["_strata_cache_deadline_ns"])
        self.assertIn("sscope=" + expected_scope, self.keys_of(sampling))
        another = {**body, "strata_conversation_id": "B",
                   "messages": [body["messages"][0], {"role": "user", "content": "a different question"}]}
        third = self.start_request(body=another, headers=self.headers)
        self.join(third)
        self.assertEqual(third[1].get("status"), 200, third[1])
        direct = self.engine.samplings[2]
        self.assertEqual(direct["_strata_cache_shared_scope"], expected_scope)
        self.assertNotEqual(direct["_strata_cache_scope"], sampling["_strata_cache_scope"])
        self.assertNotIn("_strata_cache_offer", direct)
        self.assertGreater(direct["_strata_cache_deadline_ns"], 0)
        self.assertEqual(self.peer.other, [], "sharing must not extend the private offer wire protocol")

    def test_an_effort_fallback_is_never_frozen_for_a_shared_demand(self):
        path = self.tokenizer_dir / "fallback.jinja"
        path.write_text("{{ 'generation' if add_generation_prompt else 'history' }}"
                        "{% for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")
        self.svc.template = ChatTemplate(path)
        self.svc.effort_end = True
        body = {"messages": [{"role": "user", "content": "a document and a question"}],
                "strata_shared_prefix": {"tokens": 2}, "max_tokens": 4, "reasoning_effort": "none"}
        resolved = self.svc.resolve_openai(body, load=False)
        self.svc.lifecycle_local.record = {"_cache_principal": self.principal}
        self.assertIsNone(self.svc.freeze_generation(resolved))
        self.assertTrue(self.svc.lifecycle_local.render_fallback)
        self.assertNotIn("_strata_cache_shared_scope", resolved.req)
        self.svc.prepare(resolved.messages, resolved.tools, resolved.kw, resolved.max_new_requested,
                         force=resolved.force, req=resolved.req)
        self.assertNotIn("_strata_cache_shared_scope", resolved.req)
        self.svc.lifecycle_local.record = None


class Ancillary(unittest.TestCase):
    """What the channel accepts from the engine: no descriptor at all, a whole descriptor array, or nothing."""

    def setUp(self):
        self.link = conversation_offers.ConversationOfferLink()
        self.addCleanup(self.link.close)
        self.link.start()
        sock = self.link.child_socket
        assert sock is not None
        self.sock = sock
        sock.send(hello_payload(Hello(generation=5, frontier=1, metadata_bytes=1 << 20, enabled=True, issuer_key=KEY)))
        self.assertTrue(self.link.wait_hello(5), "the engine's own HELLO was not read")

    @staticmethod
    def open_fds() -> int:
        return len(os.listdir("/proc/self/fd"))

    def wait_for(self, key, count, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.link.stats()[key] >= count:
                return
            time.sleep(0.01)
        self.fail(f"never saw {key} reach {count} (got {self.link.stats()[key]})")

    def send_with(self, fds) -> None:
        array_ = array.array("i", [fd for fd in fds])
        self.sock.sendmsg([b"\0" * HEADER.size], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array_)])

    def test_a_descriptor_from_the_engine_is_refused_closed_and_counted(self):
        before = self.open_fds()
        fd = os.open(__file__, os.O_RDONLY)
        try:
            self.send_with([fd])
        finally:
            os.close(fd)
        self.wait_for("received_fds", 1)
        self.wait_for("malformed", 1)
        self.assertIsNone(self.link.stats()["reader_error"])
        self.assertLessEqual(self.open_fds(), before, "a descriptor the channel refused was left open")

    def test_a_truncated_descriptor_array_is_refused_without_leaking_any(self):
        before = self.open_fds()
        fds = [os.open(__file__, os.O_RDONLY) for _ in range(2 * conversation_offers.FD_SPACE)]
        try:
            self.send_with(fds)                              # more than a whole control buffer: the kernel cuts it
        finally:
            for fd in fds:
                os.close(fd)
        self.wait_for("malformed", 1)
        self.assertIsNone(self.link.stats()["reader_error"])
        self.assertLessEqual(self.open_fds(), before, "a truncated descriptor array leaked descriptors")


if __name__ == "__main__":
    unittest.main()
