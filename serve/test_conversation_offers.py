"""The frontend's half of the private conversation-offer channel: stable scopes, the queued frontier's preparation
and the transport it is offered on.

    python -m unittest serve.test_conversation_offers -v

Nothing here needs a GPU or the native engine: the native side of the channel is played by this test's own socket
peer (a HELLO, then whatever the frontend sends), and the engine by a scripted stand-in that can hold its first
requests open so a real queue forms.
"""
import array
import fcntl
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
    metadata_bytes = 1 << 20
    enabled = True
    key = KEY
    batch = 1
    hold = 1

    def setUp(self):
        self.tok = ByteTokenizer()
        self.link = conversation_offers.ConversationOfferLink()
        self.addCleanup(self.link.close)
        self.peer = NativePeer(self.link, frontier=self.frontier, metadata_bytes=self.metadata_bytes,
                               enabled=self.enabled, key=self.key)
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

    def start_request(self, text="hello", body=None, path="/v1/chat/completions"):
        """Send one request on its own thread; -> (thread, outcome dict).  A queued request never returns here."""
        outcome: dict = {}

        def run():
            conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=20)
            try:
                payload = body if body is not None else {"messages": [{"role": "user", "content": text}]}
                conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
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


class NoStoreFrontier(ChannelTestCase):
    """A store that keeps nothing (a disabled HELLO) offers nothing: the demand path stays exactly as it was."""

    enabled = False
    key = b"\0" * 32

    def test_no_offer_is_ever_published(self):
        self.assertFalse(self.svc.offers_ready())
        self.assertIsNone(self.svc.scope_key())
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        self.start_request("second")
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(self.peer.offers, [])
        self.assertEqual(self.peer.other, [])
        self.assertEqual(self.svc.offer_stats()["store"], False)
        self.engine.release.set()
        self.join(first)
        self.assertEqual(first[1].get("status"), 200)


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

    def test_a_prompt_past_the_text_bound_is_declined_before_the_template_renders_it(self):
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5))
        bound = mock.patch.object(self.svc, "preparation_text_bound", return_value=8)
        bound.start()
        self.addCleanup(bound.stop)
        rendered = mock.patch.object(self.svc, "render_prompt", wraps=self.svc.render_prompt)
        render = rendered.start()
        self.addCleanup(rendered.stop)
        second = self.start_request(body={"messages": [{"role": "user", "content": "y" * 400}], "max_tokens": 4})
        self.wait_queued(1)
        self.peer.drain(0.6)
        self.assertEqual(render.call_count, 0, "the template rendered a request the preparation had not admitted")
        self.assertEqual(self.peer.offers, [])
        self.engine.release.set()
        self.join(first, second)
        self.assertEqual(second[1].get("status"), 200)   # preparation is the only thing that was declined
        self.assertEqual(render.call_count, 1, "the demand did not render the request it had to run")
        self.assertIn("y" * 400, prompt_text(self.engine.prompts[1]))

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
