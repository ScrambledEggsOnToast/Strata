"""The frontend's half of the private conversation-offer channel: stable scopes, the queued frontier's preparation
and the transport it is offered on.

    python -m unittest serve.test_conversation_offers -v

Nothing here needs a GPU or the native engine: the native side of the channel is played by this test's own socket
peer (a HELLO, then whatever the frontend sends), and the engine by a scripted stand-in that can hold its first
requests open so a real queue forms.  One test does spawn a real child process, to prove the descriptor and the
environment the engine inherits are the ones this channel needs.
"""
import array
import fcntl
import http.client
import json
import os
from pathlib import Path
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


class OtherTokenizer(ByteTokenizer):
    """The byte tokenizer under another name: a different tokenizer identity (Service.render_identity)."""


class NativePeer:
    """The native side of one link, as these tests speak it: it answers the HELLO and reads what is offered."""

    def __init__(self, link, *, generation=41, frontier=1, metadata_bytes=1 << 20, enabled=True, key=KEY):
        assert link.child_socket is not None
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
                                "seals": fcntl.fcntl(fd, fcntl.F_GET_SEALS),
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
        self.engine = QueuedEngine(self.tok, link=self.link, batch=self.batch, hold=self.hold)
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

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

    def service(self, *, key: bytes | None = KEY, tok=None, template=None, model="m", tokenizer_dir=None):
        link = None
        if key is not None:
            link = conversation_offers.ConversationOfferLink()
            self.addCleanup(link.close)
            peer = NativePeer(link, key=key)
            link.start()
            peer.start()
        engine = PlainEngine(tok or ByteTokenizer(), link=link)
        return Service(engine, engine.tok, template or ChatTemplate(TEMPLATE), model_name=model,
                       tokenizer_dir=tokenizer_dir)

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
        self.assertNotEqual(baseline, self.service(model="other-model").cache_scope(request, "tenant-a"))
        self.assertNotEqual(baseline, self.service(template=self.other_template()).cache_scope(request, "tenant-a"))
        self.assertNotEqual(baseline, self.service(tok=OtherTokenizer()).cache_scope(request, "tenant-a"))

    def test_the_tokenizer_artifacts_are_part_of_the_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vocab.json").write_text('{"a": 0}', encoding="utf-8")
            first = self.service(tokenizer_dir=root).cache_scope({"messages": []}, "tenant-a")
            (root / "vocab.json").write_text('{"a": 0, "b": 1}', encoding="utf-8")
            second = self.service(tokenizer_dir=root).cache_scope({"messages": []}, "tenant-a")
        self.assertNotEqual(first, second)

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
        for seal in (fcntl.F_SEAL_SHRINK, fcntl.F_SEAL_GROW, fcntl.F_SEAL_WRITE, fcntl.F_SEAL_SEAL):
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


class QueuedFrontier(ChannelTestCase):
    """What the real production path does with queued, varied prompts."""

    def test_the_frontier_is_prepared_once_and_the_demand_binds_that_offer(self):
        rendered = mock.patch.object(self.svc, "encode_prompt", wraps=self.svc.encode_prompt)
        spy = rendered.start()
        self.addCleanup(rendered.stop)
        first = self.start_request("first")
        self.assertTrue(self.engine.entered.wait(5), "the first request never reached the engine")
        second = self.start_request("second")
        self.wait_queued(1)
        offered = self.peer.next_offer(5)
        self.assertIsNotNone(offered, "the queue head was never prepared and offered")
        third = self.start_request("three")
        self.wait_queued(2)
        self.peer.drain(0.6)
        self.assertEqual(len(self.peer.offers), 1, "a request behind the frontier was prepared too")
        self.assertEqual(self.peer.cancels, [])
        self.assertEqual(self.peer.other, [])
        self.engine.release.set()
        self.join(first, second, third)
        self.assertEqual([outcome.get("status") for _t, outcome in (first, second, third)], [200, 200, 200])
        self.assertEqual(len(self.engine.prompts), 3)
        self.assertEqual(self.engine.prompts[1], list(self.peer.offers[0]["tokens"]))   # consumed, not re-rendered
        self.assertEqual(self.engine.prompts[2], list(self.peer.offers[1]["tokens"]))   # the next head, in turn
        bound = StrataEngine.sampling_keys(self.engine.samplings[1])
        self.assertIn(f"scope={self.engine.samplings[1]['_strata_cache_scope']}", bound)
        self.assertIn(f"cache_offer={self.peer.offers[0]['request_id']}"
                      f" cache_generation={self.peer.offers[0]['generation']}", bound)
        self.assertNotIn("cache_offer", StrataEngine.sampling_keys({"temperature": 0.5}))
        self.assertEqual(spy.call_count, 3, "a prepared prompt was rendered again")
        self.assertEqual([c[0] for c in self.peer.cancels], [o["request_id"] for o in self.peer.offers])

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
        self.assertTrue(self.svc.cancel_request(queued[0]["id"]))
        self.join(second)
        self.assertEqual(second[1].get("status"), 409)
        self.assertEqual(self.peer.next_cancel(5), (offered["request_id"], offered["generation"]))
        self.peer.drain(0.4)
        self.assertEqual(len(self.peer.cancels), 1, "a retired request left another offer outstanding")
        self.engine.release.set()
        self.join(first)
        self.assertEqual(first[1].get("status"), 200)

    def test_a_prepared_result_is_consumed_once_and_only_for_its_own_prompt(self):
        generation = conversation_offers.generation_digest([], None, {}, 4, None)
        prepared = conversation_offers.PreparedGeneration(ids=(1, 2, 3), thinking=True, max_new=4, digest=generation,
                                                          ctx=4096, link=self.link, offer_id=9,
                                                          generation=self.link.epoch())
        self.assertIsNone(self.svc.take_prepared(None, [], None, {}, 4, None))
        record = {"_prepared": prepared}
        self.assertIs(self.svc.take_prepared(record, [], None, {}, 4, None), prepared)     # this exact prompt
        self.assertIsNone(self.svc.take_prepared(record, [{"role": "user"}], None, {}, 4, None))   # once only
        request: dict = {}
        self.svc.bind_offer(request, prepared)
        self.assertEqual(request.get("_strata_cache_offer"), 9)
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
        self.svc.withdraw_offer({"_prepared": prepared})
        self.assertTrue(prepared.withdrawn)
        self.assertIsNone(self.svc.withdraw_offer({"_prepared": prepared}))
        self.assertIsNone(self.svc.withdraw_offer(None))

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


class EngineSpawn(unittest.TestCase):
    """The engine really inherits this incarnation's descriptor, and really answers on it."""

    ENGINE = r'''
import json, os, socket, struct, sys
from pathlib import Path
fd = int(os.environ["STRATA_CONVERSATION_OFFER_FD"])
out = Path(sys.argv[sys.argv.index("--report") + 1])
with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET, fileno=fd) as s:
    s.send(struct.pack("<4sHHQQQII32s", b"SCC1", 1, 1, 0, 77, 1 << 20, 2, 1, bytes(range(32))))
    out.write_text(json.dumps({"fd": fd, "env": os.environ["STRATA_CONVERSATION_OFFER_FD"]}))
print("READY 4096 stop", flush=True)
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
'''

    def test_the_engine_inherits_the_descriptor_and_its_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "engine.py"
            script.write_text("#!" + sys.executable + "\n" + self.ENGINE)
            script.chmod(0o700)
            report = root / "report.json"
            engine = StrataEngine(str(script), ["--report", str(report)])
            self.addCleanup(engine.close)
            channel = engine.offers
            assert channel is not None
            self.assertTrue(channel.wait_hello(10), "the engine's HELLO never arrived")
            self.assertTrue(channel.store_enabled)
            self.assertEqual(channel.epoch(), 77)
            self.assertEqual(channel.frontier(), 2)
            self.assertEqual(channel.scope_key(), bytes(range(32)))
            facts = json.loads(report.read_text())
            self.assertEqual(facts["fd"], channel.child_fd)
            self.assertEqual(facts["env"], str(channel.child_fd))
            before = len(os.listdir("/proc/self/fd"))
            engine.close()
            self.assertIsNone(engine.offers)
            self.assertLessEqual(len(os.listdir("/proc/self/fd")), before)


if __name__ == "__main__":
    unittest.main()
