"""Data-parallel replicas (serve/replicas.py): the config, the router, and the real Service over HTTP with two fake
engines (the fake of test_parallel.py) behind a ReplicaEngine."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import conversation_offers
from serve.conversation_offers import Hello
from serve.frontend import ChatTemplate
from serve.replicas import BLOCK, ReplicaConfigError, ReplicaEngine, Router, cpu_sets, gpu_groups, replica_cfg
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH


class Config(unittest.TestCase):
    def test_a_number_cuts_the_gpu_list(self):
        self.assertEqual(gpu_groups({"gpu": [0, 1, 2, 3], "replicas": 2}), [[0, 1], [2, 3]])
        self.assertEqual(gpu_groups({"gpu": "0,1,2,3", "replicas": 4}), [[0], [1], [2], [3]])
        self.assertEqual(gpu_groups({"gpu": [0, 1], "replicas": 1}), [])
        self.assertEqual(gpu_groups({"gpu": [0, 1]}), [])

    def test_named_groups(self):
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}, {"gpus": [2]}]}), [[0, 1], [2]])
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}]}), [])

    def test_mistakes_are_said(self):
        for cfg in ({"gpu": [0, 1, 2], "replicas": 2}, {"replicas": 2}, {"replicas": "2"},
                    {"replicas": [{"gpus": [0, 1]}, {"gpus": [1, 2]}]}, {"replicas": [{"gpus": []}, {"gpus": [1]}]}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                gpu_groups(cfg)

    def test_replica_cfg(self):
        cfg = {"gpu": [0, 1, 2, 3], "args": ["--x"], "log": "/l/strata.log", "hip_ordinal": 1, "env": {"A": "1"},
               "expert_profile_save": "/p/prof.bin",
               "replicas": [{"gpus": [0, 1], "env": {"B": 2}}, {"gpus": [2, 3], "args": ["--y"]}]}
        a, b = replica_cfg(cfg, 0, [0, 1]), replica_cfg(cfg, 1, [2, 3])
        self.assertEqual((a["gpu"], b["gpu"]), ([0, 1], [2, 3]))
        self.assertEqual((a["log"], b["log"]), ("/l/strata.r0.log", "/l/strata.r1.log"))
        self.assertEqual(a["env"], {"A": "1", "B": "2"})
        self.assertEqual(b["args"], ["--x", "--y"])
        self.assertNotIn("replicas", a)
        self.assertNotIn("hip_ordinal", a)
        self.assertEqual(a["expert_profile_save"], "/p/prof.r0.bin")
        self.assertEqual(cfg["args"], ["--x"])                       # the original is untouched
        self.assertEqual(replica_cfg({"gpu": [0, 1], "args": []}, 0, [3])["gpu"], 3)

    def test_cpu_sets_off(self):
        self.assertIsNone(cpu_sets(2, enabled=False))
        self.assertIsNone(cpu_sets(1))


class Routing(unittest.TestCase):
    def test_matched_prefix_per_replica(self):
        r = Router(3)
        a = list(range(BLOCK * 4))
        r.note(0, a)
        r.note(1, a[:BLOCK * 2])
        self.assertEqual(r.matched(a + [9] * 70), [BLOCK * 4, BLOCK * 2, 0])
        self.assertEqual(r.matched([7] * 200), [0, 0, 0])
        r.forget(0)
        self.assertEqual(r.matched(a), [0, BLOCK * 2, 0])

    def test_the_index_is_bounded(self):
        r = Router(2, cap=10)
        for i in range(50):
            r.note(i % 2, [i] * (BLOCK * 2))
        self.assertLessEqual(len(r.blocks), 10)


class FakeEngine:
    def __init__(self, n=4):
        self.batch, self.up = n, True
        self.max_context = 4096
        self.info = {}
        self.last = {}

    def alive(self):
        return self.up


class RouteChoice(unittest.TestCase):
    def test_least_loaded_then_affinity(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        picks = [e.route([i] * 100) for i in range(4)]             # nothing cached: spread
        self.assertEqual(sorted(picks), [0, 0, 1, 1])
        ids = list(range(BLOCK * 8))
        e.inflight = [0, 0]
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids + [5] * 10), k)                # the conversation goes back
        e.inflight = [0, 0]
        e.inflight[k] = 1                                           # 512 tokens held < 3000 per running request
        self.assertNotEqual(e.route(ids + [5] * 10), k)

    def test_a_long_prefix_outweighs_load(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        ids = list(range(BLOCK * 200))                              # 12,800 tokens held on replica 0
        e.router.note(0, ids)
        e.inflight = [2, 0]
        self.assertEqual(e.route(ids + [1]), 0)
        e.inflight = [5, 0]
        self.assertEqual(e.route(ids + [1]), 1)

    def test_a_dead_replica_is_skipped(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        ids = list(range(BLOCK * 3))
        e.router.note(0, ids)
        a.up = False
        self.assertEqual(e.route(ids), 1)
        b.up = False
        self.assertEqual(e.route(ids), -1)

    def test_nonfinite_forgets_the_replica(self):
        """#879: after DONE ... nonfinite the engine holds no conversation: the router forgets that replica."""
        from serve.server import EngineNonFinite

        class NonFinite(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                raise EngineNonFinite("the engine's logits became non-finite (#879)")

        e = ReplicaEngine([NonFinite(), NonFinite()])
        other, ids = [9] * (BLOCK * 4), list(range(BLOCK * 3))
        e.router.note(0, other)
        e.router.note(1, other)
        k = 0
        e.inflight[k] += 1                                          # as route() counts it
        with mock.patch.object(e, "route", return_value=k):
            with self.assertRaises(EngineNonFinite):
                list(e.generate(ids, 10, {}, threading.Event()))
        self.assertEqual(e.router.matched(ids)[k], 0)
        self.assertEqual(e.router.matched(other)[k], 0)             # its older conversations too
        self.assertEqual(e.router.matched(other)[1 - k], len(other))
        self.assertEqual(e.inflight[k], 0)

    def test_batch_is_the_sum(self):
        self.assertEqual(ReplicaEngine([FakeEngine(4), FakeEngine(4)]).batch, 8)
        self.assertEqual(ReplicaEngine([FakeEngine(0), FakeEngine(0)]).batch, 2)


class OfferOwnership(unittest.TestCase):
    """A prepared offer belongs to the replica whose store validated it.  Preparation happens before routing, so the
    keys a prepared generation carries can name another replica than the one the router picks: the *recipient* must
    not be handed an offer it never made, and the offering store must not keep a frontier slot for that demand."""

    class Engine:
        """A replica that records what it was asked to run, with its own offer channel."""

        def __init__(self, link=None):
            self.offers, self.batch, self.max_context, self.info = link, 0, 4096, {}
            self.samplings = []

        def alive(self):
            return True

        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            self.samplings.append(dict(sampling))
            yield 1

    def link(self):
        """A link whose store is enabled, as the engine's own HELLO says at startup."""
        link = conversation_offers.ConversationOfferLink()
        self.addCleanup(link.close)
        link.start()
        sock = link.child_socket
        assert sock is not None
        sock.send(conversation_offers.hello_payload(Hello(generation=7, frontier=1, metadata_bytes=1 << 20,
                                                          enabled=True, issuer_key=bytes(range(32)))))
        self.assertTrue(link.wait_hello(5), "the store's own HELLO was not read")
        return link

    def prepared_sampling(self, owner):
        """What Service.bind_offer leaves on a generation after this replica's store took its offer."""
        return {"_strata_cache_scope": "cd" * 32, "_strata_cache_offer": 3, "_strata_cache_generation": owner.epoch(),
                "_strata_cache_offer_owner": owner}

    def test_another_recipient_never_runs_an_offer_it_did_not_make(self):
        owner, other = self.link(), self.link()
        first, second = self.Engine(owner), self.Engine(other)
        engine = ReplicaEngine([first, second])
        self.assertTrue(owner.offer(3, owner.epoch(), 0, "ab" * 32, [1, 2, 3]))
        sampling = self.prepared_sampling(owner)
        engine.inflight[1] += 1                                         # as route() counts it (route is patched)
        with mock.patch.object(engine, "route", return_value=1):        # the demand lands on the other replica
            list(engine.generate([1, 2, 3], 4, sampling, threading.Event()))
        self.assertEqual(second.samplings, [{"_strata_cache_scope": "cd" * 32}],
                         "a replica was handed an offer another replica's store validated")
        self.assertEqual(first.samplings, [])
        self.assertEqual(owner.stats()["cancelled"], 1, "the offering store still holds the offer")
        self.assertEqual(owner.stats()["outstanding"], 0)
        self.assertEqual(other.stats()["cancelled"], 0)
        self.assertEqual(sampling["_strata_cache_offer"], 3, "the caller's own request was rewritten")

    def test_the_offering_replica_keeps_its_own_offer(self):
        owner, other = self.link(), self.link()
        first, second = self.Engine(owner), self.Engine(other)
        engine = ReplicaEngine([first, second])
        self.assertTrue(owner.offer(3, owner.epoch(), 0, "ab" * 32, [1, 2, 3]))
        sampling = self.prepared_sampling(owner)
        engine.inflight[0] += 1                                         # as route() counts it (route is patched)
        with mock.patch.object(engine, "route", return_value=0):        # the demand lands on the offering replica
            list(engine.generate([1, 2, 3], 4, sampling, threading.Event()))
        self.assertEqual(first.samplings[0]["_strata_cache_offer"], 3)
        self.assertIs(first.samplings[0]["_strata_cache_offer_owner"], owner)
        self.assertEqual(owner.stats()["cancelled"], 0)
        self.assertEqual(owner.stats()["outstanding"], 1)
        self.assertEqual(second.samplings, [])


class ReplicaService(unittest.TestCase):
    def start(self, slots=4, n=2):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        self.logs = [Path(self.tmp.name) / f"r{i}.log" for i in range(n)]
        engines = []
        for i in range(n):
            extra = (["--batch", str(slots)] if slots else []) + ["--log", str(self.logs[i])]
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engines.append(StrataEngine("strata", extra))
        self.engine = ReplicaEngine(engines)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
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

    def post(self, messages, max_tokens=64):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def chat(self, text, max_tokens=64):
        return self.post([{"role": "user", "content": text}], max_tokens)

    def served(self):
        return [sum(1 for l in p.read_text().splitlines() if l.split()[0] in ("GEN", "BGEN")) if p.exists() else 0
                for p in self.logs]

    def burst(self, prompts):
        out = [None] * len(prompts)

        def one(i):
            out[i] = self.chat(prompts[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_requests_spread_over_both_and_answers_are_the_solo_answer(self):
        self.start()
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(6)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 2 for g in got), got)
        self.assertEqual(set(out), {solo})

    def test_metrics_and_status_see_all_slots(self):
        self.start()
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 8)
        m = self.svc.metrics()
        self.assertEqual(len(m["live"]["slots"]), 8)
        self.assertEqual([r["replica"] for r in m["live"]["replicas"]], [0, 1])
        self.assertEqual(m["engine"]["replicas"], 2)

    def test_engines_without_slots_run_one_each_at_once(self):
        self.start(slots=0)
        out = self.burst(["a" * 40, "b" * 40])
        self.assertEqual(self.served(), [1, 1])
        self.assertEqual(len(set(out)), 1)

    def test_a_conversation_goes_back_to_its_replica(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first)
        k = self.served().index(1)
        self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                   {"role": "user", "content": "and then?"}], 16)
        self.assertEqual(self.served()[k], 2)

    def test_a_replica_that_died_is_skipped(self):
        self.start()
        self.engine.heal = lambda: None
        self.engine.engines[0].proc.kill()
        self.engine.engines[0].proc.wait()
        self.assertTrue(self.engine.alive())
        for i in range(3):
            self.assertIn("ok, done.", self.chat(f"after the loss {i}"))
        self.assertEqual(self.served()[1], 3)


class ReplicaClose(unittest.TestCase):
    """One shutdown budget covers every replica: the wrapper hands each engine what is *left* of it."""

    class Engine:
        def __init__(self, delay=0.0):
            self.delay = delay
            self.budgets: list[float | None] = []

        def alive(self):
            return False

        def close(self, timeout_s=None):
            self.budgets.append(timeout_s)
            if self.delay:
                time.sleep(self.delay)

    def test_every_replica_is_closed_within_one_total_budget(self):
        engines = [self.Engine(), self.Engine(delay=0.4), self.Engine()]
        engine = ReplicaEngine(engines, load_tokens=3000)
        start = time.monotonic()
        engine.close(timeout_s=5.0)
        elapsed = time.monotonic() - start
        self.assertEqual([len(e.budgets) for e in engines], [1, 1, 1], "a replica was not asked to end")
        for e in engines:
            self.assertIsNotNone(e.budgets[0], "a replica was closed without the shutdown's own budget")
            self.assertLessEqual(e.budgets[0], 5.0)
        self.assertLess(engines[-1].budgets[0], engines[0].budgets[0], "each replica got the whole budget again")
        self.assertLess(elapsed, 1.0, "the shutdown waited for more than the replicas' own work")

    def test_a_replica_that_fails_does_not_leave_the_others_running(self):
        class Broken(self.Engine):
            def close(self, timeout_s=None):
                super().close(timeout_s)
                raise RuntimeError("this replica cannot be reached")

        engines = [Broken(), self.Engine(), self.Engine()]
        engine = ReplicaEngine(engines, load_tokens=3000)
        with self.assertRaises(RuntimeError):
            engine.close(timeout_s=2.0)
        self.assertEqual([len(e.budgets) for e in engines], [1, 1, 1])
        for e in engines[1:]:
            self.assertIsNotNone(e.budgets[0])


if __name__ == "__main__":
    unittest.main()
