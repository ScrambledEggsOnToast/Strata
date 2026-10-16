"""CPU-only wire and HTTP proof; no model, native process or GPU is started."""
import json
import queue
import threading
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from serve.throughput import Throughput
from serve.replicas import ReplicaEngine
from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, StrataEngine, serve


class ThroughputTests(unittest.TestCase):
    def test_concurrent_decode_cached_prefill_and_expiry(self):
        now = [10.0]
        t = Throughput(clock=lambda: now[0])
        t.wire('INFO throughput_progress=1')
        t.wire('RESUME 1000')
        t.wire('PP 1100 1300 100 1000 100')
        t.wire('BT 0 42')
        t.wire('BT 1 43')
        t.wire('PP 1300 1300 300 1000 300')
        t.wire('PP 1300 1300 300 1000 300') # duplicate is not more work
        self.assertEqual(t.snapshot(),dict(window_s=5.0,decode_tok_s=.4,prefill_tok_s=60))
        # A yielded/new leg restarts its fresh count, not the aggregate window.
        t.wire('RESUME 1300')
        t.wire('PP 1400 1400 100 1000 100')
        self.assertEqual(t.snapshot()['prefill_tok_s'],80)
        # A real re-read can have a large reused prefix but fresh work starts at 0.
        t.wire('RESUME 1000')
        t.wire('PP 100 1300 100 1000 100')
        self.assertEqual(t.snapshot()['prefill_tok_s'],100)
        now[0]=15.1
        self.assertEqual(t.snapshot(),dict(window_s=5.0,decode_tok_s=0,prefill_tok_s=0))

    def test_terminal_tail_and_cancelled_work_count_once(self):
        t=Throughput(True)
        t.wire('RESUME 900')
        t.wire('PP 1000 1100 100 1000 100')
        done='DONE 0 1100 100 0 cancel 0 0 900 0 0 0 0 0 150'
        t.wire('INFO fresh_prompt_tokens=150')
        self.assertEqual(t.snapshot()['prefill_tok_s'],30)
        t.wire(done);t.wire(done)
        self.assertEqual(t.snapshot()['prefill_tok_s'],30)
        t.wire('RESUME 2000')
        t.wire('INFO fresh_prompt_tokens=1')
        self.assertEqual(t.snapshot()['prefill_tok_s'],30.2)
        t.wire('DONE 1 2001 1 1 stop 0 0 2000 0 0 0 0 0 1')
        self.assertEqual(t.snapshot()['prefill_tok_s'],30.2)

    def test_reread_ignores_legacy_done_total(self):
        t=Throughput(True)
        t.wire('RESUME 1000')
        t.wire('PP 1299 1300 100 1000 1299')
        t.wire('INFO fresh_prompt_tokens=1300')
        self.assertEqual(t.snapshot()['prefill_tok_s'],260)
        t.wire('DONE 1 1300 100 20 stop 0 0 1000 0 0 0 0 0 300')
        self.assertEqual(t.snapshot()['prefill_tok_s'],260)

    def test_replica_sum_and_unknown_member(self):
        a,b=Throughput(True),Throughput(True)
        a.wire('T 1');b.wire('BT 0 2')
        r=ReplicaEngine.__new__(ReplicaEngine)
        r.engines=[SimpleNamespace(throughput=t,alive=lambda: True) for t in (a,b)]
        self.assertEqual(r.snapshot()['decode_tok_s'],.4)
        b.prefill_supported=False
        self.assertIsNone(r.snapshot()['prefill_tok_s'])
        r.engines[1].alive=lambda: False
        self.assertIsNone(r.snapshot()['decode_tok_s'])

    def test_unknown_protocol_and_restart(self):
        t=Throughput()
        t.wire('T 5')
        t.wire('RESUME 100')
        t.wire('PP 200 200 100 1000')
        self.assertIsNone(t.snapshot()['prefill_tok_s'])
        self.assertEqual(t.snapshot()['decode_tok_s'],.2)
        self.assertEqual(Throughput().snapshot()['decode_tok_s'],0)
        t.wire('PP bad data')
        t.wire('T nonsense')
        self.assertEqual(t.snapshot()['decode_tok_s'],.2)

    def test_native_pump_captures_drained_work_and_preserves_routing(self):
        e=StrataEngine.__new__(StrataEngine)
        wire=['INFO throughput_progress=1\n','RESUME 90\n','PP 100 100 1 10000 10\n','BT 0 7\n','T 8\n','DONE cancelled\n']
        e.proc=SimpleNamespace(stdout=iter(wire))
        e.lines=queue.Queue();e.slot_q=[queue.Queue()];e.slot_cv=threading.Condition()
        e._pump()
        self.assertTrue(e.ended)
        self.assertEqual(e.throughput.snapshot()['decode_tok_s'],.4)
        self.assertEqual(e.throughput.snapshot()['prefill_tok_s'],2)
        self.assertEqual(e.slot_q[0].get(),'BT 0 7\n')
        self.assertEqual([e.lines.get() for _ in range(5)],wire[:3]+wire[4:])

    def test_http_during_newer_prefill_reports_both_rates(self):
        tok=ByteTokenizer();engine=MockEngine(tok,'answer',max_context=4096)
        engine.throughput=Throughput(True)
        engine.throughput.wire('RESUME 100')
        engine.throughput.wire('PP 600 600 100 5000 500')
        engine.throughput.wire('BT 0 42')
        svc=Service(engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        svc.status.update(busy=True,first_token=None,started=0)
        httpd=serve(svc,port=0)
        try:
            with urllib.request.urlopen('http://127.0.0.1:%s/metrics'%httpd.server_address[1]) as r:
                body=json.load(r)
            self.assertEqual(body['live']['state'],'reading')
            self.assertIsNone(body['live']['tok_s']) # legacy field stays compatible
            self.assertEqual(body['throughput']['decode_tok_s'],.2)
            self.assertEqual(body['throughput']['prefill_tok_s'],100)
        finally:
            httpd.shutdown();httpd.server_close()

if __name__=='__main__': unittest.main()
