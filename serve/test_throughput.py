import tempfile
import sys
"""CPU-only wire and HTTP proof; no model, native process or GPU is started."""
from unittest.mock import patch
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
    def test_idle_window_does_not_change_active_window(self):
        now = [0.0]
        t = Throughput(True, clock=lambda:now[0])
        t.wire('T 1')
        self.assertEqual(t.snapshot(idle=True)['window_s'], 2.0)
        self.assertEqual(t.snapshot(idle=True)['decode_tok_s'], .5)
        now[0] = 2.1
        self.assertEqual(t.snapshot(idle=True)['decode_tok_s'], 0)
        t.wire('T 2')
        self.assertEqual(t.snapshot()['decode_tok_s'], .4)
        now[0] = 5.1
        self.assertEqual(t.snapshot()['decode_tok_s'], .2)
        self.assertIsNone(Throughput().snapshot(idle=True)['prefill_tok_s'])

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

    def test_tagged_owners_count_fresh_rows_and_final_prompt_token_independently(self):
        counter = Throughput(True)
        counter.wire("BACK first 0 11 1")
        counter.wire("BACK second 1 22 1")
        counter.wire("BQPP first 0 11 1 1020 1100 20")
        counter.wire("BQPP second 1 22 1 5 6 5")
        counter.wire("BQPP first 0 11 1 1020 1100 20")
        counter.wire("BQT second 1 22 1 9")
        counter.wire("BQDONE second 1 22 1 1 6 1 1 length 0 0 0 prompt_read=6")
        counter.wire("BQDONE first 0 11 1 0 1100 1 0 cancel 0 0 1000 prompt_read=20")
        self.assertAlmostEqual(counter.snapshot()["prefill_tok_s"], 26 / 5)
        self.assertAlmostEqual(counter.snapshot()["decode_tok_s"], 1 / 5)
        self.assertEqual(counter.packed_fresh, {})
        counter.wire("BACK successor 0 33 2")
        counter.wire("BQT successor 0 11 1 9")
        self.assertAlmostEqual(counter.snapshot()["decode_tok_s"], 1 / 5)
        self.assertIsNone(counter.snapshot()["prefill_tok_s"])

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

    def test_startup_capability_reaches_runtime_counter(self):
        import serve.server as server
        from unittest.mock import patch
        peer = "import sys\nprint('INFO throughput_progress=1', flush=True)\nprint('READY 4096 stop', flush=True)\nfor line in sys.stdin:\n if line.strip() == 'QUIT': break\n"
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'peer.py'
            script.write_text(peer)
            popen = server.subprocess.Popen
            with patch.object(server.subprocess, 'Popen', lambda cmd, **kw: popen([sys.executable, str(script)], **kw)):
                engine = StrataEngine('fixture', [])
            try:
                self.assertTrue(engine.alive())
                self.assertEqual(engine.throughput.snapshot()['prefill_tok_s'], 0)
            finally:
                engine.close()
            # A new process without the announcement must not inherit capability.
            script.write_text(peer.replace("print('INFO throughput_progress=1', flush=True)\n", ''))
            with patch.object(server.subprocess, 'Popen', lambda cmd, **kw: popen([sys.executable, str(script)], **kw)):
                engine.__init__('fixture', [])
            try:
                self.assertIsNone(engine.throughput.snapshot()['prefill_tok_s'])
            finally:
                engine.close()

    def test_idle_replica_capability_before_first_request(self):
        busy, unused = Throughput(), Throughput()
        for counter in (busy, unused):
            counter.wire('INFO throughput_progress=1')
            counter.wire('READY 4096 stop')
        busy.wire('RESUME 0')
        busy.wire('PP 100 100 100 1000 100')
        busy.wire('BT 0 42')
        replicas = ReplicaEngine.__new__(ReplicaEngine)
        replicas.engines = [SimpleNamespace(throughput=t, alive=lambda: True) for t in (busy, unused)]
        self.assertEqual(replicas.snapshot()['prefill_tok_s'], 20)
        self.assertEqual(replicas.snapshot()['decode_tok_s'], .2)
        # A genuinely uninstrumented member must still leave the total unknown.
        unused.wire('RESUME 0')
        unused.wire('PP 10 10 10 1000')
        self.assertIsNone(replicas.snapshot()['prefill_tok_s'])

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

    def test_request_activity_bound_does_not_limit_active_records(self):
        tok=ByteTokenizer(); engine=MockEngine(tok,'answer',max_context=4096)
        svc=Service(engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        with svc.status_lock:
            svc.admission_active={str(i):{'id':format(i,'032x'),'state':'generating','_executing':True} for i in range(200)}
            snapshot=svc._request_activity()
            self.assertEqual(len(snapshot['records']),128)
            self.assertTrue(snapshot['truncated'])
            self.assertEqual(len(svc.admission_active),200)

    def test_partial_handler_failures_are_not_completed(self):
        tok=ByteTokenizer(); engine=MockEngine(tok,'answer',max_context=4096)
        svc=Service(engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        httpd=serve(svc,port=0)
        httpd.handle_error=lambda *args:None
        try:
            for exception in (TypeError,RuntimeError):
                def broken(handler, request):
                    handler.answer_started=True
                    raise exception('fixture failure')
                with patch.object(httpd.RequestHandlerClass,'_openai',broken):
                    request=urllib.request.Request('http://127.0.0.1:%s/v1/chat/completions'%httpd.server_address[1],
                        data=b'{"messages":[{"role":"user","content":"fixture"}]}',headers={'Content-Type':'application/json'})
                    with self.assertRaises(Exception):
                        urllib.request.urlopen(request).read()
                with svc.status_lock:
                    self.assertEqual(svc._request_activity()['records'][-1]['state'],'failed')
        finally:
            httpd.shutdown();httpd.server_close()

    def test_request_activity_is_content_free_and_tracks_outcomes(self):
        tok=ByteTokenizer(); engine=MockEngine(tok,'answer',max_context=4096)
        svc=Service(engine,tok,ChatTemplate(Path(__file__).with_name('chat_template.jinja')))
        record=svc.register_lifecycle({},threading.Event())
        with svc.status_lock:
            self.assertEqual(svc._request_activity()['records'][0]['state'],'queued')
        svc.acquire_request(record)
        svc.lifecycle_note(record,'prefill',_executing=True,prompt='must not escape',error_detail='private')
        with svc.status_lock:
            summary=svc._request_activity()
        self.assertEqual(summary['records'],[{'id':record['id'],'state':'processing'}])
        svc.finish_lifecycle(record)
        with svc.status_lock:
            self.assertEqual(svc._request_activity()['records'][0]['state'],'completed')
        for reason,expected in [('cancelled','cancelled'),('timeout','cancelled'),('error','failed')]:
            record=svc.register_lifecycle({},threading.Event())
            if reason=='error': record['error']='private failure'
            else: record['cancel_reason']=reason
            svc.finish_lifecycle(record)
            with svc.status_lock:
                item=next(r for r in svc._request_activity()['records'] if r['id']==record['id'])
            self.assertEqual(item['state'],expected)
        self.assertEqual(set(summary),{'boot_id','truncated','records'})

    @patch.dict("os.environ", {"STRATA_THROUGHPUT":"1"})
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
            self.assertEqual(body['request_activity']['records'],[])
            self.assertEqual(body['live']['state'],'reading')
            self.assertIsNone(body['live']['tok_s']) # legacy field stays compatible
            self.assertEqual(body['throughput']['decode_tok_s'],.2)
            self.assertEqual(body['throughput']['prefill_tok_s'],100)
        finally:
            httpd.shutdown();httpd.server_close()

if __name__=='__main__': unittest.main()
