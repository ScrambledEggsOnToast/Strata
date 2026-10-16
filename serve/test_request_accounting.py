"""Request accounting across thinking-budget continuations, through the HTTP APIs (no GPU or model pack).

    python -m unittest serve.test_request_accounting -v
"""
import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, request_timings, serve
from serve.test_server import ThinkingEngine


class ClockedThinkingEngine(ThinkingEngine):
    """The native DONE contract: closing a generation drains its statistics before the next one starts."""
    REUSED = 5

    def __init__(self, tok):
        super().__init__(tok)
        self.attempts = 0
        self.done = []
        self.fail_continuation = False
        self.cancel_continuation = False
        self.unknown_initial_drafts = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.attempts += 1
        if self.fail_continuation and self.attempts == 2:
            raise EngineDied("the continuation ended without a DONE line")
        first = self.attempts == 1
        generated = 0
        stopped = False
        try:
            for token in super().generate(ids, max_new, sampling, cancel, embeddings):
                generated += 1
                stopped = token in self.tok.encode("<|im_end|>", parse_special=True)
                yield token
                if self.cancel_continuation and not first and generated == 3:
                    cancel.set()
        finally:
            # The continuation reuses the original prompt plus the generated thinking.
            reused = self.REUSED if first else len(self.prompts[0]) + self.done[0]["generated"]
            self.last = dict(prompt_tokens=len(ids), generated=generated, reused=reused,
                             prompt_read=len(ids) - reused, finish="stop" if stopped else "cancel",
                             prompt_ms=40.0 if first else 7.0, decode_ms=200.0 if first else 80.0,
                             drafts_offered=18 if first else 11, drafts_accepted=8 if first else 7,
                             hits=9 if first else 7, lookups=10 if first else 8,
                             ram_blobs=1 if first else 3, file_blobs=2 if first else 4,
                             file_mb=1.5 if first else 2.5)
            if first and self.unknown_initial_drafts:
                self.last.pop("drafts_offered")
                self.last.pop("drafts_accepted")
            self.done.append(dict(self.last))


class RequestAccounting(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ClockedThinkingEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body else None,
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read().decode()
            return response.status, raw

    def chat(self, path="/v1/chat/completions", stream=False, budget=20):
        code, raw = self.request(path, {"model": "m", "messages": [{"role": "user", "content": "2+2?"}],
                                       "max_tokens": 400, "reasoning_budget_tokens": budget, "stream": stream})
        self.assertEqual(code, 200, raw)
        if stream:
            return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        return json.loads(raw)

    def assert_original_input(self, usage):
        original = len(self.engine.prompts[0])
        self.assertEqual(self.engine.attempts, 2)
        self.assertGreater(self.engine.done[1]["reused"], original, "exercise reuse beyond the API input")
        self.assertEqual(usage["prompt_tokens"], original)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], self.engine.REUSED)
        self.assertEqual(usage["total_tokens"], original + usage["completion_tokens"])

    def assert_request_work(self, timings, completion_tokens, finish="stop"):
        original = len(self.engine.prompts[0])
        self.assertEqual(len(self.engine.done), 2)
        native_generated = sum(done["generated"] for done in self.engine.done)
        self.assertEqual((timings["cache_n"], timings["prompt_n"]), (5, original - 5))
        self.assertEqual((timings["prompt_ms"], timings["predicted_ms"]), (47.0, 280.0))
        self.assertEqual(timings["predicted_n"], completion_tokens)
        self.assertGreater(completion_tokens, native_generated, "wrap-up tokens count as output, not native decode")
        self.assertEqual(timings["predicted_per_second"], round(native_generated / 0.280, 1))
        self.assertEqual(timings["predicted_per_token_ms"], round(280.0 / native_generated, 3))
        self.assertEqual((timings["draft_n"], timings["draft_n_accepted"]), (29, 15))
        code, raw = self.request("/metrics")
        self.assertEqual(code, 200)
        metrics = json.loads(raw)
        row, totals = metrics["requests"][0], metrics["totals"]
        self.assertEqual(row["finish"], finish)
        self.assertEqual((row["prompt_tokens"], row["reused"], row["prompt_read"]), (original, 5, original - 5))
        self.assertEqual(row["engine_generated"], native_generated)
        self.assertEqual(row["decode_tok_s"], timings["predicted_per_second"])
        self.assertEqual((row["prompt_ms"], row["decode_ms"]), (47.0, 280.0))
        self.assertEqual((row["ram_blobs"], row["file_blobs"], row["file_mb"]), (4, 6, 4.0))
        self.assertAlmostEqual(row["hit_rate"], 16 / 18, places=3)
        self.assertEqual((totals["requests"], totals["prompt_tokens"], totals["reused"]), (1, original, 5))
        self.assertEqual((totals["prompt_ms"], totals["decode_ms"]), (47.0, 280.0))
        self.assertEqual((totals["drafts_offered"], totals["drafts_accepted"]), (29, 15))
        code, raw = self.request("/v1/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)["last_timings"]["draft_n"], 29)

    def test_openai_nonstream_keeps_original_cache_and_all_segment_work(self):
        response = self.chat()
        self.assertEqual(response["choices"][0]["message"]["content"], self.engine.ANSWER)
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assert_original_input(response["usage"])
        self.assert_request_work(response["timings"], response["usage"]["completion_tokens"])

    def test_continuations_emit_one_response_attribution_record(self):
        prefix = "[strata] response_attribution: "
        with contextlib.redirect_stdout(io.StringIO()) as log:
            response = self.chat()
            deadline = time.monotonic() + 2
            while prefix not in log.getvalue() and time.monotonic() < deadline:
                time.sleep(0.005)
        records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                   if line.startswith(prefix)]
        self.assertEqual(len(records), 1, log.getvalue())
        record = records[0]
        self.assertEqual(record["request_id"], self.svc.lifecycle_records()[0]["id"])
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (15, 29))
        self.assertEqual(len(record["legs"]), 2)
        self.assertEqual([leg["drafts_offered"] for leg in record["legs"]], [18, 11])
        self.assertEqual(record["output_tokens"], response["usage"]["completion_tokens"])
        self.assertLess(record["committed_tokens"], record["output_tokens"],
                        "frontend wrap-up tokens must not become native commits")

    def test_unknown_earlier_drafts_cannot_become_a_known_partial_response_total(self):
        self.engine.unknown_initial_drafts = True
        with contextlib.redirect_stdout(io.StringIO()) as log:
            response = self.chat()
            deadline = time.monotonic() + 2
            while "response_attribution: " not in log.getvalue() and time.monotonic() < deadline:
                time.sleep(.005)
        self.assertNotIn("draft_n", response["timings"])
        self.assertNotIn("draft_n_accepted", response["timings"])
        record, = [json.loads(line.split("response_attribution: ", 1)[1])
                   for line in log.getvalue().splitlines() if "response_attribution: " in line]
        self.assertIsNone(record["drafts_offered"])
        self.assertIsNone(record["drafts_accepted"])
        self.assertEqual([leg["drafts_offered"] for leg in record["legs"]], [None, 11])

    def test_openai_stream_keeps_original_cache_and_all_segment_work(self):
        events = self.chat(stream=True)
        final = events[-1]
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assert_original_input(final["usage"])
        self.assert_request_work(final["timings"], final["usage"]["completion_tokens"])

    def test_cancelled_continuation_keeps_original_input_and_counts_each_done_once(self):
        self.engine.cancel_continuation = True
        events = self.chat(stream=True)
        final = events[-1]
        content = "".join(event["choices"][0]["delta"].get("content", "") for event in events)
        self.assertEqual(content, "The")
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertEqual([done["generated"] for done in self.engine.done], [20, 3])
        self.assert_original_input(final["usage"])
        self.assert_request_work(final["timings"], final["usage"]["completion_tokens"], finish="cancel")

    def test_anthropic_nonstream_classifies_only_original_input_as_cached(self):
        usage = self.chat("/v1/messages")["usage"]
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(usage["cache_read_input_tokens"], 5)
        self.assertEqual(usage["input_tokens"], len(self.engine.prompts[0]) - 5)

    def test_anthropic_stream_classifies_only_original_input_as_cached(self):
        events = self.chat("/v1/messages", stream=True)
        usage = next(event["usage"] for event in events if event["type"] == "message_delta")
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(usage["cache_read_input_tokens"], 5)
        self.assertEqual(usage["input_tokens"], len(self.engine.prompts[0]) - 5)

    def test_single_segment_keeps_its_own_usage_and_timings(self):
        response = self.chat(budget=10000)
        self.assertEqual(self.engine.attempts, 1)
        self.assertEqual(response["usage"]["prompt_tokens_details"]["cached_tokens"], 5)
        timings = response["timings"]
        self.assertEqual((timings["prompt_ms"], timings["predicted_ms"]), (40.0, 200.0))
        self.assertEqual((timings["draft_n"], timings["draft_n_accepted"]), (18, 8))

    def test_failed_continuation_does_not_count_the_first_done_twice(self):
        self.engine.fail_continuation = True
        events = self.chat(stream=True)
        self.assertTrue(any("error" in event for event in events))
        self.assertEqual(self.engine.attempts, 2)
        self.assertEqual(len(self.engine.done), 1)
        _, raw = self.request("/metrics")
        metrics = json.loads(raw)
        row = metrics["requests"][0]
        self.assertEqual(row["finish"], "error")
        self.assertEqual(row["engine_generated"], 20)
        self.assertEqual((row["prompt_ms"], row["decode_ms"]), (40.0, 200.0))
        self.assertEqual((metrics["totals"]["drafts_offered"], metrics["totals"]["drafts_accepted"]), (18, 8))


class PressureHistory(unittest.TestCase):
    def test_finished_history_and_cache_events_join_the_same_completed_sample(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "Hello.", max_context=4096)
        svc = Service(engine, tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        with contextlib.redirect_stdout(io.StringIO()):
            list(svc.run(tok.encode("hi"), False, None, 20, {}, threading.Event()))
        self.assertIsNone(svc.history[-1]["pressure_seq"])  # telemetry remains optional
        reader = mock.Mock(spec=["ok", "name", "read", "close"])
        reader.ok.return_value = False
        with mock.patch("serve.telemetry.gpu_reader", return_value=reader), \
                mock.patch.object(threading.Thread, "start"):
            svc.start_telemetry()
        sampler = svc.telemetry
        self.addCleanup(sampler._readers.close)  # no background thread was started
        cases = [(99.0, True), (101.0, True), (98.99, False), (101.01, False), (None, False)]
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("", encoding="utf-8")
            engine.log_path, engine.log_start = str(log), 0
            previous_seqs = []
            for sampled_at, fresh in cases:
                with self.subTest(sampled_at=sampled_at):
                    def extra():
                        if sampled_at is None:
                            raise RuntimeError("no pressure payload")
                        return {"pressure": {"boot_id": svc.lifecycle_boot, "sampled_at_unix_s": sampled_at}}

                    sampler.extra = extra
                    sampler._stop.clear()
                    with mock.patch("serve.server.time.time", return_value=100.0), \
                            mock.patch.object(sampler._stop, "wait", side_effect=lambda _: sampler._stop.set()), \
                            contextlib.redirect_stdout(io.StringIO()):
                        sampler._loop()
                        seq = sampler.snapshot()["now"]["seq"]
                        list(svc.run(tok.encode("hi"), False, None, 20, {}, threading.Event()))
                        with log.open("a", encoding="utf-8") as f:
                            f.write("strata serve: conversation cache: restored 2 tokens (live) in 3.5 ms; "
                                    "parked=0 bytes=0\n")
                        metrics = svc.metrics()
                    expected = seq if fresh else None
                    previous_seqs.append(expected)
                    self.assertEqual(svc.history[-1]["pressure_seq"], expected)
                    self.assertEqual(metrics["requests"][0]["pressure_seq"], expected)
                    event = metrics["conversation_cache"]["events"][-1]
                    self.assertEqual(event["pressure_seq"], expected)
                    self.assertEqual((event["restore_kind"], event["duration_ms"]), ("live", 3.5))
                    self.assertEqual([row["pressure_seq"] for row in list(svc.history)[1:]], previous_seqs)
                    self.assertTrue(all(row["finish"] == "stop" for row in svc.history))


class SlotLegAccounting(unittest.TestCase):
    """Prompt timing uses real original-input read counts, including a held-slot prefix."""

    def test_prompt_n_is_what_was_read(self):
        # the slot already held the whole prompt: 1 token read, the conversation cache off (reused 0)
        last = {"prompt_ms": 563.5, "decode_ms": 700.0, "generated": 30, "reused": 0, "prompt_read": 1, "finish": "stop"}
        t = request_timings(43086, 30, last)
        self.assertEqual((t["prompt_n"], t["cache_n"]), (1, 43085))
        self.assertEqual(t["prompt_n"] + t["cache_n"], 43086)
        self.assertLess(t["prompt_per_second"], 10)
        # a cold read, and an older engine without the field: unchanged
        cold = request_timings(2052, 10, {**last, "prompt_read": 2052})
        self.assertEqual((cold["prompt_n"], cold["cache_n"]), (2052, 0))
        old = request_timings(2052, 10, {k: v for k, v in last.items() if k != "prompt_read"})
        self.assertEqual((old["prompt_n"], old["cache_n"]), (2052, 0))
        # the conversation cache's own count still counts when larger
        both = request_timings(1000, 10, {**last, "reused": 900, "prompt_read": 100})
        self.assertEqual((both["prompt_n"], both["cache_n"]), (100, 900))


if __name__ == "__main__":
    unittest.main()
