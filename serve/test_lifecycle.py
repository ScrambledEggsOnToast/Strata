"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import json
import io
import os
import subprocess
import time
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import AdmissionError, ByteTokenizer, EngineDied, EngineStuck, MockEngine, Service, StrataEngine, serve


class ReservedSoloSlotTests(unittest.TestCase):
    def test_reserved_capacity_predicate(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, None)
        svc.reserve_solo_slot = True
        svc.engine.info = {"packed_prefill": 1}
        ordinary = svc.register_lifecycle({}, threading.Event())
        solo = svc.register_lifecycle({"strata_packing": "solo"}, threading.Event())
        self.assertFalse(svc._capacity_allows(ordinary, 3, 0, 4),
                         "three ordinary reservations leave the fourth capacity unit for solo")
        self.assertTrue(svc._capacity_allows(solo, 3, 0, 4))
        self.assertTrue(svc._capacity_allows(ordinary, 2, 1, 4))
        self.assertTrue(svc._capacity_allows(ordinary, 3, 1, 4))
        self.assertFalse(svc._capacity_allows(ordinary, 4, 1, 4))
        self.assertFalse(svc._capacity_allows(solo, 4, 1, 4))
        svc.reserve_solo_slot = False
        self.assertTrue(svc._capacity_allows(ordinary, 3, 0, 4))
        svc.reserve_solo_slot = True
        svc.engine.info = {}
        self.assertTrue(svc._capacity_allows(ordinary, 3, 0, 4), "reservation is ineffective with packing off")
        svc.engine.info = {"packed_prefill": 1}
        self.assertTrue(svc._capacity_allows(ordinary, 0, 0, 1), "a single-slot engine keeps its existing rule")


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Lifecycle(unittest.TestCase):
    def test_close_cleans_up_after_a_broken_stdin_pipe(self):
        for running in (False, True):
            with self.subTest(running=running):
                engine = StrataEngine("missing-executable", [], lazy=True)
                read_fd, write_fd = os.pipe()
                stdin = os.fdopen(write_fd, "w")
                self.addCleanup(stdin.close)
                os.close(read_fd)                       # the engine's end of the pipe is already gone
                stdin.write("pending")                  # buffered: flush() and close() will hit the broken pipe
                proc = mock.Mock(stdin=stdin, stdout=io.StringIO())
                self.addCleanup(proc.stdout.close)
                proc.poll.side_effect = [None if running else 0, 0]
                engine.proc, engine.pump, engine.log = proc, mock.Mock(), io.StringIO()
                self.addCleanup(engine.log.close)
                engine.ended, engine.progress, engine.last = False, (1, 2), {"generated": 1}

                engine.close()

                self.assertTrue(stdin.closed)
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(engine.log.closed)
                engine.pump.join.assert_called_once_with(timeout=2)
                self.assertIsNone(engine.proc)
                self.assertTrue(engine.ended)
                self.assertIsNone(engine.progress)
                self.assertEqual(engine.last, {})
                if running:
                    proc.terminate.assert_called_once()
                    proc.wait.assert_called_once_with(timeout=20)
                else:
                    proc.terminate.assert_not_called()
                    proc.wait.assert_not_called()
                proc.kill.assert_not_called()
                engine.close()                          # repeated cleanup is harmless

    def test_close_sends_eof_before_waiting_for_windows_reader(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = lambda **kwargs: self.assertTrue(proc.stdin.closed)
        engine.proc = proc
        engine.close()
        self.assertIsNone(engine.proc)
        proc.kill.assert_not_called()

    def test_close_does_not_forget_a_process_still_exiting(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 2)
        engine.proc = proc
        with self.assertRaisesRegex(EngineStuck, "still releasing"):
            engine.close()
        self.assertIs(engine.proc, proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_close_terminates_before_it_kills(self):
        # an engine that does not end on QUIT is terminated first (as the unload always did); kill is the last resort
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", 20), 0]
        engine.proc = proc
        engine.close()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_native_lazy_constructor_does_not_start_a_process(self):
        engine = StrataEngine("missing-executable", ["--max-context", "16384"], lazy=True)
        self.assertFalse(engine.alive())
        self.assertEqual(engine.max_context, 16384)
        self.assertIsNone(engine.exit_code())
        engine.close()

    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_lifecycle_event_cap_accounts_for_discarded_notes(self):
        record = self.svc.register_lifecycle({}, threading.Event())
        self.assertEqual(record["events_dropped"], 0)
        try:
            for i in range(127):
                self.svc.lifecycle_note(record, "queued", observation=i)
            full = self.svc.lifecycle_records(record["id"])
            self.assertEqual((len(full["events"]), full["events_dropped"]), (128, 0))
            for i in range(5):
                self.svc.lifecycle_note(record, "queued", observation=127 + i)
            capped = self.svc.lifecycle_records(record["id"])
            self.assertEqual((len(capped["events"]), capped["events_dropped"]), (128, 5))
            self.assertEqual(capped["events"], full["events"])
            self.assertEqual(capped["observation"], 131)  # the display cap cannot suppress state updates
        finally:
            self.svc.finish_lifecycle(record)
        finished = self.svc.lifecycle_records(record["id"])
        self.assertEqual((finished["state"], len(finished["events"]), finished["events_dropped"]),
                         ("completed", 128, 6))

    def test_discover_chat_unload_and_stream_reload(self):
        self.assertEqual(self.request("/api/health")[1]["service"], "strata")
        self.assertFalse(self.request("/v1/status")[1]["loaded"])
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertTrue(self.request("/props")[1]["models_autoload"])
        self.assertEqual(self.engine.starts, 0)
        chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}
        for _ in range(2):
            code, answer = self.request("/v1/chat/completions", chat)
            self.assertEqual(code, 200)
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(self.engine.starts, 1)
        self.assertFalse(self.request("/v1/unload", {})[1]["loaded"])
        code, stream = self.request("/v1/chat/completions", {**chat, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("data: [DONE]", stream)
        self.assertEqual(self.engine.starts, 2)

    def test_busy_auth_and_foreign_origin_cannot_unload(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/v1/unload", {})[0], 401)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret",
                                                        "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.engine.closes, 0)
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret"})[0], 200)

    def test_an_engine_that_died_on_load_is_reported_as_such(self):
        # EngineDied is a RuntimeError: it keeps its own answer ("the next request restarts it")
        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        code, body = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(code, 503)
        self.assertIn("the next request restarts it", body["error"]["message"])

    def test_explicit_load_and_model_matching(self):
        self.assertEqual(self.request("/v1/load", {"model": "other-model"})[0], 404)
        self.assertEqual(self.engine.starts, 0)
        before = time.time()
        code, status = self.request("/v1/load", {"model": self.svc.model})
        self.assertEqual(code, 200)
        self.assertTrue(status["loaded"])
        self.assertEqual(self.engine.starts, 1)
        self.assertGreaterEqual(self.svc.last_request_at, before)
        self.svc.status["queued"] = 1
        self.assertEqual(self.request("/v1/load", {})[0], 409)
        self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.assertEqual(self.engine.closes, 0)
        self.assertEqual(self.request("/v1/load", {}, {"Content-Type": "text/plain"})[0], 415)

    def test_every_json_route_enforces_the_request_body_budget(self):
        self.svc.request_body_bytes = 16
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses", "/settings", "/config"):
            with self.subTest(path=path):
                status, body = self.request(path, {"oversized": "x" * 40})
                self.assertEqual(status, 413)
                self.assertIn("request_body_bytes", body["error"]["message"])
        self.assertEqual(self.engine.starts, 0)
        self.assertEqual(self.svc.admission_waiters, [])


class RequestAdmission(unittest.TestCase):
    def service(self):
        tok = ByteTokenizer()
        return Service(MockEngine(tok, "ok"), tok, None)

    def test_frontend_partition_survives_trace_overflow_and_cancelled_reacquisition(self):
        svc = self.service()
        clock = [100.0]
        with mock.patch("serve.server.time.perf_counter", side_effect=lambda: clock[0]):
            record = svc.register_lifecycle({"strata_priority": "background"}, threading.Event())
            clock[0] += 2
            svc.acquire_request(record)
            self.assertEqual(record["admission_wait_s"], 2)
            clock[0] += 3
            for _ in range(140):
                svc.frontend_phase(record, "control_selector_wait")
                svc.lifecycle_note(record, "engine_wait")
                clock[0] += 1
                svc.frontend_phase(record, "physical_slot_wait")
                clock[0] += 2
                svc.frontend_phase(record, "control_occupied_wait")
                clock[0] += 3
                svc.native_submitted(record, 0)
                svc.lifecycle_note(record, "prefill")
                clock[0] += 4
            svc.frontend_phase(record, "control_selector_wait")
            clock[0] += 5
            record["cancel_reason"] = "cancelled"
            svc.finish_lifecycle(record)
        seconds, counts = record["frontend"]["seconds"], record["frontend"]["counts"]
        self.assertEqual(seconds["admission_wait"], 2)
        self.assertEqual(seconds["preparation"], 3)
        self.assertEqual(seconds["control_selector_wait"], 145)
        self.assertEqual(seconds["physical_slot_wait"], 280)
        self.assertEqual(seconds["control_occupied_wait"], 420)
        self.assertEqual(seconds["native_active"], 560)
        self.assertEqual(counts["control_selector_wait"], 141)
        self.assertEqual(sum(seconds.values()), record["completion_s"])
        self.assertEqual(record["finished_monotonic_s"], clock[0])
        self.assertIsNone(record["_frontend_phase"])
        self.assertEqual(record["state"], "cancelled")
        self.assertEqual(len(record["events"]), 128)
        self.assertGreater(record["events_dropped"], 128)
        self.assertEqual(len(record["native_submissions"]), 128)
        self.assertEqual(record["native_submissions_dropped"], 12)
        self.assertEqual(record["last_native_submission"]["submission_sequence"], 140)
        totals = svc.frontend_totals["background"]["cancelled"]
        self.assertEqual(totals["seconds"], seconds)
        self.assertEqual(totals["requests"], 1)
        with svc.status_lock:
            svc._finish_frontend(record, clock[0] + 100)
        self.assertEqual(totals["requests"], 1)

    def test_cleanup_and_first_observations_are_independent_of_native_timings(self):
        svc = self.service()
        clock = [10.0]
        with mock.patch("serve.server.time.perf_counter", side_effect=lambda: clock[0]):
            record = svc.register_lifecycle({}, threading.Event())
            svc.acquire_request(record)
            clock[0] = 11
            svc.frontend_phase(record, "native_active")
            record["first_token_s"], record["first_content_s"] = 1.5, 2.0
            clock[0] = 14
            svc.frontend_phase(record, "cleanup")
            clock[0] = 16
            svc.finish_lifecycle(record)
        self.assertEqual(record["frontend"]["seconds"]["cleanup"], 2)
        self.assertEqual(record["frontend"]["seconds"]["native_active"], 3)
        self.assertEqual(sum(record["frontend"]["seconds"].values()), 6)
        totals = svc.frontend_totals["foreground"]["completed"]
        self.assertEqual((totals["first_token_s"], totals["first_content_s"]), (1.5, 2.0))
        self.assertEqual((totals["first_token_count"], totals["first_content_count"]), (1, 1))

    def test_registration_refusals_close_after_http_response_and_emit_once(self):
        import contextlib
        for shutdown in (False, True):
            with self.subTest(shutdown=shutdown):
                tok = ByteTokenizer()
                svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
                svc.request_queue_limit = 0
                if not shutdown:
                    svc.register_lifecycle({}, threading.Event())
                svc.shutting_down = shutdown
                http = serve(svc, "127.0.0.1", 0)
                finished = threading.Event()
                emit = svc._emit_attribution
                def finalize(*args):
                    try:
                        emit(*args)
                    finally:
                        finished.set()
                try:
                    with contextlib.redirect_stdout(io.StringIO()) as log, \
                            mock.patch.object(svc, "_emit_attribution", side_effect=finalize):
                        request = urllib.request.Request(
                            f"http://127.0.0.1:{http.server_port}/v1/chat/completions",
                            json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                        "strata_priority": "background", "max_tokens": 1}).encode(),
                            {"Content-Type": "application/json"})
                        with self.assertRaises(urllib.error.HTTPError) as refused:
                            urllib.request.urlopen(request, timeout=2)
                        body = refused.exception.read()
                        refused.exception.close()
                        self.assertEqual(refused.exception.code, 503 if shutdown else 429, body)
                        self.assertTrue(finished.wait(2))
                        rejected = svc.lifecycles[-1]
                        self.assertEqual(refused.exception.headers["X-Strata-Request-Id"], rejected["id"])
                        svc._emit_attribution(rejected, "/v1/chat/completions")
                    records = [json.loads(line.split("response_attribution: ", 1)[1])
                               for line in log.getvalue().splitlines() if "response_attribution: " in line]
                    self.assertEqual(len(records), 1)
                    attribution = records[0]
                    self.assertEqual(attribution["request_id"], rejected["id"])
                    self.assertEqual(attribution["priority"], "background")
                    self.assertEqual(attribution["outcome"], "rejected")
                    self.assertEqual(attribution["rejection_reason"], "shutdown" if shutdown else "queue_full")
                    self.assertEqual((attribution["legs"], attribution["native_submission_count"]), ([], 0))
                    self.assertIsNone(attribution["committed_tokens"])
                    self.assertAlmostEqual(sum(attribution["frontend"]["seconds"].values()),
                                           attribution["interval"]["duration_s"])
                    self.assertEqual(svc.frontend_totals["background"]["rejected"]["requests"], 1)
                finally:
                    http.shutdown()
                    http.server_close()

    def test_real_control_reacquisitions_and_cancelled_wait_close_the_open_phase(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        engine.alive = lambda: True
        tok = ByteTokenizer()
        svc = Service(engine, tok, None)
        clock = [100.0]
        cancel = threading.Event()
        def consume(turn):
            while True:
                try:
                    next(turn)
                except StopIteration as done:
                    return done.value
        with mock.patch("serve.server.time.perf_counter", side_effect=lambda: clock[0]):
            record = svc.register_lifecycle({}, cancel)
            svc.acquire_request(record)
            for _ in range(2):
                self.assertTrue(consume(engine._take_control(cancel, 1,
                    on_phase=lambda phase: svc.frontend_phase(record, phase))))
                svc.native_submitted(record)
                clock[0] += 1
                engine._release_control()
            engine.ctl.acquire()
            def cancel_wait(timeout=None):
                clock[0] += 4
                cancel.set()
            with mock.patch.object(engine.slot_cv, "wait", side_effect=cancel_wait):
                self.assertFalse(consume(engine._take_control(cancel, 1,
                    on_phase=lambda phase: svc.frontend_phase(record, phase))))
            record["cancel_reason"] = "cancelled"
            svc.finish_lifecycle(record)
            engine.ctl.release()
        self.assertEqual(record["frontend"]["seconds"]["native_active"], 2)
        self.assertEqual(record["frontend"]["seconds"]["control_occupied_wait"], 4)
        self.assertEqual(record["completion_s"], 6)
        self.assertIsNone(record["_frontend_phase"])
        self.assertEqual(engine.waiting, 0)

    def test_drained_foreground_releases_native_deferral_but_not_http_admission(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        engine.alive = lambda: True
        engine.batch, engine.foreground_budget_s = 4, 1
        svc = Service(engine, ByteTokenizer(), None)
        foreground = svc.register_lifecycle({}, threading.Event())
        svc.acquire_request(foreground)
        def background_can_take_control():
            cancel = threading.Event()
            with mock.patch.object(engine.slot_cv, "wait", side_effect=lambda timeout=None: cancel.set()):
                turn = engine._take_control(cancel, 1, priority="background",
                                            background_due=time.monotonic() + 10,
                                            foreground_pending=lambda: svc._foreground_pending)
                with self.assertRaises(StopIteration) as finished:
                    next(turn)
            if finished.exception.value:
                engine._release_control()
            return finished.exception.value
        try:
            for state, expected in (("preparing", False), ("drained", True),
                                    ("preparing", False), ("error", True)):
                with self.subTest(state=state):
                    svc.lifecycle_note(foreground, state, _executing=False,
                                       error="failed" if state == "error" else None)
                    self.assertEqual(background_can_take_control(), expected)
                    self.assertIn(foreground["id"], svc.admission_active)
        finally:
            svc.finish_lifecycle(foreground)

    def test_priority_fairness_bounds_foreground_bypass(self):
        svc = self.service()
        records = []
        for priority in ("foreground", "foreground", "foreground", "foreground", "background"):
            records.append(svc.register_lifecycle({"strata_priority": priority}, threading.Event()))
        for index in (0, 1, 2, 4, 3):
            svc.acquire_request(records[index])
            self.assertEqual(list(svc.admission_active), [records[index]["id"]])
            svc.finish_lifecycle(records[index])
        self.assertEqual((svc.admission_active, svc.admission_waiters, svc.status["queued"]), ({}, [], 0))
        self.assertEqual([r["state"] for r in svc.lifecycle_records()], ["completed"] * 5)

    def test_queue_rejection_and_expired_waiter_do_not_take_engine_capacity(self):
        svc = self.service()
        svc.request_queue_limit = 1
        first = svc.register_lifecycle({}, threading.Event())
        svc.acquire_request(first)
        second = svc.register_lifecycle({}, threading.Event())
        with self.assertRaises(AdmissionError):
            svc.register_lifecycle({}, threading.Event())
        second["_deadline"] = 0
        with self.assertRaises(AdmissionError) as error:
            svc.acquire_request(second)
        self.assertEqual(error.exception.status, 408)
        svc.finish_lifecycle(second)
        self.assertEqual(list(svc.admission_active), [first["id"]])
        svc.finish_lifecycle(first)
        self.assertEqual((svc.admission_active, svc.admission_waiters, svc.status["queued"]), ({}, [], 0))
        self.assertEqual([r["state"] for r in svc.lifecycle_records()], ["completed", "timeout", "rejected"])

    def test_cancelled_equal_priority_waiter_leaves_fifo_without_consuming_slot(self):
        svc = self.service()
        first, cancelled, last = [svc.register_lifecycle({}, threading.Event()) for _ in range(3)]
        self.assertTrue(svc.cancel_request(cancelled["id"]))
        with self.assertRaises(AdmissionError) as error:
            svc.acquire_request(cancelled)
        self.assertEqual(error.exception.status, 409)
        svc.finish_lifecycle(cancelled)
        for record in (first, last):
            svc.acquire_request(record)
            svc.finish_lifecycle(record)
        self.assertEqual(svc.status["queued"], 0)
        public = svc.lifecycle_records(cancelled["id"])
        self.assertEqual(public["state"], "cancelled")
        self.assertFalse(any(key.startswith("_") for key in public))

    def test_score_admission_preserves_all_generation_reservations(self):
        svc = self.service()
        svc.engine.batch = 4
        svc.request_queue_limit = 0
        generation = [svc.register_lifecycle({}, threading.Event()) for _ in range(4)]
        for record in generation:
            svc.acquire_request(record)
        score = svc.register_lifecycle({}, threading.Event(), score=True)
        score["_deadline"] = time.perf_counter() + 0.25  # a regression fails, rather than hanging
        svc.acquire_request(score)
        snapshot = svc._admission_snapshot()
        self.assertEqual((snapshot["slots"], snapshot["active"], snapshot["score_active"]), (4, 5, 1))
        with self.assertRaises(AdmissionError):
            svc.register_lifecycle({}, threading.Event())
        svc.finish_lifecycle(generation.pop())
        successor = svc.register_lifecycle({}, threading.Event())
        successor["_deadline"] = time.perf_counter() + 0.25
        svc.acquire_request(successor)
        self.assertEqual(len(svc.admission_active), 5)
        for record in [*generation, successor, score]:
            svc.finish_lifecycle(record)

    def test_score_queue_is_bounded_and_cancelled_waiter_releases_its_place(self):
        svc = self.service()
        svc.engine.info = {"score_max_pending": 2}
        first = svc.register_lifecycle({}, threading.Event(), score=True)
        svc.acquire_request(first)
        cancelled = svc.register_lifecycle({}, threading.Event(), score=True)
        with self.assertRaises(AdmissionError):
            svc.register_lifecycle({}, threading.Event(), score=True)
        self.assertTrue(svc.cancel_request(cancelled["id"]))
        with self.assertRaises(AdmissionError) as error:
            svc.acquire_request(cancelled)
        self.assertEqual(error.exception.status, 409)
        svc.finish_lifecycle(cancelled)
        successor = svc.register_lifecycle({}, threading.Event(), score=True)
        svc.finish_lifecycle(first)
        svc.acquire_request(successor)
        svc.finish_lifecycle(successor)
        self.assertEqual((svc.admission_active, svc.admission_waiters, svc.status["queued"]), ({}, [], 0))

    def test_other_lane_cannot_reset_background_fairness(self):
        for score in (False, True):
            with self.subTest(score=score):
                svc = self.service()
                records = [svc.register_lifecycle({"strata_priority": priority}, threading.Event(), score=score)
                           for priority in ("foreground", "foreground", "foreground", "foreground", "background")]
                for record in records[:3]:
                    svc.acquire_request(record)
                    svc.finish_lifecycle(record)
                    other = svc.register_lifecycle({"strata_priority": "background"}, threading.Event(), score=not score)
                    svc.acquire_request(other)
                    svc.finish_lifecycle(other)
                for index in (4, 3):
                    records[index]["_deadline"] = time.perf_counter() + 0.25
                    svc.acquire_request(records[index])
                    svc.finish_lifecycle(records[index])
                self.assertEqual(svc.admission_waiters, [])



if __name__ == "__main__":
    unittest.main()
