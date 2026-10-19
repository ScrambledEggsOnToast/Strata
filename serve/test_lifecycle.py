"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import contextlib
import json
import io
import os
import socket
import subprocess
import time
import threading
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock
from serve import server as server_module
from serve import test_parallel as parallel_tests

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

    def test_lifecycle_terminal_and_frontend_totals_are_idempotent(self):
        svc = self.service()
        record = svc.register_lifecycle({}, threading.Event())
        svc.acquire_request(record)
        svc.lifecycle_adapter.terminal(record)
        svc.finish_lifecycle(record)
        snapshot = svc.lifecycle_records(record["id"])
        svc.lifecycle_adapter.terminal(record)
        svc.finish_lifecycle(record)
        self.assertEqual(svc.lifecycle_records(record["id"]), snapshot)
        self.assertEqual(svc.frontend_totals["foreground"]["completed"]["requests"], 1)
        self.assertEqual(sum(event["state"] == "completed" for event in record["events"]), 1)

    def test_adapter_terminal_drains_a_retained_legacy_consumer_once(self):
        svc = self.service()
        cancel = threading.Event()
        record = svc.register_lifecycle({}, cancel)
        svc.acquire_request(record)
        events = svc.lifecycle_adapter.events(record, [1], 8, {}, cancel)
        next(events)
        self.assertEqual(len(record["_generators"]), 1)
        svc.cancel_request(record["id"])
        svc.lifecycle_adapter.terminal(record)
        events.close()  # A dialect may release its outer iterator after terminal drain.
        self.assertEqual(record["_generators"], [])
        self.assertTrue(cancel.is_set())
        svc.finish_lifecycle(record)
        svc.lifecycle_adapter.terminal(record)
        svc.finish_lifecycle(record)
        self.assertEqual(svc.frontend_totals["foreground"]["cancelled"]["requests"], 1)

    def test_legacy_queue_keeps_timeout_message_and_existing_cancel_reason(self):
        for disconnected in (False, True):
            with self.subTest(disconnected=disconnected):
                svc = self.service()
                cancel = threading.Event()
                record = svc.register_lifecycle({}, cancel)
                record["_deadline"] = time.perf_counter() - 1
                if disconnected:
                    record["cancel_reason"] = "disconnected"
                    cancel.set()
                with self.assertRaises(AdmissionError) as failed:
                    svc.acquire_request(record)
                self.assertEqual(failed.exception.status, 409 if disconnected else 408)
                self.assertEqual(str(failed.exception), "disconnected" if disconnected else
                                 "request timed out while queued")
                self.assertTrue(cancel.is_set())
                svc.finish_lifecycle(record)

    def test_legacy_fallback_does_not_issue_unfenced_live_priority_controls(self):
        svc = self.service()
        record = svc.register_lifecycle({}, threading.Event())
        self.assertEqual(record["lifecycle_transport"], "legacy-seat")
        with self.assertRaisesRegex(ValueError, "require native lifecycle"):
            svc.lifecycle_adapter.priority(record, "background", None)
        self.assertFalse(record["_cancel"].is_set())
        svc.finish_lifecycle(record)

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



class NativeAdmissionLifecycle(parallel_tests.NativeAdmissionHarness, unittest.TestCase):
    """The migrated admission contract through HTTP, not the legacy Python selector."""

    def test_cancel_racing_first_native_token_delivers_the_committed_prefix(self):
        self.start_native()
        client = self.launch("cancel while the first token is crossing the frontend", stream=True, max_tokens=8)
        tag = self.registered_tag(1)
        identity = self.client_identity(client)
        self.wait_wire("BACK", tag=tag)
        native_next = self.engine._native_next
        cancelled = threading.Event()

        def cancel_on_first_token(owner, cancel):
            line = native_next(owner, cancel)
            if line is not None and line.startswith("BQT ") and owner["tag"] == tag and not cancelled.is_set():
                cancelled.set()
                self.assertTrue(self.svc.cancel_request(identity))
            return line

        with mock.patch.object(self.engine, "_native_next", side_effect=cancel_on_first_token):
            self.gate("service", held=False)
            self.finish_client(client)
        self.assertTrue(cancelled.is_set(), "the race must occur after the native token was committed")
        self.wait_until(lambda: self.svc.lifecycle_records(identity).get("attribution"))
        attribution = self.svc.lifecycle_records(identity)["attribution"]
        tokens = self.wire("BQT", tag=tag)
        terminal = self.wait_wire("BQDONE", tag=tag)
        self.assertEqual(len(terminal), 1)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(int(terminal[0]["raw"].split()[5]), len(tokens))
        self.assertEqual(attribution["committed_tokens"], len(tokens))
        self.assertEqual(attribution["output_tokens"], len(tokens))
        self.assertLessEqual(len(tokens), 8)
        events = [json.loads(line[6:]) for line in client["body"].splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join(event["choices"][0]["delta"].get("content", "") for event in events
                          if event.get("choices"))
        self.assertEqual(content, bytes(int(token["raw"].split()[5]) for token in tokens).decode())
        self.assertEqual(client["body"].count("data: [DONE]"), 1)
        self.assertEqual(client["body"].split("data: [DONE]", 1)[1].strip(), "")
        self.assertEqual(attribution["finish"], "cancel")
        self.assertTrue(self.engine.alive())

    def test_cancel_after_first_native_token_before_consumer_start_keeps_its_output(self):
        self.start_native(more=("--native-no-suspend",))
        self.gate("token-1")
        entered, release = threading.Event(), threading.Event()
        handler = self.httpd.RequestHandlerClass
        send_headers = handler._sse

        def hold_consumer(request):
            send_headers(request)
            entered.set()
            if not release.wait(5):
                raise AssertionError("cancelled consumer-start barrier timed out")

        try:
            with mock.patch.object(handler, "_sse", hold_consumer):
                client = self.launch("cancel after commit but before Service.run", stream=True, max_tokens=8)
                self.assertTrue(entered.wait(2))
                tag = self.registered_tag(1)
                identity = self.client_identity(client)
                self.gate("service", held=False)
                self.wait_wire("BQT", tag=tag)
                self.cancel_client(client)
                self.wait_wire("BQDONE", tag=tag)
                release.set()
                self.finish_client(client)
        finally:
            release.set()
        self.wait_until(lambda: self.svc.lifecycle_records(identity).get("attribution"))
        attribution = self.svc.lifecycle_records(identity)["attribution"]
        self.assertEqual(len(self.wire("BQT", tag=tag)), 1)
        self.assertEqual(attribution["committed_tokens"], 1)
        self.assertEqual(attribution["output_tokens"], 1)
        self.assertIn('"content": "o"', client["body"])
        self.assertEqual(client["body"].count("data: [DONE]"), 1)
        self.assertEqual(client["body"].split("data: [DONE]", 1)[1].strip(), "")
        self.assertEqual(len(self.wire("BGENQ1", tag=tag, direction="sent")), 1)
        self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)

    def test_cancel_racing_seating_does_not_discard_an_already_committed_first_token(self):
        self.start_native(more=("--native-no-suspend",))
        self.gate("token-1")
        native_next = self.engine._native_next
        cancelled = threading.Event()

        def cancel_on_seating(owner, cancel):
            line = native_next(owner, cancel)
            if line is not None and line.startswith("BACK ") and not cancelled.is_set():
                self.gate("service", held=False)
                self.wait_wire("BQT", tag=owner["tag"])
                cancelled.set()
                self.assertTrue(self.svc.cancel_request(owner["live"]["request_id"]))
            return line

        with mock.patch.object(self.engine, "_native_next", side_effect=cancel_on_seating):
            client = self.launch("cancel before the seating consumer returns", stream=True, max_tokens=8)
            self.finish_client(client)
        self.assertTrue(cancelled.is_set())
        identity = self.client_identity(client)
        self.wait_until(lambda: self.svc.lifecycle_records(identity).get("attribution"))
        attribution = self.svc.lifecycle_records(identity)["attribution"]
        self.assertEqual((attribution["committed_tokens"], attribution["output_tokens"]), (1, 1))
        self.assertIn('"content": "o"', client["body"])
        self.assertEqual(client["body"].count("data: [DONE]"), 1)
        self.assertEqual(client["body"].split("data: [DONE]", 1)[1].strip(), "")

    def test_cancel_drains_mtp_commits_before_each_dialects_terminal_without_touching_peer(self):
        # This is the frontend/wire seam, not a GPU-arithmetic fixture. The native
        # stand-in publishes two accepted in-flight tokens only when drain is released.
        self.start_native(more=("--native-cancel-catchup", "2"))
        for index, path in enumerate(("/v1/chat/completions", "/v1/messages", "/v1/responses")):
            with self.subTest(path=path):
                self.gate("service")
                cap = 5 if index == 0 else 8
                dialect = ({"thinking": {"type": "disabled"}} if path == "/v1/messages" else
                           {"reasoning": {"effort": "none"}} if path == "/v1/responses" else {})
                client = self.launch(f"{index}: cancel after three delivered tokens", path=path, stream=True,
                                     max_tokens=cap, body_updates=dialect)
                tag = self.registered_tag(2 * index + 1)
                identity = self.client_identity(client)
                self.wait_wire("BACK", tag=tag)
                self.gate("token-3", tag)
                self.gate("drain", tag)
                peer = self.launch(f"{index}: surviving peer", max_tokens=1)
                peer_tag = self.registered_tag(2 * index + 2)
                self.client_identity(peer)
                self.wait_wire("BACK", tag=peer_tag)
                self.gate("service", peer_tag)
                self.gate("service", held=False)
                self.wait_until(lambda: self.client_record(client).get("output_tokens") == 3)
                self.cancel_client(client)
                self.wait_wire("BQCANCEL", tag=tag, direction="sent")
                self.assertFalse(client["done"].is_set(), "the HTTP terminal must wait for the native drain")
                self.assertEqual(self.wire("BQDONE", tag=tag), [])
                self.gate("drain", tag, held=False)
                self.finish_client(client)
                self.wait_until(lambda: self.svc.lifecycle_records(identity).get("attribution"))
                attribution = self.svc.lifecycle_records(identity)["attribution"]
                tokens = self.wire("BQT", tag=tag)
                terminal = self.wait_wire("BQDONE", tag=tag)
                self.assertEqual(len(tokens), 5)
                self.assertLessEqual(len(tokens), cap)
                self.assertEqual(len(terminal), 1)
                self.assertEqual(terminal[0]["raw"].split()[5], "5")
                self.assertEqual((attribution["committed_tokens"], attribution["output_tokens"]), (5, 5))
                self.assertEqual((attribution["drafts_accepted"], attribution["drafts_offered"]), (2, 2))
                self.assertEqual(attribution["finish"], "cancel")
                self.assertEqual(len(self.wire("BGENQ1", tag=tag, direction="sent")), 1)
                self.assertEqual(self.wire("BQCANCEL", tag=peer_tag, direction="sent"), [])
                self.assertFalse(peer["done"].is_set())
                events = [json.loads(line[6:]) for line in client["body"].splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                if path == "/v1/chat/completions":
                    content = "".join(event["choices"][0]["delta"].get("content", "") for event in events
                                      if event.get("choices"))
                    self.assertEqual(sum("error" in event for event in events), 1)
                    self.assertIn("error", events[-1])
                    self.assertEqual(client["body"].count("data: [DONE]"), 1)
                    self.assertEqual(client["body"].split("data: [DONE]", 1)[1].strip(), "")
                elif path == "/v1/messages":
                    content = "".join(event.get("delta", {}).get("text", "") for event in events)
                    self.assertEqual(sum(event["type"] in ("message_stop", "error") for event in events), 1)
                    self.assertEqual(events[-1]["type"], "error")
                else:
                    content = "".join(event["delta"] for event in events
                                      if event["type"] == "response.output_text.delta")
                    self.assertEqual(sum(event["type"] in ("response.completed", "response.failed",
                                                          "response.incomplete") for event in events), 1)
                    self.assertEqual(events[-1]["type"], "response.failed")
                self.assertEqual(content, "ok, d")
                self.assertLess(tokens[-1]["monotonic_s"], terminal[0]["monotonic_s"])
                self.gate("service", peer_tag, held=False)
                result = self.finish_client(peer)["body"]
                self.assertEqual(result["choices"][0]["message"]["content"], "o")
                self.assertEqual(result["usage"]["completion_tokens"], 1)
                self.assertEqual(len(self.wire("BQDONE", tag=peer_tag)), 1)
                self.assertTrue(self.engine.alive())

    def test_disconnected_nonstream_native_request_drains_first_token_and_mtp_without_delivery(self):
        self.start_native(more=("--native-no-suspend", "--native-cancel-catchup", "2"))
        entered, release = threading.Event(), threading.Event()
        native_next = self.engine._native_next
        tag = None

        def hold_first_token(owner, cancel):
            line = native_next(owner, cancel)
            if line is not None and line.startswith("BQT ") and owner["tag"] == tag and not entered.is_set():
                entered.set()
                if not release.wait(8):
                    raise AssertionError("disconnected first-token barrier timed out")
            return line

        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "disconnect in flight"}],
                           "stream": False, "max_tokens": 8, "reasoning_effort": "none"}).encode()
        token_gate = mock.patch.object(self.engine, "_native_next", side_effect=hold_first_token)
        token_gate.start()
        self.addCleanup(token_gate.stop)
        sock = socket.create_connection(self.httpd.server_address, timeout=5)
        try:
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                         b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode() +
                         b"\r\n\r\n" + body)
            tag = self.registered_tag(1)
            self.wait_wire("BACK", tag=tag)
            identity = self.get("/api/lifecycle")[0]["id"]
            self.gate("token-1", tag)
            self.gate("drain", tag)
            push, feed = server_module.Detokenizer.push, server_module.OutputParser.feed
            with mock.patch.object(server_module.Detokenizer, "push", autospec=True, side_effect=push) as pushed, \
                    mock.patch.object(server_module.OutputParser, "feed", autospec=True, side_effect=feed) as fed:
                self.gate("service", held=False)
                self.assertTrue(entered.wait(2), "first-token commitment must precede the disconnect")
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                self.wait_until(lambda: self.svc.lifecycle_records(identity).get("cancel_reason") == "disconnected",
                                seconds=2)
                release.set()
                self.wait_wire("BQCANCEL", tag=tag, direction="sent")
                self.assertEqual(self.wire("BQDONE", tag=tag), [], "native drain must remain fenced")
                self.gate("drain", tag, held=False)
                self.wait_until(lambda: self.svc.lifecycle_records(identity).get("attribution"))
                attribution = self.svc.lifecycle_records(identity)["attribution"]
                self.assertEqual(attribution["outcome"], "disconnected")
                self.assertEqual((attribution["committed_tokens"], attribution["output_tokens"]), (3, 0))
                self.assertEqual(self.svc.lifecycle_records(identity).get("output_tokens", 0), 0)
                self.assertEqual(self.svc.history[-1]["output_tokens"], 0)
                pushed.assert_not_called()
                fed.assert_not_called()
            self.assertEqual(len(self.wire("BQT", tag=tag)), 3)
            self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)
            self.assertEqual(len(self.wire("BGENQ1", tag=tag, direction="sent")), 1)
            self.assertFalse(self.engine.packed_q)
            self.assertTrue(self.engine.alive())
        finally:
            release.set()
            sock.close()
            if tag is not None:
                self.gate("drain", tag, held=False)

    def test_unseated_cancel_and_timeout_keep_preheader_status_and_zero_output(self):
        self.start_native(admission_held=True)
        for number, (status, reason) in enumerate(((409, "cancel"), (408, "timeout")), 1):
            with self.subTest(status=status):
                client = self.launch("cancel without a native seat", stream=True,
                                     timeout=.15 if status == 408 else None, max_tokens=8)
                tag = self.registered_tag(number)
                identity = self.client_identity(client)
                self.wait_wire("BGENQ1", tag=tag, direction="sent")
                self.assertFalse(client["headers_ready"].is_set())
                if status == 409:
                    self.cancel_client(client)
                self.finish_client(client, status)
                self.assertNotIn("event-stream", client["headers"]["Content-Type"])
                self.assertNotIn("Retry-After", client["headers"])
                self.assert_queued_terminal(tag, reason)
                self.wait_until(lambda: self.svc.lifecycle_records(identity).get("attribution"))
                attribution = self.svc.lifecycle_records(identity)["attribution"]
                self.assertEqual((attribution["committed_tokens"], attribution["output_tokens"]), (0, 0))
                self.assertEqual(attribution["finish"], reason)
        self.assertTrue(self.engine.alive())

    def test_unloaded_native_engine_reloads_before_registering_accepted_ingress(self):
        self.start_native()
        self.gate("service", held=False)
        self.finish_client(self.launch("warm retained engine"))
        old_generation = self.engine.gen
        request = urllib.request.Request(self.base + "/v1/unload", b"{}",
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
        self.assertFalse(self.engine.alive())
        real, script = subprocess.Popen, Path(self.tmp.name) / "fake_strata.py"
        with mock.patch("serve.server.subprocess.Popen",
                        lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            reply = self.finish_client(self.launch("reload and register after startup"))
        self.assertEqual(reply["body"]["usage"]["completion_tokens"], 1)
        self.assertGreater(self.engine.gen, old_generation)
        self.assertEqual(self.client_record(reply)["native_registration_count"], 1)

    def test_reload_backfills_accepted_native_metadata_in_ingress_order(self):
        self.start_native()
        self.gate("service", held=False)
        self.finish_client(self.launch("warm before backfill"))
        self.assertEqual(self.svc.unload(), "unloaded")
        real, script = subprocess.Popen, Path(self.tmp.name) / "fake_strata.py"
        with mock.patch("serve.server.subprocess.Popen",
                        lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            with self.svc.fifo:
                first = self.launch("first accepted while unloaded")
                self.client_identity(first)
                second = self.launch("second accepted while unloaded")
                self.client_identity(second)
                self.assertEqual(len(self.wire("BQREGISTER", direction="sent")), 1)
            self.finish_client(first)
            self.finish_client(second)
        tags = [receipt["raw"].split()[1] for receipt in self.wire("BQREGISTER", direction="sent")]
        for client, tag in zip((first, second), tags[1:]):
            accepted = [event["tag"] for event in self.client_record(client)["retained_events"]
                        if event["event"] == "BQACCEPT"]
            self.assertEqual(accepted, [tag])
        self.assertEqual(len(tags), 3)

    def assert_deferred_close_excludes_load_backfill(self, *, cancel, lazy):
        self.start_native(1, lazy=lazy)
        if not lazy:
            self.assertEqual(self.svc.unload(), "unloaded")
        blocker = None
        if cancel:
            blocker = self.svc.register_lifecycle({}, threading.Event())
            self.svc.acquire_request(blocker)
        record = self.svc.register_lifecycle({}, threading.Event())
        self.assertEqual(record.get("native_registration_count", 0), 0)
        closed, finish, errors = threading.Event(), threading.Event(), []
        def closing_handler():
            try:
                if cancel:
                    try:
                        self.svc.acquire_request(record)
                    except AdmissionError as error:
                        self.assertEqual(error.status, 409)
                    else:
                        raise AssertionError("deferred request bypassed the occupied preparation permit")
                self.svc.lifecycle_adapter.terminal(record)
                closed.set()
                if not finish.wait(5):
                    raise AssertionError("lifecycle retirement barrier timed out")
                self.svc.finish_lifecycle(record)
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=closing_handler)
        worker.start()
        try:
            if cancel:
                self.wait_until(lambda: self.svc.lifecycle_records(record["id"]).get("ingress_wait_s", 0) > .02)
                self.assertTrue(self.svc.cancel_request(record["id"]))
            self.assertTrue(closed.wait(2), errors)
            # The final close has returned, but the handler has not yet retired its
            # queued record. Completing load here used to resurrect an owner behind it.
            real, script = subprocess.Popen, Path(self.tmp.name) / "fake_strata.py"
            with mock.patch("serve.server.subprocess.Popen",
                            lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                self.svc.load()
            finish.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            if blocker is not None:
                self.svc.lifecycle_adapter.terminal(blocker)
                self.svc.finish_lifecycle(blocker)
            self.assertEqual(record.get("native_registration_count", 0), 0,
                             "backfill registered a lifecycle after its last terminal close")
            self.assertEqual(record["_cancel"].is_set(), cancel,
                             "ordinary cleanup must fence backfill without relying on cancellation")
            self.assertEqual(len(self.wire("BQREGISTER", direction="sent")), 1 if blocker else 0)
            self.gate("service", held=False)
            ready = self.finish_client(self.launch("ready successor after deferred terminal cleanup"))
            self.assertEqual(ready["body"]["usage"]["completion_tokens"], 1)
            self.assertEqual(self.engine.packed_q, {}, "terminal cleanup left native routing/callback owners")
            self.assertEqual(len(self.wire("BQDONE")), 2 if blocker else 1)
        finally:
            finish.set()
            worker.join(2)
            self.svc.lifecycle_adapter.terminal(record)  # Drain any owner found by a red backfill assertion.

    def test_deferred_ingress_cancel_cannot_be_registered_after_close_during_reload(self):
        self.assert_deferred_close_excludes_load_backfill(cancel=True, lazy=False)

    def test_ownerless_lazy_cleanup_without_cancellation_cannot_be_backfilled_after_close(self):
        self.assert_deferred_close_excludes_load_backfill(cancel=False, lazy=True)

    def test_native_ingress_timeout_and_cancel_keep_queue_wait_after_finalization(self):
        self.start_native(1, depth=1)
        with self.tokenization_gate("held-ingress") as (entered, release):
            first = self.launch("held-ingress occupies the preparation permit")
            self.assertTrue(entered.wait(2))
            for timeout, cancel, status in ((.15, False, 408), (None, True, 409)):
                queued = self.launch("wait before tokenization", timeout=timeout)
                self.wait_until(lambda: self.client_record(queued)["ingress_wait_s"] >= .02)
                if cancel:
                    self.cancel_client(queued)
                self.finish_client(queued, status)
                record = self.client_record(queued)
                self.assertGreater(record["ingress_wait_s"], 0)
                self.assertGreaterEqual(record["queue_s"], record["ingress_wait_s"])
                self.assertEqual(record["native_submission_count"], 0)
            release.set()
            self.gate("service", held=False)
            self.finish_client(first)

    def test_native_cancel_and_timeout_after_headers_use_each_dialects_stream_error(self):
        self.start_native()
        handler = self.httpd.RequestHandlerClass
        original = handler._sse
        for path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
            for status in (408, 409):
                with self.subTest(path=path, status=status):
                    entered, release = threading.Event(), threading.Event()
                    def hold_after_headers(request):
                        original(request)
                        entered.set()
                        if not release.wait(3):
                            raise AssertionError("post-header test barrier timed out")
                    with mock.patch.object(handler, "_sse", hold_after_headers):
                        try:
                            client = self.launch("cancel after 200 headers", path=path, stream=True,
                                                 timeout=.2 if status == 408 else None)
                            self.assertTrue(entered.wait(2))
                            tag = self.registered_tag(len(self.wire("BQREGISTER", direction="sent")))
                            if status == 409:
                                self.cancel_client(client)
                            self.wait_wire("BQDONE", tag=tag)
                        finally:
                            release.set()
                        self.finish_client(client)
                    body = client["body"]
                    self.assertNotIn("HTTP/1.", body, "a stream must not contain a second HTTP response")
                    if path == "/v1/chat/completions":
                        self.assertIn('"error"', body)
                        self.assertIn("data: [DONE]", body)
                    elif path == "/v1/messages":
                        self.assertIn("event: error", body)
                    else:
                        self.assertIn("event: response.failed", body)

    def test_mcp_continuation_cancelled_behind_ingress_ends_the_open_stream(self):
        from serve.test_mcp import call_script, stdio
        from serve.mcp import McpHub
        with contextlib.ExitStack() as stack:
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            reply = Path(directory) / "reply"
            reply.write_text(call_script("fake__echo", text="continuation"))
            hub = McpHub({"fake": stdio()}, {"timeout_s": 10, "max_result_chars": 500, "max_rounds": 2})
            hub.start(wait=True)
            stack.callback(hub.close)
            self.start_native(1, depth=2, more=("--native-reply", str(reply)))
            self.svc.mcp = hub
            self.gate("service", held=False)
            called, release_tool = threading.Event(), threading.Event()
            original = hub.call
            def gated_tool(*args):
                called.set()
                if not release_tool.wait(5):
                    raise AssertionError("MCP continuation barrier timed out")
                return original(*args)
            stack.callback(release_tool.set)
            stack.enter_context(mock.patch.object(hub, "call", gated_tool))
            with self.tokenization_gate("held-peer") as (entered, release_peer):
                client = self.launch("MCP continuation", stream=True, max_tokens=512,
                                     body_updates={"strata_mcp": True})
                self.assertTrue(called.wait(5), client.get("body"))
                peer = self.launch("held-peer occupies the continuation preparation lane")
                self.assertTrue(entered.wait(2))
                release_tool.set()
                self.wait_until(lambda: self.client_record(client)["frontend"]["counts"]["ingress_wait"] >= 2)
                self.cancel_client(client)
                self.finish_client(client)
                self.assertNotIn("HTTP/1.", client["body"])
                self.assertIn('"error"', client["body"])
                self.assertIn("data: [DONE]", client["body"])
                self.assertEqual(self.client_record(client)["native_submission_count"], 1)
                release_peer.set()
                self.finish_client(peer)

    def assert_grant_order(self, clients, tags, order):
        self.gate("admission", held=False)
        previous_terminal = None
        for grant, index in enumerate(order, 1):
            seat = self.wait_wire("BACK", tag=tags[index])[0]
            if previous_terminal is not None:
                self.assertLessEqual(previous_terminal["monotonic_s"], seat["monotonic_s"])
            self.assertEqual([receipt["raw"].split()[1] for receipt in self.wire("BACK")],
                             [tags[i] for i in order[:grant]])
            self.cancel_client(clients[index])
            self.finish_client(clients[index], status=None)
            terminals = self.wait_wire("BQDONE", tag=tags[index])
            self.assertEqual(len(terminals), 1)
            self.assertEqual(terminals[0]["raw"].split()[5], "0")
            previous_terminal = terminals[0]
        self.assertTrue(self.engine.alive())

    def test_native_priority_fifo_admits_background_after_three_foreground_grants(self):
        self.start_native(1, admission_held=True)
        clients, tags = [], []
        for number, priority in enumerate(("foreground", "foreground", "foreground", "foreground", "background"), 1):
            client = self.launch(f"policy request {number}", priority=priority)
            tag = self.registered_tag(number)
            self.client_identity(client)
            self.wait_wire("BGENQ1", tag=tag, direction="sent")
            clients.append(client)
            tags.append(tag)
        self.assertEqual(self.wire("BACK"), [])
        self.assert_grant_order(clients, tags, (0, 1, 2, 4, 3))

    def test_native_solo_preference_yields_to_eligible_after_four_successful_grants(self):
        self.start_native(1, admission_held=True)
        clients, tags = [], []
        for number, packing in enumerate(("eligible", "solo", "solo", "solo", "solo", "solo"), 1):
            client = self.launch(f"packing request {number}", packing=packing)
            tag = self.registered_tag(number)
            self.client_identity(client)
            self.wait_wire("BGENQ1", tag=tag, direction="sent")
            clients.append(client)
            tags.append(tag)
        self.assert_grant_order(clients, tags, (1, 2, 3, 4, 0, 5))

    def test_early_metadata_preserves_fifo_when_tokenization_finishes_in_reverse_order(self):
        self.start_native()
        with self.tokenization_gate("slow-first") as (entered, release):
            first = self.launch("slow-first tokenization")
            first_tag = self.registered_tag(1)
            self.client_identity(first)
            self.wait_wire("BQACCEPT", tag=first_tag)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.wire("BGENQ1", tag=first_tag, direction="sent"), [])
            second = self.launch("ready-second tokenization")
            second_tag = self.registered_tag(2)
            self.client_identity(second)
            self.wait_wire("BGENQ1", tag=second_tag, direction="sent")
            self.assertEqual(self.wire("BACK"), [], "a later ready prompt bypassed earlier metadata")
            release.set()
            self.wait_wire("BGENQ1", tag=first_tag, direction="sent")
            self.wait_wire("BACK", count=2)
            self.assertEqual([receipt["raw"].split()[1] for receipt in self.wire("BGENQ1", direction="sent")],
                             [second_tag, first_tag])
            self.assertEqual([receipt["raw"].split()[1] for receipt in self.wire("BACK")],
                             [first_tag, second_tag])
            self.gate("service", held=False)
            for client in (first, second):
                self.assertEqual(self.finish_client(client)["body"]["usage"]["completion_tokens"], 1)

    def test_original_deadline_expires_during_tokenization_without_seating_or_reset(self):
        self.start_native()
        with self.tokenization_gate("deadline-tokenizer") as (entered, release):
            client = self.launch("deadline-tokenizer must not reset the HTTP deadline", stream=True, timeout=.37)
            tag = self.registered_tag(1)
            self.client_identity(client)
            self.assertTrue(entered.wait(2))
            terminal = self.assert_queued_terminal(tag, "timeout")
            self.assertEqual(terminal["raw"].split()[6], "0", "no complete input was published")
            self.assertFalse(client["headers_ready"].is_set())
            release.set()
            response = self.finish_client(client, 408)
            self.assertNotIn("Retry-After", response["headers"])
            self.assertEqual(len(self.wire("BQREGISTER", tag=tag, direction="sent")), 1)
            self.assertLessEqual(len(self.wire("BGENQ1", tag=tag, direction="sent")), 1)
            self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)

    def assert_retained_receipts(self, record, tag):
        names = ("BQACCEPT", "BACK", "BQPP", "BQSUSPEND", "BQRESUMED", "BQDONE", "BQCANCEL")
        expected = [receipt for name in names for receipt in
                    self.wire(name, tag=tag, direction="sent" if name == "BQCANCEL" else "received")]
        expected.sort(key=lambda receipt: receipt["monotonic_s"])
        actual = [event for event in record["retained_events"] if event["tag"] == tag and event["event"] in names]
        self.assertEqual(record["retained_events_dropped"], 0)
        self.assertTrue(record["retained_events_complete"])
        self.assertEqual([(event["direction"], event["raw"]) for event in actual],
                         [(receipt["direction"], receipt["raw"]) for receipt in expected])
        self.assertEqual([event["monotonic_s"] for event in actual],
                         sorted(event["monotonic_s"] for event in actual))
        for event in actual:
            fields = event["raw"].split()
            self.assertEqual(event["event"], fields[0])
            self.assertEqual(event["tag"], fields[1])
            self.assertEqual((event["slot"], event["request"], event["incarnation"]), tuple(map(int, fields[2:5])))
            self.assertGreater(event["monotonic_s"], 0)

    def test_native_seating_receives_zero_service_wait_before_progress_or_terminal(self):
        self.start_native()
        client = self.launch("immediately admitted service receipt")
        tag = self.registered_tag(1)
        self.wait_wire("BACK", tag=tag)
        self.wait_until(lambda: self.engine.ended or self.client_record(client)["state"] == "prefill")
        self.assertFalse(self.engine.ended, "the native live-counter receipt was not parsed")
        record = self.client_record(client)
        self.assertEqual(record["native_service_wait_s"], 0.0)
        self.assertEqual(record["native_wait_receipts_missing"]["service"], 0)
        back = next(event for event in record["retained_events"] if event["event"] == "BACK")
        self.assertEqual(back["native_service_wait_s"], 0.0)
        self.assertIn("native_service_wait_s=0.000000000", back["raw"])
        self.assertEqual(self.wire("BQPP", tag=tag), [])
        self.assertEqual(self.wire("BQDONE", tag=tag), [])
        self.gate("service", held=False)
        self.finish_client(client)
        record = self.client_record(client)
        progress = [event for event in record["retained_events"] if event["event"] == "BQPP"]
        self.assertTrue(progress)
        self.assertTrue(all(event["native_service_wait_s"] >= 0 for event in progress))
        self.assertGreaterEqual(record["native_service_wait_s"], progress[-1]["native_service_wait_s"])

    def test_missing_live_service_receipt_is_not_synthesized_from_seating(self):
        self.start_native(more=("--native-missing-service-receipt",))
        client = self.launch("seated owner with missing live service telemetry")
        tag = self.registered_tag(1)
        self.wait_wire("BACK", tag=tag)
        self.wait_until(lambda: self.client_record(client)["state"] == "prefill")
        record = self.client_record(client)
        self.assertNotIn("native_service_wait_s", record)
        self.assertEqual(record["native_wait_receipts_missing"]["service"], 1)
        back = next(event for event in record["retained_events"] if event["event"] == "BACK")
        self.assertNotIn("native_service_wait_s", back)
        self.assertEqual(self.wire("BQDONE", tag=tag), [])
        self.gate("service", held=False)
        self.finish_client(client)

    def test_immediate_zero_service_wait_is_kept_in_lifecycle_and_attribution(self):
        self.start_native()
        self.gate("service", held=False)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            client = self.finish_client(self.launch("immediate service without queue wait"))
            identity = self.client_identity(client)
            prefix = "[strata] response_attribution: "
            def matching_attributions():
                return [value for line in log.getvalue().splitlines() if line.startswith(prefix)
                        and (value := json.loads(line[len(prefix):]))["request_id"] == identity]
            self.wait_until(matching_attributions)
            self.assertEqual(len(matching_attributions()), 1)
            attribution = matching_attributions()[0]
        for record in (self.client_record(client), attribution):
            self.assertEqual(record["native_service_wait_s"], 0.0)
            self.assertEqual(record["queue_s"], record["admission_wait_s"])
            self.assertGreaterEqual(record["queue_s"], record["ingress_wait_s"])
            self.assertEqual(record["native_wait_receipts_missing"]["service"], 0)
            self.assertTrue(any(event.get("native_service_wait_s") == 0
                                for event in record["retained_events"] if event["event"] == "BACK"))

    def test_queued_acceptance_publishes_authoritative_zero_service_before_input_or_seat(self):
        self.start_native()
        with self.tokenization_gate("queued-counter") as (entered, release):
            client = self.launch("queued-counter has not published any input")
            tag = self.registered_tag(1)
            self.client_identity(client)
            self.assertTrue(entered.wait(2))
            self.wait_wire("BQACCEPT", tag=tag)
            record = self.client_record(client)
            self.assertEqual(self.wire("BACK", tag=tag), [])
            self.assertEqual(self.wire("BGENQ1", tag=tag, direction="sent"), [])
            self.assertEqual(record.get("native_service_wait_s"), 0.0)
            self.assertEqual(record["native_wait_receipts_missing"]["service"], 0)
            accepted = next(event for event in record["retained_events"] if event["event"] == "BQACCEPT")
            self.assertIn("native_service_wait_s=0.000000000", accepted["raw"])
            self.assertEqual(accepted["native_service_wait_s"], 0.0)
            self.cancel_client(client)
            release.set()
            self.finish_client(client, 409)

    def test_ingress_preparation_admission_and_service_waits_are_distinct_receipt_driven_fields(self):
        self.start_native(depth=1)
        with contextlib.redirect_stdout(io.StringIO()) as log, \
                self.tokenization_gate("ingress-A") as (entered_a, release_a), \
                self.tokenization_gate("ingress-B") as (entered_b, release_b):
            first = self.launch("ingress-A fills the first preparation lane")
            tag_a = self.registered_tag(1)
            self.client_identity(first)
            self.assertTrue(entered_a.wait(2))
            peer = self.launch("ingress-B fills the second preparation lane")
            tag_b = self.registered_tag(2)
            self.client_identity(peer)
            self.assertTrue(entered_b.wait(2))
            queued = self.launch("C waits for ingress, then admission, then service")
            tag_c = self.registered_tag(3)
            identity = self.client_identity(queued)
            self.wait_wire("BQACCEPT", tag=tag_c)
            self.assertEqual(self.wire("BGENQ1", tag=tag_c, direction="sent"), [])
            accepted = self.client_record(queued)
            self.assertNotIn("native_admission_wait_s", accepted)
            self.assertEqual(accepted["native_service_wait_s"], 0.0)
            release_a.set()
            release_b.set()
            self.wait_wire("BACK", count=2)
            self.wait_wire("BGENQ1", tag=tag_c, direction="sent")
            self.assertEqual(self.wire("BACK", tag=tag_c), [])
            self.cancel_client(first)
            self.wait_wire("BACK", tag=tag_c)
            self.wait_until(lambda: (self.gates / f"observed-service-{tag_c}").exists())
            self.gate("service", tag_b)
            self.gate("service", held=False)
            result = self.finish_client(queued)
            self.assertEqual(result["body"]["usage"]["completion_tokens"], 1)
            prefix = "[strata] response_attribution: "
            def attributions():
                return [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                        if line.startswith(prefix) and json.loads(line[len(prefix):])["request_id"] == identity]
            self.wait_until(attributions)
            self.assertEqual(len(attributions()), 1)
            attribution = attributions()[0]
            record = self.client_record(queued)
            terminal = self.wait_wire("BQDONE", tag=tag_c)[0]["raw"].split()
            counters = dict(field.split("=", 1) for field in terminal[14:])
            for observed in (record, attribution):
                self.assertGreater(observed["ingress_wait_s"], 0)
                self.assertGreater(observed["frontend"]["seconds"]["preparation"], 0)
                self.assertGreater(observed["native_admission_wait_s"], 0)
                self.assertGreater(observed["native_service_wait_s"], 0)
                self.assertEqual(observed["queue_s"], observed["admission_wait_s"])
                self.assertGreaterEqual(observed["queue_s"], observed["ingress_wait_s"])
                for name in ("native_admission_wait_s", "native_service_wait_s"):
                    self.assertAlmostEqual(observed[name], float(counters[name]), places=9)
                self.assertTrue(all(value >= 0 for value in observed["frontend"]["seconds"].values()))
                self.assert_retained_receipts(observed, tag_c)
            self.assertEqual(record["native_submission_count"], 1)
            self.assertEqual(attribution["native_submission_count"], 1)
            self.assertEqual(attribution["output_tokens"], 1)
            self.assertEqual(attribution["committed_tokens"], 1)
            self.assertEqual(len(attribution["legs"]), 1)
            self.assertFalse(peer["done"].is_set())
            self.finish_client(first, status=None)
            self.assert_retained_receipts(self.client_record(first), tag_a)
            self.cancel_client(peer)
            self.finish_client(peer, status=None)

    def test_retained_event_capacity_reports_every_dropped_receipt_as_incomplete(self):
        self.start_native(more=("--native-progress-count", "4100", "--says-context", "8192"))
        with contextlib.redirect_stdout(io.StringIO()) as log:
            client = self.launch("many retained progress receipts " * 180)
            tag = self.registered_tag(1)
            identity = self.client_identity(client)
            self.gate("service", held=False)
            self.finish_client(client)
            prefix = "[strata] response_attribution: "
            def attributions():
                return [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                        if line.startswith(prefix) and json.loads(line[len(prefix):])["request_id"] == identity]
            self.wait_until(attributions)
            self.assertEqual(len(attributions()), 1)
            attribution = attributions()[0]
        record = self.client_record(client)
        expected = sum(len(self.wire(name, tag=tag))
                       for name in ("BQACCEPT", "BACK", "BQPP", "BQSUSPEND", "BQRESUMED", "BQDONE"))
        self.assertEqual(len(self.wire("BQPP", tag=tag)), 4100)
        for observed in (record, attribution):
            self.assertEqual(len(observed["retained_events"]) + observed["retained_events_dropped"], expected)
            self.assertGreater(observed["retained_events_dropped"], 0)
            self.assertFalse(observed["retained_events_complete"])

    def test_context_validation_cancels_early_metadata_without_publishing_invalid_input(self):
        self.start_native()
        client = self.launch("over context " * 500)
        tag = self.registered_tag(1)
        self.client_identity(client)
        self.finish_client(client, 400)
        self.assertEqual(self.wire("BGENQ1", tag=tag, direction="sent"), [])
        terminal = self.assert_queued_terminal(tag, "cancel")
        self.assertEqual(terminal["raw"].split()[6], "0")
        self.assertEqual(len(self.wire("BQCANCEL", tag=tag, direction="sent")), 1)
        successor = self.launch("validation released the accepted allowance")
        successor_tag = self.registered_tag(2)
        self.client_identity(successor)
        self.wait_wire("BACK", tag=successor_tag)
        self.gate("service", held=False)
        self.finish_client(successor)

    def actual_engine_exit(self, code):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.start_native(more=("--exit-code", str(code)))
            process = self.engine.proc
            startup = self.get("/metrics")["engine"]["protocol_startup"]
            self.assertIsInstance(startup, list)
            self.assertEqual(startup[0], "INFO engine=0.1.39 batch_slots=2")
            self.assertIn("INFO retained_prompt=1 retained_prompt_version=1 retained_prompt_owners=4", startup)
            self.assertIn("INFO native_admission=1 native_admission_version=1", startup)
            plans = [json.loads(line[6:]) for line in startup if line.startswith("EPLAN ")]
            self.assertEqual([plan["admission"] for plan in plans], [{"authority": "native", "support": "on"}])
            self.assertEqual(startup[-1], "READY 4096 stop")
            self.assertNotIn("[strata] engine_exit: ", log.getvalue())
            if code:
                self.engine._send("QUIT")
                self.wait_until(lambda: process.poll() is not None)
                self.assertFalse(self.engine.alive())
            self.engine.close()
            self.engine.close()
            self.assertEqual(process.returncode, code)
            prefix = "[strata] engine_exit: "
            records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines() if line.startswith(prefix)]
            self.assertEqual(records, [{"pid": process.pid, "exit_code": code}])

    def test_raw_protocol_startup_and_clean_engine_exit_are_real_process_observations(self):
        self.actual_engine_exit(0)

    def test_nonzero_engine_exit_is_detected_and_never_reported_as_clean(self):
        self.actual_engine_exit(7)

    def test_unstarted_engine_has_no_fabricated_startup_or_exit(self):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.engine = StrataEngine("missing-executable", [], lazy=True)
            svc = Service(self.engine, ByteTokenizer(), None)
            self.assertEqual(svc.metrics()["engine"].get("protocol_startup", []), [])
            self.engine.close()
        self.assertNotIn("[strata] engine_exit: ", log.getvalue())

    def test_score_keeps_its_independent_lane_when_native_generation_allowance_is_full(self):
        self.start_native(depth=0)
        for number in range(1, 3):
            client = self.launch(f"generation reservation {number}")
            tag = self.registered_tag(number)
            self.client_identity(client)
            self.wait_wire("BACK", tag=tag)
        result = parallel_tests.ScoreTransport.score(self, prompt=[5, 6], selected=[3, 1, 0, 3])
        self.assertEqual(result["logits"], [.3333333432674408, -17.5, .125, .3333333432674408])
        self.assertEqual(result["generated_tokens"], 0)
        self.assertEqual(len(self.wire("BQREGISTER", direction="sent")), 2)
        self.assertEqual(self.wire("BQDONE"), [])
        self.gate("service", held=False)
        for client in self.clients:
            self.assertEqual(self.finish_client(client)["body"]["usage"]["completion_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
