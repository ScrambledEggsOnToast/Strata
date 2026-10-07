"""GPU reader ownership and sampler shutdown; no driver or native engine is required."""
import contextlib
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import server, telemetry


class NvmlOwnership(unittest.TestCase):
    def library(self):
        lib = mock.Mock(spec=["nvmlInit_v2", "nvmlShutdown", "nvmlDeviceGetHandleByIndex_v2",
                              "nvmlDeviceGetMemoryInfo"])
        lib.nvmlInit_v2.return_value = 0
        lib.nvmlShutdown.return_value = 0

        def handle(index, ptr):
            ptr._obj.value = index.value + 1
            return 0

        def memory(dev, ptr):
            ptr._obj.total, ptr._obj.used, ptr._obj.free = 8 << 30, 2 << 30, 5 << 30
            return 0

        lib.nvmlDeviceGetHandleByIndex_v2.side_effect = handle
        lib.nvmlDeviceGetMemoryInfo.side_effect = memory
        return lib

    def test_each_successful_init_owns_one_shutdown(self):
        lib = self.library()
        with mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib):
            first, second = telemetry._Nvml(0), telemetry._Nvml(1)
        try:
            self.assertTrue(first.ok())
            self.assertTrue(second.ok())
            first.close()
            first.close()
            self.assertFalse(first.ok())
            self.assertTrue(second.ok())
            self.assertEqual(lib.nvmlShutdown.call_count, 1)
            self.assertEqual(second.read()["mem_total"], 8 << 30)
        finally:
            first.close()
            second.close()
        self.assertEqual(lib.nvmlInit_v2.call_count, 2)
        self.assertEqual(lib.nvmlShutdown.call_count, 2)

    def test_failed_init_does_not_shutdown_an_unowned_reference(self):
        lib = self.library()
        lib.nvmlInit_v2.return_value = 1
        with mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib):
            reader = telemetry._Nvml()
        self.assertFalse(reader.ok())
        reader.close()
        lib.nvmlShutdown.assert_not_called()

    def test_handle_lookup_failures_release_the_successful_init(self):
        for error in (1, AttributeError("missing symbol"), OSError("device disappeared")):
            with self.subTest(error=error):
                lib = self.library()
                if isinstance(error, Exception):
                    lib.nvmlDeviceGetHandleByIndex_v2.side_effect = error
                else:
                    lib.nvmlDeviceGetHandleByIndex_v2.side_effect = None
                    lib.nvmlDeviceGetHandleByIndex_v2.return_value = error
                with mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib):
                    reader = telemetry._Nvml()
                self.assertFalse(reader.ok())
                lib.nvmlShutdown.assert_called_once_with()
                reader.close()
                lib.nvmlShutdown.assert_called_once_with()

    def test_missing_library_or_shutdown_never_acquires_a_reference(self):
        with mock.patch.object(telemetry.ctypes, "CDLL", side_effect=OSError("not installed")):
            reader = telemetry._Nvml()
        self.assertFalse(reader.ok())
        reader.close()
        lib = self.library()
        del lib.nvmlShutdown
        with mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib):
            reader = telemetry._Nvml()
        self.assertFalse(reader.ok())
        lib.nvmlInit_v2.assert_not_called()

    def test_temporary_readers_close_on_success_and_unreadable_memory(self):
        svc = server.Service.__new__(server.Service)
        svc.gpu_index = 3
        for query in (telemetry.free_vram_mib, svc.free_vram_mib):
            for result in (0, 1):
                with self.subTest(query=query, result=result):
                    lib = self.library()
                    if result:
                        lib.nvmlDeviceGetMemoryInfo.side_effect = None
                        lib.nvmlDeviceGetMemoryInfo.return_value = result
                    with mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib):
                        self.assertEqual(query(), None if result else 5 << 10)
                    lib.nvmlInit_v2.assert_called_once_with()
                    lib.nvmlShutdown.assert_called_once_with()
                    index = lib.nvmlDeviceGetHandleByIndex_v2.call_args.args[0].value
                    self.assertEqual(index, 3 if query == svc.free_vram_mib else 0)

    def test_temporary_reader_closes_when_read_raises(self):
        lib = self.library()
        with mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib), \
                mock.patch.object(telemetry._Nvml, "read", side_effect=RuntimeError("read failed")):
            with self.assertRaisesRegex(RuntimeError, "read failed"):
                telemetry.free_vram_mib()
        lib.nvmlShutdown.assert_called_once_with()

    def test_main_releases_nvml_before_closing_engine(self):
        lib = self.library()
        service = []
        engine = mock.Mock(max_context=4096)
        real_serve = server.serve

        def serve(svc, **kwargs):
            service.append(svc)
            return real_serve(svc, **kwargs)

        def close_engine():
            lib.nvmlShutdown.assert_called_once_with()
            self.assertFalse(service[0].telemetry._thread.is_alive())

        engine.close.side_effect = close_engine
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(server.sys, "argv", ["server", "--port", "0", "--tokenizer", d]), \
                mock.patch.dict(os.environ, {"STRATA_API_KEY": "test-key"}), \
                mock.patch.object(telemetry.ctypes, "CDLL", return_value=lib), \
                mock.patch.object(server, "MockEngine", return_value=engine), \
                mock.patch.object(server, "hub_from_config", return_value=None), \
                mock.patch.object(server, "serve", side_effect=serve), \
                mock.patch.object(server.Service, "start_idle_unload"), \
                mock.patch.object(server.signal, "signal"), \
                mock.patch.object(server.time, "sleep", side_effect=KeyboardInterrupt):
            self.assertEqual(server.main(), 0)
        lib.nvmlInit_v2.assert_called_once_with()
        lib.nvmlShutdown.assert_called_once_with()
        engine.close.assert_called_once_with()


class SamplerOwnership(unittest.TestCase):
    def reader(self):
        reader = mock.Mock(spec=["ok", "name", "read", "close"])
        reader.ok.return_value = True
        reader.name.return_value = "test GPU"
        reader.read.return_value = {}
        return reader

    def test_close_joins_inflight_sample_before_releasing_all_readers(self):
        readers = [self.reader(), self.reader()]
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()

        def read():
            entered.set()
            release.wait()
            return {}

        readers[0].read.side_effect = read
        with mock.patch.object(telemetry, "gpu_reader", side_effect=readers):
            sampler = telemetry.Telemetry(gpu_indices=[0, 1])
        closer = threading.Thread(target=lambda: (sampler.close(), closed.set()), daemon=True)
        try:
            self.assertTrue(entered.wait(2))
            closer.start()
            self.assertTrue(sampler._stop.wait(2))
            self.assertFalse(closed.is_set())
            for reader in readers:
                reader.close.assert_not_called()
            release.set()
            closer.join(2)
            self.assertTrue(closed.is_set())
            self.assertFalse(sampler._thread.is_alive())
            sampler.close()
            for reader in readers:
                reader.close.assert_called_once_with()
        finally:
            release.set()
            sampler.close()
            if closer.ident is not None:
                closer.join(2)

    def test_close_wakes_the_sampler_without_waiting_for_its_interval(self):
        reader = self.reader()
        with mock.patch.object(telemetry, "gpu_reader", return_value=reader):
            sampler = telemetry.Telemetry()
        sampler.close()
        self.assertTrue(sampler._stop.is_set())
        self.assertFalse(sampler._thread.is_alive())
        reader.close.assert_called_once_with()

    def test_constructor_failure_releases_readers_already_created(self):
        first = self.reader()
        with mock.patch.object(telemetry, "gpu_reader", side_effect=[first, RuntimeError("reader failed")]):
            with self.assertRaisesRegex(RuntimeError, "reader failed"):
                telemetry.Telemetry(gpu_indices=[0, 1])
        first.close.assert_called_once_with()

    def test_thread_start_failure_releases_readers(self):
        reader = self.reader()
        with mock.patch.object(telemetry, "gpu_reader", return_value=reader), \
                mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("thread failed")):
            with self.assertRaisesRegex(RuntimeError, "thread failed"):
                telemetry.Telemetry()
        reader.close.assert_called_once_with()

    def test_filtered_readers_are_still_owned_and_closed(self):
        readers = [self.reader(), self.reader()]
        readers[0].ok.return_value = False
        with mock.patch.object(telemetry, "gpu_reader", side_effect=readers):
            sampler = telemetry.Telemetry(gpu_indices=[0, 1])
        try:
            self.assertEqual(sampler.gpus, [(1, readers[1])])
        finally:
            sampler.close()
        for reader in readers:
            reader.close.assert_called_once_with()


class ServingOwnership(unittest.TestCase):
    def service(self):
        tok = server.ByteTokenizer()
        return server.Service(server.MockEngine(tok, "hello"), tok,
                              server.ChatTemplate(Path(__file__).parent / "chat_template.jinja"))

    def test_server_close_releases_its_sampler(self):
        with mock.patch.object(telemetry.ctypes, "CDLL", side_effect=OSError("no driver")):
            svc = self.service()
            httpd = server.serve(svc, port=0)
        sampler = svc.telemetry
        try:
            self.assertTrue(sampler._thread.is_alive())
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertFalse(sampler._thread.is_alive())
        httpd.server_close()

    def test_bind_failure_does_not_start_telemetry(self):
        svc = self.service()
        with mock.patch.object(server, "Server", side_effect=OSError("port occupied")), \
                mock.patch.object(svc, "start_telemetry") as start:
            with self.assertRaisesRegex(OSError, "port occupied"):
                server.serve(svc, port=0)
        start.assert_not_called()

    def test_http_thread_start_failure_closes_sampler_and_socket(self):
        svc = self.service()
        httpd = server.Server(("127.0.0.1", 0), server.BaseHTTPRequestHandler)
        start = threading.Thread.start

        def start_thread(thread):
            if thread._target == httpd.serve_forever:
                raise RuntimeError("HTTP thread failed")
            start(thread)

        try:
            with mock.patch.object(telemetry.ctypes, "CDLL", side_effect=OSError("no driver")), \
                    mock.patch.object(server, "Server", return_value=httpd), \
                    mock.patch.object(threading.Thread, "start", autospec=True, side_effect=start_thread):
                with self.assertRaisesRegex(RuntimeError, "HTTP thread failed"):
                    server.serve(svc, port=0)
            self.assertFalse(svc.telemetry._thread.is_alive())
            self.assertEqual(httpd.socket.fileno(), -1)
        finally:
            httpd.server_close()

    def test_sampler_startup_failure_closes_socket(self):
        svc = self.service()
        httpd = server.Server(("127.0.0.1", 0), server.BaseHTTPRequestHandler)
        try:
            with mock.patch.object(server, "Server", return_value=httpd), \
                    mock.patch.object(svc, "start_telemetry", side_effect=RuntimeError("sampler failed")):
                with self.assertRaisesRegex(RuntimeError, "sampler failed"):
                    server.serve(svc, port=0)
            self.assertEqual(httpd.socket.fileno(), -1)
        finally:
            httpd.server_close()



if __name__ == "__main__":
    unittest.main()
