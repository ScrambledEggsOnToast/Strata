"""Real temporary proc/cgroup files exercise pressure sampling without an engine or GPU."""
import errno
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import serve.psi as psi
from serve import server, telemetry


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux proc and directory-fd semantics")
class PressureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.proc = self.base / "proc"
        (self.proc / "pressure").mkdir(parents=True)
        (self.proc / "self").mkdir()
        self.mount = self.base / "cgroup"
        self.mount.mkdir()
        # Use real directory-fd mount IDs for ordinary temporary-directory fixtures.
        # The shadow regression below models the one bit requiring CAP_SYS_ADMIN.
        (self.proc / "self" / "fdinfo").symlink_to("/proc/self/fdinfo", target_is_directory=True)
        fd = os.open(self.mount, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fdinfo = (self.proc / "self" / "fdinfo" / str(fd)).read_bytes()
        finally:
            os.close(fd)
        ids = [line.split(b":", 1)[1] for line in fdinfo.split(b"\n") if line.startswith(b"mnt_id:")]
        if len(ids) != 1:
            self.skipTest("Linux directory fdinfo mnt_id is required")
        self.mount_id = int(ids[0])
        self.member = "/service"
        self.membership(self.member)
        self.mountinfo(self.mount_line())
        self.scope = self.mount / "service"
        self.populate(self.scope)
        for resource, some, full in (("cpu", 9007199254740993, 999), ("memory", 31, 17), ("io", 42, 19)):
            (self.proc / "pressure" / resource).write_text(self.pressure(some, full), encoding="utf-8")
        self.reader = psi.PressureReader("server-boot", proc_root=self.proc)

    @staticmethod
    def pressure(some, full):
        return (f"some avg10=0.00 avg60=0.00 avg300=0.00 total={some}\n"
                f"full avg10=0.00 avg60=0.00 avg300=0.00 total={full}\n")

    @staticmethod
    def escape(path):
        return os.fsencode(path).replace(b"\\", rb"\134").replace(b" ", rb"\040").replace(b"\t", rb"\011").replace(b"\n", rb"\012")

    def mount_line(self, *, root="/", mountpoint=None, mount_id=None, filesystem="cgroup2", device=None):
        mountpoint = self.mount if mountpoint is None else mountpoint
        mount_id = self.mount_id if mount_id is None else mount_id
        stat = self.mount.stat()
        device = f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}" if device is None else device
        return (f"{mount_id} 20 {device} ".encode("ascii") + self.escape(root) + b" " +
                self.escape(mountpoint) + b" rw,nosuid shared:7 master:2 - " +
                filesystem.encode("ascii") + b" cgroup rw\n")

    def mountinfo(self, text):
        (self.proc / "self" / "mountinfo").write_bytes(text.encode("utf-8") if isinstance(text, str) else text)

    def membership(self, path):
        (self.proc / "self" / "cgroup").write_bytes(b"9:memory:/legacy\n0::" + os.fsencode(path) + b"\n")

    def populate(self, directory, *, some=101, full=23, usage=987654321):
        directory.mkdir(parents=True, exist_ok=True)
        for resource in ("cpu", "memory", "io"):
            (directory / (resource + ".pressure")).write_text(self.pressure(some, full), encoding="utf-8")
        (directory / "memory.events").write_text(
            "low 0\nhigh 12\nmax 5\noom 3\noom_kill 2\noom_group_kill 1\nfuture_counter 999\n", encoding="utf-8")
        (directory / "cpu.stat").write_text(
            f"usage_usec {usage}\nuser_usec 50\nsystem_usec 20\nnr_periods 90\nnr_throttled 7\nthrottled_usec 800\n",
            encoding="utf-8")

    def test_valid_totals_identity_times_and_allowlists(self):
        with mock.patch.object(psi.time, "monotonic", side_effect=(10.0, 10.25)), \
                mock.patch.object(psi.time, "time", return_value=1234567890.5):
            sample = self.reader.snapshot()
        self.assertEqual(sample["schema_version"], 1)
        self.assertEqual(sample["boot_id"], "server-boot")
        self.assertEqual(sample["sample_started_monotonic_s"], 10.0)
        self.assertEqual(sample["sample_finished_monotonic_s"], 10.25)
        self.assertEqual(sample["sampled_at_unix_s"], 1234567890.5)
        self.assertEqual(sample["host"]["cpu"], {"some_total_usec": 9007199254740993, "reasons": {}})
        self.assertIs(type(sample["host"]["cpu"]["some_total_usec"]), int)
        self.assertEqual(sample["host"]["memory"], {"some_total_usec": 31, "full_total_usec": 17, "reasons": {}})
        self.assertEqual(sample["host"]["io"]["full_total_usec"], 19)
        cgroup = sample["cgroup"]
        stat = self.scope.stat()
        self.assertEqual(cgroup["identity"], {"path": str(self.scope), "device": stat.st_dev, "inode": stat.st_ino})
        self.assertEqual(cgroup["scope"], "frontend_service")
        self.assertIsNone(cgroup["reason"])
        for resource in ("cpu", "memory", "io"):
            self.assertEqual(cgroup[resource], {"some_total_usec": 101, "full_total_usec": 23, "reasons": {}})
        self.assertEqual(cgroup["memory_events"], {"low": 0, "high": 12, "max": 5, "oom": 3,
                                                 "oom_kill": 2, "oom_group_kill": 1, "reasons": {}})
        self.assertEqual(cgroup["cpu_stat"], {"usage_usec": 987654321, "nr_periods": 90, "nr_throttled": 7,
                                            "throttled_usec": 800, "reasons": {}})
        self.assertEqual(json.loads(json.dumps(sample))["host"]["cpu"]["some_total_usec"], 9007199254740993)

    def test_namespace_root_and_subtree_mount_mapping(self):
        for member, root, relative in (("/", "/", ""), ("/tenant/job", "/tenant", "job"),
                                       ("/tenant", "/tenant", "")):
            with self.subTest(member=member, root=root):
                mount = self.base / ("scope-" + str(len(member)) + "-" + str(len(root)))
                directory = mount / relative
                self.populate(directory, some=222)
                self.membership(member)
                self.mountinfo(self.mount_line(root=root, mountpoint=mount))
                cgroup = self.reader.snapshot()["cgroup"]
                self.assertEqual(cgroup["identity"]["path"], str(directory))
                self.assertEqual(cgroup["cpu"]["some_total_usec"], 222)

    def test_unrelated_or_inherited_mount_is_not_namespace_root(self):
        self.membership("/")
        for root in ("/foreign", "/.."):
            with self.subTest(root=root):
                self.populate(self.mount, some=999)
                self.mountinfo(self.mount_line(root=root))
                cgroup = self.reader.snapshot()["cgroup"]
                self.assertIsNone(cgroup["identity"])
                self.assertEqual(cgroup["reason"], "scope_unmapped")
                self.assertIsNone(cgroup["cpu"]["some_total_usec"])

    def test_mapping_uses_components_and_prefers_specific_accessible_mount(self):
        self.membership("/teamwork/task")
        specific = self.base / "specific"
        self.populate(specific / "task", some=333)
        self.populate(self.mount / "teamwork" / "task", some=444)
        false_prefix = self.base / "false-prefix"
        self.populate(false_prefix / "work" / "task", some=999)
        self.mountinfo(self.mount_line(root="/team", mountpoint=false_prefix) + self.mount_line() +
                       self.mount_line(root="/teamwork", mountpoint=specific))
        cgroup = self.reader.snapshot()["cgroup"]
        self.assertEqual(cgroup["identity"]["path"], str(specific / "task"))
        self.assertEqual(cgroup["cpu"]["some_total_usec"], 333)
        # A stale/hidden more-specific mount must not hide another valid cgroup2 mapping.
        (specific / "task").rename(specific / "gone")
        cgroup = self.reader.snapshot()["cgroup"]
        self.assertEqual(cgroup["identity"]["path"], str(self.mount / "teamwork" / "task"))
        self.assertEqual(cgroup["cpu"]["some_total_usec"], 444)
        self.mountinfo(self.mount_line(root="/team", mountpoint=false_prefix))
        self.assertEqual(self.reader.snapshot()["cgroup"]["reason"], "scope_unmapped")

    def test_escaped_mountinfo_paths_and_membership_colons(self):
        mount = self.base / "space tab\tnewline\nback\\slash"
        root = r"/tenant name\040literal:part"
        directory = mount / "service:child"
        self.populate(directory, some=555)
        self.membership(root + "/service:child")
        self.mountinfo(self.mount_line(root=root, mountpoint=mount))
        cgroup = self.reader.snapshot()["cgroup"]
        self.assertEqual(cgroup["identity"]["path"], str(directory))
        self.assertEqual(cgroup["io"]["some_total_usec"], 555)

    def test_non_utf8_metadata_and_scope_names_remain_json_safe(self):
        # Neither unrelated mount names nor legacy-controller paths invalidate v2.
        unrelated = self.mount_line(mountpoint=os.fsdecode(os.fsencode(self.base) + b"/foreign-\xff"),
                                    filesystem="tmpfs")
        self.mountinfo(unrelated + self.mount_line())
        (self.proc / "self" / "cgroup").write_bytes(b"9:memory:/legacy-\xff\n0::/service\n")
        sample = self.reader.snapshot()
        self.assertEqual(sample["cgroup"]["cpu_stat"]["usage_usec"], 987654321)
        self.assertIsNone(sample["cgroup"]["reason"])
        json.dumps(sample, ensure_ascii=False).encode("utf-8")

        mount = self.base / os.fsdecode(b"cgroup-\xff")
        root = os.fsdecode(b"/tenant-\xfe")
        relative = os.fsdecode(b"service-\xfd")
        directory = mount / relative
        self.populate(directory, some=666)
        self.membership(root + "/" + relative)
        self.mountinfo(unrelated + self.mount_line(root=root, mountpoint=mount))
        sample = self.reader.snapshot()
        cgroup = sample["cgroup"]
        self.assertIsNone(cgroup["reason"])
        self.assertEqual(cgroup["identity"]["path"], os.fsencode(directory).decode("utf-8", "backslashreplace"))
        self.assertEqual(cgroup["identity"]["inode"], directory.stat().st_ino)
        self.assertEqual(cgroup["io"]["some_total_usec"], 666)
        json.dumps(sample, ensure_ascii=False).encode("utf-8")

    def test_nbsp_and_carriage_return_are_path_bytes_not_field_delimiters(self):
        mount = self.base / "mount\u00a0name\rwith space\tand\\slash"
        root = "/tenant\u00a0name\rwith space\tand\\slash"
        directory = mount / "service\u00a0name\r"
        self.populate(directory, some=777)
        self.membership(root + "/" + directory.name)
        self.mountinfo(self.mount_line(root=root, mountpoint=mount))
        sample = self.reader.snapshot()
        self.assertIsNone(sample["cgroup"]["reason"])
        self.assertEqual(sample["cgroup"]["identity"]["path"], str(directory))
        self.assertEqual(sample["cgroup"]["memory"]["some_total_usec"], 777)
        json.dumps(sample, ensure_ascii=False).encode("utf-8")

    def test_absent_files_are_null_without_suppressing_other_files(self):
        (self.proc / "pressure" / "cpu").unlink()
        (self.scope / "io.pressure").unlink()
        (self.scope / "memory.events").write_text("low 0\noom_kill 4\n", encoding="utf-8")
        sample = self.reader.snapshot()
        self.assertEqual(sample["host"]["cpu"], {"some_total_usec": None, "reasons": {"some_total_usec": "missing"}})
        self.assertEqual(sample["host"]["memory"]["full_total_usec"], 17)
        cgroup = sample["cgroup"]
        self.assertEqual(cgroup["io"]["reasons"], {"some_total_usec": "missing", "full_total_usec": "missing"})
        self.assertIsNone(cgroup["io"]["some_total_usec"])
        self.assertEqual(cgroup["cpu"]["some_total_usec"], 101)
        self.assertEqual(cgroup["cpu_stat"]["usage_usec"], 987654321)
        self.assertEqual(cgroup["memory_events"]["low"], 0)
        self.assertIsNone(cgroup["memory_events"]["oom"])
        self.assertEqual(cgroup["memory_events"]["reasons"]["oom"], "missing_field")
        self.assertEqual(cgroup["memory_events"]["oom_kill"], 4)

    def test_malformed_fields_are_local_and_do_not_add_labels(self):
        (self.proc / "pressure" / "memory").write_text(
            "some avg10=0.00 total=bad\nfull avg10=0.00 total=6\nunknown total=123\n", encoding="utf-8")
        (self.scope / "cpu.stat").write_text(
            "usage_usec -5\nnr_throttled 18446744073709551616\nthrottled_usec 8\nthrottled_usec 9\n"
            "unknown 123\n", encoding="utf-8")
        (self.scope / "cpu.pressure").write_text("some avg10=0.00\nfull total=0\n", encoding="utf-8")
        (self.scope / "io.pressure").write_bytes(b"some total=2\xff\n")
        sample = self.reader.snapshot()
        memory = sample["host"]["memory"]
        self.assertIsNone(memory["some_total_usec"])
        self.assertEqual(memory["reasons"], {"some_total_usec": "malformed"})
        self.assertEqual(memory["full_total_usec"], 6)
        cgroup = sample["cgroup"]
        self.assertEqual(cgroup["cpu_stat"]["reasons"], {"usage_usec": "malformed", "nr_periods": "missing_field",
                                                        "nr_throttled": "malformed", "throttled_usec": "malformed"})
        self.assertTrue(all(cgroup["cpu_stat"][key] is None for key in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")))
        self.assertNotIn("unknown", cgroup["cpu_stat"])
        self.assertEqual(cgroup["cpu"]["reasons"], {"some_total_usec": "missing_field"})
        self.assertEqual(cgroup["cpu"]["full_total_usec"], 0)
        self.assertEqual(cgroup["io"]["reasons"]["some_total_usec"], "malformed")
        self.assertEqual(cgroup["memory_events"]["high"], 12)

    def test_permission_like_file_failure_is_local_and_reason_is_bounded(self):
        real_open = os.open

        def open_file(path, flags, *args, **kwargs):
            if str(path) == "memory.pressure" and kwargs.get("dir_fd") is not None:
                raise PermissionError(errno.EACCES, "must-not-leak-raw-error", str(path))
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(psi.os, "open", side_effect=open_file):
            sample = self.reader.snapshot()
        cgroup = sample["cgroup"]
        self.assertEqual(cgroup["memory"]["reasons"], {"some_total_usec": "permission_denied",
                                                     "full_total_usec": "permission_denied"})
        self.assertIsNone(cgroup["memory"]["some_total_usec"])
        self.assertEqual(cgroup["io"]["some_total_usec"], 101)
        self.assertEqual(cgroup["memory_events"]["high"], 12)
        self.assertEqual(sample["host"]["memory"]["some_total_usec"], 31)
        self.assertNotIn("must-not-leak", json.dumps(sample))

    def test_search_only_ancestor_does_not_hide_service_counters(self):
        # Model an unprivileged caller facing an execute/search-only ancestor
        # (mode 0711) whose leaf directory and counter files stay readable.
        # A read-open of the ancestor must fail while O_PATH traversal still
        # yields the service-cgroup counters. The seam denies directory opens
        # lacking O_PATH whenever the mode withholds group/other read
        # permission, so it stays deterministic even when this test process
        # owns the fixture or runs as root.
        ancestor = self.mount
        ancestor.chmod(0o711)
        self.addCleanup(ancestor.chmod, 0o755)
        self.assertEqual(ancestor.stat().st_mode & 0o0777, 0o711)
        real_open = os.open

        def open_directory(path, flags, *args, **kwargs):
            if flags & os.O_DIRECTORY and not flags & os.O_PATH:
                st = os.stat(path, **kwargs)
                if not st.st_mode & 0o0044:  # group/other lack read: search-only
                    raise PermissionError(errno.EACCES, "must-not-leak-raw-error", str(path))
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(psi.os, "open", side_effect=open_directory):
            sample = self.reader.snapshot()
        cgroup = sample["cgroup"]
        self.assertIsNone(cgroup["reason"])
        self.assertEqual(cgroup["identity"]["path"], self.member)
        self.assertEqual(cgroup["cpu"]["some_total_usec"], 101)
        self.assertEqual(cgroup["memory"]["some_total_usec"], 101)
        self.assertEqual(cgroup["io"]["some_total_usec"], 101)
        self.assertEqual(cgroup["memory_events"]["oom_kill"], 2)
        self.assertEqual(cgroup["cpu_stat"]["usage_usec"], 987654321)
        self.assertNotIn("must-not-leak", json.dumps(sample))

    def test_absent_or_malformed_scope_keeps_host_metrics(self):
        for membership, mountinfo, reason in (
                ("9:memory:/legacy\n", self.mount_line(), "cgroup_v2_absent"),
                ("0::relative\n", self.mount_line(), "membership_malformed"),
                ("0::/service\n0::/other\n", self.mount_line(), "membership_malformed"),
                ("0::/service\n", self.mount_line(filesystem="cgroup"), "cgroup2_mount_absent"),
                ("0::/service\n", "31 20 0:1 / /bad - cgroup2 cgroup rw\n", "mountinfo_malformed")):
            with self.subTest(reason=reason):
                (self.proc / "self" / "cgroup").write_text(membership, encoding="utf-8")
                self.mountinfo(mountinfo)
                sample = self.reader.snapshot()
                self.assertEqual(sample["host"]["io"]["some_total_usec"], 42)
                self.assertIsNone(sample["cgroup"]["identity"])
                self.assertEqual(sample["cgroup"]["reason"], reason)
                self.assertIsNone(sample["cgroup"]["cpu_stat"]["usage_usec"])
        (self.proc / "self" / "cgroup").unlink()
        self.assertEqual(self.reader.snapshot()["cgroup"]["reason"], "membership_missing")
        self.membership(self.member)
        (self.proc / "self" / "mountinfo").unlink()
        self.assertEqual(self.reader.snapshot()["cgroup"]["reason"], "mountinfo_missing")

    def test_mount_device_mismatch_does_not_claim_foreign_scope(self):
        stat = self.mount.stat()
        self.mountinfo(self.mount_line(device=f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev) + 1}"))
        sample = self.reader.snapshot()
        self.assertEqual(sample["cgroup"]["reason"], "mount_changed")
        self.assertIsNone(sample["cgroup"]["identity"])
        self.assertEqual(sample["host"]["io"]["some_total_usec"], 42)

    def test_same_device_bind_shadow_is_rejected_in_consumer_metrics(self):
        scope = self.mount / "A" / "service"
        self.populate(scope, some=999)
        self.populate(self.mount, some=888)  # Never substitute root counters.
        self.membership("/A/service")
        directories = (self.mount, self.mount / "A", scope)
        bind_id = self.mount_id + 1000000
        fdinfo = self.proc / "self" / "fdinfo"
        fdinfo.unlink()
        fdinfo.mkdir()
        svc = self.start_service_telemetry()
        real_open = os.open
        for index, shadow in enumerate(directories):
            with self.subTest(shadow=shadow):
                self.mountinfo(self.mount_line() +
                               self.mount_line(root="/B", mountpoint=shadow, mount_id=bind_id))
                shadow_ids = {(path.stat().st_dev, path.stat().st_ino) for path in directories[index:]}
                self.assertEqual(scope.stat().st_dev, self.mount.stat().st_dev)
                counter_reads = []

                def open_file(path, flags, *args, **kwargs):
                    fd = real_open(path, flags, *args, **kwargs)
                    if flags & os.O_DIRECTORY:
                        stat = os.fstat(fd)
                        mount_id = bind_id if (stat.st_dev, stat.st_ino) in shadow_ids else self.mount_id
                        # Model kernel bind-mount provenance only; directory opens,
                        # device/inode values and counter reads remain real.
                        (fdinfo / str(fd)).write_bytes(f"mnt_id:\t{mount_id}\n".encode("ascii"))
                    elif kwargs.get("dir_fd") is not None:
                        counter_reads.append(path)
                    return fd

                with mock.patch.object(psi.os, "open", side_effect=open_file):
                    sample = svc.telemetry.sample()
                with svc.telemetry.lock:
                    svc.telemetry.now = sample
                pressure = svc.metrics()["hardware"]["pressure"]
                self.assertEqual(pressure["cgroup"]["reason"], "mount_changed")
                self.assertIsNone(pressure["cgroup"]["identity"])
                self.assertIsNone(pressure["cgroup"]["cpu"]["some_total_usec"])
                self.assertEqual(pressure["cgroup"]["cpu"]["reasons"]["some_total_usec"], "mount_changed")
                self.assertEqual(pressure["host"]["io"]["some_total_usec"], 42)
                self.assertEqual(counter_reads, [])
                json.dumps(svc.metrics(), ensure_ascii=False).encode("utf-8")

    def test_missing_mount_id_does_not_weaken_scope_validation(self):
        (self.proc / "self" / "fdinfo").unlink()
        (self.proc / "self" / "fdinfo").mkdir()
        sample = self.reader.snapshot()
        self.assertEqual(sample["cgroup"]["reason"], "mount_id_missing")
        self.assertIsNone(sample["cgroup"]["identity"])
        self.assertIsNone(sample["cgroup"]["cpu_stat"]["usage_usec"])
        self.assertEqual(sample["host"]["memory"]["some_total_usec"], 31)

    def test_unavailable_or_malformed_mount_ids_fail_closed(self):
        real_read = psi._read_bytes
        for data, reason in ((b"pos:\t0\n", "mount_id_unavailable"),
                             (b"mnt_id:\tbad\n", "mount_id_malformed"),
                             (b"mnt_id:\t1\nmnt_id:\t1\n", "mount_id_malformed")):
            with self.subTest(reason=reason, data=data):
                def read_file(path, **kwargs):
                    if Path(path).parent == self.proc / "self" / "fdinfo":
                        return data, None
                    return real_read(path, **kwargs)

                with mock.patch.object(psi, "_read_bytes", side_effect=read_file):
                    sample = self.reader.snapshot()
                self.assertEqual(sample["cgroup"]["reason"], reason)
                self.assertIsNone(sample["cgroup"]["identity"])
                self.assertIsNone(sample["cgroup"]["memory_events"]["oom"])
                self.assertEqual(sample["host"]["cpu"]["some_total_usec"], 9007199254740993)

    def test_symlink_within_same_mount_does_not_redirect_membership(self):
        other = self.mount / "other"
        self.populate(other / "service", some=999)
        (self.mount / "redirect").symlink_to(other, target_is_directory=True)
        self.membership("/redirect/service")
        sample = self.reader.snapshot()
        self.assertIn(sample["cgroup"]["reason"], ("scope_missing", "scope_unreadable"))
        self.assertIsNone(sample["cgroup"]["identity"])
        self.assertIsNone(sample["cgroup"]["cpu"]["some_total_usec"])
        self.assertEqual(sample["host"]["io"]["some_total_usec"], 42)

    def test_membership_mount_and_directory_identity_are_revalidated(self):
        first = self.reader.snapshot()["cgroup"]
        other = self.mount / "other"
        self.populate(other, some=7)
        self.membership("/other")
        second = self.reader.snapshot()["cgroup"]
        self.assertEqual(second["identity"]["path"], str(other))
        self.assertNotEqual(first["identity"], second["identity"])
        self.assertEqual(second["cpu"]["some_total_usec"], 7)
        new_mount = self.base / "new-mount"
        self.populate(new_mount / "other", some=5)
        self.mountinfo(self.mount_line(mountpoint=new_mount))
        third = self.reader.snapshot()["cgroup"]
        self.assertEqual(third["identity"]["path"], str(new_mount / "other"))
        self.assertEqual(third["cpu"]["some_total_usec"], 5)
        # Retain the old directory so the filesystem cannot immediately reuse its inode.
        (new_mount / "other").rename(new_mount / "old-other")
        self.populate(new_mount / "other", some=2)
        fourth = self.reader.snapshot()["cgroup"]
        self.assertEqual(third["identity"]["path"], fourth["identity"]["path"])
        self.assertNotEqual(third["identity"]["inode"], fourth["identity"]["inode"])
        self.assertEqual(fourth["cpu"]["some_total_usec"], 2)

    def test_mid_sample_membership_change_invalidates_only_cgroup(self):
        real_read = psi._read_text

        def read_file(path, **kwargs):
            result = real_read(path, **kwargs)
            if str(path) == "cpu.pressure" and kwargs.get("dir_fd") is not None:
                self.membership("/other")
            return result

        with mock.patch.object(psi, "_read_text", side_effect=read_file):
            sample = self.reader.snapshot()
        self.assertEqual(sample["cgroup"]["reason"], "scope_changed")
        self.assertIsNone(sample["cgroup"]["identity"])
        self.assertIsNone(sample["cgroup"]["memory_events"]["oom"])
        self.assertEqual(sample["host"]["cpu"]["some_total_usec"], 9007199254740993)

    def test_directory_fd_keeps_counters_with_the_reported_identity(self):
        real_read = psi._read_text
        old_inode = self.scope.stat().st_ino

        def read_file(path, **kwargs):
            result = real_read(path, **kwargs)
            if str(path) == "cpu.pressure" and kwargs.get("dir_fd") is not None:
                self.scope.rename(self.mount / "previous-service")
                self.populate(self.scope, some=1, usage=2)
            return result

        with mock.patch.object(psi, "_read_text", side_effect=read_file):
            cgroup = self.reader.snapshot()["cgroup"]
        self.assertEqual(cgroup["identity"]["inode"], old_inode)
        self.assertEqual(cgroup["memory"]["some_total_usec"], 101)
        self.assertEqual(cgroup["cpu_stat"]["usage_usec"], 987654321)
        following = self.reader.snapshot()["cgroup"]
        self.assertNotEqual(following["identity"]["inode"], old_inode)
        self.assertEqual(following["memory"]["some_total_usec"], 1)
        self.assertEqual(following["cpu_stat"]["usage_usec"], 2)

    def test_counter_resets_are_visible_without_rate_or_clamping(self):
        before = self.reader.snapshot()
        (self.proc / "pressure" / "cpu").write_text(self.pressure(3, 0), encoding="utf-8")
        (self.scope / "cpu.pressure").write_text(self.pressure(0, 0), encoding="utf-8")
        (self.scope / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n", encoding="utf-8")
        (self.scope / "cpu.stat").write_text("usage_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n", encoding="utf-8")
        after = self.reader.snapshot()
        self.assertEqual(before["cgroup"]["identity"], after["cgroup"]["identity"])
        self.assertEqual(after["boot_id"], before["boot_id"])
        self.assertEqual(after["host"]["cpu"]["some_total_usec"], 3)
        self.assertEqual(after["cgroup"]["cpu"]["some_total_usec"], 0)
        self.assertEqual(after["cgroup"]["memory_events"]["high"], 0)
        self.assertEqual(after["cgroup"]["cpu_stat"]["usage_usec"], 0)
        self.assertGreaterEqual(after["sample_started_monotonic_s"], before["sample_finished_monotonic_s"])

    def test_oversized_file_does_not_hide_other_files(self):
        (self.scope / "memory.events").write_text("x" * (64 * 1024 + 1), encoding="utf-8")
        cgroup = self.reader.snapshot()["cgroup"]
        self.assertIsNone(cgroup["memory_events"]["high"])
        self.assertEqual(cgroup["memory_events"]["reasons"]["high"], "too_large")
        self.assertEqual(cgroup["memory"]["some_total_usec"], 101)
        self.assertEqual(cgroup["cpu_stat"]["usage_usec"], 987654321)

    def start_service_telemetry(self):
        tok = server.ByteTokenizer()
        svc = server.Service(server.MockEngine(tok, "hello"), tok,
                             server.ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        reader_type = psi.PressureReader
        gpu = mock.Mock(spec=["ok", "close", "name"])
        gpu.ok.return_value = False
        with mock.patch.object(psi, "PressureReader", side_effect=lambda boot: reader_type(boot, proc_root=self.proc)), \
                mock.patch.object(telemetry, "gpu_reader", return_value=gpu), \
                mock.patch.object(telemetry.Telemetry, "_loop", return_value=None):
            svc.start_telemetry()
        self.addCleanup(svc.telemetry.close)
        return svc

    def test_service_callback_adds_pressure_without_changing_existing_metrics(self):
        svc = self.start_service_telemetry()
        first = svc.telemetry.sample()
        with svc.status_lock:
            lifecycle_boot = svc._request_activity()["boot_id"]
        self.assertEqual(first["pressure"]["boot_id"], lifecycle_boot)
        self.assertEqual(first["pressure"]["cgroup"]["cpu_stat"]["usage_usec"], 987654321)
        for field in ("tok_s", "tok_s_mean", "prefill_tok_s_mean"):
            self.assertEqual(first[field], 0.0)
        # A missing new diagnostic file cannot prevent existing series from being emitted.
        (self.scope / "cpu.stat").unlink()
        second = svc.telemetry.sample()
        self.assertEqual(second["tok_s"], 0.0)
        self.assertEqual(second["pressure"]["cgroup"]["cpu_stat"]["reasons"]["usage_usec"], "missing")
        with svc.telemetry.lock:
            svc.telemetry.now = second
        self.assertEqual(svc.metrics()["hardware"]["pressure"], second["pressure"])


class UnsupportedPressure(unittest.TestCase):
    def test_non_linux_uses_null_counters_with_a_bounded_reason(self):
        with mock.patch.object(psi.sys, "platform", "win32"), mock.patch.object(psi.os, "open") as opened:
            sample = psi.PressureReader("server-boot").snapshot()
        opened.assert_not_called()
        self.assertEqual(sample["host"]["cpu"], {"some_total_usec": None, "reasons": {"some_total_usec": "unsupported"}})
        self.assertEqual(sample["cgroup"]["reason"], "unsupported")
        self.assertIsNone(sample["cgroup"]["identity"])


if __name__ == "__main__":
    unittest.main()
