"""Linux PSI and cgroup-v2 counters for the frontend's once-per-second sampler.

PressureReader(boot_id).snapshot() performs bounded, independent file reads; it
never computes rates or retains counters. Totals named *_usec are cumulative
microseconds. Null values have a bounded code in that file's `reasons` map.
Host CPU full pressure is intentionally absent (it is not defined system-wide).

The cgroup is the frontend/service's scope, including any other work in that
scope and its descendants, NOT exclusive engine accounting. Identity is the
resolved filesystem path and the directory's device/inode. Membership and
mountinfo are resolved anew every sample; an open directory pins file reads to
that identity. A membership change during those reads invalidates that scope's
sample. These are sequential observations, not an atomic kernel snapshot.

Mount roots and membership share cgroup-namespace coordinates; '/' membership
only maps to a mount rooted at '/'. Inherited mounts rooted outside that
namespace are not guessed to be the current scope. See cgroup_namespaces(7):
https://man7.org/linux/man-pages/man7/cgroup_namespaces.7.html
"""
from __future__ import annotations

import errno
import os
import re
import sys
import time

_FILE_LIMIT = 64 * 1024
_MOUNTINFO_LIMIT = 1024 * 1024
_MEMORY_EVENTS = ("low", "high", "max", "oom", "oom_kill", "oom_group_kill")
_CPU_STAT = ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec")
_PSI_FIELDS = ("some_total_usec", "full_total_usec")
_MOUNT_ESCAPE = re.compile(r"\\(040|011|012|134)")
_ESCAPE_CHARS = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}


def _os_reason(error):
    if error.errno in (errno.ENOENT, errno.ENOTDIR):
        return "missing"
    if error.errno in (errno.EACCES, errno.EPERM):
        return "permission_denied"
    return "unreadable"


def _read_text(path, *, dir_fd=None, limit=_FILE_LIMIT):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC, dir_fd=dir_fd)
        with os.fdopen(fd, "rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            return None, "too_large"
        return data.decode("utf-8"), None
    except OSError as error:
        return None, _os_reason(error)
    except UnicodeError:
        return None, "malformed"


def _unavailable(fields, reason):
    return {**dict.fromkeys(fields), "reasons": dict.fromkeys(fields, reason)}


def _counter(text):
    # Kernel counters are unsigned 64-bit integers, never floating-point rates.
    if not text or len(text) > 20 or not text.isascii() or not text.isdecimal():
        return None
    value = int(text)
    return value if value <= (1 << 64) - 1 else None


def _read_counters(path, fields, *, dir_fd=None, psi=False):
    text, reason = _read_text(path, dir_fd=dir_fd)
    if text is None:
        return _unavailable(fields, reason or "unreadable")
    out = _unavailable(fields, "missing_field")
    seen = set()
    for line in text.split("\n"):
        parts = line.split()
        if not parts:
            continue
        field = parts[0] + "_total_usec" if psi else parts[0]
        if field not in fields:
            continue
        values = [p[6:] for p in parts[1:] if p.startswith("total=")] if psi else parts[1:]
        value = _counter(values[0]) if len(values) == 1 else None
        if field in seen or value is None:
            out[field] = None
            out["reasons"][field] = "malformed" if values or field in seen else "missing_field"
        else:
            out[field] = value
            out["reasons"].pop(field, None)
        seen.add(field)
    return out


def _components(path, *, parents=False):
    if not path.startswith("/") or "\0" in path:
        return None
    parts = tuple(path[1:].split("/")) if path != "/" else ()
    if any(p in ("", ".") or (p == ".." and not parents) for p in parts):
        return None
    return parts


def _mount_path(text):
    # Decode once, including literal backslashes; do not unescape the result again.
    if "\\" in _MOUNT_ESCAPE.sub("", text):
        return None
    return _MOUNT_ESCAPE.sub(lambda match: _ESCAPE_CHARS[match[1]], text)


def _membership(proc_root):
    text, reason = _read_text(os.path.join(proc_root, "self", "cgroup"))
    if text is None:
        return None, "membership_" + (reason or "unreadable")
    paths = [line.split(":", 2)[2] for line in text.split("\n") if line.startswith("0::")]
    if not paths:
        return None, "cgroup_v2_absent"
    if len(paths) != 1 or _components(paths[0], parents=True) is None:
        return None, "membership_malformed"
    return paths[0], None


def _mount_candidates(proc_root, membership):
    text, reason = _read_text(os.path.join(proc_root, "self", "mountinfo"), limit=_MOUNTINFO_LIMIT)
    if text is None:
        return [], "mountinfo_" + (reason or "unreadable")
    member_parts = _components(membership, parents=True)
    if member_parts is None:
        return [], "membership_malformed"
    candidates = []
    found = malformed = False
    for line in text.split("\n"):
        if not line:
            continue
        before, separator, after = line.partition(" - ")
        tail = after.split()
        if not separator or not tail:
            malformed = True
            continue
        if tail[0] != "cgroup2":
            continue
        found = True
        fields = before.split()
        if len(fields) < 6 or len(tail) < 3:
            malformed = True
            continue
        root, mountpoint = _mount_path(fields[3]), _mount_path(fields[4])
        root_parts = _components(root, parents=True) if root is not None else None
        if root_parts is None or mountpoint is None or _components(mountpoint) is None:
            malformed = True
            continue
        device = fields[2].split(":")
        if len(device) != 2 or any(_counter(part) is None for part in device):
            malformed = True
            continue
        if member_parts[:len(root_parts)] != root_parts:
            continue
        relative = member_parts[len(root_parts):]
        if ".." in relative:       # never traverse outside the visible mount
            continue
        path = os.path.join(mountpoint, *relative)
        candidates.append((len(root_parts), path, tuple(int(p) for p in device)))
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return candidates, None if candidates else ("mountinfo_malformed" if malformed else
                                               "scope_unmapped" if found else "cgroup2_mount_absent")


def _empty_cgroup(reason):
    return {"scope": "frontend_service", "identity": None, "reason": reason,
            **{resource: _unavailable(_PSI_FIELDS, reason) for resource in ("cpu", "memory", "io")},
            "memory_events": _unavailable(_MEMORY_EVENTS, reason),
            "cpu_stat": _unavailable(_CPU_STAT, reason)}


class PressureReader:
    def __init__(self, boot_id: str, proc_root: str | os.PathLike[str] = "/proc"):
        self.boot_id = boot_id
        self.proc_root = os.fspath(proc_root)

    def _cgroup(self):
        membership, reason = _membership(self.proc_root)
        if reason:
            return _empty_cgroup(reason)
        candidates, reason = _mount_candidates(self.proc_root, membership)
        if reason:
            return _empty_cgroup(reason)
        reason = "scope_unmapped"
        for _, path, device in candidates:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            except OSError as error:
                reason = "scope_" + _os_reason(error)
                continue
            try:
                stat = os.fstat(fd)
                if (os.major(stat.st_dev), os.minor(stat.st_dev)) != device:
                    reason = "mount_changed"
                    continue
                out = {"scope": "frontend_service", "reason": None,
                       "identity": {"path": path, "device": stat.st_dev, "inode": stat.st_ino}}
                for resource in ("cpu", "memory", "io"):
                    out[resource] = _read_counters(resource + ".pressure", _PSI_FIELDS, dir_fd=fd, psi=True)
                out["memory_events"] = _read_counters("memory.events", _MEMORY_EVENTS, dir_fd=fd)
                out["cpu_stat"] = _read_counters("cpu.stat", _CPU_STAT, dir_fd=fd)
                current, reason = _membership(self.proc_root)
                if reason or current != membership:
                    return _empty_cgroup(reason or "scope_changed")
                return out
            except OSError as error:
                reason = "scope_" + _os_reason(error)
            finally:
                os.close(fd)
        return _empty_cgroup(reason)

    def snapshot(self):
        started, sampled_at = time.monotonic(), time.time()
        host = {}
        supported = sys.platform.startswith("linux")
        for resource in ("cpu", "memory", "io"):
            fields = _PSI_FIELDS[:1] if resource == "cpu" else _PSI_FIELDS
            try:
                host[resource] = _read_counters(os.path.join(self.proc_root, "pressure", resource), fields, psi=True) \
                    if supported else _unavailable(fields, "unsupported")
            except Exception:  # noqa: BLE001 - a diagnostic failure cannot hide other telemetry
                host[resource] = _unavailable(fields, "unreadable")
        try:
            cgroup = self._cgroup() if supported else _empty_cgroup("unsupported")
        except Exception:  # noqa: BLE001 - keep failures local to this optional diagnostic scope
            cgroup = _empty_cgroup("unreadable")
        return {"schema_version": 1, "boot_id": self.boot_id,
                "sample_started_monotonic_s": started, "sample_finished_monotonic_s": time.monotonic(),
                "sampled_at_unix_s": sampled_at, "host": host, "cgroup": cgroup}
