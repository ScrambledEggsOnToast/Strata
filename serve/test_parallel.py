"""#465: "parallel": N - several requests decode together in the engine's batch slots (--batch N).

A fake engine (a Python script speaking the engine's GEN / BGEN / BT / BDONE / BADM / BSTOP lines) runs behind the
real StrataEngine and Service, so the server's side is tested without a GPU: requests up to N run at once, more
wait for a slot, /metrics shows the slots, each request's history row is its own, and a conversation's next turn
goes back to the slot that holds it."""
import collections
import contextlib
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import (ATTRIBUTION_OUTCOMES, ByteTokenizer, LifecycleIdentity, Service, SpeculationPolicy,
                          StrataEngine, engine_args, parallel_args, serve)

# The fake engine: one token every STEP seconds per active slot (a "window" serves every active slot at once);
# GEN (solo) streams T lines; BGEN reads the prompt (one T line, DONE), answers BADM and continues in the slot.
# Every reply is the bytes of "ok, done." then <|im_end|> (ByteTokenizer: 257).
FAKE_BATCH = r'''import json, os, queue, sys, threading, time
args = sys.argv[1:]
slots = int(args[args.index("--batch") + 1]) if "--batch" in args else 0
fit = int(args[args.index("--fit") + 1]) if "--fit" in args else slots
fail = "--fail-window" in args
native_admission = "--native-admission" in args and "--retained" in args
native_gates = args[args.index("--native-gates") + 1] if "--native-gates" in args else None
groups = int(args[args.index("--says-groups") + 1]) if "--says-groups" in args else 0   # INFO batch_groups=G (--batch-groups auto)        # #997: the first window over two slots fails
STEP = 0.02
hold = args[args.index("--hold-replies") + 1] if "--hold-replies" in args else None
hold_badm = args[args.index("--hold-badm") + 1] if "--hold-badm" in args else None
drafts = [pair.split(",") for pair in args[args.index("--draft-counters") + 1].split(":")] if "--draft-counters" in args else None
CH = 32
lines, stop = queue.Queue(), threading.Event()
score_cancel = set()
score_lock = threading.Lock()
def reader():
    for l in sys.stdin:
        l = l.strip()
        if l == "STOP":
            stop.set()
        elif l.startswith("SCANCEL "):
            with score_lock:
                score_cancel.add(l.split()[1])
            if "--score-cancel-file" in args:
                open(args[args.index("--score-cancel-file") + 1], "w").close()
        else:
            lines.put(l)
    lines.put(None)
threading.Thread(target=reader, daemon=True).start()
print("INFO engine=0.1.39" + (f" batch_slots={fit}" if fit >= 2 or (("--packed" in args or native_admission) and fit == 1) else "") +
      (" slot_cache=1" if "--slotcache" in args else "") + (f" batch_groups={groups}" if groups > 1 else ""), flush=True)
if "--says-multiplex" in args:
    print("INFO resident_multiplex=1", flush=True)
print("INFO score=1 score_version=1 score_max_prompt=4096 score_max_selected=256 score_max_pending=8", flush=True)
if "--packed" in args:
    print("INFO packed_prefill=1 packed_prefill_version=1 packed_prefill_owners=4", flush=True)
if "--retained" in args:
    print("INFO retained_prompt=1 retained_prompt_version=1 retained_prompt_owners=4", flush=True)
if native_admission:
    print("INFO native_admission=1 native_admission_version=1", flush=True)
    print("EPLAN " + json.dumps({"admission": {"authority": "native", "support": "on"}}), flush=True)
if "--says-budget" in args:
    print("INFO foreground_budget_ms=100 background_share=0.25", flush=True)
if "--says-spec" in args:
    print("INFO spec=4", flush=True)
if "--says-residency" in args:
    for info in args[args.index("--says-residency") + 1].splitlines():
        print("INFO " + info, flush=True)
ctx = int(args[args.index("--says-context") + 1]) if "--says-context" in args else 4096
print(f"READY {ctx} stop", flush=True)
LONG = list(b"LONGREPLY")
def continued(ids):        # (the whole reply, how much of it the prompt already ends with: a request continued)
    ids = [int(x) for x in ids]
    long_one = any(ids[i:i + len(LONG)] == LONG for i in range(len(ids)))
    R = list(b"ok, " + b"la " * 15 + b"done." if long_one else b"ok, done.") + [257]
    return R, max(k for k in range(len(R)) if k == 0 or ids[-k:] == R[:k])
def reply(ids):            # the rest of the reply after what the prompt already ends with
    if native_admission and "--native-reply" in args:
        with open(args[args.index("--native-reply") + 1], "rb") as source:
            return list(source.read()) + [257]
    R, k = continued(ids)
    return R[k:]
active = {}          # slot -> [tokens left, max_new, produced]
stopped = set()      # BSTOPped slots: they end "cancel"
legacy_identity = "--legacy-identity" in args
legacy_serial, legacy_owners, legacy_terminals = 0, {}, {}
legacy_seen = set()
def legacy_done(slot, produced, finish, counts=""):
    identity = legacy_owners.get(slot)
    suffix = f" request={identity[0]} incarnation={identity[1]}" if legacy_identity and identity else ""
    terminal = f"BDONE {slot} {produced} {finish} 1.0{counts}{suffix}"
    print(terminal, flush=True)
    if legacy_identity:
        legacy_terminals[slot] = terminal
        if log:
            log.write("TERMINAL " + terminal + "\n"); log.flush()
# #879: a prompt with NFSOLO / NFADMIT ends its first GEN / BGEN with DONE ... nonfinite (nothing emitted); one with
# NFSLOT ends its slot with BDONE ... nonfinite after its second token
fired, nf_slots = set(), set()
packed, packed_cache, packed_serial = {}, {}, 0
retained_positions, retained_waiting, retained_resumed = {}, {}, set()
# Opt-in #263 peer: registrations own logical tags; only BACK grants a dynamic seat.
# These gates delay work, never the reader, so a saturated queue still accepts controls.
native_owners, native_terminals = {}, set()
native_foreground, native_solo = 0, 0
def native_gate(name, tag=None):
    return bool(native_gates and any(os.path.exists(os.path.join(native_gates, gate))
                                    for gate in (name, f"{name}-{tag}") if tag or gate == name))
def native_log(direction, raw):
    if log:
        log.write("NATIVE " + json.dumps({"monotonic_s": time.monotonic(),
                  "direction": direction, "raw": raw}) + "\n")
        log.flush()
def native_send(raw):
    native_log("received", raw)
    print(raw, flush=True)
def native_accept(owner, force=False):
    if not owner["accepted"] and (force or not native_gate("accept", owner["tag"])):
        service = "" if "--native-missing-service-receipt" in args else " native_service_wait_s=0.000000000"
        native_send(f"BQACCEPT {owner['tag']} -1 0 0" + service)
        owner["accepted"] = True
def native_finish(owner, reason):
    tag, slot = owner["tag"], owner["slot"]
    assert tag not in native_terminals, "duplicate native terminal: " + tag
    native_accept(owner, force=True)
    native_terminals.add(tag)
    prompt = len(owner["ids"] or [])
    generated = len(owner["generated"])
    request, incarnation = owner["identity"] or (0, 0)
    admission_wait = owner["admission_wait"]
    if admission_wait is None:
        admission_wait = max(0.0, time.monotonic() - owner["registered"])
    fresh = owner["position"] - owner["reuse"] if slot >= 0 else 0
    native_send(f"BQDONE {tag} {slot} {request} {incarnation} {generated} {prompt} "
                f"{5 if fresh else 0} {1 if generated else 0} {reason} "
                f"{owner.get('catchup', 0)} {owner.get('catchup', 0)} {owner['reuse']} "
                f"prompt_read={fresh} native_admission_wait_s={admission_wait:.9f} "
                f"native_service_wait_s={owner['service_wait']:.9f}")
    if slot >= 0:
        if generated:
            packed_cache[slot] = (owner["scope"], owner["ids"] + owner["generated"][:-1])
        del packed[slot]
    del native_owners[tag]
def native_choose():
    waiting = [owner for owner in native_owners.values() if owner["slot"] < 0]
    foreground = [owner for owner in waiting if owner["priority"] == "foreground"]
    background = next((owner for owner in waiting if owner["priority"] == "background"), None)
    solo = next((owner for owner in foreground if owner["packing"] == "solo"), None)
    eligible = next((owner for owner in foreground if owner["packing"] == "eligible"), None)
    if background is not None and (not foreground or native_foreground >= 3):
        return background
    if solo is not None and (eligible is None or native_solo < 4):
        return solo
    return eligible or next(iter(foreground), None)
def native_seat():
    global packed_serial, native_foreground, native_solo
    while not native_gate("admission"):
        free = next((slot for slot in range(max(1, fit)) if slot not in packed), None)
        if free is None:
            return
        owner = native_choose()
        # Metadata is the FIFO linearization point, not completion of tokenization.
        if owner is None or owner["ids"] is None or not owner["accepted"]:
            return
        packed_serial += 1
        owner["slot"], owner["identity"] = free, (packed_serial, packed_serial)
        owner["admission_wait"] = max(0.0, time.monotonic() - owner["registered"])
        owner["service_clock"] = time.monotonic()
        old_scope, cached = packed_cache.get(free, ("", []))
        if owner["scope"] and owner["scope"] == old_scope:
            while (owner["reuse"] < min(len(owner["ids"]) - 1, len(cached)) and
                   owner["ids"][owner["reuse"]] == cached[owner["reuse"]]):
                owner["reuse"] += 1
        owner["position"] = owner["reuse"]
        packed[free] = owner
        native_foreground = native_foreground + 1 if owner["priority"] == "foreground" else 0
        native_solo = native_solo + 1 if owner["priority"] == "foreground" and owner["packing"] == "solo" else 0
        service = "" if "--native-missing-service-receipt" in args else " native_service_wait_s=0.000000000"
        native_send(f"BACK {owner['tag']} {free} {packed_serial} {packed_serial} "
                    f"native_admission_wait_s={owner['admission_wait']:.9f}" + service)
def native_tick():
    now = time.monotonic()
    for owner in list(native_owners.values()):
        native_accept(owner)
        reason = "cancel" if owner["cancelled"] else "timeout" if now >= owner["deadline"] else None
        if reason and not (owner["slot"] >= 0 and native_gate("drain", owner["tag"])):
            # Transport-only stand-in for accepted rows of an already issued MTP
            # window: publish them after cancellation, before its drained terminal.
            if (reason == "cancel" and owner["slot"] >= 0 and owner["generated"] and
                    "--native-cancel-catchup" in args):
                count = int(args[args.index("--native-cancel-catchup") + 1])
                prefix = f"{owner['tag']} {owner['slot']} {owner['identity'][0]} {owner['identity'][1]}"
                for _ in range(min(count, owner["limit"] - len(owner["generated"]))):
                    token = owner["tokens"].pop(0) if owner["tokens"] else 257
                    owner["position"] = len(owner["ids"])
                    owner["generated"].append(token)
                    owner["catchup"] = owner.get("catchup", 0) + 1
                    native_send(f"BQT {prefix} {token}")
                    if token == 257:
                        break
            native_finish(owner, reason)
    native_seat()
    for slot, owner in list(packed.items()):
        tag = owner["tag"]
        now = time.monotonic()
        if native_gate("service", tag):
            owner["service_wait"] += max(0.0, now - owner["service_clock"])
            owner["service_clock"] = now
            observed = os.path.join(native_gates, f"observed-service-{tag}")
            if not os.path.exists(observed):
                open(observed, "w").close()
            continue
        owner["service_clock"] = now
        if owner["cancelled"] or now >= owner["deadline"]:
            continue
        marker = bytes(token & 255 for token in owner["ids"])
        if b"ADAPTEREXIT" in marker:
            sys.exit(7)
        if b"REFUSEME" in marker:
            native_finish(owner, "nonfinite")
            continue
        prefix = f"{tag} {slot} {owner['identity'][0]} {owner['identity'][1]}"
        if len(owner["ids"]) > 33 and not owner["resumed"] and "--native-no-suspend" not in args:
            if owner["cursor"] is None:
                count = int(args[args.index("--native-progress-count") + 1]) if "--native-progress-count" in args else 1
                position = min(len(owner["ids"]) - 1, ((max(32, owner["reuse"], count) + 31) // 32) * 32)
                begin = owner.get("progress_next", max(owner["reuse"] + 1, position - count + 1))
                end = min(position + 1, begin + 32)
                for progress in range(begin, end):
                    service = ("" if "--native-missing-service-receipt" in args else
                               f" native_service_wait_s={owner['service_wait']:.9f}")
                    native_send(f"BQPP {prefix} {progress} {len(owner['ids'])} {progress - owner['reuse']}" + service)
                owner["position"], owner["progress_next"] = end - 1, end
                if end <= position:
                    continue
                owner["cursor"] = f"{prefix} {position} 1 1 0 {len(owner['ids']) - 1} 32 0"
                native_send(f"BQSUSPEND {owner['cursor']} chunk")
            continue
        if native_gate("token-" + str(len(owner["generated"])), tag):
            continue
        owner["position"] = len(owner["ids"])
        token = owner["tokens"].pop(0) if owner["tokens"] else 257
        owner["generated"].append(token)
        native_send(f"BQT {prefix} {token}")
        if token == 257 or len(owner["generated"]) >= owner["limit"]:
            native_finish(owner, "stop" if token == 257 else "length")
    # Terminal publication precedes autonomous reuse, even if the old HTTP consumer stalls.
    native_seat()
def native_control(line):
    fields = line.split()
    tag, slot, request, incarnation = fields[1], *map(int, fields[2:5])
    owner = native_owners.get(tag)
    if (owner is None or slot < -1 or (slot >= 0 and slot != owner["slot"]) or
            ((request, incarnation) != (0, 0) and (request, incarnation) != owner["identity"])):
        native_log("ignored", line)
        return
    if fields[0] == "BQCANCEL":
        owner["cancelled"] = True
    elif fields[0] == "BQPRIORITY":
        assert fields[5] in ("foreground", "background"), line
        owner["priority"] = fields[5]
    elif fields[0] == "BQRESUME":
        cursor = owner["cursor"]
        expected = cursor.split() if cursor else []
        if not expected or fields[1:] != expected[:7] + [expected[10]]:
            native_log("ignored", line)
            return
        if not owner["resumed"] and not native_gate("resume", tag):
            native_send(f"BQRESUMED {cursor} client")
            owner["resumed"] = True
def native_frame(line):
    native_log("sent", line)
    fields = line.split()
    if fields[0] == "BQREGISTER":
        assert len(fields) == 6, line
        tag, priority, packing = fields[1:4]
        remaining_ns, capacity = map(int, fields[4:])
        assert tag not in native_owners and tag not in native_terminals, "tag reused: " + line
        assert priority in ("foreground", "background") and packing in ("eligible", "solo"), line
        if capacity < max(1, fit) or capacity > max(1, fit) + 256:
            native_terminals.add(tag)
            native_send(f"BQREJECT {tag} -1 invalid_capacity")
        elif len(native_owners) >= capacity:
            native_terminals.add(tag)
            native_send(f"BQREJECT {tag} -1 queue_full")
        else:
            now = time.monotonic()
            native_owners[tag] = {"tag": tag, "priority": priority, "packing": packing,
                "registered": now, "deadline": now + max(0, remaining_ns) / 1e9,
                "ids": None, "slot": -1, "identity": None, "accepted": False,
                "limit": 0, "tokens": [], "generated": [], "cancelled": False,
                "scope": "", "reuse": 0, "position": 0, "cursor": None, "resumed": False,
                "admission_wait": None, "service_wait": 0.0, "service_clock": now}
    elif fields[0] == "BGENQ1":
        tag, slot, limit = fields[1], int(fields[2]), int(fields[3])
        assert slot == -1, "frontend selected a native seat: " + line
        owner = native_owners.get(tag)
        # An original deadline can expire during rendering; a late publication cannot resurrect it.
        if owner is None and tag in native_terminals:
            native_log("ignored", line)
            return
        assert owner is not None and owner["ids"] is None, "unregistered or repeated input: " + line
        input_capacity = int(args[args.index("--native-input-capacity") + 1]) if "--native-input-capacity" in args else 264
        if (b"ADMISSIONREFUSE" in bytes(int(token) & 255 for token in fields[-1].split(",")) or
                sum(o["ids"] is not None and o["slot"] < 0 for o in native_owners.values()) >= input_capacity):
            native_terminals.add(tag)
            native_send(f"BQREJECT {tag} -1 transport_full")
            del native_owners[tag]
        else:
            owner["ids"] = list(map(int, fields[-1].split(",")))
            owner["limit"], owner["tokens"] = limit, reply(owner["ids"])
            owner["scope"] = next((field[6:] for field in fields[4:-1] if field.startswith("scope=")), "")
    else:
        native_control(line)
    native_tick()
def packed_window():
    for b, owner in list(packed.items()):
        tag, request, incarnation, ids, limit, tokens, generated, cancelled, scope, reuse = owner
        if not cancelled and hold and os.path.exists(hold):
            continue
        prefix = f"{tag} {b} {request} {incarnation}"
        if not cancelled and "--retained" in args and len(ids) > 33 and tag not in retained_resumed:
            if tag not in retained_waiting:
                position = max(32, reuse)
                retained_positions[tag] = position
                cursor = f"{prefix} {position} 1 1 0 {len(ids) - 1} 32 0"
                retained_waiting[tag] = cursor
                print(f"BQPP {prefix} {position} {len(ids)} {position - reuse}", flush=True)
                print(f"BQSUSPEND {cursor} chunk", flush=True)
            continue
        if cancelled:
            finish = "cancel"
        else:
            token = tokens.pop(0) if tokens else 257
            generated.append(token)
            wire_request = request + 1 if "--packed-wrong-id" in args else request
            print(f"BQT {tag} {b} {wire_request} {incarnation} {token}", flush=True)
            finish = "stop" if token == 257 else "length" if len(generated) >= limit else None
        if finish:
            read = len(ids) - reuse if generated else retained_positions.get(tag, reuse) - reuse
            print(f"BQDONE {prefix} {len(generated)} {len(ids)} 5 1 {finish} 0 0 {reuse} prompt_read={read}", flush=True)
            if generated:
                packed_cache[b] = (scope, list(map(int, ids)) + generated[:-1])
            if log:
                log.write(f"BQDONE {tag} {b} {finish}\n"); log.flush()
            del packed[b]
def window():        # one batch window: every active slot one token
    if native_admission:
        native_tick()
        return
    packed_window()
    if hold and os.path.exists(hold):
        return
    if fail and len(active) >= 2:     # as the engine: the reason on stdout, then exit code 1
        print("ERR verify batch: layer 34 never rang (graph finished)", flush=True)
        sys.exit(1)
    for b in sorted(active):
        left, max_new, produced = active[b]
        if b in nf_slots and produced >= 2:
            nf_slots.discard(b)
            legacy_done(b, produced, "nonfinite")
            del active[b]
            continue
        t = left.pop(0) if left else 257
        produced += 1
        print(f"BT {b} {t}", flush=True)
        if log and "--says-budget" in args:
            log.write(f"COMMIT {b} {produced}\n"); log.flush()
        if t == 257 or produced >= max_new:
            fin = 'stop' if t == 257 else 'cancel' if b in stopped else 'length'
            stopped.discard(b)
            accepted, offered = drafts[b % len(drafts)] if drafts else (None, None)
            counts = f" draft_accepted={accepted} draft_offered={offered}" if drafts else ""
            legacy_done(b, produced, fin, counts)
            del active[b]
        else:
            active[b] = [left, max_new, produced]
log = open(args[args.index("--log") + 1], "a") if "--log" in args else None
PPK = int(args[args.index("--pp-every") + 1]) if "--pp-every" in args else 1
READF = "--readfield" in args   # DONE's 15th field: the prompt tokens the engine read (a continued prompt: the 1 it copies)
REUSE = "--reuse" in args     # DONE's reused field as the engine gives it: a continued prompt is all held but its last
def reused(ids):              # token (the request's own output included), a fresh one the first 3 tokens
    return 0 if not REUSE else len(ids) - 1 if continued(ids)[1] else 3
def score_done(request_id, status, result):
    print(f"SDONE {request_id} {status} " + json.dumps(result, separators=(",", ":")), flush=True)
parked = {}
while True:
    try:
        line = lines.get(timeout=STEP if active or packed or (native_admission and native_owners) else None)
    except queue.Empty:
        line = ""
    if line is None or line == "QUIT":
        break
    if native_admission and line.startswith(("BQREGISTER ", "BGENQ1 ", "BQCANCEL ", "BQPRIORITY ", "BQRESUME ")):
        native_frame(line)
        continue
    if line.startswith("BPRIORITY "):
        continue
    if line.startswith("BYIELD "):
        continue                              # too late: the read it was for has ended
    if line.startswith("BSTOP "):
        fields = line.split()
        b = int(fields[1])
        if legacy_identity:
            assert len(fields) == 4, "identity-aware cancellation lost its fence: " + line
            assert tuple(map(int, fields[2:])) in legacy_seen, line
            if log:
                log.write(line + "\n"); log.flush()
            if tuple(map(int, fields[2:])) != legacy_owners.get(b):
                if log:
                    log.write("IGNORED " + line + "\n"); log.flush()
                continue
        parked.pop(b, None)
        if b in active:
            active[b][1] = 0
            stopped.add(b)
        continue
    if line.startswith("BQCANCEL "):
        fields = line.split()
        tag, slot, request, incarnation = fields[1], int(fields[2]), int(fields[3]), int(fields[4])
        owner = packed.get(slot)
        if owner and owner[0] == tag and ((request, incarnation) == (0, 0) or (request, incarnation) == tuple(owner[1:3])):
            owner[7] = True
        continue
    if line.startswith("BQRESUME "):
        fields = line.split()
        tag, slot = fields[1], int(fields[2])
        owner = packed.get(slot)
        if owner and owner[0] == tag:
            cursor = retained_waiting[tag]
            expected = cursor.split()
            assert fields[1:] == expected[:7] + [expected[10]], line
            if log:
                log.write(line + "\n"); log.flush()
            if "--retained-refuse" in args:
                fresh = retained_positions[tag] - owner[9]
                print(f"BQDONE {' '.join(expected[:4])} 0 {len(owner[3])} 5 1 refused 0 0 {owner[9]} prompt_read={fresh}", flush=True)
                del packed[slot]
            elif "--hold-resume" not in args:
                print(f"BQRESUMED {cursor} client", flush=True)
                retained_resumed.add(tag)
        continue
    if line.startswith("BGENQ1 "):
        fields = line.split()
        tag, slot, max_new, ids = fields[1], int(fields[2]), int(fields[3]), fields[-1].split(",")
        if slot in packed:
            print(f"BQREJECT {tag} {slot} physical slot is not free", flush=True)
            continue
        packed_serial += 1
        scope = next((f[6:] for f in fields[4:-1] if f.startswith("scope=")), "")
        old_scope, cached = packed_cache.get(slot, ("", []))
        reuse = 0
        if scope and scope == old_scope:
            while reuse < min(len(ids) - 1, len(cached)) and int(ids[reuse]) == cached[reuse]:
                reuse += 1
        packed[slot] = [tag, packed_serial, packed_serial, ids, max_new, reply(ids), [], False, scope, reuse]
        if log:
            packing = next((f[8:] for f in fields[4:-1] if f.startswith("packing=")), "")
            priority = next((f[9:] for f in fields[4:-1] if f.startswith("priority=")), "")
            log.write(f"BGENQ {tag} {slot} {len(ids)} packing={packing} priority={priority}\n"); log.flush()
        print(f"BACK {tag} {slot} {packed_serial} {packed_serial}", flush=True)
        if "--packed-fatal" in args:
            print("FATAL packed admission failed: destination lease invalid", flush=True)
        if "--packed-duplicate-ack" in args:
            print(f"BACK {tag} {slot} {packed_serial} {packed_serial}", flush=True)
        continue
    if line.startswith("SCORE "):
        _, request_id, payload = line.split(" ", 2)
        fields = payload.split()
        prompt = [int(t) for t in fields[1].removeprefix("prompt=").split(",")]
        selected = [int(t) for t in fields[2].removeprefix("selected=").split(",")]
        result = {"version": 1, "request_id": request_id, "generated_tokens": 0}
        if log:
            log.write(f"SCORE {request_id} {len(prompt)} {len(active)}\n"); log.flush()
        vocab = int(args[args.index("--score-vocab") + 1]) if "--score-vocab" in args else 258
        if any(t >= vocab for t in prompt + selected):
            result["error"] = "a token id is outside the vocabulary"
            score_done(request_id, "REJECT", result)
            continue
        admission_n = len(prompt) + (1 if "--score-wrong-admission" in args else 0)
        print(f"SADMIT {request_id} v1 prompt_tokens={admission_n} selected_tokens={len(selected)}", flush=True)
        if "--score-flood" in args:
            for _ in range(50):
                print(f"SADMIT {request_id} v1 prompt_tokens={len(prompt)} selected_tokens={len(selected)}", flush=True)
            continue
        for q in range(0, len(prompt), CH):
            if "--score-hold-file" in args:
                gate = args[args.index("--score-hold-file") + 1]
                while os.path.exists(gate):
                    with score_lock:
                        if request_id in score_cancel:
                            break
                    time.sleep(.005)
            time.sleep(STEP / 2)
            if active:
                window()
            with score_lock:
                cancelled = request_id in score_cancel
            if cancelled:
                if "--score-drain-gate" in args:
                    gate = args[args.index("--score-drain-gate") + 1]
                    while os.path.exists(gate):
                        time.sleep(.005)
                result["error"] = "cancelled"
                score_done(request_id, "CANCELLED", result)
                break
        else:
            # Independent FP32 row fixture: these are not sampling probabilities.
            row = {0: .125, 1: -17.5, 2: 2.25, 3: .3333333432674408}
            result.update(prompt_tokens=len(prompt), selected_token_ids=selected,
                          logits=[row.get(t, -3.0) for t in selected],
                          receipt={"prompt_tokens": len(prompt), "selected_tokens": len(selected),
                                   "logit_passes": 1, "dtype": "float32", "scale": 1})
            if "--score-missing-terminal" in args:
                continue
            if "--score-wrong-terminal" in args:
                print("DONE 0 0 0 0 stop", flush=True)
                continue
            if "--score-wrong-batch-terminal" in args:
                print(f"BDONE {request_id} 0 stop 0", flush=True)
                continue
            if "--score-wrong-id" in args:
                request_id += "x"
            if "--score-wrong-boundary" in args:
                result["prompt_tokens"] += 1
            if "--score-partial" in args:
                result["logits"] = result["logits"][:-1]
            if "--score-wrong-pass" in args:
                result["receipt"]["logit_passes"] = 2
            if "--score-generated" in args:
                result["generated_tokens"] = 1
            if "--score-reordered" in args:
                result["selected_token_ids"] = sorted(selected)
            if "--score-nonfinite" in args:
                result["logits"][0] = float("nan")
            if "--score-error" in args:
                result = {"version": 1, "request_id": request_id, "generated_tokens": 0,
                          "error": "device state corrupted"}
            score_done(request_id, "ERROR" if "--score-error" in args else "OK", result)
        continue
    if line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        slot = int(f[1]) if f[0] == "BGEN" else None
        max_new = int(f[2] if f[0] == "BGEN" else f[1])
        ids = f[-1].split(",")
        if b"ADAPTEREXIT" in bytes(int(x) & 255 for x in ids):
            if log:
                log.write(f"{f[0]} {slot} {len(ids)} {len(active)}\n"); log.flush()
            sys.exit(7)
        if bytes(int(x) & 255 for x in ids).find(b"REFUSEME") >= 0:   # #1059: refused up front, then idle (no DONE)
            print("ERR refused: images are not enabled", flush=True)
            continue
        if bytes(int(x) & 255 for x in ids).find(b"ADMISSIONREFUSE") >= 0:
            print("REJECT bounded=1 prompt + max_new exceeds the context", flush=True)
            if slot is not None:
                boundary = slot + 1 if "--bad-refusal-boundary" in args else slot
                print(f"BADM {boundary} 0", flush=True)
            continue
        # #1994: the engine's own view of its slots differs: BUSYSLOT is refused once, BUSYALL always
        pbx = bytes(int(x) & 255 for x in ids)
        if slot is not None and (b"BUSYALL" in pbx or (b"BUSYSLOT" in pbx and "busyslot" not in fired)):
            fired.add("busyslot")
            print(f"ERR BGEN: no such free slot (--batch {slots}) or a bad max_new", flush=True)
            if log:
                log.write(f"REFUSED {slot}\n"); log.flush()
            continue
        if log:
            log.write(f"{f[0]} {slot} {len(ids)} {max(len(active), 0)}\n"); log.flush()
        pb = bytes(int(x) & 255 for x in ids)
        mark = b"NFSOLO" if f[0] == "GEN" else b"NFADMIT"
        if mark in pb and mark not in fired:
            fired.add(mark)
            print(f"DONE 0 {len(ids)} 5.0 0.0 nonfinite 0 0 0", flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        if slot is not None and b"NFSLOT" in pb:
            nf_slots.add(slot)
        toks = reply(ids)
        stop.clear()
        # the prompt read: CH tokens a chunk, a PP line each; a BYIELD <s> gives way at a chunk boundary
        given = None
        saved = parked.pop(slot, None) if slot is not None else None
        start = saved[1] if saved and saved[0] == ids else 0
        read_reused = max(reused(ids), start)   # Resumed legs include this request's own parked work.
        for q in range(start + CH, len(ids) - 1, CH):
            time.sleep(STEP / 2)
            if (q // CH) % PPK:   # --pp-every K: a PP line and a boundary every K chunks (the pipelined groups' pieces)
                continue
            if "--quiet-prefill" not in args:
                print(f"PP {q} {len(ids)} 1 1", flush=True)
            if log and "--says-budget" in args:
                log.write(f"PREFILL {slot} {q}\n"); log.flush()
            held = []
            while True:
                try:
                    l2 = lines.get_nowait()
                except queue.Empty:
                    break
                if l2 and l2.startswith("BYIELD "):
                    ys = int(l2.split()[1])
                    if len(ids) - 1 - q > CH and (slot is None or ys == slot) and given is None:
                        given = (ys, q)
                else:
                    held.append(l2)
            for l2 in held:
                lines.put(l2)
            if given:
                break
            if active and (q // CH) % 4 == 0:     # the slots decode between the prompt's chunks (as the engine)
                window()
        if given:
            parked[given[0]] = (ids, given[1])
            if log:
                log.write(f"YIELD {given[0]} {given[1]}\n"); log.flush()
            print(f"YIELDED {given[0]} {given[1]}", flush=True)
            print(f"DONE 0 {len(ids)} 5.0 0.0 cancel 0 0 {read_reused}" +
                  (f" 0 0 0 0 0.0 {given[1] - read_reused}" if READF else ""), flush=True)
            if slot is not None:
                print(f"BADM {slot} 0", flush=True)
            continue
        n = 1 if slot is not None else max_new
        out = 0
        for t in toks[:n]:
            while slot is None and hold and os.path.exists(hold) and not stop.is_set():
                time.sleep(0.005)
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
        fin = "stop" if out and toks[out - 1] == 257 else ("cancel" if stop.is_set() else "length")
        print(f"DONE {out} {len(ids)} 5.0 {out * STEP * 1000:.1f} {fin} 0 0 {read_reused}" +
              (f" 0 0 0 0 0.0 {len(ids) - read_reused}" if READF else ""), flush=True)
        if slot is not None:
            cont = fin == "length" and max_new > 1
            previous_terminal = legacy_terminals.get(slot)
            if cont and legacy_identity:
                legacy_serial += 1
                legacy_owners[slot] = (legacy_serial, legacy_owners.get(slot, (0, 0))[1] + 1)
                legacy_seen.add(legacy_owners[slot])
            if cont:
                active[slot] = [toks[1:], max_new, 1]
            while hold_badm and os.path.exists(hold_badm):
                time.sleep(.005)
            identity = legacy_owners.get(slot) if cont and legacy_identity else None
            suffix = f" request={identity[0]} incarnation={identity[1]}" if identity else ""
            print(f"BADM {slot} {1 if cont else 0}{suffix}", flush=True)
            if cont and previous_terminal and "--legacy-stale-done" in args:
                print(previous_terminal, flush=True)
                print(previous_terminal, flush=True)
        continue
    if line:
        print("ERR unknown " + line[:20], flush=True)
        continue
    window()
if "--exit-code" in args:
    sys.exit(int(args[args.index("--exit-code") + 1]))
'''


class ParallelArgs(unittest.TestCase):
    def test_parallel_config_key(self):
        self.assertEqual(parallel_args({"parallel": 4}, []), ["--batch", "4"])
        self.assertEqual(parallel_args({"parallel": 1}, []), [])
        self.assertEqual(parallel_args({}, []), [])
        self.assertEqual(parallel_args({"parallel": 2}, ["--batch", "3"]), [])       # the args' own value wins
        self.assertEqual(parallel_args({"parallel": "4"}, []), [])                   # said and ignored
        self.assertEqual(parallel_args({"parallel": True}, []), [])
        self.assertEqual(parallel_args({"parallel": 12}, []), ["--batch", "12"])     # the engine warns and caps
    def test_parallel_above_the_window_says_so_even_with_batch_mtp(self):
        import contextlib, io, os
        def said(args, env):
            buf = io.StringIO()
            old = os.environ.get("STRATA_BATCH_MTP")
            if env is None:
                os.environ.pop("STRATA_BATCH_MTP", None)
            else:
                os.environ["STRATA_BATCH_MTP"] = env
            try:
                with contextlib.redirect_stdout(buf):
                    r = parallel_args({"parallel": 12}, args)
            finally:
                if old is None:
                    os.environ.pop("STRATA_BATCH_MTP", None)
                else:
                    os.environ["STRATA_BATCH_MTP"] = old
            self.assertEqual(r, ["--batch", "12"])
            return "at most" in buf.getvalue()
        self.assertTrue(said([], None))                  # 0.1.39: the window holds 8 rows
        self.assertTrue(said([], "0"))
        self.assertTrue(said(["--batch-mtp"], None))     # native admission also normalizes MTP slots to eight
        self.assertTrue(said([], "1"))

        args = engine_args({"args": ["--pack", "p"], "parallel": 2})
        self.assertEqual(args[-2:], ["--batch", "2"])

    def test_resident_multiplex_is_explicit_and_demands_slots(self):
        self.assertEqual(engine_args({"args": []}), [])
        self.assertEqual(engine_args({"args": [], "parallel": 2, "resident_multiplex": True}),
                         ["--batch", "2", "--resident-multiplex"])
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            engine_args({"args": [], "resident_multiplex": True})
        with self.assertRaises(ValueError):
            engine_args({"args": [], "resident_multiplex": "true"})


class AdaptiveSpeculation(unittest.TestCase):
    @staticmethod
    def waiter(context=100, due=2.0, priority="foreground"):
        return [context, object(), priority, due]

    def test_cold_policy_never_withholds_waiting_capacity(self):
        policy = SpeculationPolicy(2.0, 4)
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter()], now=0))
        self.assertTrue(policy.prefer_solo(100, 64, now=0))
        self.assertFalse(policy.yield_solo(100, 64, [], now=0))

    def test_wire_counts_commits_not_proposals_or_terminal_totals(self):
        policy = SpeculationPolicy(2.0, 4)
        policy.begin("solo", 100, transition=True, now=0)
        policy.wire("REUSED 99", now=1)
        policy.wire("DRAFT 999 rejected 999", now=1.05)
        for i in range(8):
            policy.wire(f"T {10 + i}", now=1.1 + i / 10)
        key = policy._key("solo", 1, 100)
        before = policy.costs[key][0] / policy.costs[key][1]
        policy.wire("DONE 8 100 1 1 length 999 999 99", now=2)
        after = policy.costs[key][0] / policy.costs[key][1]
        self.assertEqual(policy.costs[key][2], 8)
        self.assertLess(after, before, "zero-output tail drafting/drain still costs wall time")
        self.assertAlmostEqual(policy.handoffs[key], 1.1)
        policy.begin("batch", 100, slot=0, now=3)
        policy.wire("T 20", now=3.2)  # BGEN's first committed target token
        policy.wire("DONE 1 100 1 1 length 0 0 99", now=3.3)
        policy.wire("BADM 0 1", now=3.4)
        policy.begin("batch", 200, slot=1, now=3.5)
        policy.wire("T 21", now=3.6)
        policy.wire("DONE 1 200 1 1 length 0 0 0", now=3.7)
        policy.wire("BADM 1 1", now=3.8)
        for i in range(8):
            policy.wire(f"BT {i % 2} {30 + i}", now=4 + i / 10)
        policy.wire("BDONE 0 5 cancel 1700", now=5)
        policy.wire("BDONE 1 5 length 1900", now=5.2)
        self.assertEqual(sum(value[2] for k, value in policy.costs.items() if k[0] == "batch"), 10)
        self.assertEqual(policy.slots, {})
        self.assertIsNone(policy.epoch)

    def test_context_load_and_fixed_slack_change_actual_choice(self):
        policy = SpeculationPolicy(2.0, 4)
        policy.observe("solo", 1, 100, 80, 1)
        policy.observe("batch", 2, 100, 20, 1)
        waiters = [self.waiter()]
        self.assertFalse(policy.yield_solo(100, 64, waiters, now=0))
        self.assertFalse(policy.yield_solo(100, 64, waiters, now=.49))
        self.assertTrue(policy.yield_solo(100, 64, waiters, now=.5),
                        "a profitable solo burst cannot keep extending its own deadline")
        policy.yield_solo(100, 64, [], now=.6)
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter(due=.7)], now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter(due=0, priority="background")], now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [], foreground_pending=True, now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter(priority="background")],
                                         foreground_pending=True, now=.6))
        self.assertTrue(policy.yield_solo(100, 64, [self.waiter() for _ in range(3)], now=.6),
                        "unmeasured higher load retains immediate ordinary-batch fallback")
        policy.observe("solo", 1, 4096, 10, 1)
        policy.observe("batch", 2, 4096, 50, 1)
        self.assertTrue(policy.yield_solo(4096, 64, [self.waiter(4096)], now=0))

    def test_gain_must_repay_catchup_and_clear_hysteresis(self):
        policy = SpeculationPolicy(2.0, 2)
        policy.observe("solo", 1, 100, 80, 1)
        policy.observe("batch", 1, 100, 20, 1)
        key = policy._key("solo", 1, 100)
        policy.probed[key] = 0
        policy.handoffs[key] = 3.0
        self.assertFalse(policy.prefer_solo(100, 64, now=1))
        self.assertTrue(policy.prefer_solo(100, 120, now=1))
        policy.observe("solo", 1, 200, 21, 1)
        policy.observe("batch", 1, 200, 20, 1)
        self.assertFalse(policy.prefer_solo(200, 1000, now=1), "a 5% noisy gain is not a switch")
        policy.observe("solo", 1, 300, 10, 1)
        policy.observe("batch", 1, 300, 40, 1)
        key = policy._key("solo", 1, 300)
        policy.probed[key] = 0
        self.assertFalse(policy.prefer_solo(300, 64, now=15.99))
        self.assertTrue(policy.prefer_solo(300, 64, now=16), "stale acceptance is remeasured, not frozen")
        policy.begin("solo", 300, now=16)
        self.assertFalse(policy.prefer_solo(300, 64, now=16.01))

    def test_new_process_has_no_predecessor_cost_or_deadline_state(self):
        before = SpeculationPolicy(2.0, 2)
        before.observe("solo", 1, 100, 100, 1)
        before.observe("batch", 2, 100, 20, 1)
        self.assertFalse(before.yield_solo(100, 64, [self.waiter()], now=0))
        after = SpeculationPolicy(2.0, 2)
        self.assertTrue(after.yield_solo(100, 64, [self.waiter()], now=.1))
        self.assertIsNone(after.defer_until)


class ControlBudget(unittest.TestCase):
    def test_control_wait_partition_obeys_gate_precedence_without_holding_condition(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        stage, clock = [0], [100.0]
        class ScriptedCondition(threading.Condition):
            def wait(self, timeout=None):
                clock[0] += 1
                stage[0] += 1
                if stage[0] == 3:
                    engine.ctl.release()
        engine.slot_cv = ScriptedCondition()
        engine.ctl.acquire()
        engine.foreground_budget_s, engine.background_share = 1, .25
        engine.alive = lambda: True
        svc = Service(engine, ByteTokenizer(), None)
        def observe(phase):
            if phase == "control_selector_wait":
                inspected = threading.Event()
                def inspect_waiters():
                    with engine.slot_cv:
                        inspected.set()
                probe = threading.Thread(target=inspect_waiters, daemon=True)
                probe.start()
                self.assertTrue(inspected.wait(1), "phase accounting must not hold the engine condition")
                probe.join(1)
            svc.frontend_phase(record, phase)
        try:
            with mock.patch("serve.server.time.perf_counter", side_effect=lambda: clock[0]):
                record = svc.register_lifecycle({}, threading.Event())
                svc.acquire_request(record)
                turn = engine._take_control(threading.Event(), 10, priority="background",
                                            background_due=time.monotonic() + 100,
                                            foreground_pending=lambda: stage[0] == 0,
                                            slot_ready=lambda: stage[0] >= 2, on_phase=observe)
                with self.assertRaises(StopIteration) as finished:
                    next(turn)
                self.assertTrue(finished.exception.value)
                svc.finish_lifecycle(record)
            seconds = record["frontend"]["seconds"]
            self.assertEqual(seconds["control_selector_wait"], 1)
            self.assertEqual(seconds["physical_slot_wait"], 1)
            self.assertEqual(seconds["control_occupied_wait"], 1)
            self.assertEqual(sum(seconds.values()), record["completion_s"])
            self.assertEqual(engine.waiting, 0)
            self.assertEqual(list(engine.control_waiters), [])
        finally:
            engine.ctl.release()

    def test_arrival_trace_bounds_background_wait_without_reordering_foreground(self):
        import collections
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        engine.background_share = .25
        background = [3000, object(), "background", 1.0]
        foreground = [[20, object(), "foreground", float("inf")] for _ in range(100)]
        engine.control_waiters = collections.deque([background, *foreground])
        self.assertIs(engine._control_turn(0), foreground[0])
        engine.control_waiters.remove(foreground[0])
        self.assertIs(engine._control_turn(.99), foreground[1])
        self.assertIs(engine._control_turn(1.0), background,
                      "new foreground arrivals cannot overtake a due background continuation")
        with mock.patch("serve.server.time.monotonic", return_value=2.0):
            self.assertAlmostEqual(engine._background_next(1.8), 2.6)
            self.assertEqual(engine._background_next(0), 8.0,
                             "background requeue must repay measured service, not force a chunk every budget")
        self.assertTrue(engine._can_yield(100), "the legacy two-yield cap bypasses the configured policy")
        engine.foreground_budget_s = 0
        self.assertIs(engine._control_turn(1.0), background)
        self.assertFalse(engine._can_yield(2), "rollback preserves the legacy policy")

    def test_short_first_orders_new_reads_without_bypassing_background_deadlines(self):
        import collections
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        background = [10, object(), "background", 40.0, 0.0, True]
        long = [3000, object(), "foreground", float("inf"), 0.0, True]
        short = [20, object(), "foreground", float("inf"), 2.0, True]
        engine.wait_lens = [background, long, short]
        engine.control_waiters = collections.deque(engine.wait_lens)
        with mock.patch("serve.server.ADMIT_SHORT_FIRST", 100), \
                mock.patch("serve.server.ADMIT_MAX_WAIT_S", 30.0), \
                mock.patch("serve.server.time.monotonic", return_value=10.0):
            self.assertIs(engine._control_turn(10.0), short, "short-first stays within the foreground tier")
            with mock.patch("serve.server.time.monotonic", return_value=31.0):
                self.assertIs(engine._control_turn(31.0), long, "an aged long prompt cannot starve")
            self.assertIs(engine._control_turn(40.0), background, "a due background read still goes first")
            engine.control_waiters.remove(background)
            engine.wait_lens.remove(background)
            long[5] = False
            self.assertIs(engine._control_turn(10.0), long, "a parked continuation keeps its FIFO turn")
        with mock.patch("serve.server.ADMIT_SHORT_FIRST", 0):
            self.assertIs(engine._control_turn(10.0), long, "disabled short-first retains FIFO")

    def test_foreground_waiting_for_a_physical_slot_stays_in_the_control_queue(self):
        import collections
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        engine.background_share = .25
        engine.slot_cv = threading.Condition()
        engine.ctl = threading.Lock()
        engine.ctl_epoch = 0
        engine.waiting = 0
        engine.wait_lens = []
        engine.control_waiters = collections.deque()
        engine.alive = lambda: True
        ready, acquired, cancel = threading.Event(), threading.Event(), threading.Event()
        results = []
        def request():
            def wait():
                results.append((yield from engine._take_control(cancel, 20, lambda _: acquired.set(),
                                                               slot_ready=ready.is_set)))
            list(wait())
        thread = threading.Thread(target=request)
        thread.start()
        with engine.slot_cv:
            self.assertTrue(engine.slot_cv.wait_for(lambda: engine.waiting == 1, timeout=2))
            foreground = engine.control_waiters[0]
            background = [3000, object(), "background", time.monotonic() + 10]
            engine.control_waiters.append(background)
            self.assertIs(engine._control_turn(time.monotonic()), foreground,
                          "a foreground slot waiter must not disappear and make background appear alone")
            self.assertFalse(engine.ctl.locked())
            background[3] = 0
            self.assertIs(engine._control_turn(time.monotonic()), background,
                          "keeping a slot waiter visible must not starve due background work")
            engine.control_waiters.remove(background)
            ready.set()
            engine.slot_cv.notify_all()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [True])
        self.assertTrue(acquired.is_set())
        self.assertEqual(engine.waiting, 0)
        self.assertEqual(list(engine.control_waiters), [])
        engine.ctl.release()

    def test_releasing_control_wakes_a_blocked_owner_without_a_poll_timeout(self):
        import collections
        waiting, wakeups = threading.Event(), []
        class WaitingCondition(threading.Condition):
            def wait(self, timeout=None):
                waiting.set()
                notified = super().wait(timeout=2)
                wakeups.append(notified)
                return notified
        engine = object.__new__(StrataEngine)
        engine.slot_cv = WaitingCondition()
        engine.ctl = threading.Lock()
        engine.ctl.acquire()
        engine.ctl_epoch, engine.waiting = 0, 0
        engine.wait_lens, engine.control_waiters = [], collections.deque()
        engine.alive = lambda: True
        results = []
        def request():
            def turn():
                results.append((yield from engine._take_control(threading.Event(), 77)))
            list(turn())
        thread = threading.Thread(target=request)
        thread.start()
        try:
            self.assertTrue(waiting.wait(2))
            engine._release_control()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(results, [True])
            self.assertEqual(wakeups, [True], "ownership must wake on release, not the polling timeout")
            self.assertEqual(list(engine.control_waiters), [])
        finally:
            with engine.slot_cv:
                engine.slot_cv.notify_all()
            thread.join(3)
            if engine.ctl.locked():
                engine.ctl.release()

    def test_http_foreground_waiter_is_visible_but_cannot_starve_due_background(self):
        import collections
        service = object.__new__(Service)
        foreground = {"priority": "foreground", "state": "queued"}
        service.admission_waiters = [foreground]
        service.admission_active = {}
        service._refresh_foreground_pending()
        self.assertTrue(service._foreground_pending)
        service.admission_waiters.remove(foreground)
        service.admission_active["foreground"] = foreground
        foreground["state"] = "admitted"
        service._refresh_foreground_pending()
        self.assertTrue(service._foreground_pending, "admitted but unsubmitted requests are still waiting")
        foreground["_executing"] = True
        service._refresh_foreground_pending()
        self.assertFalse(service._foreground_pending, "an executing decoder is not an admission waiter")
        foreground["_executing"] = False
        service._refresh_foreground_pending()
        engine = object.__new__(StrataEngine)
        engine.foreground_budget_s = 1.0
        engine.background_share = .25
        engine.slot_cv = threading.Condition()
        engine.ctl = threading.Lock()
        engine.ctl_epoch = 0
        engine.waiting = 0
        engine.wait_lens = []
        engine.control_waiters = collections.deque()
        engine.alive = lambda: True
        acquired, cancel = threading.Event(), threading.Event()
        results = []
        due = time.monotonic() + 1.0
        def request():
            def wait():
                results.append((yield from engine._take_control(
                    cancel, 3000, lambda _: acquired.set(), priority="background", background_due=due,
                    foreground_pending=lambda: service._foreground_pending)))
            list(wait())
        thread = threading.Thread(target=request)
        thread.start()
        with engine.slot_cv:
            self.assertTrue(engine.slot_cv.wait_for(lambda: engine.waiting == 1, timeout=.5))
            self.assertFalse(engine.ctl.locked(), "HTTP foreground demand forbids early background reacquisition")
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(acquired.is_set(), "a due background request advances even under persistent demand")
        self.assertEqual(results, [True])
        engine.ctl.release()



class PickSlot(unittest.TestCase):
    def engine(self, n):
        e = StrataEngine("missing-executable", [], lazy=True)
        e.batch = n
        e.slot_order = list(range(n))
        e.slot_busy = [False] * n
        e.slot_held = [[] for _ in range(n)]
        e.slot_scope = ["a"] * n
        e.slot_groups = 1
        e.slot_used = [0.0] * n
        e.slot_live = [None] * n
        e.slot_claims = set()
        e.slot_releasing = set()
        e.slot_group = [0] * n
        return e

    def test_the_slot_that_holds_the_conversation(self):
        e = self.engine(3)
        e.slot_held = [[1, 2, 3], [], [1, 2, 3, 4, 5]]
        e.slot_used = [5.0, 0.0, 9.0]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5, 6], 'a'), 2)       # the longest held start
        self.assertEqual(e.pick_slot([1, 2, 3, 9], 'a'), 0)
        self.assertEqual(e.pick_slot([7, 8], 'a'), 1)                    # no match: the empty slot first
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([7, 8], 'a'), 0)                    # then the one used longest ago
        e.slot_busy = [True] * 3
        self.assertIsNone(e.pick_slot([7, 8], 'a'))
        e.slot_busy = [False] * 3
        self.assertEqual(e.pick_slot([1, 2, 3], 'a'), 1)                 # a held prompt is never the WHOLE prompt

    def test_a_rewritten_next_turn_goes_to_the_slot_of_its_chat(self):
        # #2518: two chats in two slots; each next turn renders the history again but not the reply it decoded or the
        # assistant header, so the held stream (prompt + reply) is no start of it.  The engine's turn checkpoint sits at
        # the last <|im_start|> of the old prompt: the prompt that agrees up to there belongs to that slot.  Before, both
        # went to the least recently used slot in turn, and the first one overwrote the other chat's cache.
        T = StrataEngine.TURN_ID
        hist_a, hist_b = list(range(1000, 1600)), list(range(2000, 2600))
        held_a = hist_a + [T, 7, 8, 9] + [50, 51, 52]                  # prompt (ends with the assistant header) + reply
        held_b = hist_b + [T, 7, 8, 9] + [60, 61]
        e = self.engine(2)
        e.slot_held = [held_a, held_b]
        e.slot_used = [10.0, 20.0]                                      # slot 0 was used longer ago
        next_a = hist_a + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))  # the history again, then new turns
        next_b = hist_b + [T, 5, 6, 7, 8, 9] + list(range(3000, 3100))
        self.assertEqual(e.pick_slot(next_b, "a"), 1)                   # the LRU slot 0 holds the other chat
        self.assertEqual(e.pick_slot(next_a, "a"), 0)
        e.slot_busy[0] = True                                           # a busy slot is never picked
        self.assertEqual(e.pick_slot(next_a, "a"), 1)
        e.slot_busy[0] = False
        # a chat that agrees with neither (a new one, or one that rewrote its history) keeps the old order
        self.assertEqual(e.pick_slot(list(range(5000, 5200)), "a"), 0)
        self.assertEqual(e.pick_slot(hist_a[:300] + [T, 5, 6] + list(range(3000, 3100)), "a"), 0)
        e.slot_held[0] = []                                             # an empty slot still comes before evicting one
        self.assertEqual(e.pick_slot(list(range(5000, 5200)), "a"), 0)
        # the whole held stream continued is still the longer match
        e.slot_held = [held_a, held_b + [T, 1, 2]]
        self.assertEqual(e.pick_slot(held_b + [T, 1, 2, 3, 4], "a"), 1)
        self.assertEqual(e.pick_slot(next_b, "other"), 0, "a turn checkpoint never crosses an authorized scope")

    def test_batch_groups_spread_requests_over_the_groups(self):
        # #1249: with --batch-groups 2 over 4 slots (groups {0,1} and {2,3}) a second request goes to the other group,
        # and a short held prefix does not pull it back into the busy one
        e = self.engine(4)
        e.slot_groups = 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], 'a'), 2)
        e.slot_held[1] = list(range(1, 601))                       # a long prefix is worth the unbalanced group
        self.assertEqual(e.pick_slot(list(range(1, 700)), 'a'), 1)
        e.slot_groups, e.slot_group = 1, [0] * 4                   # without groups: any held prefix, as before
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.pick_slot([1, 2, 3, 4, 5], 'a'), 1)

    def test_pipelined_prefix_affinity_is_scoped_and_precedes_hole_filling(self):
        e = self.engine(4)
        e.slot_groups, e.slot_gs = 2, 2
        e.slot_group = [0, 0, 1, 1]
        e.slot_busy[0] = True
        e.slot_held[2] = list(range(600))
        e.slot_scope[2] = "a"
        prompt = list(range(601))
        self.assertEqual(e.pick_slot(prompt, "a"), 2)
        self.assertEqual(e.pick_slot(prompt, "other"), 1)
        self.assertEqual(e.pick_slot(prompt, ""), 1)

    def test_slot_states_never_describe_a_reserved_or_releasing_slot_as_idle(self):
        e = self.engine(6)
        e.slot_busy = [False, True, True, True, True, True]
        e.slot_claims.update((1, 2, 3, 4, 5))
        e.slot_releasing.add(5)
        for b, state in ((2, "parked"), (3, "reading"), (4, "decoding"), (5, "decoding")):
            e.slot_live[b] = {"state": state, "prompt_tokens": 9, "generated": 1,
                              "started": time.time(), "first_token": None}
        self.assertEqual([r["state"] for r in e.slots_view()],
                         ["idle", "reserved", "parked", "reading", "decoding", "releasing"])

    def test_slots_view(self):
        e = self.engine(2)
        e.slot_held[1] = [1, 2]
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual(v[0]["state"], "decoding")
        self.assertEqual(v[0]["generated"], 4)
        self.assertEqual(v[1], {"slot": 1, "state": "idle", "held_tokens": 2})

    def test_slots_view_keeps_reported_residency_while_decoding(self):
        e = self.engine(2)
        e.info.update(slot_0_kv_mode="streamed", slot_0_kv_window=65536,
                      slot_0_draft_kv_mode="resident", slot_0_draft_kv_window=262144,
                      slot_1_kv_mode="resident", slot_1_kv_window=262144,
                      slot_1_draft_kv_mode="off", slot_1_draft_kv_window=0)
        e.slot_live[0] = {"state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        view = e.slots_view()
        self.assertEqual((view[0]["kv_mode"], view[0]["kv_window"],
                          view[0]["draft_kv_mode"], view[0]["draft_kv_window"]),
                         ("streamed", 65536, "resident", 262144))
        self.assertEqual(view[1], {"slot": 1, "state": "idle", "held_tokens": 0,
                                  "kv_mode": "resident", "kv_window": 262144,
                                  "draft_kv_mode": "off", "draft_kv_window": 0})


class ParallelService(unittest.TestCase):
    """The real StrataEngine and Service over HTTP, the fake engine behind them."""

    def start(self, slots, fit=None, slot_cache=False, fail=False, reuse=False, says_groups=0, more=(), lazy=False):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        extra = ["--batch", str(slots)] + (["--fit", str(fit)] if fit is not None else []) + ["--log", str(self.log)]
        extra += ["--slotcache"] if slot_cache else []
        extra += ["--fail-window"] if fail else []
        extra += ["--reuse"] if reuse else []
        extra += ["--says-groups", str(says_groups)] if says_groups else []
        extra += list(more)
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            self.engine = StrataEngine("strata", extra, lazy=lazy)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
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

    def chat(self, text, max_tokens=64, priority="foreground", packing=None):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "strata_priority": priority}
        if packing is not None:
            body["strata_packing"] = packing
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def message(self, text, max_tokens=64):
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens}
        req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def _two_at_once(self, text_b):
        """A long request first, then one that has to go to a batch slot (the engine is busy with the first)."""
        out = {}
        def go(k, text):
            try:
                out[k] = ("ok", self.chat(text, max_tokens=64))
            except urllib.error.HTTPError as e:
                out[k] = ("http", e.code, e.read().decode(), dict(e.headers))
        ta = threading.Thread(target=go, args=("a", "LONGREPLY please"))
        ta.start()
        time.sleep(0.15)
        tb = threading.Thread(target=go, args=("b", text_b))
        tb.start()
        ta.join(30); tb.join(30)
        return out

    def attributed_chat(self, *, max_tokens=4, stream=False, stop=None, text="hello", priority="foreground"):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "stream": stream, "strata_priority": priority}
        if stop is not None:
            body["stop"] = stop
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            identity = response.headers["X-Strata-Request-Id"]
            raw = response.read().decode()
        result = ([json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")][-1]
                  if stream else json.loads(raw))
        return identity, result

    def test_bdone_draft_counters_reach_the_http_response(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "79,107"))
        identity, result = self.attributed_chat()
        self.assertTrue(identity)
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (79, 107))
        self.assertEqual(result["usage"]["completion_tokens"], 4)

    def recorded_chat(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as log:
            identity, result = self.attributed_chat(**kwargs)
            deadline = time.monotonic() + 2
            prefix = "[strata] response_attribution: "
            matching = []
            while time.monotonic() < deadline:
                records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                           if line.startswith(prefix)]
                matching = [record for record in records if record["request_id"] == identity]
                if matching:
                    break
                time.sleep(0.005)
        self.assertEqual(len(matching), 1, log.getvalue())
        return result, matching[0]

    def _yielded_accounting(self, *, reuse=False, yields=1):
        self.start(2, reuse=reuse, more=("--resident-multiplex", "--says-multiplex", "--says-budget",
                                        "--readfield", "--draft-counters", "0,0"))
        replies, failures = [], []
        def long_request():
            try:
                replies.append(self.recorded_chat(text="long " * 600, priority="background"))
            except Exception as error:
                failures.append(error)
        worker = threading.Thread(target=long_request)
        worker.start()
        def wait_read(admissions):
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                lines = self.log.read_text().splitlines() if self.log.exists() else []
                long_reads = [i for i, line in enumerate(lines) if line.startswith("BGEN ") and int(line.split()[2]) > 500]
                if len(long_reads) >= admissions and any(line.startswith("PREFILL ") for line in lines[long_reads[-1] + 1:]):
                    return
                time.sleep(.005)
            self.fail("long admission did not reach its prefill barrier")
        try:
            for wave in range(yields):
                wait_read(wave + 1)
                self.attributed_chat(text=f"short {wave}")
        finally:
            worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        result, record = replies[0]
        self.assertEqual(result["usage"]["completion_tokens"], 4)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], 3 if reuse else 0)
        self.assertEqual(result["timings"]["prompt_ms"], 5 * (yields + 1))
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (4, 4))
        self.assertEqual(len(record["legs"]), yields + 1)
        self.assertEqual([leg["generated"] for leg in record["legs"]], [0] * yields + [4])
        self.assertEqual(record["legs"][-1]["admission"]["generated"], 1)
        self.assertEqual(sum(leg["prompt_read"] for leg in record["legs"]) + record["legs"][0]["reused"],
                         result["usage"]["prompt_tokens"])
        self.assertEqual(sum(leg["prompt_ms"] for leg in record["legs"]), result["timings"]["prompt_ms"])

    def test_cold_yielded_prefill_is_read_work_not_an_external_cache_hit(self):
        self._yielded_accounting()

    def test_warm_yield_preserves_original_cache_and_sums_remaining_reads(self):
        self._yielded_accounting(reuse=True)

    def test_two_yields_keep_disjoint_prompt_work_and_one_final_batch_commit(self):
        self._yielded_accounting(yields=2)

    def test_bdone_response_record_binds_identity_interval_and_slot_work(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "79,107"))
        result, record = self.recorded_chat()
        self.assertEqual(record["path"], "/v1/chat/completions")
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (4, 4))
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (79, 107))
        interval = record["interval"]
        self.assertLessEqual(interval["started_at"], interval["finished_at"])
        self.assertGreater(interval["duration_s"], 0)
        self.assertEqual(len(record["legs"]), 1)
        leg = record["legs"][0]
        self.assertEqual((leg["kind"], leg["slot"], leg["generated"], leg["decode_ms"]), ("batch", 0, 4, 1.0))
        self.assertEqual((leg["drafts_accepted"], leg["drafts_offered"]), (79, 107))
        self.assertEqual(leg["admission"]["generated"], 1)
        self.assertEqual(result["timings"]["draft_n"], 107)

    def test_close_after_admission_done_drains_badm_and_one_batch_terminal(self):
        import http.client
        import socket
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold-badm"
            gate.touch()
            self.start(2, more=("--resident-multiplex", "--says-multiplex", "--says-budget",
                                "--hold-badm", str(gate), "--draft-counters", "1,2"))
            parsed = threading.Event()
            parse_done = self.engine._parse_done
            def note_done(*args, **kwargs):
                parse_done(*args, **kwargs)
                parsed.set()
            connection = http.client.HTTPConnection(*self.httpd.server_address, timeout=5)
            response, transport = None, None
            with contextlib.redirect_stdout(io.StringIO()) as log, \
                    mock.patch.object(self.engine, "_parse_done", side_effect=note_done):
                try:
                    connection.request("POST", "/v1/chat/completions", json.dumps({
                        "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4,
                        "reasoning_effort": "none", "stream": True}), {"Content-Type": "application/json"})
                    transport = connection.sock.dup()
                    response = connection.getresponse()
                    identity = response.getheader("X-Strata-Request-Id")
                    self.assertTrue(parsed.wait(3), "admission DONE was not ingested before BADM gate")
                    transport.shutdown(socket.SHUT_RDWR)
                    response.close()
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline and not any(
                            row["id"] == identity and row["state"] == "draining" for row in self.svc.lifecycle_records()):
                        time.sleep(.005)
                    self.assertTrue(any(row["id"] == identity and row["state"] == "draining"
                                        for row in self.svc.lifecycle_records()))
                    self.assertEqual(len(self.engine.slot_claims), 1)
                    claimed = next(iter(self.engine.slot_claims))
                    self.assertTrue(self.engine.slot_busy[claimed], "claim released before BADM/BDONE")
                    gate.unlink()
                    prefix, records = "[strata] response_attribution: ", []
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                                   if line.startswith(prefix)]
                        records = [record for record in records if record["request_id"] == identity]
                        if records: break
                        time.sleep(.005)
                    self.assertEqual(len(records), 1, log.getvalue())
                    record = records[0]
                    self.assertEqual(len(record["legs"]), 1)
                    terminal = record["legs"][0]
                    self.assertEqual((terminal["kind"], terminal["admission"]["generated"]), ("batch", 1))
                    self.assertEqual(record["committed_tokens"], terminal["generated"])
                    self.assertEqual(record["drafts_offered"], 2)
                    self.assertFalse(self.engine.slot_claims)
                    self.assertFalse(any(self.engine.slot_busy))
                finally:
                    gate.unlink(missing_ok=True)
                    if transport: transport.close()
                    if response: response.close()
                    connection.close()

    def test_bdone_counters_survive_early_stop_and_synchronous_drain(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "1,2"))
        result, record = self.recorded_chat(max_tokens=64, stop="ok")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (1, 2))
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (3, 2))
        self.assertEqual(record["legs"][0]["kind"], "batch")
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (1, 2))

    def test_bdone_counters_survive_streaming_eos_drain(self):
        import serve.server as server
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "7,9"))
        with mock.patch.object(server, "EOS_IDS", {257}):
            result, record = self.recorded_chat(max_tokens=64, stream=True)
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (7, 9))
        self.assertEqual((record["committed_tokens"], record["output_tokens"]), (10, 10))
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (7, 9))
        self.assertEqual(record["legs"][0]["generated"], 10)

    def test_bdone_without_drafts_reports_explicit_zeros(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "0,0"))
        result, record = self.recorded_chat()
        self.assertEqual((result["timings"]["draft_n_accepted"], result["timings"]["draft_n"]), (0, 0))
        self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), (0, 0))
        self.assertEqual((record["legs"][0]["drafts_accepted"], record["legs"][0]["drafts_offered"]), (0, 0))

    def test_bdone_without_counters_does_not_borrow_the_admission_zeros(self):
        self.start(2, more=("--resident-multiplex", "--says-multiplex"))
        result, record = self.recorded_chat()
        self.assertNotIn("draft_n", result["timings"])
        self.assertNotIn("draft_n_accepted", result["timings"])
        self.assertIsNone(record["drafts_accepted"])
        self.assertIsNone(record["drafts_offered"])
        self.assertIsNone(record["legs"][0]["drafts_offered"])
        self.assertEqual(record["legs"][0]["admission"]["drafts_offered"], 0)

    def test_equal_count_responses_keep_their_slot_counters_by_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            gate.touch()
            self.start(2, more=("--resident-multiplex", "--says-multiplex", "--draft-counters", "1,2:2,3",
                                "--hold-replies", str(gate)))
            responses, errors = [], []

            def go():
                try:
                    responses.append(self.attributed_chat())
                except Exception as error:
                    errors.append(error)

            prefix = "[strata] response_attribution: "
            with contextlib.redirect_stdout(io.StringIO()) as log:
                threads = [threading.Thread(target=go) for _ in range(2)]
                for thread in threads:
                    thread.start()
                try:
                    deadline = time.monotonic() + 3
                    ready = False
                    while not ready and time.monotonic() < deadline:
                        with self.engine.slot_cv:
                            ready = sum(bool(slot) and slot["state"] == "decoding"
                                        for slot in self.engine.slot_live) == 2
                        if not ready:
                            time.sleep(0.005)
                    self.assertTrue(ready, "both requests must enter different native slots")
                finally:
                    gate.unlink(missing_ok=True)
                    for thread in threads:
                        thread.join(10)
                deadline = time.monotonic() + 2
                while log.getvalue().count(prefix) < 2 and time.monotonic() < deadline:
                    time.sleep(0.005)
            self.assertEqual(errors, [])
            self.assertEqual(len(responses), 2)
            records = [json.loads(line[len(prefix):]) for line in log.getvalue().splitlines()
                       if line.startswith(prefix)]
            self.assertEqual(len(records), 2, log.getvalue())
            by_id = {record["request_id"]: record for record in records}
            self.assertEqual(len(by_id), 2)
            self.assertEqual({record["legs"][0]["slot"] for record in records}, {0, 1})
            for identity, response in responses:
                record = by_id[identity]
                slot = record["legs"][0]["slot"]
                expected = (1, 2) if slot == 0 else (2, 3)
                self.assertEqual((response["timings"]["draft_n_accepted"], response["timings"]["draft_n"]), expected)
                self.assertEqual((record["drafts_accepted"], record["drafts_offered"]), expected)
                self.assertEqual((record["committed_tokens"], record["output_tokens"]), (4, 4))

    def test_bgen_refused_by_the_engine_is_retried_not_a_400(self):
        """#1994: the engine refuses a BGEN ("no such free slot") although the server saw the slot free: the request
        goes to another slot (or waits) and is answered; the same conversation's next turn works too."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.2
        out = self._two_at_once("BUSYSLOT hello")
        self.assertEqual(out["b"][0], "ok", out["b"])
        self.assertIn("REFUSED", self.log.read_text())
        again = self.chat("BUSYSLOT hello")                     # the conversation is not poisoned
        self.assertEqual(again["choices"][0]["message"]["content"], "ok, done.")
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])

    def test_bgen_refused_for_good_is_a_retryable_503_and_frees_the_slots(self):
        """#1994: when the engine keeps refusing, the answer is a 503 (retryable), never a 400, and no slot stays busy."""
        self.start(2)
        self.engine.SLOT_PARK_S = 0.05
        out = self._two_at_once("BUSYALL hello")
        self.assertEqual(out["b"][0], "http", out["b"])
        self.assertEqual(out["b"][1], 503, out["b"])
        time.sleep(0.5)
        self.assertEqual(self.engine.slot_busy, [False, False])
        self.assertEqual(self.chat("fine again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_reports_fewer_slots(self):
        self.start(4, fit=2)
        self.assertEqual(self.engine.batch, 2)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 2)

    def test_requested_nine_slots_report_effective_eight(self):
        """Frontend uses native INFO after normalization; the CPU planner test pins native 9 -> 8."""
        self.start(9, fit=8, says_groups=2, more=("--batch-groups", "2"))
        self.assertEqual(self.engine.batch, 8)
        self.assertEqual(self.engine.slot_groups, 2)
        self.assertEqual(len(self.engine.slot_busy), 8)
        self.assertEqual(self.get("/v1/status")["concurrency"]["serving"], 8)

    def test_native_info_reports_each_effective_slot_and_its_draft(self):
        """Scripted INFO -> HTTP reporting only, not proof of native K/V allocation or execution."""
        info = ("working_kv_mode=resident working_kv_window=262144 "
                "working_draft_kv_mode=resident working_draft_kv_window=262144\n"
                "slot_0_kv_mode=streamed slot_0_kv_window=65536 "
                "slot_0_draft_kv_mode=streamed slot_0_draft_kv_window=65536\n"
                "slot_1_kv_mode=streamed slot_1_kv_window=65536 "
                "slot_1_draft_kv_mode=resident slot_1_draft_kv_window=262144")
        self.start(4, fit=2, more=("--slot-kv-window", "65533", "--says-context", "262144",
                                   "--says-residency", info))
        expected = [{"slot": 0, "state": "idle", "held_tokens": 0,
                     "kv_mode": "streamed", "kv_window": 65536,
                     "draft_kv_mode": "streamed", "draft_kv_window": 65536},
                    {"slot": 1, "state": "idle", "held_tokens": 0,
                     "kv_mode": "streamed", "kv_window": 65536,
                     "draft_kv_mode": "resident", "draft_kv_window": 262144}]
        metrics = self.get("/metrics")
        self.assertEqual(metrics["live"]["slots"], expected)
        self.assertEqual(metrics["engine"]["working_kv_mode"], "resident")
        self.assertEqual(metrics["engine"]["working_draft_kv_mode"], "resident")
        self.assertEqual(metrics["engine"]["slot_0_kv_window"], 65536)
        self.assertEqual(self.get("/v1/status")["slots"], expected)
        self.assertEqual(self.get("/slots"),
                         [{"id": 0, "n_ctx": 262144, "is_processing": False, "n_prompt_tokens": 0,
                           "kv_mode": "streamed", "kv_window": 65536,
                           "draft_kv_mode": "streamed", "draft_kv_window": 65536},
                          {"id": 1, "n_ctx": 262144, "is_processing": False, "n_prompt_tokens": 0,
                           "kv_mode": "streamed", "kv_window": 65536,
                           "draft_kv_mode": "resident", "draft_kv_window": 262144}])

    def test_reported_resident_context_and_absent_draft_override_requested_window(self):
        info = ("slot_0_kv_mode=resident slot_0_kv_window=4096 "
                "slot_0_draft_kv_mode=off slot_0_draft_kv_window=0\n"
                "slot_1_kv_mode=resident slot_1_kv_window=4096 "
                "slot_1_draft_kv_mode=off slot_1_draft_kv_window=0")
        self.start(2, more=("--slot-kv-window", "65536", "--says-residency", info))
        for path in ("/slots", "/metrics", "/v1/status"):
            with self.subTest(path=path):
                reply = self.get(path)
                slots = reply if path == "/slots" else reply["live"]["slots"] if path == "/metrics" else reply["slots"]
                for slot in slots:
                    self.assertEqual((slot["kv_mode"], slot["kv_window"],
                                      slot["draft_kv_mode"], slot["draft_kv_window"]),
                                     ("resident", 4096, "off", 0))

    def test_legacy_info_leaves_status_unchanged(self):
        self.start(2)
        self.assertEqual(self.get("/slots"),
                         [{"id": 0, "n_ctx": 4096, "is_processing": False, "n_prompt_tokens": 0},
                          {"id": 1, "n_ctx": 4096, "is_processing": False, "n_prompt_tokens": 0}])
        self.assertEqual(self.get("/metrics")["live"]["slots"],
                         [{"slot": 0, "state": "idle", "held_tokens": 0},
                          {"slot": 1, "state": "idle", "held_tokens": 0}])
        self.assertNotIn("slots", self.get("/v1/status"))

    def test_multiplex_requires_native_acknowledgement(self):
        self.start(2)
        self.engine.close()
        self.engine.spawn[1].append("--resident-multiplex")
        import serve.server as server
        real = server.subprocess.Popen
        script = Path(self.tmp.name) / "fake_strata.py"
        with mock.patch.object(server.subprocess, "Popen",
                               lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
            with self.assertRaisesRegex(RuntimeError, "INFO resident_multiplex=1"):
                self.engine.restart(tries=1)
        self.assertFalse(self.engine.alive())
        self.assertIsNone(self.engine.proc)


    def test_quiet_resident_cancel_does_not_wait_for_another_output_token(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            gate.touch()
            self.start(2, more=("--hold-replies", str(gate)))
            engine = self.engine
            engine.slot_busy[1] = True
            engine.slot_claims.add(1)  # another owner prevents solo promotion
            cancel, admitted, stop_written = threading.Event(), threading.Event(), threading.Event()
            errors, commands = [], []
            send = engine._send

            def observe(command):
                send(command)
                commands.append(command)
                if command.startswith("BSTOP "):
                    stop_written.set()

            output = engine.generate_batched(list(b"LONGREPLY"), 64, {}, cancel)
            def consume():
                try:
                    for token in output:
                        if isinstance(token, int):
                            cancel.set()  # admission token; the peer withholds every resident BT
                            admitted.set()
                except Exception as error:
                    errors.append(error)

            worker = threading.Thread(target=consume, daemon=True)
            with mock.patch.object(engine, "_send", side_effect=observe):
                worker.start()
                try:
                    self.assertTrue(admitted.wait(2), "the request never reached its resident phase")
                    self.assertTrue(stop_written.wait(2), "quiet cancellation waited for output or the old ten-second poll")
                    self.assertTrue(worker.is_alive(), "ownership was released before the peer's terminal")
                    self.assertIn(0, engine.slot_claims)
                finally:
                    gate.unlink(missing_ok=True)  # permit the peer's matching cancellation terminal
                    worker.join(3)
                    engine.slot_busy[1] = False
                    engine.slot_claims.discard(1)
                    if not worker.is_alive():
                        output.close()
            self.assertFalse(worker.is_alive(), "the cancellation lifecycle did not drain")
            self.assertEqual(errors, [])
            self.assertNotIn("STOP", commands, "resident cancellation must not stop another owner's prompt read")
            self.assertNotIn(0, engine.slot_claims)
            self.assertEqual(self.chat("successor")["choices"][0]["message"]["content"], "ok, done.")

    def test_early_close_does_not_release_successor_reservation(self):
        self.start(2)
        engine = self.engine
        engine.slot_busy[1] = True  # peer reservation forces the real BGEN/BDONE path
        engine.slot_claims.add(1)
        release = engine._release_slot_when_done
        successor = []
        def reserve_after_drain(slot, stream, owner):
            release(slot, stream, owner)
            with engine.slot_cv:
                chosen = engine.pick_slot([], "")
                self.assertEqual(chosen, slot)
                self.assertNotIn(slot, engine.slot_claims, "drain must retire the old claim before publishing availability")
                engine.slot_busy[chosen] = True
                engine.slot_claims.add(chosen)
                successor.append(chosen)
        output = engine.generate_batched(list(b'LONGREPLY'), 64, {}, threading.Event())
        try:
            with mock.patch.object(engine, '_release_slot_when_done', side_effect=reserve_after_drain):
                self.assertIsInstance(next(output), int)
                self.assertIsInstance(next(output), int)  # first BT after BADM, not admission T
                output.close()
            self.assertEqual(len(successor), 1)
            self.assertTrue(engine.slot_busy[successor[0]], 'old consumer released the successor reservation')
            self.assertIn(successor[0], engine.slot_claims, "old finalizer erased the successor claim")
            with engine.slot_cv:
                engine.ORPHAN_SLOT_S = 0
                self.assertEqual(engine._reap_orphan_slots(), [])
                self.assertTrue(engine.slot_busy[successor[0]], "orphan scan stole an owned successor")
        finally:
            for slot in successor:
                engine.slot_busy[slot] = False
                engine.slot_claims.discard(slot)
            engine.slot_busy[1] = False
            engine.slot_claims.discard(1)
            output.close()

    def test_old_generation_finalizer_keeps_new_generation_claim(self):
        self.start(2)
        engine = self.engine
        engine.slot_busy[1] = True  # force slot admission rather than solo generation
        output = engine.generate_batched(list(b"LONGREPLY"), 64, {}, threading.Event())
        try:
            self.assertIsInstance(next(output), int)
            self.assertIsInstance(next(output), int)
            old_busy, old_claims = engine.slot_busy, engine.slot_claims
            self.assertEqual(old_claims, {0})
            import serve.server as server
            real = server.subprocess.Popen
            script = Path(self.tmp.name) / "fake_strata.py"
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine.restart(tries=1)
            with engine.slot_cv:
                chosen = engine.pick_slot([], "")
                self.assertEqual(chosen, 0)
                engine.slot_busy[chosen] = True
                engine.slot_claims.add(chosen)
            output.close()  # old-generation finally executes only now
            self.assertFalse(old_busy[0])
            self.assertEqual(old_claims, set(), "old generation's claim was not retired")
            self.assertTrue(engine.slot_busy[chosen])
            self.assertEqual(engine.slot_claims, {chosen}, "old generation discarded the new generation's claim")
            with engine.slot_cv:
                engine.ORPHAN_SLOT_S = 0
                self.assertEqual(engine._reap_orphan_slots(), [])
        finally:
            output.close()
            with engine.slot_cv:
                engine.slot_busy[:] = [False] * engine.batch
                engine.slot_claims.clear()

    def test_batch_groups_auto_follows_the_engine(self):
        """--batch-groups auto: the engine picks the groups and says so (INFO batch_groups=G); the server spreads the
        requests over those groups, and a plain number is read as before."""
        self.start(8, says_groups=4, more=["--batch-groups", "auto"])
        e = self.engine
        self.assertEqual((e.batch, e.slot_groups), (8, 4))
        self.assertEqual(e.slot_group, [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(e.slot_order[:4], [0, 2, 4, 6])
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, more=["--batch-groups", "auto"])          # an engine that reports none: one group
        self.assertEqual(self.engine.slot_groups, 1)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, more=["--batch-groups", "2"])
        self.assertEqual(self.engine.slot_groups, 2)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, says_groups=4)                           # 0.1.41: no flag on a layer split = the engine's choice
        self.assertEqual(self.engine.slot_groups, 4)
        self.tearDown(); self.httpd = self.engine = self.tmp = None
        self.start(8, says_groups=4, more=["--batch-groups", "1"])   # the opt-out
        self.assertEqual(self.engine.slot_groups, 1)

    def test_a_burst_of_connections_is_not_reset(self):
        """30-40 clients at once got "connection reset by peer" with the listen backlog of 5 (4 x R9700 burst): the
        server listens with a deep backlog (STRATA_HTTP_BACKLOG, 256), so 60 simultaneous requests all get answers.
        This fork admits a bounded number of requests at once, so an answer may be a 429 refusal: what the deep
        backlog has to deliver is that every connection is answered rather than reset or left hanging."""
        import serve.server as server
        self.assertGreaterEqual(server.Server.request_queue_size, 64)
        self.start(2)
        errs, ok, refused = [], [], []
        def one(i):
            try:
                self.chat(f"burst {i}", max_tokens=8)
                ok.append(i)
            except urllib.error.HTTPError as e:
                refused.append(e.code)
            except Exception as e:                      # noqa: BLE001
                errs.append(repr(e))
        th = [threading.Thread(target=one, args=(i,)) for i in range(60)]
        for t in th: t.start()
        for t in th: t.join()
        self.assertEqual(errs[:2], [])
        self.assertEqual(len(ok) + len(refused), 60)
        self.assertLessEqual(set(refused), {429}, refused)   # only the fork's bounded admission may refuse
        self.assertGreater(len(ok), 0, 'a burst is still served, not only refused')

    def test_stop_strings_in_a_batch_slot(self):
        """#454: a stop string cuts the answer in --batch mode too, and the slot is freed for the next request."""
        self.start(2)
        body = {"messages": [{"role": "user", "content": "hi LONGREPLY"}], "max_tokens": 64, "reasoning_effort": "none",
                "stop": ["la la"]}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            c = json.loads(r.read().decode())["choices"][0]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok, ", "stop"))
        self.assertEqual(self.chat("again")["choices"][0]["message"]["content"], "ok, done.")

    def test_engine_turns_batching_off(self):
        self.start(4, fit=0)
        self.assertEqual(self.engine.batch, 0)
        self.assertEqual(self.chat("hi")["choices"][0]["message"]["content"], "ok, done.")

    def queue_boundary(self, depth):
        """Effective HTTP reservations: two slots plus a configured waiting depth."""
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            gate.touch()
            self.start(4, fit=2, more=("--hold-replies", str(gate)))
            self.svc.configure_admission({"request_queue_limit": depth})
            replies, errors = [], []

            def go(index):
                try:
                    replies.append(self.chat(f"queued question {index}"))
                except Exception as error:  # reported in the test thread
                    errors.append(error)

            threads = [threading.Thread(target=go, args=(i,)) for i in range(2 + depth)]
            try:
                for thread in threads:
                    thread.start()
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    snapshot = self.get("/status")["admission"]
                    if (snapshot["active"], snapshot["waiting"], snapshot.get("executing"),
                            snapshot.get("native_waiting")) == (2, depth, 1, 1):
                        break
                    time.sleep(0.005)
                self.assertEqual((snapshot["slots"], snapshot["request_queue_limit"],
                                  snapshot["active"], snapshot["waiting"]), (2, depth, 2, depth))
                self.assertEqual((snapshot["executing"], snapshot["native_waiting"]), (1, 1))
                if depth:
                    self.assertGreater(snapshot["oldest_wait_s"], 0)
                else:
                    self.assertEqual(snapshot["oldest_wait_s"], 0)
                for path in ("/metrics", "/health", "/v1/status"):
                    observed = self.get(path)["admission"]
                    self.assertEqual((observed["active"], observed["waiting"],
                                      observed["request_queue_limit"]), (2, depth, depth))
                    self.assertEqual((observed["executing"], observed["native_waiting"]), (1, 1))
                    self.assertGreaterEqual(observed["retry_after_s"], 1)
                    self.assertLessEqual(observed["retry_after_s"], observed["retry_after_ceiling_s"])
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.chat("beyond the bounded capacity")
                with refused.exception as response:
                    self.assertEqual(response.code, 429)
                    self.assertIn(int(response.headers["Retry-After"]), range(1, 31))
                    self.assertIn("queue is full", json.load(response)["error"]["message"])
                with self.svc.status_lock:
                    if depth:
                        self.svc.admission_waiters[0]["_clock"] -= 40
                    else:
                        next(iter(self.svc.admission_active.values()))["_clock"] -= 40
                for path in ("/status", "/metrics", "/health", "/v1/status"):
                    old = self.get(path)["admission"]
                    self.assertEqual((old["active"], old["waiting"]), (2, depth))
                    self.assertEqual((old["retry_after_s"], old["retry_after_ceiling_s"]), (30, 30))
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.chat("retry while the queue is still full")
                with refused.exception as response:
                    self.assertEqual(response.code, 429)
                    self.assertEqual(response.headers["Retry-After"], "30")
            finally:
                gate.unlink(missing_ok=True)
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(replies), 2 + depth)
            self.assertTrue(all(reply["choices"][0]["message"]["content"] == "ok, done." for reply in replies))
            snapshot = self.get("/status")["admission"]
            self.assertEqual((snapshot["active"], snapshot["waiting"], snapshot["oldest_wait_s"]), (0, 0, 0))
            self.assertEqual(snapshot["retry_after_s"], 1)

    def test_configured_depth_refuses_at_effective_slots_plus_depth(self):
        self.queue_boundary(3)

    def test_zero_depth_refuses_at_effective_slots(self):
        self.queue_boundary(0)

    def test_native_budget_refusal_drains_without_poisoning_a_survivor(self):
        self.start(2)
        replies, errors = [], []

        def survivor():
            try:
                replies.append(self.chat("survivor LONGREPLY"))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=survivor)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not self.log.exists():
                time.sleep(0.005)
            self.assertTrue(self.log.exists(), "the survivor submitted work before the refusal")
            with self.assertRaises(urllib.error.HTTPError) as refused:
                self.chat("ADMISSIONREFUSE")
            with refused.exception as response:
                self.assertEqual(response.code, 400)
                self.assertIn("exceeds the context", json.load(response)["error"]["message"])
        finally:
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(replies[0]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertTrue(self.engine.alive(), "an explicit bounded refusal must not kill coexisting work")
        self.assertEqual(self.chat("replacement")["choices"][0]["message"]["content"], "ok, done.")
        refused_record = next(row for row in self.svc.lifecycle_records() if row["state"] == "error")
        self.assertEqual(refused_record.get("output_tokens", 0), 0)
        self.assertFalse(any(self.engine.slot_busy))
        snapshot = self.get("/metrics")["admission"]
        self.assertEqual((snapshot["active"], snapshot["waiting"], snapshot["executing"],
                          snapshot["native_waiting"]), (0, 0, 0, 0))

    def test_mixed_context_refusal_never_submits_or_truncates_work(self):
        self.start(2, more=("--says-context", "256"))
        self.assertEqual(self.chat("short", max_tokens=4)["usage"]["completion_tokens"], 4)
        before = self.log.read_text()
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.chat("long context " * 30, max_tokens=4)
        with refused.exception as response:
            self.assertEqual(response.code, 400)
            self.assertIn("never truncated", json.load(response)["error"]["message"])
        self.assertEqual(self.log.read_text(), before, "an over-context request must not reach GEN/BGEN")
        self.assertEqual(self.chat("short successor", max_tokens=4)["usage"]["completion_tokens"], 4)
        records = self.svc.lifecycle_records()
        self.assertEqual([row.get("output_tokens", 0) for row in records], [4, 0, 4])
        self.assertEqual([row["state"] for row in records], ["completed", "error", "completed"])

    def test_bounded_refusal_with_wrong_slot_boundary_still_fails_closed(self):
        self.start(2, more=("--bad-refusal-boundary",))
        # A one-token request enters BGEN even when alone.
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.chat("ADMISSIONREFUSE", max_tokens=1)
        with refused.exception as response:
            self.assertEqual(response.code, 503)
            self.assertIn("missing its slot boundary", json.load(response)["error"]["message"])
        self.assertFalse(self.engine.alive())

    def test_concurrent_requests_share_the_slots(self):
        self.start(2)
        # the first one decodes long enough for the others to arrive while it runs (also on a busy PC)
        texts = ["first question LONGREPLY", "second question, longer", "a third one waits for a slot"]
        reply = {t: "ok, " + "la " * 15 + "done." if "LONGREPLY" in t else "ok, done." for t in texts}
        out, errors, seen = {}, [], []

        def go(t):
            try:
                out[t] = self.chat(t)
            except Exception as e:   # noqa: BLE001 - reported below
                errors.append(e)

        threads = [threading.Thread(target=go, args=(t,)) for t in texts]
        for th in threads:
            th.start()
            time.sleep(0.05)
        deadline = time.time() + 30
        while any(th.is_alive() for th in threads) and time.time() < deadline:
            m = self.get("/metrics")["live"]
            seen.append(m)
            time.sleep(0.01)
        for th in threads:
            th.join(30)
        self.assertEqual(errors, [])
        for t in texts:
            self.assertEqual(out[t]["choices"][0]["message"]["content"], reply[t])
        # the slots were reported, and a request was admitted while another decoded in a slot (the fake logs how many
        # slots were active at each admission)
        self.assertTrue(any(m.get("parallel") == 2 and len(m.get("slots", [])) == 2 for m in seen))
        admitted = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertIn(1, admitted)
        # every request recorded its own row; nothing is left running
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)
            self.assertEqual(self.svc.live_reqs, {})
            self.assertFalse(self.svc.status["busy"])
        self.assertEqual(sorted(r["output_tokens"] for r in self.svc.history),
                         sorted(len(reply[t]) + 1 for t in texts))
        # never more than two in the slots at once (the fake logs how many were active at each admission)
        active = [int(x.split()[3]) for x in self.log.read_text().splitlines() if x.startswith("BGEN")]
        self.assertTrue(all(a <= 1 for a in active), active)
        identities = {row["id"] for row in self.svc.lifecycle_records()}
        seen_owners = {}
        for snapshot in seen:
            active_slots = [slot for slot in snapshot.get("slots", []) if slot["state"] != "idle"]
            request_ids = [slot["request_id"] for slot in active_slots]
            self.assertEqual(len(request_ids), len(set(request_ids)))
            self.assertLessEqual(set(request_ids), identities)
            for slot in active_slots:
                seen_owners.setdefault(slot["slot"], set()).add(slot["request_id"])
        self.assertTrue(any(len(owners) > 1 for owners in seen_owners.values()),
                        "a departing slot takes a new identity while a peer still runs")

    def test_wire_trace_reconciles_committed_tokens_not_scheduled_rows(self):
        import serve.server as server
        self.start(2)
        trace = io.StringIO()
        with mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            replies = self.race("trace owner LONGREPLY", "trace peer", delay=0.05)
        frames = [line.split(" wire< ", 1)[1] for line in trace.getvalue().splitlines() if " wire< " in line]
        emitted = sum(frame.startswith(("T ", "BT ")) for frame in frames)
        self.assertTrue(any(frame.startswith("BT ") for frame in frames), "trace exposes resident commits")
        self.assertEqual(emitted, sum(row["output_tokens"] for row in self.get("/metrics")["requests"]))
        self.assertEqual(replies["trace owner LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertEqual(replies["trace peer"][0], "ok, done.")
        self.assertTrue(any(frame.startswith("BADM ") for frame in frames))
        self.assertTrue(any(frame.startswith("BDONE ") for frame in frames))

    def race(self, first, second, delay=0.15, max_tokens=64):
        """`first` sent, `second` after `delay` seconds; -> {text: (reply, seconds it took)}"""
        res = {}

        def go(t):
            t0 = time.time()
            r = self.chat(t, max_tokens=max_tokens)
            res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
        a = threading.Thread(target=go, args=(first,))
        a.start()
        time.sleep(delay)
        go(second)
        a.join(30)
        return res

    def test_a_long_prompt_gives_way_to_a_short_one(self):
        """#656: a long solo prompt read gives way at a chunk boundary; the short request is answered first, then the
        long one goes on (in the slot its part waited in) and is answered too."""
        self.start(2)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        # the long read takes ~1 s (3,000 tokens, 32 a chunk, 10 ms a chunk): the short one did not wait for it
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)
        self.assertFalse(any(self.engine.slot_busy))

    def test_the_way_is_asked_between_far_apart_progress_lines(self):
        """#656 when the PP lines come far apart (the pipelined groups' read in pieces of one chunk: one PP per piece,
        at its end): the BYIELD goes out while the engine is quiet, so the read gives way at the first boundary after
        the short request came.  Asked only at a PP line, it reached the engine after that boundary's check and the
        read gave way one piece later."""
        self.start(2, more=["--pp-every", "80"])   # a boundary every 80 chunks (~0.8 s): one in this read
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        yields = [x for x in self.log.read_text().splitlines() if x.startswith("YIELD ")]
        self.assertEqual([int(x.split()[2]) for x in yields], [80 * 32], yields)

    def test_three_requests_long_long_short_do_not_deadlock(self):
        """ENGINE_REVIEW finding 1 (0.1.41 check): two long prompts fill both slots, a short one arrives a moment later
        and wants one of them (the yield path).  The reviewer's repro: long, long, short at 0.25 s steps.  Every
        request must be answered; none may wait on the control lock that the yielding read holds."""
        self.start(2)
        texts = ["long " * 600, "lung " * 600, "short"]
        res, errors = {}, []

        def go(t):
            t0 = time.time()
            try:
                r = self.chat(t, max_tokens=64)
                res[t] = (r["choices"][0]["message"]["content"], time.time() - t0)
            except Exception as e:      # noqa: BLE001 - a timeout is the failure this test looks for
                errors.append((t[:6], repr(e)))
        threads = []
        for t in texts:
            th = threading.Thread(target=go, args=(t,))
            th.start()
            threads.append(th)
            time.sleep(0.25)
        for th in threads:
            th.join(45)
        self.assertFalse(any(th.is_alive() for th in threads), "a request never finished (deadlock)")
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 3)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 3)

    def test_an_admission_gives_way_too(self):
        """The same while another request decodes in a slot: the long one is being admitted, a short one waits."""
        self.start(3)
        bg = threading.Thread(target=lambda: self.chat("background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertLess(res[short_q][1], res[long_q][1] - 0.5, res)
        log = self.log.read_text().splitlines()
        self.assertTrue(any(x.startswith("YIELD ") for x in log), log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_time_budget_silent_prefill_yields_before_the_ten_second_heartbeat(self):
        self.start(2, says_groups=2, more=("--says-budget", "--quiet-prefill"))
        replies, failures = {}, []
        def request(name, text):
            try:
                replies[name] = self.chat(text, max_tokens=12,
                                          priority="background" if name == "long" else "foreground")
            except Exception as error:
                failures.append(str(error))
        long = threading.Thread(target=request, args=("long", "long " * 600))
        long.start()
        time.sleep(.1)
        short = threading.Thread(target=request, args=("short", "short chat"))
        short.start()
        for thread in (short, long):
            thread.join(5)
        self.assertFalse(long.is_alive() or short.is_alive(), "silent prefill bypassed cooperative yield polling")
        self.assertEqual(failures, [])
        self.assertEqual(set(replies), {"long", "short"})
        log = self.log.read_text().splitlines()
        yielded = next(index for index, line in enumerate(log) if line.startswith("YIELD "))
        continued = next(index for index, line in enumerate(log) if line.startswith("PREFILL ") and
                         int(line.split()[2]) > 2500)
        self.assertLess(yielded, continued, "the short request waited for the whole silent prefill")
        self.assertFalse(any(self.engine.slot_busy))

    def test_time_budget_background_progresses_under_foreground_arrivals(self):
        self.start(3, says_groups=3, more=("--says-budget",))
        failures, replies = [], {}
        def request(name, text, priority="foreground"):
            try:
                replies[name] = self.chat(text, max_tokens=12, priority=priority)
            except Exception as error:
                failures.append(str(error))
        background = threading.Thread(target=request, args=("background", "long " * 600, "background"))
        background.start()
        arrivals = []
        for index in range(8):
            time.sleep(.08)
            thread = threading.Thread(target=request, args=(f"short-{index}", f"short arrival {index}"))
            thread.start()
            arrivals.append(thread)
        for thread in [background, *arrivals]:
            thread.join(15)
        self.assertFalse(any(thread.is_alive() for thread in [background, *arrivals]), "an arrival starved")
        self.assertEqual(failures, [])
        self.assertEqual(len(replies), 9)
        self.assertTrue(all(reply["choices"][0]["message"]["content"] == "ok, done." for reply in replies.values()))
        self.assertEqual(replies["background"]["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
                         "a cold request's own parked prefill is not reused input")
        log = self.log.read_text().splitlines()
        progress = [int(line.split()[2]) for line in log if line.startswith("PREFILL ") and int(line.split()[2]) > 64]
        self.assertGreater(max(progress), 2500, "background prefill did not keep advancing")
        yields = [int(line.split()[2]) for line in log if line.startswith("YIELD ")]
        self.assertGreater(len(yields), 2, "sustained arrivals must exercise beyond the legacy yield cap")
        self.assertEqual(yields, sorted(set(yields)), "a resumed read lost its arithmetic position")
        self.assertTrue(any(line.startswith("COMMIT ") for line in log), "decoders ran during the read")
        self.assertFalse(any(self.engine.slot_busy))

    def test_time_budget_several_decoders_and_prefill_complete_in_isolation(self):
        self.start(4, says_groups=2, more=("--says-budget",))
        replies, failures = {}, []
        def request(name, text, priority="foreground"):
            try:
                replies[name] = self.chat(text, max_tokens=60, priority=priority)
            except Exception as error:
                failures.append(str(error))
        threads = [threading.Thread(target=request, args=(str(i), f"decoder {i} LONGREPLY")) for i in range(3)]
        for thread in threads:
            thread.start()
            time.sleep(.03)
        prefill = threading.Thread(target=request, args=("prefill", "long " * 500, "background"))
        prefill.start()
        time.sleep(.15)
        short = threading.Thread(target=request, args=("short", "arriving short chat"))
        short.start()
        for thread in [*threads, prefill, short]:
            thread.join(15)
        self.assertFalse(any(thread.is_alive() for thread in [*threads, prefill, short]))
        self.assertEqual(failures, [])
        self.assertEqual(replies["prefill"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(replies["short"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(replies["prefill"]["usage"]["prompt_tokens_details"]["cached_tokens"], 0,
                         "a cold BGEN's first parked leg must not receive internal-prefix credit")
        for index in range(3):
            self.assertEqual(replies[str(index)]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        frames = self.log.read_text().splitlines()
        first_read = next(i for i, frame in enumerate(frames) if frame.startswith("PREFILL ") and int(frame.split()[2]) > 64)
        self.assertTrue(any(frame.startswith("COMMIT ") for frame in frames[first_read + 1:]),
                        "active decoders must commit while long prefill advances")
        self.assertFalse(any(self.engine.slot_busy))

    def test_an_admission_is_asked_between_far_apart_progress_lines(self):
        """As above for a read admitted beside a decoding slot (BGEN, `ask()`): with one PP line per 80 chunks the
        short request's BYIELD still reaches the first boundary."""
        self.start(3, more=["--pp-every", "80"])
        bg = threading.Thread(target=lambda: self.chat("LONGREPLY background", max_tokens=60))
        bg.start()
        time.sleep(0.05)
        long_q, short_q = "long " * 600, "short"
        res = self.race(long_q, short_q)
        bg.join(30)
        self.assertEqual(res[long_q][0], "ok, done.")
        self.assertEqual(res[short_q][0], "ok, done.")
        log = self.log.read_text().splitlines()
        admitted = [x.split() for x in log if x.startswith("BGEN ") and int(x.split()[2]) > 2600]
        self.assertTrue(admitted, log)                                 # the long read was an admission
        yields = [x.split()[1:] for x in log if x.startswith("YIELD ")]
        self.assertEqual(yields[:1], [[admitted[0][1], str(80 * 32)]], log)
        self.assertFalse(any(self.engine.slot_busy))

    def test_no_way_given_to_a_long_one(self):
        """Two prompts of about the same length: no BYIELD (it only goes to a much shorter one)."""
        self.start(2)
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_adaptive_policy_preserves_solo_concurrent_solo_identity_and_wire_totals(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            self.start(2, slot_cache=True, reuse=True,
                       more=("--says-budget", "--says-spec", "--mtp", "fake", "--spec", "4"))
        policy = self.engine.speculation_policy
        self.assertIsNotNone(policy)
        for exponent in range(1, 13):
            policy.observe("solo", 1, 1 << exponent, 200, 1)
            policy.observe("batch", 1, 1 << exponent, 20, 1)
            policy.observe("batch", 2, 1 << exponent, 20, 1)
        trace = io.StringIO()
        with mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            replies = self.race("adaptive owner LONGREPLY", "adaptive peer", delay=.05, max_tokens=200)
        self.assertEqual(replies["adaptive owner LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertEqual(replies["adaptive peer"][0], "ok, done.")
        kinds = [line.split()[0] for line in self.log.read_text().splitlines()
                 if line.startswith(("GEN ", "BGEN "))]
        self.assertEqual(kinds[0], "GEN")
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN")
        frames = [line.split(" wire< ", 1)[1] for line in trace.getvalue().splitlines() if " wire< " in line]
        owners = [line.split() for line in trace.getvalue().splitlines() if " owner " in line]
        requests = self.get("/metrics")["requests"]
        self.assertEqual(sum(frame.startswith(("T ", "BT ")) for frame in frames),
                         sum(request["output_tokens"] for request in requests))
        self.assertEqual(len({line[4] for line in owners}), 2, "one stable identity per request across returns")
        self.assertTrue(any(line[6] != "None" for line in owners), "the cohort was admitted through a physical slot")
        self.assertEqual(kinds.count("GEN"), 2, "the owner's own solo leg, and its return after the peer left")
        self.assertFalse(any(self.engine.slot_busy))
        self.assertEqual(policy.slots, {})

    def test_adaptive_unprofitable_solo_uses_real_ordinary_singleton(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            self.start(2, slot_cache=True,
                       more=("--says-budget", "--says-spec", "--mtp", "fake", "--spec", "4"))
        policy = self.engine.speculation_policy
        for exponent in range(1, 13):
            policy.observe("solo", 1, 1 << exponent, 8, 2)
            policy.observe("batch", 1, 1 << exponent, 100, 1)
        reply = self.chat("ordinary fallback", max_tokens=200)
        self.assertEqual(reply["choices"][0]["message"]["content"], "ok, done.")
        kinds = [line.split()[0] for line in self.log.read_text().splitlines()
                 if line.startswith(("GEN ", "BGEN "))]
        self.assertEqual(kinds, ["BGEN"], "no invented target-only command or spurious return to MTP")
        self.assertFalse(any(self.engine.slot_busy))

    def test_adaptive_without_mtp_reports_inactive_and_retains_static_paths(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.start(2, more=("--says-budget",))
        self.assertIn("no loaded solo MTP", output.getvalue())
        self.assertIsNone(self.engine.speculation_policy)
        self.chat("ordinary no drafter")
        self.assertEqual(self.log.read_text().split()[0], "GEN")

    def test_adaptive_refuses_a_different_batch_mechanism(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            with self.assertRaisesRegex(ValueError, "ordinary batches"):
                self.start(2, slot_cache=True,
                           more=("--says-budget", "--says-spec", "--mtp", "fake", "--batch-mtp"))

    def test_adaptive_requires_the_fixed_foreground_budget(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "adaptive"}):
            with self.assertRaisesRegex(ValueError, "foreground-budget-ms"):
                self.start(2, slot_cache=True, more=("--says-spec", "--mtp", "fake", "--spec", "4"))

    def test_static_default_does_not_construct_or_train_a_policy(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ):
            server.os.environ.pop("STRATA_SPEC_POLICY", None)
            self.start(2, slot_cache=True,
                       more=("--says-budget", "--says-spec", "--mtp", "fake", "--spec", "4"))
        self.assertIsNone(self.engine.speculation_policy)
        self.chat("static unchanged")
        self.assertEqual(self.log.read_text().split()[0], "GEN")

    def test_invalid_policy_is_refused_before_any_native_process(self):
        import serve.server as server
        with mock.patch.dict(server.os.environ, {"STRATA_SPEC_POLICY": "invalid"}), \
                mock.patch.object(server.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "static or adaptive"):
                StrataEngine("not-launched", [])
            spawn.assert_not_called()

    def test_no_way_given_to_a_long_one_between_far_apart_progress_lines(self):
        """The ask between PP lines keeps the rule: a waiting request about as long gets no BYIELD there either."""
        self.start(2, more=["--pp-every", "80"])
        res = self.race("long " * 600, "lung " * 600)
        self.assertEqual(sorted(r[0] for r in res.values()), ["ok, done."] * 2)
        self.assertFalse(any(x.startswith("YIELD ") for x in self.log.read_text().splitlines()))

    def test_left_alone_it_goes_back_to_the_solo_path(self):
        """A request decoding in a slot whose neighbour finished goes back to the solo path (MTP drafts), continued
        from its slot - only with an engine that keeps the slots' conversations (INFO slot_cache=1)."""
        self.start(2, slot_cache=True, reuse=True, more=("--readfield", "--draft-counters", "7,11"))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.recorded_chat(text="LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.05)
        short = self.chat("short", max_tokens=200)
        long.join(30)
        self.assertEqual(short["choices"][0]["message"]["content"], "ok, done.")
        response, record = res["long"]
        self.assertEqual(response["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertEqual([leg["kind"] for leg in record["legs"]], ["solo", "batch", "solo"])
        self.assertTrue(all(leg["generated"] > 0 for leg in record["legs"]))
        self.assertEqual(record["committed_tokens"], response["usage"]["completion_tokens"])
        self.assertEqual(record["committed_tokens"], sum(leg["generated"] for leg in record["legs"]))
        self.assertEqual((response["timings"]["draft_n_accepted"], response["timings"]["draft_n"]), (7, 11))
        self.assertEqual(response["timings"]["prompt_ms"], sum(leg["prompt_ms"] for leg in record["legs"]))
        self.assertEqual(response["timings"]["prompt_ms"], 15.0)
        self.assertEqual(response["usage"]["prompt_tokens_details"]["cached_tokens"], 3)
        self.assertEqual(sum(leg["prompt_read"] for leg in record["legs"]), response["usage"]["prompt_tokens"] - 3 + 2)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        # solo, then both in slots, then the long one alone again: a GEN after the BGENs
        self.assertIn("BGEN", kinds)
        self.assertEqual(kinds[-1], "GEN", kinds)
        self.assertFalse(any(self.engine.slot_busy))
        with self.svc.status_lock:
            self.assertEqual(len(self.svc.history), 2)

    def test_next_turn_goes_back_to_its_slot(self):
        self.start(2)
        a = "conversation A, first turn"
        r1 = {}
        th = threading.Thread(target=lambda: r1.setdefault("x", self.chat("blocker LONGREPLY", max_tokens=200)))
        th.start()
        time.sleep(0.05)
        first = self.chat(a)
        th.join(30)
        deadline = time.time() + 10                     # a slot left at its stop token is freed at its BDONE
        while not [h for h in self.engine.slot_held if h] and time.time() < deadline:
            time.sleep(0.02)
        held = [h for h in self.engine.slot_held if h]
        self.assertTrue(held, "a finished slot request keeps what its slot holds")
        # the held tokens are the prompt and every token fed: all of the reply but its last token
        reply = list(b"ok, done.") + [257]
        mine = [h for h in held if h[-len(reply) + 1:] == reply[:-1] and bytes(x for x in h if x < 256).find(b"conversation A") >= 0]
        self.assertEqual(len(mine), 1, held)
        slot = self.engine.slot_held.index(mine[0])
        self.assertEqual(self.engine.pick_slot(mine[0] + [10, 11], self.engine.slot_scope[slot]), slot)
        self.assertEqual(first["choices"][0]["message"]["content"], "ok, done.")

    def test_a_failed_window_names_its_error(self):
        # #997 #890: a batch window fails while every request reads its own slot's lines, so no request reads the
        # engine's ERR line before it exits; the server's note on the death gives it instead of a guess
        self.start(2, fail=True)
        errors = []

        def go(t):
            try:
                self.chat(t)
            except urllib.error.HTTPError as e:
                errors.append(e.code)

        threads = [threading.Thread(target=go, args=(t,)) for t in ("first question LONGREPLY", "second question")]
        for th in threads:
            th.start()
            time.sleep(0.05)
        for th in threads:
            th.join(30)
        self.assertEqual(len(errors), 2, errors)
        note = self.engine.death_note()
        self.assertIn("exited (code 1)", note)
        self.assertIn("verify batch: layer 34 never rang (graph finished)", note)

    def test_nonfinite_in_batch_mode(self):
        """#879 in --batch mode: DONE ... nonfinite on the solo path or at an admission (nothing emitted yet) sends the
        request again once; BDONE ... nonfinite in a slot ends the reply that had started, as "length"."""
        self.start(2)
        self.assertEqual(self.chat("NFSOLO alone")["choices"][0]["message"]["content"], "ok, done.")
        gens = [x for x in self.log.read_text().splitlines() if x.startswith("GEN")]
        self.assertEqual(len(gens), 2, gens)                              # sent again once
        res = self.race("first question LONGREPLY", "NFADMIT second")
        self.assertEqual(res["NFADMIT second"][0], "ok, done.")
        self.assertEqual(res["first question LONGREPLY"][0], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(x.startswith("BGEN") for x in self.log.read_text().splitlines()))
        out = {}

        def go(t):
            body = {"messages": [{"role": "user", "content": t}], "max_tokens": 64, "reasoning_effort": "none"}
            req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out[t] = json.loads(r.read().decode())["choices"][0]
        a = threading.Thread(target=go, args=("third question LONGREPLY",))
        a.start()
        time.sleep(0.15)
        go("NFSLOT fourth")
        a.join(30)
        c = out["NFSLOT fourth"]
        self.assertEqual((c["message"]["content"], c["finish_reason"]), ("ok", "length"))
        self.assertEqual(out["third question LONGREPLY"]["message"]["content"], "ok, " + "la " * 15 + "done.")
        # the engine dropped that slot's state: the server does not route the conversation's next turn to it
        self.assertFalse([h for h in self.engine.slot_held if bytes(t & 255 for t in h).find(b"NFSLOT") >= 0])
        with self.svc.status_lock:
            self.assertEqual(self.svc.live_reqs, {})

    def test_a_continued_request_reports_its_own_prompt_reuse(self):
        """A request STOPped on the solo path and continued in a slot (or back on the solo path) is sent again as its
        prompt + what it generated, and the engine holds all of that: the usage must still split the request's OWN
        prompt (input_tokens + cache_read_input_tokens = the prompt, input_tokens >= 1).  input_tokens 0 makes Claude
        Code count the prompt twice (message_start's input_tokens + cache_read_input_tokens): "Prompt is too long"
        at half the context."""
        self.start(2, slot_cache=True, reuse=True)
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.message("LONGREPLY please", 200)))
        long.start()
        time.sleep(0.05)
        self.message("short", 200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertIn("BGEN", kinds)                    # it was continued: the case under test
        u = res["long"]["usage"]
        prompt = self.svc.history[[h["output_tokens"] for h in self.svc.history].index(
            max(h["output_tokens"] for h in self.svc.history))]["prompt_total"]
        self.assertGreaterEqual(u["input_tokens"], 1, u)
        self.assertEqual(u["input_tokens"] + u["cache_read_input_tokens"], prompt, u)
        self.assertEqual(u["cache_read_input_tokens"], 3, u)    # the first read's reuse, not the continuation's

    def test_a_slot_request_ended_by_its_stop_token_reports_the_slot_leg(self):
        """#1869: the consumer leaves at the stop token, before the BDONE: the request kept the admission's one token and
        its few ms.  Its figures are the slot leg's now (finish=stop was the broken case, finish=length read the BDONE)."""
        import serve.server as server
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        with mock.patch.object(server, "EOS_IDS", {257}):          # the fake engine's stop token
            a = threading.Thread(target=lambda: res.setdefault("a", self.chat("List three primes please", 200)))
            b = threading.Thread(target=lambda: res.setdefault("b", self.chat("Name two rivers in Germany", 200)))
            a.start(); b.start(); a.join(30); b.join(30)
        self.assertEqual(len(self.svc.history), 2)
        for h in self.svc.history:
            self.assertEqual(h["engine_generated"], h["output_tokens"], h)
            self.assertGreaterEqual(h["decode_ms"], 1.0, h)
        for r in res.values():
            t = r["timings"]
            self.assertEqual(t["prompt_n"] + t["cache_n"], r["usage"]["prompt_tokens"], t)
            self.assertEqual(t["predicted_n"], 10, t)
        self.assertFalse(any(self.engine.slot_busy))                 # the slots are free (no BDONE left for a wait)

    def test_a_request_moved_between_solo_and_slot_keeps_its_totals(self):
        """#1869/#1870: a request that read its prompt and decoded on the solo path, was STOPped when another arrived and
        went on in a slot (its admission copies the held prefix: prompt_read 1, one token) reports the prompt it read and
        the decode time and tokens of ALL its legs, not the last one's."""
        self.start(2, slot_cache=True, reuse=True, more=("--readfield",))
        res = {}
        long = threading.Thread(target=lambda: res.setdefault("long", self.chat("LONGREPLY please", max_tokens=200)))
        long.start()
        time.sleep(0.25)                                   # a few solo tokens first
        self.chat("short", max_tokens=200)
        long.join(30)
        kinds = [x.split()[0] for x in self.log.read_text().splitlines()]
        self.assertEqual(kinds[:2], ["GEN", "BGEN"], kinds)           # solo, then continued in a slot: the case under test
        n = res["long"]["usage"]["completion_tokens"]
        row = max(self.svc.history, key=lambda h: h["output_tokens"])
        self.assertEqual(row["engine_generated"], n, row)
        self.assertGreater(row["decode_ms"], 20.0 * 3, row)          # the solo leg's STEP-long tokens are in it
        self.assertGreater(row["prompt_read"], 1, row)                # the first read, not the copy leg's 1
        t = res["long"]["timings"]
        self.assertEqual(t["prompt_n"] + t["cache_n"], res["long"]["usage"]["prompt_tokens"], t)
        self.assertEqual(t["cache_n"], 3, t)                          # the first read's reuse only
        self.assertGreater(t["predicted_ms"], 60.0, t)

    def test_a_leaked_busy_slot_does_not_hang_the_next_request(self):
        """#1603: a slot whose busy flag has no owner (no request holds it, no BDONE wait is running) kept every later
        request waiting for a slot nothing would free: running 0, queued 0, and the frozen-engine check never looked.
        The waiting request is now counted (admitting), the slot is given back after ORPHAN_SLOT_S, and it is
        answered."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.4
        with e.slot_cv:
            e.slot_busy[0] = e.slot_busy[1] = True       # the leak: busy, claimed by nobody, nothing ending them
        # the solo path needs a quiet engine; a busy flag sends the request to the slot path, which waits
        res, errs = {}, []
        def go():
            try:
                res["r"] = self.chat("after the leak", max_tokens=16)
            except Exception as ex:                       # noqa: BLE001
                errs.append(ex)
        t = threading.Thread(target=go)
        t0 = time.time()
        t.start()
        seen = []
        while t.is_alive() and time.time() - t0 < 20:
            live = self.get("/metrics")["live"]
            seen.append((live.get("running"), live.get("queued"), live.get("admitting")))
            time.sleep(0.05)
        t.join(5)
        self.assertEqual(errs, [])
        self.assertFalse(t.is_alive(), "the request hung on a slot nobody owns")
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        # while it waited it was visible: running as a request, and counted as waiting for a slot - never 0 / 0 / 0
        self.assertTrue(any(r == 1 and a == 1 for r, q, a in seen), seen)
        first = next(i for i, x in enumerate(seen) if sum(v or 0 for v in x))
        self.assertTrue(all(sum(v or 0 for v in x) for x in seen[first:-1]), seen)   # never invisible once it arrived
        self.assertEqual(e.admitting, 0)
        self.assertEqual(e.slot_claims, set())

    def test_an_owned_busy_slot_is_never_taken(self):
        """The leak check frees only slots with no owner: a slot a request holds stays busy however long it waits."""
        self.start(2)
        e = self.engine
        e.ORPHAN_SLOT_S = 0.0
        with e.slot_cv:
            e.slot_busy[0] = True
            e.slot_claims.add(0)
            e.slot_busy[1] = True
            e.slot_releasing.add(1)
            self.assertEqual(e._reap_orphan_slots(), [])
            self.assertEqual(e.slot_busy, [True, True])
            e.slot_claims.discard(0)
            self.assertEqual(e._reap_orphan_slots(), [0])
            self.assertEqual(e.slot_busy, [False, True])

    def test_a_parallel_request_does_not_wait_for_the_fifo_on_a_loaded_engine(self):
        """#1603: parallel requests used to take the FIFO (held by an image encode, /v1/vram, load) just to find the
        engine loaded; between 'queued' and 'running' they were neither.  Loaded: no fifo.  Unloaded: they wait for it
        and are counted as queued the whole time."""
        self.start(2)
        self.assertEqual(self.chat("warm")["choices"][0]["message"]["content"], "ok, done.")
        self.svc.fifo.acquire()
        try:
            r = self.chat("with the fifo held")             # would block here before
            self.assertEqual(r["choices"][0]["message"]["content"], "ok, done.")
        finally:
            self.svc.fifo.release()
        import serve.server as server
        script, real = Path(self.tmp.name) / "fake_strata.py", server.subprocess.Popen
        patch = mock.patch.object(server.subprocess, "Popen",      # the reload starts the fake again
                                  lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        patch.start()
        self.addCleanup(patch.stop)
        self.engine.unload()
        self.svc.fifo.acquire()
        res = {}
        t = threading.Thread(target=lambda: res.setdefault("r", self.chat("reloads", max_tokens=16)))
        t.start()
        try:
            deadline = time.time() + 10
            live = {}
            while time.time() < deadline:
                live = self.get("/metrics")["live"]
                if live.get("queued"):
                    break
                time.sleep(0.05)
            self.assertEqual(live.get("queued"), 1, live)    # waiting for the fifo: queued, not nothing
            self.assertEqual(live.get("running"), 0, live)
        finally:
            self.svc.fifo.release()
        t.join(60)
        self.assertEqual(res["r"]["choices"][0]["message"]["content"], "ok, done.")
        self.assertEqual(self.get("/metrics")["live"]["queued"], 0)


class LegacyRequestIdentity(unittest.TestCase):
    """#260: the real legacy consumer and cancellation lifecycle over the scripted engine pipe."""
    start = ParallelService.start
    tearDown = ParallelService.tearDown

    def force_slot(self, more=()):
        self.start(2, more=("--legacy-identity", *more))
        self.engine.slot_busy[1] = True
        self.engine.slot_claims.add(1)  # another owner keeps each request on BGEN, not the solo fast path

    def wait_until(self, predicate):
        until = time.monotonic() + 3
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.005)
        self.fail("legacy identity lifecycle did not reach its bounded condition")

    def test_matching_terminal_completes_once(self):
        self.force_slot()
        engine = self.engine
        with mock.patch.object(engine, "_apply_bdone", wraps=engine._apply_bdone) as apply:
            tokens = list(engine.generate_batched([1, 2], 3, {}, threading.Event()))
        self.assertEqual(tokens, list(b"ok,"))
        self.assertEqual(engine._slot_identity[0], (1, 1))
        self.assertEqual((engine.last["generated"], engine.last["finish"]), (3, "length"))
        self.assertEqual(len(engine.last["legs"]), 1)
        self.assertEqual(apply.call_count, 1)
        self.assertFalse(engine.slot_busy[0])
        self.assertNotIn(0, engine.slot_claims)
        self.assertEqual(list(engine._ignored_bdone), [])

    def test_solo_transition_stop_uses_the_admitted_identity(self):
        self.force_slot(("--slotcache",))
        engine = self.engine
        output = engine.generate_batched([1, 2], 64, {}, threading.Event())
        with mock.patch.object(engine, "_send", wraps=engine._send) as send:
            tokens = [next(output), next(output)]
            engine.slot_busy[1] = False
            engine.slot_claims.discard(1)
            tokens.extend(output)
        stops = [call.args[0] for call in send.call_args_list if call.args[0].startswith("BSTOP ")]
        self.assertEqual(stops, ["BSTOP 0 1 1"])
        self.assertEqual(tokens, [*b"ok, done.", 257])
        self.assertEqual(len(engine.last["legs"]), 2)
        self.assertEqual([leg["finish"] for leg in engine.last["legs"]], ["cancel", "stop"])
        self.assertFalse(any(engine.slot_busy))
        self.assertEqual(engine.slot_claims, set())

    def test_identity_cancel_is_ignored_after_slot_reuse_without_another_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "hold"
            self.force_slot(("--hold-replies", str(gate)))
            engine = self.engine
            cancel = threading.Event()
            output = engine.generate_batched([1, 2], 64, {}, cancel)
            completed, errors = [], []
            with mock.patch.object(engine, "_send", wraps=engine._send) as send:
                self.assertEqual(next(output), ord("o"))  # admission token, BADM not read yet
                self.assertEqual(next(output), ord("k"))  # resident token, admitted identity now known
                cancel.set()
                list(output)
                self.assertEqual(engine.last["finish"], "cancel")
                stops = [call.args[0] for call in send.call_args_list if call.args[0].startswith("BSTOP ")]
                self.assertEqual(stops, ["BSTOP 0 1 1"])
            gate.touch()
            def successor():
                try:
                    tokens = list(engine.generate_batched([3, 4], 4, {}, threading.Event()))
                    completed.append((tokens, dict(engine.last)))
                except Exception as error:
                    errors.append(error)
            worker = threading.Thread(target=successor, daemon=True)
            worker.start()
            try:
                self.wait_until(lambda: engine._slot_identity[0] == (2, 2))
                engine._send(stops[0])  # replay the previous owner's actual cancel after the new BADM
                self.wait_until(lambda: "IGNORED BSTOP 0 1 1" in self.log.read_text())
                self.assertTrue(engine.slot_busy[0], "ignored stale cancellation released the successor")
                self.assertIn(0, engine.slot_claims)
                self.assertTrue(worker.is_alive())
            finally:
                gate.unlink(missing_ok=True)
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0][0], list(b"ok, "))
            self.assertEqual((completed[0][1]["generated"], completed[0][1]["finish"]), (4, "length"))
            self.assertEqual(self.log.read_text().count("TERMINAL BDONE "), 2)
            self.assertFalse(engine.slot_busy[0])
            self.assertNotIn(0, engine.slot_claims)

    def test_stale_terminal_after_reuse_is_not_attributed_to_the_successor(self):
        self.force_slot(("--legacy-stale-done",))
        engine = self.engine
        first = list(engine.generate_batched([1, 2], 3, {}, threading.Event()))
        first_figures = dict(engine.last)
        self.assertEqual(first, list(b"ok,"))
        with mock.patch.object(engine, "_apply_bdone", wraps=engine._apply_bdone) as apply:
            second = list(engine.generate_batched([3, 4], 5, {}, threading.Event()))
        self.assertEqual(second, list(b"ok, d"))
        self.assertEqual(engine._slot_identity[0], (2, 2))
        self.assertEqual((first_figures["generated"], first_figures["finish"]), (3, "length"))
        self.assertEqual((engine.last["generated"], engine.last["finish"]), (5, "length"))
        self.assertEqual(len(engine.last["legs"]), 1)
        self.assertEqual(len(engine._ignored_bdone), 2)
        self.assertEqual(sum(call.args[0][2] == "5" for call in apply.call_args_list), 1)
        self.assertEqual(self.log.read_text().count("TERMINAL BDONE "), 2)
        self.assertFalse(engine.slot_busy[0])
        self.assertNotIn(0, engine.slot_claims)


class PackedTransport(unittest.TestCase):
    """Real HTTP/frontend transport coverage; arithmetic is exercised by the native interleave vehicle."""
    start = ParallelService.start
    tearDown = ParallelService.tearDown
    chat = ParallelService.chat
    get = ParallelService.get

    def wait_until(self, predicate, seconds=5):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.005)
        self.fail("packed transport did not reach its bounded condition")

    def test_retained_resume_submits_input_once_and_keeps_one_output_budget(self):
        self.start(2, more=("--retained",))
        response = self.chat("retained " * 20 + "LONGREPLY", max_tokens=8)
        lines = self.log.read_text().splitlines()
        self.assertEqual(response["usage"]["completion_tokens"], 8)
        submitted = [line for line in lines if line.startswith("BGENQ ")]
        resumed = [line for line in lines if line.startswith("BQRESUME ")]
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(resumed), 1)
        self.assertEqual(submitted[0].split()[1:3], resumed[0].split()[1:3])
        self.assertEqual(len([line for line in lines if line.startswith("BQDONE ")]), 1)
        self.assertFalse(any(line.startswith(("GEN ", "BGEN ", "BYIELD ")) for line in lines))
        self.assertTrue(self.engine.alive())
        self.assertFalse(any(self.engine.slot_busy))

    def test_retained_without_budget_waits_for_a_physical_slot(self):
        with tempfile.TemporaryDirectory() as root:
            hold = Path(root) / "replies-held"
            hold.touch()
            self.start(2, more=("--retained", "--hold-replies", str(hold)))
            self.assertEqual(self.engine.info.get("retained_prompt"), 1)
            self.assertNotEqual(self.engine.info.get("packed_prefill"), 1)
            self.assertEqual(self.engine.foreground_budget_s, 0)
            results, errors = {}, []
            waiting, waiter_done = threading.Event(), threading.Event()
            def phase(value):
                if value == "physical_slot_wait":
                    waiting.set()
            def request(index):
                try:
                    sampling = {"_strata_cache_scope": "a" * 64}
                    if index == 2:
                        sampling["_strata_frontend_phase"] = phase
                    results[index] = [token for token in self.engine.generate_batched(
                        [65, 66, 67 + index], 3, sampling, threading.Event()) if token is not None]
                except Exception as error:
                    errors.append(error)
                finally:
                    if index == 2:
                        waiter_done.set()
            threads = [threading.Thread(target=request, args=(index,)) for index in range(3)]
            try:
                for thread in threads[:2]:
                    thread.start()
                self.wait_until(lambda: len(self.engine.packed_q) == 2 and
                                all(owner["identity"] is not None for owner in self.engine.packed_q.values()))
                self.assertTrue(all(self.engine.slot_busy))
                threads[2].start()
                self.wait_until(lambda: waiting.is_set() or waiter_done.is_set())
                self.assertEqual(errors, [], "a full retained engine must wait, not lose its reservation")
                self.assertTrue(waiting.is_set())
                self.assertFalse(waiter_done.is_set())
                self.assertFalse(self.engine.ctl.locked())
                self.assertEqual(self.log.read_text().count("BGENQ "), 2)
            finally:
                hold.unlink(missing_ok=True)
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual({index: len(tokens) for index, tokens in results.items()}, {0: 3, 1: 3, 2: 3})
            self.assertEqual(self.log.read_text().count("BGENQ "), 3)
            self.assertTrue(self.engine.alive())
            self.assertFalse(any(self.engine.slot_busy))
            self.assertFalse(self.engine.packed_q)

    def test_retained_suspended_cancel_keeps_peer_at_full_output(self):
        self.start(2, more=("--retained", "--hold-resume"))
        cancel = threading.Event()
        result, errors = [], []
        def reading():
            try:
                result.extend(self.engine.generate_batched([65] * 96, 8, {}, cancel))
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=reading)
        thread.start()
        self.wait_until(lambda: any(live and live["state"] == "suspended" for live in self.engine.slot_live))
        cancel.set()
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([token for token in result if token is not None], [])
        # A short peer does not need suspension and must retain its entire budget.
        peer = list(self.engine.generate_batched([66, 67], 8, {}, threading.Event()))
        self.assertEqual(len([token for token in peer if token is not None]), 8)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.log.read_text().count(" cancel"), 1)
        self.assertFalse(any(self.engine.slot_busy))

    def test_retained_resume_refusal_is_one_request_local_terminal(self):
        self.start(2, more=("--retained", "--retained-refuse"))
        with self.assertRaisesRegex(ValueError, "continuation was refused"):
            list(self.engine.generate_batched([65] * 96, 8, {}, threading.Event()))
        peer = list(self.engine.generate_batched([66, 67], 8, {}, threading.Event()))
        self.assertEqual(len([token for token in peer if token is not None]), 8)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.log.read_text().count("BQRESUME "), 1)
        self.assertFalse(any(self.engine.slot_busy))

    def test_one_effective_slot_still_uses_the_tagged_native_path(self):
        self.start(1, more=("--packed",))
        self.assertEqual(self.engine.batch, 1)
        reply = self.chat("one general slot", max_tokens=1)
        self.assertEqual(reply["usage"]["completion_tokens"], 1)
        self.assertIn("BGENQ ", self.log.read_text())
        self.assertFalse(any(self.engine.slot_busy))
        self.assertFalse(self.engine.packed_q)

    def test_native_admission_flag_does_not_migrate_packed_only_capability(self):
        self.start(1, more=("--packed", "--native-admission"))
        reply = self.chat("packed-only remains frontend-seated", max_tokens=1)
        self.assertEqual(reply["usage"]["completion_tokens"], 1)
        self.assertNotEqual(self.engine.info.get("native_admission"), 1)
        self.assertEqual(self.log.read_text().count("BGENQ "), 1)
        self.assertNotIn("BQREGISTER", self.log.read_text())
        self.assertFalse(self.engine.packed_q)

    def test_packing_policy_is_per_request_and_reaches_the_wire(self):
        self.start(2, more=("--packed",))
        self.chat("ordinary request", max_tokens=1)
        self.chat("priority request", max_tokens=1, packing="solo")
        lines = [line for line in self.log.read_text().splitlines() if line.startswith("BGENQ ")]
        self.assertEqual(len(lines), 2)
        self.assertIn("packing=eligible", lines[0])
        self.assertIn("packing=solo", lines[1])

    def test_invalid_packing_is_refused_before_the_engine(self):
        self.start(1, more=("--packed",))
        with self.assertRaises(urllib.error.HTTPError) as refused:
            self.chat("bad policy", max_tokens=1, packing="packed")
        self.assertEqual(refused.exception.code, 400)
        self.assertIn("strata_packing", refused.exception.read().decode())
        self.assertFalse(self.engine.packed_q)

    def test_reservation_ack_releases_control_before_prefill_with_four_general_slots(self):
        with tempfile.TemporaryDirectory() as root:
            hold = Path(root) / "prefill-held"
            hold.touch()
            self.start(4, more=("--packed", "--hold-replies", str(hold), "--says-budget"))
            replies, errors = {}, []
            def request(index):
                try:
                    replies[index] = self.chat(f"owner-{index} " + "different prompt " * (index + 1),
                                               max_tokens=1, priority="background" if index == 3 else "foreground")
                except Exception as error:
                    errors.append(error)
            threads = [threading.Thread(target=request, args=(index,)) for index in range(4)]
            try:
                for thread in threads:
                    thread.start()
                self.wait_until(lambda: self.log.exists() and self.log.read_text().count("BGENQ ") == 4)
                self.wait_until(lambda: not self.engine.ctl.locked())
                self.assertEqual(sum(self.engine.slot_busy), 4)
                self.assertEqual(replies, {}, "ACK must not pretend the prompt or first token completed")
                self.assertEqual(len(self.engine.packed_q), 4)
                self.assertTrue(all(owner["identity"] is not None for owner in self.engine.packed_q.values()))
            finally:
                hold.unlink(missing_ok=True)
                for thread in threads:
                    thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(replies), 4)
            self.assertTrue(all(reply["usage"]["completion_tokens"] == 1 for reply in replies.values()))
            self.assertFalse(any(self.engine.slot_busy))
            self.assertFalse(self.engine.packed_q)

    def test_cancel_during_prefill_is_owner_local_and_request_reuse_gets_a_new_identity(self):
        with tempfile.TemporaryDirectory() as root:
            hold = Path(root) / "prefill-held"
            hold.touch()
            self.start(2, more=("--packed", "--hold-replies", str(hold), "--slotcache"))
            cancels = [threading.Event(), threading.Event()]
            results, errors = {}, []
            def request(index):
                try:
                    tokens = list(self.engine.generate_batched(
                        [20 + index, 40 + index, 60 + index], 1,
                        {"_strata_cache_scope": ("a" if index == 0 else "b") * 64}, cancels[index]))
                    results[index] = ([token for token in tokens if token is not None], dict(self.engine.last))
                except Exception as error:
                    errors.append(error)
            threads = [threading.Thread(target=request, args=(index,)) for index in range(2)]
            try:
                for thread in threads:
                    thread.start()
                self.wait_until(lambda: len(self.engine.packed_q) == 2 and
                                all(owner["identity"] is not None for owner in self.engine.packed_q.values()))
                old = set(self.engine.packed_q)
                cancels[0].set()
                self.wait_until(lambda: 0 in results)
                self.assertEqual(results[0][0], [])
                self.assertEqual(results[0][1]["finish"], "cancel")
                self.assertNotIn(1, results)
            finally:
                hold.unlink(missing_ok=True)
                for thread in threads:
                    thread.join(10)
            self.assertEqual(errors, [])
            self.assertEqual(len(results[1][0]), 1)
            replacement = list(self.engine.generate_batched(
                [21, 41, 61, 70], 1, {"_strata_cache_scope": "b" * 64}, threading.Event()))
            self.assertEqual(len([token for token in replacement if token is not None]), 1)
            tags = {line.split()[1] for line in self.log.read_text().splitlines() if line.startswith("BGENQ ")}
            self.assertEqual(len(tags), 3)
            self.assertEqual(len(tags - old), 1)
            self.assertGreater(self.engine.last["reused"], 0)
            self.assertFalse(any(self.engine.slot_busy))
            self.assertFalse(self.engine.ctl.locked())

    def test_mismatched_identity_and_duplicate_ack_fail_closed(self):
        for flag in ("--packed-wrong-id", "--packed-duplicate-ack"):
            with self.subTest(flag=flag):
                self.start(2, more=("--packed", flag))
                try:
                    with self.assertRaises(urllib.error.HTTPError) as failed:
                        self.chat("must keep exact correlation", max_tokens=1)
                    with failed.exception as response:
                        self.assertEqual(response.code, 503)
                    self.assertFalse(self.engine.alive())
                    self.assertFalse(any(self.engine.slot_busy))
                finally:
                    self.tearDown()
                    self.httpd = self.engine = self.tmp = None

    def test_native_fatal_invalidates_packed_owner_before_trailing_output(self):
        self.start(2, more=("--packed", "--packed-fatal"))
        with self.assertRaises(urllib.error.HTTPError) as failed:
            self.chat("the failed owner cannot emit later tokens", max_tokens=16)
        with failed.exception as response:
            self.assertEqual(response.code, 503)
            request_id = response.headers["X-Strata-Request-Id"]
        self.assertIsNotNone(request_id)
        self.assertFalse(self.engine.alive())
        self.assertFalse(any(self.engine.slot_busy))
        self.assertFalse(self.engine.ctl.locked())
        record = self.svc.lifecycle_records(request_id)
        self.assertEqual(record.get("output_tokens", 0), 0)


class NativeAdmissionHarness:
    """HTTP clients plus raw wire receipts; gates never stand in for frontend admission decisions."""
    start = ParallelService.start
    get = ParallelService.get

    def start_native(self, slots=2, *, fit=None, depth=8, admission_held=False, more=(), lazy=False):
        self.gate_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.gate_tmp.cleanup)
        self.gates = Path(self.gate_tmp.name)
        self.clients = []
        self.gate("service")
        if admission_held:
            self.gate("admission")
        self.start(slots, fit=fit, lazy=lazy, more=("--retained", "--native-admission",
                                                   "--native-gates", str(self.gates), *more))
        self.svc.configure_admission({"request_queue_limit": depth, "lifecycle_api": True})

    def tearDown(self):
        try:
            if hasattr(self, "gates"):
                for gate in self.gates.iterdir():
                    gate.unlink(missing_ok=True)
            for client in getattr(self, "clients", []):
                client["thread"].join(5)
                self.assertFalse(client["thread"].is_alive(), client["text"])
                self.assertNotIn("exception", client, repr(client.get("exception")))
        finally:
            ParallelService.tearDown(self)

    def gate(self, name, tag=None, *, held=True):
        path = self.gates / (f"{name}-{tag}" if tag is not None else name)
        if held:
            path.touch()
        else:
            path.unlink(missing_ok=True)

    @contextlib.contextmanager
    def tokenization_gate(self, marker):
        entered, release = threading.Event(), threading.Event()
        encode, encode_within = self.svc.tok.encode, self.svc.tok.encode_within
        prompt_encoder = self.svc.prompts
        prompt_encode = prompt_encoder.encode if prompt_encoder is not None else None
        def pause(text):
            if marker in text:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("tokenizer gate was not released: " + marker)
        def delayed_encode(text, *args, **kwargs):
            pause(text)
            return encode(text, *args, **kwargs)
        def delayed_within(text, *args, **kwargs):
            pause(text)
            return encode_within(text, *args, **kwargs)
        def delayed_prompt(text, *args, **kwargs):
            pause(text)
            return prompt_encode(text, *args, **kwargs)
        try:
            # Demand tokenization normally uses the incremental encoder, which
            # captured the tokenizer method when Service was constructed.
            with mock.patch.object(self.svc.tok, "encode", side_effect=delayed_encode), \
                    mock.patch.object(self.svc.tok, "encode_within", side_effect=delayed_within), \
                    (mock.patch.object(prompt_encoder, "encode", side_effect=delayed_prompt)
                     if prompt_encoder is not None else contextlib.nullcontext()):
                yield entered, release
        finally:
            release.set()

    def wait_until(self, predicate, seconds=5):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            result = predicate()
            if result:
                return result
            time.sleep(.005)
        raw = self.log.read_text() if self.log.exists() else ""
        self.fail("native admission did not reach its finite gate:\n" + raw)

    def wire(self, event, *, tag=None, direction="received"):
        raw = self.log.read_text() if self.log.exists() else ""
        lines = raw.splitlines()
        if raw and not raw.endswith("\n"):
            lines = lines[:-1]  # A concurrent append has not published its complete receipt yet.
        receipts = [json.loads(line[7:]) for line in lines if line.startswith("NATIVE ")]
        return [receipt for receipt in receipts if receipt["direction"] == direction and
                receipt["raw"].split()[0] == event and
                (tag is None or receipt["raw"].split()[1] == tag)]

    def wait_wire(self, event, *, count=1, tag=None, direction="received"):
        self.wait_until(lambda: len(self.wire(event, tag=tag, direction=direction)) >= count)
        return self.wire(event, tag=tag, direction=direction)

    def registered_tag(self, number):
        return self.wait_wire("BQREGISTER", count=number, direction="sent")[number - 1]["raw"].split()[1]

    def launch(self, text, *, stream=False, path="/v1/chat/completions", priority="foreground",
               packing="eligible", timeout=None, max_tokens=1, body_updates=None):
        before = {record["id"] for record in self.get("/api/lifecycle")}
        body = {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", "stream": stream,
                "strata_priority": priority, "strata_packing": packing}
        if path == "/v1/responses":
            body["input"] = text
            body["max_output_tokens"] = body.pop("max_tokens")
            del body["messages"]
        if timeout is not None:
            body["strata_timeout_s"] = timeout
        body.update(body_updates or {})
        client = {"text": text, "before": before, "headers_ready": threading.Event(), "done": threading.Event()}
        def request():
            try:
                req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
                try:
                    response = urllib.request.urlopen(req, timeout=10)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    client.update(status=response.status, headers=dict(response.headers),
                                  id=response.headers.get("X-Strata-Request-Id"))
                    client["headers_ready"].set()
                    raw = response.read().decode()
                    client["body"] = raw if "event-stream" in response.headers.get("Content-Type", "") else json.loads(raw)
            except Exception as error:
                client["exception"] = error
            finally:
                client["done"].set()
        client["thread"] = threading.Thread(target=request, daemon=True)
        self.clients.append(client)
        client["thread"].start()
        return client

    def client_identity(self, client):
        if client.get("id"):
            return client["id"]
        def find():
            added = [record["id"] for record in self.get("/api/lifecycle")
                     if record["id"] not in client["before"]]
            return added[0] if len(added) == 1 else None
        client["id"] = self.wait_until(find)
        return client["id"]

    def client_record(self, client):
        return self.get("/api/lifecycle?id=" + self.client_identity(client))

    def cancel_client(self, client):
        identity = self.client_identity(client)
        req = urllib.request.Request(self.base + "/api/lifecycle/cancel",
                                     json.dumps({"id": identity}).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 202)
            self.assertEqual(json.load(response), {"id": identity, "cancel_requested": True})

    def finish_client(self, client, status=200):
        self.assertTrue(client["done"].wait(5), client["text"])
        client["thread"].join(1)
        self.assertNotIn("exception", client, repr(client.get("exception")))
        if status is not None:
            self.assertEqual(client["status"], status, client.get("body"))
        return client

    def assert_queued_terminal(self, tag, reason=None):
        terminals = self.wait_wire("BQDONE", tag=tag)
        self.assertEqual(len(terminals), 1)
        fields = terminals[0]["raw"].split()
        self.assertEqual(fields[2:6], ["-1", "0", "0", "0"])
        self.assertEqual(fields[7:9], ["0", "0"])
        self.assertEqual(fields[10:14], ["0", "0", "0", "prompt_read=0"])
        if reason is not None:
            self.assertEqual(fields[9], reason)
        self.assertGreaterEqual(float(fields[14].split("=")[1]), 0)
        self.assertEqual(fields[15], "native_service_wait_s=0.000000000")
        for event in ("BACK", "BQT", "BQPP", "BQSUSPEND", "BQRESUMED"):
            self.assertEqual(self.wire(event, tag=tag), [], event)
        return terminals[0]


class NativeAdmissionTransport(NativeAdmissionHarness, unittest.TestCase):
    def test_same_unseated_input_takes_only_the_cancelled_seat_while_peer_remains(self):
        self.start_native(depth=1)
        first = self.launch("A holds a seat")
        tag_a = self.registered_tag(1)
        self.client_identity(first)
        seat_a = self.wait_wire("BACK", tag=tag_a)[0]["raw"].split()
        peer = self.launch("B must stay resident")
        tag_b = self.registered_tag(2)
        self.client_identity(peer)
        self.wait_wire("BACK", tag=tag_b)
        waiting = self.launch("C is submitted exactly once")
        tag_c = self.registered_tag(3)
        self.client_identity(waiting)
        self.wait_wire("BQACCEPT", tag=tag_c)
        self.wait_wire("BGENQ1", tag=tag_c, direction="sent")
        self.assertEqual(self.wire("BACK", tag=tag_c), [])
        self.assertFalse(waiting["headers_ready"].is_set())
        self.cancel_client(first)
        seat_c = self.wait_wire("BACK", tag=tag_c)[0]["raw"].split()
        self.assertEqual(seat_c[2], seat_a[2])
        self.assertNotEqual(seat_c[3:5], seat_a[3:5])
        self.assertEqual(len(self.wire("BGENQ1", tag=tag_c, direction="sent")), 1)
        self.assertEqual(self.wire("BQDONE", tag=tag_b), [])
        self.assertEqual(self.wire("BQT", tag=tag_b), [])
        self.gate("service", tag_b)
        self.gate("service", held=False)
        result = self.finish_client(waiting)
        self.assertEqual(result["body"]["usage"]["completion_tokens"], 1)
        self.assertFalse(peer["done"].is_set())
        self.assertEqual(self.wire("BQDONE", tag=tag_b), [])
        self.cancel_client(peer)
        self.finish_client(first, status=None)
        self.finish_client(peer, status=None)
        self.assertEqual(len(self.wire("BQDONE", tag=tag_a)), 1)
        self.assertEqual(len(self.wire("BQDONE", tag=tag_b)), 1)
        self.assertEqual(len(self.wire("BQDONE", tag=tag_c)), 1)

    def test_full_queue_cancellation_before_accept_receipt_has_no_seat_or_work(self):
        self.start_native(depth=1)
        residents = []
        for index in range(2):
            client = self.launch(f"resident {index}")
            tag = self.registered_tag(index + 1)
            self.client_identity(client)
            self.wait_wire("BACK", tag=tag)
            residents.append(client)
        self.gate("accept")
        queued = self.launch("cancel before BQACCEPT can be consumed")
        tag = self.registered_tag(3)
        self.client_identity(queued)
        self.wait_wire("BGENQ1", tag=tag, direction="sent")
        self.assertEqual(self.wire("BQACCEPT", tag=tag), [])
        overflow = self.launch("full even with an unacknowledged logical owner")
        self.finish_client(overflow, 429)
        self.assertIn("Retry-After", overflow["headers"])
        self.cancel_client(queued)
        self.finish_client(queued, 409)
        terminal = self.assert_queued_terminal(tag, "cancel")
        self.assertGreater(int(terminal["raw"].split()[6]), 0)
        self.assertEqual(len(self.wire("BQACCEPT", tag=tag)), 1)
        self.assertTrue(all(not client["done"].is_set() for client in residents))
        # Late logical controls cannot resurrect a terminal or mutate either resident.
        self.engine._send(f"BQCANCEL {tag} -1 0 0")
        self.engine._send(f"BQPRIORITY {tag} -1 0 0 background")
        self.wait_wire("BQPRIORITY", tag=tag, direction="ignored")
        self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)
        self.assertEqual(len(self.wire("BACK")), 2)
        self.gate("accept", held=False)
        successor = self.launch("freed unfinished allowance")
        successor_tag = self.registered_tag(4)
        self.client_identity(successor)
        self.wait_wire("BGENQ1", tag=successor_tag, direction="sent")
        self.assertEqual(self.wire("BACK", tag=successor_tag), [])
        self.gate("service", held=False)
        for client in (*residents, successor):
            self.assertEqual(self.finish_client(client)["body"]["usage"]["completion_tokens"], 1)
        self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)
        self.assertEqual(len(self.wire("BQDONE", tag=successor_tag)), 1)

    def native_queue_boundary(self, depth):
        self.start_native(4, fit=2, depth=depth, admission_held=True)
        accepted = []
        for index in range(2 + depth):
            client = self.launch(f"bounded owner {index}")
            tag = self.registered_tag(index + 1)
            self.client_identity(client)
            self.wait_wire("BQACCEPT", tag=tag)
            self.wait_wire("BGENQ1", tag=tag, direction="sent")
            accepted.append(client)
        self.assertEqual(self.wire("BACK"), [])
        self.assertEqual(self.engine.batch, 2)
        for receipt in self.wire("BQREGISTER", direction="sent"):
            self.assertEqual(int(receipt["raw"].split()[5]), 2 + depth)
        rejected = self.launch("one beyond the one total allowance")
        self.finish_client(rejected, 429)
        self.assertIn(int(rejected["headers"]["Retry-After"]), range(1, 31))
        self.assertEqual(len(self.wire("BQREGISTER", direction="sent")), 2 + depth)
        self.gate("admission", held=False)
        self.wait_wire("BACK", count=2)
        self.assertTrue(all(not client["done"].is_set() for client in accepted))
        self.gate("service", held=False)
        for client in accepted:
            self.assertEqual(self.finish_client(client)["body"]["usage"]["completion_tokens"], 1)
        self.assertEqual(len(self.wire("BGENQ1", direction="sent")), 2 + depth)
        terminals = self.wait_wire("BQDONE", count=2 + depth)
        self.assertEqual(len({receipt["raw"].split()[1] for receipt in terminals}), 2 + depth)

    def test_total_allowance_uses_effective_slots_plus_depth(self):
        self.native_queue_boundary(2)

    def test_zero_depth_still_accepts_effective_slots_before_native_seating(self):
        self.native_queue_boundary(0)

    def test_native_timeout_is_408_before_any_stream_headers_or_dialect_preamble(self):
        self.start_native()
        for index in range(2):
            client = self.launch(f"deadline blocker {index}")
            tag = self.registered_tag(index + 1)
            self.client_identity(client)
            self.wait_wire("BACK", tag=tag)
        routes = (("/v1/chat/completions", False), ("/v1/chat/completions", True),
                  ("/v1/messages", True), ("/v1/responses", True))
        for number, (path, stream) in enumerate(routes, 3):
            with self.subTest(path=path, stream=stream):
                queued = self.launch("deadline in native admission", path=path, stream=stream, timeout=.37)
                tag = self.registered_tag(number)
                self.client_identity(queued)
                self.wait_wire("BGENQ1", tag=tag, direction="sent")
                self.assertFalse(queued["headers_ready"].is_set())
                response = self.finish_client(queued, 408)
                self.assertNotIn("Retry-After", response["headers"])
                self.assertIn("error", response["body"])
                self.assertNotIn("event-stream", response["headers"]["Content-Type"])
                self.assert_queued_terminal(tag)
                registration = self.wire("BQREGISTER", tag=tag, direction="sent")
                self.assertEqual(len(registration), 1)
                self.assertIn(int(registration[0]["raw"].split()[4]), range(1, 370000001))

    def test_transport_refusal_does_not_block_logical_cancellation_or_successor(self):
        self.start_native(1, depth=2, admission_held=True, more=("--native-input-capacity", "1"))
        queued = self.launch("fills bounded native input transport")
        tag = self.registered_tag(1)
        self.client_identity(queued)
        self.wait_wire("BGENQ1", tag=tag, direction="sent")
        refused = self.launch("input transport is full")
        refused_tag = self.registered_tag(2)
        self.client_identity(refused)
        self.finish_client(refused, 429)
        self.assertEqual([receipt["raw"] for receipt in self.wire("BQREJECT", tag=refused_tag)],
                         [f"BQREJECT {refused_tag} -1 transport_full"])
        self.cancel_client(queued)
        self.finish_client(queued, 409)
        self.assert_queued_terminal(tag, "cancel")
        successor = self.launch("transport capacity was released")
        successor_tag = self.registered_tag(3)
        self.client_identity(successor)
        self.wait_wire("BGENQ1", tag=successor_tag, direction="sent")
        self.gate("admission", held=False)
        self.gate("service", held=False)
        self.assertEqual(self.finish_client(successor)["body"]["usage"]["completion_tokens"], 1)
        self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)
        self.assertEqual(self.wire("BQDONE", tag=refused_tag), [])

    def test_delayed_old_output_finalizer_and_stale_controls_cannot_clear_http_successor(self):
        self.start_native(1, depth=1)
        self.gate("service", held=False)
        old = self.engine.generate_batched([65, 66, 67], 1, {}, threading.Event())
        try:
            self.assertIsInstance(next(old), int)
            old_seat = self.wait_wire("BACK")[0]["raw"].split()
            old_tag = old_seat[1]
            self.wait_wire("BQDONE", tag=old_tag)
            # Leave the old generator suspended after its token, before its terminal/finally.
            self.gate("service")
            successor = self.launch("HTTP successor owns the autonomously reused seat")
            tag = self.registered_tag(2)
            self.client_identity(successor)
            seat = self.wait_wire("BACK", tag=tag)[0]["raw"].split()
            self.assertEqual(old_seat[2], seat[2])
            self.assertNotEqual(old_seat[3:5], seat[3:5])
            self.wait_until(lambda: any(event["event"] == "BACK"
                                       for event in self.client_record(successor)["retained_events"]))
            old.close()
            self.engine._send(f"BQCANCEL {old_tag} {' '.join(old_seat[2:5])}")
            self.engine._send(f"BQPRIORITY {old_tag} {' '.join(old_seat[2:5])} background")
            self.engine._send(f"BQRESUME {old_tag} {' '.join(old_seat[2:5])} 0 1 1 32")
            self.engine._send(f"BQCANCEL {tag} {seat[2]} {int(seat[3]) + 1} {seat[4]}")
            self.engine._send(f"BQCANCEL {tag} {int(seat[2]) + 1} 0 0")
            self.wait_wire("BQCANCEL", tag=tag, count=2, direction="ignored")
            self.assertEqual(len(self.wire("BQDONE", tag=old_tag)), 1)
            self.assertEqual(self.wire("BQDONE", tag=tag), [])
            self.assertTrue(self.get("/slots")[0]["is_processing"])
            self.assertFalse(successor["done"].is_set())
            waiting = self.launch("new waiter cannot steal successor's seat")
            waiting_tag = self.registered_tag(3)
            self.client_identity(waiting)
            self.wait_wire("BGENQ1", tag=waiting_tag, direction="sent")
            self.assertEqual(self.wire("BACK", tag=waiting_tag), [])
            self.gate("service", held=False)
            for client in (successor, waiting):
                self.assertEqual(self.finish_client(client)["body"]["usage"]["completion_tokens"], 1)
            self.assertEqual(len(self.wire("BQDONE", tag=tag)), 1)
            self.assertEqual(len(self.wire("BQDONE", tag=waiting_tag)), 1)
        finally:
            old.close()


class LifecycleAdapterMatrix:
    """The same client lifecycle across the native mode and its explicit legacy fallback; no GPU claims."""

    def start_adapter(self, *, held=False):
        if self.native_mode:
            self.start_native(1)
            if not held:
                self.gate("service", held=False)
        else:
            self.gate_tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self.gate_tmp.cleanup)
            self.gates = Path(self.gate_tmp.name)
            self.clients = []
            if held:
                self.gate("legacy-replies")
            self.start(2, more=("--resident-multiplex", "--says-multiplex", "--legacy-identity",
                                "--hold-replies", str(self.gates / "legacy-replies")))
            self.svc.configure_admission({"lifecycle_api": True})

    def release_adapter(self):
        self.gate("service" if self.native_mode else "legacy-replies", held=False)

    def wait_adapter_owner(self, client):
        identity = self.client_identity(client)
        if self.native_mode:
            self.wait_until(lambda: self.client_record(client).get("native_request") is not None)
        else:
            self.wait_until(lambda: any(live and live.get("request_id") == identity and
                                        self.engine._slot_identity[slot] is not None
                                        for slot, live in enumerate(self.engine.slot_live)))

    def final_record(self, client, outcome):
        record = self.wait_until(lambda: (row if (row := self.client_record(client)).get("state") == outcome and
                                         row.get("finished_monotonic_s") is not None else None))
        self.assertEqual(record["lifecycle_transport"], "native-retained-v1" if self.native_mode else "legacy-seat")
        terminals = [event for event in record["events"] if event["state"] in ATTRIBUTION_OUTCOMES]
        self.assertEqual(len(terminals), 1, record["events"])
        self.assertAlmostEqual(sum(record["frontend"]["seconds"].values()), record["completion_s"], places=5)
        receipts = collections.Counter(event["tag"] for event in record.get("retained_events", [])
                                       if event["direction"] == "received" and
                                       event["event"] in ("BQDONE", "BQREJECT"))
        self.assertTrue(all(count == 1 for count in receipts.values()), receipts)
        body = client.get("body")
        if isinstance(body, str):
            self.assertLessEqual(body.count("data: [DONE]"), 1, body)
            finishes = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
            self.assertLessEqual(sum(any(choice.get("finish_reason") is not None for choice in row.get("choices", []))
                                     for row in finishes), 1, body)
        return record

    def test_adapter_completion_and_phase_partition(self):
        self.start_adapter()
        client = self.launch("adapter completion", stream=True, max_tokens=4)
        response = self.finish_client(client)
        self.assertEqual(response["body"].count("data: [DONE]"), 1)
        record = self.final_record(client, "completed")
        self.assertEqual(record["native_submission_count"], 1)
        if self.native_mode:
            self.assertEqual(record["frontend"]["counts"]["control_selector_wait"], 0)
            self.assertEqual(record["frontend"]["counts"]["physical_slot_wait"], 0)

    def test_adapter_cancel_is_one_terminal_without_resubmission(self):
        self.start_adapter(held=True)
        client = self.launch("adapter cancel", stream=True, max_tokens=8)
        self.wait_adapter_owner(client)
        self.cancel_client(client)
        if not self.native_mode:
            self.wait_until(lambda: "BSTOP " in self.log.read_text())
        self.release_adapter()
        self.finish_client(client)
        record = self.final_record(client, "cancelled")
        self.assertEqual(record["native_submission_count"], 1)

    def test_adapter_disconnect_drains_without_rerun(self):
        import socket
        self.start_adapter(held=True)
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "adapter disconnect"}],
                           "reasoning_effort": "none", "stream": True, "max_tokens": 8}).encode()
        sock = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]), timeout=5)
        try:
            sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n" +
                         f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            self.wait_until(lambda: any(row.get("slot") is not None for row in self.svc.lifecycle_records()))
            identity = self.svc.lifecycle_records()[0]["id"]
            self.wait_adapter_owner({"id": identity})
            sock.shutdown(socket.SHUT_RDWR)
        finally:
            sock.close()
        self.wait_until(lambda: self.svc.lifecycle_records(identity).get("cancel_reason") == "disconnected")
        self.release_adapter()
        client = {"id": identity}
        record = self.final_record(client, "disconnected")
        self.assertEqual(record["native_submission_count"], 1)
        successor = self.launch("peer after disconnect")
        self.finish_client(successor)
        self.final_record(successor, "completed")

    def test_adapter_refusal_is_request_local(self):
        self.start_adapter()
        client = self.launch("ADMISSIONREFUSE", stream=True)
        self.finish_client(client, 429 if self.native_mode else 200)
        record = self.final_record(client, "rejected" if self.native_mode else "error")
        self.assertEqual(record.get("output_tokens", 0), 0)
        self.assertTrue(self.engine.alive())
        successor = self.launch("peer after refusal")
        self.finish_client(successor)
        self.final_record(successor, "completed")

    def test_adapter_recoverable_engine_error_keeps_successor_usable(self):
        self.start_adapter()
        client = self.launch("REFUSEME", stream=True)
        self.finish_client(client, None)  # Native reports its drained nonfinite terminal after the SSE preamble.
        record = self.final_record(client, "error")
        self.assertEqual(record.get("output_tokens", 0), 0)
        if self.native_mode:
            self.assertEqual(record["native_registration_count"], 2)
        self.assertTrue(self.engine.alive())
        successor = self.launch("peer after recoverable error")
        self.finish_client(successor)
        self.final_record(successor, "completed")

    def test_adapter_engine_exit_never_replays_accepted_input(self):
        self.start_adapter()
        client = self.launch("ADAPTEREXIT", stream=True)
        self.finish_client(client, None)
        record = self.final_record(client, "error")
        self.assertEqual(record["native_submission_count"], 1)
        self.assertFalse(self.engine.alive())
        self.assertEqual(self.engine.exit_code(), 7)
        if self.native_mode:
            self.assertEqual(len(self.wire("BQREGISTER", direction="sent")), 1)
        else:
            self.assertEqual(sum(line.startswith(("GEN ", "BGEN ")) for line in self.log.read_text().splitlines()), 1)


class LegacyLifecycleAdapterMatrix(LifecycleAdapterMatrix, NativeAdmissionHarness, unittest.TestCase):
    native_mode = False


class NativeLifecycleAdapterMatrix(LifecycleAdapterMatrix, NativeAdmissionHarness, unittest.TestCase):
    native_mode = True

    def test_priority_and_cancel_use_tag_and_incarnation_not_reused_seat(self):
        self.start_native(1, admission_held=True)
        client = self.launch("fenced native controls", stream=True)
        tag = self.registered_tag(1)
        self.wait_wire("BGENQ1", tag=tag, direction="sent")
        record = self.svc.admission_active[self.client_identity(client)]
        with self.engine.slot_cv:
            queued = LifecycleIdentity.of(record["_native_owner"])
        adapter = self.svc.lifecycle_adapter
        self.assertTrue(adapter.priority(record, "background", queued))
        self.wait_wire("BQPRIORITY", tag=tag, direction="sent")
        self.gate("admission", held=False)
        self.wait_wire("BACK", tag=tag)
        with self.engine.slot_cv:
            seated = LifecycleIdentity.of(record["_native_owner"])
        self.assertNotEqual(queued, seated)
        self.assertFalse(adapter.priority(record, "foreground", queued))
        wrong = LifecycleIdentity(tag, seated.slot, seated.request + 1, seated.incarnation)
        self.assertFalse(adapter.cancel(record, wrong))
        self.assertFalse(record["_cancel"].is_set())
        self.assertTrue(adapter.priority(record, "foreground", seated))
        self.cancel_client(client)
        self.finish_client(client)
        self.final_record(client, "cancelled")
        successor = self.launch("same seat with a different owner")
        new_tag = self.registered_tag(2)
        self.wait_wire("BACK", tag=new_tag)
        new_record = self.svc.admission_active[self.client_identity(successor)]
        self.assertFalse(adapter.priority(new_record, "background", seated))
        self.assertFalse(adapter.cancel(new_record, seated))
        self.assertFalse(new_record["_cancel"].is_set())
        self.assertEqual(len(self.wire("BQPRIORITY", tag=tag, direction="sent")), 2)
        self.gate("service", held=False)
        self.finish_client(successor)
        self.final_record(successor, "completed")


class ScoreTransport(unittest.TestCase):
    """#233: the public bounded adapter over the existing scripted engine pipe."""
    start = ParallelService.start
    tearDown = ParallelService.tearDown
    chat = ParallelService.chat
    get = ParallelService.get

    def score(self, prompt=None, selected=None, **extra):
        body = {"version": 1, "prompt_token_ids": prompt if prompt is not None else [5, 6],
                "selected_token_ids": selected if selected is not None else [3, 1, 0, 3]}
        return ScoreTransport.post_score(self, body | extra)

    def post_score(self, body, path="/v1/score", *, headers=None):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"} | (headers or {}))
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def group_body(self, count=3, prompt=None, group_id=None, branches=1):
        from serve import scoring
        children = [{"id": f"child_{i}" if branches == 1 else f"child_{i}_{branch}",
                     "question_index": i, "branch_index": branch,
                     "plan_sha256": scoring.fingerprint(["plan", i, branch]),
                     "prompt_token_ids": prompt or [5, 6], "selected_token_ids": [3, 1]}
                    for i in range(count) for branch in range(branches)]
        group = {"id": group_id or uuid.uuid4().hex, "scoring_identity_sha256": "a" * 64,
                 "bundle_sha256": "b" * 64, "request_sha256": "c" * 64,
                 "token_plan_sha256": scoring.fingerprint([[c["plan_sha256"] for c in children if c["question_index"] == i]
                                                         for i in range(count)]),
                 "readout_family": "direct_letter_logits" if branches == 1 else "per_option_yes_no_log_odds",
                 "temperature": 2.0, "question_count": count, "deadline_unix_s": time.time() + 10}
        group["manifest_sha256"] = scoring.fingerprint(group | {"children": children})
        return {"version": 1, "group": group, "children": children}

    def test_group_is_complete_ordered_and_pins_scoring_identity(self):
        self.start(2)
        body = self.group_body()
        result = self.post_score(body)
        self.assertEqual(result["group"], body["group"])
        self.assertEqual([c["id"] for c in result["children"]], ["child_0", "child_1", "child_2"])
        self.assertEqual(result["outcome"]["kind"], "completed")
        self.assertEqual(result["outcome"]["completed_children"], 3)
        self.assertTrue(result["outcome"]["drained"])
        ids = [c["result"]["request_id"] for c in result["children"]]
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(i.startswith(result["request_id"] + "_") for i in ids))
        self.assertFalse(self.engine.score_q)
        record = self.svc.lifecycle_records(result["request_id"])
        self.assertEqual(record["lifecycle_transport"], "score-control")
        self.assertEqual(record["score_outcome"]["kind"], "completed")
        with self.assertRaises(urllib.error.HTTPError) as duplicate:
            self.post_score(body)
        self.assertEqual(duplicate.exception.code, 409)
        self.assertEqual(self.score()["generated_tokens"], 0)

    def test_late_bad_group_child_and_identity_skew_issue_no_native_work(self):
        self.start(2)
        for change in (lambda b: b["children"][-1].update(prompt_token_ids=[]),
                       lambda b: b["children"][-1].update(id="child_0"),
                       lambda b: b["children"][-1].update(branch_index=1),
                       lambda b: b["group"].update(token_plan_sha256="d" * 64),
                       lambda b: b["group"].update(question_count=65)):
            body = self.group_body()
            change(body)
            with self.subTest(change=change), self.assertRaises(urllib.error.HTTPError) as failed:
                self.post_score(body)
            self.assertEqual(failed.exception.code, 400)
        self.assertEqual(self.log.read_text(), "")
        self.assertEqual(self.score()["generated_tokens"], 0)

    def test_native_refusal_in_later_child_returns_one_structured_failure_without_partial_results(self):
        from serve import scoring
        self.start(2)
        body = self.group_body()
        body["children"][1]["selected_token_ids"] = [999]
        group = body["group"]
        group["manifest_sha256"] = scoring.fingerprint(
            {k: v for k, v in group.items() if k != "manifest_sha256"} | {"children": body["children"]})
        with self.assertRaises(urllib.error.HTTPError) as failed:
            self.post_score(body)
        with failed.exception as response:
            self.assertEqual(response.code, 400)
            error = json.load(response)
        self.assertEqual(error["group"], group)
        self.assertEqual(error["outcome"]["kind"], "refused")
        self.assertEqual(error["outcome"]["completed_children"], 1)
        self.assertEqual(error["outcome"]["failed_child"], "child_1")
        self.assertTrue(error["outcome"]["drained"])
        self.assertNotIn("children", error)
        self.assertNotIn("logits", error)
        self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 2)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.score()["generated_tokens"], 0)
        self.assertEqual(self.chat("independent refusal peer")["choices"][0]["message"]["content"], "ok, done.")

    def test_group_deadline_between_branches_does_not_issue_a_dependent_child(self):
        self.start(2)
        original = self.engine.score
        def slow_after_drain(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(.08)
            return result
        with mock.patch.object(self.engine, "score", side_effect=slow_after_drain):
            with self.assertRaises(urllib.error.HTTPError) as failed:
                self.post_score(self.group_body() | {"strata_timeout_s": .05})
        with failed.exception as response:
            self.assertEqual(response.code, 408)
            error = json.load(response)
        self.assertEqual(error["outcome"]["kind"], "deadline")
        self.assertEqual(error["outcome"]["completed_children"], 1)
        self.assertEqual(error["outcome"]["failed_child"], "child_1")
        self.assertTrue(error["outcome"]["drained"])
        self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 1)
        self.assertEqual(self.score()["generated_tokens"], 0)

    def test_queued_and_active_group_timeouts_are_408_with_distinct_bounded_cleanup(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hold, drain, cancelled = (Path(directory.name) / name for name in ("hold", "drain", "cancelled"))
        hold.touch()
        drain.touch()
        self.start(2, more=("--score-hold-file", str(hold), "--score-drain-gate", str(drain),
                            "--score-cancel-file", str(cancelled)))
        failures = []
        def active():
            try:
                self.post_score(self.group_body() | {"strata_timeout_s": .15})
            except urllib.error.HTTPError as error:
                with error:
                    failures.append((error.code, json.load(error)))
        thread = threading.Thread(target=active)
        thread.start()
        try:
            self.wait_for(cancelled.exists, "active deadline sends SCANCEL")
            self.assertTrue(self.engine.score_q)
            self.assertTrue(self.engine.ctl.locked())
            with self.assertRaises(urllib.error.HTTPError) as queued:
                self.post_score(self.group_body() | {"strata_timeout_s": .05})
            with queued.exception as response:
                self.assertEqual(response.code, 408)
                body = json.load(response)
            self.assertEqual(body["outcome"]["kind"], "deadline")
            self.assertEqual(body["outcome"]["completed_children"], 0)
            self.assertTrue(body["outcome"]["drained"])
            self.assertTrue(thread.is_alive(), "timeout notification cannot release issued state ahead of SDONE")
            self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 1)
        finally:
            drain.unlink(missing_ok=True)
            hold.unlink(missing_ok=True)
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures[0][0], 408)
        outcome = failures[0][1]["outcome"]
        self.assertEqual(outcome["kind"], "deadline")
        self.assertTrue(outcome["drained"])
        self.assertGreater(outcome["cleanup_s"], .05)
        self.assertFalse(self.engine.score_q)
        self.assertEqual(self.score()["generated_tokens"], 0)

    def test_pre_registration_cancel_is_bounded_and_overload_does_not_evict_a_live_abort(self):
        from serve.server import AdmissionError
        self.start(2)
        body = self.group_body(group_id="cancel_first")
        request = {"version": 1, "group_id": body["group"]["id"],
                   "deadline_unix_s": body["group"]["deadline_unix_s"], "pre_registration": True}
        self.assertTrue(self.post_score(request, "/v1/score/cancel")["outcome"]["drained"])
        for i in range(31):
            self.svc.cancel_score_group(request | {"group_id": f"cancel_{i}"})
        with self.assertRaises(AdmissionError) as full:
            self.svc.cancel_score_group(request | {"group_id": "one_too_many"})
        self.assertEqual(full.exception.status, 429)
        with self.assertRaises(urllib.error.HTTPError) as cancelled:
            self.post_score(body)
        self.assertEqual(cancelled.exception.code, 409)
        self.assertEqual(self.log.read_text(), "")
        self.assertEqual(len(self.post_score(self.group_body())["children"]), 3)

    def test_deadline_and_priority_validation_precedes_both_http_and_direct_score_issue(self):
        self.start(2)
        for controls in ({"strata_timeout_s": 0}, {"strata_timeout_s": True}, {"strata_timeout_s": float("nan")},
                         {"strata_timeout_s": float("inf")}, {"strata_timeout_s": 301},
                         {"strata_priority": "urgent"}, {"strata_priority": None}):
            with self.subTest(controls=controls):
                with self.assertRaises(urllib.error.HTTPError) as failed:
                    self.score(**controls)
                self.assertEqual(failed.exception.code, 400)
                with self.assertRaises(ValueError):
                    self.engine.score({"version": 1, "prompt_token_ids": [5], "selected_token_ids": [0]} | controls,
                                      threading.Event())
        self.assertEqual(self.log.read_text(), "")
        self.assertEqual(self.score()["generated_tokens"], 0)

    def launch_group(self, body, headers=None):
        results, errors = [], []
        def run():
            try:
                results.append(self.post_score(body, headers=headers))
            except urllib.error.HTTPError as error:
                with error:
                    errors.append((error.code, json.load(error)))
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, results, errors

    def join_group(self, call):
        call[0].join(8)
        self.assertFalse(call[0].is_alive(), "group must reach bounded drained retirement")

    def test_authenticated_scope_binds_cancel_precancel_and_history_not_untrusted_tenant_headers(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hold, cancelled = (Path(directory.name) / name for name in ("hold", "cancelled"))
        hold.touch()
        self.start(2, more=("--score-hold-file", str(hold), "--score-cancel-file", str(cancelled)))
        self.svc.api_key = ["gateway-1", "gateway-2", "direct"]
        self.svc.tenant_token = ["gateway-1", "gateway-2"]
        self.svc.lifecycle_api = True
        alpha = {"Authorization": "Bearer gateway-1", "X-Strata-Tenant": "alpha", "X-Strata-Operation": "search/one"}
        beta = {"Authorization": "Bearer gateway-1", "X-Strata-Tenant": "beta"}
        future = self.group_body(group_id="same_local_id")
        future_cancel = {"version": 1, "group_id": future["group"]["id"],
                         "deadline_unix_s": future["group"]["deadline_unix_s"], "pre_registration": True}
        # A pre-cancel belongs only to beta's authenticated namespace; it cannot reserve alpha's ID.
        self.assertTrue(self.post_score(future_cancel, "/v1/score/cancel", headers=beta)["outcome"]["drained"])
        body = self.group_body(group_id="alpha_active")
        cancel = {"version": 1, "group_id": body["group"]["id"], "deadline_unix_s": body["group"]["deadline_unix_s"]}
        first, successor = self.launch_group(body, alpha), None
        try:
            self.wait_for(lambda: self.log.exists() and "SCORE " in self.log.read_text(), "alpha submitted")
            request_id = self.log.read_text().split()[1].rsplit("_", 1)[0]
            for foreign in (beta, {"Authorization": "Bearer direct", "X-Strata-Tenant": "alpha"},
                            {"Authorization": "Bearer gateway-2", "X-Strata-Tenant": "alpha"}):
                with self.subTest(foreign=foreign), self.assertRaises(urllib.error.HTTPError) as denied:
                    self.post_score(cancel, "/v1/score/cancel", headers=foreign)
                self.assertEqual(denied.exception.code, 403)
                with self.assertRaises(urllib.error.HTTPError) as private_denied:
                    self.post_score({"id": request_id}, "/api/lifecycle/cancel", headers=foreign)
                self.assertEqual(private_denied.exception.code, 403)
            self.assertFalse(cancelled.exists(), "another tenant or rotated credential cannot send alpha's SCANCEL")
            successor = self.launch_group(future, alpha)
            self.wait_for(lambda: bool(self.svc.score_ready), "alpha's successor is not beta's pre-cancel")
            with self.assertRaises(urllib.error.HTTPError) as stale:
                self.post_score(cancel | {"deadline_unix_s": cancel["deadline_unix_s"] - 1},
                                "/v1/score/cancel", headers=alpha)
            self.assertEqual(stale.exception.code, 409)
            self.assertFalse(cancelled.exists())
            outcome = self.post_score(cancel, "/v1/score/cancel", headers=alpha)["outcome"]
            self.assertTrue(outcome["drained"])
        finally:
            hold.unlink(missing_ok=True)
            self.join_group(first)
            if successor:
                self.join_group(successor)
        self.assertEqual(first[1], [])
        self.assertEqual(first[2][0][0], 409)
        self.assertEqual(successor[2], [])
        self.assertEqual(successor[1][0]["correlation"]["operation_id"], "search/one")
        self.assertEqual(successor[1][0]["correlation"]["operation_source"], "gateway_attested")
        with self.assertRaises(urllib.error.HTTPError) as history:
            self.post_score(cancel, "/v1/score/cancel", headers=beta)
        self.assertEqual(history.exception.code, 403)
        self.assertTrue(self.post_score(cancel, "/v1/score/cancel", headers=alpha)["outcome"]["drained"])
        self.assertEqual(len(self.post_score(self.group_body(), headers=beta)["children"]), 3)

    def test_512_child_group_yields_to_urgent_peer_after_one_drained_child_and_has_one_terminal_per_issue(self):
        self._assert_large_group_fairness(2)

    def test_serial_512_child_group_yields_to_an_urgent_peer_after_one_drained_child(self):
        self._assert_large_group_fairness(0)

    def test_cold_serial_peer_reaches_child_selection_after_loading_instead_of_staying_on_startup_fifo(self):
        self._assert_large_group_fairness(0, cold=True)

    def test_cold_parallel_peer_reaches_child_selection_after_loading_instead_of_staying_on_startup_fifo(self):
        self._assert_large_group_fairness(2, cold=True)

    def test_parallel_arrival_during_visible_restart_waits_before_capturing_native_generation(self):
        self.start(2)
        entered, snapshots = threading.Event(), []
        native_score = self.engine.score
        def score(*args, **kwargs):
            snapshots.append((self.engine.gen, self.engine.starting))
            entered.set()
            return native_score(*args, **kwargs)
        self.svc.fifo.acquire()
        with self.engine.slot_cv:
            self.engine.starting = True  # alive/READY is already visible, but restart has not finished.
        with mock.patch.object(self.engine, "score", side_effect=score):
            call = self.launch_group(self.group_body(1))
            try:
                self.wait_for(lambda: self.svc.score_child_owner is not None, "peer owns a selected host child")
                early = entered.wait(.05)
                self.assertEqual(self.log.read_text(), "", "startup must not issue a native child")
            finally:
                with self.engine.slot_cv:
                    self.engine.starting = False
                    self.engine.slot_cv.notify_all()
                self.svc.fifo.release()
                self.join_group(call)
        self.assertFalse(early, "native SCORE must not capture generation while initialization owns the FIFO")
        self.assertEqual(snapshots, [(self.engine.gen, False)])
        self.assertEqual(call[2], [])
        self.assertEqual(call[1][0]["outcome"]["kind"], "completed")

    def _assert_large_group_fairness(self, slots, cold=False):
        import serve.server as server
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hold, cancelled = (Path(directory.name) / name for name in ("hold", "cancelled"))
        hold.touch()
        self.start(slots, more=("--score-hold-file", str(hold), "--score-cancel-file", str(cancelled)))
        self.svc.api_key = self.svc.tenant_token = ["gateway"]
        alpha = {"Authorization": "Bearer gateway", "X-Strata-Tenant": "alpha"}
        beta = {"Authorization": "Bearer gateway", "X-Strata-Tenant": "beta"}
        body = self.group_body(64, branches=8) | {"strata_priority": "background"}
        urgent_body = self.group_body(1) | {"strata_priority": "foreground"}
        loading, loaded, release_loading, startup_wait = (threading.Event() for _ in range(4))
        service, real_fifo = self.svc, self.svc.fifo
        urgent_id = urgent_body["group"]["id"]
        def load():
            loading.set()
            if not release_loading.wait(5):
                raise AssertionError("test did not release the no-load startup gate")
            loaded.set()
        def is_loaded():
            record = getattr(service.lifecycle_local, "record", None)
            if record is not None and record.get("_score_group", {}).get("id") == urgent_id:
                # Expose readiness after the loader's first child owns the native turn.
                return self.log.exists() and "SCORE " in self.log.read_text()
            return loaded.is_set()
        class NonFairStartupFIFO:
            def acquire(_self, *args, **kwargs):
                record = getattr(service.lifecycle_local, "record", None)
                if record is not None and record.get("_score_group", {}).get("id") == urgent_id:
                    selected = service.score_child_owner
                    if (not selected or selected["record"] is not record) and not any(
                            entry["record"] is record for entry in service.score_ready):
                        # A lock does not promise fairness. Cold waiters must become
                        # visible to the child selector, not win a FIFO reacquire race.
                        startup_wait.set()
                        time.sleep(.01)
                        return False
                return real_fifo.acquire(*args, **kwargs)
            def release(_self):
                return real_fifo.release()
        trace = io.StringIO()
        large = urgent = None
        with contextlib.ExitStack() as patches, mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            if cold:
                patches.enter_context(mock.patch.object(service, "loaded", side_effect=is_loaded))
                patches.enter_context(mock.patch.object(service, "ensure_loaded", side_effect=load))
                patches.enter_context(mock.patch.object(service, "fifo", NonFairStartupFIFO()))
            try:
                large = self.launch_group(body, alpha)
                if cold:
                    self.wait_for(loading.is_set, "large owner is loading")
                    urgent = self.launch_group(urgent_body, beta)
                    self.wait_for(startup_wait.is_set, "urgent peer entered the cold startup FIFO")
                    release_loading.set()
                self.wait_for(lambda: self.log.exists() and "SCORE " in self.log.read_text(), "large first child issued")
                if not cold:
                    urgent = self.launch_group(urgent_body, beta)
                self.wait_for(lambda: bool(self.svc.score_ready), "urgent peer is ready, not behind the complete group")
                self.assertEqual(len(self.svc.score_groups), 2)
                hold.unlink()
                self.join_group(urgent)
                self.assertEqual(urgent[2], [])
                request_id = urgent[1][0]["request_id"]
                issues = [line.split()[1] for line in self.log.read_text().splitlines() if line.startswith("SCORE ")]
                self.assertEqual(issues[1], request_id + "_0")
                self.assertTrue(large[0].is_alive(), "urgent completion cannot require finishing 512 siblings")
                hold.touch()
                before = len(issues)
                self.wait_for(lambda: len(self.log.read_text().splitlines()) > before, "large successor child owns native state")
                outcome = self.post_score({"version": 1, "group_id": body["group"]["id"],
                    "deadline_unix_s": body["group"]["deadline_unix_s"]}, "/v1/score/cancel", headers=alpha)["outcome"]
                self.assertTrue(outcome["drained"])
            finally:
                release_loading.set()
                if cold:
                    with service.admission_cv:
                        for record in service.score_groups.values():
                            record["_cancel"].set()
                        service.admission_cv.notify_all()
                hold.unlink(missing_ok=True)
                if large:
                    self.join_group(large)
                if urgent:
                    self.join_group(urgent)
        self.assertEqual(large[1], [])
        self.assertEqual(large[2][0][0], 409)
        self.assertNotIn("children", large[2][0][1])
        self.assertLess(large[2][0][1]["outcome"]["completed_children"], 512)
        issues = [line.split()[1] for line in self.log.read_text().splitlines() if line.startswith("SCORE ")]
        terminals = [line.split(" wire< ", 1)[1].split()[1] for line in trace.getvalue().splitlines()
                     if " wire< SDONE " in line]
        self.assertEqual(sorted(terminals), sorted(issues))
        self.assertEqual(len(set(terminals)), len(terminals))
        self.assertFalse(self.svc.score_caller_work)
        self.assertFalse(self.svc.score_ready)
        self.assertTrue(self.engine.alive())

    def test_caller_round_robin_and_group_caps_leave_a_small_peer_a_turn(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hold = Path(directory.name) / "hold"
        hold.touch()
        self.start(2, more=("--score-hold-file", str(hold)))
        self.svc.api_key = self.svc.tenant_token = ["gateway"]
        self.engine.info["score_max_pending"] = 5
        alpha = {"Authorization": "Bearer gateway", "X-Strata-Tenant": "alpha"}
        beta = {"Authorization": "Bearer gateway", "X-Strata-Tenant": "beta"}
        calls, peer = [], None
        try:
            calls.append(self.launch_group(self.group_body(), alpha))
            self.wait_for(lambda: self.log.exists() and "SCORE " in self.log.read_text(), "first caller owns one child")
            calls.extend(self.launch_group(self.group_body(), alpha) for _ in range(3))
            self.wait_for(lambda: len(self.svc.score_groups) == 4, "four bounded groups belong to alpha")
            with self.assertRaises(urllib.error.HTTPError) as caller_full:
                self.post_score(self.group_body(1), headers=alpha)
            self.assertEqual(caller_full.exception.code, 429)
            peer = self.launch_group(self.group_body(1), beta)
            self.wait_for(lambda: len(self.svc.score_groups) == 5 and len(self.svc.score_ready) == 4, "peer keeps its own ready turn")
            with self.assertRaises(urllib.error.HTTPError) as total_full:
                self.post_score(self.group_body(1), headers=beta)
            self.assertEqual(total_full.exception.code, 429)
            hold.unlink()
            self.join_group(peer)
            self.assertEqual(peer[2], [])
            issues = [line.split()[1] for line in self.log.read_text().splitlines() if line.startswith("SCORE ")]
            self.assertEqual(issues[1], peer[1][0]["request_id"] + "_0", "caller FIFO, not one turn for every group from alpha")
        finally:
            hold.unlink(missing_ok=True)
            for call in calls:
                self.join_group(call)
            if peer:
                self.join_group(peer)
        self.assertTrue(all(not call[2] for call in calls))
        self.assertFalse(self.svc.score_caller_work)
        self.assertEqual(len(self.post_score(self.group_body(1), headers=beta)["children"]), 1)

    def test_declared_caller_and_global_work_bounds_refuse_before_native_issue(self):
        from serve.server import AdmissionError
        self.start(2)
        alpha = self.svc.score_scope("alpha", gateway=True, credential="gateway")
        beta = self.svc.score_scope("beta", gateway=True, credential="gateway")
        gamma = self.svc.score_scope("gamma", gateway=True, credential="gateway")
        records = []
        try:
            for scope in (alpha, alpha, beta, beta):
                body = self.group_body(64, branches=8, prompt=[5] * 512)
                records.append(self.svc.register_lifecycle(body, threading.Event(), score=True, score_scope=scope))
            with self.svc.status_lock:
                pending = self.svc._admission_snapshot()["score_pending_work"]
            self.assertEqual((pending["groups"], pending["children"], pending["prompt_tokens"]), (4, 2048, 1048576))
            for scope in (alpha, gamma):
                with self.assertRaises(AdmissionError) as full:
                    self.svc.register_lifecycle(self.group_body(1), threading.Event(), score=True, score_scope=scope)
                self.assertEqual(full.exception.status, 429)
                self.assertIn("caller/work", str(full.exception))
                self.svc.finish_lifecycle(full.exception.lifecycle)
            self.assertEqual(self.log.read_text(), "", "declared aggregate work cannot reach native dispatch")
        finally:
            for record in records:
                record["cancel_reason"] = "cancelled"
                record["_cancel"].set()
                self.svc.finish_lifecycle(record)
        self.assertFalse(self.svc.score_caller_work)
        self.assertEqual(len(self.post_score(self.group_body(1))["children"]), 1)

    def test_group_input_has_a_fixed_body_bound_even_if_generic_http_budget_is_larger(self):
        self.start(2)
        self.svc.request_body_bytes = 8 << 20
        from http.client import HTTPConnection
        connection = HTTPConnection(*self.httpd.server_address, timeout=3)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/score",
                           headers={"Content-Type": "application/json", "Content-Length": str((4 << 20) + 1)})
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        self.assertEqual(self.log.read_text(), "")
        self.assertEqual(self.score()["generated_tokens"], 0)

    def test_scoped_pre_registration_cancel_can_share_a_foreign_live_group_id(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hold, cancelled = (Path(directory.name) / name for name in ("hold", "cancelled"))
        hold.touch()
        self.start(2, more=("--score-hold-file", str(hold), "--score-cancel-file", str(cancelled)))
        self.svc.api_key = self.svc.tenant_token = ["gateway"]
        alpha = {"Authorization": "Bearer gateway", "X-Strata-Tenant": "alpha"}
        beta = {"Authorization": "Bearer gateway", "X-Strata-Tenant": "beta"}
        body = self.group_body(group_id="same_id_different_owners")
        cancel = {"version": 1, "group_id": body["group"]["id"], "deadline_unix_s": body["group"]["deadline_unix_s"]}
        foreign = self.launch_group(body, beta)
        try:
            self.wait_for(lambda: self.log.exists() and "SCORE " in self.log.read_text(), "beta owns its ID")
            with self.assertRaises(urllib.error.HTTPError) as denied:
                self.post_score(cancel, "/v1/score/cancel", headers=alpha)
            self.assertEqual(denied.exception.code, 403)
            self.assertTrue(self.post_score(cancel | {"pre_registration": True},
                "/v1/score/cancel", headers=alpha)["outcome"]["drained"])
            with self.assertRaises(urllib.error.HTTPError) as aborted:
                self.post_score(body, headers=alpha)
            self.assertEqual(aborted.exception.code, 409)
            self.assertFalse(cancelled.exists(), "alpha's pre-cancel cannot touch beta's identically named group")
            self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 1)
        finally:
            hold.unlink(missing_ok=True)
            self.join_group(foreign)
        self.assertEqual(foreign[2], [])
        self.assertEqual(len(foreign[1][0]["children"]), 3)
        self.assertEqual(len(self.post_score(self.group_body(1), headers=alpha)["children"]), 1)

    def test_retirement_capacity_protects_replays_and_rejected_record_cannot_retire_a_new_owner(self):
        import serve.server as server
        from serve.server import AdmissionError
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        hold = Path(directory.name) / "hold"
        self.start(2, more=("--score-hold-file", str(hold)))
        with mock.patch.object(server.score_api, "MAX_RETIRED_IDENTITIES", 1):
            original = self.group_body(1, group_id="retained_original")
            self.assertEqual(len(self.post_score(original)["children"]), 1)
            rejected_body = self.group_body(1, group_id="retry_after_retention_pressure")
            with self.assertRaises(AdmissionError) as full:
                self.svc.register_lifecycle(rejected_body, threading.Event(), score=True)
            self.assertEqual(full.exception.status, 429)
            self.assertIn("identity retention", str(full.exception))
            rejected = full.exception.lifecycle  # Deliberately delay this rejected handler's terminal cleanup.
            with self.assertRaises(urllib.error.HTTPError) as replay:
                self.post_score(original)
            self.assertEqual(replay.exception.code, 409)
            self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 1)
            future = time.time() + 60
            with mock.patch.object(server.time, "time", return_value=future):
                replacement_body = self.group_body(1, group_id=rejected_body["group"]["id"])
                cancellation = threading.Event()
                replacement = self.svc.register_lifecycle(replacement_body, cancellation, score=True)
                self.svc.acquire_request(replacement)
                hold.touch()
                errors = []
                def retire_old_while_new_child_is_issued():
                    try:
                        self.wait_for(lambda: len(self.log.read_text().splitlines()) == 2, "replacement submitted")
                        self.svc.finish_lifecycle(rejected)
                        key = replacement["_score_group_key"]
                        self.assertIs(self.svc.score_groups[key], replacement)
                        self.assertIs(self.svc.score_identity_owners[key], replacement)
                        self.assertNotIn(key, self.svc.score_group_history)
                        self.assertFalse(replacement["_score_finished"].is_set())
                        self.assertTrue(self.engine.score_q)
                    except Exception as error:
                        errors.append(error)
                    finally:
                        self.svc.cancel_request(replacement["id"])
                thread = threading.Thread(target=retire_old_while_new_child_is_issued)
                thread.start()
                try:
                    with self.assertRaises(AdmissionError) as stopped:
                        self.svc.score(replacement_body, cancellation)
                    self.assertEqual(stopped.exception.status, 409)
                finally:
                    hold.unlink(missing_ok=True)
                    thread.join(5)
                    self.svc.finish_lifecycle(replacement)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertFalse(self.engine.score_q)
                self.assertFalse(self.svc.score_groups)
                self.assertFalse(self.svc.score_identity_owners)
        self.assertEqual(self.score()["generated_tokens"], 0)

    def test_complete_score_has_ordered_raw_fp32_and_identity_receipt(self):
        self.start(2)
        proc = self.engine.proc
        result = self.score()
        self.assertEqual(result["logits"], [.3333333432674408, -17.5, .125, .3333333432674408])
        self.assertEqual(result["selected_token_ids"], [3, 1, 0, 3])
        self.assertEqual(result["generated_tokens"], 0)
        self.assertEqual(result["receipt"], {"prompt_tokens": 2, "selected_tokens": 4,
                                           "logit_passes": 1, "dtype": "float32", "scale": 1})
        record = self.svc.lifecycle_records(result["request_id"])
        self.assertEqual((record["state"], record["output_tokens"]), ("completed", 0))
        self.assertIs(self.engine.proc, proc)
        self.assertFalse(self.engine.score_q)

    def test_rejected_score_is_nonfatal_while_generation_interleaves_and_successor_scores(self):
        self.start(2)
        result, errors = [], []
        def generate():
            try:
                result.append(self.chat("survivor LONGREPLY"))
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=generate)
        thread.start()
        try:
            deadline = time.monotonic() + 3
            while not self.log.exists() and time.monotonic() < deadline:
                time.sleep(.005)
            proc = self.engine.proc
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.score(selected=[999])
            with rejected.exception as response:
                self.assertEqual(response.code, 400)
            self.assertEqual(self.score(prompt=[5] * 96)["generated_tokens"], 0)
            self.assertIs(self.engine.proc, proc)
            self.assertTrue(self.engine.alive())
        finally:
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result[0]["choices"][0]["message"]["content"], "ok, " + "la " * 15 + "done.")
        self.assertTrue(any(line.startswith("SCORE ") and int(line.split()[3]) > 0
                            for line in self.log.read_text().splitlines()))
        self.assertEqual(self.chat("successor")["choices"][0]["message"]["content"], "ok, done.")

    def assert_protocol_failure(self, flag, text):
        self.start(2, more=(flag,))
        self.engine.SCORE_TIMEOUT_S = .2
        self.engine.SCORE_DRAIN_S = .2
        with self.assertRaises(urllib.error.HTTPError) as failed:
            self.score()
        with failed.exception as response:
            self.assertEqual(response.code, 503)
            body = json.load(response)
            self.assertIn(text, body["error"]["message"])
            self.assertNotIn("logits", body)
        self.assertFalse(self.engine.alive(), "ambiguous or corrupt score state must be retired")
        self.assertFalse(self.engine.score_q)
        self.assertFalse(self.engine.ctl.locked())

    def test_wrong_generation_terminal_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-terminal", "wrong score terminal")

    def test_wrong_score_id_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-id", "score terminal ID")

    def test_missing_terminal_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-missing-terminal", "missing matching SDONE")

    def test_mismatched_prompt_boundary_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-boundary", "receipt boundary mismatch")

    def test_mismatched_admission_boundary_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-admission", "admission boundary")

    def test_score_cannot_hide_a_wrong_terminal_in_generation_slot_routing(self):
        self.assert_protocol_failure("--score-wrong-batch-terminal", "wrong score terminal")

    def test_partial_option_set_is_never_a_successful_answer(self):
        self.assert_protocol_failure("--score-partial", "incomplete option set")

    def test_wrong_pass_receipt_is_a_protocol_failure(self):
        self.assert_protocol_failure("--score-wrong-pass", "receipt boundary mismatch")

    def test_generated_token_count_cannot_be_disguised_as_scoring(self):
        self.assert_protocol_failure("--score-generated", "generated-token boundary mismatch")

    def test_selected_token_order_receipt_cannot_be_changed(self):
        self.assert_protocol_failure("--score-reordered", "receipt boundary mismatch")

    def test_nonfinite_logit_never_publishes_an_option_set(self):
        self.assert_protocol_failure("--score-nonfinite", "non-finite")

    def test_device_or_state_error_is_not_an_ordinary_rejection(self):
        self.assert_protocol_failure("--score-error", "fatal score engine/state error")

    def test_duplicate_nonterminal_cannot_flood_the_reply_queue(self):
        self.assert_protocol_failure("--score-flood", "duplicate score admission")

    def wait_for(self, predicate, text):
        deadline = time.monotonic() + 5
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate(), text)

    def test_disconnect_during_prompt_read_keeps_admission_until_matching_terminal_drains(self):
        import socket
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gate, cancelled = Path(directory.name) / "drain", Path(directory.name) / "cancelled"
        gate.touch()
        self.start(2, more=("--score-drain-gate", str(gate), "--score-cancel-file", str(cancelled)))
        body = json.dumps({"version": 1, "prompt_token_ids": [5] * 4096, "selected_token_ids": [3, 1]}).encode()
        client = socket.create_connection(self.httpd.server_address, timeout=3)
        self.addCleanup(client.close)
        client.sendall(b"POST /v1/score HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: " +
                       str(len(body)).encode() + b"\r\n\r\n" + body)
        self.wait_for(lambda: self.log.exists() and "SCORE " in self.log.read_text(), "score submitted")
        request_id = self.log.read_text().split()[1]
        client.close()
        self.wait_for(cancelled.exists, "disconnect sends SCANCEL during prompt read")
        self.assertIn(request_id, self.engine.score_q)
        self.assertTrue(self.engine.ctl.locked())
        self.assertIn(request_id, self.svc.admission_active)
        self.assertFalse(self.engine.score_q[request_id]["terminal"])
        self.assertEqual(self.svc.unload(), "busy", "disconnect cannot unload before score terminal drain")
        results, errors = [], []
        def successor():
            try:
                results.append(self.score())
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=successor)
        thread.start()
        try:
            self.wait_for(lambda: len(self.svc.admission_waiters) == 1, "successor waits behind the draining score owner")
            self.assertEqual(list(self.svc.admission_active), [request_id])
            self.assertEqual(sum(line.startswith("SCORE ") for line in self.log.read_text().splitlines()), 1)
        finally:
            gate.unlink(missing_ok=True)
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results[0]["generated_tokens"], 0)
        self.assertNotEqual(results[0]["request_id"], request_id)
        self.wait_for(lambda: not self.svc.admission_active, "both admission owners released after drain")
        self.assertFalse(self.engine.score_q)
        self.assertTrue(self.engine.alive())

    def test_http_score_uses_its_own_bounded_admission_queue(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gate = Path(directory.name) / "hold"
        gate.touch()
        self.start(2, more=("--score-hold-file", str(gate)))
        self.svc.request_queue_limit = 0
        self.engine.info["score_max_pending"] = 2
        results, errors = [], []
        def request():
            try:
                results.append(self.score())
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=request) for _ in range(2)]
        for thread in threads:
            thread.start()
        try:
            self.wait_for(lambda: len(self.svc.admission_active) == 1 and len(self.svc.admission_waiters) == 1,
                          "one score owns control and the other occupies its bounded queue")
            with self.assertRaises(urllib.error.HTTPError) as rejected:
                self.score()
            with rejected.exception as response:
                self.assertEqual(response.code, 429)
                self.assertIn("queue is full", json.load(response)["error"]["message"])
            self.assertEqual(len(self.engine.score_q), 1)
            self.assertEqual(next(iter(self.engine.score_q.values()))["queue"].maxsize, 2)
        finally:
            gate.unlink(missing_ok=True)
            for thread in threads:
                thread.join(10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertTrue(self.engine.alive())

    def test_direct_native_waiter_count_is_bounded(self):
        from serve.server import AdmissionError
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gate = Path(directory.name) / "hold"
        gate.touch()
        self.start(2, more=("--score-hold-file", str(gate)))
        self.engine.info["score_max_pending"] = 2
        request = {"version": 1, "prompt_token_ids": [5], "selected_token_ids": [0]}
        cancels, errors = [threading.Event(), threading.Event()], []
        def native(cancel):
            try:
                self.engine.score(request, cancel)
            except AdmissionError as error:
                errors.append(error)
        threads = [threading.Thread(target=native, args=(cancel,)) for cancel in cancels]
        for thread in threads:
            thread.start()
        try:
            self.wait_for(lambda: self.engine.score_waiters == 2, "bounded direct callers own the queue")
            with self.assertRaisesRegex(AdmissionError, "score queue is full"):
                self.engine.score(request, threading.Event())
        finally:
            for cancel in cancels:
                cancel.set()
            gate.unlink(missing_ok=True)
            for thread in threads:
                thread.join(10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(errors), 2)
        self.assertEqual(self.engine.score_waiters, 0)
        self.assertFalse(self.engine.score_q)

    def test_work_and_payload_shape_are_bounded_before_submission(self):
        self.start(2)
        invalid = [({"prompt": [5] * 4097}, "prompt_token_ids"),
                   ({"selected": [0] * 257}, "selected_token_ids"),
                   ({"selected": [True]}, "int32"), ({"selected": [1.0]}, "int32"),
                   ({"prompt": []}, "prompt_token_ids"), ({"selected": []}, "selected_token_ids"),
                   ({"version": 2}, "version"), ({"temperature": .5}, "unsupported score fields"),
                   ({"max_tokens": 0}, "unsupported score fields"),
                   # the frontend's internal cache fields stay server-owned on score too: a client's own
                   # deadline (or scope) is refused by the closed field set, never honored
                   ({"_strata_cache_deadline_ns": 1}, "unsupported score fields"),
                   ({"_strata_cache_scope": "ab" * 32}, "unsupported score fields")]
        for body, reason in invalid:
            with self.subTest(body_keys=list(body)):
                with self.assertRaises(urllib.error.HTTPError) as refused:
                    self.score(**body)
                with refused.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn(reason, json.load(response)["error"]["message"])
        self.assertEqual(self.log.read_text(), "", "refused work never reaches any native command")
        self.assertEqual(self.score()["generated_tokens"], 0)
        self.assertEqual(self.get("/health")["scoring"],
                         {"available": True, "version": 1, "max_prompt_tokens": 4096,
                          "max_selected_tokens": 256, "max_pending": 8})

    def test_serial_generation_and_score_keep_their_protocols_separate(self):
        import serve.server as server
        self.start(0)
        trace = io.StringIO()
        with mock.patch.object(server, "_BTRACE", True), contextlib.redirect_stderr(trace):
            self.assertEqual(self.score()["generated_tokens"], 0)
        frames = [line.split(" wire< ", 1)[1] for line in trace.getvalue().splitlines() if " wire< " in line]
        self.assertEqual(sum(frame.startswith("SDONE ") for frame in frames), 1)
        self.assertFalse(any(frame.startswith(("T ", "BT ", "DONE ", "BDONE ")) for frame in frames))
        self.assertEqual(self.chat("generation")["choices"][0]["message"]["content"], "ok, done.")

    def test_failed_score_write_never_reuses_ambiguous_ownership(self):
        from serve.server import EngineDied
        self.start(2)
        request = {"version": 1, "prompt_token_ids": [5], "selected_token_ids": [0]}
        with mock.patch.object(self.engine, "_send", side_effect=EngineDied("partial score write")):
            with self.assertRaises(EngineDied):
                self.engine.score(request, threading.Event())
        self.assertFalse(self.engine.alive(), "a failed write and failed cancel drain cannot release live ownership")
        self.assertFalse(self.engine.score_q)
        self.assertFalse(self.engine.ctl.locked())



class ParkedCancellation(unittest.TestCase):
    def test_cancelled_park_is_not_continued_by_an_identical_request(self):
        # Stateful protocol peer: the server must end parked ownership, not merely
        # recycle its frontend reservation. This is not CUDA isolation evidence.
        peer = r'''import queue,sys,threading
lines=queue.Queue()
def reader():
    for line in sys.stdin: lines.put(line.strip())
threading.Thread(target=reader,daemon=True).start()
print('INFO batch_slots=2 slot_cache=1',flush=True)
print('READY 4096 stop',flush=True)
parked=False
yielded=False
while True:
    line=lines.get(timeout=10)
    if line=='QUIT': break
    if line.startswith('BSTOP '):
        parked=False
        continue
    if not line.startswith(('GEN ','BGEN ')): continue
    fields=line.split();slot=int(fields[1]) if fields[0]=='BGEN' else None
    long=len(fields[-1].split(','))>100
    if long and not yielded:
        while True:
            print('PP 64 481 1 1',flush=True)
            try: command=lines.get(timeout=.02)
            except queue.Empty: continue
            if command.startswith('BYIELD '):
                destination=int(command.split()[1]);break
        yielded=parked=True
        print(f'YIELDED {destination} 64',flush=True)
        print('DONE 0 481 1 0 cancel 0 0 0 0 0 0 0 0.0 64 0',flush=True)
        if slot is not None: print(f'BADM {slot} 0',flush=True)
        continue
    # A later identical request gets the parked identity unless cancelled.
    print('T '+str(71 if long and parked else 72 if long else 73),flush=True)
    print('DONE 1 481 1 1 stop 0 0 0',flush=True)
    if slot is not None: print(f'BADM {slot} 0',flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'peer.py'
            script.write_text(peer)
            import serve.server as server
            popen = server.subprocess.Popen
            with mock.patch.object(server.subprocess, 'Popen',
                                   lambda cmd, **kw: popen([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = StrataEngine('strata', [])
            long_cancel, short_release, long_started = threading.Event(), threading.Event(), threading.Event()
            errors, output, native_work = [], {}, {}

            def consume(name, prompt, cancel, ready):
                try:
                    output[name] = [t for t in engine.generate(prompt, 2, {'_strata_control_ready': ready}, cancel)
                                    if t is not None]
                    native_work[name] = dict(engine.last)
                except Exception as error:
                    errors.append(error)

            def short_holds_control(_):
                long_cancel.set()
                if not short_release.wait(5):
                    raise TimeoutError('parked cancellation did not release its frontend slot')

            long = threading.Thread(target=consume, args=('long', [1] * 481, long_cancel,
                                                          lambda _: long_started.set()))
            short = threading.Thread(target=consume, args=('short', [2] * 4, threading.Event(), short_holds_control))
            try:
                long.start()
                self.assertTrue(long_started.wait(3))
                short.start()
                long.join(4)
                self.assertFalse(long.is_alive(), 'cancelled parked request did not terminate')
                self.assertEqual(len(native_work["long"]["legs"]), 1)
                self.assertEqual((native_work["long"]["prompt_read"], native_work["long"]["prompt_ms"]), (64, 1.0))
                self.assertEqual(native_work["long"]["generated"], 0)
                cancelled = threading.Event()
                cancelled.set()
                engine.last = {"generated": 999, "prompt_ms": 123, "legs": [{"kind": "solo", "generated": 999}]}
                self.assertEqual([t for t in engine.generate([3] * 4, 2, {}, cancelled) if t is not None], [])
                self.assertEqual(engine.last["legs"], [])
                self.assertIsNone(engine.last.get("generated"))
                short_release.set()
                short.join(4)
                self.assertFalse(short.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(output, {'long': [], 'short': [73]})
                again = [t for t in engine.generate([1] * 481, 2, {}, threading.Event()) if t is not None]
                self.assertEqual(again, [72], 'new request inherited cancelled parked ownership')
                self.assertFalse(any(engine.slot_busy))
            finally:
                short_release.set()
                long_cancel.set()
                engine.unload()
                long.join(5)
                if short.ident is not None:
                    short.join(5)


if __name__ == "__main__":
    unittest.main()
