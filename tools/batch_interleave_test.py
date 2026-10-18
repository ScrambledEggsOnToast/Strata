#!/usr/bin/env python3
"""Batch slots, the parts batch_test.py does not reach (#465): a long prompt read WHILE other slots decode (its
chunks interleaved with their windows), and a conversation's next turn continued from the slot that holds it.

  1. solo references (GEN): A, then A's next turn (from the live session), B, C (a long prompt)
  2. the same in the slots: A in slot 0, B in slot 1 next to it, then C in slot 2 while A and B decode - every slot's
     tokens must equal its solo tokens
  3. A's next turn (BGEN into slot 0, which still holds A): read from the slot, its tokens must equal the solo next turn
  4. a long prompt D gives way (BYIELD) at its first chunk boundary, a short E is admitted, D goes on from its slot
     while E decodes: both equal their solo tokens
  6. A in a slot, stopped after 50 tokens and continued on the solo path (GEN of A + its tokens, from the slot): the
     whole equals solo A (what the server does with a request left alone in a slot)
  5. (a measurement) a solo next turn continued from a slot: the drafts accepted
  7. F, A's history as a client sends it back without the reply's thinking, from slot 1's turn checkpoint: equal to
     F solo (which continues from the same checkpoint of the live chain)

Exact comparisons need the same settings as batch_test.py (this script sets STRATA_IQ_MT_MIN=1):
  python tools/batch_interleave_test.py --exe build/strata --config strata-<model>.json \\
      --extra "--pcie-frac 0 --adapt-every 1000000 --no-prefill-borrow"
Without --no-prefill-borrow the slots decoding during C's read see the expert cache without the slots C's prompt
borrowed (those experts run on the CPU, which rounds differently): A's and B's text may then drift from solo.
"""
import argparse, json, os, queue, sys, tempfile, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_test import Engine, tokenizer  # noqa: E402

LONG = ("The history of mathematics is long and full of surprising turns. Early civilizations counted with tally marks, "
        "then with symbols, and later with place-value systems that made arithmetic far easier. ")


def run(eng, out, line, slot=None):
    """Send one GEN/BGEN; returns (tokens of the request line, the BADM flag) - BT/BDONE of other slots are kept."""
    eng.send(line)
    got = []
    for l in out:
        if l.startswith("T "):
            got.append(int(l.split()[1]))
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif slot is None and l.startswith("DONE"):
            return got, None
        elif slot is not None and l.startswith("BADM "):
            return got, l.split()[2] == "1"
    raise SystemExit(f"the engine ended during: {line[:40]} (see {eng.log_path})")


def packed_cases(eng, prompts):
    """Exercise the real segmented caller and lifecycle, not greedy parity against a different arithmetic partition."""
    incoming = queue.Queue(maxsize=64)
    def reader():
        try:
            for line in eng.lines():
                incoming.put(line)
        finally:
            incoming.put(None)
    threading.Thread(target=reader, daemon=True).start()
    deadline = time.monotonic() + 300
    owners, events = {}, []
    def record(tag, slot, prompt, maximum=1, scope=None, packing=None):
        owners[tag] = {"slot": slot, "prompt": list(prompt), "maximum": maximum, "identity": None,
                       "tokens": [], "terminal": None, "progress": 0}
        scope = scope or format(slot + 1, "x") * 64
        extra = f" packing={packing}" if packing else ""
        return f"{tag} {slot} {maximum} scope={scope}{extra} " + ",".join(map(str, prompt))
    def read_frame(allow_fatal=False):
        left = deadline - time.monotonic()
        if left <= 0:
            raise AssertionError("native packed fixture exceeded 300 seconds")
        try:
            line = incoming.get(timeout=left)
        except queue.Empty:
            raise AssertionError("native packed fixture missed its bounded reply") from None
        if line is None:
            raise AssertionError("native engine ended before the tagged terminal")
        fields = line.split()
        events.append(fields)
        if line.startswith(("FATAL", "ERR")):
            if allow_fatal:
                return fields
            raise AssertionError(line)
        if not fields or fields[0] not in ("BACK", "BQPP", "BQT", "BQDONE", "BQREJECT"):
            return fields
        tag = fields[1]
        assert tag in owners, ("reply for unknown request", line)
        owner = owners[tag]
        assert int(fields[2]) == owner["slot"], ("cross-slot reply", line)
        assert owner["terminal"] is None, ("duplicate terminal or late reply", line)
        if fields[0] == "BQREJECT":
            assert owner["identity"] is None, ("refusal after reservation", line)
            owner["terminal"] = fields
            return fields
        identity = tuple(map(int, fields[3:5]))
        assert min(identity) > 0, line
        if fields[0] == "BACK":
            assert owner["identity"] is None, ("duplicate ACK", line)
            owner["identity"] = identity
        else:
            assert owner["identity"] == identity, ("request incarnation crossed", line)
            if fields[0] == "BQT":
                owner["tokens"].append(int(fields[5]))
                assert len(owner["tokens"]) <= owner["maximum"], line
            elif fields[0] == "BQPP":
                position, total, fresh = map(int, fields[5:8])
                assert owner["progress"] <= position <= total - 1, line
                assert total == len(owner["prompt"]) and 0 <= fresh <= position, line
                owner["progress"] = position
            else:
                assert int(fields[5]) == len(owner["tokens"]), ("terminal token accounting", line)
                assert int(fields[6]) == len(owner["prompt"]), line
                assert len(fields) == 14 and fields[13].startswith("prompt_read="), line
                reused, fresh = int(fields[12]), int(fields[13].split("=", 1)[1])
                assert 0 <= reused <= len(owner["prompt"]) - 1, line
                assert 0 <= fresh <= len(owner["prompt"]) - reused, line
                if owner["tokens"]:
                    assert reused + fresh == len(owner["prompt"]), ("last prompt token missing or duplicated", line)
                owner["terminal"] = fields
        return fields
    def drain(tags):
        while any(owners[tag]["terminal"] is None for tag in tags):
            read_frame()
    a, b, long_prompt, short = prompts
    initial = ["single-cell", "max-new-one", "cancel-cold", "fourth-general"]
    records = [record(initial[0], 0, a[-1:]), record(initial[1], 1, a),
               record(initial[2], 2, long_prompt, 4), record(initial[3], 3, b)]
    eng.send("BGENQ4 " + "\t".join(records))
    eng.send("BQCANCEL cancel-cold 2 0 0")
    drain(initial)
    first_token = next(i for i, event in enumerate(events) if event[0] == "BQT")
    assert all(any(event[:2] == ["BACK", tag] for event in events[:first_token]) for tag in initial)
    assert len(owners["single-cell"]["tokens"]) == len(owners["max-new-one"]["tokens"]) == 1
    assert owners["single-cell"]["terminal"][13] == "prompt_read=1"
    ready_done = next(i for i, event in enumerate(events) if event[:2] == ["BQDONE", "single-cell"])
    assert not any(event[0] == "BQPP" for event in events[:ready_done]), \
        "an endpoint-ready owner waited for unrelated prefill"
    assert owners["cancel-cold"]["terminal"][9] == "cancel"
    assert not owners["cancel-cold"]["tokens"]
    # Exact resend has zero segmented rows; only the independent final-token
    # verifier operation runs. Its prior max_new=1 draft state must be coherent.
    eng.send("BGENQ1 " + record("zero-remaining", 1, a))
    drain(["zero-remaining"])
    assert int(owners["zero-remaining"]["terminal"][12]) == len(a) - 1
    suffix = a + owners["zero-remaining"]["tokens"] + [a[-1]]
    eng.send("BGENQ1 " + record("one-row-suffix", 1, suffix))
    drain(["one-row-suffix"])
    assert int(owners["one-row-suffix"]["terminal"][12]) == len(a)
    # Unequal restored frontiers and exact 1/63/64-row suffixes share one
    # complete admission frame. Include the unconsumed predicted token first:
    # each max_new=1 source retains its prompt, not its emitted next token.
    boundary = [("suffix-1", "single-cell", 1),
                ("suffix-63", "one-row-suffix", 63),
                ("suffix-64", "fourth-general", 64)]
    records = []
    for tag, previous, rows in boundary:
        source = owners[previous]
        assert len(source["tokens"]) == 1
        prompt = source["prompt"] + source["tokens"] + [source["prompt"][-1]] * rows
        records.append(record(tag, source["slot"], prompt))
    eng.send("BGENQ3 " + "\t".join(records))
    drain([tag for tag, _, _ in boundary])
    for tag, previous, rows in boundary:
        owner = owners[tag]
        reused = int(owner["terminal"][12])
        fresh = int(owner["terminal"][13].split("=", 1)[1])
        assert len(owner["tokens"]) == 1
        assert reused == len(owners[previous]["prompt"]), ("wrong suffix source frontier", tag, reused)
        assert len(owner["prompt"]) - reused - 1 == rows, ("wrong segmented suffix extent", tag)
        assert fresh == rows + 1, ("final prompt token missing or duplicated", tag, fresh)
    # A ready singleton starts immediately. The second complete request arrives
    # after its ACK and must get its own ACK before the long reply completes.
    eng.send("BGENQ1 " + record("late-long", 0, long_prompt, 2))
    while owners["late-long"]["identity"] is None:
        read_frame()
    eng.send("BGENQ1 " + record("late-short", 2, short))
    old_request, old_incarnation = owners["single-cell"]["identity"]
    eng.send(f"BQCANCEL single-cell 0 {old_request} {old_incarnation}")
    drain(["late-long", "late-short"])
    late_ack = next(i for i, event in enumerate(events) if event[:2] == ["BACK", "late-short"])
    long_done = next(i for i, event in enumerate(events) if event[:2] == ["BQDONE", "late-long"])
    assert late_ack < long_done, "late ready owner waited for the entire first prefill"
    short_done = next(i for i, event in enumerate(events) if event[:2] == ["BQDONE", "late-short"])
    assert short_done < long_done, "a completed short owner waited for an unrelated generation terminal"
    assert owners["late-long"]["terminal"][9] != "cancel", "stale control reached a reused physical slot"
    # One invalid reservation rejects the entire complete two-owner frame. The
    # free destination remains usable by an unrelated subsequent request.
    eng.send("BGENQ1 " + record("atomic-held", 0, long_prompt, 2, scope="e" * 64))
    while owners["atomic-held"]["identity"] is None:
        read_frame()
    eng.send("BGENQ2 " + "\t".join([record("atomic-busy", 0, a), record("atomic-free", 3, b)]))
    while any(owners[tag]["terminal"] is None for tag in ("atomic-busy", "atomic-free")):
        read_frame()
    assert all(owners[tag]["terminal"][0] == "BQREJECT" for tag in ("atomic-busy", "atomic-free"))
    eng.send("BGENQ1 " + record("after-atomic", 3, short))
    drain(["atomic-held", "after-atomic"])
    # #249: ordinary owners share one pack; a priority owner asking for the solo policy is admitted
    # late and executes alone.  The engine receipts, not the configured keys, prove the composition.
    mixed_offset = Path(eng.log_path).stat().st_size
    mixed = ["mixed-ordinary-a", "mixed-ordinary-b"]
    eng.send("BGENQ2 " + "\t".join(record(tag, slot, long_prompt, 2, packing="eligible")
                                   for slot, tag in enumerate(mixed)))
    while any(owners[tag]["identity"] is None for tag in mixed):
        read_frame()
    # The priority request must arrive during ordinary prefill, after the ordinary owners really
    # shared a pack (2048 of their ~5000 rows), not before their first selection.
    mixed_until = time.monotonic() + 30
    while True:
        tail = Path(eng.log_path).read_text()[mixed_offset:]
        if sum(1 for line in tail.splitlines() if "event=frontier" in line and
               ("tag=mixed-ordinary-a" in line or "tag=mixed-ordinary-b" in line)) >= 2:
            break
        if time.monotonic() >= mixed_until:
            raise AssertionError("ordinary owners did not share a pack before the priority arrival")
        time.sleep(0.01)
    eng.send("BGENQ1 " + record("mixed-priority", 2, long_prompt, 2, packing="solo"))
    drain(mixed + ["mixed-priority"])
    tail = Path(eng.log_path).read_text()[mixed_offset:]
    executions = {}
    for line in tail.splitlines():
        if "PACKED_PREFILL event=drained" not in line:
            continue
        fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
        executions[fields["execution"]] = {"kind": fields["kind"], "owners": int(fields["owners"]), "frontiers": []}
    for line in tail.splitlines():
        if "PACKED_PREFILL event=frontier" not in line:
            continue
        fields = dict(item.split("=", 1) for item in line.split() if "=" in item)
        executions[fields["execution"]]["frontiers"].append((fields["tag"], fields["packing"]))
    assert any(e["kind"] == "packed" and e["owners"] > 1 and {tag for tag, _ in e["frontiers"]} == set(mixed)
               for e in executions.values()), ("ordinary owners did not share one pack", executions)
    solo = [e for e in executions.values() if e["kind"] == "solo"]
    assert solo, ("the priority owner did not run alone", executions)
    assert all(e["owners"] == 1 and e["frontiers"] == [("mixed-priority", "solo")] for e in solo), executions
    assert all(e["owners"] == 1 for e in executions.values() if e["kind"] == "fairness"), executions
    assert any("tag=mixed-priority" in line and "packing=solo" in line for line in tail.splitlines()), \
        "the priority reservation did not report its packing policy"
    eng.send("BGENQ1 " + record("eos-probe", 1, a[-1:], scope="f" * 64))
    drain(["eos-probe"])
    eos_token = owners["eos-probe"]["tokens"][0]
    # A fragment is never a runnable owner. EOF rejects it rather than treating
    # it as one final complete admission. The caller supplies a disposable store.
    eng.p.stdin.write("BGENQ1 fragment 0 1 1,2")
    eng.p.stdin.flush()
    eng.p.stdin.close()
    fatal = read_frame(allow_fatal=True)
    assert fatal and fatal[0] == "FATAL", ("incomplete frame was not refused", fatal)
    eng.p.wait(timeout=30)
    assert eng.p.returncode != 0
    print("PASS real native packed: four owners, early independent ACK, late arrival, cancellation, "
          "one-cell/zero-remaining/1/63/64-row suffix first token, exact reuse, atomic refusal, EOF fragment", flush=True)
    return eos_token


def packed_eos_case(eng, prompt):
    """The observed one-row greedy probe is now an explicit EOS, with unchanged arithmetic."""
    frames = queue.Queue(maxsize=16)
    def reader():
        try:
            for line in eng.lines():
                frames.put(line)
        finally:
            frames.put(None)
    threading.Thread(target=reader, daemon=True).start()
    eng.send("BGENQ1 eos-first 1 4 scope=" + "f" * 64 + " " + ",".join(map(str, prompt)))
    identity, tokens = None, []
    until = time.monotonic() + 60
    while True:
        left = until - time.monotonic()
        if left <= 0:
            raise AssertionError("native EOS first-token case exceeded 60 seconds")
        line = frames.get(timeout=left)
        assert line is not None and not line.startswith(("ERR", "FATAL")), line
        fields = line.split()
        if fields[:2] == ["BACK", "eos-first"]:
            identity = fields[2:5]
        elif fields[:2] == ["BQT", "eos-first"]:
            assert fields[2:5] == identity, line
            tokens.append(int(fields[5]))
        elif fields[:2] == ["BQDONE", "eos-first"]:
            assert fields[2:5] == identity and fields[9] == "stop", line
            assert len(tokens) == int(fields[5]) == 1, line
            eng.send("QUIT")
            eng.p.wait(timeout=30)
            assert eng.p.returncode == 0
            print("PASS real native packed: first-token EOS with max_new=4 and mandatory MTP catch-up", flush=True)
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--long", type=int, default=5000, help="tokens of the long prompt C (several prompt chunks)")
    ap.add_argument("--extra", default="")
    ap.add_argument("--packed-prefill", action="store_true",
                    help="real tagged packed-prefill behavioral cases; no exact-token parity gate")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    tok = tokenizer(cfg["tokenizer"])

    def chat(q):
        return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                          parse_special=True)
    body = LONG * (a.long // max(1, len(tok.encode(LONG))) + 1)
    A = chat("Write a Python function that merges overlapping intervals, then explain it.")
    B = chat("Compare TCP and QUIC: handshake, congestion control, multiplexing.")
    C = chat(body + "\nSummarize the text above in three sentences.")
    D = chat(body + "\nWhat are the three most important ideas in the text above?")
    E = chat("Describe the life cycle of a star like the Sun.")
    print(f"prompts: A {len(A)}, B {len(B)}, C {len(C)} tokens", flush=True)
    if a.packed_prefill:
        with tempfile.TemporaryDirectory(prefix="strata-packed-fixture-") as cache:
            eng = Engine(a.exe, cfg, 4, {"STRATA_PACKED_PREFILL": "1"},
                         a.extra.split() + ["--conversation-cache-dir", cache])
            try:
                eos_token = packed_cases(eng, [A, B, C, E])
            finally:
                if eng.p.poll() is None:
                    eng.send("QUIT")
                    eng.p.wait(timeout=30)
            eng.log.close()
            # Preserve the first native log; do not truncate its retained pack,
            # cancellation and EOF receipts when starting the EOS-only case.
            previous_log = os.environ.get("BATCH_TEST_LOG")
            descriptor, eos_log = tempfile.mkstemp(prefix="batch-packed-eos-", suffix=".log",
                                                   dir=str(Path(eng.log_path).parent))
            os.close(descriptor)
            os.environ["BATCH_TEST_LOG"] = eos_log
            try:
                eos = Engine(a.exe, cfg, 4, {"STRATA_PACKED_PREFILL": "1"},
                             a.extra.split() + ["--conversation-cache-dir", str(Path(cache) / "eos"),
                                               "--eos-ids", str(eos_token)])
                try:
                    packed_eos_case(eos, A[-1:])
                finally:
                    if eos.p.poll() is None:
                        eos.send("QUIT")
                        eos.p.wait(timeout=30)
                    eos.log.close()
                print(f"native logs: {eng.log_path}; {eos_log}", flush=True)
            finally:
                if previous_log is None:
                    os.environ.pop("BATCH_TEST_LOG", None)
                else:
                    os.environ["BATCH_TEST_LOG"] = previous_log
            return 0
    eng = Engine(a.exe, cfg, 3, {"STRATA_IQ_MT_MIN": "1"}, a.extra.split())
    eng.pending = []
    out = eng.lines()
    ids = lambda v: ",".join(map(str, v))
    M = a.max_new

    # 1. solo
    sA, _ = run(eng, out, f"GEN {M} {ids(A)}")
    A2 = A + sA + chat("Now do the same in Rust.")
    sA2, _ = run(eng, out, f"GEN {M} {ids(A2)}")
    # F: A's history as a client sends it back WITHOUT the reply's thinking - it shares A only up to A's last turn
    # boundary (the checkpoint there); solo it continues from that checkpoint of the live chain
    turn_at = max(i for i, t in enumerate(A) if t == A[0])          # A[0] is <|im_start|>
    F = A[:turn_at] + tok.encode("<|im_start|>assistant\nA short answer.<|im_end|>\n", parse_special=True) + \
        chat("Now in Go.")
    sF, _ = run(eng, out, f"GEN {M} {ids(F)}")
    sB, _ = run(eng, out, f"GEN {M} {ids(B)}")
    sC, _ = run(eng, out, f"GEN {M} {ids(C)}")
    sD, _ = run(eng, out, f"GEN {M} {ids(D)}")
    sE, _ = run(eng, out, f"GEN {M} {ids(E)}")
    print(f"solo: A {len(sA)}, A2 {len(sA2)}, B {len(sB)}, C {len(sC)}, D {len(sD)}, E {len(sE)} tokens", flush=True)

    # 2. the slots: A, B, then the long C while they decode
    got = {0: [], 1: [], 2: []}
    done = {}
    t0 = time.time()
    for s, P in ((0, A), (1, B), (2, C)):
        first, cont = run(eng, out, f"BGEN {s} {M} {ids(P)}", slot=s)
        got[s] += first
        if not cont:
            done[s] = True
    t_admit = time.time() - t0
    during_c = sum(1 for l in eng.pending if l.startswith("BT "))   # windows the slots ran while C was admitted

    def drain():
        while eng.pending or len(done) < 3:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT "):
                _, s, y = l.split()
                got[int(s)].append(int(y))
            elif l.startswith("BDONE "):
                done[int(l.split()[1])] = True
            elif l.startswith("ERR"):
                raise SystemExit("engine: " + l)
    drain()
    print(f"slots: admissions {t_admit:.1f}s; {during_c} slot tokens arrived while the admissions read their prompts",
          flush=True)
    ok = True
    for s, ref, name in ((0, sA, "A"), (1, sB, "B"), (2, sC, "C")):
        same = got[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got[s]), len(ref))) if got[s][k] != ref[k]), None)
        print(f"slot {s} ({name}): {len(got[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 3. A's next turn from slot 0 (the engine's log says "slot 0 gave back ...")
    first, cont = run(eng, out, f"BGEN 0 {M} {ids(A2)}", slot=0)
    got2, done = first, {}
    if cont:
        while True:
            l = eng.pending.pop(0) if eng.pending else next(out)
            if l.startswith("BT 0 "):
                got2.append(int(l.split()[2]))
            elif l.startswith("BDONE 0 "):
                break
    same = got2 == sA2
    ok &= same
    d = next((k for k in range(min(len(got2), len(sA2))) if got2[k] != sA2[k]), None)
    print(f"A's next turn from its slot: {len(got2)} tokens, solo {len(sA2)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 4. a long prompt D gives way (BYIELD) to a short E at its first chunk boundary, then goes on from its slot
    eng.send(f"BGEN 2 {M} {ids(D)}")
    eng.send("BYIELD 2")
    yielded = None
    for l in out:
        if l.startswith("YIELDED "):
            yielded = l
        elif l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("ERR"):
            raise SystemExit("engine: " + l)
        elif l.startswith("BADM 2 "):
            break
    print(f"D gave way: {yielded}", flush=True)
    ok &= yielded is not None
    got4 = {1: [], 2: []}
    first, cont1 = run(eng, out, f"BGEN 1 {M} {ids(E)}", slot=1)
    got4[1] += first
    first, cont2 = run(eng, out, f"BGEN 2 {M} {ids(D)}", slot=2)
    got4[2] += first
    done4 = {s for s, c in ((1, cont1), (2, cont2)) if not c}
    while len(done4) < 2:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT "):
            _, s, y = l.split()
            if int(s) in got4:
                got4[int(s)].append(int(y))
        elif l.startswith("BDONE "):
            done4.add(int(l.split()[1]))
    for s, ref, name in ((1, sE, "E (short, admitted while D waited)"), (2, sD, "D (gave way, then went on)")):
        same = got4[s] == ref
        ok &= same
        d = next((k for k in range(min(len(got4[s]), len(ref))) if got4[s][k] != ref[k]), None)
        print(f"slot {s} {name}: {len(got4[s])} tokens, solo {len(ref)}: "
              f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    # 6. back to the solo path (what the server does with a request left alone in a slot): A in slot 1, BSTOP after
    # 50 tokens, then GEN of A + what it produced - the engine continues from the slot with MTP drafts again
    first, cont = run(eng, out, f"BGEN 1 {M} {ids(A)}", slot=1)
    got6, sent = list(first), False
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 1 "):
            got6.append(int(l.split()[2]))
            if len(got6) >= 50 and not sent:
                eng.send("BSTOP 1")
                sent = True
        elif l.startswith("BDONE 1 "):
            break
    tail6, _ = run(eng, out, f"GEN {M - len(got6)} {ids(A + got6)}") if len(got6) < M else ([], None)
    same = got6 + tail6 == sA
    ok &= same
    both = got6 + tail6
    d = next((k for k in range(min(len(both), len(sA))) if both[k] != sA[k]), None)
    print(f"A in a slot, then solo again after {len(got6)} tokens: {len(both)} tokens, solo {len(sA)}: "
          f"{'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)

    # 5. (a measurement, not a check) a SOLO next turn continued from slot 0: its tokens are exact either way, but the
    # draft layer's own K/V was built for another conversation, so fewer drafts may be accepted than in a solo next
    # turn whose drafter read the conversation (A2 in step 1)
    A3 = A2 + got2 + chat("Now in Go.")
    eng.send(f"GEN {M} {ids(A3)}")
    for l in out:
        if l.startswith(("BT ", "BDONE ")):
            eng.pending.append(l)
        elif l.startswith("DONE"):
            f = l.split()
            print(f"solo next turn from a slot: drafts accepted {f[6]} of {f[7]} ({f[1]} tokens in {f[4]} ms)",
                  flush=True)
            break
    # 7. F from slot 1's turn checkpoint (slot 1 holds A since step 6): the engine log says "(its turn checkpoint)"
    first, cont = run(eng, out, f"BGEN 2 {M} {ids(F)}", slot=2)
    got7 = list(first)
    while cont:
        l = eng.pending.pop(0) if eng.pending else next(out)
        if l.startswith("BT 2 "):
            got7.append(int(l.split()[2]))
        elif l.startswith("BDONE 2 "):
            break
    same = got7 == sF
    ok &= same
    d = next((k for k in range(min(len(got7), len(sF))) if got7[k] != sF[k]), None)
    print(f"F (A's history without the reply's thinking) from slot 1's turn checkpoint: {len(got7)} tokens, solo "
          f"{len(sF)}: {'IDENTICAL' if same else f'DIFFERS at {d}'}", flush=True)
    eng.send("QUIT")
    eng.p.wait(timeout=180)
    log = Path(eng.log_path).read_text(errors="replace")
    for l in log.splitlines():
        if "drafts accepted" in l:
            print("  log:", l.split("strata serve: ")[-1][:160], flush=True)
    for key in ("gave back", "its turn checkpoint", "takes", "the prompt was read in", "gives way"):
        print(f"engine log '{key}': {log.count(key)} lines", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
