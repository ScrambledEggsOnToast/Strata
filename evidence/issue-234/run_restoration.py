#!/usr/bin/env python3
"""Bounded #234 restoration proof, never a workload qualification campaign.

Use prepare before any evaluation. Run fixture/model only through the machine
queue and a finite transient unit with host.py's unchanged profile properties.
The model arm owns its own private MPS lifecycle; no other daemon is touched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import judge_reference as ref


def prepare(args):
    profile = ref.load(args.profile)
    started = time.monotonic()
    compiler = ref.PromptCompiler(profile["tokenizer"])
    selected = [compiler.single_token(letter) for letter in "ABCD"]
    background = "Read the context before answering.\n" + (
        "The following sentence is neutral background, not an answer choice.\n" * 80)
    requests = []
    for branch, question in (("restoration-A", "2 + 2"), ("restoration-B", "2 + 3"),
                             ("restoration-A-replay", "2 + 2")):
        request = compiler.compile(branch, [{"role": "user", "content": background +
            f"What is {question}? Reply only with the letter.\nA. 3\nB. 4\nC. 5\nD. 6"}], selected)
        request["evidence_role"] = "owned arithmetic restoration probe; not human gold or independent source groups"
        requests.append(request)
    prefix = ref.common_prefix(requests)
    fixtures = {"schema_version": 1, "issue": 234, "requests": requests,
                "prefix_tokens": prefix, "tokenisation_s": time.monotonic() - started}
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / "serial-profile.json"
    if profile_path != Path(args.profile).resolve():
        ref.save(profile_path, profile)
    ref.save(out / "restoration-fixtures.json", fixtures)
    plan = {"schema_version": 1, "issue": 234, "fixed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evidence_scope": "restoration only; one owned synthetic source with A/B/A variants, not qualification",
            "execution_leg": "serial_GEN", "source": ref.source_identity(),
            "binary": {"path": profile["binary"], "sha256": ref.digest(profile["binary"])},
            "profile_sha256": ref.digest(profile_path),
            "fixtures_sha256": ref.digest(out / "restoration-fixtures.json"),
            "mps_caps_bytes": profile["mps_caps_bytes"], "reserve_bytes": profile["reserve_bytes"],
            "memory_max_bytes": profile["memory_max_bytes"], "timeout_s": profile["timeout_s"],
            "prefix_tokens": prefix, "branches": [{"id": request["branch_id"], "token_digest": request["token_digest"],
                "tokens": len(request["prompt_token_ids"])} for request in requests],
            "fixture_matrix": {"formats": ["f16", "int8", "q4", "k8v4"], "kv_modes": [0, 1],
                "prefix_lengths": [1, 3, 4, 5, 63, 64, 65], "cases": 56,
                "negative": "retain suffix PLE history while restoring every other element; production verification must reject",
                "other_negatives": ["one-byte GDN/PLE/tail/dead/block-position omission", "PLE current and prior token scalars",
                    "pooled boundary rebuild omission", "prefix KV contamination", "reuse after release"]},
            "model_sequence": ["fresh A", "fresh A control", "fresh B", "one prefix prefill then restored A/B/A"],
            "restoration_criteria": ["all 56 GPU fixture cases pass their positive and deliberate rejection assertions",
                "every reused serial GEN row has full coverage, correct position, restoration_verified and no generated answer",
                "one PREFIX_CAPTURE, three PREFIX_RESTORE; complete stage suffix-state inventory",
                "A replay has identical FP32 selected logits and identical committed GDN/PLE/KV fingerprints on every stage",
                "all safety telemetry valid; private MPS caps exactly equal daily values and clean teardown"],
            "timing_scope": "retained tokenisation + measured queue wait + complete process/call wall time including checkpoint and restore",
            "numerical_campaign": {"disposition": "blocked", "required_groups_per_workload_per_mode": 3000,
                "primary": "one preselected branch per independent source group", "modes": ["idle", "maximum_coexistence"],
                "maximum_coexistence": {"generation_slots": 4, "concurrent_submitted_judgments": 2},
                "failure": "argmax change OR max abs delta probability >0.02 OR TV>0.05",
                "upper_bound": "exact one-sided 95% failure-rate upper bound <=0.01",
                "all_rows": {"max_difference": 0.05, "max_tv": 0.10, "p95_tv": 0.02,
                    "missing_nonfinite_wrong_position_identity_contamination_generated_answer": 0},
                "fresh_fresh_control": "required on identical full token plans",
                "blockers": ["#224 independent-human-labelled frozen source-state corpus is absent",
                    "contract conflict: #224 200+400+600=1200 groups total cannot supply #234 3000 per workload and mode",
                    "maximum-load contended logits dump wiring is absent; head rows exist, but extending wiring is outside #234"],
                "no_parallel_claim": True, "no_synthetic_qualification": True}}
    ref.save(out / "restoration-plan.json", plan)
    print(json.dumps({"fixed_utc": plan["fixed_utc"], "prefix_tokens": prefix,
                      "plan_sha256": ref.digest(out / "restoration-plan.json")}))
    return 0


def drift(a, b):
    p, q = ref.softmax(a["logits"]), ref.softmax(b["logits"])
    return {"argmax_change": p.index(max(p)) != q.index(max(q)),
            "max_probability_difference": max(abs(x - y) for x, y in zip(p, q)),
            "tv": sum(abs(x - y) for x, y in zip(p, q)) / 2,
            "max_logit_difference": max(abs(x - y) for x, y in zip(a["logits"], b["logits"]))}


def run(args):
    if os.environ.get("ISSUE_234_QUEUE_WINDOW") != "measure":
        raise RuntimeError("run only through window.py run --class measure --ticket 234")
    sys.path.insert(0, "/home/joshkirklin/ai-optimise/tools/host-inference")
    import host
    profile, plan = ref.load(args.profile), ref.load(args.plan)
    if (profile["reserve_bytes"] < 8 << 30 or not 60 <= profile["timeout_s"] <= 1800 or
            ref.digest(args.profile) != plan["profile_sha256"] or
            ref.digest(args.fixtures) != plan["fixtures_sha256"] or
            ref.digest(profile["binary"]) != plan["binary"]["sha256"]):
        raise RuntimeError("finite frozen restoration plan/profile/binary binding required")
    if ref.source_identity() != plan["source"]:
        raise RuntimeError("source changed after the prospective restoration plan was fixed")
    daily = ref.load("/home/joshkirklin/ai-optimise/tools/host-inference/profile.json")
    if profile["mps_caps_bytes"] != daily["mps_caps_bytes"] or profile["mps_caps_bytes"] != plan["mps_caps_bytes"]:
        raise RuntimeError("restoration capsule cannot change any frozen MPS cap")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    lock = (host.STATE / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    queued_at = float(os.environ["ISSUE_234_QUEUED_AT"])
    result = {"schema_version": 1, "issue": 234, "kind": args.action, "disposition": "blocked",
              "plan_sha256": ref.digest(args.plan), "profile_sha256": ref.digest(args.profile),
              "execution_leg": "serial_GEN" if args.action == "model" else "CUDA fixtures (no model readout)",
              "queueing_s": max(0, time.time() - queued_at), "numerical_qualification": "blocked; no campaign rows",
              "source": ref.source_identity(), "binary": plan["binary"], "receipts": []}
    mps = False
    pipe = None
    def interrupted(signum, frame):
        raise RuntimeError(f"owned restoration proof interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    os.environ.update(profile.get("env", {}))
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(profile["gpu_uuids"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      CUDA_MODULE_LOADING="LAZY", STRATA_SUPERVISED="1")
    with (out / "telemetry.jsonl").open("x") as telemetry:
        def safety():
            if time.monotonic() - started >= profile["timeout_s"] - 30:
                raise RuntimeError("restoration proof lifetime exceeded; retaining cleanup reserve")
            reading = host.sample(profile)
            telemetry.write(json.dumps(reading) + "\n")
            telemetry.flush()
            return reading
        def control(text, timeout=5):
            return subprocess.run([host.CONTROL], input=text + "\n", text=True, env=os.environ,
                                  capture_output=True, timeout=timeout, check=False)
        try:
            safety()
            host.verify_power_limits(profile)
            if args.action == "fixture":
                argv = [str(Path(profile["binary"]).parent / "conversation_snapshot_test")]
                with (out / "fixture.stdout").open("xb") as stdout, (out / "fixture.stderr").open("xb") as stderr:
                    process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=os.environ)
                    try:
                        while process.poll() is None:
                            safety(); time.sleep(1)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                process.kill(); process.wait(timeout=15)
                result["command"] = argv
                result["exit_code"] = process.returncode
                if process.returncode:
                    raise RuntimeError(f"restoration fixture suite exited {process.returncode}")
                rows = [line for line in (out / "fixture.stdout").read_text().splitlines()
                        if line.startswith("PREFIX_RESTORATION ")]
                if len(rows) != plan["fixture_matrix"]["cases"]:
                    raise RuntimeError("restoration fixture matrix denominator differs")
                result["fixture_cases"] = rows
                result["omission_negative"] = "expected rejection retained in each of 56 cases; not repaired into an accepting case"
                result["disposition"] = "keep"
            else:
                result["mps_preflight"] = host.report_mps_preflight(profile)
                pipe = Path("/tmp") / ("s234-" + str(os.getpid()))
                pipe.mkdir(exist_ok=False)
                os.environ.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(out / "mps-log"),
                    CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=",".join(f"{i}={cap}" for i, cap in enumerate(profile["mps_caps_bytes"])))
                with (out / "mps-start.log").open("x") as log:
                    subprocess.run([host.CONTROL, "-d"], env=os.environ, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=30)
                mps = True
                ready_until = time.monotonic() + 20
                while control("get_server_list").stderr.strip():
                    safety()
                    if time.monotonic() >= ready_until:
                        raise RuntimeError("private MPS daemon readiness failed")
                    time.sleep(.5)
                declarations = []
                for ordinal, cap in enumerate(profile["mps_caps_bytes"]):
                    command = f"set_default_device_pinned_mem_limit {ordinal} {cap // 2**20}M"
                    applied = control(command)
                    declarations.append({"command": command, "cap_bytes": cap, "frozen_cap_bytes": daily["mps_caps_bytes"][ordinal],
                                         "exit_code": applied.returncode, "stderr": applied.stderr})
                    if applied.returncode or applied.stderr.strip():
                        raise RuntimeError("private MPS cap initialisation refused")
                result["mps_declarations"] = declarations
                server = control("start_server", timeout=10)
                if server.returncode or server.stderr.strip():
                    raise RuntimeError("private MPS start_server refused")
                manifest_path = "/home/joshkirklin/ai-optimise/tools/host-inference/release-manifest.json"
                manifest = ref.load(manifest_path)
                identities = dict(manifest["model"]["shards_sha256"])
                pack = Path(manifest["model"]["pack"]["path"])
                identities.update({str(pack / name): expected for name, expected in manifest["model"]["pack"]["files_sha256"].items()})
                model = {"release_manifest_sha256": ref.digest(manifest_path), "revision": manifest["model"]["revision"],
                         "quantization": manifest["model"]["quantization"], "files": {}}
                for path, expected in identities.items():
                    safety()
                    actual = ref.digest(path)
                    if actual != expected:
                        raise RuntimeError("frozen model artifact differs: " + path)
                    model["files"][path] = {"sha256": actual, "bytes": Path(path).stat().st_size}
                ref.save(out / "model-identity.json", model)
                fixtures = ref.load(args.fixtures)
                requests = fixtures["requests"]
                fresh = []
                for name, request in (("fresh-A", requests[0]), ("fresh-A-control", requests[0]), ("fresh-B", requests[1])):
                    fresh.append(ref.capture(profile, request, out / name, check_safety=safety, model_identity=model))
                reused = ref.capture(profile, requests[0], out / "reused-A-B-A", check_safety=safety,
                                     model_identity=model, branch_requests=requests)
                stderr = (out / "reused-A-B-A" / "engine.stderr").read_text()
                if stderr.count("PREFIX_CAPTURE ") != 1 or stderr.count("PREFIX_RESTORE ") != 3:
                    raise RuntimeError("prefix prefill/restore call inventory differs")
                pattern = r"PREFIX_SUFFIX_STATE branch=(\d+) stage=(\d+) cells=(\d+) gdn=([0-9a-f]+) ple=([0-9a-f]+) kv=([0-9a-f]+) leg=serial_GEN"
                states = {(int(branch), int(stage)): (cells, gdn, ple, kv)
                          for branch, stage, cells, gdn, ple, kv in re.findall(pattern, stderr)}
                expected_stages = len(profile["engine_args"][profile["engine_args"].index("--layer-split") + 1].split(",")) + 1
                if len(states) != 3 * expected_stages or any(states[(0, stage)] != states[(2, stage)] for stage in range(expected_stages)):
                    raise RuntimeError("A/B/A committed state differs or stage inventory is incomplete")
                if reused[0]["logits"] != reused[2]["logits"]:
                    raise RuntimeError("A/B/A selected FP32 logits differ after restoration")
                result["suffix_states"] = [{"branch": branch, "stage": stage, "cells": values[0], "gdn": values[1],
                    "ple": values[2], "kv": values[3]} for (branch, stage), values in sorted(states.items())]
                result["observational_drift"] = {"fresh_fresh_A": drift(fresh[0], fresh[1]),
                    "fresh_reused_A": drift(fresh[0], reused[0]), "fresh_reused_B": drift(fresh[2], reused[1]),
                    "reused_A_replay": drift(reused[0], reused[2])}
                result["receipts"] = [str(out / name / "receipt.json") for name in ("fresh-A", "fresh-A-control", "fresh-B")]
                result["receipts"] += [str(out / "reused-A-B-A" / ("receipt.json" + (f".{i}" if i else ""))) for i in range(3)]
                result["timing"] = {"tokenisation_s": fixtures["tokenisation_s"], "queueing_s": result["queueing_s"],
                    "fresh_call_s": [receipt["elapsed_s"] for receipt in fresh], "reused_call_s": reused[-1]["elapsed_s"],
                    "checkpoint_ms": reused[0]["checkpoint_ms"], "restore_ms": [receipt["restore_ms"] for receipt in reused]}
                result["disposition"] = "keep"
                result["claim"] = "one synthetic serial GEN A/B/A restoration probe only; not numerical workload qualification"
            safety()
        except Exception as error:
            result["error"] = str(error)
            result["disposition"] = "blocked"
        finally:
            if mps:
                try:
                    cleanup = control("quit", timeout=15)
                    result["mps_cleanup"] = {"exit_code": cleanup.returncode, "stderr": cleanup.stderr}
                    if cleanup.returncode or cleanup.stderr.strip():
                        result["disposition"] = "blocked"
                        result["cleanup_error"] = "private MPS lifecycle did not tear down cleanly"
                except Exception as error:
                    result["disposition"] = "blocked"; result["cleanup_error"] = str(error)
            lock.close()
            result["elapsed_s"] = time.monotonic() - started
            ref.save(out / "result.json", result)
    print(json.dumps({"disposition": result["disposition"], "error": result.get("error"), "out": str(out)}))
    return 0 if result["disposition"] == "keep" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fixture", "model"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fixtures")
    parser.add_argument("--plan")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return prepare(args) if args.action == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
