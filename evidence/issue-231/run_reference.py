#!/usr/bin/env python3
"""Issue-231 bounded serial acceptance capsule; run only through window.py.

Uses the host launcher's normal owned per-run MPS lifecycle and unchanged daily
UUID/cap envelope. No engine/model/profile promotion and no other holder touched.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import judge_reference as ref


def prepare(args):
    profile = ref.load(args.profile)
    compiler = ref.PromptCompiler(profile["tokenizer"])
    options = [{"id": c, "text": value} for c, value in zip("ABCD", ("3", "4", "5", "6"))]
    letter_ids = [compiler.single_token(c) for c in "ABCD"]
    base = [{"role": "user", "content": "What is 2 + 2? Reply only with the letter.\nA. 3\nB. 4\nC. 5\nD. 6"}]
    direct = compiler.compile("arithmetic-sanity-direct", base, letter_ids)
    direct.update(family=ref.FAMILIES[0], declared_options=options, expected_sanity_option="B",
                  evidence_role="arithmetic sanity only, not issue-224 human gold or semantic screen")
    coverage = {**direct, "branch_id": "selected-coverage-beyond-256", "selected_token_ids":
                letter_ids + [i for i in range(300) if i not in letter_ids],
                "evidence_role": "complete selected-token transport/coverage probe, not semantic option gold"}
    long_messages = [{"role": "user", "content": "Read the context before answering.\n" +
        "The following sentence is neutral background, not an answer choice.\n" * 80 +
        "What is 2 + 2? Reply only with the letter.\nA. 3\nB. 4\nC. 5\nD. 6"}]
    long = compiler.compile("prefill-plus-final-input", long_messages, letter_ids)
    long.update(family=ref.FAMILIES[0], declared_options=options, expected_sanity_option="B",
                evidence_role="multi-chunk prompt/read-position probe, not semantic screen")
    requests = [direct, coverage, long]
    for option in options:
        messages = [{"role": "user", "content":
                     f"Question: What is 2 + 2?\nIs {option['text']} the correct answer? Reply only Yes or No."}]
        request = compiler.compile("arithmetic-sanity-yes-no-" + option["id"], messages,
                                   [compiler.single_token("Yes"), compiler.single_token("No")])
        request.update(family=ref.FAMILIES[1], declared_option=option,
                       evidence_role="arithmetic sanity only, not issue-224 human gold or semantic screen")
        requests.append(request)
    ref.save(args.out, {"schema_version": 1, "issue": 231, "requests": requests,
                        "semantic_screen": {"disposition": "blocked", "blocking_issue": 224,
                                            "reason": "200 independently human-labelled development source-state groups are not frozen"}})
    return 0


def run(args):
    # Must be declared by the outer queue command, never an alternate GPU route.
    if os.environ.get("ISSUE_231_QUEUE_WINDOW") != "measure":
        raise RuntimeError("run only through window.py run --class measure --ticket 231")
    sys.path.insert(0, "/home/joshkirklin/ai-optimise/tools/host-inference")
    import host
    profile = ref.load(args.profile)
    if profile["reserve_bytes"] < 8 << 30 or not 60 <= profile["timeout_s"] <= 1800:
        raise RuntimeError("finite lifetime and >=8 GiB host reserve required")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    lock = (host.STATE / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pipe = Path("/tmp") / ("s231-" + str(os.getpid()))
    pipe.mkdir(exist_ok=False)
    os.environ.update(profile.get("env", {}))
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(profile["gpu_uuids"]), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                      CUDA_MODULE_LOADING="LAZY", STRATA_SUPERVISED="1",
                      CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(out / "mps-log"),
                      CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=",".join(f"{i}={cap}" for i, cap in enumerate(profile["mps_caps_bytes"])))
    started = time.monotonic()
    mps = False
    receipts = []
    result = {"schema_version": 1, "issue": 231, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "half_a": "blocked", "half_b": "blocked", "blocking_issue": 224,
              "semantic_screen": "not run; independent-human-labelled 200-group development corpus is absent",
              "profile": str(Path(args.profile).resolve()), "profile_sha256": ref.digest(args.profile),
              "fixtures_sha256": ref.digest(args.fixtures), "receipts": []}
    def interrupted(signum, frame):
        raise RuntimeError(f"owned experiment interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (out / "telemetry.jsonl").open("x") as telemetry:
        def safety():
            if time.monotonic() - started >= profile["timeout_s"] - 30:
                raise RuntimeError("suite lifetime exceeded; keeping cleanup reserve")
            reading = host.sample(profile)
            telemetry.write(json.dumps(reading) + "\n")
            telemetry.flush()
            return reading
        try:
            safety()
            host.verify_power_limits(profile)
            result["mps_preflight"] = host.report_mps_preflight(profile)
            manifest_path = "/home/joshkirklin/ai-optimise/tools/host-inference/release-manifest.json"
            manifest = ref.load(manifest_path)
            identities = dict(manifest["model"]["shards_sha256"])
            pack = Path(manifest["model"]["pack"]["path"])
            identities.update({str(pack / name): expected for name, expected in manifest["model"]["pack"]["files_sha256"].items()})
            model = {"release_manifest_sha256": ref.digest(manifest_path),
                     "quantization": manifest["model"]["quantization"], "revision": manifest["model"]["revision"], "files": {}}
            for path, expected in identities.items():
                safety()
                actual = ref.digest(path)
                if actual != expected:
                    raise RuntimeError("frozen model artifact hash mismatch: " + path)
                model["files"][path] = {"sha256": actual, "bytes": Path(path).stat().st_size}
            ref.save(out / "model-identity.json", model)
            with (out / "mps-start.log").open("x") as log:
                subprocess.run([host.CONTROL, "-d"], env=os.environ, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=30)
            mps = True
            for request in ref.load(args.fixtures)["requests"]:
                safety()
                receipt = ref.capture(profile, request, out / request["branch_id"], check_safety=safety, model_identity=model)
                receipts.append(receipt)
                result["receipts"].append({"branch_id": request["branch_id"],
                    "path": str(out / request["branch_id"] / "receipt.json"),
                    "sha256": ref.digest(out / request["branch_id"] / "receipt.json"),
                    "position": receipt["position"], "selected": len(receipt["selected_token_ids"]),
                    "generated_tokens": receipt["generated_tokens"]})
            result["half_a"] = "keep"
            direct = receipts[0]
            result["arithmetic_sanity_direct"] = {"option_ids": list("ABCD"), "logits": direct["logits"],
                "probabilities": ref.softmax(direct["logits"]), "expected_sanity_option": "B"}
            odds = [receipt["logits"][0] - receipt["logits"][1] for receipt in receipts[-4:]]
            result["arithmetic_sanity_yes_no"] = {"option_ids": list("ABCD"), "log_odds": odds,
                                                   "probabilities": ref.softmax(odds), "expected_sanity_option": "B"}
            result["sanity_claim"] = "sanity-only observations; no semantic screen, family selection or calibration"
        except Exception as error:
            result["half_a"] = "blocked"
            result["error"] = str(error)
        finally:
            if mps:
                try:
                    cleanup = subprocess.run([host.CONTROL], input="quit\n", text=True, env=os.environ,
                                             capture_output=True, timeout=15)
                    result["mps_cleanup"] = {"exit_code": cleanup.returncode, "stderr": cleanup.stderr}
                    if cleanup.returncode or cleanup.stderr.strip():
                        result["half_a"] = "blocked"
                        result["cleanup_error"] = "owned MPS lifecycle did not stop cleanly"
                except Exception as error:
                    result["half_a"] = "blocked"
                    result["cleanup_error"] = str(error)
            lock.close()
            result["elapsed_s"] = time.monotonic() - started
            ref.save(out / "result.json", result)
    print(json.dumps({"half_a": result["half_a"], "half_b": result["half_b"], "error": result.get("error"), "out": str(out)}))
    return 0 if result["half_a"] == "keep" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--fixtures")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return prepare(args) if args.action == "prepare" else run(args)


if __name__ == "__main__":
    sys.exit(main())
