#!/usr/bin/env python3
"""Exact assigned #234 CUDA build, queued; transient refusals back off 180 s."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
G = "/home/joshkirklin/strata-workspace/toolchains/cuda129"
BUILD = "/home/joshkirklin/strata-workspace/scratch/judge-234-build"
WINDOW = "/home/joshkirklin/ai-optimise/tools/host-inference/window.py"
CONFIGURE = ["cmake", "-S", str(ROOT), "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CUDA_ARCHITECTURES=70;86", f"-DCMAKE_CUDA_COMPILER={G}/bin/nvcc",
    f"-DCMAKE_CUDA_HOST_COMPILER={G}/bin/x86_64-conda-linux-gnu-g++",
    f"-DCMAKE_C_COMPILER={G}/bin/x86_64-conda-linux-gnu-cc",
    f"-DCMAKE_CXX_COMPILER={G}/bin/x86_64-conda-linux-gnu-g++", f"-DCUDAToolkit_ROOT={G}",
    "-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=ON",
    "-DSTRATA_EXPERIMENTAL_SM60=ON", "-DSTRATA_GGML_DIR=/home/joshkirklin/strata-workspace/scratch/repin-0.1.40.1-ggml",
    f"-DSTRATA_PLE_FIXTURE_DIR={ROOT}/bench/micro"]
COMPILE = ["cmake", "--build", BUILD, "-j6"]
ARGV = [WINDOW, "run", "--class", "build", "--ticket", "234", "--purpose",
        "Build exact serial prefix restoration candidate and fixtures", "--max-s", "1800", "--",
        "bash", "-c", shlex.join(CONFIGURE) + " && " + shlex.join(COMPILE)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "issue": 234, "configure": CONFIGURE, "compile": COMPILE,
              "argv": ARGV, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "attempts": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with (out / "command.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    while True:
        attempt = len(record["attempts"]) + 1
        started = time.monotonic()
        process = subprocess.run(ARGV, cwd=ROOT, capture_output=True, text=True)
        (out / f"attempt-{attempt}.stdout").write_text(process.stdout)
        (out / f"attempt-{attempt}.stderr").write_text(process.stderr)
        record["attempts"].append({"attempt": attempt, "exit_code": process.returncode, "elapsed_s": time.monotonic() - started})
        if process.returncode != 75 or "machine on hold" in process.stdout + process.stderr:
            record["exit_code"] = process.returncode
            record["disposition"] = "keep" if process.returncode == 0 else "blocked"
            with (out / "result.json").open("x") as handle:
                json.dump(record, handle, indent=2)
            print(json.dumps({"exit_code": process.returncode, "disposition": record["disposition"], "out": str(out)}), flush=True)
            print(process.stdout, end="", flush=True)
            print(process.stderr, end="", file=sys.stderr, flush=True)
            return process.returncode
        print(json.dumps({"attempt": attempt, "transient_refusal": process.stdout.strip(), "backoff_s": 180}), flush=True)
        time.sleep(180)


if __name__ == "__main__":
    raise SystemExit(main())
