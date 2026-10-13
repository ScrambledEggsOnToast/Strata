#!/usr/bin/env python3
"""Run the fixed #234 restoration arm through the exclusive machine queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/joshkirklin/strata-workspace/toolchains/mm-het002/bin/python"
WINDOW = "/home/joshkirklin/ai-optimise/tools/host-inference/window.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fixture", "model"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--plan-root", type=Path, default=ROOT / "evidence/issue-234")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    plan_root = args.plan_root.resolve()
    profile = json.loads((plan_root / "serial-profile.json").read_text())
    sys.path.insert(0, "/home/joshkirklin/ai-optimise/tools/host-inference")
    import host
    queued = time.time()
    unit = f"strata-prefix234-{args.action}-{int(queued)}"
    argv = [WINDOW, "run", "--class", "measure", "--ticket", "234", "--purpose",
            f"Serial prefix {args.action} restoration proof (not workload qualification)", "--max-s", "1800", "--",
            "systemd-run", "--user", "--wait", "--pipe", "--collect", "--unit", unit,
            *host.unit_properties(profile, profile["timeout_s"]), f"--working-directory={ROOT}",
            "env", "ISSUE_234_QUEUE_WINDOW=measure", f"ISSUE_234_QUEUED_AT={queued}", PYTHON,
            str(ROOT / "evidence/issue-234/run_restoration.py"), args.action,
            "--profile", str(plan_root / "serial-profile.json"),
            "--fixtures", str(plan_root / "restoration-fixtures.json"),
            "--plan", str(plan_root / "restoration-plan.json"), "--out", str(output)]
    logs = Path(str(output) + "-queue")
    logs.mkdir(parents=True, exist_ok=False)
    with (logs / "command.json").open("x") as stream:
        json.dump({"schema_version": 1, "issue": 234, "argv": argv, "unit": unit, "queued_at": queued}, stream, indent=2)
    attempt = 0
    while True:
        attempt += 1
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        (logs / f"attempt-{attempt}.stdout").write_text(result.stdout)
        (logs / f"attempt-{attempt}.stderr").write_text(result.stderr)
        if result.returncode != 75 or "machine on hold" in result.stdout + result.stderr:
            with (logs / "result.json").open("x") as stream:
                json.dump({"exit_code": result.returncode, "attempts": attempt, "unit": unit}, stream, indent=2)
            print(result.stdout, end="", flush=True)
            print(result.stderr, end="", file=sys.stderr, flush=True)
            return result.returncode
        print(json.dumps({"transient_refusal": result.stdout.strip(), "backoff_s": 180}), flush=True)
        time.sleep(180)


if __name__ == "__main__":
    raise SystemExit(main())
