#!/usr/bin/env python3
"""Fresh final-position reference receipts and a full-denominator development screen.

Capture must run inside an authorised, supervised machine window. This module
never creates an MPS daemon or changes clocks, caps, profiles or model files.
Screen preparation/evaluation are CPU-only; absent issue-224 human gold is a block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate, literal_tags, mark_think_literals, unmark_think_literals
from strata_tokenizer import Tokenizer

FAMILIES = ("direct_letter_logits", "per_option_yes_no_log_odds")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_digest(ids):
    return hashlib.sha256(b"".join(struct.pack("<i", value) for value in ids)).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def serial_profile(daily, binary, frontend, timeout_s=900):
    """Copy the daily profile, retaining model, context, placement and safety caps."""
    if type(timeout_s) is not int or not 60 <= timeout_s <= 1800:
        raise ValueError("reference lifetime must be an integer in 60..1800 seconds")
    profile = json.loads(json.dumps(daily))
    args = list(profile["engine_args"])
    for flag, value in (("--batch", "0"), ("--batch-groups", "1"), ("--prompt-cache", "0")):
        args[args.index(flag) + 1] = value
    for flag in ("--batch-mtp", "--batch-mtp-pipeline"):
        args.remove(flag)
    mtp = args.index("--mtp")
    del args[mtp:mtp + 2]
    # Separate storage reserve: recorded as a placement change, not merely hygiene.
    args += ["--no-prefill-borrow", "--pipeline-windows", "0", "--serve"]
    profile.update(binary=str(Path(binary).resolve()), frontend=str(Path(frontend).resolve()),
                   timeout_s=timeout_s, engine_args=args)
    differences = {key: {"daily": daily.get(key), "serial": value}
                   for key, value in profile.items() if daily.get(key) != value}
    return profile, differences


class PromptCompiler:
    def __init__(self, tokenizer_dir):
        self.path = Path(tokenizer_dir).resolve()
        template = self.path / "chat_template.jinja"
        if not template.is_file():
            raise ValueError("pack chat template missing; fallback is refused")
        self.template = ChatTemplate(template)
        vocab = load(self.path / "vocab.json")
        tokens = [None] * len(vocab)
        for text, token in vocab.items():
            if type(token) is not int or not 0 <= token < len(tokens) or tokens[token] is not None:
                raise ValueError("pack vocabulary IDs must be unique and contiguous")
            tokens[token] = text
        if any(text is None for text in tokens):
            raise ValueError("pack vocabulary has a hole")
        self.tok = Tokenizer(tokens, (self.path / "merges.txt").read_text().split("\n"),
                             load(self.path / "token_type.json"))
        self.literals = literal_tags(self.tok.control_tokens)
        self.identities = {name: digest(self.path / name) for name in
                           ("chat_template.jinja", "vocab.json", "merges.txt", "token_type.json", "tokenizer.json")}

    def compile(self, branch_id, messages, selected_token_ids):
        if not isinstance(branch_id, str) or not branch_id:
            raise ValueError("branch identity is required")
        if not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or
                not isinstance(m.get("content"), str) or set(m) != {"role", "content"} for m in messages):
            raise ValueError("reference messages must be explicit text-only role/content objects")
        marked, _, changed = mark_think_literals(messages, None, self.literals)
        rendered = self.template.render(marked, enable_thinking=False, add_generation_prompt=True)
        plain = None
        if changed:
            rendered, plain = unmark_think_literals(rendered, self.literals)
        ids = self.tok.encode(rendered, parse_special=True, plain=plain)
        validate_ids(selected_token_ids, len(self.tok.tokens))
        return {"schema_version": 1, "branch_id": branch_id, "messages": messages,
                "rendered_prompt": rendered, "prompt_token_ids": ids,
                "token_digest": token_digest(ids), "selected_token_ids": selected_token_ids,
                "prompt_contract": {"template": str(self.path / "chat_template.jinja"),
                                    "template_sha256": self.identities["chat_template.jinja"],
                                    "enable_thinking": False, "add_generation_prompt": True,
                                    "reasoning_effort_injection": False, "effort_at_end": False,
                                    "template_fallback": False}, "tokenizer_sha256": self.identities}

    def single_token(self, text):
        ids = self.tok.encode(text, parse_special=False)
        if len(ids) != 1:
            raise ValueError(f"declared marker {text!r} is not exactly one token")
        return ids[0]


def validate_ids(ids, vocab=None):
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 or i > 2147483647 or
            (vocab is not None and i >= vocab) for i in ids):
        raise ValueError("complete nonempty in-vocabulary token ID set required")


def validate_readout(result, request):
    n = len(request["prompt_token_ids"])
    if (result.get("schema_version") != 1 or result.get("mode") != "fresh" or
            result.get("engine_mode") != "serial_solo" or result.get("position") != n or
            result.get("consumed_prompt_tokens") != n or result.get("row_position") != n - 1 or
            result.get("prefix_tokens") != 0 or result.get("generated_tokens") != 0 or
            result.get("sampling") is not False or result.get("mtp") is not False or
            result.get("selected_token_ids") != request["selected_token_ids"]):
        raise ValueError("reference identity, read position or complete selected coverage mismatch")
    values = result.get("logits")
    log_probs = result.get("log_probs")
    lse = result.get("vocab_logsumexp")
    if (not isinstance(values, list) or len(values) != len(request["selected_token_ids"]) or
            not isinstance(log_probs, list) or len(log_probs) != len(values) or
            not isinstance(lse, (int, float)) or not math.isfinite(lse) or any(
                not isinstance(x, (int, float)) or not math.isfinite(x) for x in values + log_probs)):
        raise ValueError("missing or nonfinite complete logits/log-probability vector")
    validate_ids(result["selected_token_ids"], result["n_vocab"])
    for value, lp in zip(values, log_probs):
        if not math.isclose(value - lse, lp, abs_tol=1e-5):
            raise ValueError("readout log probability differs from unscaled logit minus full-vocabulary LSE")


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    source: dict[str, object] = {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
              "base_commit": "32e14d4febdd026b5f50901c8c76b5be0b128ac5"}
    changes = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"])
    source["dirty_diff_sha256"] = hashlib.sha256(changes).hexdigest()
    source["capture_source_sha256"] = {str(path.relative_to(ROOT)): digest(path) for path in
        (ROOT / "src/program/generate.cpp", ROOT / "include/strata/program/readout_reference.hpp",
         Path(__file__).resolve())}
    return source


def capture(profile, request, output_dir, check_safety=None, model_identity=None):
    """Token-only seam used by the shared backend's fresh reference mode."""
    if os.environ.get("STRATA_SUPERVISED") != "1" or not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise ValueError("capture requires an approved inherited supervised MPS window")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(profile["gpu_uuids"]):
        raise ValueError("inherited physical GPU selection differs from the serial profile")
    validate_ids(request["prompt_token_ids"])
    validate_ids(request["selected_token_ids"])
    if request.get("token_digest") != token_digest(request["prompt_token_ids"]):
        raise ValueError("compiled prompt token digest mismatch")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "request.json", request)
    tokens = output / "prompt.tokens"
    with tokens.open("x") as stream:
        stream.write(",".join(map(str, request["prompt_token_ids"])) + "\n")
    raw = output / "readout.json"
    argv = [profile["binary"], *profile["engine_args"], "--tokens-file", str(tokens),
            "--score-reference", str(raw), "--score-select", ",".join(map(str, request["selected_token_ids"]))]
    binary_hash = digest(profile["binary"])
    source = source_identity()
    began = time.monotonic()
    process = None
    try:
        if check_safety:
            check_safety()
        with (output / "engine.stdout").open("xb") as stdout, (output / "engine.stderr").open("xb") as stderr:
            environment = os.environ.copy()
            environment.update(profile.get("env", {}))
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       cwd=profile["frontend"], env=environment)
            starttime = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
            incarnation = {"id": str(uuid.uuid4()), "pid": process.pid, "proc_start_ticks": starttime,
                           "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            while process.poll() is None:
                if time.monotonic() - began >= profile["timeout_s"]:
                    raise RuntimeError("reference experiment deadline exceeded")
                if check_safety:
                    check_safety()
                time.sleep(1)
        if process.returncode:
            raise RuntimeError(f"reference engine exited {process.returncode}")
        result = load(raw)
        validate_readout(result, request)
        lines = (output / "engine.stdout").read_text().splitlines()
        if any(line.split(" ", 1)[0] in ("T", "BT", "DONE", "BDONE") for line in lines):
            raise ValueError("generation channel was emitted during no-decode capture")
        receipt = {**result, "branch_identity": {"id": request["branch_id"], "token_digest": request["token_digest"]},
                   "token_digest": request["token_digest"], "engine_incarnation": incarnation,
                   "source": {**source, "model": profile["name"]},
                   "binary": {"path": profile["binary"], "sha256": binary_hash},
                   "profile_sha256": hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
                   "model_identity": model_identity,
                   "environment": {key: value for key, value in environment.items()
                                   if key.startswith(("STRATA_", "CUDA_", "LD_"))},
                   "prompt_contract": request.get("prompt_contract"), "request_sha256": digest(output / "request.json"),
                   "readout_sha256": digest(raw), "argv": argv, "elapsed_s": time.monotonic() - began,
                   "exit_code": process.returncode}
        save(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        save(output / "failure.json", {"error": str(error), "argv": argv, "elapsed_s": time.monotonic() - began})
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def corpus_digest(corpus):
    payload = {key: value for key, value in corpus.items() if key != "freeze_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_corpus(corpus):
    if corpus.get("schema_version") != 1 or corpus.get("issue") != 224 or corpus.get("fold") != "development":
        raise ValueError("issue-224 frozen development corpus required")
    if corpus.get("label_provenance") != "independent_human" or not corpus.get("freeze_sha256"):
        raise ValueError("independent human label provenance and pre-outcome freeze binding required")
    if corpus["freeze_sha256"] != corpus_digest(corpus):
        raise ValueError("development corpus or predeclared gates changed after freeze")
    groups, anchors = corpus.get("groups"), corpus.get("easy_anchors")
    if not isinstance(groups, list) or len(groups) != 200 or len({g["group_id"] for g in groups}) != 200:
        raise ValueError("the full 200 independent development groups are required")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("predeclared easy anchors required")
    gate = corpus.get("gate", {})
    if any(type(gate.get(k)) not in (int, float) or not 0 <= gate[k] <= 1 for k in
           ("minimum_group_accuracy", "minimum_anchor_accuracy")):
        raise ValueError("accuracy gates must be frozen before candidate outcomes")
    items = []
    for group in groups:
        if not isinstance(group["group_id"], str) or not group["group_id"] or group["group_id"] == "easy_anchor":
            raise ValueError("independent group identities must be explicit and nonreserved")
        if not isinstance(group.get("items"), list) or not group["items"]:
            raise ValueError("each source-state group must contain its complete typed items")
        items += [(group["group_id"], item) for item in group["items"]]
    items += [("easy_anchor", item) for item in anchors]
    if len({item["id"] for _, item in items}) != len(items):
        raise ValueError("duplicate branch identity")
    for group, item in items:
        if (set(item) != {"schema_version", "id", "group_id", "request", "gold"} or
                item["schema_version"] != 1 or item["group_id"] != group or
                not isinstance(item["id"], str) or not item["id"] or
                not isinstance(item["request"], dict) or not isinstance(item["gold"], dict)):
            raise ValueError("shared evaluator's complete typed request/gold case is required")
    return items


def screen_jobs(corpus, scoring, bundles):
    """Compile through #223's sole method; this module does not author screen prompts."""
    jobs = []
    for family in FAMILIES:
        bundle = bundles[family]
        if bundle.identity["readout_family"] != family or bundle.temperature != 1.0:
            raise ValueError("the categorical screen needs both pinned raw T=1 readout families")
        for group, item in validate_corpus(corpus):
            normalized = scoring.normalize_request(bundle, item["request"])
            compiled = scoring.compile_request(bundle, normalized)
            if set(item["gold"]) != {q.question.id for q in compiled.questions}:
                raise ValueError("gold must cover every typed question, never an inner join")
            for question in compiled.questions:
                options = [o.id for o in question.question.options]
                if item["gold"][question.question.id] not in options:
                    raise ValueError("independent gold is outside the complete declared option set")
                for plan in question.plans:
                    branch = hashlib.sha256(json.dumps(
                        [family, item["id"], plan.branch_id, plan.plan_sha256]).encode()).hexdigest()
                    jobs.append({"schema_version": 1, "branch_id": branch, "family": family,
                        "group_id": group, "item_id": item["id"], "question_id": question.question.id,
                        "option_ids": options, "plan_branch_id": plan.branch_id,
                        "plan_sha256": plan.plan_sha256, "bundle_sha256": compiled.bundle_sha256,
                        "rendered_prompt": plan.prompt, "literal_spans": [list(span) for span in plan.literal_spans],
                        "prompt_token_ids": list(plan.prompt_token_ids),
                        "selected_token_ids": list(plan.selected_token_ids),
                        "token_digest": token_digest(plan.prompt_token_ids),
                        "prompt_contract": {"template_sha256": bundle.identity["artifact_sha256"]["chat_template"],
                            "enable_thinking": False, "add_generation_prompt": True, "reasoning_effort_injection": False,
                            "effort_at_end": False, "template_fallback": False}})
    return jobs

def load_screen_method(shared_root, bundle_set):
    import importlib
    root = Path(shared_root).resolve()
    source = root / "tools/judge_scoring.py"
    if bundle_set.get("schema_version") != 1 or digest(source) != bundle_set.get("shared_code_sha256"):
        raise ValueError("shared scoring implementation differs from the external pin")
    sys.path.insert(0, str(root))
    scoring = importlib.import_module("tools.judge_scoring")
    if Path(scoring.__file__).resolve() != source:
        raise ValueError("a different shared compiler is already imported")
    declarations = bundle_set.get("families", {})
    if set(declarations) != set(FAMILIES):
        raise ValueError("both complete pinned family bundles are required")
    bundles = {family: scoring.load_scoring_bundle(declarations[family]["path"],
               expected_sha256=declarations[family]["sha256"]) for family in FAMILIES}
    for bundle in bundles.values():
        if (bundle.temperature != 1.0 or bundle.identity["artifact_sha256"]["chat_template"] !=
                "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"):
            raise ValueError("screen requires raw T=1 and the frozen IQ3_S pack template")
    return scoring, bundles



def softmax(values):
    maximum = max(values)
    masses = [math.exp(value - maximum) for value in values]
    denominator = sum(masses)
    return [value / denominator for value in masses]


def evaluate_screen(corpus, jobs, receipts, scoring, bundles):
    items = validate_corpus(corpus)
    # Recompile via the same shared method before accepting retained branches.
    if jobs != screen_jobs(corpus, scoring, bundles):
        raise ValueError("missing, reordered or mutated shared declared-option token plans")
    by_branch = {}
    for receipt in receipts:
        key = receipt["branch_identity"]["id"]
        if key in by_branch:
            raise ValueError("duplicate reference receipt")
        by_branch[key] = receipt
    if set(by_branch) != {job["branch_id"] for job in jobs}:
        raise ValueError("missing or extra branches: full screen denominator required")
    for job in jobs:
        receipt = by_branch[job["branch_id"]]
        validate_readout(receipt, job)
        if (receipt["token_digest"] != job["token_digest"] or
                receipt["branch_identity"].get("token_digest") != job["token_digest"] or
                not receipt.get("engine_incarnation") or not receipt.get("source")):
            raise ValueError("unbound reference receipt")
    families = {}
    for family in FAMILIES:
        outcomes, grouped = [], {}
        for group, item in items:
            table = {job["plan_sha256"]: job for job in jobs
                     if job["family"] == family and job["item_id"] == item["id"]}
            used = set()
            class RetainedReference:
                def validate_identity(self, identity):
                    for job in table.values():
                        receipt = by_branch[job["branch_id"]]
                        if (receipt["source"].get("model") != identity["model"] or
                                job["bundle_sha256"] != identity["bundle_sha256"]):
                            raise ValueError("retained source/model or shared bundle identity mismatch")
                def score(self, plan):
                    if plan.plan_sha256 not in table or plan.plan_sha256 in used:
                        raise ValueError("missing or duplicate canonical shared branch")
                    job = table[plan.plan_sha256]
                    used.add(plan.plan_sha256)
                    receipt = by_branch[job["branch_id"]]
                    return scoring.LogitRow(plan.plan_sha256, len(plan.prompt_token_ids),
                        plan.selected_token_ids, tuple(receipt["logits"]), 0)
            # Gold stays outside the shared compiler/backend. Only score_call
            # combines direct logits or yes/no odds and normalises the options.
            call = scoring.score_call(bundles[family], item["request"], RetainedReference())
            if used != set(table):
                raise ValueError("unused reference branches cannot disappear from the denominator")
            for question, readout in zip(call.compiled.questions, call.readouts):
                qid = question.question.id
                tied = sum(p == max(readout.probabilities) for p in readout.probabilities) != 1
                correct = not tied and readout.selected_id == item["gold"][qid]
                outcomes.append({"group_id": group, "item_id": item["id"], "question_id": qid,
                    "option_ids": list(readout.option_ids), "probabilities": list(readout.probabilities),
                    "selected_option_id": readout.selected_id, "gold_option_id": item["gold"][qid],
                    "correct": correct, "tied": tied})
                grouped.setdefault(group, []).append(correct)
        anchor_accuracy = sum(grouped["easy_anchor"]) / len(grouped["easy_anchor"])
        group_accuracy = sum(all(values) for group, values in grouped.items() if group != "easy_anchor") / 200
        passed = (anchor_accuracy >= corpus["gate"]["minimum_anchor_accuracy"] and
                  group_accuracy >= corpus["gate"]["minimum_group_accuracy"])
        families[family] = {"pass": passed, "group_accuracy": group_accuracy,
                            "anchor_accuracy": anchor_accuracy, "groups": 200, "outcomes": outcomes}
    passed = [family for family in FAMILIES if families[family]["pass"]]
    selected = max(passed, key=lambda f: (families[f]["group_accuracy"], families[f]["anchor_accuracy"], -FAMILIES.index(f))) if passed else None
    return {"schema_version": 1, "disposition": "keep" if selected else "reject", "family_selection": selected,
            "stop_program": not bool(selected), "corpus_freeze_sha256": corpus["freeze_sha256"], "families": families,
            "shared_bundles": {family: bundles[family].identity for family in FAMILIES},
            "calibration": "not applied; categorical-order gate only (pinned T=1)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("profile")
    p.add_argument("--daily", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--differences", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--request", required=True, help="JSON branch_id, messages, selected_token_ids")
    p.add_argument("--out", required=True)
    p = commands.add_parser("capture")
    p.add_argument("--profile", required=True)
    p.add_argument("--request", required=True, help="already compiled request (never retokenized)")
    p.add_argument("--out-dir", required=True)
    p = commands.add_parser("prepare-screen")
    p.add_argument("--shared-root", required=True)
    p.add_argument("--bundle-set", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("screen")
    p.add_argument("--shared-root", required=True)
    p.add_argument("--bundle-set", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--receipts", required=True, help="directory containing reference receipt.json files")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "profile":
        daily = load(args.daily)
        profile, differences = serial_profile(daily, args.binary, ROOT)
        save(args.out, profile)
        save(args.differences, {"daily_file_sha256": digest(args.daily), "differences": differences})
    elif args.command == "prepare":
        request = load(args.request)
        save(args.out, PromptCompiler(args.tokenizer).compile(request["branch_id"], request["messages"], request["selected_token_ids"]))
    elif args.command == "capture":
        capture(load(args.profile), load(args.request), args.out_dir)
    elif args.command == "prepare-screen":
        scoring, bundles = load_screen_method(args.shared_root, load(args.bundle_set))
        save(args.out, screen_jobs(load(args.corpus), scoring, bundles))
    else:
        scoring, bundles = load_screen_method(args.shared_root, load(args.bundle_set))
        result = evaluate_screen(load(args.corpus), load(args.jobs),
            [load(path) for path in Path(args.receipts).rglob("receipt.json")], scoring, bundles)
        save(args.out, result)
        return 0 if result["disposition"] == "keep" else 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"reference refused: {error}", file=sys.stderr)
        sys.exit(2)
