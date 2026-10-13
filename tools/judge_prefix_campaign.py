#!/usr/bin/env python3
"""#234 prospective full-denominator prefix-reuse declaration and evaluation.

No model readout, corpus generation, labels, calibration or GPU launch lives here.
Probabilities are raw T=1 softmax of the complete submitted selected-logit vector.
Missing independent source groups or a maximum-coexistence leg are blocks, not
fresh-path evidence for reuse. CPU fixture results never qualify a workload.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import judge_reference as ref

GROUPS = 3000
MODES = ("idle", "maximum_coexistence")
LOADS = {"idle": {"generation_slots": 0, "concurrent_submitted_judgments": 1},
         "maximum_coexistence": {"generation_slots": 4, "concurrent_submitted_judgments": 2}}
CRITERIA = {"groups_per_workload_per_mode": GROUPS, "confidence": .95, "upper_failure_rate": .01,
            "primary_failure": {"argmax_change": True, "max_difference_gt": .02, "tv_gt": .05},
            "all_rows": {"max_difference_le": .05, "max_tv_le": .10, "p95_tv_le": .02,
                         "missing_nonfinite_wrong_position_identity_contamination_generated_answer": 0},
            "fresh_fresh_control": True, "probability_basis": "complete selected logits, raw softmax T=1",
            "p95_definition": "nearest rank ceil(0.95*N)", "no_parallel_claim": True,
            "timing": "tokenisation + queueing + whole call, including checkpoint and restore"}


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("campaign timestamps require a timezone")
    return parsed


def validate_groups(groups):
    if not isinstance(groups, list) or not groups:
        raise ValueError("independent source-state groups are absent")
    counts, identities, sources = Counter(), set(), set()
    for group in groups:
        workload, identity = group.get("workload"), group.get("group_id")
        if not isinstance(workload, str) or not workload or not isinstance(identity, str) or not identity:
            raise ValueError("explicit workload and source-group identities are required")
        key = (workload, identity)
        source = (workload, group.get("source_sha256"))
        if key in identities or not sha(source[1]) or source in sources:
            raise ValueError("duplicate group/source hash cannot count as an independent source")
        identities.add(key); sources.add(source); counts[workload] += 1
        requests = group.get("branches")
        prefix = ref.common_prefix(requests)
        primary = group.get("primary_branch_id")
        if primary not in {request["branch_id"] for request in requests}:
            raise ValueError("one primary branch must be preselected in every source group")
        if "prefix_tokens" in group and group["prefix_tokens"] != prefix:
            raise ValueError("declared prefix differs from the identical full token plans")
        for request in requests:
            ref.validate_ids(request["selected_token_ids"])
            if any(key in request for key in ("logits", "probabilities", "prediction", "selected_id")):
                raise ValueError("candidate outcomes cannot enter a prospective declaration")
            prompt = request.get("prompt_contract", {})
            if (prompt.get("enable_thinking") is not False or prompt.get("add_generation_prompt") is not True or
                    prompt.get("template_fallback") is not False or prompt.get("reasoning_effort_injection") is not False or
                    prompt.get("effort_at_end") is not False or not sha(prompt.get("template_sha256")) or
                    not request.get("tokenizer_sha256") or any(not sha(value) for value in request["tokenizer_sha256"].values())):
                raise ValueError("compiled plans require the frozen pack prompt/tokenizer identity")
    if any(count != GROUPS for count in counts.values()):
        raise ValueError(f"exactly {GROUPS} independent source groups are required per workload: {dict(counts)}")
    return counts


def declaration(corpus, binary, corpus_sha256):
    if (corpus.get("schema_version") != 1 or corpus.get("fixture_only") or
            corpus.get("source_provenance") != "independent_human" or corpus.get("outcome_blind_selection") is not True or
            not sha(corpus.get("source_snapshot_sha256"))):
        raise ValueError("outcome-blind independently acquired/human-labelled frozen corpus required; no synthetic qualification")
    counts = validate_groups(corpus.get("groups"))
    traces = corpus.get("arrival_trace_sha256", {})
    if set(traces) != set(MODES) or any(not sha(traces[mode]) for mode in MODES):
        raise ValueError("idle and maximum-coexistence arrival traces must be frozen before outcomes")
    groups = [{**group, "prefix_tokens": ref.common_prefix(group["branches"])} for group in corpus["groups"]]
    return {"schema_version": 1, "issue": 234, "disposition": "declared",
            "fixed_utc": datetime.now(timezone.utc).isoformat(), "criteria": CRITERIA,
            "corpus_sha256": corpus_sha256, "source_snapshot_sha256": corpus["source_snapshot_sha256"],
            "source_provenance": corpus["source_provenance"], "outcome_blind_selection": True,
            "source": ref.source_identity(), "binary": {"path": str(Path(binary).resolve()), "sha256": ref.digest(binary)},
            "modes": list(MODES), "loads": LOADS, "arrival_trace_sha256": traces,
            "workloads": dict(counts), "groups": groups,
            "expected_rows": sum(len(group["branches"]) for group in groups) * len(MODES)}


def binomial_cdf(k, n, p):
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k == n)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) +
             i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(value - maximum) for value in terms)


def exact_upper(failures, groups, confidence=.95):
    """One-sided Clopper-Pearson bound: invert P_p[X<=failures]=1-confidence."""
    if type(groups) is not int or groups < 1 or type(failures) is not int or not 0 <= failures <= groups or not 0 < confidence < 1:
        raise ValueError("valid full binomial denominator and confidence required")
    if failures == groups:
        return 1.0
    if failures == 0:
        return -math.expm1(math.log1p(-confidence) / groups)
    lower, upper = failures / groups, 1.0
    for _ in range(70):
        mid = (lower + upper) / 2
        if binomial_cdf(failures, groups, mid) > 1 - confidence:
            lower = mid
        else:
            upper = mid
    return upper


def pair(a, b):
    p, q = ref.softmax(a["logits"]), ref.softmax(b["logits"])
    maximum = max(abs(x - y) for x, y in zip(p, q))
    tv = sum(abs(x - y) for x, y in zip(p, q)) / 2
    argmax = p.index(max(p)) != q.index(max(q))
    return {"argmax_change": argmax, "max_difference": maximum, "tv": tv,
            "primary_failure": argmax or maximum > .02 or tv > .05}


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summary(rows):
    tv = [row["tv"] for row in rows]
    delta = [row["max_difference"] for row in rows]
    result = {"rows": len(rows), "argmax_changes": sum(row["argmax_change"] for row in rows),
              "primary_drift_failures_all_rows": sum(row["primary_failure"] for row in rows),
              "max_difference": max(delta), "max_tv": max(tv), "p50_tv": quantile(tv, .5),
              "p95_tv": quantile(tv, .95), "p99_tv": quantile(tv, .99)}
    result["hard_gates_pass"] = result["max_difference"] <= .05 and result["max_tv"] <= .10 and result["p95_tv"] <= .02
    return result


def validate_receipt(receipt, request, declared, reused, prefix):
    ref.validate_readout(receipt, request, mode="reused" if reused else "fresh", prefix_tokens=prefix if reused else 0)
    if (receipt.get("execution_leg") != "serial_GEN" or receipt.get("token_digest") != request["token_digest"] or
            receipt.get("branch_identity") != {"id": request["branch_id"], "token_digest": request["token_digest"]} or
            receipt.get("prompt_contract") != request["prompt_contract"] or not receipt.get("engine_incarnation") or
            receipt.get("binary", {}).get("sha256") != declared["binary"]["sha256"] or
            not receipt.get("model_identity") or not receipt.get("profile_sha256") or
            not sha(receipt.get("request_sha256")) or not sha(receipt.get("readout_sha256"))):
        raise ValueError("identity contamination or absent prompt/model/binary/engine binding")
    source = receipt.get("source", {})
    if any(source.get(key) != value for key, value in declared["source"].items()):
        raise ValueError("receipt source differs from the predeclared compiled candidate")


def evaluate(declared, manifest, expected_declaration_sha=None):
    result = {"schema_version": 1, "issue": 234, "disposition": "blocked", "criteria": CRITERIA,
              "no_parallel_claim": True, "blockers": []}
    try:
        if (declared.get("schema_version") != 1 or declared.get("issue") != 234 or declared.get("disposition") != "declared" or
                declared.get("criteria") != CRITERIA or declared.get("modes") != list(MODES) or declared.get("loads") != LOADS or
                declared.get("source_provenance") != "independent_human" or declared.get("outcome_blind_selection") is not True):
            raise ValueError("fixed prospective #234 declaration required")
        counts = validate_groups(declared.get("groups"))
        if declared.get("workloads") != dict(counts):
            raise ValueError("workload source denominator differs from the declaration")
        wanted_sha = expected_declaration_sha or content_digest(declared)
        if (manifest.get("schema_version") != 1 or manifest.get("declaration_sha256") != wanted_sha or
                utc(manifest.get("started_utc", "")) < utc(declared["fixed_utc"])):
            raise ValueError("results are not bound to a declaration fixed before evaluation")
        result["declaration_sha256"] = wanted_sha
        expected = {}
        for group in declared["groups"]:
            for mode in MODES:
                for request in group["branches"]:
                    key = (group["workload"], group["group_id"], request["branch_id"], mode)
                    expected[key] = (group, request)
        rows = manifest.get("rows")
        result["expected_rows"] = len(expected)
        if not isinstance(rows, list) or len(rows) != len(expected) or declared.get("expected_rows") != len(expected):
            raise ValueError("full campaign denominator is missing or has extra rows")
        seen, drift, controls, primaries, timings = set(), [], [], defaultdict(list), defaultdict(list)
        for row in rows:
            key = (row.get("workload"), row.get("group_id"), row.get("branch_id"), row.get("mode"))
            if key not in expected or key in seen:
                raise ValueError("duplicate, undeclared or wrong-mode result row")
            seen.add(key)
            group, request = expected[key]
            if (row.get("observed_load") != LOADS[key[3]] or
                    row.get("arrival_trace_sha256") != declared["arrival_trace_sha256"][key[3]]):
                raise ValueError("maximum-coexistence or idle load/arrival evidence differs")
            a, b, control = row["fresh"], row["reused"], row["fresh_control"]
            for receipt, reused in ((a, False), (b, True), (control, False)):
                validate_receipt(receipt, request, declared, reused, group["prefix_tokens"])
            for identity in ("model_identity", "profile_sha256", "n_vocab"):
                if not a[identity] == b[identity] == control[identity]:
                    raise ValueError("paired reference identities differ")
            timing = row.get("timing", {})
            for name, receipt in (("fresh", a), ("reused", b), ("fresh_control", control)):
                interval = timing.get(name, {})
                values = [interval.get(field) for field in ("tokenisation_s", "queueing_s", "call_s")]
                if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError("complete tokenisation, queueing and whole-call timing is absent")
                if name == "reused" and (interval.get("includes_checkpoint_restore") is not True or
                        1000 * values[2] + .001 < receipt["checkpoint_ms"] + receipt["restore_ms"]):
                    raise ValueError("forward-only timing cannot claim checkpoint/restore latency")
                timings[name].append(sum(values))
            observed, baseline = pair(a, b), pair(a, control)
            metadata = {"workload": key[0], "group_id": key[1], "branch_id": key[2], "mode": key[3]}
            drift.append({**metadata, **observed}); controls.append({**metadata, **baseline})
            if key[2] == group["primary_branch_id"]:
                primaries[(key[0], key[3], "fresh_reused")].append(observed["primary_failure"])
                primaries[(key[0], key[3], "fresh_fresh")].append(baseline["primary_failure"])
        if seen != set(expected):
            raise ValueError("full campaign denominator is missing results")
        gates = []
        for workload in counts:
            for mode in MODES:
                for comparison in ("fresh_reused", "fresh_fresh"):
                    outcomes = primaries[(workload, mode, comparison)]
                    if len(outcomes) != GROUPS:
                        raise ValueError("preselected primary denominator is not 3000 independent groups")
                    failures = sum(outcomes)
                    upper = exact_upper(failures, GROUPS)
                    gates.append({"workload": workload, "mode": mode, "comparison": comparison, "groups": GROUPS,
                                  "failures": failures, "one_sided_exact_95_upper": upper, "pass": upper <= .01})
        result.update(valid_rows=len(rows), primary_gates=gates, fresh_reused=summary(drift), fresh_fresh=summary(controls),
                      drift_rows=drift, control_rows=controls,
                      timing={name: {"p50_total_s": quantile(values, .5), "p95_total_s": quantile(values, .95),
                                     "max_total_s": max(values)} for name, values in timings.items()})
        result["disposition"] = "keep" if all(gate["pass"] for gate in gates) and result["fresh_reused"]["hard_gates_pass"] and result["fresh_fresh"]["hard_gates_pass"] else "reject"
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        result["blockers"].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declare = commands.add_parser("declare")
    declare.add_argument("--corpus", required=True); declare.add_argument("--binary", required=True)
    declare.add_argument("--out", required=True)
    assess = commands.add_parser("evaluate")
    assess.add_argument("--declaration", required=True); assess.add_argument("--manifest", required=True)
    assess.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "declare":
            value = declaration(ref.load(args.corpus), args.binary, ref.digest(args.corpus))
        else:
            value = evaluate(ref.load(args.declaration), ref.load(args.manifest), ref.digest(args.declaration))
    except (OSError, ValueError, TypeError, KeyError) as error:
        value = {"schema_version": 1, "issue": 234, "disposition": "blocked", "criteria": CRITERIA,
                 "fixed_utc": datetime.now(timezone.utc).isoformat(), "blockers": [str(error)],
                 "no_parallel_claim": True}
    ref.save(args.out, value)
    print(json.dumps({"disposition": value["disposition"], "blockers": value.get("blockers", []), "out": args.out}))
    return 0 if value["disposition"] in ("declared", "keep") else 1


if __name__ == "__main__":
    raise SystemExit(main())
