#!/usr/bin/env python3
"""CPU-only integration with #223's real shared scorer and its synthetic fixtures.

No weights, engine launches or human gold. Explicit project checkout is required
because the shared method lives in ai-optimise, not the public engine repository.
"""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import tempfile

import judge_reference as ref


def run(shared_root):
    root = Path(shared_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    scoring = importlib.import_module("tools.judge_scoring")
    fixtures = importlib.import_module("judge_fixtures")
    if Path(scoring.__file__).resolve() != root / "tools/judge_scoring.py":
        raise ValueError("wrong shared module loaded")
    with tempfile.TemporaryDirectory(prefix="strata-shared-screen-test-") as temporary:
        bundles = {}
        for family in ref.FAMILIES:
            manifest, sha = fixtures.make_bundle(Path(temporary) / family, family, temperature=1.0)
            bundles[family] = scoring.load_scoring_bundle(manifest, expected_sha256=sha)
        def item(name, group):
            return {"schema_version": 1, "id": name, "group_id": group, "request": fixtures.request(),
                    "gold": {"pick": "two", "truth": "true", "level": "1"}}
        corpus = {"schema_version": 1, "issue": 224, "fold": "development",
                  "label_provenance": "independent_human", "fixture_only": True,
                  "groups": [{"group_id": str(i), "items": [item(str(i), str(i))]} for i in range(200)],
                  "easy_anchors": [item("anchor", "easy_anchor")],
                  "gate": {"minimum_group_accuracy": 1.0, "minimum_anchor_accuracy": 1.0}}
        corpus["freeze_sha256"] = ref.corpus_digest(corpus)
        jobs = ref.screen_jobs(corpus, scoring, bundles)
        receipts = []
        for job in jobs:
            qid, branch = job["question_id"], job["plan_branch_id"]
            correct = {"pick": 1, "truth": 0, "level": 1}[qid]
            if job["family"] == ref.FAMILIES[0]:
                logits = [0.0] * len(job["option_ids"])
                logits[correct] = 5.0
            else:
                candidate = int(branch.rsplit(":", 1)[1])
                logits = [5.0, 0.0] if candidate == correct else [0.0, 5.0]
            lse = max(logits) + math.log(sum(math.exp(x - max(logits)) for x in logits))
            n = len(job["prompt_token_ids"])
            receipts.append({"schema_version": 1, "mode": "fresh", "engine_mode": "serial_solo",
                "position": n, "consumed_prompt_tokens": n, "row_position": n - 1, "prefix_tokens": 0,
                "generated_tokens": 0, "sampling": False, "mtp": False, "n_vocab": 1000,
                "selected_token_ids": job["selected_token_ids"], "logits": logits,
                "vocab_logsumexp": lse, "log_probs": [x - lse for x in logits],
                "token_digest": job["token_digest"],
                "branch_identity": {"id": job["branch_id"], "token_digest": job["token_digest"]},
                "engine_incarnation": {"id": "CPU fixture only"},
                "source": {"commit": "CPU fixture only", "model": "fixture-model"}})
        result = ref.evaluate_screen(corpus, jobs, receipts, scoring, bundles)
        assert result["disposition"] == "keep" and result["family_selection"] == ref.FAMILIES[0]
        assert all(family["group_accuracy"] == 1.0 and len(family["outcomes"]) == 603
                   for family in result["families"].values())
        try:
            ref.evaluate_screen(corpus, jobs, receipts[:-1], scoring, bundles)
        except ValueError as error:
            assert "denominator" in str(error)
        else:
            raise AssertionError("missing branch was accepted")
        wrong_model = list(receipts)
        wrong_model[0] = {**wrong_model[0], "source": {"commit": "CPU fixture only", "model": "different-model"}}
        try:
            ref.evaluate_screen(corpus, jobs, wrong_model, scoring, bundles)
        except ValueError as error:
            assert "identity mismatch" in str(error)
        else:
            raise AssertionError("retained model identity skew was accepted")
        return {"schema_version": 1, "test": "real shared compiler/readout + retained-reference CPU integration",
                "disposition": "pass", "shared_scoring_sha256": ref.digest(root / "tools/judge_scoring.py"),
                "jobs": len(jobs), "typed_question_outcomes_per_family": 603,
                "covers": ["Choice, Noul and Score use real normalize_request/compile_request/score_call",
                           "both readout families", "raw T=1", "source/model identity attestation",
                           "complete declared option vectors", "missing-branch denominator refusal",
                           "retained model identity skew refusal"],
                "evidence_boundary": "Synthetic CPU fixtures only; zero engine launches, no real semantic screen or family selection."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.shared_root)
    ref.save(args.out, result)
    print(json.dumps(result))
