"""Synthetic CPU fixtures only: none of these groups or labels is qualification evidence."""
import copy
import hashlib
import math
import unittest

import judge_prefix_campaign as campaign
import judge_reference as ref

HASH = "a" * 64
PROMPT = {"enable_thinking": False, "add_generation_prompt": True, "template_fallback": False,
          "reasoning_effort_injection": False, "effort_at_end": False, "template_sha256": HASH}


def fixture_group(index):
    def request(branch, last):
        tokens = [index + 10, 1, last]
        return {"branch_id": branch, "prompt_token_ids": tokens, "token_digest": ref.token_digest(tokens),
                "selected_token_ids": [65, 66], "prompt_contract": PROMPT, "tokenizer_sha256": {"vocab.json": HASH}}
    return {"workload": "CPU-fixture-not-real-workload", "group_id": str(index),
            "source_sha256": hashlib.sha256(f"CPU-source-{index}".encode()).hexdigest(),
            "primary_branch_id": "A", "prefix_tokens": 2, "branches": [request("A", 2), request("B", 3)]}


def fixture_declaration():
    return {"schema_version": 1, "issue": 234, "disposition": "declared", "fixed_utc": "2026-10-13T00:00:00Z",
            "criteria": campaign.CRITERIA, "source_provenance": "independent_human", "outcome_blind_selection": True,
            "source_snapshot_sha256": HASH, "modes": list(campaign.MODES), "loads": campaign.LOADS,
            "arrival_trace_sha256": {mode: HASH for mode in campaign.MODES},
            "source": {"commit": "CPU-fixture"}, "binary": {"sha256": HASH},
            "workloads": {"CPU-fixture-not-real-workload": 3000},
            "groups": [fixture_group(index) for index in range(3000)], "expected_rows": 12000}


def receipt(request, reused, incarnation):
    values = [0.0001, 0.0]
    lse = max(values) + math.log(sum(math.exp(value - max(values)) for value in values))
    result = {"schema_version": 1, "mode": "reused" if reused else "fresh", "engine_mode": "serial_solo",
              "execution_leg": "serial_GEN", "position": 3, "consumed_prompt_tokens": 3, "row_position": 2,
              "prefix_tokens": 2 if reused else 0, "generated_tokens": 0, "sampling": False, "mtp": False,
              "n_vocab": 4096, "selected_token_ids": [65, 66], "logits": values,
              "log_probs": [value - lse for value in values], "vocab_logsumexp": lse,
              "token_digest": request["token_digest"],
              "branch_identity": {"id": request["branch_id"], "token_digest": request["token_digest"]},
              "prompt_contract": PROMPT, "engine_incarnation": {"id": incarnation},
              "binary": {"sha256": HASH}, "source": {"commit": "CPU-fixture"},
              "model_identity": {"sha256": HASH}, "profile_sha256": HASH, "request_sha256": HASH, "readout_sha256": HASH}
    if reused:
        result.update(restoration_verified=True, checkpoint_ms=1, restore_ms=1, call_elapsed_ms=3)
    return result


def fixture_manifest(declared):
    rows = []
    for group in declared["groups"]:
        for mode in campaign.MODES:
            for request in group["branches"]:
                prefix = f"{mode}/{group['group_id']}"
                rows.append({"workload": group["workload"], "group_id": group["group_id"], "branch_id": request["branch_id"],
                    "mode": mode, "observed_load": campaign.LOADS[mode], "arrival_trace_sha256": HASH,
                    "fresh": receipt(request, False, prefix + "/fresh"),
                    "reused": receipt(request, True, prefix + "/reused"),
                    "fresh_control": receipt(request, False, prefix + "/control"),
                    "timing": {name: {"tokenisation_s": .001, "queueing_s": .002, "call_s": .003,
                                      "includes_checkpoint_restore": True} for name in ("fresh", "reused", "fresh_control")}})
    return {"schema_version": 1, "declaration_sha256": campaign.content_digest(declared),
            "started_utc": "2026-10-13T00:01:00Z", "rows": rows}


def set_logits(row, values):
    lse = max(values) + math.log(sum(math.exp(value - max(values)) for value in values))
    row.update(logits=values, vocab_logsumexp=lse, log_probs=[value - lse for value in values])


class PrefixCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.declared = fixture_declaration()
        cls.manifest = fixture_manifest(cls.declared)

    def changed_row(self, index, **patch):
        manifest = {**self.manifest, "rows": list(self.manifest["rows"])}
        manifest["rows"][index] = {**manifest["rows"][index], **patch}
        return manifest

    def test_exact_one_sided_bound_and_primary_boundary(self):
        self.assertAlmostEqual(campaign.exact_upper(0, 1), .95)
        self.assertEqual(campaign.exact_upper(1, 1), 1)
        self.assertAlmostEqual(campaign.exact_upper(0, 3000), .000998079, places=9)
        self.assertLessEqual(campaign.exact_upper(20, 3000), .01)
        self.assertGreater(campaign.exact_upper(21, 3000), .01)
        with self.assertRaises(ValueError):
            campaign.exact_upper(0, 0)

    def test_complete_fixture_matrix_and_fresh_control(self):
        result = campaign.evaluate(self.declared, self.manifest)
        self.assertEqual(result["disposition"], "keep")
        self.assertEqual(result["valid_rows"], 12000)
        self.assertEqual(len(result["primary_gates"]), 4)
        self.assertTrue(all(gate["groups"] == 3000 and gate["failures"] == 0 for gate in result["primary_gates"]))
        self.assertEqual(result["timing"]["reused"]["p95_total_s"], .006)

    def test_missing_and_duplicate_rows_do_not_shrink_denominator(self):
        missing = {**self.manifest, "rows": self.manifest["rows"][:-1]}
        result = campaign.evaluate(self.declared, missing)
        self.assertEqual(result["disposition"], "blocked")
        self.assertIn("denominator", result["blockers"][0])
        duplicate = self.changed_row(1, **self.manifest["rows"][0])
        self.assertEqual(campaign.evaluate(self.declared, duplicate)["disposition"], "blocked")

    def test_source_variants_cannot_count_as_independent_groups(self):
        groups = list(self.declared["groups"])
        groups[1] = {**groups[1], "source_sha256": groups[0]["source_sha256"]}
        with self.assertRaisesRegex(ValueError, "independent"):
            campaign.validate_groups(groups)
        with self.assertRaisesRegex(ValueError, "3000"):
            campaign.validate_groups(self.declared["groups"][:1200])
        wrong_primary = list(self.declared["groups"])
        wrong_primary[0] = {**wrong_primary[0], "primary_branch_id": "not-declared"}
        with self.assertRaisesRegex(ValueError, "preselected"):
            campaign.validate_groups(wrong_primary)

    def test_no_synthetic_declaration_or_missing_corpus(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            campaign.declaration({"schema_version": 1, "fixture_only": True}, "/not-a-binary", HASH)
        with self.assertRaisesRegex(ValueError, "absent"):
            campaign.validate_groups([])

    def test_reuse_missing_state_position_finite_or_identity_is_blocked(self):
        original = self.manifest["rows"][0]["reused"]
        patches = ({"mode": "fresh"}, {"restoration_verified": False}, {"position": 2}, {"row_position": 1},
                   {"generated_tokens": 1}, {"logits": [float("nan"), 0]}, {"token_digest": "b" * 64},
                   {"binary": {"sha256": "b" * 64}}, {"source": {"commit": "wrong"}},
                   {"selected_token_ids": [65]}, {"execution_leg": "batch_slot"})
        for patch in patches:
            with self.subTest(patch=patch):
                result = campaign.evaluate(self.declared, self.changed_row(0, reused={**original, **patch}))
                self.assertEqual(result["disposition"], "blocked")

    def test_nonprimary_row_hard_gate_cannot_hide_behind_primary_pass(self):
        reused = copy.deepcopy(self.manifest["rows"][1]["reused"])
        set_logits(reused, [-4, 4])
        result = campaign.evaluate(self.declared, self.changed_row(1, reused=reused))
        self.assertEqual(result["disposition"], "reject")
        self.assertTrue(all(gate["pass"] for gate in result["primary_gates"]))
        self.assertFalse(result["fresh_reused"]["hard_gates_pass"])

    def test_argmax_primary_failures_reject_even_when_all_row_magnitudes_pass(self):
        manifest = {**self.manifest, "rows": list(self.manifest["rows"])}
        for index in range(21):
            offset = index * 4
            row = dict(manifest["rows"][offset])
            row["reused"] = copy.deepcopy(row["reused"])
            set_logits(row["reused"], [0, .0001])
            manifest["rows"][offset] = row
        result = campaign.evaluate(self.declared, manifest)
        self.assertEqual(result["disposition"], "reject")
        self.assertTrue(result["fresh_reused"]["hard_gates_pass"])
        gate = next(gate for gate in result["primary_gates"] if gate["comparison"] == "fresh_reused" and gate["mode"] == "idle")
        self.assertEqual(gate["failures"], 21)
        self.assertFalse(gate["pass"])

    def test_fresh_control_failure_is_not_reuse_evidence(self):
        control = copy.deepcopy(self.manifest["rows"][0]["fresh_control"])
        set_logits(control, [-4, 4])
        result = campaign.evaluate(self.declared, self.changed_row(0, fresh_control=control))
        self.assertEqual(result["disposition"], "reject")
        self.assertTrue(result["fresh_reused"]["hard_gates_pass"])
        self.assertFalse(result["fresh_fresh"]["hard_gates_pass"])

    def test_timing_load_and_pre_outcome_freeze_are_required(self):
        for row in ({"observed_load": {"generation_slots": 0, "concurrent_submitted_judgments": 0}},
                    {"timing": {}}, {"arrival_trace_sha256": "b" * 64}):
            self.assertEqual(campaign.evaluate(self.declared, self.changed_row(0, **row))["disposition"], "blocked")
        timings = copy.deepcopy(self.manifest["rows"][0]["timing"])
        timings["reused"].update(call_s=.0001, includes_checkpoint_restore=False)
        self.assertEqual(campaign.evaluate(self.declared, self.changed_row(0, timing=timings))["disposition"], "blocked")
        late = {**self.manifest, "started_utc": "2026-10-12T00:00:00Z"}
        self.assertEqual(campaign.evaluate(self.declared, late)["disposition"], "blocked")
        altered = {**self.manifest, "declaration_sha256": "b" * 64}
        self.assertEqual(campaign.evaluate(self.declared, altered)["disposition"], "blocked")


if __name__ == "__main__":
    unittest.main()
