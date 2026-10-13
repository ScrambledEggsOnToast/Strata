"""CPU-only fixtures: synthetic labels here are never model-screen evidence."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import judge_reference as ref
from strata_tokenizer import BYTE_TO_UNICODE


class FakeCompiler:
    def single_token(self, text):
        return {"Yes": 80, "No": 81}.get(text, ord(text[0]))

    def compile(self, branch_id, messages, selected_token_ids):
        ids = [1, 2, 3]
        return {"schema_version": 1, "branch_id": branch_id, "messages": messages,
                "prompt_token_ids": ids, "token_digest": ref.token_digest(ids),
                "selected_token_ids": selected_token_ids}


def fixture_corpus():
    def item(name):
        return {"id": name, "messages": [{"role": "user", "content": "CPU test only"}],
                "options": [{"id": "a", "text": "a"}, {"id": "b", "text": "b"}], "gold_option_id": "b"}
    corpus = {"schema_version": 1, "issue": 224, "fold": "development", "label_provenance": "independent_human",
              "fixture_only": True, "groups": [{"group_id": str(i), "items": [item(str(i))]} for i in range(200)],
              "easy_anchors": [item("anchor")], "gate": {"minimum_group_accuracy": .9, "minimum_anchor_accuracy": 1}}
    corpus["freeze_sha256"] = ref.corpus_digest(corpus)
    return corpus


def readout(job, logits):
    lse = max(logits) + math.log(sum(math.exp(x - max(logits)) for x in logits))
    return {"schema_version": 1, "mode": "fresh", "engine_mode": "serial_solo", "position": 3,
            "consumed_prompt_tokens": 3, "row_position": 2, "prefix_tokens": 0,
            "generated_tokens": 0, "sampling": False, "mtp": False, "n_vocab": 1000,
            "selected_token_ids": job["selected_token_ids"], "logits": logits,
            "vocab_logsumexp": lse, "log_probs": [x - lse for x in logits],
            "token_digest": job["token_digest"],
            "branch_identity": {"id": job["branch_id"], "token_digest": job["token_digest"]},
            "engine_incarnation": {"id": job["branch_id"]}, "source": {"commit": "CPU fixture"}}


class ReferenceTests(unittest.TestCase):
    def test_serial_changes_are_recorded_and_daily_not_mutated(self):
        daily = {"binary": "/daily", "frontend": "/daily-tree", "timeout_s": 0,
                 "engine_args": ["--batch", "4", "--batch-groups", "2", "--prompt-cache", "6",
                                 "--batch-mtp", "--batch-mtp-pipeline", "--mtp", "/mtp", "--native", "/iq3"],
                 "reserve_bytes": 8 << 30}
        old = copy.deepcopy(daily)
        profile, delta = ref.serial_profile(daily, "/reference", "/ref-tree")
        self.assertEqual(daily, old)
        self.assertEqual(set(delta), {"binary", "frontend", "timeout_s", "engine_args"})
        self.assertEqual(profile["reserve_bytes"], 8 << 30)
        self.assertNotIn("--mtp", profile["engine_args"])
        self.assertEqual(profile["engine_args"][profile["engine_args"].index("--batch") + 1], "0")

    def test_template_fallback_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "fallback is refused"):
                ref.PromptCompiler(root)

    def test_frozen_prompt_arguments_and_no_effort_reconstruction(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            vocab = {BYTE_TO_UNICODE[i]: i for i in range(256)}
            vocab["ab"] = 256
            (path / "vocab.json").write_text(json.dumps(vocab))
            (path / "merges.txt").write_text("a b")
            (path / "token_type.json").write_text(json.dumps([1] * 257))
            (path / "tokenizer.json").write_text("{}")
            (path / "chat_template.jinja").write_text(
                "{% if enable_thinking != false or add_generation_prompt != true or reasoning_effort is defined %}"
                "{{ raise_exception('contract drift') }}{% endif %}"
                "{% for m in messages %}{{m.content}}{% endfor %}<assistant>")
            compiler = ref.PromptCompiler(path)
            request = compiler.compile("branch", [{"role": "user", "content": "question"}], [65, 66])
            self.assertEqual(request["rendered_prompt"], "question<assistant>")
            self.assertEqual(request["token_digest"], ref.token_digest(request["prompt_token_ids"]))
            self.assertFalse(request["prompt_contract"]["template_fallback"])
            with self.assertRaises(ValueError):
                compiler.compile("branch", [{"role": "user", "content": "question", "reasoning_effort": "high"}], [65])

    def test_readout_requires_position_n_and_complete_ordered_vector(self):
        job = FakeCompiler().compile("x", [], [65, 66])
        result = readout(job, [1, 2])
        ref.validate_readout(result, job)
        for patch in ({"position": 2}, {"consumed_prompt_tokens": 2}, {"generated_tokens": 1},
                      {"sampling": True}, {"mtp": True}, {"prefix_tokens": 1}, {"engine_mode": "batch"},
                      {"selected_token_ids": [66, 65]}, {"selected_token_ids": [65]}, {"logits": [1]},
                      {"logits": [float("nan"), 2]}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                ref.validate_readout({**result, **patch}, job)

    def test_both_readout_families_and_complete_group_denominator(self):
        corpus = fixture_corpus()
        jobs = ref.screen_jobs(corpus, FakeCompiler())
        receipts = [readout(job, [0, 10] if job["family"] == ref.FAMILIES[0] or job["option_id"] == "a"
                            else [10, 0]) for job in jobs]
        result = ref.evaluate_screen(corpus, jobs, receipts)
        self.assertEqual(result["disposition"], "keep")
        self.assertEqual(result["family_selection"], ref.FAMILIES[0])
        self.assertTrue(all(f["group_accuracy"] == 1 for f in result["families"].values()))
        with self.assertRaisesRegex(ValueError, "denominator"):
            ref.evaluate_screen(corpus, jobs, receipts[:-1])
        partial = copy.deepcopy(jobs)
        partial[0]["selected_token_ids"] = [65]
        with self.assertRaisesRegex(ValueError, "declared-option"):
            ref.evaluate_screen(corpus, partial, receipts)

    def test_semantic_failure_stops_without_family_selection(self):
        corpus = fixture_corpus()
        jobs = ref.screen_jobs(corpus, FakeCompiler())
        receipts = [readout(job, [10, 0]) for job in jobs]
        result = ref.evaluate_screen(corpus, jobs, receipts)
        self.assertEqual(result["disposition"], "reject")
        self.assertIsNone(result["family_selection"])
        self.assertTrue(result["stop_program"])

    def test_missing_human_corpus_and_mutated_freeze_are_refused(self):
        corpus = fixture_corpus()
        for change in ({"label_provenance": "agent"}, {"groups": corpus["groups"][:-1]},
                       {"freeze_sha256": "wrong"}):
            with self.assertRaises(ValueError):
                ref.validate_corpus({**corpus, **change})
        corpus["groups"][0]["items"][0]["gold_option_id"] = "a"
        with self.assertRaisesRegex(ValueError, "after freeze"):
            ref.validate_corpus(corpus)

    def test_no_unprotected_capture(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "supervised"):
            ref.capture({}, {}, "/not-created")


if __name__ == "__main__":
    unittest.main()
