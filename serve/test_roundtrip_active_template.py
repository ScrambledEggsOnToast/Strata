"""#272: measure the installed seeded template at the offline production API roundtrip.

Fixture tokens are UTF-8 bytes plus real special-token boundaries, not model-vocabulary counts.
No engine, GPU, download or browser is used. To save the measurements while running this test:

    python -m serve.test_roundtrip_active_template -v --evidence ../evidence/issue-272/active-template.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from jinja2 import TemplateError

from serve.frontend import (
    ChatTemplate,
    OutputParser,
    anthropic_to_messages,
    openai_to_messages,
)
from serve.server import (
    Detokenizer,
    MockEngine,
    Service,
    anthropic_collect,
    anthropic_events,
    openai_chunks,
    openai_collect,
)
from serve.test_prompt_encoder import ThinkTokenizer
from serve.test_roundtrip_prefix import FOLLOWUP, QUESTION, token_lcp

ACTIVE_TEMPLATE = Path.home() / "strata-workspace/releases/sharp-seeded-v10-20261017/tokenizer/chat_template.jinja"
GENERATION_SEED = "<think>\nLet me"
SEED_TEXT = "Let me"
# The prompt already contains "Let me". Only these newly generated tokens reach the parser/collector.
THOUGHT = " check.\n"
RAW_REPLY = THOUGHT + "</think>\n\nHello."


class ActiveTemplateRoundtrip(unittest.TestCase):
    evidence: ClassVar[dict] = {}

    @classmethod
    def setUpClass(cls):
        cls.evidence = {
            "schema_version": 1,
            "issue": 272,
            "status": "not_measured",
            "template": {"path": str(ACTIVE_TEMPLATE), "sha256": None},
            "tokenizer_kind": "serve.test_prompt_encoder.ThinkTokenizer (UTF-8 bytes plus special-token ids)",
            "fixture": {
                "question": QUESTION,
                "followup": FOLLOWUP,
                "thinking": "enabled, medium effort, no tools",
                "generation_seed": GENERATION_SEED,
                "raw_generated_text": RAW_REPLY,
                "end_of_turn": "<|im_end|> (included in raw generated ids, consumed before parsing)",
                "seam": "normalize -> Service.prepare -> Detokenizer/OutputParser -> production stream assembler/"
                        "collector -> exact assistant payload + user followup -> normalize -> Service.prepare",
            },
            "measurements": {},
            "limits": [
                "Installed template with deterministic fixture tokens, not active model BPE counts or model output.",
                "No native engine, GPU, browser, sent HTTP body capture, model-state cache hit or performance claim.",
            ],
        }
        try:
            cls.evidence["template"]["sha256"] = hashlib.sha256(ACTIVE_TEMPLATE.read_bytes()).hexdigest()
            cls.template = ChatTemplate(ACTIVE_TEMPLATE)
        except (OSError, TemplateError) as exc:
            cls.evidence.update(status="blocked", decision="blocked",
                                missing_artifact=str(ACTIVE_TEMPLATE), reason=str(exc))
            raise unittest.SkipTest(f"Cannot load active installed template: {ACTIVE_TEMPLATE}: {exc}; no download") from exc

    def roundtrip(self, api):
        tok = ThinkTokenizer()
        engine = MockEngine(tok, "", max_context=1 << 16)
        svc = Service(engine, tok, self.template)
        reference = Service(MockEngine(tok, "", max_context=1 << 16), tok, self.template)
        reference.prompts = None  # Independent fresh full encoding, not a second incremental cache.
        req: dict = {"messages": [{"role": "user", "content": QUESTION}]}
        if api == "openai":
            normalize = openai_to_messages
            req["reasoning_effort"] = "medium"
        else:
            normalize = anthropic_to_messages
            req["thinking"] = {"type": "enabled"}
            req["output_config"] = {"effort": "medium"}
        messages, tools, kwargs = normalize(req)
        first_ids, thinking, max_new = svc.prepare(messages, tools, kwargs, 256)
        self.assertTrue(thinking)
        self.assertEqual(first_ids, reference.prepare(messages, tools, kwargs, 256)[0])
        assert svc.prompts is not None
        source = svc.prompts.entries[0]
        first_text = source[0]
        input_history = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        input_ids = tok.encode(input_history, parse_special=True)
        self.assertTrue(first_text.startswith(input_history))
        self.assertEqual(first_ids[:len(input_ids)], input_ids)
        self.assertTrue(first_text.endswith(GENERATION_SEED))
        stable_ids = tok.encode(first_text[:-len(SEED_TEXT)], parse_special=True)
        self.assertEqual(first_ids, stable_ids + tok.encode(SEED_TEXT, parse_special=True))

        raw_ids = tok.encode(RAW_REPLY, parse_special=True)
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
        decoder = Detokenizer(tok)
        events = []
        for token in raw_ids:
            events.extend(parser.feed(decoder.push(token)))
        events.extend(parser.finish())
        raw_ids += tok.encode("<|im_end|>", parse_special=True)
        run = [("event", event) for event in events] + [
            ("done", {"finish": "stop", "completion_tokens": len(raw_ids)})]
        # Substitute only the engine-facing source, exactly as the original roundtrip fixture does.
        with mock.patch.object(engine, "generate", side_effect=AssertionError("offline fixture must not generate")), \
                mock.patch.object(svc, "run", return_value=iter(run)) as replay:
            args = (svc, req, first_ids, thinking, tools, max_new, threading.Event())
            if api == "openai":
                response = openai_collect(openai_chunks(*args))
                assistant = response["choices"][0]["message"]
            else:
                response = anthropic_collect(anthropic_events(*args))
                assistant = {"role": response["role"], "content": response["content"],
                             "reasoning_content_is_continuation": True}
            replay.assert_called_once()
        req["messages"].extend([assistant, {"role": "user", "content": FOLLOWUP}])
        messages, tools, kwargs = normalize(req)
        self.assertEqual(messages[1]["reasoning_content"], THOUGHT)
        self.assertEqual(messages[1]["content"], "Hello.")
        self.assertEqual(messages[0]["content"], QUESTION)
        self.assertEqual(messages[-1]["content"], FOLLOWUP)

        incremental_ids = svc.prepare(messages, tools, kwargs, 256)[0]
        full_ids = reference.prepare(messages, tools, kwargs, 256)[0]
        self.assertEqual(incremental_ids, full_ids)
        # "Input history" excludes the generation prologue. The full prompt includes the seed, so asserting it
        # remains a prefix would hide the very boundary loss this measurement is meant to distinguish.
        self.assertEqual(full_ids[:len(input_ids)], input_ids)
        self.assertEqual(full_ids[:len(stable_ids)], stable_ids)  # Stable through <think>\n, before "Let me".
        reused_chars = svc.prompts.last_reused
        self.assertGreater(reused_chars, 0)
        reused_tokens = source[4][source[3].index(reused_chars)]

        retained_ids = first_ids + raw_ids
        lcp = token_lcp(retained_ids, full_ids)
        first_difference = None
        if lcp == len(retained_ids):
            decision = "prefix preserved"
        else:
            at_seed = len(stable_ids) <= lcp < len(first_ids)
            decision = "seeded-boundary loss" if at_seed else "other loss"
            first_difference = {
                "token_index": lcp,
                "relative_to_turn_one_prompt_end": lcp - len(first_ids),
                "location": "seeded <think>\\nLet me boundary" if at_seed else "later render change",
                "retained_token_id": retained_ids[lcp] if lcp < len(retained_ids) else None,
                "turn_two_token_id": full_ids[lcp] if lcp < len(full_ids) else None,
                "retained_token": tok.decode(retained_ids[lcp:lcp + 1]),
                "turn_two_token": tok.decode(full_ids[lcp:lcp + 1]),
                "retained_tail": tok.decode(retained_ids[lcp:lcp + 32]),
                "turn_two_tail": tok.decode(full_ids[lcp:lcp + 32]),
            }
        # The fixed renderer must preserve the whole prior turn. The earlier seeded-loss measurement is retained
        # separately in the evidence; this test asserts the repaired behavior so a regression cannot pass silently.
        self.assertEqual(lcp, len(retained_ids), "the full prior turn is not a prefix after seed restoration")
        result = {
            "assistant_payload": assistant,
            "turn_one_prompt_tokens": len(first_ids),
            "turn_one_input_history_tokens": len(input_ids),
            "stable_prefix_through_think_tokens": len(stable_ids),
            "raw_generated_tokens_including_eot": len(raw_ids),
            "retained_turn_tokens": len(retained_ids),
            "turn_two_prompt_tokens": len(full_ids),
            "incremental_equals_full": incremental_ids == full_ids,
            "input_history_prefix_preserved": full_ids[:len(input_ids)] == input_ids,
            "full_prior_prompt_preserved": full_ids[:len(first_ids)] == first_ids,
            "encoder_reused_chars": reused_chars,
            "encoder_reused_tokens": reused_tokens,
            "encoder_fresh_suffix_tokens": len(full_ids) - reused_tokens,
            "retained_turn_lcp_tokens": lcp,
            "retained_generated_prefix_tokens": max(0, lcp - len(first_ids)),
            "first_divergence": first_difference,
            "decision": decision,
        }
        print(f"{api}: retained-turn LCP {lcp}/{len(retained_ids)}; {decision}; "
              f"first divergence={json.dumps(first_difference, ensure_ascii=False)}")
        return result

    def test_installed_template_roundtrip_measures_actual_prefix(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                self.evidence["measurements"][api] = self.roundtrip(api)
        if len(self.evidence["measurements"]) == 2:
            decisions = {m["decision"] for m in self.evidence["measurements"].values()}
            self.evidence.update(status="measured", decision=next(iter(decisions)) if len(decisions) == 1 else "mixed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, help="write compact measured or blocked evidence JSON")
    args, remaining = parser.parse_known_args()
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    if args.evidence is not None:
        evidence = ActiveTemplateRoundtrip.evidence
        if not program.result.wasSuccessful():
            evidence["status"] = "failed"
        evidence["checks"] = {"active_template_unittest": {
            "command": "python3 -m serve.test_roundtrip_active_template -v "
                       "--evidence ../evidence/issue-272/active-template.json",
            "tests_run": program.result.testsRun,
            "failures": len(program.result.failures),
            "errors": len(program.result.errors),
            "skipped": len(program.result.skipped),
            "exit_code": int(not program.result.wasSuccessful()),
        }}
        args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(not program.result.wasSuccessful())


class SeedHistoryContract(unittest.TestCase):
    def _frontend(self, seed):
        from serve.frontend import ChatTemplate
        f = ChatTemplate.__new__(ChatTemplate)
        f.thinking_seed = seed
        return f

    def test_empty_continuation_keeps_the_prefill(self):
        out = self._frontend("Let me")._seed_history([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "A", "reasoning_content": "",
             "reasoning_content_is_continuation": True},
        ])
        self.assertEqual(out[1]["reasoning_content"], "Let me")

    def test_tool_result_continuation_is_not_seeded(self):
        out = self._frontend("Let me")._seed_history([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f"}}],
             "reasoning_content": "check", "reasoning_content_is_continuation": True},
            {"role": "tool", "content": "r"},
            {"role": "assistant", "content": "A", "reasoning_content": "done",
             "reasoning_content_is_continuation": True},
        ])
        self.assertEqual(out[1]["reasoning_content"], "Let mecheck")
        self.assertEqual(out[3]["reasoning_content"], "done")

    def test_unmarked_histories_are_unchanged_through_both_converters(self):
        from serve.frontend import openai_to_messages, anthropic_to_messages
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "A", "reasoning_content": "I checked"}]})
        self.assertNotIn("reasoning_content_is_continuation", msgs[1])
        self.assertEqual(self._frontend("Let me")._seed_history(msgs)[1]["reasoning_content"], "I checked")
        msgs, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I checked"}, {"type": "text", "text": "A"}]}]})
        self.assertNotIn("reasoning_content_is_continuation", msgs[1])
        self.assertEqual(self._frontend("Let me")._seed_history(msgs)[1]["reasoning_content"], "I checked")

    def test_marked_empty_continuation_survives_both_converters(self):
        from serve.frontend import openai_to_messages, anthropic_to_messages
        msgs, _, _ = openai_to_messages({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "A", "reasoning_content": "",
             "reasoning_content_is_continuation": True}]})
        self.assertEqual(msgs[1]["reasoning_content"], "")
        self.assertTrue(msgs[1]["reasoning_content_is_continuation"])
        self.assertEqual(self._frontend("Let me")._seed_history(msgs)[1]["reasoning_content"], "Let me")
        msgs, _, _ = anthropic_to_messages({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "A"}],
             "reasoning_content_is_continuation": True}]})
        self.assertEqual(msgs[1]["reasoning_content"], "")
        self.assertTrue(msgs[1]["reasoning_content_is_continuation"])
        self.assertEqual(self._frontend("Let me")._seed_history(msgs)[1]["reasoning_content"], "Let me")

    def test_collector_preserves_a_marked_empty_continuation_through_normalization(self):
        from serve.server import openai_collect
        from serve.frontend import openai_to_messages
        def chunk(delta, finish=None):
            return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        final = chunk({}, "stop")
        final["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        msg = openai_collect([chunk({"role": "assistant", "content": "",
                                     "reasoning_content_is_continuation": True}),
                              chunk({"content": "Hello."}), final])["choices"][0]["message"]
        self.assertEqual(msg["reasoning_content"], "")
        self.assertTrue(msg["reasoning_content_is_continuation"])
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": "q"}, msg]})
        self.assertEqual(self._frontend("Let me")._seed_history(msgs)[1]["reasoning_content"], "Let me")

    def test_unmarked_history_is_unchanged(self):
        out = self._frontend("Let me")._seed_history([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "A", "reasoning_content": "I checked"},
        ])
        self.assertEqual(out[1]["reasoning_content"], "I checked")

if __name__ == "__main__":
    raise SystemExit(main())
