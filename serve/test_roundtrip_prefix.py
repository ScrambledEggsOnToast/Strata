"""#272: token-prefix preservation at the client / Service roundtrip, without an engine or model pack.

The byte tokenizer has real special-token boundaries (including thinking tags); counts are fixture tokens,
not model-vocabulary counts. Both APIs use their production parser, stream assembler and response collector.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path
from unittest import mock

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

ROOT = Path(__file__).resolve().parents[1]

QUESTION = "Name the café file."
FOLLOWUP = "Thanks; next?"
THOUGHT = "Let me check.\n"
TOOL = {"name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
REPLIES = {
    "plain": "Hello.",
    "reasoning": THOUGHT + "</think>\n\nHello.",
    # Deliberately non-canonical XML inside reasoning: the API preserves the call's meaning, not its spelling
    # or its location relative to </think>. This is a bounded negative result, not a cache-eviction change.
    "tool_result": THOUGHT + "<tool_call><function=read_file><parameter=path>café.txt</parameter>"
                            "</function></tool_call>\n</think>\n\n",
}


def token_lcp(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))


class RoundtripPrefix(unittest.TestCase):
    def roundtrip(self, api, kind, *, drop_reasoning=False):
        """Exercise the public request/response seam; return the bounded measurements saved in project evidence."""
        tok = ThinkTokenizer()
        template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        engine = MockEngine(tok, "", max_context=1 << 16)
        svc = Service(engine, tok, template)
        reference = Service(MockEngine(tok, "", max_context=1 << 16), tok, template)
        reference.prompts = None
        req = {"messages": [{"role": "user", "content": QUESTION}]}
        if api == "openai":
            normalize = openai_to_messages
            req["reasoning_effort"] = "none" if kind == "plain" else "medium"
            if kind == "tool_result":
                req["tools"] = [{"type": "function", "function": TOOL}]
        else:
            normalize = anthropic_to_messages
            req["thinking"] = {"type": "disabled"} if kind == "plain" else {"type": "enabled"}
            req["output_config"] = {"effort": "medium"}
            if kind == "tool_result":
                req["tools"] = [{"name": TOOL["name"], "description": TOOL["description"],
                                 "input_schema": TOOL["parameters"]}]
        messages, tools, kwargs = normalize(req)
        first_ids, thinking, max_new = svc.prepare(messages, tools, kwargs, 256)
        self.assertEqual(first_ids, reference.prepare(messages, tools, kwargs, 256)[0])
        source = svc.prompts.entries[0]                 # (text, plain spans, ids, boundary chars, boundary token counts)

        # These are raw generated ids, including the end-of-turn token consumed by Service.run, not a re-encoding
        # of the assembled assistant message. Feed the same ids one at a time through the production detokenizer.
        raw_ids = tok.encode(REPLIES[kind], parse_special=True)
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
        decoder = Detokenizer(tok)
        events = []
        for token in raw_ids:
            events.extend(parser.feed(decoder.push(token)))
        events.extend(parser.finish())
        raw_ids += tok.encode("<|im_end|>", parse_special=True)
        run = [("event", event) for event in events] + [
            ("done", {"finish": "stop", "completion_tokens": len(raw_ids)})]
        # Only substitute the engine-facing event source. Neither assembly path nor the normalizers are mocked.
        with mock.patch.object(engine, "generate", side_effect=AssertionError("offline fixture must not generate")), \
                mock.patch.object(svc, "run", return_value=iter(run)) as replay:
            args = (svc, req, first_ids, thinking, tools, max_new, threading.Event())
            if api == "openai":
                response = openai_collect(openai_chunks(*args))
                assistant = response["choices"][0]["message"]
            else:
                response = anthropic_collect(anthropic_events(*args))
                assistant = {"role": response["role"], "content": response["content"]}
            replay.assert_called_once()

        # Replay the exact assistant payload the collector gave the client. The counterfactual models the old
        # non-tool web builder: it discarded reasoning_content whenever an answer was present.
        if drop_reasoning:
            assistant = {k: v for k, v in assistant.items() if k != "reasoning_content"}
        req["messages"].append(assistant)
        if kind == "tool_result":
            if api == "openai":
                req["messages"].append({"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"],
                                        "content": "file contents"})
            else:
                call = next(b for b in assistant["content"] if b["type"] == "tool_use")
                req["messages"].append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": call["id"], "content": "file contents"}]})
        req["messages"].append({"role": "user", "content": FOLLOWUP})
        messages, tools, kwargs = normalize(req)
        historical = messages[1]
        self.assertEqual(historical["content"], "" if kind == "tool_result" else "Hello.")
        if kind != "plain" and not drop_reasoning:
            self.assertEqual(historical["reasoning_content"], THOUGHT)
        if kind == "tool_result":
            self.assertEqual(historical["tool_calls"], [
                {"function": {"name": "read_file", "arguments": {"path": "café.txt"}}}])
            self.assertEqual(messages[-2], {"role": "tool", "content": "file contents"})
        self.assertEqual(messages[0]["content"], QUESTION)
        self.assertEqual(messages[-1]["content"], FOLLOWUP)

        with mock.patch.object(tok, "encode_marked", wraps=tok.encode_marked) as encoded:
            incremental_ids = svc.prepare(messages, tools, kwargs, 256)[0]
        full_ids = reference.prepare(messages, tools, kwargs, 256)[0]
        self.assertEqual(incremental_ids, full_ids)
        rendered, plain = svc.prompts.entries[0][:2]
        reused_chars = svc.prompts.last_reused            # NOT a token count: translate via the stored boundary
        self.assertGreater(reused_chars, 0)
        boundary = source[3].index(reused_chars)
        reused_tokens = source[4][boundary]
        self.assertNotEqual(reused_chars, reused_tokens)  # specials and UTF-8 make a mistaken char slice observable
        suffix_plain = tuple((max(a - reused_chars, 0), b - reused_chars) for a, b in plain if b > reused_chars)
        encoded.assert_called_once_with(rendered[reused_chars:], parse_special=True, plain=suffix_plain)
        fresh_suffix = tok.encode(rendered[reused_chars:], parse_special=True, plain=suffix_plain)
        self.assertEqual(source[2][:reused_tokens], full_ids[:reused_tokens])
        self.assertEqual(source[2][:reused_tokens] + fresh_suffix, full_ids)
        self.assertEqual(len(fresh_suffix), len(full_ids) - reused_tokens)
        self.assertGreater(len(fresh_suffix), 0)

        retained_ids = first_ids + raw_ids
        lcp = token_lcp(retained_ids, full_ids)
        if drop_reasoning:
            self.assertEqual(lcp, len(first_ids))
        elif kind == "tool_result":
            self.assertEqual(lcp, len(first_ids) + len(tok.encode(THOUGHT, parse_special=True)))
            self.assertEqual(tok.decode(retained_ids[lcp:])[:11], "<tool_call>")
            self.assertTrue(tok.decode(full_ids[lcp:]).startswith("</think>\n\n<tool_call>\n<function=read_file>\n"))
        else:
            self.assertEqual(lcp, len(retained_ids))      # the complete committed turn remains a prefix
        return {"turn_one_prompt_tokens": len(first_ids), "raw_generated_tokens_including_eot": len(raw_ids),
                "turn_two_prompt_tokens": len(full_ids), "encoder_reused_chars": reused_chars,
                "encoder_reused_tokens": reused_tokens, "encoder_fresh_suffix_tokens": len(fresh_suffix),
                "retained_turn_lcp_tokens": lcp, "retained_generated_prefix_tokens": lcp - len(first_ids)}

    def test_both_api_roundtrips_measure_prefix_and_fresh_suffix(self):
        for api in ("openai", "anthropic"):
            for kind in REPLIES:
                with self.subTest(api=api, kind=kind):
                    self.roundtrip(api, kind)

    def test_web_answer_keeps_reasoning_at_the_python_boundary(self):
        kept = self.roundtrip("openai", "reasoning")
        dropped = self.roundtrip("openai", "reasoning", drop_reasoning=True)
        self.assertGreater(kept["retained_turn_lcp_tokens"], dropped["retained_turn_lcp_tokens"])


if __name__ == "__main__":
    unittest.main()
