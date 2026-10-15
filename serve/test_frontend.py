"""Chat history normalization before rendering the model's prompt."""
import copy
import os
from pathlib import Path
import unittest
from unittest import mock

from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service


class EmptyAssistantHistory(unittest.TestCase):
    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_openai_empty_replies_are_not_rendered(self):
        for content in (None, "", " \n\t", [], [{"type": "text", "text": ""}]):
            for preserve in (True, False):
                with self.subTest(content=content, preserve_thinking=preserve):
                    history = []
                    for _ in range(3):
                        history.extend([{"role": "user", "content": "haz un ls"},
                                        {"role": "assistant", "content": content,
                                         "reasoning_content": "Let me run ls."}])
                    history.append({"role": "user", "content": "haz un ls"})
                    messages, tools, kwargs = openai_to_messages({"messages": history})
                    kwargs["preserve_thinking"] = preserve
                    clean = [m for m in messages if m["role"] == "user"]
                    self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                                     self.service.encode_prompt(clean, tools, kwargs))

    def test_anthropic_thinking_only_replies_are_not_rendered(self):
        messages, tools, kwargs = anthropic_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "Let me run ls."}]},
            {"role": "user", "content": "haz un ls"},
        ]})
        self.assertEqual(self.service.encode_prompt(messages, tools, kwargs),
                         self.service.encode_prompt([messages[0], messages[2]], tools, kwargs))

    def test_text_and_tool_calls_are_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "text", "text": "Running ls."}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"type": "function", "function": {"name": "run", "arguments": '{"command":"ls"}'}},
            ]},
            {"role": "tool", "content": "file.txt"},
            {"role": "user", "content": "thanks"},
        ]})
        before = copy.deepcopy(messages)
        prompt = self.template.render(messages, tools=tools, **kwargs)
        self.assertIn("Running ls.", prompt)
        self.assertIn("<function=run>", prompt)
        self.assertIn("file.txt", prompt)
        self.assertEqual(messages, before)

    def test_image_only_assistant_turn_is_kept(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "describe the image"},
            {"role": "assistant", "content": [{"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "thanks"},
        ]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(messages, tools=tools, **kwargs))

    def test_final_empty_assistant_turn_is_kept(self):
        prompt = self.template.render([{"role": "user", "content": "hi"},
                                       {"role": "assistant", "content": ""}], add_generation_prompt=False)
        self.assertIn("<|im_start|>assistant\n", prompt)

    def test_empty_user_and_tool_turns_are_kept(self):
        prompt = self.template.render([{"role": "user", "content": ""},
                                       {"role": "tool", "content": ""},
                                       {"role": "user", "content": "hi"}])
        self.assertEqual(prompt.count("<|im_start|>user\n"), 3)
        self.assertIn("<tool_response>\n\n</tool_response>", prompt)


class UnreadContentParts(unittest.TestCase):
    """A content part the server does not read is refused naming it and where it sits, as /v1/responses does; chat
    and messages used to drop it without a word, and the model answered about a file it had never seen."""

    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        tok = ByteTokenizer()
        self.service = Service(MockEngine(tok, "ok"), tok, self.template)

    def test_chat_file_and_audio_parts_are_left_out(self):
        parts = ({"type": "file", "file": {"filename": "a.pdf", "file_data": "data:application/pdf;base64,AA"}},
                 {"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}})
        plain = [{"role": "user", "content": "summarize"}, {"role": "user", "content": "this one"}]
        for part in parts:
            with self.subTest(kind=part["type"]):
                got = openai_to_messages({"messages": [
                    {"role": "user", "content": "summarize"},
                    {"role": "user", "content": [{"type": "text", "text": "this one"}, part]}]})
                want = openai_to_messages({"messages": [
                    {"role": "user", "content": "summarize"},
                    {"role": "user", "content": [{"type": "text", "text": "this one"}]}]})
                self.assertEqual(got, want)

    def test_messages_blocks_real_clients_send_are_left_out(self):
        # tool_reference (Claude Code's tool search), documents, search results, server tool blocks, signed thinking
        extra = [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA"}},
                 {"type": "search_result", "source": "s", "title": "t", "content": [{"type": "text", "text": "x"}]},
                 {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}},
                 {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
                 {"type": "container_upload", "file_id": "f"}]
        for block in extra:
            with self.subTest(kind=block["type"]):
                got = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}, block]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}, block]}]})
                want = anthropic_to_messages({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": [{"type": "text", "text": "go"}]}]})
                self.assertEqual(got, want)

    def test_messages_tool_reference_in_a_tool_result_is_left_out(self):
        def run(extra):
            return anthropic_to_messages({"messages": [
                {"role": "user", "content": "find a tool"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch", "input": {}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "found"}] + extra}]}]})
        self.assertEqual(run([{"type": "tool_reference", "tool_name": "mcp__x"}]), run([]))

    def test_a_system_prompt_with_an_image_part_is_read_as_text_as_before(self):
        got = openai_to_messages({"messages": [
            {"role": "system", "content": [{"type": "text", "text": "be brief"},
                                           {"type": "image_url", "image_url": "image.png"}]},
            {"role": "user", "content": "hi"}]})
        self.assertEqual(got[0][0]["content"], "be brief")

    def test_messages_redacted_thinking_is_dropped(self):
        msgs = {"messages": [
            {"role": "user", "content": "haz un ls"},
            {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "ZW50cnk="},
                                              {"type": "text", "text": "Running ls."}]},
            {"role": "user", "content": "thanks"}]}
        without = copy.deepcopy(msgs)
        del without["messages"][1]["content"][0]
        with_block = anthropic_to_messages(msgs)
        without_block = anthropic_to_messages(without)
        self.assertEqual(with_block[0], without_block[0])
        self.assertEqual(self.service.encode_prompt(with_block[0], with_block[1], with_block[2]),
                         self.service.encode_prompt(without_block[0], without_block[1], without_block[2]))

    def test_chat_refusal_part_is_read_as_its_text(self):
        messages, tools, kwargs = openai_to_messages({"messages": [
            {"role": "user", "content": "do a thing"},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "I cannot do that."}]},
            {"role": "user", "content": "ok"}]})
        self.assertEqual(messages[1]["content"], "I cannot do that.")
        self.assertIn("I cannot do that.", self.template.render(messages, tools=tools, **kwargs))

    def test_text_and_image_requests_render_as_before(self):
        parts = [{"role": "user", "content": [{"type": "text", "text": "haz un ls"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "file.txt"}]}]
        plain = [{"role": "user", "content": "haz un ls"}, {"role": "assistant", "content": "file.txt"}]
        for convert in (openai_to_messages, anthropic_to_messages):
            with self.subTest(convert=convert.__name__):
                self.assertEqual(self.service.encode_prompt(*convert({"messages": parts})),
                                 self.service.encode_prompt(*convert({"messages": plain})))
        chat = openai_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image_url", "image_url": "image.png"}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(chat[0], tools=chat[1], **chat[2]))
        msgs = anthropic_to_messages({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "describe"},
                                         {"type": "image", "source": {"type": "url", "url": "image.png"}}]}]})
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>",
                      self.template.render(msgs[0], tools=msgs[1], **msgs[2]))


if __name__ == "__main__":
    unittest.main()


class EmptyTurnsSwitch(unittest.TestCase):
    def test_keep_empty_turns_restores_the_old_prompt(self):
        import os
        from unittest import mock
        template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": ""},
                {"role": "user", "content": "b"}]
        clean = template.render([msgs[0], msgs[2]])
        self.assertEqual(template.render(msgs), clean)
        with mock.patch.dict(os.environ, {"STRATA_KEEP_EMPTY_TURNS": "1"}):
            self.assertNotEqual(template.render(msgs), clean)


class BoundedRendering(unittest.TestCase):
    """The preparation-only bounds of the template and of its literal restoration: a prompt is rendered and restored
    under a stated character bound, and over it the result is None - never a shortened prompt."""

    def setUp(self):
        self.template = ChatTemplate(Path(__file__).with_name("chat_template.jinja"))
        self.messages = [{"role": "user", "content": "a long enough turn to render " * 40}]

    def test_the_bounded_render_is_the_unbounded_one_until_the_bound_is_passed(self):
        whole = self.template.render(self.messages)
        self.assertEqual(self.template.render_within(self.messages, limit=len(whole)), whole)
        self.assertEqual(self.template.render_within(self.messages, limit=len(whole) + 1), whole)
        self.assertIsNone(self.template.render_within(self.messages, limit=len(whole) - 1))
        self.assertIsNone(self.template.render_within(self.messages, limit=16))
        # None renders exactly as render does, for every caller that is not the queued frontier
        self.assertEqual(self.template.render_within(self.messages, limit=None), whole)
        # the empty-turn filtering is the same on both paths (#843)
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""},
                   {"role": "user", "content": "q2"}]
        self.assertEqual(self.template.render_within(history, limit=None), self.template.render(history))
        self.assertEqual(self.template.render_within(history, limit=len(self.template.render(history))),
                         self.template.render(history))

    def test_a_restoration_over_its_bounds_is_refused_rather_than_shortened(self):
        from serve.frontend import mark_think_literals, unmark_think_literals
        marked, _tools, changed = mark_think_literals([{"role": "user", "content": "<think>a</think>" * 40}], None)
        self.assertTrue(changed)
        prompt = marked[0]["content"]
        restored, spans = unmark_think_literals(prompt)
        self.assertEqual(spans, unmark_think_literals(prompt, restore_limit=len(restored),
                                                      span_limit=len(spans))[1])
        self.assertEqual(unmark_think_literals(prompt, restore_limit=len(restored), span_limit=len(spans))[0],
                         restored)
        self.assertIsNone(unmark_think_literals(prompt, restore_limit=len(restored) - 1))
        self.assertIsNone(unmark_think_literals(prompt, span_limit=len(spans) - 1))
        self.assertIsNone(unmark_think_literals(prompt, restore_limit=len(restored), span_limit=0))
        self.assertEqual(len(spans), 80)                                   # forty <think> and forty </think>


def pack_template():
    """The deployed pack's own chat template - the one production renders with - or None when there is no pack here.

    serve/server.py constructs `ChatTemplate(pack/.../chat_template.jinja)` whenever the pack ships one, and that
    template (with its `render_content` macro and `tojson` filter) is what the preparation bounds must hold for, not
    only this repository's fixture.
    """
    dirs = [os.environ.get("STRATA_TOKENIZER", "")]
    dirs += [str(p) for p in Path("/srv/development/models").glob("*/pack/tokenizer")]
    for directory in dirs:
        if directory and (Path(directory) / "chat_template.jinja").exists():
            return Path(directory) / "chat_template.jinja"
    return None


@unittest.skipIf(pack_template() is None, "no deployed pack template here")
class PackTemplateRendering(unittest.TestCase):
    """The preparation path with the template production really uses: the bounded render and its literal restoration
    produce exactly the demand's own ids (the effort split, a literal <think>, and a prefix head included), and a
    bound below what the prompt needs refuses instead of shortening it."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.template = ChatTemplate(pack_template())
        cls.svc = Service(MockEngine(cls.tok, "ok", max_context=32768), cls.tok, cls.template)

    def setUp(self):
        self.svc.effort_end = True
        self.messages = [{"role": "system", "content": "You are a careful assistant. " * 20},
                         {"role": "user", "content": "Read this: " + ("the document text " * 300)},
                         {"role": "assistant", "content": "Done."},
                         {"role": "user", "content": "Now: <think>a quoted marker</think> and a question."}]
        self.tools = [{"name": "read", "description": "read a file " * 10,
                       "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]

    def test_the_bounded_path_produces_the_demands_own_ids(self):
        for kwargs in ({}, {"reasoning_effort": "low"}, {"enable_thinking": False}):
            with self.subTest(kwargs=kwargs):
                bound = self.svc.preparation_token_limit()
                self.assertEqual(self.svc.encode_prompt(self.messages, self.tools, dict(kwargs), limit=bound),
                                 self.svc.encode_prompt(self.messages, self.tools, kwargs))

    def test_the_prefix_head_is_bounded_the_same_way(self):
        req = {"strata_prefix": {"messages": 3}}
        prepared = self.svc.prepare(self.messages, self.tools, {}, 16, req=req)
        self.assertIsNotNone(prepared)
        pinned = dict(req["strata_prefix"])
        self.assertGreater(pinned.get("tokens", 0), 0)
        bounded_req = {"strata_prefix": {"messages": 3}}
        self.assertEqual(self.svc.prepare(self.messages, self.tools, {}, 16, req=bounded_req,
                                          limit=self.svc.preparation_token_limit()), prepared)
        self.assertEqual(bounded_req["strata_prefix"], pinned)

    def test_a_long_context_prompt_is_bounded_and_still_the_demands_own(self):
        """A prompt of the length the frontier really prepares (a long turn, the tools included) on the template
        production renders with: the bounded path produces the demand's own ids, and both figure are inside the
        frontend's own bounds."""
        svc = Service(MockEngine(self.tok, "ok", max_context=262144), self.tok, self.template)
        messages = [{"role": "user", "content": "a document line that is long enough to matter\n" * 4000},
                    {"role": "user", "content": "and the question"}]
        ids = svc.encode_prompt(messages, self.tools, {}, limit=svc.preparation_token_limit())
        self.assertIsNotNone(ids)
        self.assertEqual(ids, svc.encode_prompt(messages, self.tools, {}))
        rendered = svc.render_prompt(messages, self.tools, {})
        self.assertGreater(len(rendered), 1 << 17)
        self.assertLessEqual(len(rendered), svc.preparation_render_chars())
        self.assertLessEqual(len(ids), svc.preparation_token_limit())
        self.assertEqual(svc.render_prompt(messages, self.tools, {}, limit=svc.preparation_render_chars()), rendered)

    def test_a_render_bound_below_the_prompt_refuses_it(self):
        rendered = self.svc.render_prompt(self.messages, self.tools, {})
        self.assertIsNone(self.svc.render_prompt(self.messages, self.tools, {}, limit=len(rendered) - 1))
        self.assertEqual(self.svc.render_prompt(self.messages, self.tools, {}, limit=len(rendered)), rendered)
        bounded = self.svc.encode_prompt(self.messages, self.tools, {}, limit=self.svc.preparation_token_limit())
        self.assertIsNotNone(bounded)
        with mock.patch.object(self.svc, "preparation_render_chars", return_value=64):
            self.assertIsNone(self.svc.encode_prompt(self.messages, self.tools, {},
                                                     limit=self.svc.preparation_token_limit()))


class ForcedCallOpening(unittest.TestCase):
    def test_required_with_one_tool_names_it(self):
        from serve.frontend import forced_call
        one, two = [{"name": "a"}], [{"name": "a"}, {"name": "b"}]
        self.assertTrue(forced_call("required", one).endswith("<function=a>\n"))
        self.assertTrue(forced_call("required", two).endswith("<function="))
        self.assertIsNone(forced_call("auto", one))


class QuotedToolCallCloser(unittest.TestCase):
    """#2026: a `</tool_call>` a user quotes is text, not the end-of-call token (type 4: matched even without
    parse_special); an assistant message replaying raw call text keeps the model's own tokens."""

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import strata_tokenizer as ST
        b2u = ST.bytes_to_unicode()
        self.tok = ST.Tokenizer([b2u[b] for b in range(256)] + ["</tool_call>", "<|im_end|>"], [], [1] * 256 + [4, 3])
        self.tags = __import__("serve.frontend", fromlist=["literal_tags"]).literal_tags(self.tok.control_tokens)

    def encode(self, role, text):
        from serve.frontend import mark_think_literals, unmark_think_literals
        marked, _, changed = mark_think_literals([{"role": role, "content": text}], None, self.tags)
        prompt, plain = unmark_think_literals(marked[0]["content"], self.tags)
        return self.tok.encode(prompt, parse_special=True, plain=plain), changed

    def test_user_text_stays_text(self):
        text = "output verbatim: </tool_call>."
        ids, changed = self.encode("user", text)
        self.assertTrue(changed)
        self.assertEqual(ids, [*text.encode()])
        self.assertNotEqual(self.tok.encode(text, parse_special=True), ids)     # unmarked, it was the special token

    def test_assistant_replay_keeps_its_token(self):
        text = "<tool_call>x</tool_call>"
        ids, _ = self.encode("assistant", text)
        self.assertEqual(ids, self.tok.encode(text, parse_special=True))


class FencedJsonCallInWrapper(unittest.TestCase):
    """#2037: with recovery on, a declared tool's JSON call in a Markdown fence inside <tool_call> is the call."""
    TOOLS = [{"name": "write_file", "parameters": {"type": "object", "properties": {"content": {"type": "string"}}}}]
    PAYLOAD = '{"name": "write_file", "arguments": {"content": "hello `world`"}}'

    def run_parser(self, raw, chunk, stream_tools, recover=True, tools=None):
        from serve.frontend import OutputParser
        p = OutputParser(thinking=False, tools=self.TOOLS if tools is None else tools, recover=recover,
                         stream_tools=stream_tools)
        events = []
        for i in range(0, len(raw), chunk):
            events += p.feed(raw[i:i + chunk])
        events += p.finish()
        return ([e.call.arguments for e in events if e.kind == "tool_call"],
                "".join(e.text for e in events if e.kind == "content"))

    def test_fenced_and_bare_agree(self):
        for fence in ("```json\n", "```\n", "```JSON\r\n"):
            raw = f"<tool_call>\n{fence}{self.PAYLOAD}\n```\n</tool_call>"
            for chunk in (len(raw), 1, 7):
                for st in (False, True):
                    with self.subTest(fence=fence, chunk=chunk, stream_tools=st):
                        self.assertEqual(self.run_parser(raw, chunk, st), ([{"content": "hello `world`"}], ""))

    def test_other_fences_and_prose_stay_text(self):
        raw = "<tool_call>\n```python\nx = 1\n```\n</tool_call>"
        for chunk in (len(raw), 1):
            self.assertEqual(self.run_parser(raw, chunk, False), ([], raw))
        raw = f"Here:\n```json\n{self.PAYLOAD}\n```\n"                 # no wrapper: just text
        self.assertEqual(self.run_parser(raw, len(raw), False), ([], raw))

    def test_off_without_recovery_or_for_an_undeclared_tool(self):
        raw = f"<tool_call>\n```json\n{self.PAYLOAD}\n```\n</tool_call>"
        calls, text = self.run_parser(raw, len(raw), False, recover=False)
        self.assertEqual((calls, text), ([], raw))
        other = [{"name": "other", "parameters": {}}]
        calls, text = self.run_parser(raw, len(raw), False, tools=other)
        self.assertEqual((calls, text), ([], raw))
