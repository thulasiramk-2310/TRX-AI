from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import cli_input
from cli_input import COMMANDS, StatusState, build_completer, build_prompt_fn, normalize_command


def _complete(text: str) -> list[str]:
    completer = build_completer()
    return [c.text for c in completer.get_completions(Document(text, len(text)), CompleteEvent())]


class NormalizeCommandTests(unittest.TestCase):
    def test_slash_alias(self) -> None:
        self.assertEqual(normalize_command("/review a.py"), "review a.py")
        self.assertEqual(normalize_command("  /help "), "help")

    def test_non_commands_untouched(self) -> None:
        self.assertEqual(normalize_command("/"), "/")
        self.assertEqual(normalize_command("/usr/bin is a path"), "/usr/bin is a path")
        self.assertEqual(normalize_command("/reviewer notes"), "/reviewer notes")
        self.assertEqual(normalize_command("// comment"), "// comment")
        self.assertEqual(normalize_command("what is SQL?"), "what is SQL?")


class CompleterTests(unittest.TestCase):
    def test_command_names(self) -> None:
        self.assertIn("review", _complete("rev"))
        self.assertIn("review", _complete("/rev"))
        self.assertEqual(set(_complete("")), set(COMMANDS))

    def test_subcommands(self) -> None:
        self.assertEqual(_complete("mode o"), ["optimize"])
        self.assertEqual(set(_complete("personality ")), {"list", "set", "add", "delete"})
        self.assertEqual(_complete("status x"), [])

    def test_paths_for_path_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "buggy_code.py").write_text("x = 1\n", encoding="utf-8")
            old = os.getcwd()
            os.chdir(tmp)
            try:
                self.assertIn("gy_code.py", _complete("review bug"))
                self.assertIn("gy_code.py", _complete("/fix bug"))
                self.assertEqual(_complete("explain bug"), [])
            finally:
                os.chdir(old)


class StatusStateTests(unittest.TestCase):
    def test_toolbar_text(self) -> None:
        state = StatusState(model="qwen3:8b", mode="code", personality="mentor", runs=3)
        text = state.toolbar_text()
        self.assertTrue(text.startswith("  qwen3:8b · code · mentor · 3 runs"))
        self.assertIn("/help", text)
        self.assertNotIn("last", text)

        state.last_latency, state.last_source, state.llm_enabled = 6.25, "llm", False
        text = state.toolbar_text()
        self.assertIn("last 6.2s llm", text)
        self.assertIn("qwen3:8b (off)", text)

        fragments = state.toolbar_fragments()
        self.assertEqual(fragments[0], ("class:toolbar.model", "  qwen3:8b (off)"))
        self.assertEqual("".join(t for _, t in fragments), text)

    def test_rprompt_is_short(self) -> None:
        state = StatusState(model="qwen3:8b", mode="auto")
        self.assertEqual(state.rprompt_text(), "qwen3:8b · auto")
        state.last_latency, state.llm_enabled = 6.25, False
        self.assertEqual(state.rprompt_fragments(), [("class:rprompt", " qwen3:8b (off) · auto · 6.2s ")])

    def test_single_run_wording(self) -> None:
        self.assertIn("· 1 run ", StatusState(runs=1).toolbar_text() + " ")
        self.assertNotIn("run", StatusState(runs=0).toolbar_text().split("/help")[0])


class BuildPromptFnTests(unittest.TestCase):
    def test_non_interactive_uses_fallback_with_slash_normalization(self) -> None:
        with patch.object(cli_input.sys.stdin, "isatty", return_value=False):
            prompt = build_prompt_fn(StatusState(), fallback=lambda _m: "/status")
        self.assertEqual(prompt("trx >"), "status")

    def test_force_basic(self) -> None:
        prompt = build_prompt_fn(StatusState(), fallback=lambda _m: "help", force_basic=True)
        self.assertEqual(prompt("x"), "help")


if __name__ == "__main__":
    unittest.main()
