from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

import observability
from formatter import OutputFormatter


def _render(analysis: dict, *, encoding: str = "utf-8") -> str:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding, errors="strict")
    console = Console(file=stream, force_terminal=False, color_system=None, width=110)
    OutputFormatter(console).render(analysis, "debug")
    stream.flush()
    return buffer.getvalue().decode(encoding)


REVIEW = {
    "response_mode": "analysis",
    "current_input": "review app.py",
    "model_name": "qwen3:8b",
    "analysis_source": "llm",
    "confidence_score": 80,
    "debug_analysis": [
        "Line 2: returns arr[0] on an empty list",
        "Impacted module: find_max touches other.py",
        "Call relation: find_max -> len",
    ],
    "improvements": ["Guard empty input"],
    "system_status": ["source=llm"],
    "elapsed_seconds": 3.25,
}


class ReviewCardTests(unittest.TestCase):
    def test_graph_lines_move_out_of_issues_into_meta(self) -> None:
        out = _render(dict(REVIEW))
        self.assertIn("REVIEW ▸ app.py", out)
        self.assertIn("returns arr[0] on an empty list", out)
        self.assertNotIn("Impacted module", out)
        self.assertIn("graph: 1 related module, 1 call", out)
        self.assertIn("qwen3:8b · 3.2s · llm", out)
        self.assertIn("▰▰▰▰▰▰▰▰▱▱ 80%", out)

    def test_diff_shown_when_original_known(self) -> None:
        analysis = dict(REVIEW, original_code="def f(a):\n    return a[0]\n",
                        fixed_code="# FILE: app.py\ndef f(a):\n    if not a:\n        return None\n    return a[0]\n")
        out = _render(analysis)
        self.assertIn("SUGGESTED CHANGE  +2 -0", out)
        self.assertIn("+    if not a:", out)
        self.assertNotIn("FILE: app.py", out)
        self.assertNotIn("@@", out)  # hunk headers are hidden

    def test_no_box_drawing_or_trailing_spaces(self) -> None:
        # Borders end up in copied text; padded lines re-wrap into blank lines on window resize.
        out = _render(dict(REVIEW, original_code="a = 1\n", fixed_code="a = 2\n"))
        self.assertFalse(set(out) & set("│┃╭╮╰╯┌┐└┘"))  # header rule ─ is fine
        for line in out.splitlines():
            self.assertEqual(line, line.rstrip(), repr(line))

    def test_unchanged_fix_is_not_dumped(self) -> None:
        code = "def f(a):\n    return a[0]\n"
        out = _render(dict(REVIEW, original_code=code, fixed_code="# FILE: app.py\n" + code))
        self.assertIn("no code changes suggested", out)
        self.assertNotIn("SUGGESTED", out)

    def test_fix_without_original_shows_full_code(self) -> None:
        out = _render(dict(REVIEW, fixed_code="x = 1\n"))
        self.assertIn("SUGGESTED FIX", out)
        self.assertIn("x = 1", out)

    def test_ascii_fallback_on_cp1252(self) -> None:
        out = _render(dict(REVIEW), encoding="cp1252")
        self.assertIn("/ REVIEW > app.py", out)
        self.assertIn("    x Line 2: returns arr[0]", out)
        self.assertIn("qwen3:8b | 3.2s | llm", out)

    def test_unencodable_text_does_not_poison_later_prints(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
        formatter = OutputFormatter(Console(file=stream, force_terminal=False, color_system=None, width=80))
        formatter.render({"response_mode": "chat", "chat_response": "bad \x81 char"}, "debug")
        formatter.render({"response_mode": "chat", "chat_response": "second answer"}, "debug")
        stream.flush()
        self.assertIn("second answer", buffer.getvalue().decode("cp1252"))


class ChatAndMetaTests(unittest.TestCase):
    def test_source_labels(self) -> None:
        base = {"response_mode": "chat", "chat_response": "Hi.", "model_name": "m"}
        self.assertIn("m · built-in knowledge", _render(dict(base, analysis_source="knowledge-base")))
        fallback = dict(base, analysis_source="llm", system_status=["llm_unavailable"])
        self.assertIn("rule fallback (LLM unavailable)", _render(fallback))

    def test_tiny_elapsed_is_hidden(self) -> None:
        out = _render({"response_mode": "chat", "chat_response": "Hi.", "model_name": "m",
                       "analysis_source": "llm", "elapsed_seconds": 0.01})
        self.assertIn("m · llm", out)
        self.assertNotIn("0.0s", out)


class StructuredLogTests(unittest.TestCase):
    def test_events_go_to_file_not_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "events.jsonl"
            env = {"RD_STRUCTURED_LOGS": "true", "RD_LOG_FILE": str(log)}
            with patch.dict(os.environ, env), patch("builtins.print") as printed:
                observability.log_event("unit_test", value=1)
            printed.assert_not_called()
            row = json.loads(log.read_text(encoding="utf-8").strip())
            self.assertEqual((row["event"], row["value"]), ("unit_test", 1))

    def test_stdout_mode_and_disabled(self) -> None:
        with patch.dict(os.environ, {"RD_STRUCTURED_LOGS": "stdout"}), patch("builtins.print") as printed:
            observability.log_event("x")
        printed.assert_called_once()
        with patch.dict(os.environ, {"RD_STRUCTURED_LOGS": "false"}), patch("builtins.print") as printed:
            observability.log_event("x")
        printed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
