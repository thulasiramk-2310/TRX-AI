from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import cli_input
from cli_input import StatusState, build_prompt_fn


class PromptSessionEndToEndTests(unittest.TestCase):
    """Drives the real prompt_toolkit session with simulated keystrokes."""

    def _run(self, keys: list[str], history_path: Path) -> list[str]:
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            with patch.object(cli_input.sys.stdin, "isatty", return_value=True), patch.object(
                cli_input.sys.stdout, "isatty", return_value=True
            ):
                prompt = build_prompt_fn(StatusState(model="qwen3:8b"), history_path=history_path)
            answers = []
            for key in keys:
                # Send everything except the final Enter now, and Enter a moment later, the way a
                # person types; otherwise Enter can beat the asynchronous completion.
                # (History also loads in the background, so the first keys wait a moment too.)
                typing = threading.Timer(0.3, pipe.send_text, args=(key[:-1],))
                enter = threading.Timer(0.6, pipe.send_text, args=(key[-1],))
                typing.start()
                enter.start()
                answers.append(prompt("ignored"))
                typing.join()
                enter.join()
            return answers

    def test_slash_command_tab_completion_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / "hist.txt"
            # "/stat" + Tab completes to "/status"; Enter submits; slash alias normalized.
            answers = self._run(["/stat\t\r", "explain recursion\r"], history)
            self.assertEqual(answers, ["status", "explain recursion"])
            saved = history.read_text(encoding="utf-8")
            self.assertIn("/status", saved)
            self.assertIn("explain recursion", saved)

            # New session: Up arrow recalls the last entry from the history file.
            answers = self._run(["\x1b[A\r"], history)
            self.assertEqual(answers, ["explain recursion"])


if __name__ == "__main__":
    unittest.main()
