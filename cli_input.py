"""Interactive input layer for the TRX CLI (prompt_toolkit).

Adds persistent history, history-based suggestions, Tab completion for commands and file
paths, /slash command aliases, and an always-visible bottom status toolbar. Falls back to a
plain Rich prompt when stdin is not an interactive terminal or prompt_toolkit is missing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Command words, with their sub-commands. Keep in sync with the dispatcher in main.run_cli.
COMMANDS: dict[str, tuple[str, ...]] = {
    "review": (),
    "fix": (),
    "watch": (),
    "explain": (),
    "analyze": (),
    "chat": (),
    "code": (),
    "auto": (),
    "mode": ("debug", "optimize", "predict"),
    "personality": ("list", "set", "add", "delete"),
    "user": ("list", "create", "login"),
    "cache": ("clear",),
    "metrics": ("export",),
    "correct": (),
    "status": (),
    "history": (),
    "save": (),
    "export": (),
    "agents": (),
    "benchmark": (),
    "test": (),
    "help": (),
    "exit": (),
    "quit": (),
}

# Commands whose argument is a file or folder path.
PATH_COMMANDS = {"review", "fix", "watch", "save", "export"}

COMMAND_HELP: dict[str, str] = {
    "review": "multi-agent code review of a file or folder",
    "fix": "generate a fixed copy of a file",
    "watch": "re-review a path on every save",
    "explain": "ask a general question",
    "mode": "fallback profile",
    "personality": "manage personalities",
    "user": "manage user profiles",
    "status": "runtime status",
    "help": "list commands",
    "exit": "quit TRX",
}


def normalize_command(text: str) -> str:
    """Accepts `/review x` as an alias for `review x`, like other AI CLIs."""
    stripped = text.strip()
    if stripped.startswith("/"):
        first_word = stripped[1:].split(" ", 1)[0].lower()
        if first_word in COMMANDS:
            return stripped[1:]
    return stripped


@dataclass
class StatusState:
    """Values shown in the bottom toolbar; main.run_cli updates them as the session runs."""

    model: str = "unknown"
    mode: str = "auto"
    personality: str = "friendly"
    llm_enabled: bool = True
    last_latency: float | None = None
    last_source: str = ""
    runs: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def toolbar_text(self) -> str:
        model = self.model if self.llm_enabled else f"{self.model} (off)"
        parts = [f"  {model}", self.mode, self.personality]
        if self.runs:
            parts.append(f"{self.runs} run{'s' if self.runs != 1 else ''}")
        if self.last_latency is not None:
            source = f" {self.last_source}" if self.last_source else ""
            parts.append(f"last {self.last_latency:.1f}s{source}")
        parts.extend(f"{k} {v}" for k, v in self.extra.items())
        return " · ".join(parts) + "    /help · Tab · Ctrl+D "

    def rprompt_text(self) -> str:
        """Short status for the right edge of the prompt line: model, mode, last answer time."""
        model = self.model if self.llm_enabled else f"{self.model} (off)"
        parts = [model, self.mode]
        if self.last_latency is not None:
            parts.append(f"{self.last_latency:.1f}s")
        return " · ".join(parts)

    def rprompt_fragments(self) -> list[tuple[str, str]]:
        return [("class:rprompt", f" {self.rprompt_text()} ")]

    def toolbar_fragments(self) -> list[tuple[str, str]]:
        """Same text as toolbar_text, split so the model name stands out from the muted rest."""
        text = self.toolbar_text()
        head, _, tail = text.partition(" · ")
        return [("class:toolbar.model", head), ("class:toolbar", " · " + tail if tail else "")]


def _try_import_prompt_toolkit() -> Any | None:
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return None
    return prompt_toolkit


def build_completer() -> Any:
    """Completes command names, sub-commands, and file paths for path-taking commands."""
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document

    path_completer = PathCompleter(expanduser=True)

    class TrxCompleter(Completer):
        def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
            text = document.text_before_cursor
            slash = text.startswith("/")
            body = text[1:] if slash else text
            words = body.split(" ")

            if len(words) == 1:
                prefix = words[0].lower()
                for name in COMMANDS:
                    if name.startswith(prefix):
                        yield Completion(
                            name,
                            start_position=-len(words[0]),
                            display_meta=COMMAND_HELP.get(name, ""),
                        )
                return

            command = words[0].lower()
            if command in PATH_COMMANDS:
                arg = body[len(words[0]) + 1 :]
                sub_doc = Document(arg, cursor_position=len(arg))
                yield from path_completer.get_completions(sub_doc, complete_event)
                return

            subs = COMMANDS.get(command, ())
            if subs and len(words) == 2:
                prefix = words[1].lower()
                for sub in subs:
                    if sub.startswith(prefix):
                        yield Completion(sub, start_position=-len(words[1]))

    return TrxCompleter()


def build_prompt_fn(
    state: StatusState,
    *,
    history_path: str | Path = "sessions/cli_history.txt",
    fallback: Callable[[str], str] | None = None,
    force_basic: bool = False,
) -> Callable[[str], str]:
    """Returns a `prompt(message) -> str` callable for the CLI loop.

    Uses prompt_toolkit on an interactive terminal, else `fallback` (a plain Rich input by default).
    The returned text has `/command` aliases already normalized.
    """
    if fallback is None:
        from rich.console import Console

        plain_console = Console()

        def fallback(message: str) -> str:
            return plain_console.input(f"{message} ")

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if force_basic or not interactive or _try_import_prompt_toolkit() is None:
        return lambda message: normalize_command(fallback(message))

    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.shortcuts import CompleteStyle
    from prompt_toolkit.history import FileHistory, InMemoryHistory
    from prompt_toolkit.styles import Style

    try:
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        history: Any = FileHistory(str(history_path))
    except OSError:
        history = InMemoryHistory()

    # Same palette as formatter.THEME: neutral greys, one green accent.
    style = Style.from_dict(
        {
            "prompt.arrow": "#a78bfa bold",
            "prompt.glyph": "#22d3ee bold",
            "prompt.name": "#818cf8 bold",
            "rprompt": "#64748b bg:#0f172a",
            "toolbar": "#52525b",
            "toolbar.model": "#71717a",
            "auto-suggestion": "#52525b",
        }
    )
    # Layout choices that keep the screen still (see docs/ui.md):
    # - no bottom toolbar and no reserved menu space: both add lines under the prompt that make the
    #   whole screen shift up and down;
    # - Tab-only, readline-style completion: candidates print once below the prompt like bash,
    #   instead of a popup that opens and closes on every keystroke;
    # - status lives in a dim right-hand prompt on the same line.
    session: Any = PromptSession(
        history=history,
        auto_suggest=AutoSuggestFromHistory(),
        completer=build_completer(),
        complete_while_typing=False,
        complete_style=CompleteStyle.READLINE_LIKE,
        reserve_space_for_menu=0,
        rprompt=state.rprompt_fragments,
        style=style,
        enable_history_search=True,
    )

    def prompt(_message: str) -> str:
        print()  # one blank line between the last answer and the next prompt
        message = HTML("<prompt.glyph>◈ </prompt.glyph><prompt.name>trx</prompt.name><prompt.arrow> ▸ </prompt.arrow>")
        return normalize_command(session.prompt(message))

    return prompt
