"""TRX full-screen terminal app.

Layout: masthead / (navigation | conversation stream | live telemetry) / command bar / footer.
Commands typed in the command bar run in a worker thread through tui.engine.Engine, which reuses
the classic CLI's analyzer, review loading, guardrails and history.

Run:  trx            (or: python -m tui.app)      F2 cycles themes · Ctrl+L focuses the command bar
"""

from __future__ import annotations

import difflib
import io
from pathlib import Path
import time
from typing import Any

from rich.console import Console, Group
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Digits, Footer, Input, Markdown, OptionList, ProgressBar, Sparkline, Static
from textual.widgets.option_list import Option

from cli_input import normalize_command
from formatter import SOURCE_LABELS, OutputFormatter
from guardrails import Finding, strip_terminal_controls
from tui.themes import THEMES

DEFAULT_THEME = "trx-ops"
NAV = (("chat", "Chat"), ("review", "Code review"), ("fix", "Generate fix"), ("help", "Help"))
HELP_TEXT = """\
**Commands** (type in the bar below; a leading `/` is optional)

| command | what it does |
|---|---|
| `review <file or folder>` | multi-agent code review |
| `fix <file>` | review, show the patch and guardrail checks, then ask before saving `<name>_fixed` |
| anything else | ask TRX a question |
| `clear` | clear the conversation |
| `theme` | switch colour theme (or press **F2**) |
| `inspect` | show or hide diagnostics (or press **F3**) |
| `exit` | quit (or **Ctrl+Q**) |

Use **Tab** and arrow keys to navigate the sidebar; **Enter** selects an action.
**Ctrl+L** returns to the command bar. **Escape** cancels a fix confirmation.
Classic scrolling CLI, including history and benchmarks: `trx --classic`.
"""

# Shared helpers from the classic renderer, so both UIs pick sections the same way.
_HELPER = OutputFormatter(Console(file=io.StringIO(), width=120))


def _source_label(result: dict[str, Any]) -> str:
    source = str(result.get("analysis_source", "")).lower()
    if any("rule-fallback" in s or s == "llm_unavailable" for s in map(str.lower, map(str, result.get("system_status", [])))):
        source = "rule-fallback"
    return SOURCE_LABELS.get(source, source)


def _meta(result: dict[str, Any]) -> str:
    parts = [str(result.get("model_name", "")).strip()]
    elapsed = result.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed >= 0.05:
        parts.append(f"{elapsed:.1f}s")
    parts.append(_source_label(result))
    return " · ".join(p for p in parts if p)


def _status_value(result: dict[str, Any], key: str) -> str:
    for item in result.get("system_status", []):
        text = str(item)
        if text.startswith(f"{key}="):
            return text.split("=", 1)[1]
    return ""


# ----------------------------------------------------------------------------------- widgets


class Masthead(Static):
    """Compact workspace status; detailed diagnostics live in the inspector."""

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh)

    def render(self) -> Table:
        theme = self.app.current_theme
        status = getattr(self.app, "sys_status", None)
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        workspace = getattr(self.app, "workspace_name", "workspace")
        title = Text.assemble(("TRX", f"bold {theme.primary}"), (f"  /  {workspace}", theme.foreground))
        if status is None:
            detail = Text("starting engine", style=theme.warning)
        else:
            state = "ready" if status.llm_reachable else ("offline" if status.llm_enabled else "disabled")
            detail = Text.assemble((status.model, theme.foreground), (f"  {state}", theme.success if status.llm_reachable else theme.warning))
            if self.size.width >= 90:
                detail.append(f"  ·  {status.compute}  ·  graph {status.graph}", style="dim")
        grid.add_row(title, detail)
        return grid


class ReviewCard(Static):
    """One code review: confidence bar and Issues / Fixes / Performance / Summary sections."""

    COLORS = {"Issues": "error", "Fixes": "success", "Performance": "warning", "Summary": "accent"}
    ICONS = {"Issues": "!", "Fixes": "+", "Performance": "~", "Summary": ">"}

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(classes="review-card")
        self.result = result
        verb, _, target = str(result.get("current_input", "review")).partition(" ")
        self.border_title = f" {verb.upper() or 'REVIEW'}  {escape(target or 'code')} "
        context = _HELPER._graph_context_summary(result)
        self.border_subtitle = f" {_meta(result)}{' · ' + context if context else ''} "

    def render(self) -> Group:
        t = self.app.current_theme
        blocks: list[Any] = []
        score = self.result.get("confidence_score")
        if isinstance(score, (int, float)):
            blocks += [Text(f"Confidence {int(score)}%", style="dim"), Text("")]
        sections = [s for s in _HELPER._review_sections(self.result) if s[3]]
        for label, _icon, _color, items in sections:
            color = getattr(t, self.COLORS.get(label, "primary")) or t.primary
            table = Table.grid(padding=(0, 1))
            table.add_column(width=2)
            table.add_column(ratio=1)
            table.add_row(Text(""), Text(label, style=f"bold {t.foreground}"))
            for item in items:
                table.add_row(Text(self.ICONS.get(label, "•"), style=f"bold {color}"),
                              _HELPER._inline(item))
            blocks += [table, Text("")]
        if not sections:
            blocks.append(Text("✓ No issues found.", style=f"bold {t.success}"))
        return Group(*blocks[:-1] if blocks and blocks[-1].plain == "" else blocks)


class Patch(Static):
    def __init__(self, original: str, fixed: str) -> None:
        original = OutputFormatter._strip_file_header(original).strip("\n")
        fixed = OutputFormatter._strip_file_header(strip_terminal_controls(fixed)).strip("\n")
        diff = [d for d in difflib.unified_diff(
            [line.rstrip() for line in original.splitlines()],
            [line.rstrip() for line in fixed.splitlines()], lineterm="", n=2)][2:]
        diff = [d for d in diff if not d.startswith("@@")]
        plus = sum(1 for d in diff if d.startswith("+"))
        minus = sum(1 for d in diff if d.startswith("-"))
        super().__init__(Syntax("\n".join(diff) or "  (no changes)", "diff", theme="monokai",
                                background_color="default"), classes="patch")
        self.border_title = f" SUGGESTED CHANGE  +{plus} -{minus} "
        self.has_changes = bool(diff)


class ConfirmFix(ModalScreen[bool]):
    """Asks before saving a fix. A blocked fix needs the word 'force' typed, not a reflexive click."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    def __init__(self, target: str, findings: list[Finding]) -> None:
        super().__init__()
        self.target = target
        self.blocking = [f for f in findings if f.blocking]

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            if self.blocking:
                yield Static(Text.assemble(("Guardrails blocked this fix", "bold"), f" for {self.target}"), id="confirm-title")
                for f in self.blocking:
                    yield Static(Text(f"  ✗ {f.message}"), classes="confirm-finding")
                yield Static("Type [b]force[/b] to save it anyway, or press Cancel.", classes="confirm-hint")
                yield Input(placeholder="force", id="force-input")
            else:
                yield Static(Text.assemble(("Save the fix?", "bold"), f"  It is written next to {self.target} as *_fixed."), id="confirm-title")
            with Horizontal(id="confirm-buttons"):
                yield Button("Save fix", variant="warning" if self.blocking else "success", id="save",
                             disabled=bool(self.blocking))
                yield Button("Cancel", id="cancel")

    def on_input_changed(self, event: Input.Changed) -> None:
        self.query_one("#save", Button).disabled = event.value.strip().lower() != "force"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "save")

    def action_cancel(self) -> None:
        self.dismiss(False)


# ----------------------------------------------------------------------------------- app


class TRXApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "TRX"
    BINDINGS = [
        ("f2", "cycle_theme", "Theme"),
        ("f3", "toggle_inspector", "Inspect"),
        Binding("ctrl+l", "focus_command", "Command", priority=True),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, theme: str = DEFAULT_THEME, demo: bool = False, engine: Any = None) -> None:
        super().__init__()
        self._initial_theme = theme
        self.demo = demo
        self.engine = engine
        self.sys_status = None
        self.busy = False
        self.workspace_name = Path.cwd().name or "workspace"
        self.inspector_visible = False
        self._busy_started_at = 0.0
        self._busy_label = ""
        self._command_hint: Static | None = None
        self._activity: Static | None = None
        self._stages: list[str] = []

    # -- layout ------------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Masthead(id="masthead")
        with Horizontal(id="body"):
            with Vertical(id="nav"):
                yield Static("WORKSPACE", classes="nav-label")
                yield OptionList(*(Option(label, id=key) for key, label in NAV), id="nav-list")
                yield Static("SESSION", classes="nav-label")
                yield Static("", id="session")
            yield VerticalScroll(id="stream")
            with VerticalScroll(id="telemetry"):
                confidence = Vertical(Digits("--", id="confidence-digits"), classes="tile", id="confidence")
                confidence.border_title = "Confidence"
                yield confidence
                latency = Vertical(Sparkline([0.0], id="latency-spark"), Static("", id="latency-text"), classes="tile")
                latency.border_title = "Response time"
                yield latency
                pipeline = Vertical(Static("[dim]idle[/]", id="pipeline-text"),
                                    ProgressBar(total=1, show_eta=False, show_percentage=False, id="pipeline-bar"),
                                    classes="tile")
                pipeline.border_title = "Review pipeline"
                yield pipeline
                guard = Vertical(Static("[dim]no checks yet[/]", id="guard-text"), classes="tile")
                guard.border_title = "Safety checks"
                yield guard
        yield Static("", id="activity", markup=False)
        yield Static("Enter submit   ·   Ctrl+L focus   ·   F3 diagnostics", id="command-hint")
        yield Input(placeholder="Ask a question or enter a command…", id="command")
        yield Footer()

    def on_mount(self) -> None:
        for theme in THEMES:
            self.register_theme(theme)
        self.theme = self._initial_theme
        self.query_one("#command", Input).focus()
        self.query_one("#command", Input).border_title = " Command "
        self.screen.set_class(self.inspector_visible, "inspecting")
        self._command_hint = self.query_one("#command-hint", Static)
        self._activity = self.query_one("#activity", Static)
        self._activity.display = False
        self.set_interval(0.15, self._animate_busy)
        self._update_session()
        if self.demo:
            from tui.demo import DEMO_STATUS, demo_results

            self.sys_status = DEMO_STATUS
            for prompt, result in demo_results():
                self._add_user(prompt)
                self.show_result(result, for_fix=False)
            return
        self._mount(Static(Text.assemble(
            ("Workspace ready", "bold"),
            "\n\nAsk a development question, or inspect a file.\n\n",
            ("review path/to/file.py", "bold"), "   review code\n",
            ("fix path/to/file.py", "bold"), "      preview a patch\n",
            ("help", "bold"), "                     commands and shortcuts",
        ), id="welcome"))
        if self.engine is None:
            self._note("Starting TRX engine…", "dim")
            self.boot()
        else:
            self.refresh_status()
        self.set_interval(20, self.refresh_status)

    def on_resize(self, event: Resize) -> None:
        screen = self.screen_stack[0]
        screen.set_class(event.size.width < 120, "compact")
        screen.set_class(event.size.width < 90, "narrow")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        command = self.query_one("#command", Input)
        action = event.option.id
        if action == "help":
            self._mount(Markdown(HELP_TEXT, classes="answer"))
        elif action in {"review", "fix"}:
            command.value = f"{action} "
            command.cursor_position = len(command.value)
        elif action == "chat":
            command.value = ""
        command.focus()

    # -- workers (run off the UI thread) ---------------------------------------------------

    @work(thread=True, exclusive=True, group="boot")
    def boot(self) -> None:
        from tui.engine import Engine

        try:
            engine = Engine()
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self.call_from_thread(self._note, f"Engine failed to start: {exc}", "error")
            return
        self.engine = engine
        self.call_from_thread(self._note, "Ready. Type a question, or `review <file>`. `help` lists commands.", "dim")
        self.refresh_status()

    @work(thread=True, exclusive=True, group="status")
    def refresh_status(self) -> None:
        if self.engine is None:
            return
        try:
            status = self.engine.status()
        except Exception:  # noqa: BLE001 - status lights must never crash the app
            return
        self.call_from_thread(setattr, self, "sys_status", status)

    @work(thread=True, exclusive=True, group="ops")
    def run_chat(self, text: str) -> None:
        try:
            result = self.engine.chat(text, progress=self._worker_progress)
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._finish_error, str(exc))
            return
        self.call_from_thread(self.show_result, result, False)

    @work(thread=True, exclusive=True, group="ops")
    def run_review(self, target: str, for_fix: bool) -> None:
        try:
            result = self.engine.review(target, for_fix=for_fix, progress=self._worker_progress)
        except FileNotFoundError:
            self.call_from_thread(self._finish_error, f"File or folder not found: {target}")
            return
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._finish_error, str(exc))
            return
        self.call_from_thread(self.show_result, result, for_fix)

    # -- input -----------------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "command":
            return
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return
        text = normalize_command(raw)
        if text.startswith("/") and text[1:].partition(" ")[0].lower() in {"clear", "theme", "inspect"}:
            text = text[1:]
        command, _, arg = text.partition(" ")
        command = command.lower()
        if command in {"exit", "quit"}:
            self.exit()
            return
        if command == "clear":
            self.query_one("#stream").remove_children()
            return
        if command == "theme":
            self.action_cycle_theme()
            return
        if command == "inspect":
            self.action_toggle_inspector()
            return
        if command == "help":
            self._add_user(raw)
            self._mount(Markdown(HELP_TEXT, classes="answer"))
            return
        if self.engine is None:
            event.input.value = raw
            self._note("Engine is still starting; try again in a moment.", "warning")
            return
        if self.busy:
            event.input.value = raw
            self._note("Still working on the previous request.", "warning")
            return
        self._add_user(raw)
        if command in {"review", "fix"}:
            if not arg.strip():
                self._note(f"Usage: {command} <file>", "warning")
                return
            self._start_busy(f"{command} {arg.strip()}")
            self.run_review(arg.strip(), command == "fix")
        else:
            self._start_busy("thinking")
            self.run_chat(raw)

    # -- rendering results -------------------------------------------------------------------

    def show_result(self, result: dict[str, Any], for_fix: bool = False) -> None:
        self.busy = False
        self._hide_activity()
        self._refresh_command_hint()
        if result.get("response_mode") == "analysis":
            self._mount(ReviewCard(result))
            fixed = str(result.get("fixed_code", "")).strip()
            original = str(result.get("original_code", ""))
            patch = Patch(original, fixed) if fixed and original else None
            if patch is not None and patch.has_changes:
                self._mount(patch)
            if for_fix and fixed and patch is not None and patch.has_changes:
                self._confirm_fix(result)
            elif for_fix:
                self._note("No code changes suggested; nothing to save.", "dim")
        else:
            reply = strip_terminal_controls(str(result.get("chat_response", ""))).strip() or "…"
            answer = Markdown(reply, classes="answer")
            answer.border_subtitle = f" {_meta(result)} "
            self._mount(answer)
        self._update_telemetry(result)
        self._update_session()

    def _confirm_fix(self, result: dict[str, Any]) -> None:
        target = str(result.get("review_target", ""))
        findings = [f for f in result.get("guardrails", []) if isinstance(f, Finding)]

        def done(save: bool | None) -> None:
            if not save:
                self._note("Fix not saved.", "dim")
                return
            try:
                path = self.engine.apply_fix(target, result)
                self._note(f"Fix saved as {path}", "success")
            except Exception as exc:  # noqa: BLE001
                self._note(f"Could not save the fix: {exc}", "error")

        self.push_screen(ConfirmFix(target, findings), done)

    def _update_telemetry(self, result: dict[str, Any]) -> None:
        score = result.get("confidence_score")
        if not isinstance(score, (int, float)):
            conf = result.get("intent_confidence")
            score = round(float(conf) * 100) if isinstance(conf, (int, float)) else None
        self.query_one("#confidence-digits", Digits).update(f"{int(score)}%" if isinstance(score, (int, float)) else "--")

        stats = getattr(self.engine, "stats", None)
        latencies = list(getattr(stats, "latencies", []) or [])
        if not latencies and isinstance(result.get("elapsed_seconds"), (int, float)):
            latencies = [float(result["elapsed_seconds"])]
        if latencies:
            # A single sample would fill the chart as a solid block; anchor it at zero.
            self.query_one("#latency-spark", Sparkline).data = latencies if len(latencies) > 1 else [0.0] + latencies
            p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
            self.query_one("#latency-text", Static).update(
                f"[dim]last[/] {latencies[-1]:.1f}s   [dim]p95[/] {p95:.1f}s   "
                f"[dim]avg[/] {sum(latencies) / len(latencies):.1f}s")

        bar = self.query_one("#pipeline-bar", ProgressBar)
        bar.update(total=1, progress=1)
        if result.get("response_mode") == "analysis":
            steps = _status_value(result, "steps") or "?"
            critic = _status_value(result, "critic_score") or "-"
            lines = [f"[green]✓[/] source       [b]{_source_label(result)}[/]",
                     f"[green]✓[/] agent steps  [b]{steps}[/]",
                     f"[green]✓[/] critic score [b]{critic}[/]"]
            if "cache_hit" in [str(s) for s in result.get("system_status", [])]:
                lines.append("[green]✓[/] served from cache")
        else:
            lines = [f"[green]✓[/] chat · [b]{_source_label(result)}[/]"]
        self.query_one("#pipeline-text", Static).update("\n".join(lines))

        findings = [f for f in result.get("guardrails", []) if isinstance(f, Finding)]
        if findings:
            rows = [f"[{'red' if f.blocking else 'yellow'}]{'✗' if f.blocking else '!'}[/] {f.message}" for f in findings]
            self.query_one("#guard-text", Static).update("\n".join(rows))
        elif result.get("response_mode") == "analysis":
            self.query_one("#guard-text", Static).update(
                "No findings from available checks." if "guardrails" in result else
                "No fix-safety checks performed for this request.")
        else:
            self.query_one("#guard-text", Static).update("No fix-safety checks performed for this request.")

    def _update_session(self) -> None:
        stats = getattr(self.engine, "stats", None)
        if stats is None:
            self.query_one("#session", Static).update("[dim]no runs yet[/]")
            return
        self.query_one("#session", Static).update(
            Text(f"Runs     {stats.runs}\nReviews  {stats.reviews}\n"
                 f"Cache    {stats.cache_rate:.0%}\nAverage  {stats.avg_latency:.1f}s"))

    # -- small helpers -------------------------------------------------------------------------

    def _mount(self, widget: Any) -> None:
        stream = self.query_one("#stream", VerticalScroll)
        stream.mount(widget)
        # Newly mounted content has no final height until the next layout pass.
        self.call_after_refresh(stream.scroll_end, animate=False)

    def _add_user(self, text: str) -> None:
        for welcome in self.query("#welcome"):
            welcome.remove()
        self._mount(Static(f"> {text}", classes="user-msg", markup=False))

    def _note(self, text: str, level: str = "dim") -> None:
        color = {"error": "red", "warning": "yellow", "success": "green"}.get(level, "")
        widget = Static(Text(text, style=color or "dim"), classes="note")
        self._mount(widget)

    def _start_busy(self, label: str) -> None:
        self.busy = True
        self._busy_started_at = time.monotonic()
        self._busy_label = label
        self._stages = []
        if self._activity is not None:
            self._activity.display = True
        self._refresh_command_hint()
        self.query_one("#pipeline-text", Static).update(Text.assemble(("◐ ", "yellow"), "working: ", (label, "bold")))
        self.query_one("#pipeline-bar", ProgressBar).update(total=None)
        self.query_one("#guard-text", Static).update("Checks pending for this request.")

    def _finish_error(self, message: str) -> None:
        self.busy = False
        self._hide_activity()
        self._refresh_command_hint()
        self.query_one("#pipeline-text", Static).update("[red]✗[/] failed")
        self.query_one("#pipeline-bar", ProgressBar).update(total=1, progress=0)
        self.query_one("#guard-text", Static).update("Checks incomplete for this request.")
        self._note(message, "error")

    def _refresh_command_hint(self) -> None:
        if self._command_hint is None or not self._command_hint.is_mounted:
            return
        if self.busy:
            elapsed = max(0, int(time.monotonic() - self._busy_started_at))
            frame = ("[.  ]", "[ . ]", "[  .]", "[ . ]")[int(time.monotonic() * 6) % 4]
            hint = Text.assemble((f"{frame} Working", "bold"), f"  ·  {self._busy_label}  ·  {elapsed}s")
            if self._activity is not None and self._activity.is_mounted:
                current = self._stages[-1] if self._stages else "Preparing request"
                trail = " > ".join(s.partition(" · ")[0] for s in self._stages[-5:])
                self._activity.update(Text.assemble(
                    (current, "bold"), (f"\n{trail}" if trail else "", "dim")))
        else:
            hint = Text("Enter submit   ·   Ctrl+L focus   ·   F3 diagnostics")
        self._command_hint.update(hint)

    def _animate_busy(self) -> None:
        if self.busy:
            self._refresh_command_hint()

    def _hide_activity(self) -> None:
        if self._activity is not None:
            self._activity.display = False

    def _worker_progress(self, label: str) -> None:
        self.call_from_thread(self._show_progress, label)

    def _show_progress(self, label: str) -> None:
        if not self.busy:
            return
        label = strip_terminal_controls(label)
        if not self._stages or self._stages[-1] != label:
            self._stages.append(label)
            self._stages = self._stages[-12:]
        self._refresh_command_hint()
        trace = Text()
        for previous in self._stages[:-1]:
            trace.append(f"  {previous}\n", style="dim")
        trace.append(f"> {label}", style="bold")
        self.query_one("#pipeline-text", Static).update(trace)

    def action_cycle_theme(self) -> None:
        names = [t.name for t in THEMES]
        self.theme = names[(names.index(self.theme) + 1) % len(names)] if self.theme in names else names[0]

    def action_focus_command(self) -> None:
        self.query_one("#command", Input).focus()

    def action_toggle_inspector(self) -> None:
        self.inspector_visible = not self.inspector_visible
        self.screen_stack[0].set_class(self.inspector_visible, "inspecting")
        if self.inspector_visible and self.size.width < 120:
            self._note("Diagnostics need at least 120 terminal columns. Widen the terminal to show them.", "dim")


def main() -> None:
    import sys

    TRXApp(demo="--demo" in sys.argv).run()


if __name__ == "__main__":
    main()
