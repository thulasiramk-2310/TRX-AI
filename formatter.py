"""Futuristic dashboard formatting utilities for TRX-AI outputs."""

from __future__ import annotations

import difflib
import re
import time
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class OutputFormatter:
    """Renders a futuristic multi-panel dashboard for chat and analysis responses."""

    def __init__(
        self,
        console: Console,
        typing_effect: bool = False,
        typing_delay: float = 0.01,
        ui_transitions: bool = True,
        ui_transition_delay: float = 0.35,
    ) -> None:
        self.console = console
        self.typing_effect = typing_effect
        self.typing_delay = typing_delay
        self.ui_transitions = ui_transitions
        self.ui_transition_delay = ui_transition_delay

    @staticmethod
    def _section_items(analysis: dict[str, Any], key: str, fallback: str) -> list[str]:
        value = analysis.get(key, [])
        if isinstance(value, list) and value:
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return [fallback]

    @staticmethod
    def structured_text(analysis: dict[str, Any]) -> str:
        def items(key: str, fallback: str) -> list[str]:
            return OutputFormatter._section_items(analysis, key, fallback)

        def code_mode() -> bool:
            intent = str(analysis.get("intent", "")).lower()
            return intent == "review" or bool(str(analysis.get("fixed_code", "")).strip())

        def complexity_notes() -> list[str]:
            pool = []
            for section_key in ("predictions", "improvements", "debug_analysis"):
                val = analysis.get(section_key, [])
                if isinstance(val, list):
                    pool.extend(str(v).strip() for v in val if str(v).strip())
                elif isinstance(val, str) and val.strip():
                    pool.append(val.strip())
            selected = []
            for line in pool:
                low = line.lower()
                if "o(" in low or "complexity" in low or "exponential" in low or "quadratic" in low:
                    selected.append(line)
            if selected:
                return selected[:4]
            return []

        def optimization_approaches() -> list[str]:
            pool = []
            for section_key in ("improvements", "predictions"):
                val = analysis.get(section_key, [])
                if isinstance(val, list):
                    pool.extend(str(v).strip() for v in val if str(v).strip())
                elif isinstance(val, str) and val.strip():
                    pool.append(val.strip())
            selected = []
            for line in pool:
                low = line.lower()
                if any(
                    marker in low
                    for marker in (
                        "optimiz", "memo", "cache", "reduce", "refactor", "improve performance",
                        "time complexity", "space complexity", "early exit", "index bound",
                    )
                ):
                    selected.append(line)
            deduped = []
            seen = set()
            for line in selected:
                key = line.lower()
                if key not in seen:
                    deduped.append(line)
                    seen.add(key)
            return deduped[:5]

        def changed_lines_summary() -> list[str]:
            original = str(analysis.get("original_code", "")).strip()
            fixed = str(analysis.get("fixed_code", "")).strip()
            if not original or not fixed:
                return []
            old_lines = original.splitlines()
            new_lines = fixed.splitlines()
            matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines)
            out: list[str] = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                old_snippet = " | ".join(line.strip() for line in old_lines[i1:i2] if line.strip())[:100]
                new_snippet = " | ".join(line.strip() for line in new_lines[j1:j2] if line.strip())[:100]
                out.append(f"OLD: {old_snippet or '<none>'}")
                out.append(f"NEW: {new_snippet or '<none>'}")
                if len(out) >= 8:
                    break
            reason = ""
            final_insight = analysis.get("final_insight", [])
            if isinstance(final_insight, list) and final_insight:
                reason = str(final_insight[0]).strip()
            elif isinstance(final_insight, str):
                reason = final_insight.strip()
            if reason:
                out.insert(0, f"Reason: {reason}")
            return out

        lines: list[str] = []

        lines.append("[DEBUG ANALYSIS]")
        for item in items("debug_analysis", "No debug analysis available"):
            lines.append(f"- {item}")

        lines.append("")
        lines.append("[IMPROVEMENTS]")
        for item in items("improvements", "No improvements available"):
            lines.append(f"- {item}")

        lines.append("")
        lines.append("[PREDICTIONS]")
        for item in items("predictions", "No predictions available"):
            lines.append(f"- {item}")

        lines.append("")
        lines.append("[FINAL INSIGHT]")
        for item in items("final_insight", "Prioritize one corrective action now"):
            lines.append(f"- {item}")

        if code_mode():
            changes = changed_lines_summary()
            if changes:
                lines.append("")
                lines.append("[CHANGES APPLIED]")
                for item in changes:
                    lines.append(f"- {item}")

            complexity = complexity_notes()
            if complexity:
                lines.append("")
                lines.append("[CODE COMPLEXITY]")
                for item in complexity:
                    lines.append(f"- {item}")

            approaches = optimization_approaches()
            if approaches:
                lines.append("")
                lines.append("[OPTIMIZATION APPROACHES]")
                for item in approaches:
                    lines.append(f"- {item}")

            fixed_code = str(analysis.get("fixed_code", "")).strip()
            if fixed_code:
                lines.append("")
                lines.append("[LLM FIXED CODE]")
                lines.append("```")
                lines.extend(fixed_code.splitlines())
                lines.append("```")

        lines.append("")
        lines.append("[CONFIDENCE SCORE]")
        lines.append(f"- {int(analysis.get('confidence_score', 50))}%")

        return "\n".join(lines)

    def _supports_utf8(self) -> bool:
        stream = getattr(self.console, "file", None)
        encoding = str(getattr(stream, "encoding", "") or "").lower()
        return "utf" in encoding

    def _progress_bar(self, percent: int, width: int = 26) -> str:
        clamped = max(0, min(100, int(percent)))
        filled = int((clamped / 100.0) * width)
        if self._supports_utf8():
            return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    @staticmethod
    def _markdown_from_items(items: list[str], fallback: str, *, limit: int = 6) -> str:
        cleaned: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            text = re.sub(r"^[-*]\s+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            cleaned.append(text)
        if not cleaned:
            return f"- {fallback}"
        return "\n".join(f"- {line}" for line in cleaned[:limit])

    @staticmethod
    def _is_placeholder_items(items: list[str], placeholder_texts: set[str]) -> bool:
        if not items:
            return True
        normalized = {str(item).strip().lower() for item in items if str(item).strip()}
        if not normalized:
            return True
        return normalized.issubset({value.lower() for value in placeholder_texts})

    @staticmethod
    def _extract_raw_section(raw_text: str, headers: list[str], *, max_items: int = 8) -> list[str]:
        if not raw_text.strip():
            return []

        lines = raw_text.splitlines()

        def normalize_header(line: str) -> str:
            cleaned = line.strip()
            cleaned = re.sub(r"^[#>\-*\s`]+", "", cleaned)
            cleaned = cleaned.replace("**", "").replace("__", "").strip()
            cleaned = cleaned.rstrip(":").strip().upper()
            return cleaned

        section_header_set = {
            "CODE DEBUG",
            "DEBUG ANALYSIS",
            "CODE IMPROVEMENTS",
            "IMPROVEMENTS",
            "PERFORMANCE",
            "PERFORMANCE / PREDICTIONS",
            "SECURITY",
            "FIX SUGGESTIONS",
            "FIXED CODE",
            "FINAL SUMMARY",
            "CONFIDENCE",
        }

        target_headers = {header.upper() for header in headers}
        start_idx = -1
        for idx, raw in enumerate(lines):
            if normalize_header(raw) in target_headers:
                start_idx = idx + 1
                break
        if start_idx == -1:
            return []

        collected: list[str] = []
        in_code_block = False
        for idx in range(start_idx, len(lines)):
            current = lines[idx].strip()
            if not current:
                continue
            if current.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            normalized = normalize_header(current)
            if normalized in section_header_set:
                break
            if re.fullmatch(r"[-=_]{3,}", current):
                continue

            item = re.sub(r"^[>\-*\u2022\d\.\)\s]+", "", current).strip()
            item = item.replace("**", "").replace("__", "").strip()
            if not item:
                continue
            collected.append(item)
            if len(collected) >= max_items:
                break

        return collected

    def _smooth_print(self, renderable: Any) -> None:
        try:
            self.console.print(renderable)
        except UnicodeEncodeError:
            safe_text = self._renderable_to_safe_text(renderable)
            self.console.print(safe_text)
        if self.ui_transitions:
            time.sleep(max(0.04, min(0.18, self.ui_transition_delay / 2.5)))

    def _renderable_to_safe_text(self, renderable: Any) -> str:
        with self.console.capture() as capture:
            self.console.print(renderable)
        rendered = capture.get()
        stream = getattr(self.console, "file", None)
        encoding = getattr(stream, "encoding", None) or "utf-8"
        return rendered.encode(encoding, errors="replace").decode(encoding, errors="replace")

    def _build_modules_panel(self, analysis: dict[str, Any]) -> Panel:
        active = {
            str(item).strip().lower()
            for item in analysis.get("active_agents", [])
            if str(item).strip()
        }
        current_input = str(analysis.get("current_input", "")).strip().lower()
        statuses = [str(item).lower() for item in analysis.get("system_status", [])]
        intent = str(analysis.get("intent", "")).lower()
        is_review_flow = (
            intent == "review"
            or current_input.startswith("review ")
            or current_input.startswith("fix ")
            or any("intent=review" in item for item in statuses)
            or bool(str(analysis.get("fixed_code", "")).strip())
        )

        rows = [
            ("[D]", "Debug Agent", ("debug" in active) or is_review_flow),
            ("[I]", "Improve Agent", ("improve" in active) or is_review_flow),
            ("[P]", "Predict Agent", ("predict" in active) or is_review_flow),
            ("[R]", "Review Agent", ("review" in active) or is_review_flow),
        ]

        modules = Table.grid(expand=True)
        modules.add_column(ratio=1)
        for icon, label, is_active in rows:
            status = "[bright_cyan]ACTIVE[/bright_cyan]" if is_active else "[dim]IDLE[/dim]"
            border = "dark_orange3" if is_active else "#5f4b8b"
            glow_dot = "[bright_cyan]*[/bright_cyan]" if is_active else "[dim].[/dim]"
            item = Table.grid(expand=True)
            item.add_column(ratio=1)
            item.add_column(width=3, justify="right")
            item.add_row(f"{icon} [bold]{label}[/bold]\n{status}", glow_dot)
            modules.add_row(
                Panel(
                    item,
                    border_style=border,
                    box=ROUNDED,
                    padding=(0, 1),
                    style="on #130d1f",
                )
            )

        return Panel(
            modules,
            title="[bold #ffb266]TRX-AI MODULES[/bold #ffb266]",
            border_style="#ff8c42",
            box=ROUNDED,
            style="on #0f0818",
            padding=(1, 1),
        )

    def _build_analysis_panel(self, analysis: dict[str, Any]) -> Panel:
        if analysis.get("response_mode") != "analysis":
            chat_response = str(analysis.get("chat_response", "I am here to help.")).strip()
            if not chat_response:
                chat_response = "I am here to help."
            confidence_label = "High" if analysis.get("analysis_source") == "llm" else "Medium"
            branded = (
                "[TRX RESPONSE]\n\n"
                "💬 Response:\n"
                f"{chat_response}\n\n"
                "Insight:\n"
                "Ask follow-up questions for deeper detail.\n\n"
                f"Confidence: {confidence_label}"
            )

            body = Table.grid(expand=True)
            body.add_column(ratio=1)
            body.add_row(
                Panel(
                    Markdown(branded, code_theme="monokai"),
                    title="[bold]TRX RESPONSE[/bold]",
                    border_style="#7dd3fc",
                    box=ROUNDED,
                    style="on #102030",
                    padding=(0, 1),
                )
            )
            body.add_row(
                Panel(
                    Markdown("- Ask a follow-up for deeper detail or examples.", code_theme="monokai"),
                    title="[bold]NEXT STEP[/bold]",
                    border_style="#c084fc",
                    box=ROUNDED,
                    style="on #1d1330",
                    padding=(0, 1),
                )
            )

            return Panel(
                body,
                title="[bold #d8b4fe]ANALYSIS OUTPUT[/bold #d8b4fe]",
                border_style="#8b5cf6",
                box=ROUNDED,
                style="on #0c0716",
                padding=(1, 1),
            )

        raw_output = str(analysis.get("raw_llm_output", "")).strip()

        debug_items = self._section_items(analysis, "debug_analysis", "No debug issues detected.")
        if self._is_placeholder_items(debug_items, {"No critical bugs were explicitly identified.", "No debug issues detected."}):
            extracted = self._extract_raw_section(raw_output, ["CODE DEBUG", "DEBUG ANALYSIS"], max_items=8)
            if extracted:
                debug_items = extracted

        improve_items = self._section_items(analysis, "improvements", "No improvement suggestions yet.")
        if self._is_placeholder_items(improve_items, {"Apply incremental refactoring for readability and maintainability.", "No improvement suggestions yet."}):
            extracted = self._extract_raw_section(raw_output, ["CODE IMPROVEMENTS", "IMPROVEMENTS", "SECURITY"], max_items=8)
            if extracted:
                improve_items = extracted

        performance_items = self._section_items(analysis, "predictions", "No performance insights yet.")
        if self._is_placeholder_items(performance_items, {"LLM output parsing incomplete - check raw output", "No performance insights yet."}):
            extracted = self._extract_raw_section(raw_output, ["PERFORMANCE"], max_items=8)
            if extracted:
                performance_items = extracted

        summary_items = self._section_items(analysis, "final_insight", "Prioritize one high-impact fix first.")
        if self._is_placeholder_items(summary_items, {"Prioritize correctness issues first, then refactor and optimize.", "Prioritize one high-impact fix first."}):
            extracted = self._extract_raw_section(raw_output, ["FINAL SUMMARY"], max_items=4)
            if extracted:
                summary_items = extracted

        section_rows = [
            ("DEBUG", debug_items, "#ff6b6b", "#2a0d14"),
            ("IMPROVEMENTS", improve_items, "#c084fc", "#1d1330"),
            ("PERFORMANCE / PREDICTIONS", performance_items, "#7dd3fc", "#102030"),
            ("SUMMARY", summary_items, "#ffb266", "#2b1b12"),
        ]

        body = Table.grid(expand=True)
        body.add_column(ratio=1)
        for title, items, border, bg in section_rows:
            md_text = self._markdown_from_items(items, "No data available.", limit=6)
            block = Panel(
                Markdown(md_text, code_theme="monokai"),
                title=f"[bold]{title}[/bold]",
                border_style=border,
                box=ROUNDED,
                style=f"on {bg}",
                padding=(0, 1),
            )
            body.add_row(block)

        return Panel(
            body,
            title="[bold #d8b4fe]ANALYSIS OUTPUT[/bold #d8b4fe]",
            border_style="#8b5cf6",
            box=ROUNDED,
            style="on #0c0716",
            padding=(1, 1),
        )

    def _build_input_results_panel(self, analysis: dict[str, Any]) -> Panel:
        raw_input = str(analysis.get("current_input", "")).strip()
        if not raw_input:
            raw_input = "Enter problem or code..."

        preview_lines = []
        if analysis.get("response_mode") != "analysis":
            chat_preview = str(analysis.get("chat_response", "")).strip()
            if chat_preview:
                preview_lines.append(chat_preview)
        for section in ("debug_analysis", "improvements", "predictions", "final_insight"):
            values = analysis.get(section, [])
            if isinstance(values, list):
                preview_lines.extend(str(v).strip() for v in values if str(v).strip())
            elif isinstance(values, str) and values.strip():
                preview_lines.append(values.strip())

        if not preview_lines:
            raw_output = str(analysis.get("raw_llm_output", "")).strip()
            preview_lines = self._extract_raw_section(
                raw_output,
                ["CODE DEBUG", "CODE IMPROVEMENTS", "PERFORMANCE", "FINAL SUMMARY"],
                max_items=3,
            )
        preview = preview_lines[:3] if preview_lines else ["No analysis preview available yet."]
        recent = [str(item) for item in analysis.get("system_status", [])[:4]]
        if not recent:
            recent = ["Run a review or fix command to populate recent analyses."]

        content = Table.grid(expand=True)
        content.add_column(ratio=1)

        input_box = Panel(
            Text(f"{raw_input[:90]}", style="white"),
            title="[bold #7dd3fc]INPUT[/bold #7dd3fc]",
            border_style="#22d3ee",
            box=ROUNDED,
            style="on #0f1a22",
            padding=(0, 1),
        )
        preview_box = Panel(
            Markdown(self._markdown_from_items(preview, "No results yet.", limit=3), code_theme="monokai"),
            title="[bold #ffb266]Results Preview[/bold #ffb266]",
            border_style="#f59e0b",
            box=ROUNDED,
            style="on #22160d",
            padding=(0, 1),
        )
        recent_box = Panel(
            Markdown(self._markdown_from_items(recent, "No history entries.", limit=4), code_theme="monokai"),
            title="[bold #d8b4fe]Recent Analyses[/bold #d8b4fe]",
            border_style="#a78bfa",
            box=ROUNDED,
            style="on #18112a",
            padding=(0, 1),
        )

        content.add_row(input_box)
        content.add_row(preview_box)
        content.add_row(recent_box)

        return Panel(
            content,
            title="[bold #93c5fd]INPUT & RESULTS[/bold #93c5fd]",
            border_style="#38bdf8",
            box=ROUNDED,
            style="on #0b1019",
            padding=(1, 1),
        )

    def _build_status_bar(self, analysis: dict[str, Any], mode: str, total_runs: int | None) -> Panel:
        model_name = str(analysis.get("model_name", "qwen3:8b"))
        score = int(analysis.get("confidence_score", int(float(analysis.get("intent_confidence", 0.8)) * 100)))
        run_label = total_runs if total_runs is not None else "-"
        score_bar = escape(self._progress_bar(score, width=12))

        status = Table.grid(expand=True)
        status.add_column(ratio=1)
        status.add_column(ratio=1)
        status.add_column(ratio=1)
        status.add_column(ratio=1)
        status.add_row(
            f"[bold]Run:[/bold] {run_label}",
            f"[bold]Model:[/bold] {model_name[:14]}",
            f"[bold]Mode:[/bold] {mode}",
            f"[bold]Score:[/bold] {score}% [bright_cyan]{score_bar}[/bright_cyan]",
        )

        return Panel(
            status,
            title="[bold #86efac]SYSTEM STATUS[/bold #86efac]",
            border_style="#22c55e",
            box=ROUNDED,
            style="on #0b1611",
            padding=(0, 1),
        )

    def render(self, analysis: dict[str, Any], mode: str, total_runs: int | None = None) -> None:
        if not analysis.get("current_input"):
            analysis["current_input"] = "Enter problem or code..."

        width = self.console.size.width
        modules = self._build_modules_panel(analysis)
        analysis_panel = self._build_analysis_panel(analysis)
        input_results = self._build_input_results_panel(analysis)

        grid = Table.grid(expand=True)
        if width < 110:
            # Compact stacked layout for narrow terminals.
            grid.add_column(ratio=1)
            if width >= 85:
                grid.add_row(modules)
            grid.add_row(analysis_panel)
            grid.add_row(input_results)
        else:
            # Wide layout with three columns.
            grid.add_column(ratio=24)
            grid.add_column(ratio=46)
            grid.add_column(ratio=30)
            grid.add_row(modules, analysis_panel, input_results)

        shell = Panel(
            grid,
            title="[bold #f8fafc]TRX-AI Futuristic Developer Dashboard[/bold #f8fafc]",
            subtitle="[dim]Neon Glass Interface[/dim]",
            border_style="#7c3aed",
            box=ROUNDED,
            style="on #07050d",
            padding=(1, 1),
            expand=True,
        )

        # Redraw dashboard cleanly so prompt interaction doesn't keep pushing prior frames upward.
        self.console.clear()
        self._smooth_print(shell)
        self._smooth_print(self._build_status_bar(analysis, mode, total_runs))

    def render_help_dashboard(
        self,
        *,
        mode: str,
        total_runs: int,
        model_name: str,
        active_agents: list[str],
    ) -> None:
        help_payload = {
            "response_mode": "analysis",
            "intent": "help",
            "current_input": "help",
            "model_name": model_name,
            "active_agents": active_agents,
            "debug_analysis": [
                "help -> Show this command guide. Example: help",
                "history -> List previous prompts. Example: history",
                "save <path> -> Save session history JSON. Example: save sessions/my_run.json",
                "export <file> -> Export latest analysis report (.txt/.pdf). Example: export review_report.pdf",
                "export compare <file> -> Compare last two analyses in PDF. Example: export compare run_diff.pdf",
            ],
            "improvements": [
                "review <file|folder> -> Run code review. Example: review dsa_test.py",
                "fix <file> -> Generate fixed code with preview + confirm (y/n). Example: fix demo.py",
                "watch <folder> -> Auto-review changes in supported code files. Example: watch .",
                "agents all | agents debug improve predict -> Control agent modules. Example: agents all",
            ],
            "predictions": [
                "mode debug|optimize|predict -> Change analysis profile. Example: mode optimize",
                "Model in use: " + model_name,
                "Supported exit commands: exit, quit, or Ctrl+C",
                "Best flow: review file -> inspect output -> fix file -> export report",
            ],
            "final_insight": [
                "Quick Start 1: review dsa_test.py",
                "Quick Start 2: fix dsa_test.py",
                "Quick Start 3: export report.pdf",
                "Tip: Keep review targets focused (single file) for higher quality fixes",
            ],
            "system_status": [
                "help_view",
                "source=local_commands",
                "ui=dashboard",
                "loader=enabled",
                f"model={model_name}",
            ],
            "confidence_score": 100,
        }
        self.render(help_payload, mode=mode, total_runs=total_runs)
