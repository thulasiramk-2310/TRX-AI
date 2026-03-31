"""Minimal formatting utilities for Reality Debugger outputs."""

from __future__ import annotations

import difflib
import re
from typing import Any

from rich.console import Console


class OutputFormatter:
    """Renders minimal CLI output for chat and analysis responses."""

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

    def render(self, analysis: dict[str, Any], mode: str, total_runs: int | None = None) -> None:
        from rich.panel import Panel
        from rich.text import Text

        intent = str(analysis.get("intent", "chat"))
        source = (
            str(analysis.get("analysis_source", analysis.get("intent_source", "rule")))
            if analysis.get("response_mode") == "analysis"
            else str(analysis.get("intent_source", "rule"))
        )
        encoding = (getattr(getattr(self.console, "file", None), "encoding", "") or "").lower()
        bullet = "-" if encoding in {"ascii"} else "•"
        terminal_codec = "cp1252" if encoding in {"cp1252", "windows-1252"} else ("ascii" if encoding == "ascii" else "")

        def safe_terminal_text(value: str) -> str:
            text = str(value)
            if not terminal_codec:
                return text
            return text.encode(terminal_codec, errors="replace").decode(terminal_codec, errors="replace")

        def clean_line(value: str) -> str:
            text = re.sub(r"\*\*(.*?)\*\*", r"\1", str(value))
            text = re.sub(r"\s+", " ", text).strip()
            return safe_terminal_text(text)

        def compact_line(value: str, max_len: int = 80) -> str:
            text = clean_line(value)
            if len(text) <= max_len:
                return text
            return text[: max_len - 3].rstrip() + "..."

        def with_line_hint(value: str) -> str:
            text = compact_line(value)
            match = re.search(
                r"\bline(?:s)?\s*(\d+(?:\s*[-–]\s*\d+)?)\b\)?\s*[:\-]?\s*(.*)",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                line_info = match.group(1).replace(" ", "")
                message = re.sub(r"^[\)\:\-\s]+", "", match.group(2)).strip()
                if not message:
                    return f"[Line {line_info}]"
                return f"[Line {line_info}] {message}"
            return text

        def section(text_obj: Text, title: str, items: list[str], *, limit: int = 3) -> None:
            cleaned = [item for item in items if clean_line(item)]
            if not cleaned:
                return
            title_icons = {
                "DEBUG": "[D]",
                "IMPROVEMENTS": "[I]",
                "PERFORMANCE": "[P]",
                "FIX": "[F]",
                "SUMMARY": "[S]",
                "CONFIDENCE": "[C]",
            }
            icon = title_icons.get(title, "[*]")
            text_obj.append(f"{icon} {title}\n", style="bold cyan")
            for item in cleaned[:limit]:
                rendered = with_line_hint(item)
                style = "white"
                if title == "DEBUG":
                    lowered = rendered.lower()
                    if "error" in lowered or "crash" in lowered:
                        style = "bold red"
                text_obj.append(f"{bullet} {rendered}\n", style=style)
            text_obj.append("\n")

        def divider(text_obj: Text) -> None:
            divider_char = "-" if encoding in {"cp1252", "ascii"} else "─"
            text_obj.append(divider_char * 40 + "\n", style="dim")
            text_obj.append("\n")

        content = Text(no_wrap=False, overflow="fold")

        if analysis.get("response_mode") == "analysis":
            debug_items = self._section_items(analysis, "debug_analysis", "")
            improve_items = self._section_items(analysis, "improvements", "")
            performance_items = self._section_items(analysis, "predictions", "")
            fix_items = self._section_items(analysis, "final_insight", "")

            section(content, "DEBUG", debug_items, limit=3)
            divider(content)
            section(content, "IMPROVEMENTS", improve_items, limit=3)
            divider(content)
            section(content, "PERFORMANCE", performance_items, limit=3)
            divider(content)
            section(content, "FIX", fix_items, limit=2)
            if fix_items:
                priority = with_line_hint(fix_items[0])
                content.append(f"{bullet} [PRIORITY] {priority}\n", style="bold yellow")
                content.append("\n")
            divider(content)

            summary_line = ""
            if fix_items:
                summary_line = clean_line(fix_items[1] if len(fix_items) > 1 else fix_items[0])
            elif improve_items:
                summary_line = clean_line(improve_items[0])
            if summary_line:
                section(content, "SUMMARY", [summary_line], limit=1)
                divider(content)

            content.append("[C] CONFIDENCE\n", style="bold cyan")
            score = int(analysis.get("confidence_score", 80))
            if score >= 85:
                confidence_color = "green"
            elif score >= 70:
                confidence_color = "yellow"
            else:
                confidence_color = "red"
            content.append(f"{bullet} {score}%\n", style=confidence_color)
        else:
            response = compact_line(str(analysis.get("chat_response", "I am here to help.")), max_len=220)
            if not response.strip():
                response = "I am here to help."
            content.append("TRX-AI\n", style="bold")
            content.append(f"{response}\n", style="white")
            content.append("\n")
            divider(content)
            content.append("[C] CONFIDENCE\n", style="bold cyan")
            score = int(float(analysis.get("intent_confidence", 0.8)) * 100)
            if score >= 85:
                confidence_color = "green"
            elif score >= 70:
                confidence_color = "yellow"
            else:
                confidence_color = "red"
            content.append(f"{bullet} {score}%\n", style=confidence_color)

        if total_runs is not None:
            content.append("\n")
            content.append(f"Run: {total_runs} | Mode: {mode.upper()}\n", style="dim")
        content.append("\n")
        content.append(f"intent: {intent}\n", style="dim")
        content.append(f"source: {source}", style="dim")
        statuses = analysis.get("system_status", [])
        if isinstance(statuses, list) and statuses:
            preview = ", ".join(str(item) for item in statuses[:5])
            content.append("\n")
            content.append(f"status: {preview}", style="dim")

        panel = Panel(
            content,
            title="[bold cyan]TRX-AI Assistant[/bold cyan]",
            border_style="dim",
            padding=(0, 1),
            expand=True,
        )
        self.console.print(panel)
