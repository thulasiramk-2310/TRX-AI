"""Transcript-style output rendering for TRX-AI (Rich)."""

from __future__ import annotations

import difflib
import re
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from guardrails import strip_terminal_controls
from personality import personality_meta  # noqa: F401 - kept for callers importing it from here

# Neon HUD palette: cyan/violet/magenta chrome, semantic colours for findings.
THEME = {
    "accent": "#22d3ee",   # neon cyan: glyphs, prompt, headings
    "accent2": "#a78bfa",  # violet
    "accent3": "#f472b6",  # magenta
    "text": "#e2e8f0",
    "muted": "#64748b",
    "border": "#334155",
    "chip": "on #0f172a",  # dark pill background for status chips
    "bug": "#fb7185",
    "fix": "#34d399",
    "perf": "#fbbf24",
    "note": "#60a5fa",
    "code": "#7dd3fc",
}
CODE_THEME = "monokai"
GRADIENT = ("#22d3ee", "#38bdf8", "#818cf8", "#a78bfa", "#c084fc", "#f472b6")


def gradient_text(text: str, *, bold: bool = True, colors: tuple[str, ...] = GRADIENT) -> Text:
    """Colours each character along a cyan -> violet -> magenta ramp (static, printed once)."""
    out = Text()
    visible = max(1, len(text) - 1)
    for i, ch in enumerate(text):
        color = colors[min(len(colors) - 1, round(i / visible * (len(colors) - 1)))]
        out.append(ch, style=f"bold {color}" if bold else color)
    return out

# How each analysis_source is shown in the meta line under an answer.
SOURCE_LABELS = {
    "llm": "llm",
    "rule-fallback": "rule fallback (LLM unavailable)",
    "rule-fastpath": "built-in",
    "rule": "rule-based",
    "knowledge-base": "built-in knowledge",
}

_GRAPH_PREFIXES = ("impacted module:", "call relation:", "dependency chain:", "graph:")


# Spaces Rich pads lines with, optionally followed by trailing colour-reset codes.
_TRAILING_WS = re.compile(r"[ \t]+((?:\x1b\[[0-9;]*m)*)$", re.MULTILINE)


def _is_graph_line(text: str) -> bool:
    return str(text).strip().lower().startswith(_GRAPH_PREFIXES)


class OutputFormatter:
    """Renders chat replies and code reviews as compact cards in a scrolling transcript."""

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

    def _renderable_to_safe_text(self, renderable: Any) -> str:
        with self.console.capture() as capture:
            self.console.print(renderable)
        rendered = capture.get()
        stream = getattr(self.console, "file", None)
        encoding = getattr(stream, "encoding", None) or "utf-8"
        return rendered.encode(encoding, errors="replace").decode(encoding, errors="replace")

    # ------------------------------------------------------------------ transcript UI

    def _glyphs(self) -> dict[str, str]:
        """Unicode glyphs where the terminal can encode them, ASCII otherwise (e.g. cp1252)."""
        stream = getattr(self.console, "file", None)
        encoding = str(getattr(stream, "encoding", "") or "utf-8")
        # Single-width glyphs only: emoji like ⚡ are double-width in some terminals and break alignment.
        fancy = {"dot": "◈", "bug": "✗", "fix": "✓", "perf": "↯", "note": "→", "sep": "·",
                 "hud_l": "◢", "hud_r": "◣", "arrow": "▸", "rule": "─", "on": "▰", "off": "▱"}
        try:
            "".join(fancy.values()).encode(encoding)
            return fancy
        except (LookupError, UnicodeEncodeError):
            return {"dot": "*", "bug": "x", "fix": "+", "perf": "!", "note": ">", "sep": "|",
                    "hud_l": "/", "hud_r": "\\", "arrow": ">", "rule": "-", "on": "#", "off": "."}

    def _hud_width(self) -> int:
        # Capped so rules don't re-wrap into a mess when the window is narrowed later.
        return max(40, min(self.console.width, 96))

    def _confidence_bar(self, score: int, cells: int = 10) -> Text:
        g = self._glyphs()
        filled = max(0, min(cells, round(score / 100 * cells)))
        bar = gradient_text(g["on"] * filled, bold=False)
        bar.append(g["off"] * (cells - filled), style=THEME["border"])
        bar.append(f" {score}%", style=f"bold {THEME['text']}")
        return bar

    def _hud_rule(self, label: str, subject: str = "", right: Text | None = None) -> Text:
        """`◢ LABEL ▸ subject ───────── right ◣` — the HUD header used above every answer."""
        g = self._glyphs()
        line = Text.assemble((f"{g['hud_l']} ", THEME["accent"]))
        line.append_text(gradient_text(label))
        if subject:
            line.append(f" {g['arrow']} ", style=THEME["accent2"])
            line.append(subject, style=f"bold {THEME['text']}")
        tail = Text(" ")
        if right is not None:
            tail.append_text(right)
            tail.append(" ")
        tail.append(g["hud_r"], style=THEME["accent3"])
        fill = self._hud_width() - line.cell_len - tail.cell_len - 1
        if fill > 2:
            line.append(" " + g["rule"] * fill, style=THEME["border"])
        line.append_text(tail)
        return line

    def _section_header(self, label: str, color: str, count: int | None = None) -> Text:
        head = Text.assemble((f"  {self._glyphs()['arrow']} ", THEME["accent2"]), (label.upper(), f"bold {color}"))
        if count and count > 1:
            head.append(f"  {count}", style=THEME["muted"])
        return head

    def _meta_line(self, analysis: dict[str, Any], extra: list[str] | None = None) -> Text:
        sep = f" {self._glyphs()['sep']} "
        parts: list[str] = []
        model = str(analysis.get("model_name", "")).strip()
        if model:
            parts.append(model)
        elapsed = analysis.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and elapsed >= 0.05:
            parts.append(f"{elapsed:.1f}s")
        source = str(analysis.get("analysis_source", "")).strip().lower()
        statuses = [str(s).lower() for s in analysis.get("system_status", [])]
        if any("rule-fallback" in s or s == "llm_unavailable" for s in statuses):
            source = "rule-fallback"
        label = SOURCE_LABELS.get(source, source)
        if label:
            parts.append(label)
        if "cache_hit" in statuses:
            parts.append("cached")
        parts.extend(extra or [])
        if not parts:
            return Text("")
        # Status "chip": a dark pill behind the meta text.
        return Text.assemble(("  ", ""), (self._glyphs()["dot"] + " ", THEME["accent"]),
                             (f" {sep.join(parts)} ", f"{THEME['muted']} {THEME['chip']}"))

    @staticmethod
    def _inline(text: str) -> Text:
        """Light inline formatting: `code` and **bold**, without Markdown's heavy code blocks."""
        clean = re.sub(r"\s+", " ", strip_terminal_controls(str(text))).strip()
        clean = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", clean)
        out = Text()
        for token in re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", clean):
            if not token:
                continue
            if token.startswith("`") and token.endswith("`") and len(token) > 1:
                out.append(token[1:-1], style=THEME["code"])
            elif token.startswith("**") and token.endswith("**") and len(token) > 3:
                out.append(token[2:-2], style=f"bold {THEME['text']}")
            else:
                out.append(token, style=THEME["text"])
        return out

    def _print(self, renderable: Any) -> None:
        """Renders once, strips Rich's padding to the terminal width, and writes the block in one go.

        Trailing spaces end up in copied text and turn into blank wrapped lines when the window is
        resized; writing each block once (no live redraws) keeps the terminal's own scrollback stable.
        """
        with self.console.capture() as capture:
            self.console.print(renderable)
        rendered = _TRAILING_WS.sub(r"\1", capture.get()).rstrip("\n")
        # Make the text encodable *before* writing: if Rich hits a UnicodeEncodeError mid-write it
        # keeps the failed text buffered, and every later print fails on it again.
        encoding = str(getattr(getattr(self.console, "file", None), "encoding", None) or "utf-8")
        try:
            rendered.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            rendered = rendered.encode(encoding, errors="replace").decode(encoding, errors="replace")
        self.console.print(Text.from_ansi(rendered) if rendered else Text(""), soft_wrap=True)

    def _render_chat(self, analysis: dict[str, Any]) -> None:
        g = self._glyphs()
        # Model text never reaches the terminal raw: no escape sequences (screen clear, title, clipboard).
        reply = strip_terminal_controls(str(analysis.get("chat_response", ""))).strip() or "I am here to help."
        row = Table.grid(padding=(0, 1))
        row.add_column(width=1, no_wrap=True)
        row.add_column(ratio=1)
        row.add_row(gradient_text(g["dot"]), Markdown(reply, code_theme=CODE_THEME))
        self._print(Group(Text(""), row, Text(""), self._meta_line(analysis)))

    def _review_sections(self, analysis: dict[str, Any]) -> list[tuple[str, str, str, list[str]]]:
        raw_output = str(analysis.get("raw_llm_output", "")).strip()

        def pick(key: str, placeholders: set[str], headers: list[str], limit: int) -> list[str]:
            # Graph lines ("Impacted module: ...") are context, not findings; they get their own line.
            items = [i for i in self._section_items(analysis, key, "") if i and not _is_graph_line(i)]
            if self._is_placeholder_items(items, placeholders):
                items = self._extract_raw_section(raw_output, headers, max_items=limit) or []
            return items[:limit]

        g = self._glyphs()
        return [
            ("Issues", g["bug"], THEME["bug"], pick(
                "debug_analysis",
                {"No critical bugs were explicitly identified.", "No debug issues detected."},
                ["CODE DEBUG", "DEBUG ANALYSIS"], 6)),
            ("Fixes", g["fix"], THEME["fix"], pick(
                "improvements",
                {"Apply incremental refactoring for readability and maintainability."},
                ["CODE IMPROVEMENTS", "IMPROVEMENTS", "FIX SUGGESTIONS"], 6)),
            ("Performance", g["perf"], THEME["perf"], pick(
                "predictions",
                {"LLM output parsing incomplete - check raw output"},
                ["PERFORMANCE"], 4)),
            ("Summary", g["note"], THEME["note"], pick(
                "final_insight",
                {"Prioritize correctness issues first, then refactor and optimize."},
                ["FINAL SUMMARY"], 3)),
        ]

    @staticmethod
    def _language_from_input(analysis: dict[str, Any]) -> str:
        target = str(analysis.get("current_input", "")).lower()
        for ext, lang in ((".py", "python"), (".js", "javascript"), (".ts", "typescript"), (".java", "java"),
                          (".go", "go"), (".cpp", "cpp"), (".c", "c"), (".sql", "sql"), (".rs", "rust")):
            if re.search(re.escape(ext) + r"\b", target):
                return lang
        return "python"

    @staticmethod
    def _review_title(analysis: dict[str, Any]) -> str:
        target = str(analysis.get("current_input", "")).strip()
        for prefix in ("review ", "fix ", "analyze "):
            if target.lower().startswith(prefix):
                return f"{prefix.strip().capitalize()} {target[len(prefix):].strip()}"
        return "Code review"

    def _render_review(self, analysis: dict[str, Any]) -> None:
        """Borderless review: a header line, then one short section per finding type.

        No box-drawing characters, so selecting and copying the text gives clean text.
        """
        g = self._glyphs()
        title = self._review_title(analysis)
        verb, _, subject = ("Review", "", "") if title == "Code review" else title.partition(" ")
        score = analysis.get("confidence_score")
        bar = self._confidence_bar(int(score)) if isinstance(score, (int, float)) and score else None
        header = self._hud_rule(verb.upper(), subject, bar)

        sections = [s for s in self._review_sections(analysis) if s[3]]
        blocks: list[Any] = [Text(""), header]
        for label, icon, color, items in sections:
            blocks += [Text(""), self._section_header(label, color, len(items))]
            # Grid gives a hanging indent: wrapped lines line up with the text, not the icon.
            grid = Table.grid(padding=(0, 1))
            grid.add_column(width=5, no_wrap=True)
            grid.add_column()
            for item in items:
                grid.add_row(Text(f"    {icon}", style=color), self._inline(item))
            blocks.append(grid)
        if not sections:
            blocks += [Text(""), Text(f"  {g['fix']} No issues found.", style=THEME["fix"])]
        self._print(Group(*blocks))

        self._render_fix(analysis)
        self.render_guardrails(analysis.get("guardrails") or [])
        context = self._graph_context_summary(analysis)
        self._print(Text(""))
        self._print(self._meta_line(analysis, [context] if context else None))

    @staticmethod
    def _strip_file_header(code: str) -> str:
        # The reviewer prefixes fixes with "# FILE: name"; it isn't part of the change.
        return re.sub(r"^\s*(#|//|--)\s*FILE:[^\n]*\n", "", code, count=1)

    def _render_fix(self, analysis: dict[str, Any]) -> None:
        fixed = self._strip_file_header(strip_terminal_controls(str(analysis.get("fixed_code", "")))).strip("\n")
        if not fixed.strip():
            return
        original = self._strip_file_header(str(analysis.get("original_code", ""))).strip("\n")
        language = str(analysis.get("language", "") or self._language_from_input(analysis))

        if original:
            old_lines = [line.rstrip() for line in original.splitlines()]
            new_lines = [line.rstrip() for line in fixed.splitlines()]
            if old_lines == new_lines:
                self._print(Text(""))
                self._print(Text(f"  {self._glyphs()['fix']} no code changes suggested", style=THEME["muted"]))
                return
            diff = list(difflib.unified_diff(old_lines, new_lines, lineterm="", n=2))[2:]  # drop ---/+++ headers
            diff = [d for d in diff if not d.startswith("@@")]  # hunk headers are noise for a reader
            added = sum(1 for d in diff if d.startswith("+"))
            removed = sum(1 for d in diff if d.startswith("-"))
            body: Any = Syntax("\n".join(diff), "diff", theme=CODE_THEME, background_color="default")
            title = self._section_header("Suggested change", THEME["fix"])
            title.append(f"  +{added} ", style=f"bold {THEME['fix']}")
            title.append(f"-{removed}", style=f"bold {THEME['bug']}")
        else:
            # No line numbers: the fix should copy-paste straight into an editor.
            body = Syntax(fixed, language, theme=CODE_THEME, background_color="default")
            title = self._section_header("Suggested fix", THEME["fix"])

        self._print(Group(Text(""), title, Padding(body, (0, 0, 0, 4))))

    def render_guardrails(self, findings: list[Any]) -> None:
        """`▸ GUARDRAILS` section. Accepts guardrails.Finding objects or (level, message) pairs."""
        rows = []
        for item in findings:
            level, message = (item.level, item.message) if hasattr(item, "level") else (item[0], item[1])
            rows.append((str(level), str(message)))
        if not rows:
            return
        blocking = any(level == "block" for level, _ in rows)
        g = self._glyphs()
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=5, no_wrap=True)
        grid.add_column()
        for level, message in rows:
            icon, color = (g["bug"], THEME["bug"]) if level == "block" else ("!", THEME["perf"])
            grid.add_row(Text(f"    {icon}", style=f"bold {color}"), self._inline(message))
        header = self._section_header("Guardrails", THEME["bug"] if blocking else THEME["perf"], len(rows))
        header.append("  blocked" if blocking else "  warning", style=f"bold {THEME['bug'] if blocking else THEME['perf']}")
        self._print(Group(Text(""), header, grid))

    def _graph_context_summary(self, analysis: dict[str, Any]) -> str:
        lines = [
            str(item)
            for key in ("debug_analysis", "improvements", "predictions", "final_insight")
            for item in (analysis.get(key) or [])
            if _is_graph_line(str(item))
        ]
        if not lines:
            return ""
        modules = sum(1 for line in lines if line.lower().startswith("impacted module"))
        calls = sum(1 for line in lines if line.lower().startswith("call relation"))
        bits = [f"{modules} related module{'s' if modules != 1 else ''}"] if modules else []
        if calls:
            bits.append(f"{calls} call{'s' if calls != 1 else ''}")
        return "graph: " + ", ".join(bits) if bits else "graph context"

    def render(self, analysis: dict[str, Any], mode: str, total_runs: int | None = None) -> None:
        """Prints one answer into the scrolling transcript (never clears the screen)."""
        if analysis.get("response_mode") == "analysis":
            self._render_review(analysis)
        else:
            self._render_chat(analysis)

    def render_help_dashboard(
        self,
        *,
        mode: str,
        total_runs: int,
        model_name: str,
        active_agents: list[str],
    ) -> None:
        fancy = self._glyphs()["dot"] != "*"
        groups = [
            ("Code", [
                ("review <file|folder>", "multi-agent code review"),
                ("fix <file>", "generate a fixed copy, preview, confirm"),
                ("watch <folder>", "re-review files on every save"),
                ("explain \"<question>\"", "ask a general question"),
            ]),
            ("Session", [
                ("history", "previous inputs"),
                ("save <path>", "save session JSON"),
                ("export <file>", "export last analysis (.txt / .pdf)"),
                ("export compare <file>", "compare last two analyses (PDF)"),
            ]),
            ("Settings", [
                ("chat | code | auto", "routing mode"),
                ("mode debug|optimize|predict", "fallback profile"),
                ("personality list|set|add|delete", "assistant voice"),
                ("agents all | agents debug improve predict", "active agents"),
                ("status", "runtime status"),
                ("exit", "quit (or Ctrl+D)"),
            ]),
            ("Keys", [
                ("/command", "every command also works with a slash"),
                ("Tab  ↑  →" if fancy else "Tab  Up  Right", "complete, history, accept suggestion"),
            ]),
        ]
        table = Table.grid(padding=(0, 3))
        table.add_column(no_wrap=True)
        table.add_column(style=THEME["muted"])
        for i, (title, rows) in enumerate(groups):
            if i:
                table.add_row("", "")
            table.add_row(self._section_header(title, THEME["accent"]), "")
            for cmd, desc in rows:
                table.add_row(Text(f"    {cmd}", style=THEME["code"]), desc)
        self._print(Group(Text(""), self._hud_rule("COMMANDS", model_name), Text(""), table))
