"""Session history handling for Reality Debugger."""

from __future__ import annotations

import json
import textwrap
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SessionHistory:
    """In-memory session state and persistence helpers."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add_entry(self, user_input: str, mode: str, analysis: dict[str, Any]) -> None:
        self.entries.append(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": mode,
                "input": user_input,
                "analysis": analysis,
            }
        )

    def list_inputs(self) -> list[str]:
        return [entry["input"] for entry in self.entries]

    def recent_context(self, limit: int = 3) -> list[dict[str, Any]]:
        """Returns the most recent session items for contextual reasoning."""
        if limit <= 0:
            return []
        return self.entries[-limit:]

    def latest_entry(self) -> dict[str, Any] | None:
        """Returns the latest session entry if available."""
        if not self.entries:
            return None
        return self.entries[-1]

    def latest_analysis_entries(self, limit: int = 2) -> list[dict[str, Any]]:
        """Returns the latest analysis-only session entries."""
        if limit <= 0:
            return []
        analysis_entries = [
            entry
            for entry in self.entries
            if isinstance(entry.get("analysis"), dict)
            and entry["analysis"].get("response_mode") == "analysis"
        ]
        return analysis_entries[-limit:]

    def save(self, output_path: str | None = None) -> Path:
        output_dir = Path("sessions")
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_path:
            destination = Path(output_path)
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = output_dir / f"reality_debugger_session_{stamp}.json"

        with destination.open("w", encoding="utf-8") as session_file:
            json.dump(self.entries, session_file, indent=2)

        return destination

    def export_report(
        self,
        file_name: str,
        *,
        user_input: str,
        mode: str,
        structured_output: str,
    ) -> Path:
        """Exports a human-readable analysis report under the reports folder."""
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_name = file_name.strip() or "report.txt"
        if not safe_name.lower().endswith(".txt"):
            safe_name += ".txt"

        destination = reports_dir / safe_name
        content = (
            "TRX-AI (Reality Debugger) Report\n"
            "================================\n"
            f"Input: {user_input}\n"
            f"Mode: {mode}\n\n"
            f"{structured_output}\n"
        )

        destination.write_text(content, encoding="utf-8")
        return destination

    def export_pdf_report(
        self,
        filename: str,
        user_input: str,
        mode: str,
        structured_output: str,
    ) -> Path:
        """Exports a premium branded PDF report with card-based layout."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.pdfgen import canvas
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "PDF export requires reportlab. Install it with: pip install reportlab"
            ) from exc

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_name = filename.strip() or "report.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        destination = reports_dir / safe_name

        logo_path = self._ensure_logo_image()
        diagram_path = Path("assets") / "diagram.png"

        width, height = A4
        margin_x = 1.8 * cm
        content_width = width - (2 * margin_x)
        footer_y = 1.8 * cm
        page_number = 1

        doc = canvas.Canvas(str(destination), pagesize=A4)
        theme = {
            "bg": colors.HexColor("#F7FAFF"),
            "ink": colors.HexColor("#0E2238"),
            "muted": colors.HexColor("#4A5D73"),
            "line": colors.HexColor("#BFDDF2"),
            "brand": colors.HexColor("#00AEEF"),
            "brand_dark": colors.HexColor("#1A4F8B"),
            "card_bg": colors.HexColor("#FFFFFF"),
            "card_line": colors.HexColor("#D6E8F8"),
        }

        y = self._draw_page_header(
            doc,
            width,
            height,
            margin_x,
            logo_path,
            theme,
            subtitle="Reality Debugger",
            tagline="Debugging Real Life Like Code",
        )

        parsed_sections = self._parse_structured_sections(structured_output)
        confidence = parsed_sections.get("CONFIDENCE SCORE", ["N/A"])
        summary_lines = self._build_summary_lines(mode, parsed_sections, confidence)

        fixed_code_lines = parsed_sections.get("LLM FIXED CODE", [])
        if fixed_code_lines:
            fixed_code_lines = self._trim_code_block_lines(fixed_code_lines, max_lines=120)

        cards = [
            ("SUMMARY", summary_lines),
            ("INPUT", [user_input]),
            ("MODE", [mode.upper()]),
            ("DEBUG ANALYSIS", parsed_sections.get("DEBUG ANALYSIS", parsed_sections.get("INPUT ANALYSIS", ["N/A"]))),
            ("IMPROVEMENTS", parsed_sections.get("IMPROVEMENTS", parsed_sections.get("FIX SUGGESTIONS", ["N/A"]))),
            ("PREDICTIONS", parsed_sections.get("PREDICTIONS", ["N/A"])),
            ("FINAL INSIGHT", parsed_sections.get("FINAL INSIGHT", ["N/A"])),
            ("CHANGES APPLIED", parsed_sections.get("CHANGES APPLIED", [])),
            ("CODE COMPLEXITY", parsed_sections.get("CODE COMPLEXITY", [])),
            ("OPTIMIZATION APPROACHES", parsed_sections.get("OPTIMIZATION APPROACHES", [])),
            ("LLM FIXED CODE", fixed_code_lines),
            ("CONFIDENCE SCORE", confidence),
        ]

        cards = [(title, lines) for title, lines in cards if lines]

        for title, lines in cards:
            required = self._estimate_card_height(lines)
            if y - required < footer_y + 2.0 * cm:
                self._draw_footer(doc, width, margin_x, page_number)
                doc.showPage()
                page_number += 1
                y = self._draw_page_header(
                    doc,
                    width,
                    height,
                    margin_x,
                    logo_path,
                    theme,
                    subtitle="Reality Debugger",
                    tagline="Debugging Real Life Like Code",
                )

            y = self._draw_card_section(
                doc,
                title,
                lines,
                y,
                margin_x,
                content_width,
                theme,
            )

        if diagram_path.exists() and y - 7.5 * cm > footer_y + 1.4 * cm:
            try:
                if y - 7.5 * cm < footer_y + 1.4 * cm:
                    self._draw_footer(doc, width, margin_x, page_number)
                    doc.showPage()
                    page_number += 1
                    y = self._draw_page_header(
                        doc,
                        width,
                        height,
                        margin_x,
                        logo_path,
                        theme,
                        subtitle="Reality Debugger",
                        tagline="Debugging Real Life Like Code",
                    )

                y = self._draw_card_image(
                    doc,
                    "OPTIONAL DIAGRAM",
                    diagram_path,
                    y,
                    margin_x,
                    content_width,
                    theme,
                )
            except Exception:
                pass

        self._draw_footer(doc, width, margin_x, page_number)
        doc.save()
        return destination

    def export_comparison_pdf_report(
        self,
        filename: str,
        *,
        first_input: str,
        second_input: str,
        mode: str,
        first_structured_output: str,
        second_structured_output: str,
        first_label: str = "Run 1",
        second_label: str = "Run 2",
    ) -> Path:
        """Exports a visual comparison PDF with metric chart for two analyses."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.pdfgen import canvas
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "Comparison PDF export requires reportlab. Install it with: pip install reportlab"
            ) from exc

        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        safe_name = filename.strip() or "comparison_report.pdf"
        if not safe_name.lower().endswith(".pdf"):
            safe_name += ".pdf"
        destination = reports_dir / safe_name

        logo_path = self._ensure_logo_image()

        width, height = A4
        margin_x = 1.8 * cm
        content_width = width - (2 * margin_x)

        doc = canvas.Canvas(str(destination), pagesize=A4)

        theme = {
            "bg": colors.HexColor("#F7FAFF"),
            "ink": colors.HexColor("#0E2238"),
            "muted": colors.HexColor("#4A5D73"),
            "line": colors.HexColor("#BFDDF2"),
            "brand": colors.HexColor("#00AEEF"),
            "brand_dark": colors.HexColor("#1A4F8B"),
            "card_bg": colors.HexColor("#FFFFFF"),
            "card_line": colors.HexColor("#D6E8F8"),
        }

        y = self._draw_page_header(
            doc,
            width,
            height,
            margin_x,
            logo_path,
            theme,
            subtitle="Reality Debugger",
            tagline="Debugging Real Life Like Code",
        )
        page_number = 1

        y = self._draw_figure_header(
            doc,
            y=y,
            x=margin_x,
            content_width=content_width,
            theme=theme,
            title="Figure 5.2: Side-by-Side Comparison Report",
            subtitle="Comparison between two analysis sessions (Run 1 vs Run 2)",
        )

        parsed_first = self._parse_structured_sections(first_structured_output)
        parsed_second = self._parse_structured_sections(second_structured_output)
        metrics_first = self._extract_comparison_metrics(parsed_first)
        metrics_second = self._extract_comparison_metrics(parsed_second)

        y = self._draw_comparison_summary_block(
            doc,
            y=y,
            x=margin_x,
            content_width=content_width,
            mode=mode.upper(),
            first_label=first_label,
            second_label=second_label,
            first_input=first_input,
            second_input=second_input,
            theme=theme,
        )

        card_h = 9.2 * cm
        doc.setFillColor(theme["card_bg"])
        doc.setStrokeColor(theme["card_line"])
        doc.setLineWidth(1)
        doc.roundRect(margin_x, y - card_h, content_width, card_h, 8, fill=1, stroke=1)

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 11)
        doc.drawString(margin_x + 12, y - 18, "VISUAL METRIC COMPARISON")
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(margin_x + 10, y - 23, margin_x + content_width - 10, y - 23)

        self._draw_manual_comparison_chart(
            doc,
            x=margin_x + 12,
            y=y - card_h + 14,
            width=content_width - 24,
            height=card_h - 42,
            first_label=first_label,
            second_label=second_label,
            first_metrics=metrics_first,
            second_metrics=metrics_second,
            theme=theme,
        )
        y -= card_h + 16
        issue_diff = self._build_issue_diff(parsed_first, parsed_second)
        issue_card_height = self._estimate_issue_diff_card_height(issue_diff)
        if y - issue_card_height < 2.8 * cm:
            self._draw_footer(doc, width, margin_x, page_number)
            doc.showPage()
            page_number += 1
            y = self._draw_page_header(
                doc,
                width,
                height,
                margin_x,
                logo_path,
                theme,
                subtitle="Reality Debugger",
                tagline="Debugging Real Life Like Code",
            )
            y = self._draw_figure_header(
                doc,
                y=y,
                x=margin_x,
                content_width=content_width,
                theme=theme,
                title="Figure 5.2: Side-by-Side Comparison Report",
                subtitle="Comparison between two analysis sessions (Run 1 vs Run 2)",
            )
        y = self._draw_issue_diff_card(
            doc,
            y=y,
            x=margin_x,
            content_width=content_width,
            issue_diff=issue_diff,
            first_label=first_label,
            second_label=second_label,
            theme=theme,
        )

        recommendation = [
            "Run 2 demonstrates improved confidence compared to Run 1. "
            "It is recommended to continue this strategy while enhancing input "
            "specificity to achieve more precise diagnostics."
        ]
        y = self._draw_recommendation_card(
            doc,
            y,
            margin_x,
            content_width,
            recommendation,
            theme,
        )

        self._draw_footer(doc, width, margin_x, page_number)
        doc.save()
        return destination

    @staticmethod
    def _parse_structured_sections(structured_output: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current: str | None = None

        for raw_line in structured_output.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current = stripped[1:-1].strip().upper()
                sections.setdefault(current, [])
                continue
            if current is None:
                continue

            # Preserve indentation exactly for code blocks in reports.
            if current == "LLM FIXED CODE":
                sections[current].append(raw_line.rstrip("\n"))
                continue

            if stripped.startswith("- "):
                sections[current].append(stripped[2:].strip())
            else:
                sections[current].append(stripped)

        return sections

    @staticmethod
    def _wrap_lines(text: str, max_chars: int) -> list[str]:
        wrapped = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)
        return wrapped if wrapped else [""]

    @staticmethod
    def _trim_code_block_lines(lines: list[str], max_lines: int = 120) -> list[str]:
        cleaned: list[str] = []
        for line in lines:
            value = line.rstrip()
            if value.strip() == "```":
                continue
            cleaned.append(value)
        if len(cleaned) <= max_lines:
            return cleaned
        trimmed = cleaned[:max_lines]
        trimmed.append("... [truncated for PDF readability]")
        return trimmed

    @staticmethod
    def _estimate_card_height(lines: list[str]) -> float:
        approx_rows = 0
        for item in lines:
            row_count = max(1, (len(item) // 92) + 1)
            approx_rows += row_count
        return 30 + (approx_rows * 14) + 18

    @staticmethod
    def _build_summary_lines(
        mode: str,
        sections: dict[str, list[str]],
        confidence_lines: list[str],
    ) -> list[str]:
        debug_count = len(sections.get("DEBUG ANALYSIS", sections.get("INPUT ANALYSIS", [])))
        improve_count = len(sections.get("IMPROVEMENTS", sections.get("FIX SUGGESTIONS", [])))
        predict_count = len(sections.get("PREDICTIONS", []))
        confidence = confidence_lines[0] if confidence_lines else "N/A"
        return [
            "This report analyzes the issue and provides actionable debugging steps.",
            f"Mode selected: {mode.upper()}",
            f"Debug analysis points: {debug_count}",
            f"Improvement points: {improve_count}",
            f"Prediction points: {predict_count}",
            f"Confidence score: {confidence}",
        ]

    @staticmethod
    def _extract_comparison_metrics(parsed_sections: dict[str, list[str]]) -> dict[str, float]:
        confidence = 0.0
        for item in parsed_sections.get("CONFIDENCE SCORE", []):
            match = re.search(r"(\d{1,3})\s*%", item)
            if match:
                confidence = float(max(0, min(100, int(match.group(1)))))
                break

        return {
            "confidence": confidence,
            "bugs": float(len(parsed_sections.get("DEBUG ANALYSIS", parsed_sections.get("BUGS DETECTED", [])))),
            "fixes": float(len(parsed_sections.get("IMPROVEMENTS", parsed_sections.get("FIX SUGGESTIONS", [])))),
            "analysis": float(len(parsed_sections.get("PREDICTIONS", parsed_sections.get("INPUT ANALYSIS", [])))),
        }

    @staticmethod
    def _comparison_recommendation(
        first: dict[str, float],
        second: dict[str, float],
        first_label: str,
        second_label: str,
    ) -> list[str]:
        lines: list[str] = []
        if second["confidence"] > first["confidence"]:
            lines.append(f"{second_label} has higher confidence than {first_label}; continue this strategy.")
        elif second["confidence"] < first["confidence"]:
            lines.append(f"{second_label} confidence dropped vs {first_label}; review blockers and adjust plan.")
        else:
            lines.append("Confidence is stable across both runs; focus on reducing repeated bugs.")

        if second["bugs"] > first["bugs"]:
            lines.append("Bug count increased; prioritize root-cause cleanup before optimization.")
        elif second["bugs"] < first["bugs"]:
            lines.append("Bug count reduced; keep current corrective actions in place.")
        else:
            lines.append("Bug count unchanged; improve specificity in input details for sharper diagnosis.")

        return lines

    @staticmethod
    def _extract_issue_map(parsed_sections: dict[str, list[str]]) -> dict[str, str]:
        """Extracts normalized issue candidates from key structured sections."""
        source_sections = (
            "DEBUG ANALYSIS",
            "BUGS DETECTED",
            "IMPROVEMENTS",
            "FIX SUGGESTIONS",
            "PREDICTIONS",
            "SECURITY",
            "FINAL INSIGHT",
        )
        issue_map: dict[str, str] = {}
        for section_name in source_sections:
            for raw in parsed_sections.get(section_name, []):
                text = str(raw).strip()
                if not text:
                    continue
                normalized = re.sub(r"[^a-z0-9\s]+", " ", text.lower())
                normalized = re.sub(r"\s+", " ", normalized).strip()
                if len(normalized) < 6:
                    continue
                issue_map.setdefault(normalized, text)
        return issue_map

    def _build_issue_diff(
        self,
        first_sections: dict[str, list[str]],
        second_sections: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        first_map = self._extract_issue_map(first_sections)
        second_map = self._extract_issue_map(second_sections)

        first_keys = set(first_map.keys())
        second_keys = set(second_map.keys())

        resolved = sorted(first_keys - second_keys)
        remaining = sorted(first_keys & second_keys)
        new_findings = sorted(second_keys - first_keys)

        resolved_items = [first_map[key] for key in resolved]
        remaining_items = [second_map.get(key, first_map[key]) for key in remaining]
        new_items = [second_map[key] for key in new_findings]

        return {
            "resolved": self._dedupe_issue_lines(resolved_items),
            "remaining": self._dedupe_issue_lines(remaining_items),
            "new": self._dedupe_issue_lines(new_items),
        }

    @staticmethod
    def _dedupe_issue_lines(lines: list[str]) -> list[str]:
        """Removes duplicate meaning and vague/non-actionable issue lines."""
        skip_patterns = (
            "risk remains",
            "risk reduced",
            "risk medium",
            "no obvious",
            "no immediate",
        )
        alias_patterns: list[tuple[str, str]] = [
            (r"sql.*(concat|concatenat|string)", "Fixed SQL string concatenation"),
            (r"null check.*login|login.*null check", "Missing null check in login"),
            (r"input validation", "Add input validation"),
            (r"parameterized quer", "Use parameterized queries"),
            (r"file read.*exception", "File read lacks exception handling"),
        ]

        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in lines:
            text = str(raw).strip()
            if not text:
                continue
            low = text.lower()
            if any(pattern in low for pattern in skip_patterns):
                continue

            canonical = re.sub(r"[^a-z0-9\s]+", " ", low)
            canonical = re.sub(r"\s+", " ", canonical).strip()
            for pattern, replacement in alias_patterns:
                if re.search(pattern, canonical):
                    canonical = replacement.lower()
                    text = replacement
                    break

            if canonical in seen:
                continue
            seen.add(canonical)
            cleaned.append(text)

        return cleaned[:8]

    def _estimate_issue_diff_card_height(self, issue_diff: dict[str, list[str]]) -> float:
        max_chars = 90
        rows = 11
        for key in ("resolved", "remaining", "new"):
            entries = issue_diff.get(key, [])[:10]
            rows += 4
            for entry in entries:
                rows += max(1, (len(entry) // max_chars) + 1)
            if not entries:
                rows += 2
        return 38 + (rows * 12)

    def _draw_issue_diff_card(
        self,
        doc: Any,
        *,
        y: float,
        x: float,
        content_width: float,
        issue_diff: dict[str, list[str]],
        first_label: str,
        second_label: str,
        theme: dict[str, Any],
    ) -> float:
        from reportlab.lib import colors
        card_h = self._estimate_issue_diff_card_height(issue_diff)
        doc.setFillColor(theme["card_bg"])
        doc.setStrokeColor(theme["card_line"])
        doc.setLineWidth(1)
        doc.roundRect(x, y - card_h, content_width, card_h, 8, fill=1, stroke=1)
        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 11)
        doc.drawString(x + 12, y - 18, "ISSUE DIFFERENCE (SIDE-BY-SIDE)")
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(x + 10, y - 23, x + content_width - 10, y - 23)
        doc.setFillColor(theme["muted"])
        doc.setFont("Helvetica", 9)
        doc.drawString(x + 12, y - 35, f"Compared: {first_label} -> {second_label}")
        cursor_y = y - 52
        categories = [
            ("[OK] RESOLVED ISSUES", issue_diff.get("resolved", [])[:10], colors.HexColor("#1FAF5A"), colors.HexColor("#F3FBF5")),
            ("[X] REMAINING ISSUES", issue_diff.get("remaining", [])[:10], colors.HexColor("#D64545"), colors.HexColor("#FFF4F4")),
            ("[!] NEW FINDINGS", issue_diff.get("new", [])[:10], colors.HexColor("#D08B00"), colors.HexColor("#FFF9EF")),
        ]
        for title, lines, color, tint in categories:
            row_units = sum(max(1, len(self._wrap_lines(line, max_chars=90))) for line in lines) if lines else 1
            block_h = 22 + (row_units * 12) + 12
            doc.setFillColor(tint)
            doc.setStrokeColor(theme["card_line"])
            doc.setLineWidth(0.6)
            doc.roundRect(x + 10, cursor_y - block_h + 6, content_width - 20, block_h, 6, fill=1, stroke=1)
            doc.setFillColor(color)
            doc.setFont("Helvetica-Bold", 10)
            doc.drawString(x + 18, cursor_y - 8, title)
            cursor_y -= 19
            doc.setFillColor(theme["ink"])
            doc.setFont("Helvetica", 9)
            if not lines:
                doc.drawString(x + 26, cursor_y, "- none")
                cursor_y -= 12
            else:
                for line in lines:
                    wrapped = self._wrap_lines(line, max_chars=90)
                    for idx, part in enumerate(wrapped):
                        prefix = "- " if idx == 0 else "  "
                        doc.drawString(x + 26, cursor_y, f"{prefix}{part}")
                        cursor_y -= 12
            cursor_y -= 12
        return y - card_h - 10
    @staticmethod
    def _draw_manual_comparison_chart(
        doc: Any,
        *,
        x: float,
        y: float,
        width: float,
        height: float,
        first_label: str,
        second_label: str,
        first_metrics: dict[str, float],
        second_metrics: dict[str, float],
        theme: dict[str, Any],
    ) -> None:
        from reportlab.lib import colors
        metrics = [
            ("Confidence", first_metrics["confidence"], second_metrics["confidence"]),
            ("Bugs", first_metrics["bugs"], second_metrics["bugs"]),
            ("Fixes", first_metrics["fixes"], second_metrics["fixes"]),
            ("Analysis", first_metrics["analysis"], second_metrics["analysis"]),
        ]
        chart_left = x + 52
        chart_bottom = y + 30
        chart_width = width - 86
        chart_height = height - 84
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.8)
        doc.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)
        doc.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)
        max_val = max(10.0, max(max(first, second) for _, first, second in metrics))
        scale = chart_height / max_val
        doc.setStrokeColor(colors.HexColor("#DCE9F5"))
        doc.setLineWidth(0.6)
        for i in range(1, 6):
            gy = chart_bottom + (chart_height * i / 6)
            doc.line(chart_left, gy, chart_left + chart_width, gy)
        group_width = chart_width / len(metrics)
        bar_width = min(18, (group_width * 0.66) / 2)
        bar_gap = 6
        for idx, (label, first_val, second_val) in enumerate(metrics):
            gx = chart_left + (idx * group_width) + ((group_width - ((bar_width * 2) + bar_gap)) / 2)
            h1 = first_val * scale
            h2 = second_val * scale
            doc.setFillColor(colors.HexColor("#7FC8FF"))
            doc.rect(gx, chart_bottom, bar_width, h1, fill=1, stroke=0)
            doc.setFillColor(colors.HexColor("#1A4F8B"))
            doc.rect(gx + bar_width + bar_gap, chart_bottom, bar_width, h2, fill=1, stroke=0)
            doc.setFillColor(theme["muted"])
            doc.setFont("Helvetica", 8.8)
            doc.drawCentredString(gx + bar_width + (bar_gap / 2), chart_bottom - 12, label)
        legend_y = chart_bottom + chart_height + 16
        doc.setFillColor(colors.HexColor("#7FC8FF"))
        doc.rect(chart_left, legend_y - 4, 8, 8, fill=1, stroke=0)
        doc.setFillColor(theme["muted"])
        doc.setFont("Helvetica", 9)
        doc.drawString(chart_left + 12, legend_y - 2, "Run 1 (Baseline)")
        legend2_x = chart_left + 148
        doc.setFillColor(colors.HexColor("#1A4F8B"))
        doc.rect(legend2_x, legend_y - 4, 8, 8, fill=1, stroke=0)
        doc.setFillColor(theme["muted"])
        doc.drawString(legend2_x + 12, legend_y - 2, "Run 2 (Improved)")
        doc.setFont("Helvetica", 8.5)
        doc.drawCentredString(chart_left + (chart_width / 2), chart_bottom - 24, "Metrics")
        doc.saveState()
        doc.translate(chart_left - 22, chart_bottom + (chart_height / 2))
        doc.rotate(90)
        doc.drawCentredString(0, 0, "Score / Count")
        doc.restoreState()
    @staticmethod
    def _draw_figure_header(
        doc: Any,
        *,
        y: float,
        x: float,
        content_width: float,
        theme: dict[str, Any],
        title: str,
        subtitle: str,
    ) -> float:
        doc.setFillColor(theme["ink"])
        doc.setFont("Helvetica-Bold", 15)
        doc.drawCentredString(x + (content_width / 2), y, title)
        y -= 14
        doc.setFillColor(theme["muted"])
        doc.setFont("Helvetica", 10)
        doc.drawCentredString(x + (content_width / 2), y, subtitle)
        y -= 14
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.8)
        doc.line(x, y, x + content_width, y)
        return y - 18
    @staticmethod
    def _draw_comparison_summary_block(
        doc: Any,
        *,
        y: float,
        x: float,
        content_width: float,
        mode: str,
        first_label: str,
        second_label: str,
        first_input: str,
        second_input: str,
        theme: dict[str, Any],
    ) -> float:
        from reportlab.lib import colors
        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 12)
        doc.drawString(x, y, "COMPARISON SUMMARY")
        y -= 8
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(x, y, x + content_width, y)
        y -= 12
        table_h = 60
        doc.setFillColor(theme["card_bg"])
        doc.setStrokeColor(theme["card_line"])
        doc.roundRect(x, y - table_h, content_width, table_h, 6, fill=1, stroke=1)
        rows = [
            ("Mode", mode),
            (first_label, first_input),
            (second_label, second_input),
        ]
        row_y = y - 16
        label_x = x + 14
        value_x = x + 92
        for idx, (label, value) in enumerate(rows):
            doc.setFillColor(theme["brand_dark"])
            doc.setFont("Helvetica-Bold", 10)
            doc.drawString(label_x, row_y, f"{label:<6}")
            doc.drawString(label_x + 48, row_y, ":")
            doc.setFillColor(theme["ink"])
            doc.setFont("Helvetica", 10)
            doc.drawString(value_x, row_y, value)
            row_y -= 17
            if idx < len(rows) - 1:
                doc.setStrokeColor(colors.HexColor("#E6EEF7"))
                doc.setLineWidth(0.5)
                doc.line(x + 10, row_y + 8, x + content_width - 10, row_y + 8)
        y -= table_h + 14
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.8)
        doc.line(x, y, x + content_width, y)
        return y - 16
    @staticmethod
    def _draw_recommendation_card(
        doc: Any,
        y: float,
        x: float,
        content_width: float,
        lines: list[str],
        theme: dict[str, Any],
    ) -> float:
        from reportlab.lib import colors

        wrapped: list[str] = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=100, break_long_words=False, break_on_hyphens=False))
        card_h = 34 + (len(wrapped) * 14)

        doc.setFillColor(colors.HexColor("#F8FBFF"))
        doc.setStrokeColor(colors.HexColor("#BFDDF2"))
        doc.setLineWidth(1)
        doc.roundRect(x, y - card_h, content_width, card_h, 8, fill=1, stroke=1)

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 12)
        doc.drawString(x + 12, y - 18, "RECOMMENDATION")

        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(x + 10, y - 23, x + content_width - 10, y - 23)

        body_y = y - 40
        doc.setFillColor(theme["ink"])
        doc.setFont("Helvetica", 10.5)
        for row in wrapped:
            doc.drawString(x + 14, body_y, row)
            body_y -= 14

        return y - card_h - 10

    @staticmethod
    def _draw_page_header(
        doc: Any,
        page_width: float,
        page_height: float,
        margin_x: float,
        logo_path: Path | None,
        theme: dict[str, Any],
        *,
        subtitle: str,
        tagline: str,
    ) -> float:
        from reportlab.lib.units import cm

        doc.setFillColor(theme["bg"])
        doc.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        doc.setFillColor(theme["brand"])
        doc.rect(0, page_height - 0.85 * cm, page_width, 0.85 * cm, fill=1, stroke=0)

        top_y = page_height - 1.55 * cm
        logo_w = 2.6 * cm
        logo_h = 2.6 * cm

        if logo_path and logo_path.exists():
            try:
                doc.drawImage(
                    str(logo_path),
                    margin_x,
                    top_y - logo_h,
                    width=logo_w,
                    height=logo_h,
                    preserveAspectRatio=True,
                    mask="auto",
                )
            except Exception:
                pass

        title_x = margin_x + logo_w + 0.45 * cm
        doc.setFillColor(theme["ink"])
        doc.setFont("Helvetica-Bold", 22)
        doc.drawString(title_x, top_y - 10, "TRX-AI")

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica", 11)
        doc.drawString(title_x, top_y - 28, subtitle)

        doc.setFillColor(theme["muted"])
        doc.setFont("Helvetica", 10)
        doc.drawString(title_x, top_y - 42, tagline)

        sep_y = top_y - 52
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(1)
        doc.line(margin_x, sep_y, page_width - margin_x, sep_y)

        return sep_y - 16

    def _draw_card_section(
        self,
        doc: Any,
        title: str,
        lines: list[str],
        y: float,
        x: float,
        content_width: float,
        theme: dict[str, Any],
    ) -> float:
        if title.upper() == "LLM FIXED CODE":
            return self._draw_code_card_section(
                doc,
                title,
                lines,
                y,
                x,
                content_width,
                theme,
            )

        max_chars = 98
        wrapped: list[str] = []
        for entry in lines:
            wrapped.extend(self._wrap_lines(entry, max_chars=max_chars))

        line_height = 13
        card_h = 26 + (len(wrapped) * line_height) + 10

        doc.setFillColor(theme["card_bg"])
        doc.setStrokeColor(theme["card_line"])
        doc.setLineWidth(1)
        doc.roundRect(x, y - card_h, content_width, card_h, 8, fill=1, stroke=1)

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 11)
        doc.drawString(x + 12, y - 18, title)

        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(x + 10, y - 23, x + content_width - 10, y - 23)

        body_y = y - 36
        doc.setFillColor(theme["ink"])
        doc.setFont("Helvetica", 10)

        for idx, text_line in enumerate(wrapped):
            prefix = "-> " if idx == 0 else "  "
            doc.drawString(x + 14, body_y, f"{prefix}{text_line}")
            body_y -= line_height

        return y - card_h - 10

    def _draw_code_card_section(
        self,
        doc: Any,
        title: str,
        lines: list[str],
        y: float,
        x: float,
        content_width: float,
        theme: dict[str, Any],
    ) -> float:
        from reportlab.lib import colors

        max_chars = 100
        wrapped: list[str] = []
        for raw in lines:
            line = raw.expandtabs(4).rstrip("\n")
            if line.strip() == "```":
                continue
            parts = textwrap.wrap(
                line,
                width=max_chars,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            wrapped.extend(parts if parts else [""])

        line_height = 12
        card_h = 30 + (len(wrapped) * line_height) + 14

        doc.setFillColor(theme["card_bg"])
        doc.setStrokeColor(theme["card_line"])
        doc.setLineWidth(1)
        doc.roundRect(x, y - card_h, content_width, card_h, 8, fill=1, stroke=1)

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 11)
        doc.drawString(x + 12, y - 18, title)

        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(x + 10, y - 23, x + content_width - 10, y - 23)

        code_x = x + 12
        code_top = y - 30
        code_w = content_width - 24
        code_h = card_h - 42
        doc.setFillColor(colors.HexColor("#F2F6FA"))
        doc.roundRect(code_x, code_top - code_h, code_w, code_h, 4, fill=1, stroke=0)

        body_y = code_top - 14
        doc.setFillColor(colors.HexColor("#0E2238"))
        doc.setFont("Courier", 9)

        for text_line in wrapped:
            doc.drawString(code_x + 8, body_y, text_line)
            body_y -= line_height

        return y - card_h - 10

    @staticmethod
    def _draw_card_image(
        doc: Any,
        title: str,
        image_path: Path,
        y: float,
        x: float,
        content_width: float,
        theme: dict[str, Any],
    ) -> float:
        from reportlab.lib.units import cm

        card_h = 6.8 * cm
        doc.setFillColor(theme["card_bg"])
        doc.setStrokeColor(theme["card_line"])
        doc.setLineWidth(1)
        doc.roundRect(x, y - card_h, content_width, card_h, 8, fill=1, stroke=1)

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 11)
        doc.drawString(x + 12, y - 18, title)

        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.7)
        doc.line(x + 10, y - 23, x + content_width - 10, y - 23)

        img_x = x + 12
        img_y = y - card_h + 12
        img_w = content_width - 24
        img_h = card_h - 42
        doc.drawImage(
            str(image_path),
            img_x,
            img_y,
            width=img_w,
            height=img_h,
            preserveAspectRatio=True,
            mask="auto",
        )

        return y - card_h - 10

    @staticmethod
    def _draw_footer(
        doc: Any,
        page_width: float,
        margin_x: float,
        page_number: int,
    ) -> None:
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        y = 1.7 * cm
        doc.setStrokeColor(colors.HexColor("#C8E2F4"))
        doc.setLineWidth(1)
        doc.line(margin_x, y + 0.5 * cm, page_width - margin_x, y + 0.5 * cm)

        doc.setFillColor(colors.HexColor("#4A5D73"))
        doc.setFont("Helvetica", 9)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc.drawString(margin_x, y, "Generated by TRX-AI v1.0 | Evaluation Module")
        doc.drawRightString(page_width - margin_x, y, f"{timestamp} | Page {page_number}")

    def _draw_section_heading(
        self,
        doc: Any,
        title: str,
        y: float,
        x_left: float,
        x_right: float,
    ) -> float:
        from reportlab.lib import colors

        doc.setFillColor(colors.HexColor("#0A67A3"))
        doc.setFont("Helvetica-Bold", 12)
        doc.drawString(x_left, y, title)
        y -= 4
        doc.setStrokeColor(colors.HexColor("#88C8EC"))
        doc.setLineWidth(0.7)
        doc.line(x_left, y, x_right, y)
        return y - 14

    def _draw_section_block(
        self,
        doc: Any,
        title: str,
        lines: list[str],
        y: float,
        x_left: float,
        x_right: float,
        line_gap: int,
        *,
        nested: bool = False,
    ) -> float:
        from reportlab.lib import colors
        from reportlab.lib.units import cm

        y = self._draw_section_heading(doc, title, y, x_left, x_right)

        body_left = x_left + (14 if nested else 6)
        max_chars = 94 if nested else 98
        doc.setFillColor(colors.HexColor("#102A43"))
        doc.setFont("Helvetica", 10)

        for item in lines:
            wrapped_item = self._wrap_lines(item, max_chars=max_chars)
            for idx, part in enumerate(wrapped_item):
                prefix = "- " if idx == 0 else "  "
                doc.drawString(body_left, y, f"{prefix}{part}")
                y -= line_gap
                if y < 3.2 * cm:
                    doc.showPage()
                    y = 26.5 * cm
                    doc.setFillColor(colors.HexColor("#102A43"))
                    doc.setFont("Helvetica", 10)

        return y - 6

    @staticmethod
    def _ensure_logo_image() -> Path | None:
        """Creates a simple logo concept under assets/logo.png if missing."""
        logo_path = Path("assets") / "logo.png"
        logo_path.parent.mkdir(parents=True, exist_ok=True)

        if logo_path.exists():
            return logo_path

        try:
            from reportlab.graphics import renderPM
            from reportlab.graphics.shapes import Drawing, Rect, String
            from reportlab.lib import colors

            drawing = Drawing(360, 120)
            drawing.add(Rect(0, 0, 360, 120, fillColor=colors.HexColor("#0B1E33"), strokeColor=colors.HexColor("#00AEEF"), strokeWidth=2.5))
            drawing.add(Rect(14, 16, 332, 88, fillColor=colors.HexColor("#132F4A"), strokeColor=colors.HexColor("#2EA4E7"), strokeWidth=1.2))
            drawing.add(String(26, 62, "TRX-AI", fontName="Helvetica-Bold", fontSize=40, fillColor=colors.HexColor("#00D8FF")))
            drawing.add(String(28, 28, "Reality Debugger", fontName="Helvetica", fontSize=15, fillColor=colors.HexColor("#83CAFF")))
            renderPM.drawToFile(drawing, str(logo_path), fmt="PNG")
            return logo_path if logo_path.exists() else None
        except Exception:
            return None

