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

        cards = [
            ("SUMMARY", summary_lines),
            ("INPUT", [user_input]),
            ("MODE", [mode.upper()]),
            ("DEBUG ANALYSIS", parsed_sections.get("DEBUG ANALYSIS", parsed_sections.get("INPUT ANALYSIS", ["N/A"]))),
            ("IMPROVEMENTS", parsed_sections.get("IMPROVEMENTS", parsed_sections.get("FIX SUGGESTIONS", ["N/A"]))),
            ("PREDICTIONS", parsed_sections.get("PREDICTIONS", ["N/A"])),
            ("FINAL INSIGHT", parsed_sections.get("FINAL INSIGHT", ["N/A"])),
            ("CONFIDENCE SCORE", confidence),
        ]

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

        doc.setFillColor(theme["brand_dark"])
        doc.setFont("Helvetica-Bold", 13)
        doc.drawString(margin_x, y, "COMPARISON SUMMARY")
        y -= 8
        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.8)
        doc.line(margin_x, y, width - margin_x, y)
        y -= 14

        parsed_first = self._parse_structured_sections(first_structured_output)
        parsed_second = self._parse_structured_sections(second_structured_output)
        metrics_first = self._extract_comparison_metrics(parsed_first)
        metrics_second = self._extract_comparison_metrics(parsed_second)

        summary_lines = [
            f"Mode: {mode.upper()}",
            f"{first_label}: {first_input}",
            f"{second_label}: {second_input}",
        ]
        doc.setFillColor(theme["ink"])
        doc.setFont("Helvetica", 10)
        for line in summary_lines:
            wrapped = self._wrap_lines(line, max_chars=105)
            for part in wrapped:
                doc.drawString(margin_x, y, f"- {part}")
                y -= 14

        y -= 6

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
        y -= card_h + 12

        recommendation = self._comparison_recommendation(metrics_first, metrics_second, first_label, second_label)
        y = self._draw_card_section(
            doc,
            "RECOMMENDATION",
            recommendation,
            y,
            margin_x,
            content_width,
            theme,
        )

        self._draw_footer(doc, width, margin_x, 1)
        doc.save()
        return destination

    @staticmethod
    def _parse_structured_sections(structured_output: str) -> dict[str, list[str]]:
        sections: dict[str, list[str]] = {}
        current: str | None = None

        for raw_line in structured_output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip().upper()
                sections.setdefault(current, [])
                continue
            if current is None:
                continue
            if line.startswith("- "):
                sections[current].append(line[2:].strip())
            else:
                sections[current].append(line)

        return sections

    @staticmethod
    def _wrap_lines(text: str, max_chars: int) -> list[str]:
        wrapped = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)
        return wrapped if wrapped else [""]

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

        chart_left = x + 22
        chart_bottom = y + 22
        chart_width = width - 44
        chart_height = height - 58

        doc.setStrokeColor(theme["line"])
        doc.setLineWidth(0.8)
        doc.line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height)
        doc.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)

        max_val = max(10.0, max(max(first, second) for _, first, second in metrics))
        scale = chart_height / max_val

        group_width = chart_width / len(metrics)
        bar_width = min(14, (group_width * 0.62) / 2)

        for idx, (label, first_val, second_val) in enumerate(metrics):
            gx = chart_left + (idx * group_width) + (group_width * 0.18)
            h1 = first_val * scale
            h2 = second_val * scale

            doc.setFillColor(colors.HexColor("#00AEEF"))
            doc.rect(gx, chart_bottom, bar_width, h1, fill=1, stroke=0)

            doc.setFillColor(colors.HexColor("#1A4F8B"))
            doc.rect(gx + bar_width + 4, chart_bottom, bar_width, h2, fill=1, stroke=0)

            doc.setFillColor(theme["muted"])
            doc.setFont("Helvetica", 8)
            doc.drawCentredString(gx + bar_width, chart_bottom - 11, label)

        doc.setFillColor(theme["muted"])
        doc.setFont("Helvetica", 8)
        doc.drawString(chart_left, y + 6, f"{first_label} = cyan")
        doc.drawString(chart_left + 130, y + 6, f"{second_label} = blue")

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
        doc.drawString(margin_x, y, "Generated by TRX-AI | v1.0")
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
