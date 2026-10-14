"""Compact, paginated PDF reports for developers."""
from datetime import datetime
import re
from typing import Any

from guardrails import strip_terminal_controls


def parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    inside_code = False
    for raw in text.splitlines():
        line = raw.strip()
        if current == "LLM FIXED CODE" and line.startswith("```"):
            inside_code = not inside_code
            continue
        if inside_code:
            sections[current].append(raw)
        elif line.startswith("[") and line.endswith("]"):
            current = line[1:-1].upper().strip()
            sections.setdefault(current, [])
        elif current and (line or current == "LLM FIXED CODE"):
            sections[current].append(raw if current == "LLM FIXED CODE" else re.sub(r"^[-*] ", "", line))
    return sections


def _plain(text: str) -> str:
    return strip_terminal_controls(text).replace("**", "").replace("`", "").translate(str.maketrans({
        "\u2013": "-", "\u2014": "-", "\u2011": "-", "\u2192": "->", "\u2713": "OK", "\u2717": "!",
    }))


def _items(sections: dict[str, list[str]], *names: str) -> list[str]:
    for name in names:
        values = [line for line in sections.get(name, [])
                  if line.strip().lower() not in {"", "n/a", "none", "no predictions available", "no improvements available"}]
        if values:
            return values
    return []


def _confidence(sections: dict[str, list[str]]) -> float | None:
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*%", " ".join(sections.get("CONFIDENCE SCORE", [])))
    return min(100.0, float(match[1])) if match else None


class Report:
    """Canvas flow layout: measured lines, small running headers, no oversized cards."""

    def __init__(self, canvas: Any, size: tuple[float, float], title: str, metadata: str) -> None:
        from reportlab.lib.colors import HexColor

        self.doc = canvas
        self.width, self.height = size
        self.margin = 50.0
        self.content_width = self.width - self.margin * 2
        self.title = title
        self.metadata = metadata
        self.page = 1
        self.section = ""
        self.ink = HexColor("#222830")
        self.muted = HexColor("#68717D")
        self.accent = HexColor("#455E7A")
        self.line = HexColor("#DFE3E8")
        self.surface = HexColor("#F4F6F8")
        self.doc.setTitle(title)
        self.doc.setAuthor("TRX-AI")
        self._start_page(first=True)

    def _start_page(self, *, first: bool = False) -> None:
        d = self.doc
        d.setFillColor(self.accent)
        d.setFont("Courier-Bold", 9)
        d.drawString(self.margin, self.height - 42, "TRX / REPORT")
        d.setFillColor(self.muted)
        d.setFont("Helvetica", 8)
        d.drawRightString(self.width - self.margin, self.height - 42, datetime.now().strftime("%d %b %Y"))
        if first:
            d.setFillColor(self.ink)
            d.setFont("Helvetica-Bold", 24)
            d.drawString(self.margin, self.height - 83, self.title)
            d.setFillColor(self.muted)
            d.setFont("Helvetica", 9)
            d.drawString(self.margin, self.height - 104, self.metadata)
            self.y = self.height - 119
        else:
            self.y = self.height - 57
        d.setStrokeColor(self.line)
        d.setLineWidth(0.6)
        d.line(self.margin, self.y, self.width - self.margin, self.y)
        self.y -= 22

    def _footer(self) -> None:
        d = self.doc
        d.setStrokeColor(self.line)
        d.setLineWidth(0.5)
        d.line(self.margin, 44, self.width - self.margin, 44)
        d.setFillColor(self.muted)
        d.setFont("Helvetica", 8)
        d.drawString(self.margin, 29, "TRX-AI 0.2")
        d.drawRightString(self.width - self.margin, 29, str(self.page))

    def ensure(self, height: float) -> None:
        if self.y - height >= 65:
            return
        self._footer()
        self.doc.showPage()
        self.page += 1
        self._start_page()
        if self.section:
            self.doc.setFillColor(self.muted)
            self.doc.setFont("Helvetica-Bold", 10)
            self.doc.drawString(self.margin, self.y, self.section + " / continued")
            self.y -= 24

    @staticmethod
    def wrap(text: str, width: float, font: str, size: float, *, code: bool = False) -> list[str]:
        from reportlab.pdfbase.pdfmetrics import stringWidth

        rows = []
        for raw in str(text).splitlines() or [""]:
            remaining = raw.expandtabs(4)
            while remaining and stringWidth(remaining, font, size) > width:
                low, high = 1, len(remaining)
                while low < high:
                    mid = (low + high + 1) // 2
                    if stringWidth(remaining[:mid], font, size) <= width:
                        low = mid
                    else:
                        high = mid - 1
                cut = low
                if not code:
                    space = remaining.rfind(" ", 0, cut + 1)
                    if space > 0:
                        cut = space
                rows.append(remaining[:cut])
                remaining = remaining[cut:] if code else remaining[cut:].lstrip()
            rows.append(remaining)
        return rows

    def heading(self, title: str) -> None:
        self.section = ""
        self.ensure(45)
        self.section = title
        self.y -= 10
        self.doc.setFillColor(self.ink)
        self.doc.setFont("Helvetica-Bold", 11)
        self.doc.drawString(self.margin, self.y, title)
        self.y -= 22

    def paragraph(self, text: str, *, bullet: bool = False, muted: bool = False) -> None:
        x = self.margin + (12 if bullet else 0)
        rows = self.wrap(_plain(text), self.content_width - (12 if bullet else 0), "Helvetica", 10)
        for index, row in enumerate(rows):
            self.ensure(15)
            self.doc.setFillColor(self.muted if muted else self.ink)
            self.doc.setFont("Helvetica", 10)
            if bullet and index == 0:
                self.doc.drawString(self.margin, self.y, "-")
            self.doc.drawString(x, self.y, row)
            self.y -= 15
        self.y -= 5

    def code(self, lines: list[str]) -> None:
        self.heading("Suggested code")
        number = 0
        for raw in lines:
            if raw.strip().startswith("```"):
                continue
            number += 1
            # Keep code indentation and all lines; wrapping is purely visual.
            rows = self.wrap(strip_terminal_controls(raw), self.content_width - 48, "Courier", 8.5, code=True)
            for index, row in enumerate(rows):
                self.ensure(14)
                d = self.doc
                d.setFillColor(self.surface)
                d.rect(self.margin, self.y - 4, self.content_width, 13, stroke=0, fill=1)
                d.setFillColor(self.muted)
                d.setFont("Courier", 7.5)
                if index == 0:
                    d.drawRightString(self.margin + 28, self.y, str(number))
                d.setFillColor(self.ink)
                d.setFont("Courier", 8.5)
                d.drawString(self.margin + 40, self.y, row)
                self.y -= 13
        self.y -= 16

    def metrics(self, rows: list[tuple[str, str, str, str]], first_label: str, second_label: str) -> None:
        self.heading("At a glance")
        positions = [self.margin, self.margin + self.content_width * 0.40,
                     self.margin + self.content_width * 0.61, self.margin + self.content_width * 0.82]
        for index, row in enumerate([("Metric", first_label, second_label, "Change"), *rows]):
            wrapped = [self.wrap(_plain(value), self.content_width * (0.37 if i == 0 else 0.18),
                                 "Helvetica-Bold" if index == 0 else "Helvetica", 9) for i, value in enumerate(row)]
            height = max(map(len, wrapped)) * 13 + 14
            self.ensure(height)
            d = self.doc
            d.setFillColor(self.muted if index == 0 else self.ink)
            d.setFont("Helvetica-Bold" if index == 0 else "Helvetica", 9)
            for x, values in zip(positions, wrapped, strict=True):
                for number, value in enumerate(values):
                    d.drawString(x, self.y - number * 13, value)
            self.y -= height
            d.setStrokeColor(self.line)
            d.setLineWidth(0.5)
            d.line(self.margin, self.y + 7, self.width - self.margin, self.y + 7)
        self.y -= 8

    def save(self) -> None:
        self._footer()
        self.doc.save()


def render_report(canvas: Any, size: tuple[float, float], user_input: str, mode: str, text: str) -> None:
    sections = parse_sections(text)
    score = _confidence(sections)
    metadata = _plain(mode).upper() + (f"  /  Reported confidence {score:g}%" if score is not None else "")
    title = "Code review" if (mode.lower() in {"review", "code", "fix"} or "LLM FIXED CODE" in sections
                              or user_input.lower().startswith(("review ", "fix "))) else "Analysis report"
    report = Report(canvas, size, title, metadata)
    if user_input.strip():
        report.paragraph(user_input, muted=True)
    mapping = [
        ("Summary", ("FINAL INSIGHT", "FINAL SUMMARY")),
        ("Response", ("CHAT RESPONSE",)),
        ("Findings", ("DEBUG ANALYSIS", "BUGS DETECTED", "INPUT ANALYSIS")),
        ("Suggestions", ("IMPROVEMENTS", "FIX SUGGESTIONS")),
        ("Performance and outlook", ("PREDICTIONS", "PERFORMANCE")),
        ("Security", ("SECURITY",)),
        ("Proposed changes", ("CHANGES APPLIED",)),
        ("Complexity", ("CODE COMPLEXITY",)),
        ("Optimization options", ("OPTIMIZATION APPROACHES",)),
    ]
    shown = {"CONFIDENCE SCORE", "LLM FIXED CODE", *(name for _, names in mapping for name in names)}
    for title, names in mapping:
        values = _items(sections, *names)
        if values:
            report.heading(title)
            for value in values:
                report.paragraph(value, bullet=title not in {"Summary", "Response"})
    for name in sections:
        if name in shown:
            continue
        values = _items(sections, name)
        if values:
            report.heading(name.capitalize())
            for value in values:
                report.paragraph(value, bullet=True)
    if "LLM FIXED CODE" in sections:
        report.code(sections["LLM FIXED CODE"])
    if not sections:
        report.paragraph(text)
    report.save()


def render_comparison(canvas: Any, size: tuple[float, float], *, mode: str, first_input: str, second_input: str,
                      first_text: str, second_text: str, first_label: str, second_label: str) -> None:
    first, second = parse_sections(first_text), parse_sections(second_text)
    report = Report(canvas, size, "Review comparison", _plain(mode).upper())
    for label, target in ((first_label, first_input), (second_label, second_input)):
        report.paragraph(f"{label}: {target}", muted=True)
    findings_first = _items(first, "DEBUG ANALYSIS", "BUGS DETECTED", "INPUT ANALYSIS")
    findings_second = _items(second, "DEBUG ANALYSIS", "BUGS DETECTED", "INPUT ANALYSIS")
    c1, c2 = _confidence(first), _confidence(second)
    rows = [("Reported confidence", f"{c1:g}%" if c1 is not None else "-", f"{c2:g}%" if c2 is not None else "-",
             f"{c2 - c1:+g} pp" if c1 is not None and c2 is not None else "-")]
    for label, a, b in (("Reported findings", findings_first, findings_second),
                        ("Suggestions", _items(first, "IMPROVEMENTS", "FIX SUGGESTIONS"),
                         _items(second, "IMPROVEMENTS", "FIX SUGGESTIONS"))):
        rows.append((label, str(len(a)), str(len(b)), f"{len(b) - len(a):+d}"))
    report.metrics(rows, first_label, second_label)
    report.paragraph("Confidence is a reported estimate, not measured correctness. Counts describe report content.", muted=True)
    normalize = lambda value: re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    left = {normalize(value): value for value in findings_first}
    right = {normalize(value): value for value in findings_second}
    for title, keys, lookup in (("Only in earlier report", left.keys() - right.keys(), left),
                                ("In both reports", left.keys() & right.keys(), right),
                                ("Only in later report", right.keys() - left.keys(), right)):
        if keys:
            report.heading(title)
            for key in sorted(keys):
                report.paragraph(lookup[key], bullet=True)
    report.heading("Interpretation")
    if c1 is not None and c2 is not None:
        change = "increased" if c2 > c1 else "dropped" if c2 < c1 else "is unchanged"
        report.paragraph(f"{second_label} confidence {change} compared with {first_label}.")
    report.paragraph("Findings are matched by text. A finding missing from the later report does not prove it was fixed.", muted=True)
    report.save()
