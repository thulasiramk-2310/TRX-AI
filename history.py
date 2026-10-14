"""Session history handling for Reality Debugger."""

from __future__ import annotations

from guardrails import redact, redact_secrets
import json
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

    def recent_context(self, limit: int = 3, *, owner: str | None = None) -> list[dict[str, Any]]:
        """Returns the most recent session items for contextual reasoning."""
        if limit <= 0:
            return []
        entries = self.entries
        if owner is not None:
            entries = [entry for entry in entries
                       if entry.get("analysis", {}).get("current_user", "guest") == owner]
        return entries[-limit:]

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
            json.dump(redact(self.entries), session_file, indent=2)

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
        destination = self._report_path(file_name, default_name="report.txt", suffix=".txt")
        content = (
            "TRX-AI (Reality Debugger) Report\n"
            "================================\n"
            f"Input: {user_input}\n"
            f"Mode: {mode}\n\n"
            f"{structured_output}\n"
        )

        destination.write_text(redact_secrets(content)[0], encoding="utf-8")
        return destination

    def export_pdf_report(self, filename: str, user_input: str, mode: str, structured_output: str) -> Path:
        """Export a compact, paginated report with developer-friendly code listings."""
        from pdf_reports import render_report

        _colors, size, _cm, canvas = self._reportlab_or_raise(
            "PDF export requires reportlab. Install it with: pip install reportlab"
        )
        destination = self._prepare_report_path(filename, default_name="report.pdf")
        render_report(canvas.Canvas(str(destination), pagesize=size), size,
                      redact_secrets(user_input)[0], mode, redact_secrets(structured_output)[0])
        return destination

    def export_comparison_pdf_report(
        self, filename: str, *, first_input: str, second_input: str, mode: str,
        first_structured_output: str, second_structured_output: str,
        first_label: str = "Run 1", second_label: str = "Run 2",
    ) -> Path:
        """Compare report content with a compact metrics table and explicit limits."""
        from pdf_reports import render_comparison

        _colors, size, _cm, canvas = self._reportlab_or_raise(
            "Comparison PDF export requires reportlab. Install it with: pip install reportlab"
        )
        destination = self._prepare_report_path(filename, default_name="comparison_report.pdf")
        values = redact({"first_input": first_input, "second_input": second_input,
                         "first_text": first_structured_output, "second_text": second_structured_output,
                         "first_label": first_label, "second_label": second_label})
        render_comparison(canvas.Canvas(str(destination), pagesize=size), size, mode=mode, **values)
        return destination

    @staticmethod
    def _reportlab_or_raise(error_message: str):
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.pdfgen import canvas
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(error_message) from exc
        return colors, A4, cm, canvas

    @staticmethod
    def _prepare_report_path(filename: str, *, default_name: str) -> Path:
        return SessionHistory._report_path(filename, default_name=default_name, suffix=".pdf")

    @staticmethod
    def _report_path(filename: str, *, default_name: str, suffix: str) -> Path:
        reports_dir = Path("reports").resolve()
        safe_name = filename.strip() or default_name
        if not safe_name.lower().endswith(suffix):
            safe_name += suffix
        destination = (reports_dir / safe_name).resolve()
        if not destination.is_relative_to(reports_dir):
            raise ValueError("Report path must stay inside the reports folder.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

