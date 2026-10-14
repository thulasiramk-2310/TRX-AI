"""Inspect actual exported PDF content, including pagination and code preservation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader
from history import SessionHistory


class PdfReportTests(unittest.TestCase):
    def export(self, content: str, destination: Path):
        history = SessionHistory()
        with patch.object(history, "_prepare_report_path", return_value=destination):
            history.export_pdf_report("preview.pdf", "review sample.py", "review", content)
        return PdfReader(str(destination))

    def test_report_omits_empty_sections_and_preserves_fenced_code(self):
        content = "[DEBUG ANALYSIS]\n- Missing validation.\n[PREDICTIONS]\n- N/A\n[LLM FIXED CODE]\n```python\n[NAME]\nresult = 1\n```\n[CONFIDENCE SCORE]\n80%"
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self.export(content, Path(tmp) / "report.pdf")
            text = "\n".join(page.extract_text() for page in pdf.pages)
        self.assertIn("Code review", text)
        self.assertIn("Missing validation", text)
        self.assertIn("[NAME]", text)
        self.assertNotIn("N/A", text)
        self.assertNotIn("Performance and outlook", text)

    def test_plex_fonts_are_embedded_in_the_export(self):
        content = "[DEBUG ANALYSIS]\n- Validate input.\n[LLM FIXED CODE]\n```python\nvalue = 1\n```"
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self.export(content, Path(tmp) / "report.pdf")
            fonts = [font.get_object() for font in pdf.pages[0]["/Resources"]["/Font"].values()]
            plex = [font for font in fonts if "IBMPlex" in str(font.get("/BaseFont", ""))]
            self.assertGreaterEqual(len(plex), 3)
            self.assertTrue(any("Mono" in str(font["/BaseFont"]) for font in plex))
            for font in plex:
                descriptor = font["/FontDescriptor"].get_object()
                self.assertGreater(len(descriptor["/FontFile2"].get_object().get_data()), 0)

    def test_long_code_is_paginated_without_truncation(self):
        content = "[LLM FIXED CODE]\n```python\n" + "\n".join(f"value_{i} = {i}" for i in range(300)) + "\n```"
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self.export(content, Path(tmp) / "report.pdf")
            text = "\n".join(page.extract_text() for page in pdf.pages)
        self.assertGreater(len(pdf.pages), 1)
        self.assertIn("value_299 = 299", text)
        self.assertNotIn("truncated", text)
        self.assertIn("Suggested code / continued", text)

    def test_comparison_does_not_claim_missing_findings_are_resolved(self):
        history = SessionHistory()
        with tempfile.TemporaryDirectory() as tmp, patch.object(history, "_prepare_report_path",
                                                               return_value=Path(tmp) / "report.pdf"):
            path = history.export_comparison_pdf_report("report.pdf", first_input="before", second_input="after",
                mode="review", first_structured_output="[DEBUG ANALYSIS]\nMissing validation.\n[CONFIDENCE SCORE]\n80%",
                second_structured_output="[IMPROVEMENTS]\nCheck input.\n[CONFIDENCE SCORE]\n50%")
            pdf = PdfReader(str(path))
            text = "\n".join(page.extract_text() for page in pdf.pages)
        self.assertEqual(len(pdf.pages), 1)
        self.assertIn("-30 pp", text)
        self.assertIn("Only in earlier report", text)
        self.assertIn("does not prove it was fixed", text)
        self.assertNotIn("RESOLVED", text)
