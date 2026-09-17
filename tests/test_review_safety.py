"""Review cache recovery and complete-input fix generation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analyzer_helpers import analyzer_stub


class ReviewSafetyTests(unittest.TestCase):
    def test_review_cache_expires_in_memory_and_after_disk_reload(self):
        analyzer = analyzer_stub()
        with tempfile.TemporaryDirectory() as tmp:
            analyzer.config.review_cache_disk_enabled = True
            analyzer._review_cache_disk_path = Path(tmp) / "cache.json"
            with patch("analyzer.time.time", return_value=1000):
                analyzer._review_cache_set("key", {"analysis_source": "llm", "fixed_code": "x = 1"})
            restored = analyzer_stub()
            restored.config.review_cache_disk_enabled = True
            restored._review_cache_disk_path = analyzer._review_cache_disk_path
            restored._load_review_cache_disk()
            with patch("analyzer.time.time", return_value=1100):
                self.assertIsNotNone(restored._review_cache_get("key"))
            with patch("analyzer.time.time", return_value=1301):
                self.assertIsNone(analyzer._review_cache_get("key"))
                self.assertIsNone(restored._review_cache_get("key"))

    def test_degraded_review_is_not_cached(self):
        analyzer = analyzer_stub()
        analyzer._review_cache_set("key", {"analysis_source": "rule", "fixed_code": "x = 1"})
        self.assertIsNone(analyzer._review_cache_get("key"))

    def test_review_retry_recovers_after_model_outage(self):
        analyzer = analyzer_stub()
        successful = {"ok": True, "text": "FIXED CODE:\n```python\nx = 1\n```", "steps": []}
        with patch.object(analyzer, "_build_graph_context", return_value={}), \
                patch.object(analyzer, "_attach_graph_insights"), \
                patch.object(analyzer, "_record_failure"), \
                patch.object(analyzer, "_mcp_status_tags", return_value=[]), \
                patch.object(analyzer, "_run_code_review_multi_agent_pipeline",
                             side_effect=[{"ok": False}, successful]) as pipeline:
            self.assertEqual(analyzer.analyze_code_multi_agent("x = 1")["analysis_source"], "rule")
            self.assertEqual(analyzer.analyze_code_multi_agent("x = 1")["analysis_source"], "llm")
            self.assertEqual(pipeline.call_count, 2)

    def test_incomplete_review_cannot_return_whole_file_replacement(self):
        analyzer = analyzer_stub()
        code = "def first():\n    return 1\n" + "# filler\n" * 300 + "def last():\n    return 2\n"
        partial = {"ok": True, "text": "FIXED CODE:\n```python\ndef first():\n    return 1\n```", "steps": []}
        with patch.object(analyzer, "_build_graph_context", return_value={}), \
                patch.object(analyzer, "_attach_graph_insights"), \
                patch.object(analyzer, "_record_failure"), \
                patch.object(analyzer, "_mcp_status_tags", return_value=[]), \
                patch.object(analyzer, "_run_code_review_multi_agent_pipeline", return_value=partial):
            result = analyzer.analyze_code_multi_agent(code)
        self.assertEqual(result["fixed_code"], "")
        self.assertIn("fix_blocked_incomplete_input", result["system_status"])
        self.assertIn("Partial review", result["analysis_text"])
        with patch.object(analyzer, "_call_local") as llm:
            self.assertEqual(analyzer._generate_fixed_code_only(code, "python"), "")
            llm.assert_not_called()

    def test_fix_generation_includes_entire_small_file(self):
        analyzer = analyzer_stub()
        code = "# filler\n" * 248 + "def last():\n    return 2\n"
        self.assertGreater(len(code), 2200)
        with patch.object(analyzer, "_call_local", return_value={"ok": True, "text": code}) as llm:
            self.assertEqual(analyzer._generate_fixed_code_only(code, "python"), code.strip())
        self.assertIn("def last():", llm.call_args.args[0])
        self.assertIn("untrusted input", llm.call_args.args[0])
