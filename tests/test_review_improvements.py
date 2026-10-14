"""Review improvements must not invent quality judgments or execute user code."""
import unittest
from unittest.mock import patch

from analyzer_helpers import analyzer_stub
from offline_review import scan_python


class ReviewImprovementTests(unittest.TestCase):
    def test_critic_requests_constrained_json_and_invalid_response_skips_rewrites(self):
        analyzer = analyzer_stub()
        with patch.object(analyzer, "_failure_context_snippet", return_value=""), \
                patch.object(analyzer, "_call_local", side_effect=[
                    {"ok": True, "text": "notes"}, {"ok": True, "text": "review"},
                    {"ok": True, "text": '{"score": null, "issues": []}'},
                ]) as model:
            result = analyzer._run_code_review_multi_agent_pipeline("x = 1", "python")
        self.assertFalse(result["critic_available"])
        self.assertEqual(result["text"], "review")
        self.assertEqual(model.call_count, 3)
        options = model.call_args.kwargs
        self.assertEqual(options["response_format"]["required"], ["score", "issues"])
        self.assertEqual(options["temperature"], 0.0)

    def test_critic_transport_failure_preserves_candidate(self):
        analyzer = analyzer_stub()
        with patch.object(analyzer, "_failure_context_snippet", return_value=""), \
                patch.object(analyzer, "_call_local", side_effect=[
                    {"ok": True, "text": "notes"}, {"ok": True, "text": "review"}, {"ok": False},
                ]) as model:
            result = analyzer._run_code_review_multi_agent_pipeline("x = 1", "python")
        self.assertTrue(result["ok"])
        self.assertIn("critic_unavailable", result["steps"])
        self.assertEqual(model.call_count, 3)

    def test_python_scanner_finds_concrete_risks(self):
        cases = [
            ("def first(items):\n    return items[0]\n", "empty sequence", "code_debug"),
            ("def add(item, values=[]):\n    values.append(item)\n", "shared across calls", "code_debug"),
            ("result = value / 0\n", "ZeroDivisionError", "code_debug"),
            ("if value is 'yes':\n    pass\n", "Identity comparison", "code_debug"),
            ("sorted(items)\n", "discarded", "code_debug"),
            ("def fib(n):\n    return fib(n-1) + fib(n-2)\n", "exponential", "performance"),
            ("requests.get(url)\n", "timeout", "security"),
        ]
        for code, message, category in cases:
            with self.subTest(message=message):
                self.assertIn(message, " ".join(scan_python(code)[category]))

    def test_guarded_access_and_finite_timeout_are_not_flagged(self):
        code = "def first(items):\n    if not items:\n        return None\n    return items[0]\n\nrequests.get(url, timeout=5)\n"
        findings = scan_python(code)
        self.assertFalse(findings["code_debug"])
        self.assertFalse(findings["security"])

    def test_other_languages_and_invalid_python_fail_gracefully(self):
        for code in ("function foo() { return x; }", "def invalid(:", ""):
            self.assertFalse(any(scan_python(code).values()))
