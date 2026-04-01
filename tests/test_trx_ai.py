from __future__ import annotations

import io
import warnings
import unittest
from pathlib import Path
from unittest.mock import patch

warnings.simplefilter("ignore", DeprecationWarning)

from rich.console import Console

import main
from analyzer import RealityAnalyzer
from config import AppConfig
from formatter import OutputFormatter
from history import SessionHistory
from semantic_scoring import SemanticMatcher
from watcher import CodeChangeHandler

class AnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = RealityAnalyzer(AppConfig.from_env())

    def test_intent_priority_greeting_not_vague(self) -> None:
        result = self.analyzer.detect_intent_hybrid("hi")
        self.assertEqual(result.get("intent"), "greeting")
        self.assertEqual(result.get("source"), "rule")

    def test_parse_code_review_sections_markdown_headers(self) -> None:
        text = """### CODE DEBUG:
- Line 5: index issue

### CODE IMPROVEMENTS:
- Add bounds check

### PERFORMANCE:
- Use memoization

```python
def ignored():
    pass
```

### FIX SUGGESTIONS:
- Change loop

### FINAL SUMMARY:
- Fix index first

### CONFIDENCE:
- 88%
"""
        sections = RealityAnalyzer._parse_code_review_sections(text)
        self.assertIn("Line 5: index issue", sections["code_debug"][0])
        self.assertTrue(sections["fix_suggestions"])
        self.assertEqual(sections["confidence_score"], 88)

    def test_parse_code_review_sections_json(self) -> None:
        text = """{
  "code_debug": [{"issue":"Find Max","description":"Missing empty-list check"}],
  "code_improvements": ["Add guard clause"],
  "performance": ["Use memoization for fibonacci"],
  "security": ["Validate file paths"],
  "fix_suggestions": ["Add try/except for file IO"],
  "final_summary": "Fix input validation first",
  "confidence": "88%"
}"""
        sections = RealityAnalyzer._parse_code_review_sections(text)
        self.assertIn("Find Max: Missing empty-list check", sections["code_debug"][0])
        self.assertEqual(sections["confidence_score"], 88)
        self.assertTrue(sections["fix_suggestions"])

    def test_extract_fixed_code_from_json_payload(self) -> None:
        text = """{
  "fixed_code": [
    {"issue":"one","code":"def a():\\n    return 1"},
    {"issue":"two","code":"def b():\\n    return 2"}
  ]
}"""
        fixed = RealityAnalyzer._extract_fixed_code(text)
        self.assertIn("def a():", fixed)
        self.assertIn("def b():", fixed)

    def test_review_cache_hit(self) -> None:
        analyzer = RealityAnalyzer(AppConfig.from_env())
        call_count = {"n": 0}

        def fake_pipeline(_code: str, _lang: str) -> dict:
            call_count["n"] += 1
            return {
                "ok": True,
                "text": (
                    "CODE DEBUG:\n- Line 1 bug\n\n"
                    "CODE IMPROVEMENTS:\n- Improve naming\n\n"
                    "PERFORMANCE:\n- Use better loop\n\n"
                    "SECURITY:\n- Validate input\n\n"
                    "FIX SUGGESTIONS:\n- Add guard\n\n"
                    "FIXED CODE:\n```python\ndef x():\n    return 1\n```\n\n"
                    "FINAL SUMMARY:\n- Fix guard first\n\n"
                    "CONFIDENCE:\n- 80%\n"
                ),
                "critic_score": 0.9,
                "steps": ["analyzer", "generator", "critic_1"],
                "truncated": False,
            }

        with patch.object(analyzer, "_run_code_review_multi_agent_pipeline", side_effect=fake_pipeline):
            first = analyzer.analyze_code_multi_agent("def x():\n    return 1\n")
            second = analyzer.analyze_code_multi_agent("def x():\n    return 1\n")

        self.assertEqual(call_count["n"], 1)
        self.assertIn("cache_hit", second.get("system_status", []))
        self.assertEqual(first.get("fixed_code"), second.get("fixed_code"))

    def test_normalize_review_sections_dedupes(self) -> None:
        raw = {
            "code_debug": ["Line 1 issue", "Line 1 issue", "Line 2 issue"],
            "code_improvements": ["Refactor loop", "Refactor loop"],
            "performance": [],
            "security": [],
            "fix_suggestions": [],
            "final_summary": ["Fix line 1 first", "Fix line 1 first"],
            "confidence_score": 70,
        }
        normalized = RealityAnalyzer._normalize_review_sections(raw)
        self.assertEqual(len(normalized["code_debug"]), 2)
        self.assertEqual(len(normalized["code_improvements"]), 1)
        self.assertTrue(normalized["fix_suggestions"])

    def test_python_validation(self) -> None:
        self.assertTrue(RealityAnalyzer._is_valid_python_code("def f():\n    return 1\n"))
        self.assertFalse(RealityAnalyzer._is_valid_python_code("def f(:\n    return 1\n"))


    def test_fixed_code_rejects_undefined_identifier(self) -> None:
        analyzer = RealityAnalyzer(AppConfig.from_env())
        bad_fixed = (
            "def fibonacci(n):\n"
            "    if n < 0:\n"
            "        raise ValueError('Incorrect input')\n"
            "    elif n < len(FibArray):\n"
            "        return FibArray[n]\n"
            "    FibArray.append(fibonacci(n - 1) + fibonacci(n - 2))\n"
            "    return FibArray[n]\n"
        )
        self.assertFalse(analyzer._is_valid_fixed_code(bad_fixed, "python"))


    def test_optimization_gate_rejects_recursive_fibonacci_fix(self) -> None:
        analyzer = RealityAnalyzer(AppConfig.from_env())
        original = (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        )
        still_recursive = (
            "def fibonacci(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
        )
        self.assertFalse(analyzer._passes_python_optimization_gate(original, still_recursive))


class SemanticScoreTests(unittest.TestCase):
    def test_semantic_match_synonym(self) -> None:
        matcher = SemanticMatcher()
        score = matcher.score("add empty check", "include guard clause for empty input array")
        self.assertGreaterEqual(score, 0.20)

    def test_semantic_match_exact_phrase(self) -> None:
        matcher = SemanticMatcher()
        score = matcher.score("sql injection risk", "There is a SQL injection risk due to string concatenation.")
        self.assertGreaterEqual(score, 0.95)

class FormatterTests(unittest.TestCase):
    def test_cp1252_render_does_not_crash(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
        console = Console(file=stream, force_terminal=False, color_system=None, width=100)
        formatter = OutputFormatter(console)
        analysis = {
            "response_mode": "analysis",
            "debug_analysis": ["Fibonacci uses O(2â¿) recursion"],
            "improvements": ["Use iterative DP"],
            "predictions": ["Runtime will grow rapidly"],
            "final_insight": ["Prioritize complexity fix"],
            "confidence_score": 75,
            "intent": "review",
            "analysis_source": "llm",
            "system_status": ["intent=review", "source=llm"],
        }
        formatter.render(analysis, "debug", total_runs=1)
        stream.flush()
        output = buffer.getvalue().decode("cp1252", errors="replace")
        self.assertTrue(("TRX-AI Assistant" in output) or ("TRX-AI Futuristic Developer Dashboard" in output))


class CLISmokeTests(unittest.TestCase):
    def test_cli_help_and_exit_with_mocked_analyzer(self) -> None:
        class FakeAnalyzer:
            def __init__(self, _config: AppConfig) -> None:
                self._agents = ["debug", "improve", "predict"]

            def active_agents(self) -> list[str]:
                return self._agents

            def set_active_agents(self, agents: list[str]) -> list[str]:
                self._agents = ["debug", "improve", "predict"] if "all" in agents else agents
                return self._agents

            def analyze(self, _user_input: str, mode: str = "debug", past_context: list | None = None) -> dict:
                return {
                    "response_mode": "chat",
                    "chat_response": "ok",
                    "intent": "chat",
                    "intent_source": "rule",
                    "intent_confidence": 0.8,
                }

            def analyze_code_multi_agent(self, _code: str) -> dict:
                return {
                    "response_mode": "analysis",
                    "debug_analysis": ["Line 1: mock issue"],
                    "improvements": ["mock improvement"],
                    "predictions": ["mock performance"],
                    "final_insight": ["mock fix"],
                    "confidence_score": 80,
                    "intent": "review",
                    "intent_source": "llm",
                    "analysis_source": "llm",
                    "fixed_code": "print('ok')\n",
                    "system_status": ["intent=review", "source=llm"],
                }

        inputs = iter(["help", "exit"])

        def fake_input(_prompt: str) -> str:
            return next(inputs)

        with patch.object(main, "RealityAnalyzer", FakeAnalyzer), patch.object(main, "_print_startup", lambda _c: None):
            main.run_cli(input_fn=fake_input)


class WatcherTests(unittest.TestCase):
    def test_unchanged_content_is_skipped(self) -> None:
        handler = CodeChangeHandler(analyzer=object(), formatter=object())
        src = "demo.py"
        self.assertFalse(handler._is_unchanged_content(src, "print('x')\n"))
        self.assertTrue(handler._is_unchanged_content(src, "print('x')\n"))
        self.assertFalse(handler._is_unchanged_content(src, "print('y')\n"))


class HistoryExportTests(unittest.TestCase):
    def test_comparison_pdf_export_smoke(self) -> None:
        history = SessionHistory()
        try:
            out = history.export_comparison_pdf_report(
                "test_cmp_report.pdf",
                first_input="review run1",
                second_input="review run2",
                mode="review",
                first_structured_output="[DEBUG ANALYSIS]\n- A\n[CONFIDENCE SCORE]\n- 70%",
                second_structured_output="[DEBUG ANALYSIS]\n- B\n[CONFIDENCE SCORE]\n- 80%",
                first_label="Run 1",
                second_label="Run 2",
            )
            self.assertTrue(Path(out).exists())
        except RuntimeError:
            self.skipTest("reportlab not available in test environment")


if __name__ == "__main__":
    unittest.main()












