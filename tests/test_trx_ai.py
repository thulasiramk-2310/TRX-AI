from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from rich.console import Console

import main
from analyzer import RealityAnalyzer
from config import AppConfig
from formatter import OutputFormatter


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

    def test_python_validation(self) -> None:
        self.assertTrue(RealityAnalyzer._is_valid_python_code("def f():\n    return 1\n"))
        self.assertFalse(RealityAnalyzer._is_valid_python_code("def f(:\n    return 1\n"))


class FormatterTests(unittest.TestCase):
    def test_cp1252_render_does_not_crash(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
        console = Console(file=stream, force_terminal=False, color_system=None, width=100)
        formatter = OutputFormatter(console)
        analysis = {
            "response_mode": "analysis",
            "debug_analysis": ["Fibonacci uses O(2ⁿ) recursion"],
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
        self.assertIn("TRX-AI Assistant", output)


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


if __name__ == "__main__":
    unittest.main()
