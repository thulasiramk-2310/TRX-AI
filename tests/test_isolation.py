from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import evaluation
from analyzer import RealityAnalyzer
from config import AppConfig
from evaluation import EvalCase, evaluate_trx_ai


class FailureContextTests(unittest.TestCase):
    """Failure memory must never inject another input's code into a review prompt."""

    def setUp(self) -> None:
        with patch.object(RealityAnalyzer, "_load_failure_memory", return_value=[]):
            self.analyzer = RealityAnalyzer(AppConfig.from_env())

    def test_snippet_contains_reasons_not_foreign_code(self) -> None:
        self.analyzer._failure_memory = [
            {"input": "def bubble(arr):\n    n=len(arr)", "bad_output": "", "failure_reason": "llm_unavailable"},
            {"input": "def fib(n):\n    return fib(n-1)", "bad_output": "", "failure_reason": "fixed_code_missing"},
            {"input": "def sqrt(x): pass", "bad_output": "", "failure_reason": "fixed_code_missing"},
        ]
        snippet = self.analyzer._failure_context_snippet()

        self.assertNotIn("def bubble", snippet)
        self.assertNotIn("def fib", snippet)
        self.assertNotIn("input=", snippet)
        self.assertIn("fixed_code_missing", snippet)
        self.assertEqual(snippet.count("fixed_code_missing"), 1)  # deduplicated

    def test_review_prompt_has_only_the_code_under_review(self) -> None:
        self.analyzer._failure_memory = [
            {"input": "def bubble(arr): pass", "bad_output": "", "failure_reason": "llm_unavailable"}
        ]
        prompt = self.analyzer._build_code_review_prompt("def percent(a, b):\n    return a / b", "python")
        self.assertIn("def percent", prompt)
        self.assertNotIn("def bubble", prompt)


class EvalIsolationTests(unittest.TestCase):
    def test_failure_memory_reset_per_case_and_not_persisted(self) -> None:
        seen: list[int] = []

        class Analyzer:
            config = SimpleNamespace(local_llm_model="m")

            def __init__(self) -> None:
                self._failure_memory = [{"input": "user stuff", "failure_reason": "x"}]
                self.saved = 0

            def _save_failure_memory(self) -> None:
                self.saved += 1

            def analyze_code_multi_agent(self, code: str) -> dict:
                seen.append(len(self._failure_memory))
                self._failure_memory.append({"input": code, "failure_reason": "fixed_code_missing"})
                self._save_failure_memory()
                return {"debug_analysis": ["x"], "system_status": ["source=llm"]}

        analyzer = Analyzer()
        cases = [EvalCase(f"code-{i}", "a", "b", "bug", "python", id=str(i)) for i in range(3)]
        evaluate_trx_ai(analyzer, cases, run_baseline=False, save_report=False, verbose=False)

        self.assertEqual(seen, [0, 0, 0])
        self.assertEqual(analyzer.saved, 0)  # eval never writes the user's sessions/failure_memory.json
        self.assertEqual(analyzer._failure_memory, [{"input": "user stuff", "failure_reason": "x"}])
        self.assertIsNot(evaluation, None)


if __name__ == "__main__":
    unittest.main()
