from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evaluation
from evaluation import EvalCase, evaluate_trx_ai, load_dataset, wilson_interval


class FakeAnalyzer:
    """Returns canned reviews keyed by input code; records cache clears."""

    def __init__(self, reviews: dict[str, dict]) -> None:
        self.reviews = reviews
        self.config = SimpleNamespace(
            local_llm_model="fake-model",
            local_llm_url="http://invalid",
            local_llm_timeout_seconds=1,
            local_llm_retries=1,
        )
        self._review_response_cache = {"stale": {}}
        self.cache_sizes_seen: list[int] = []

    def analyze_code_multi_agent(self, code: str) -> dict:
        self.cache_sizes_seen.append(len(self._review_response_cache))
        self._review_response_cache[code] = {}
        return self.reviews[code]


def _review(debug: str, fix: str, source: str = "llm") -> dict:
    return {
        "debug_analysis": [debug],
        "improvements": [fix],
        "predictions": ["Performance is fine."],
        "system_status": [f"source={source}"],
    }


CASES = [
    EvalCase("code-a", "sql injection", "parameterized query", "security", "python", id="a"),
    EvalCase("code-b", "mutable default argument", "use None default", "bug", "python", id="b"),
    EvalCase("code-c", "buffer overflow", "use strncpy with bounds", "security", "c", id="c"),
]


class WilsonIntervalTests(unittest.TestCase):
    def test_known_value(self) -> None:
        low, high = wilson_interval(9, 10)
        self.assertAlmostEqual(low, 0.5958, places=3)
        self.assertAlmostEqual(high, 0.9821, places=3)

    def test_bounds_and_empty(self) -> None:
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))
        low, high = wilson_interval(0, 5)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = wilson_interval(5, 5)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_interval_narrows_with_more_samples(self) -> None:
        small = wilson_interval(9, 10)
        large = wilson_interval(90, 100)
        self.assertLess(large[1] - large[0], small[1] - small[0])


class DatasetTests(unittest.TestCase):
    def test_bundled_dataset_is_valid(self) -> None:
        cases = load_dataset()
        self.assertGreaterEqual(len(cases), 50)
        self.assertEqual(len({c.id for c in cases}), len(cases))
        for case in cases:
            self.assertTrue(case.input_text.strip(), case.id)
            self.assertTrue(case.expected_debug.strip(), case.id)
            self.assertTrue(case.expected_fix.strip(), case.id)

    def test_generic_review_floor_stays_low_on_bundled_dataset(self) -> None:
        # Guards against labels so vague that a code-agnostic answer would "pass" them.
        cases = load_dataset()
        threshold = evaluation.DEFAULT_HIT_THRESHOLD
        debug_hits = sum(evaluation._semantic_match(c.expected_debug, evaluation.GENERIC_REVIEW) >= threshold for c in cases)
        fix_hits = sum(evaluation._semantic_match(c.expected_fix, evaluation.GENERIC_REVIEW) >= threshold for c in cases)
        self.assertLessEqual(debug_hits / len(cases), 0.05)
        self.assertLessEqual(fix_hits / len(cases), 0.10)

    def test_duplicate_ids_rejected(self) -> None:
        row = {"id": "x", "code": "c", "expected_debug": "d", "expected_fix": "f", "category": "bug", "language": "py"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dataset(path)


class EvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = FakeAnalyzer(
            {
                "code-a": _review("The query is vulnerable to SQL injection.", "Use a parameterized query."),
                "code-b": _review("Variable names are unclear.", "Add a docstring."),
                "code-c": _review("strcpy causes a buffer overflow.", "Add logging.", source="rule-fallback"),
            }
        )

    def test_hit_rates_categories_and_sources(self) -> None:
        results = evaluate_trx_ai(self.analyzer, CASES, run_baseline=False, save_report=False, verbose=False)

        self.assertEqual(results["total_cases"], 3)
        self.assertEqual(results["debug_detection"]["hits"], 2)
        self.assertEqual(results["fix_suggestion"]["hits"], 1)
        self.assertAlmostEqual(results["accuracy_score"], 66.67, places=2)
        self.assertEqual(results["llm_served_cases"], 2)
        self.assertEqual(results["fallback_served_cases"], 1)
        self.assertEqual(results["by_category"]["security"]["debug_detection"]["hits"], 2)
        self.assertEqual(results["by_category"]["bug"]["debug_detection"]["hits"], 0)
        self.assertEqual(results["provenance"]["model"], "fake-model")
        self.assertEqual(results["provenance"]["dataset_size"], 3)
        low, high = results["debug_detection"]["ci95"]
        self.assertLess(low, 66.67)
        self.assertGreater(high, 66.67)

    def test_review_cache_cleared_before_each_case(self) -> None:
        evaluate_trx_ai(self.analyzer, CASES, run_baseline=False, save_report=False, verbose=False)
        self.assertEqual(self.analyzer.cache_sizes_seen, [0, 0, 0])

    def test_report_files_written_with_fallback_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "eval.txt"
            results = evaluate_trx_ai(
                self.analyzer, CASES, run_baseline=False, report_path=text_path, verbose=False
            )
            text = text_path.read_text(encoding="utf-8")
            payload = json.loads(text_path.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertIn("Debug detection: 66.7% (2/3, 95% CI", text)
        self.assertIn("WARNING: some cases were served by the rule fallback", text)
        self.assertEqual(payload["debug_detection"], results["debug_detection"])
        self.assertEqual(len(payload["cases"]), 3)

    def test_baseline_comparison_counts(self) -> None:
        baseline_texts = {
            "code-a": "Nothing obviously wrong.",
            "code-b": "Mutable default argument: use None default instead.",
            "code-c": "strcpy causes a buffer overflow.",
        }

        def fake_baseline(_analyzer, code, _language):
            return {"ok": True, "text": baseline_texts[code]}

        with patch.object(evaluation, "_baseline_simple_code_review", side_effect=fake_baseline):
            results = evaluate_trx_ai(self.analyzer, CASES, save_report=False, verbose=False)

        baseline = results["baseline"]
        self.assertEqual(baseline["debug_detection"]["hits"], 2)
        self.assertEqual(baseline["trx_wins"], 1)
        self.assertEqual(baseline["trx_losses"], 1)
        self.assertEqual(baseline["ties"], 1)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_trx_ai(self.analyzer, [], save_report=False, verbose=False)
        with self.assertRaises(ValueError):
            evaluate_trx_ai(self.analyzer, CASES, threshold=0, save_report=False, verbose=False)


class EvalCliTests(unittest.TestCase):
    def test_filters_with_no_matches_return_error(self) -> None:
        with patch("builtins.print"):
            self.assertEqual(evaluation.main(["--category", "does-not-exist"]), 1)

    def test_cli_runs_filtered_subset(self) -> None:
        captured = {}

        def fake_eval(analyzer, cases, **kwargs):
            captured["cases"] = cases
            captured["kwargs"] = kwargs
            return {}

        with patch.object(evaluation, "RealityAnalyzer"), patch.object(evaluation, "evaluate_trx_ai", fake_eval):
            code = evaluation.main(["--category", "security", "--language", "python", "--limit", "2", "--no-baseline"])

        self.assertEqual(code, 0)
        self.assertEqual(len(captured["cases"]), 2)
        self.assertTrue(all(c.category == "security" and c.language == "python" for c in captured["cases"]))
        self.assertFalse(captured["kwargs"]["run_baseline"])


if __name__ == "__main__":
    unittest.main()
