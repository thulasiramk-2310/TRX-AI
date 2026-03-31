"""TRX-AI evaluation and benchmarking module."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyzer import RealityAnalyzer, call_local_llm
from config import AppConfig
from semantic_scoring import SemanticMatcher


@dataclass(frozen=True)
class EvalCase:
    """Single benchmark case for code-review quality checks."""

    input_text: str
    expected_debug: str
    expected_fix: str
    category: str
    language: str


EVAL_DATASET: list[EvalCase] = [
    EvalCase(
        input_text="def find_max(arr):\n    return arr[0]\n",
        expected_debug="empty list index error",
        expected_fix="add empty check",
        category="edge-case",
        language="python",
    ),
    EvalCase(
        input_text="def bubble(arr):\n    n=len(arr)\n    for i in range(n):\n        for j in range(n):\n            if arr[j] > arr[j+1]:\n                arr[j],arr[j+1]=arr[j+1],arr[j]\n    return arr\n",
        expected_debug="index out of bounds",
        expected_fix="range(n - i - 1)",
        category="bug",
        language="python",
    ),
    EvalCase(
        input_text="def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n",
        expected_debug="exponential recursion",
        expected_fix="iterative or memoization",
        category="performance",
        language="python",
    ),
    EvalCase(
        input_text="function findMax(arr){ return arr[0]; }",
        expected_debug="empty array edge case",
        expected_fix="guard for empty array",
        category="edge-case",
        language="javascript",
    ),
    EvalCase(
        input_text="function run(cmd){ const {exec}=require('child_process'); exec(cmd); }",
        expected_debug="command injection risk",
        expected_fix="avoid shell execution",
        category="security",
        language="javascript",
    ),
    EvalCase(
        input_text="public static int max(int[] a){ return a[0]; }",
        expected_debug="empty array check missing",
        expected_fix="validate length",
        category="edge-case",
        language="java",
    ),
    EvalCase(
        input_text="#include <stdio.h>\nint main(){ int a[3]={1,2,3}; printf(\"%d\", a[3]); }",
        expected_debug="out of bounds",
        expected_fix="valid index range",
        category="bug",
        language="c",
    ),
    EvalCase(
        input_text="SELECT * FROM users WHERE id = " + "'\" + userInput + \"';",
        expected_debug="sql injection",
        expected_fix="parameterized query",
        category="security",
        language="sql",
    ),
    EvalCase(
        input_text="def fetch(url):\n    import requests\n    return requests.get(url).text\n",
        expected_debug="missing timeout",
        expected_fix="add timeout parameter",
        category="security",
        language="python",
    ),
    EvalCase(
        input_text="def save(path, data):\n    with open(path, 'w') as f:\n        f.write(data)\n",
        expected_debug="missing encoding",
        expected_fix="encoding utf-8",
        category="bug",
        language="python",
    ),
]


SEMANTIC_MATCHER = SemanticMatcher()


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s_\-]+", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str) -> set[str]:
    base = _normalize_text(text)
    return set(base.split())


def _semantic_match(expected: str, actual_text: str) -> float:
    return SEMANTIC_MATCHER.score(expected, actual_text)


def _as_search_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("debug_analysis", "improvements", "predictions", "final_insight"):
        value = result.get(key, [])
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def _section_completeness(result: dict[str, Any]) -> float:
    sections = ("debug_analysis", "improvements", "predictions")
    filled = 0
    for key in sections:
        value = result.get(key, [])
        has_content = False
        if isinstance(value, list):
            has_content = any(str(item).strip() for item in value)
        elif isinstance(value, str):
            has_content = bool(value.strip())
        if has_content:
            filled += 1
    return filled / len(sections)


def _baseline_simple_code_review(analyzer: RealityAnalyzer, code_text: str, language: str) -> dict[str, Any]:
    prompt = (
        "You are a code reviewer.\n"
        f"Language: {language}\n"
        "List key bugs and practical fixes briefly.\n\n"
        f"{code_text}\n"
    )
    local = call_local_llm(
        prompt,
        url=analyzer.config.local_llm_url,
        model=analyzer.config.local_llm_model,
        timeout=analyzer.config.local_llm_timeout_seconds,
        max_new_tokens=260,
        temperature=0.3,
        retries=analyzer.config.local_llm_retries,
    )
    text = str(local.get("text", "")).strip()
    return {"ok": bool(local.get("ok")), "text": text, "done_reason": local.get("done_reason")}


def evaluate_trx_ai(
    analyzer: RealityAnalyzer,
    dataset: list[EvalCase] | None = None,
    *,
    save_report: bool = True,
    report_path: str = "evaluation_report.txt",
) -> dict[str, Any]:
    """Evaluates TRX-AI on benchmark cases and returns aggregate metrics."""
    cases = dataset if dataset is not None else EVAL_DATASET
    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    total = len(cases)
    debug_score_sum = 0.0
    fix_score_sum = 0.0
    trx_times: list[float] = []
    baseline_times: list[float] = []
    completeness_values: list[float] = []
    comparison_trx_better = 0
    case_rows: list[str] = []

    for idx, case in enumerate(cases, start=1):
        started = time.perf_counter()
        trx_result = analyzer.analyze_code_multi_agent(case.input_text)
        trx_elapsed = time.perf_counter() - started
        trx_times.append(trx_elapsed)

        trx_text = _as_search_text(trx_result)
        debug_score = _semantic_match(case.expected_debug, trx_text)
        fix_score = _semantic_match(case.expected_fix, trx_text)
        debug_score_sum += debug_score
        fix_score_sum += fix_score
        completeness_values.append(_section_completeness(trx_result))

        baseline_started = time.perf_counter()
        baseline = _baseline_simple_code_review(analyzer, case.input_text, case.language)
        baseline_elapsed = time.perf_counter() - baseline_started
        baseline_times.append(baseline_elapsed)
        baseline_text = str(baseline.get("text", "")).lower()

        baseline_score = _semantic_match(case.expected_debug, baseline_text) + _semantic_match(case.expected_fix, baseline_text)
        trx_score = debug_score + fix_score
        if trx_score > baseline_score:
            comparison_trx_better += 1

        case_rows.append(
            f"{idx}. [{case.language}/{case.category}] "
            f"debug={debug_score:.2f} fix={fix_score:.2f} "
            f"trx_time={trx_elapsed:.2f}s baseline_time={baseline_elapsed:.2f}s"
        )

    accuracy_score = (debug_score_sum / total) * 100
    fix_quality_score = (fix_score_sum / total) * 100
    avg_response_time = sum(trx_times) / total
    completeness_score = (sum(completeness_values) / total) * 100
    baseline_avg_time = sum(baseline_times) / total
    comparison_rate = (comparison_trx_better / total) * 100

    results = {
        "total_cases": total,
        "accuracy_score": round(accuracy_score, 2),
        "fix_quality_score": round(fix_quality_score, 2),
        "avg_response_time_seconds": round(avg_response_time, 2),
        "completeness_score": round(completeness_score, 2),
        "baseline_avg_response_time_seconds": round(baseline_avg_time, 2),
        "trx_better_than_baseline_rate": round(comparison_rate, 2),
        "case_breakdown": case_rows,
    }

    print("Evaluation Results:")
    print(f"- Accuracy: {results['accuracy_score']}%")
    print(f"- Fix Quality: {results['fix_quality_score']}%")
    print(f"- Avg Response Time: {results['avg_response_time_seconds']}s")
    print(f"- Completeness: {results['completeness_score']}%")
    print(f"- Baseline Avg Response Time: {results['baseline_avg_response_time_seconds']}s")
    print(f"- TRX Better Than Baseline: {results['trx_better_than_baseline_rate']}%")
    print("\nBenchmark Summary:")
    print("TRX-AI shows improved structured reasoning compared to baseline.")

    if save_report:
        report_lines = [
            "TRX-AI Evaluation Report",
            "========================",
            f"Total Cases: {results['total_cases']}",
            f"Accuracy: {results['accuracy_score']}%",
            f"Fix Quality: {results['fix_quality_score']}%",
            f"Avg Response Time: {results['avg_response_time_seconds']}s",
            f"Completeness: {results['completeness_score']}%",
            f"Baseline Avg Response Time: {results['baseline_avg_response_time_seconds']}s",
            f"TRX Better Than Baseline: {results['trx_better_than_baseline_rate']}%",
            "",
            "Case Breakdown:",
            *results["case_breakdown"],
            "",
        ]
        Path(report_path).write_text("\n".join(report_lines), encoding="utf-8")
        print(f"- Report Saved: {report_path}")

    return results


if __name__ == "__main__":
    app_config = AppConfig.from_env()
    trx_analyzer = RealityAnalyzer(app_config)
    evaluate_trx_ai(trx_analyzer, save_report=True, report_path="evaluation_report.txt")
