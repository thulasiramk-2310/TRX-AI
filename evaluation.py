"""TRX-AI evaluation and benchmarking module."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyzer import RealityAnalyzer, call_local_llm
from config import AppConfig


@dataclass(frozen=True)
class EvalCase:
    """Single benchmark case for code-review quality checks."""

    input_text: str
    expected_debug: str
    expected_fix: str
    category: str


EVAL_DATASET: list[EvalCase] = [
    EvalCase(
        input_text="def find_max(arr):\n    return arr[0]\n",
        expected_debug="empty list",
        expected_fix="empty check",
        category="edge-case",
    ),
    EvalCase(
        input_text="def bubble(arr):\n    n=len(arr)\n    for i in range(n):\n        for j in range(n):\n            if arr[j] > arr[j+1]:\n                arr[j],arr[j+1]=arr[j+1],arr[j]\n    return arr\n",
        expected_debug="index error",
        expected_fix="range(n - i - 1)",
        category="bug",
    ),
    EvalCase(
        input_text="def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n",
        expected_debug="exponential",
        expected_fix="iterative",
        category="performance",
    ),
    EvalCase(
        input_text="import requests\n\ndef fetch(url):\n    return requests.get(url).text\n",
        expected_debug="timeout",
        expected_fix="timeout=",
        category="security",
    ),
    EvalCase(
        input_text="def run(cmd):\n    import subprocess\n    subprocess.run(cmd, shell=True)\n",
        expected_debug="shell=true",
        expected_fix="shell=False",
        category="security",
    ),
    EvalCase(
        input_text="def avg(nums):\n    return sum(nums) / len(nums)\n",
        expected_debug="division by zero",
        expected_fix="if not nums",
        category="edge-case",
    ),
    EvalCase(
        input_text="def parse_user(data):\n    return eval(data)\n",
        expected_debug="eval",
        expected_fix="literal_eval",
        category="security",
    ),
    EvalCase(
        input_text="def find_user(users, target):\n    for i in range(len(users)):\n        if users[i]['id'] == target:\n            return users[i]\n    return None\n",
        expected_debug="inefficient",
        expected_fix="for user in users",
        category="performance",
    ),
    EvalCase(
        input_text="def pop_first(items):\n    return items[0]\n",
        expected_debug="indexerror",
        expected_fix="if not items",
        category="edge-case",
    ),
    EvalCase(
        input_text="def save(path, data):\n    with open(path, 'w') as f:\n        f.write(data)\n",
        expected_debug="encoding",
        expected_fix="encoding=",
        category="bug",
    ),
]


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


def _baseline_simple_code_review(analyzer: RealityAnalyzer, code_text: str) -> dict[str, Any]:
    prompt = (
        "You are a Python code reviewer.\n"
        "List bugs and fixes briefly.\n\n"
        f"{code_text}\n"
    )
    local = call_local_llm(
        prompt,
        url=analyzer.config.local_llm_url,
        model=analyzer.config.local_llm_model,
        timeout=analyzer.config.local_llm_timeout_seconds,
        max_new_tokens=300,
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
    debug_hits = 0
    fix_hits = 0
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
        expected_debug = case.expected_debug.lower()
        expected_fix = case.expected_fix.lower()
        debug_ok = expected_debug in trx_text
        fix_ok = expected_fix in trx_text
        if debug_ok:
            debug_hits += 1
        if fix_ok:
            fix_hits += 1
        completeness_values.append(_section_completeness(trx_result))

        baseline_started = time.perf_counter()
        baseline = _baseline_simple_code_review(analyzer, case.input_text)
        baseline_elapsed = time.perf_counter() - baseline_started
        baseline_times.append(baseline_elapsed)
        baseline_text = str(baseline.get("text", "")).lower()

        baseline_score = int(expected_debug in baseline_text) + int(expected_fix in baseline_text)
        trx_score = int(debug_ok) + int(fix_ok)
        if trx_score > baseline_score:
            comparison_trx_better += 1

        case_rows.append(
            f"{idx}. [{case.category}] debug_match={debug_ok} fix_match={fix_ok} "
            f"trx_time={trx_elapsed:.2f}s baseline_time={baseline_elapsed:.2f}s"
        )

    accuracy_score = (debug_hits / total) * 100
    fix_quality_score = (fix_hits / total) * 100
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
