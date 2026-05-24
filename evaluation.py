"""TRX-AI evaluation and benchmarking module.

Runs the labelled benchmark in ``benchmarks/cases.jsonl`` through the code-review
pipeline and reports hit rates with 95% Wilson confidence intervals, per-category
breakdowns, and full provenance (dataset hash, model, git commit) so every
headline number can be traced back to a concrete run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyzer import RealityAnalyzer, call_local_llm
from config import AppConfig
from semantic_scoring import SemanticMatcher

DATASET_PATH = Path(__file__).resolve().parent / "benchmarks" / "cases.jsonl"
DEFAULT_REPORT_DIR = Path("reports") / "eval"
DEFAULT_HIT_THRESHOLD = 0.5

# Case-agnostic review used to measure the chance floor: how often a generic answer
# that ignores the code still "hits" the expected labels at the chosen threshold.
GENERIC_REVIEW = (
    "Line 1: The function lacks input validation and error handling. Consider adding type hints "
    "and a docstring. Improvements: add unit tests, check for edge cases such as empty or None input, "
    "handle exceptions, use descriptive variable names. Performance: the code runs in linear time; "
    "no major issues. Security: no obvious vulnerabilities, but validate untrusted input. Fix: add a "
    "guard clause at the start of the function and return early when input is invalid."
).lower()


@dataclass(frozen=True)
class EvalCase:
    """Single benchmark case for code-review quality checks."""

    input_text: str
    expected_debug: str
    expected_fix: str
    category: str
    language: str
    id: str = ""


def load_dataset(path: str | Path = DATASET_PATH) -> list[EvalCase]:
    """Loads benchmark cases from a JSONL file (one case per line)."""
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get("id") or f"line-{line_no}")
            if case_id in seen_ids:
                raise ValueError(f"Duplicate case id {case_id!r} at line {line_no}.")
            seen_ids.add(case_id)
            cases.append(
                EvalCase(
                    input_text=row["code"],
                    expected_debug=row["expected_debug"],
                    expected_fix=row["expected_fix"],
                    category=row["category"],
                    language=row["language"],
                    id=case_id,
                )
            )
    return cases


EVAL_DATASET: list[EvalCase] = load_dataset() if DATASET_PATH.exists() else []

SEMANTIC_MATCHER = SemanticMatcher()


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Returns the Wilson score interval for a binomial proportion, as fractions."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


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


def _result_source(result: dict[str, Any]) -> str:
    statuses = {str(item).lower() for item in result.get("system_status", [])}
    if "source=llm" in statuses:
        return "llm"
    if "source=rule-fallback" in statuses or "llm_unavailable" in statuses:
        return "rule-fallback"
    return "unknown"


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


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
        return f"{commit}-dirty" if commit and dirty else commit or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dataset_fingerprint(cases: list[EvalCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(
            json.dumps(
                [case.id, case.input_text, case.expected_debug, case.expected_fix, case.category, case.language],
                ensure_ascii=False,
            ).encode("utf-8")
        )
    return digest.hexdigest()[:16]


def _rate_block(hits: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, total)
    return {
        "hits": hits,
        "total": total,
        "rate": round(100 * hits / total, 2) if total else 0.0,
        "ci95": [round(100 * low, 2), round(100 * high, 2)],
    }


def _clear_review_cache(analyzer: Any) -> None:
    cache = getattr(analyzer, "_review_response_cache", None)
    if hasattr(cache, "clear"):
        cache.clear()


def evaluate_trx_ai(
    analyzer: RealityAnalyzer,
    dataset: list[EvalCase] | None = None,
    *,
    threshold: float = DEFAULT_HIT_THRESHOLD,
    run_baseline: bool = True,
    save_report: bool = True,
    report_path: str | Path | None = None,
    json_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluates TRX-AI on benchmark cases and returns aggregate metrics.

    A case counts as a debug *hit* when the semantic similarity between the expected
    issue label and the review output reaches ``threshold``; fix hits work the same way.
    The review cache is cleared before every case so latency and quality reflect real work.
    """
    cases = dataset if dataset is not None else EVAL_DATASET
    if not cases:
        raise ValueError("Evaluation dataset is empty.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")

    total = len(cases)
    case_results: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        _clear_review_cache(analyzer)
        started = time.perf_counter()
        trx_result = analyzer.analyze_code_multi_agent(case.input_text)
        trx_elapsed = time.perf_counter() - started

        trx_text = _as_search_text(trx_result)
        debug_sim = _semantic_match(case.expected_debug, trx_text)
        fix_sim = _semantic_match(case.expected_fix, trx_text)

        row: dict[str, Any] = {
            "id": case.id or f"case-{idx:03d}",
            "language": case.language,
            "category": case.category,
            "source": _result_source(trx_result),
            "debug_similarity": round(debug_sim, 4),
            "fix_similarity": round(fix_sim, 4),
            "debug_hit": debug_sim >= threshold,
            "fix_hit": fix_sim >= threshold,
            "completeness": round(_section_completeness(trx_result), 4),
            "latency_seconds": round(trx_elapsed, 3),
            "chance_debug_hit": _semantic_match(case.expected_debug, GENERIC_REVIEW) >= threshold,
            "chance_fix_hit": _semantic_match(case.expected_fix, GENERIC_REVIEW) >= threshold,
        }

        if run_baseline:
            baseline_started = time.perf_counter()
            baseline = _baseline_simple_code_review(analyzer, case.input_text, case.language)
            row["baseline_latency_seconds"] = round(time.perf_counter() - baseline_started, 3)
            baseline_text = str(baseline.get("text", "")).lower()
            row["baseline_ok"] = bool(baseline.get("ok"))
            b_debug = _semantic_match(case.expected_debug, baseline_text)
            b_fix = _semantic_match(case.expected_fix, baseline_text)
            row["baseline_debug_hit"] = b_debug >= threshold
            row["baseline_fix_hit"] = b_fix >= threshold
            diff = (debug_sim + fix_sim) - (b_debug + b_fix)
            row["vs_baseline"] = "win" if diff > 0.05 else "loss" if diff < -0.05 else "tie"

        case_results.append(row)
        if verbose:
            print(
                f"[{idx}/{total}] {row['id']} {case.language}/{case.category} "
                f"debug={'HIT' if row['debug_hit'] else 'miss'} fix={'HIT' if row['fix_hit'] else 'miss'} "
                f"src={row['source']} {trx_elapsed:.2f}s"
            )

    results = summarize(case_results, threshold=threshold, run_baseline=run_baseline)
    results["provenance"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": getattr(analyzer.config, "local_llm_model", "unknown"),
        "dataset_size": total,
        "dataset_fingerprint": _dataset_fingerprint(list(cases)),
        "hit_threshold": threshold,
    }
    results["cases"] = case_results

    if verbose:
        print()
        print(render_text_report(results))

    if save_report:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_target = Path(report_path) if report_path else DEFAULT_REPORT_DIR / f"eval_{stamp}.txt"
        json_target = Path(json_path) if json_path else text_target.with_suffix(".json")
        text_target.parent.mkdir(parents=True, exist_ok=True)
        json_target.parent.mkdir(parents=True, exist_ok=True)
        text_target.write_text(render_text_report(results) + "\n", encoding="utf-8")
        json_target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results["report_paths"] = {"text": str(text_target), "json": str(json_target)}
        if verbose:
            print(f"- Report saved: {text_target} / {json_target}")

    return results


def summarize(case_results: list[dict[str, Any]], *, threshold: float, run_baseline: bool) -> dict[str, Any]:
    """Aggregates per-case rows into overall and per-category metrics."""
    total = len(case_results)
    debug_hits = sum(1 for r in case_results if r["debug_hit"])
    fix_hits = sum(1 for r in case_results if r["fix_hit"])
    latencies = sorted(r["latency_seconds"] for r in case_results)
    llm_served = sum(1 for r in case_results if r["source"] == "llm")

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_results:
        by_category[row["category"]].append(row)

    results: dict[str, Any] = {
        "total_cases": total,
        "debug_detection": _rate_block(debug_hits, total),
        "fix_suggestion": _rate_block(fix_hits, total),
        "chance_floor": {
            "debug_detection": _rate_block(sum(1 for r in case_results if r.get("chance_debug_hit")), total),
            "fix_suggestion": _rate_block(sum(1 for r in case_results if r.get("chance_fix_hit")), total),
        },
        "mean_debug_similarity": round(sum(r["debug_similarity"] for r in case_results) / total, 4),
        "mean_fix_similarity": round(sum(r["fix_similarity"] for r in case_results) / total, 4),
        "completeness_score": round(100 * sum(r["completeness"] for r in case_results) / total, 2),
        "avg_response_time_seconds": round(sum(latencies) / total, 3),
        "p95_response_time_seconds": latencies[min(total - 1, math.ceil(0.95 * total) - 1)],
        "llm_served_cases": llm_served,
        "fallback_served_cases": total - llm_served,
        "by_category": {
            name: {
                "debug_detection": _rate_block(sum(1 for r in rows if r["debug_hit"]), len(rows)),
                "fix_suggestion": _rate_block(sum(1 for r in rows if r["fix_hit"]), len(rows)),
            }
            for name, rows in sorted(by_category.items())
        },
    }
    # Backwards-compatible keys used by older reports/scripts.
    results["accuracy_score"] = results["debug_detection"]["rate"]
    results["fix_quality_score"] = results["fix_suggestion"]["rate"]

    if run_baseline:
        outcomes = [r.get("vs_baseline", "tie") for r in case_results]
        results["baseline"] = {
            "debug_detection": _rate_block(sum(1 for r in case_results if r.get("baseline_debug_hit")), total),
            "fix_suggestion": _rate_block(sum(1 for r in case_results if r.get("baseline_fix_hit")), total),
            "avg_response_time_seconds": round(
                sum(r.get("baseline_latency_seconds", 0.0) for r in case_results) / total, 3
            ),
            "available_cases": sum(1 for r in case_results if r.get("baseline_ok")),
            "trx_wins": outcomes.count("win"),
            "ties": outcomes.count("tie"),
            "trx_losses": outcomes.count("loss"),
        }
    return results


def _fmt_rate(block: dict[str, Any]) -> str:
    low, high = block["ci95"]
    return f"{block['rate']:.1f}% ({block['hits']}/{block['total']}, 95% CI {low:.1f}-{high:.1f}%)"


def render_text_report(results: dict[str, Any]) -> str:
    """Renders a human-readable report. Every rate carries its n and confidence interval."""
    lines = [
        "TRX-AI Evaluation Report",
        "========================",
        f"Cases: {results['total_cases']}",
        f"Debug detection: {_fmt_rate(results['debug_detection'])}",
        f"Fix suggestion:  {_fmt_rate(results['fix_suggestion'])}",
        f"Chance floor (generic review): debug={results['chance_floor']['debug_detection']['rate']:.1f}%"
        f" fix={results['chance_floor']['fix_suggestion']['rate']:.1f}%",
        f"Mean similarity: debug={results['mean_debug_similarity']:.3f} fix={results['mean_fix_similarity']:.3f}",
        f"Completeness: {results['completeness_score']:.1f}%",
        f"Latency: avg={results['avg_response_time_seconds']:.2f}s p95={results['p95_response_time_seconds']:.2f}s",
        f"Served by LLM: {results['llm_served_cases']}/{results['total_cases']}"
        f" (rule fallback: {results['fallback_served_cases']})",
    ]
    prov = results.get("provenance")
    if prov:
        lines.append(
            f"Provenance: model={prov['model']} commit={prov['git_commit']} "
            f"dataset={prov['dataset_fingerprint']} threshold={prov['hit_threshold']} at {prov['timestamp']}"
        )
    if results["fallback_served_cases"]:
        lines.append("WARNING: some cases were served by the rule fallback; numbers do not reflect LLM quality.")

    lines += ["", "By category:"]
    for name, block in results["by_category"].items():
        lines.append(f"  {name:<12} debug {_fmt_rate(block['debug_detection'])}")
        lines.append(f"  {'':<12} fix   {_fmt_rate(block['fix_suggestion'])}")

    baseline = results.get("baseline")
    if baseline:
        lines += [
            "",
            "Baseline (single plain prompt, same model):",
            f"  Debug detection: {_fmt_rate(baseline['debug_detection'])}",
            f"  Fix suggestion:  {_fmt_rate(baseline['fix_suggestion'])}",
            f"  Avg latency: {baseline['avg_response_time_seconds']:.2f}s"
            f" (available on {baseline['available_cases']}/{results['total_cases']} cases)",
            f"  TRX vs baseline: {baseline['trx_wins']} wins / {baseline['ties']} ties / {baseline['trx_losses']} losses",
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trx eval", description="Run the TRX-AI code-review benchmark.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Path to a JSONL benchmark file.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_HIT_THRESHOLD, help="Similarity needed for a hit.")
    parser.add_argument("--category", action="append", help="Only run cases in this category (repeatable).")
    parser.add_argument("--language", action="append", help="Only run cases in this language (repeatable).")
    parser.add_argument("--limit", type=int, default=0, help="Run at most N cases.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the plain-prompt baseline.")
    parser.add_argument("--out", default=None, help="Text report path (JSON is written alongside).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_dataset(args.dataset)
    if args.category:
        cases = [c for c in cases if c.category in set(args.category)]
    if args.language:
        cases = [c for c in cases if c.language in set(args.language)]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        print("No cases match the given filters.")
        return 1
    analyzer = RealityAnalyzer(AppConfig.from_env())
    evaluate_trx_ai(
        analyzer,
        cases,
        threshold=args.threshold,
        run_baseline=not args.no_baseline,
        report_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
