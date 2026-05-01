"""Consolidated TRX smoke runner with machine-readable JSON report."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from analyzer import RealityAnalyzer
from config import AppConfig


def _check(name: str, category: str, ok: bool, details: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "pass": bool(ok),
        "details": details,
    }


def run_smoke(*, disable_llm: bool = False) -> dict[str, Any]:
    cfg = AppConfig.from_env()
    cfg.assistant_mode = "auto"
    cfg.debug_cache = True
    cfg.cache_ttl_seconds = max(10, int(cfg.cache_ttl_seconds))
    if disable_llm:
        cfg.use_local_llm = False
    analyzer = RealityAnalyzer(cfg)

    matrix: list[dict[str, Any]] = []
    started = time.time()

    # Routing and semantic checks.
    r1 = analyzer.analyze("hi", mode="debug", past_context=[])
    matrix.append(_check("greeting_general", "routing", str(r1.get("response_mode")) == "chat", str(r1.get("intent", ""))))

    r2 = analyzer.analyze("fix login bug in auth.py", mode="debug", past_context=[])
    matrix.append(_check("code_bug_route", "routing", str(r2.get("response_mode")) == "analysis", str(r2.get("intent", ""))))

    r3 = analyzer.analyze("improve API performance", mode="debug", past_context=[])
    matrix.append(_check("semantic_tech_route", "semantic", str(r3.get("response_mode")) == "analysis", str(r3.get("intent", ""))))

    # UX guardrails.
    r4 = analyzer.analyze("what is data warehouse", mode="debug", past_context=[])
    chat = str(r4.get("chat_response", "")).lower()
    matrix.append(_check("no_meta_response", "ux", "the user asked" not in chat and "the user is asking" not in chat, chat[:120]))

    # Circuit breaker exposure.
    matrix.append(
        _check(
            "breaker_in_output",
            "observability",
            "circuit_breaker_state" in r4 and isinstance(r4.get("system_health"), dict),
            str(r4.get("circuit_breaker_state", "")),
        )
    )
    status = analyzer.runtime_status()
    matrix.append(
        _check(
            "breaker_in_runtime_status",
            "observability",
            "circuit_breaker_state" in status,
            str(status.get("circuit_breaker_state", "")),
        )
    )

    # Cache hit semantics.
    q = "what is network"
    c1 = analyzer.analyze(q, mode="debug", past_context=[])
    c2 = analyzer.analyze(q, mode="debug", past_context=[])
    matrix.append(_check("cache_hit_second_run", "cache", bool(c2.get("cache_hit")), str(c2.get("cache_debug", {}))))
    matrix.append(_check("cache_first_miss", "cache", not bool(c1.get("cache_hit")), "first run should miss"))

    passed = sum(1 for item in matrix if item["pass"])
    total = len(matrix)
    score = round((passed / total) * 10, 2) if total else 0.0
    duration_ms = int((time.time() - started) * 1000)

    by_category: dict[str, dict[str, int]] = {}
    for item in matrix:
        cat = str(item["category"])
        by_category.setdefault(cat, {"pass": 0, "total": 0})
        by_category[cat]["total"] += 1
        by_category[cat]["pass"] += 1 if item["pass"] else 0

    return {
        "generated_at_epoch_ms": int(time.time() * 1000),
        "duration_ms": duration_ms,
        "score_out_of_10": score,
        "passed": passed,
        "total": total,
        "category_summary": by_category,
        "matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRX consolidated smoke e2e")
    parser.add_argument("--json-out", default="sessions/smoke_e2e_report.json")
    parser.add_argument("--disable-llm", action="store_true")
    args = parser.parse_args()

    report = run_smoke(disable_llm=args.disable_llm)
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "score_out_of_10": report["score_out_of_10"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

