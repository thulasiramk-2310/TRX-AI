"""Question-driven smoke test for TRX with mixed general/code prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path

from analyzer import RealityAnalyzer
from config import AppConfig


def run() -> dict:
    cfg = AppConfig.from_env()
    # Deterministic local smoke behavior.
    cfg.use_local_llm = False
    cfg.assistant_mode = "auto"
    analyzer = RealityAnalyzer(cfg)

    cases = [
        ("hi", "chat"),
        ("bye", "chat"),
        ("what is data warehouse", "chat"),
        ("what u can do", "chat"),
        ("ok", "chat"),
        ("fix login bug in auth.py", "analysis"),
        ("improve API performance", "analysis"),
        ("make login faster", "analysis"),
    ]

    results = []
    passed = 0
    started = time.time()
    for text, expected_mode in cases:
        out = analyzer.analyze(text, mode="debug", past_context=[])
        actual_mode = str(out.get("response_mode", "")).lower()
        ok = actual_mode == expected_mode
        passed += 1 if ok else 0
        results.append(
            {
                "input": text,
                "expected_mode": expected_mode,
                "actual_mode": actual_mode,
                "pass": ok,
                "intent": out.get("intent"),
                "cache_hit": bool(out.get("cache_hit", False)),
                "mcp_query_status": out.get("mcp_query_status"),
                "circuit_breaker_state": out.get("circuit_breaker_state"),
            }
        )

    total = len(cases)
    report = {
        "generated_at_epoch_ms": int(time.time() * 1000),
        "duration_ms": int((time.time() - started) * 1000),
        "passed": passed,
        "total": total,
        "score_out_of_10": round((passed / total) * 10, 2) if total else 0.0,
        "results": results,
    }
    return report


if __name__ == "__main__":
    report = run()
    out = Path("sessions") / "smoke_questions_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "score_out_of_10": report["score_out_of_10"]}, ensure_ascii=False))
