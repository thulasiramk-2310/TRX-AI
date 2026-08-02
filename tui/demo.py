"""Sample data for screenshots and `python -m tui.app --demo` (no LLM or cluster needed)."""

from __future__ import annotations

from tui.engine import SystemStatus

DEMO_STATUS = SystemStatus(model="qwen3:8b", llm_enabled=True, llm_reachable=True, compute="GPU",
                           graph="ready", mode="auto")

_ORIGINAL = "def find_max(arr):\n    return arr[0]\n"
_FIXED = "def find_max(arr):\n    if not arr:\n        raise ValueError(\"empty list\")\n    return max(arr)\n"


def demo_results() -> list[tuple[str, dict]]:
    review = {
        "response_mode": "analysis",
        "current_input": "review examples/dsa_test.py",
        "model_name": "qwen3:8b",
        "elapsed_seconds": 6.4,
        "analysis_source": "llm",
        "confidence_score": 86,
        "debug_analysis": [
            "Line 2: `find_max` returns `arr[0]` on an empty list (IndexError).",
            "Line 7: inner loop reads `arr[j+1]` past the end of the list.",
        ],
        "improvements": [
            "Guard empty input: `raise ValueError(\"empty list\")`.",
            "Use `range(n - i - 1)` in the inner loop of bubble sort.",
        ],
        "predictions": ["`fib` is O(2^n); memoize or iterate for O(n)."],
        "final_insight": ["Fix the out-of-bounds read first; it crashes on every non-empty input."],
        "original_code": _ORIGINAL,
        "fixed_code": _FIXED,
        "system_status": ["intent=review", "source=llm", "critic_score=0.92", "steps=3"],
        "guardrails": [],
    }
    chat = {
        "response_mode": "chat",
        "model_name": "qwen3:8b",
        "elapsed_seconds": 2.1,
        "analysis_source": "llm",
        "intent_confidence": 0.93,
        "chat_response": "Each pass compares neighbours across the whole list and it needs up to *n* passes, "
                         "so the work grows as **n × n**. Stopping early when a pass makes no swaps helps on "
                         "nearly sorted input, but the worst case stays O(n²).",
    }
    return [("review examples/dsa_test.py", review), ("why is bubble sort O(n²)?", chat)]
