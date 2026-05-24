"""Rule-based intent signals used by the TRX-AI router."""

from __future__ import annotations

import re
from typing import Any


TRX_GREETING = "Hi there! 😊 What can I help you with today?"
BASIC_KB = {
    "data warehouse": "A data warehouse is a centralized system used to store and analyze large structured datasets for reporting and business intelligence.",
    "network": "A network is a system of interconnected computers and devices that communicate and share resources.",
}
STRONG_CODE_HINTS = ["fix", "bug", "error", "exception", "stacktrace"]
TECH_HINTS = [
    "optimize", "performance", "latency",
    "api", "system", "database", "backend",
    "faster", "slow", "throughput", "login",
]


def classify_intent(text: str) -> str:
    text = text.lower().strip()

    if text.startswith(("what is", "define", "explain")):
        return "theory"

    if any(k in text for k in ["review", "fix", "code", "bug"]):
        return "code"

    if any(k in text for k in ["i failed", "improve", "problem"]):
        return "problem"

    return "chat"


def is_general_question(text: str) -> bool:
    keywords = ["what is", "explain", "define", "how does", "why"]
    normalized = text.lower().strip()
    return any(normalized.startswith(keyword) for keyword in keywords)


def detect_intent(text: str) -> str:
    text_l = text.lower().strip()

    code_keywords = [
        "fix", "bug", "error", "exception",
        "function", "class", "file", "repo",
        "refactor", "optimize", "review",
    ]
    general_patterns = [
        "what is", "explain", "define", "why", "how",
    ]

    if any(keyword in text_l for keyword in code_keywords):
        return "code"
    if has_tech_context(text_l):
        return "code"
    if any(text_l.startswith(pattern) for pattern in general_patterns):
        return "general"
    return "general"


def detect_intent_confidence(text: str) -> tuple[str, float]:
    normalized = text.lower().strip()
    if normalized in {"hi", "hello", "hey"}:
        return "general", 1.0

    code_patterns = [
        r"\bfix\b", r"\bbug\b", r"\berror\b", r"\bexception\b",
        r"\bfunction\b", r"\bclass\b", r"\bfile\b", r"\brepo\b",
        r"\brefactor\b", r"\boptimize\b", r"\breview\b",
        r"\bperformance\b", r"\blatency\b", r"\bapi\b", r"\bbackend\b", r"\bdatabase\b",
        r"```", r"^\s*def\s+\w+\s*\(", r"^\s*class\s+\w+",
    ]
    hits = 0
    for pattern in code_patterns:
        if re.search(pattern, normalized, flags=re.MULTILINE):
            hits += 1
    if hits > 0:
        return "code", min(0.98, 0.52 + (0.15 * hits))
    return "general", 0.55


def has_strong_code_signal(text: str) -> bool:
    text_l = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", text_l) for word in STRONG_CODE_HINTS)


def has_file_signal(text: str) -> bool:
    return bool(re.search(r"\b\w+\.(py|js|ts|java|cpp|go)\b", text.lower()))


def has_tech_context(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in TECH_HINTS)


def build_routing_debug(text: str, auto_intent: str, auto_conf: float, threshold: float) -> dict[str, Any]:
    normalized = text.lower().strip()
    return {
        "input": text,
        "intent": auto_intent,
        "confidence": round(float(auto_conf), 3),
        "threshold": float(threshold),
        "signals": {
            "strong_code": has_strong_code_signal(text),
            "file_signal": has_file_signal(text),
            "tech_context": has_tech_context(text),
            "greeting": normalized in {"hi", "hello", "hey"},
        },
    }
