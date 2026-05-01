"""Core analysis engine for TRX-AI with local Ollama inference and rule fallback."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import builtins
import sys
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import requests

from config import AppConfig
from conversation_memory import ConversationMemory
from graph_query_adapter import CodeReviewGraphAdapter
from mcp_graph import detect_mcp_graph_status
from observability import METRICS, log_event
from router_memory import RouterMemory
from semantic_router import SemanticRouter


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


class RuleEngine:
    """Deterministic reasoning layer used as fallback."""

    STOPWORDS = {
        "a", "an", "the", "is", "am", "are", "was", "were", "and", "or", "but", "for",
        "with", "to", "of", "in", "on", "at", "by", "from", "this", "that", "it", "my",
        "our", "we", "i", "you", "they", "he", "she", "them", "as", "be", "been", "being",
    }

    def analyze(self, user_input: str, mode: str, past_context: list[dict[str, Any]]) -> dict[str, Any]:
        text = user_input.lower()
        nlp = self.extract_nlp_signals(user_input)

        debug_analysis: list[str] = []
        improvements: list[str] = []
        predictions: list[str] = []

        if nlp["keywords"]:
            debug_analysis.append("Primary themes: " + ", ".join(nlp["keywords"][:8]))
        if len(nlp["clauses"]) > 1:
            debug_analysis.append("Multiple linked conditions detected, indicating layered causes.")

        if "failed exam" in text or ("exam" in text and "fail" in text):
            debug_analysis.append("Preparation depth appears insufficient for exam-level performance.")
            improvements.append("Use a daily timed-practice schedule with weekly review checkpoints.")

        if "tired" in text or "exhausted" in text or "burnout" in text:
            debug_analysis.append("Energy management is reducing focus quality and consistency.")
            improvements.append("Stabilize sleep and protect one distraction-free recovery hour daily.")

        if "procrast" in text:
            debug_analysis.append("Task-start friction is causing avoidable delays.")
            improvements.append("Break work into 20-25 minute blocks and start from the easiest block.")

        if any(token in text for token in ("deadline", "late", "delay", "delays", "missed")):
            debug_analysis.append("Delivery reliability is impacted by weak milestone planning.")
            improvements.append("Set dated milestones with clear done criteria and owner per milestone.")

        if any(token in text for token in ("team", "communication", "misaligned", "inconsistent")):
            debug_analysis.append("Coordination gaps are creating execution inconsistency.")
            improvements.append("Run short daily syncs and enforce handoff standards.")

        repeated = self._extract_repeated_topics(text, past_context)
        if repeated:
            debug_analysis.append("Recurring topics detected: " + ", ".join(repeated[:3]))
            improvements.append("Prioritize these recurring issues first for maximum leverage.")

        if mode == "optimize":
            improvements.append("Rank actions by impact-to-effort ratio and execute top priority first.")

        if mode == "predict":
            predictions.append("If unchanged, current behavior is likely to increase near-term risk.")
            predictions.append("If top fixes are applied consistently, outcomes should stabilize in 2-8 weeks.")

        if not debug_analysis:
            debug_analysis.append("No strong trigger found. Problem framing needs more specifics.")
            improvements.append("Clarify goal, blocker, and success metric before next iteration.")

        if not predictions:
            predictions.append("Without corrective action, current issues are likely to persist.")
            predictions.append("With consistent execution, reliability should improve over upcoming cycles.")

        confidence = 64 + min(10, len(nlp["keywords"])) + min(6, 2 * len(repeated))
        confidence = max(45, min(90, confidence))

        return {
            "debug_analysis": self._dedupe(debug_analysis),
            "improvements": self._dedupe(improvements),
            "predictions": self._dedupe(predictions),
            "confidence_score": confidence,
        }

    def extract_nlp_signals(self, text: str) -> dict[str, Any]:
        normalized = re.sub(r"[^a-zA-Z0-9\s,.;:!?'-]", " ", text.lower())
        tokens = [tok for tok in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", normalized)]
        keywords = [tok for tok in tokens if tok not in self.STOPWORDS]
        clauses = [
            part.strip()
            for part in re.split(r"[.;!?]|\bbut\b|\bbecause\b|\bwhile\b|\balthough\b", normalized)
            if part.strip()
        ]
        return {
            "keywords": self._top_keywords(keywords, 10),
            "clauses": clauses[:6],
        }

    @staticmethod
    def _top_keywords(values: list[str], limit: int) -> list[str]:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        return [item[0] for item in ranked[:limit]]

    @staticmethod
    def _extract_repeated_topics(text: str, context: list[dict[str, Any]]) -> list[str]:
        matched: list[str] = []
        topics = ("exam", "sleep", "deadline", "team", "stress", "procrast")
        for entry in context:
            previous = str(entry.get("input", "")).lower()
            for topic in topics:
                if topic in text and topic in previous and topic not in matched:
                    matched.append(topic)
        return matched

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                result.append(item.strip())
                seen.add(key)
        return result


def call_local_llm(
    prompt: str,
    *,
    url: str = "http://localhost:11434/api/generate",
    model: str = "qwen3:8b",
    timeout: int = 120,
    max_new_tokens: int = 600,
    temperature: float = 0.3,
    retries: int = 2,
) -> dict[str, Any]:
    """Calls local Ollama and returns normalized response metadata."""
    last_status: int | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {
                        "num_predict": max(32, int(max_new_tokens)),
                        "temperature": max(0.0, min(1.0, float(temperature))),
                    },
                },
                timeout=timeout,
            )

            last_status = response.status_code
            if response.status_code != 200:
                continue

            try:
                data = response.json()
            except ValueError:
                continue

            text = str(data.get("response", "")).strip()
            if not text:
                text = str(data.get("generated_text", "")).strip()
            if not text:
                text = str(data.get("output_text", "")).strip()
            if not text:
                message = data.get("message")
                if isinstance(message, dict):
                    text = str(message.get("content", "")).strip()
            if not text:
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        text = str(first.get("text", "")).strip()
                        if not text and isinstance(first.get("message"), dict):
                            text = str(first["message"].get("content", "")).strip()
            if text:
                return {
                    "ok": True,
                    "text": text,
                    "status_code": response.status_code,
                    "done_reason": data.get("done_reason"),
                    "attempts": attempt + 1,
                }
        except requests.RequestException:
            pass

        if attempt < attempts - 1:
            time.sleep(0.5 * (attempt + 1))

    return {"ok": False, "text": "", "status_code": last_status, "attempts": attempts}


class RealityAnalyzer:
    """Orchestrates TRX-AI local chat/reasoning and deterministic fallback."""

    MODE_INSTRUCTIONS = {
        "debug": "Focus on root-cause diagnosis and fault isolation.",
        "optimize": "Focus on efficiency improvements and high-impact optimization.",
        "predict": "Focus on likely outcomes over the next 2-8 weeks if patterns stay unchanged.",
    }

    GREETING_EXACT = {
        "hi", "hello", "hey", "yo", "hola", "hi trx", "hey trx", "hello trx", "trx", "trx-ai",
    }

    COMMAND_PREFIXES = ("export", "agents", "mode", "help", "exit", "history", "save")

    STRONG_PROBLEM_SIGNALS = {
        "problem", "issue", "error", "failed", "fail", "stuck", "deadline", "exam", "tired",
        "debug", "optimize", "predict", "procrast", "stress", "anxious", "broken", "cannot", "can't",
        "delay", "delays", "late", "communication", "burnout",
    }
    REVIEW_PROMPT_MAX_CHARS = 2500

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._response_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._review_response_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._agent_flags: dict[str, bool] = {
            "debug": True,
            "improve": True,
            "predict": True,
        }
        self._failure_memory_path = Path("sessions") / "failure_memory.json"
        self._failure_memory = self._load_failure_memory()
        self._review_cache_disk_path = Path(self.config.review_cache_disk_path)
        self._load_review_cache_disk()
        self._graph_adapter = CodeReviewGraphAdapter()
        self._router_memory = RouterMemory()
        self._conversation_memory = ConversationMemory(max_turns=5)
        self._semantic_router = SemanticRouter()

    def set_active_agents(self, agents: list[str]) -> list[str]:
        normalized = [item.strip().lower() for item in agents if item.strip()]
        if not normalized:
            raise ValueError("No agents provided. Use: agents all OR agents debug improve predict")

        if "all" in normalized:
            selected = {"debug", "improve", "predict"}
        else:
            allowed = {"debug", "improve", "predict"}
            unknown = [name for name in normalized if name not in allowed]
            if unknown:
                raise ValueError("Unknown agent(s): " + ", ".join(unknown))
            selected = set(normalized)

        for name in self._agent_flags:
            self._agent_flags[name] = name in selected

        if not any(self._agent_flags.values()):
            self._agent_flags["debug"] = True

        return self.active_agents()

    def active_agents(self) -> list[str]:
        return [name for name in ("debug", "improve", "predict") if self._agent_flags.get(name, False)]

    def analyze(
        self,
        user_input: str,
        mode: str = "debug",
        past_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cleaned = user_input.strip()
        if not cleaned:
            raise ValueError("Input cannot be empty. Please describe your problem.")
        if mode not in self.MODE_INSTRUCTIONS:
            raise ValueError("Invalid mode. Use one of: debug, optimize, predict")

        normalized = re.sub(r"\s+", " ", cleaned.lower())
        started = time.perf_counter()
        configured_mode = str(getattr(self.config, "assistant_mode", "auto") or "auto").lower()
        if configured_mode not in {"auto", "general", "code"}:
            configured_mode = "auto"

        route_mode = detect_intent(cleaned) if configured_mode == "auto" else configured_mode
        rule_intent, rule_conf = detect_intent_confidence(cleaned)
        semantic_result = self._semantic_router.semantic_intent(cleaned)
        if semantic_result is not None and semantic_result.confidence > 0.6:
            auto_intent = semantic_result.intent
            auto_conf = float(semantic_result.confidence)
        else:
            auto_intent = rule_intent
            auto_conf = rule_conf
        code_threshold = 0.7
        routing_debug = build_routing_debug(cleaned, auto_intent, auto_conf, code_threshold)
        if configured_mode == "auto":
            if (
                auto_intent == "code"
                or has_strong_code_signal(cleaned)
                or has_file_signal(cleaned)
                or has_tech_context(cleaned)
            ) and (
                auto_conf >= code_threshold
                or has_strong_code_signal(cleaned)
                or has_file_signal(cleaned)
                or has_tech_context(cleaned)
            ):
                route_mode = "code"
            else:
                route_mode = "general"
        # Adaptive override from historical routing memory (after normal detection).
        bias = self._router_memory.get_bias(cleaned)
        if configured_mode == "auto" and bias in {"code", "general"}:
            if not (
                bias == "general"
                and (has_strong_code_signal(cleaned) or has_file_signal(cleaned) or has_tech_context(cleaned))
            ):
                route_mode = bias
        if normalized in {"hi", "hello", "hey"}:
            route_mode = "general"

        if self._looks_like_command(normalized):
            result = {
                "response_mode": "chat",
                "chat_response": "Command detected. Use CLI command handling for this action.",
                "intent": "command",
                "intent_source": "rule",
                "intent_confidence": 1.0,
            }
            return self._finalize_runtime_fields(result, started, routing_debug)

        if route_mode == "code":
            try:
                result = self.analyze_code_multi_agent(cleaned)
                self._router_memory.record(cleaned, "code")
                return self._finalize_runtime_fields(result, started, routing_debug)
            except ValueError:
                pass

        if route_mode == "general":
            context = past_context or []
            cache_key = self._cache_key(cleaned, mode, context)
            cached_general = self._cache_get(cache_key)
            if cached_general:
                return self._finalize_runtime_fields(cached_general, started, routing_debug)
            if classify_intent(cleaned) == "theory":
                result = self._handle_theory(cleaned)
                self._cache_set(cache_key, result)
                self._router_memory.record(cleaned, "general")
                return self._finalize_runtime_fields(result, started, routing_debug)
            general = self._handle_general_qa(cleaned)
            general["intent"] = "general"
            general["intent_source"] = "mode"
            general["intent_confidence"] = 1.0
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            METRICS.observe("latency_ms", float(elapsed_ms))
            log_event("intent_route", route="general", confidence=auto_conf, latency_ms=elapsed_ms)
            self._cache_set(cache_key, general)
            self._router_memory.record(cleaned, "general")
            return self._finalize_runtime_fields(general, started, routing_debug)

        if classify_intent(cleaned) == "theory":
            result = self._handle_theory(cleaned)
            self._router_memory.record(cleaned, "general")
            return self._finalize_runtime_fields(result, started, routing_debug)

        if is_general_question(cleaned):
            general = self._handle_general_qa(cleaned)
            general["intent"] = "general"
            general["intent_source"] = "rule"
            general["intent_confidence"] = 0.95
            self._router_memory.record(cleaned, "general")
            return self._finalize_runtime_fields(general, started, routing_debug)

        intent_data = self.detect_intent_hybrid(cleaned)
        intent = intent_data.get("intent", "chat")
        source = intent_data.get("source", "rule")
        confidence = float(intent_data.get("confidence", 0.5))

        if self.config.dev_mode:
            print(f"[Intent: {intent} | Source: {source} | Confidence: {confidence:.2f}]")

        if intent == "command":
            result = {
                "response_mode": "chat",
                "chat_response": "Command detected. Use CLI command handling for this action.",
                "intent": intent,
                "intent_source": source,
                "intent_confidence": confidence,
            }
            return self._finalize_runtime_fields(result, started, routing_debug)

        if intent == "greeting":
            result = {
                "response_mode": "chat",
                "chat_response": TRX_GREETING,
                "intent": intent,
                "intent_source": source,
                "intent_confidence": confidence,
            }
            return self._finalize_runtime_fields(result, started, routing_debug)

        if intent == "vague":
            general = self._handle_general_qa(cleaned)
            general["intent"] = intent
            general["intent_source"] = source
            general["intent_confidence"] = confidence
            return self._finalize_runtime_fields(general, started, routing_debug)

        context = past_context or []
        cache_key = self._cache_key(cleaned, mode, context)
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        if intent == "problem":
            result = self._handle_problem(cleaned, mode, context)
        elif intent in {"theory", "general"}:
            result = self._handle_general_qa(cleaned)
        else:
            result = self._handle_chat(cleaned)

        result["intent"] = intent
        result["intent_source"] = source
        result["intent_confidence"] = confidence
        self._router_memory.record(cleaned, "code" if intent == "code" else "general")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        METRICS.observe("latency_ms", float(elapsed_ms))
        log_event("intent_route", route=intent, confidence=confidence, latency_ms=elapsed_ms)
        self._cache_set(cache_key, result)
        return self._finalize_runtime_fields(result, started, routing_debug)

    def _finalize_runtime_fields(
        self,
        result: dict[str, Any],
        started: float,
        routing_debug: dict[str, Any],
    ) -> dict[str, Any]:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result["elapsed_ms"] = elapsed_ms
        statuses = [str(item).lower() for item in result.get("system_status", []) if str(item).strip()]
        if any("mcp_query=active" in item for item in statuses):
            result["mcp_query_status"] = "ACTIVE"
        elif any("mcp_query=degraded" in item for item in statuses):
            result["mcp_query_status"] = "DEGRADED"
        else:
            result["mcp_query_status"] = "UNKNOWN"
        result["cache_hit"] = any(item == "cache_hit" for item in statuses)
        breaker_state = str(METRICS.snapshot().get("states", {}).get("circuit_breaker_state", "CLOSED"))
        result["circuit_breaker_state"] = breaker_state
        METRICS.set_state("circuit_breaker_state", breaker_state)
        METRICS.set_state("mcp_query_status", str(result.get("mcp_query_status", "UNKNOWN")))
        result["system_health"] = {
            "mcp_query_status": result.get("mcp_query_status", "UNKNOWN"),
            "circuit_breaker": breaker_state,
        }
        log_event(
            "response_runtime",
            mcp_query_status=result.get("mcp_query_status", "UNKNOWN"),
            circuit_breaker_state=breaker_state,
            cache_status=("hit" if bool(result.get("cache_hit")) else "miss"),
            elapsed_ms=elapsed_ms,
        )
        result["routing_debug"] = routing_debug
        if str(result.get("response_mode", "")) == "chat":
            user_input = str(routing_debug.get("input", "")).strip()
            response = str(result.get("chat_response", "")).strip()
            if user_input and response:
                try:
                    self._conversation_memory.add(user_input, response)
                except Exception:
                    pass
        return result

    def detect_intent_hybrid(self, user_input: str) -> dict[str, Any]:
        """Priority order: code -> command -> greeting -> problem -> general."""
        normalized = re.sub(r"\s+", " ", user_input.strip().lower())
        if not normalized:
            return {"intent": "general", "confidence": 1.0, "source": "rule"}

        if self._looks_like_code(normalized):
            return {"intent": "code", "confidence": 0.95, "source": "rule"}

        if self._is_command(normalized):
            return {"intent": "command", "confidence": 1.0, "source": "rule"}

        if self._is_greeting(normalized):
            return {"intent": "greeting", "confidence": 1.0, "source": "rule"}

        classified = classify_intent(normalized)
        if classified == "theory":
            return {"intent": "theory", "confidence": 1.0, "source": "rule"}
        if classified == "code":
            return {"intent": "code", "confidence": 0.9, "source": "rule"}
        if classified == "problem":
            return {"intent": "problem", "confidence": 0.9, "source": "rule"}

        if self._is_strong_problem_signal(normalized):
            return {"intent": "problem", "confidence": 0.85, "source": "rule"}

        if self._is_vague(normalized):
            return {"intent": "general", "confidence": 0.6, "source": "rule"}

        llm_result = self._classify_intent_with_llm(normalized)
        if llm_result is not None:
            return llm_result

        return {"intent": "general", "confidence": 0.5, "source": "rule"}

    def _is_command(self, normalized: str) -> bool:
        if normalized in {"help", "exit", "history"}:
            return True

        return (
            normalized.startswith("export ")
            or normalized.startswith("agents ")
            or normalized.startswith("mode ")
            or normalized.startswith("save ")
        )

    def _looks_like_command(self, normalized: str) -> bool:
        return self._is_command(normalized)

    @staticmethod
    def _looks_like_code(normalized: str) -> bool:
        if any(re.search(pattern, normalized, flags=re.MULTILINE) for pattern in (
            r"^\s*def\s+\w+\s*\(",
            r"^\s*class\s+\w+",
            r"\bfunction\b",
            r"\bimport\b",
            r"#\s*file:",
            r"```",
            r"\{",
            r"\};",
        )):
            return True
        return any(re.search(pattern, normalized) for pattern in (
            r"\breview\b", r"\bfix\b", r"\bbug\b", r"\bstack\s+trace\b", r"\bexception\b",
        ))

    def _is_greeting(self, normalized: str) -> bool:
        if normalized in self.GREETING_EXACT:
            return True

        tokens = normalized.split()
        if len(tokens) <= 3 and any(tok in self.GREETING_EXACT for tok in tokens):
            return True

        return False

    def _is_strong_problem_signal(self, normalized: str) -> bool:
        score = sum(1 for marker in self.STRONG_PROBLEM_SIGNALS if marker in normalized)
        token_count = len([part for part in normalized.split(" ") if part])
        return score >= 1 and token_count >= 4

    @staticmethod
    def _is_vague(normalized: str) -> bool:
        vague_phrases = {
            "why i didnt do well",
            "why i didn't do well",
            "why i failed",
            "i didnt do well",
            "i didn't do well",
            "not doing well",
            "what went wrong",
            "help me",
        }
        token_count = len([part for part in normalized.split(" ") if part])
        if normalized in vague_phrases:
            return True
        if token_count < 3:
            return True
        return False

    @staticmethod
    def _is_definition_query(normalized: str) -> bool:
        token_count = len([part for part in normalized.split(" ") if part])
        definition_markers = (
            "full form",
            "stands for",
            "meaning of",
            "what is",
            "what does",
            "define",
        )
        if token_count > 10:
            return False
        return any(marker in normalized for marker in definition_markers)

    def _classify_intent_with_llm(self, user_input: str) -> dict[str, Any] | None:
        prompt = (
            "Classify intent for a CLI assistant.\n"
            "Allowed labels: greeting, chat, problem, command, theory, code, general\n"
            "Return valid JSON only:\n"
            "{\"intent\": \"...\", \"confidence\": 0.0}\n"
            f"Input: {user_input}"
        )

        local = self._call_local(prompt)
        if not local.get("ok"):
            return None

        text = str(local.get("text", "")).strip()
        if not text:
            return None

        parsed: dict[str, Any] | None = None
        try:
            parsed = json.loads(text)
        except ValueError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except ValueError:
                    parsed = None

        if not isinstance(parsed, dict):
            return None

        intent = str(parsed.get("intent", "")).strip().lower()
        if intent not in {"greeting", "chat", "problem", "command", "theory", "code", "general"}:
            return None

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        return {
            "intent": intent,
            "confidence": max(0.0, min(1.0, confidence)),
            "source": "llm",
        }

    def _handle_chat(self, user_input: str) -> dict[str, Any]:
        prompt = self._build_chat_prompt(user_input)
        local = self._call_local(prompt)

        if local.get("ok"):
            text = self._sanitize_identity(local.get("text", ""))
            return {
                "response_mode": "chat",
                "chat_response": text or "I am here to help.",
            }

        return {
            "response_mode": "chat",
            "chat_response": "I am here and ready to help. Share what you need, and we will solve it together.",
        }

    def _handle_theory(self, user_input: str) -> dict[str, Any]:
        prompt = self._build_theory_prompt(user_input, simple=False)
        response_text = self.call_llm(prompt)
        if self._contains_theory_contamination(response_text):
            # Failsafe: re-run with tighter prompt if analysis/debug text leaks in.
            response_text = self.call_llm(self._build_theory_prompt(user_input, simple=True))

        cleaned = response_text.strip()
        if not cleaned:
            lowered = user_input.lower()
            kb_answer = next((answer for key, answer in BASIC_KB.items() if key in lowered), "")
            cleaned = kb_answer or "Definition: A concise definition is currently unavailable."
        return {
            "response_mode": "chat",
            "chat_response": cleaned,
            "intent": "theory",
            "intent_source": "rule",
            "intent_confidence": 1.0,
            "analysis_source": "llm",
        }

    def _handle_general_qa(self, query: str) -> dict[str, Any]:
        q = query.lower().strip()

        # Greeting fast path before any LLM call.
        if q in {"hi", "hello", "hey"}:
            return {
                "response_mode": "chat",
                "chat_response": "Hey! 👋 I'm TRX. How can I help you today?",
                "confidence": 1.0,
                "status": ["general_fastpath"],
                "analysis_source": "rule-fastpath",
            }

        if q in {"hi trx", "hey trx", "hello trx", "trx", "trx-ai"}:
            return {
                "response_mode": "chat",
                "chat_response": TRX_GREETING,
                "confidence": 1.0,
                "status": ["general_fastpath"],
                "analysis_source": "rule-fastpath",
            }

        # Farewell fast path before any LLM call.
        if q in {"bye", "goodbye", "see you", "exit", "quit"}:
            return {
                "response_mode": "chat",
                "chat_response": "Goodbye! 👋 It was nice talking to you. Come back anytime.",
                "confidence": 1.0,
                "status": ["general_fastpath"],
                "analysis_source": "rule-fastpath",
            }

        if "what can you do" in q or "what u can do" in q:
            return {
                "response_mode": "chat",
                "chat_response": (
                    "Hey! 👋 I can help you with:\n\n"
                    "• Explaining concepts (like data warehouse, networks, etc.)\n"
                    "• Solving coding problems and debugging\n"
                    "• Reviewing and improving your code\n"
                    "• Answering general questions or helping you study\n\n"
                    "Just tell me what you need 🙂"
                ),
                "confidence": 1.0,
                "status": ["general_fastpath"],
                "analysis_source": "rule-fastpath",
            }

        # Short conversational prompts should get a natural assistant nudge.
        if len(q.split()) <= 2:
            return {
                "response_mode": "chat",
                "chat_response": "Hey! 👋 What would you like to explore today?",
                "confidence": 0.9,
                "analysis_source": "rule-fastpath",
            }

        prompt = f"""
You are TRX, an intelligent AI assistant.

Answer the user's question directly and naturally.

DO NOT:
- explain what the user asked
- describe the question
- analyze intent

ONLY:
- give a clear, helpful answer

User: {query}
Answer:
"""
        response_text = self.call_llm(prompt, temperature=max(0.7, self.config.local_llm_temperature))
        cleaned = self.clean_response(response_text.strip())
        if not cleaned:
            lowered = query.lower()
            kb_answer = next((answer for key, answer in BASIC_KB.items() if key in lowered), "")
            cleaned = kb_answer or (
                "I'm here to help! Could you rephrase or add a bit more detail? "
                "You can ask me about concepts, code, or problems you're working on."
            )
        return {
            "response_mode": "chat",
            "chat_response": cleaned,
            "analysis_source": "llm" if response_text.strip() else "rule-fallback",
        }

    @staticmethod
    def clean_response(text: str) -> str:
        bad_patterns = [
            "the user asked",
            "this means",
            "in simple terms the user",
            "the user is asking",
        ]

        lowered = text.lower()
        for pattern in bad_patterns:
            if pattern in lowered:
                return "Let me answer that directly: " + text.split(":")[-1].strip()
        return text

    def call_llm(self, prompt: str, temperature: float | None = None) -> str:
        local = self._call_local(prompt, temperature=temperature)
        if local.get("ok"):
            return self._sanitize_identity(str(local.get("text", ""))).strip()
        return ""

    @staticmethod
    def _contains_theory_contamination(response_text: str) -> bool:
        lowered = response_text.lower()
        blocked_markers = ("debug", "analysis", "user is referring")
        return any(marker in lowered for marker in blocked_markers)

    @staticmethod
    def _build_theory_prompt(user_input: str, *, simple: bool) -> str:
        if simple:
            return (
                "You are a strict technical assistant.\n\n"
                "Give only a textbook-style definition.\n"
                "Do not include debugging, analysis, or meta commentary.\n\n"
                f"Question: {user_input}\n\n"
                "Format:\n"
                "Definition:\n"
                "<clear explanation>"
            )
        return (
            "You are a strict technical assistant.\n\n"
            "Answer the question with a clear, textbook-style definition.\n"
            "Do NOT analyze. Do NOT debug.\n\n"
            "Question:\n"
            f"{user_input}\n\n"
            "Format:\n"
            "Definition:\n"
            "<clear explanation>\n\n"
            "Example:\n"
            "<optional example>"
        )

    def _handle_problem(self, user_input: str, mode: str, context: list[dict[str, Any]]) -> dict[str, Any]:
        fallback = RuleEngine().analyze(user_input, mode, context)

        debug_result, debug_source = self.run_debug_agent(user_input, context, fallback["debug_analysis"])
        improve_result, improve_source = self.run_improve_agent(user_input, debug_result, fallback["improvements"])
        predict_result, predict_source = self.run_predict_agent(user_input, debug_result, improve_result, fallback["predictions"])

        final_insight = self._build_final_insight(debug_result, improve_result, predict_result)

        llm_scores = [source for source in (debug_source, improve_source, predict_source) if source == "llm"]
        overall_source = "llm" if llm_scores else "rule"

        return {
            "response_mode": "analysis",
            "debug_analysis": debug_result,
            "improvements": improve_result,
            "predictions": predict_result,
            "final_insight": final_insight,
            "confidence_score": int(fallback.get("confidence_score", 60)),
            "analysis_source": overall_source,
            "active_agents": self.active_agents(),
        }

    def run_debug_agent(
        self,
        user_input: str,
        context: list[dict[str, Any]],
        fallback_lines: list[str],
    ) -> tuple[list[str], str]:
        if not self._agent_flags.get("debug", True):
            return ["Debug agent is disabled."], "rule"

        prompt = self._build_debug_agent_prompt(user_input, context)
        local = self._call_local(prompt)
        if local.get("ok"):
            lines = self.parse_section_items(self._sanitize_identity(str(local.get("text", ""))), "[DEBUG ANALYSIS]")
            if lines:
                return lines, "llm"

        return fallback_lines, "rule"

    def run_improve_agent(
        self,
        user_input: str,
        debug_result: list[str],
        fallback_lines: list[str],
    ) -> tuple[list[str], str]:
        if not self._agent_flags.get("improve", True):
            return ["Improve agent is disabled."], "rule"

        prompt = self._build_improve_agent_prompt(user_input, debug_result)
        local = self._call_local(prompt)
        if local.get("ok"):
            lines = self.parse_section_items(self._sanitize_identity(str(local.get("text", ""))), "[IMPROVEMENTS]")
            if lines:
                return lines, "llm"

        return fallback_lines, "rule"

    def run_predict_agent(
        self,
        user_input: str,
        debug_result: list[str],
        improve_result: list[str],
        fallback_lines: list[str],
    ) -> tuple[list[str], str]:
        if not self._agent_flags.get("predict", True):
            return ["Predict agent is disabled."], "rule"

        prompt = self._build_predict_agent_prompt(user_input, debug_result, improve_result)
        local = self._call_local(prompt)
        if local.get("ok"):
            lines = self.parse_section_items(self._sanitize_identity(str(local.get("text", ""))), "[PREDICTIONS]")
            if lines:
                return lines, "llm"

        return fallback_lines, "rule"

    def _call_local(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        if not self.config.use_local_llm:
            METRICS.set_state("circuit_breaker_state", "OPEN")
            return {"ok": False, "text": "", "status_code": None}

        if self.config.dev_mode:
            print("[DEBUG] Prompt sent to LLM:")
            self._debug_print_safe(prompt[:1200])
        response = call_local_llm(
            prompt,
            url=self.config.local_llm_url,
            model=self.config.local_llm_model,
            timeout=self.config.local_llm_timeout_seconds,
            max_new_tokens=(max_new_tokens if max_new_tokens is not None else self.config.local_llm_max_new_tokens),
            temperature=(
                self.config.local_llm_temperature
                if temperature is None
                else max(0.0, min(1.0, float(temperature)))
            ),
            retries=self.config.local_llm_retries,
        )
        METRICS.set_state("circuit_breaker_state", "CLOSED" if response.get("ok") else "OPEN")
        if self.config.dev_mode:
            print("[DEBUG] Raw response:")
            self._debug_print_safe(str(response.get("text", ""))[:1200] or "<empty>")
        return response

    def analyze_code_multi_agent(self, code: str) -> dict[str, Any]:
        """Runs structured multi-agent style code review on provided code content."""
        cleaned_code = code.strip()
        if not cleaned_code:
            raise ValueError("Code content is empty.")
        language = self._detect_language(cleaned_code)
        review_cache_key = self._review_cache_key(cleaned_code, language)
        cached_review = self._review_cache_get(review_cache_key)
        if cached_review is not None:
            cached_copy = copy.deepcopy(cached_review)
            statuses = cached_copy.get("system_status", [])
            if isinstance(statuses, list) and "cache_hit" not in statuses:
                statuses.append("cache_hit")
                statuses.extend(self._mcp_status_tags())
                cached_copy["system_status"] = statuses
            return cached_copy

        # Guard against oversized payloads that can degrade model quality.
        code_for_prompt = cleaned_code[: self.REVIEW_PROMPT_MAX_CHARS]
        graph_context = self._build_graph_context(cleaned_code)
        if self.config.dev_mode or self.config.review_logging:
            print("[Code Review -> Using LLM]")
        started = time.perf_counter()
        try:
            pipeline = self._run_code_review_multi_agent_pipeline(
                code_for_prompt,
                language,
                graph_context=graph_context,
            )
        except TypeError:
            # Backward compatibility for tests/mocks expecting legacy two-arg signature.
            pipeline = self._run_code_review_multi_agent_pipeline(code_for_prompt, language)
        elapsed = time.perf_counter() - started
        if self.config.dev_mode or self.config.review_logging:
            print(f"[LLM time: {elapsed:.2f}s]")

        if not pipeline.get("ok"):
            self._record_failure(cleaned_code, "", "llm_unavailable")
            result = self._build_review_failure_result(cleaned_code, language)
            self._attach_graph_insights(result, {"graph_context": graph_context})
            self._review_cache_set(review_cache_key, result)
            return result

        response_text = str(pipeline.get("text", ""))
        if self.config.dev_mode or self.config.review_logging:
            print("[RAW LLM OUTPUT]")
            self._debug_print_safe(response_text)
        if "no high-confidence" in response_text.lower():
            response_text += "\n\nNOTE: Response may be generic due to weak prompt or model limitations."
        sections = self._parse_code_review_sections(response_text)
        sections = self._normalize_review_sections(sections)
        truncated = bool(pipeline.get("truncated"))
        fixed_code, repaired_from_invalid = self._resolve_fixed_code(
            response_text=response_text,
            cleaned_code=cleaned_code,
            language=language,
        )
        if not fixed_code:
            self._record_failure(cleaned_code, response_text, "fixed_code_missing")

        result = self._build_review_success_result(
            sections=sections,
            pipeline=pipeline,
            cleaned_code=cleaned_code,
            fixed_code=fixed_code,
            truncated=truncated,
            repaired_from_invalid=repaired_from_invalid,
            raw_llm_output=response_text,
        )
        self._attach_graph_insights(result, pipeline)
        self._review_cache_set(review_cache_key, result)
        return result

    def _build_review_failure_result(self, cleaned_code: str, language: str) -> dict[str, Any]:
        fallback_sections = self._fallback_code_review_sections(cleaned_code)
        heuristic_fixed = self._rule_based_fixed_code(cleaned_code, language)
        confidence_score = int(fallback_sections.get("confidence_score", 65))
        return {
            "response_mode": "analysis",
            "analysis_text": self._compose_code_review_text(fallback_sections),
            "system_status": [
                "intent=review",
                "source=rule-fallback",
                "llm_unavailable",
                "fixed_code_heuristic",
                *self._mcp_status_tags(),
            ],
            "confidence": round(confidence_score / 100.0, 2),
            "confidence_score": confidence_score,
            "debug_analysis": fallback_sections.get("code_debug", []),
            "improvements": fallback_sections.get("code_improvements", []) + fallback_sections.get("security", []),
            "predictions": fallback_sections.get("performance", []),
            "final_insight": fallback_sections.get("final_summary", []),
            "fixed_code": heuristic_fixed,
            "analysis_source": "rule",
            "intent": "review",
            "intent_source": "rule",
        }

    def _resolve_fixed_code(
        self,
        *,
        response_text: str,
        cleaned_code: str,
        language: str,
    ) -> tuple[str, bool]:
        fixed_code = self._extract_fixed_code(response_text)
        repaired_from_invalid = False
        needs_optimization_gate = language == "python" and self._requires_python_optimization(cleaned_code)

        if not fixed_code:
            fixed_code = self._generate_fixed_code_only(cleaned_code, language)

        if fixed_code and not self._is_valid_fixed_code(fixed_code, language):
            repaired = self._repair_invalid_code(fixed_code, cleaned_code, language)
            if repaired and self._is_valid_fixed_code(repaired, language):
                fixed_code = repaired
                repaired_from_invalid = True
            else:
                fixed_code = ""
        if fixed_code and needs_optimization_gate:
            if not self._passes_python_optimization_gate(cleaned_code, fixed_code):
                fixed_code = ""

        if not fixed_code:
            heuristic_fixed = self._rule_based_fixed_code(cleaned_code, language)
            if heuristic_fixed and self._is_valid_fixed_code(heuristic_fixed, language):
                if needs_optimization_gate and not self._passes_python_optimization_gate(cleaned_code, heuristic_fixed):
                    heuristic_fixed = ""
                fixed_code = heuristic_fixed

        return fixed_code, repaired_from_invalid

    def _build_review_success_result(
        self,
        *,
        sections: dict[str, Any],
        pipeline: dict[str, Any],
        cleaned_code: str,
        fixed_code: str,
        truncated: bool,
        repaired_from_invalid: bool,
        raw_llm_output: str,
    ) -> dict[str, Any]:
        confidence_score = int(sections.get("confidence_score", 70))
        analysis_text = self._compose_code_review_text(sections)
        status = ["intent=review", "source=llm", "llm_ok", *self._mcp_status_tags()]
        if len(cleaned_code) > self.REVIEW_PROMPT_MAX_CHARS:
            status.append("code_truncated_for_review")
        if truncated:
            status.append("llm_output_truncated")
        status.append("fixed_code_ready" if fixed_code else "fixed_code_missing")
        if repaired_from_invalid:
            status.append("fixed_code_repaired")
        weak_parsing = (
            "LLM output parsing incomplete - check raw output" in sections.get("performance", [])
            or "No critical bugs were explicitly identified." in sections.get("code_debug", [])
        )
        if weak_parsing:
            status.append("structured_parse_weak")
        status.append(f"critic_score={pipeline.get('critic_score', 0.0):.2f}")
        status.append(f"steps={len(pipeline.get('steps', []))}")

        return {
            "response_mode": "analysis",
            "analysis_text": analysis_text,
            "system_status": status,
            "confidence": round(confidence_score / 100.0, 2),
            "confidence_score": confidence_score,
            "debug_analysis": sections.get("code_debug", []),
            "improvements": sections.get("code_improvements", []) + sections.get("security", []),
            "predictions": sections.get("performance", []),
            "final_insight": sections.get("final_summary", []),
            "fixed_code": fixed_code,
            "raw_llm_output": raw_llm_output[:18000],
            "analysis_source": "llm",
            "intent": "review",
            "intent_source": "llm",
        }

    def _run_code_review_multi_agent_pipeline(
        self,
        code_for_prompt: str,
        language: str,
        *,
        graph_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        steps: list[str] = []
        truncated = False
        failure_context = self._failure_context_snippet()
        graph_context = graph_context or {}
        graph_context_text = self._graph_context_text(graph_context)

        analyzer_prompt = (
            "You are Analyzer Agent.\n"
            f"Inspect the {language} code and list concrete bug/performance/security risks in short bullets.\n"
            "Use failure memory to avoid repeated weak outputs.\n"
            "Return plain text.\n\n"
            f"Failure Memory:\n{failure_context}\n\n"
            f"Graph Context:\n{graph_context_text}\n\n"
            f"{code_for_prompt}"
        )
        analyzer_resp = self._call_local(analyzer_prompt, max_new_tokens=350)
        if not analyzer_resp.get("ok"):
            return {
                "ok": False,
                "text": "",
                "steps": ["analyzer_failed"],
                "critic_score": 0.0,
                "truncated": False,
                "graph_context": graph_context,
            }
        analyzer_notes = str(analyzer_resp.get("text", "")).strip()
        steps.append("analyzer")

        generator_prompt = (
            "You are Generator Agent.\n"
            "Produce STRICT structured output with sections:\n"
            "CODE DEBUG:\nCODE IMPROVEMENTS:\nPERFORMANCE:\nSECURITY:\nFIX SUGGESTIONS:\n"
            "FIXED CODE:\nFINAL SUMMARY:\nCONFIDENCE:\n"
            "Be specific.\n\n"
            f"Failure Memory:\n{failure_context}\n\n"
            f"Graph Context:\n{graph_context_text}\n\n"
            f"Analyzer Notes:\n{analyzer_notes}\n\n"
            f"Code:\n{code_for_prompt}"
        )
        generator_resp = self._call_local(generator_prompt, max_new_tokens=1000)
        if not generator_resp.get("ok"):
            return {
                "ok": False,
                "text": "",
                "steps": steps + ["generator_failed"],
                "critic_score": 0.0,
                "truncated": False,
                "graph_context": graph_context,
            }
        candidate = str(generator_resp.get("text", "")).strip()
        steps.append("generator")
        if str(generator_resp.get("done_reason", "")).lower() == "length":
            truncated = True

        critic_score = 0.0
        for loop in range(2):
            critic_prompt = (
                "You are Critic Agent.\n"
                "Evaluate this review quality for specificity, section completeness, and actionable fixes.\n"
                "Return JSON only: {\"score\": 0.0, \"issues\": [\"...\"]}\n\n"
                f"{candidate}"
            )
            critic_resp = self._call_local(critic_prompt, max_new_tokens=220)
            issues_text = ""
            if critic_resp.get("ok"):
                critic_payload = str(critic_resp.get("text", "")).strip()
                match = re.search(r"\{[\s\S]*\}", critic_payload)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        critic_score = float(parsed.get("score", 0.0))
                        issues = parsed.get("issues", [])
                        if isinstance(issues, list):
                            issues_text = "\n".join(f"- {str(i)}" for i in issues[:8])
                    except (ValueError, TypeError):
                        critic_score = 0.0
            steps.append(f"critic_{loop+1}")
            if critic_score >= 0.75:
                break
            if loop == 0 and critic_score >= 0.65:
                # Avoid extra latency for near-pass quality.
                break

            fixer_prompt = (
                "You are Fixer Agent.\n"
                "Improve the review text below.\n"
                "- Keep exact required sections\n"
                "- Increase specificity\n"
                f"- Ensure FIXED CODE is complete and valid {language} code\n\n"
                f"Critic Issues:\n{issues_text or '- low confidence'}\n\n"
                f"Review Text:\n{candidate}\n\n"
                f"Original Code:\n{code_for_prompt}"
            )
            fixer_resp = self._call_local(fixer_prompt, max_new_tokens=1200)
            if not fixer_resp.get("ok"):
                break
            candidate = str(fixer_resp.get("text", "")).strip()
            steps.append(f"fixer_{loop+1}")
            if str(fixer_resp.get("done_reason", "")).lower() == "length":
                truncated = True

        return {
            "ok": True,
            "text": candidate,
            "steps": steps,
            "critic_score": critic_score,
            "truncated": truncated,
            "graph_context": graph_context,
        }

    def _generate_fixed_code_only(self, original_code: str, language: str) -> str:
        code_for_prompt = original_code[:2200] if len(original_code) > 2200 else original_code
        prompt = (
            f"You are a senior {language} engineer.\n"
            f"Return a fully corrected version of this {language} code.\n"
            "Rules:\n"
            f"- Return only {language} code\n"
            "- No markdown, no explanations\n"
            f"- Ensure code is syntactically valid {language}\n"
            "- Do not use undefined identifiers\n"
            "- Avoid introducing new global mutable state\n"
            "- Keep behavior correct and improve algorithmic efficiency when obvious\n"
            "- Preserve intent while fixing bugs, edge cases, and robustness\n\n"
            f"{code_for_prompt}\n"
        )

        best_candidate = ""
        for token_budget in (900, 1200, 1500):
            response = self._call_local(prompt, max_new_tokens=token_budget)
            if not response.get("ok"):
                continue
            text = str(response.get("text", "")).strip()
            candidate = self._extract_code_candidate(text)
            if candidate and len(candidate) > len(best_candidate):
                best_candidate = candidate
            if candidate and self._is_valid_fixed_code(candidate, language):
                return candidate

            if str(response.get("done_reason", "")).lower() == "length" and candidate:
                continuation = self._continue_fixed_code(candidate, code_for_prompt, language)
                if continuation and self._is_valid_fixed_code(continuation, language):
                    return continuation
                if continuation and len(continuation) > len(best_candidate):
                    best_candidate = continuation

        return best_candidate

    def _continue_fixed_code(self, partial_code: str, original_code: str, language: str) -> str:
        prompt = (
            f"Continue and complete this {language} code so it becomes syntactically valid.\n"
            "Return full corrected code only.\n"
            "No markdown.\n\n"
            "ORIGINAL CODE:\n"
            f"{original_code[:1800]}\n\n"
            "PARTIAL FIXED CODE:\n"
            f"{partial_code}\n"
        )
        response = self._call_local(prompt, max_new_tokens=1400)
        if not response.get("ok"):
            return ""
        text = str(response.get("text", "")).strip()
        return self._extract_code_candidate(text)

    def _repair_invalid_code(self, invalid_code: str, original_code: str, language: str) -> str:
        prompt = (
            f"Repair syntax errors in this {language} code.\n"
            f"Return only valid {language} code. No markdown.\n"
            "Preserve behavior while fixing syntax.\n\n"
            "ORIGINAL CODE:\n"
            f"{original_code[:1800]}\n\n"
            "INVALID FIXED CODE:\n"
            f"{invalid_code}\n"
        )
        response = self._call_local(prompt, max_new_tokens=1400)
        if not response.get("ok"):
            return ""
        return self._extract_code_candidate(str(response.get("text", "")).strip())

    @staticmethod
    def _extract_code_candidate(text: str) -> str:
        if not text:
            return ""
        fenced = re.search(r"```(?:[a-zA-Z0-9_+\-#.]*)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        cleaned = re.sub(r"^\s*```(?:[a-zA-Z0-9_+\-#.]*)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _is_valid_python_code(code: str) -> bool:
        text = code.strip()
        if not text:
            return False
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False

    def _is_valid_fixed_code(self, code: str, language: str) -> bool:
        text = code.strip()
        if not text:
            return False
        if language == "python":
            return self._is_valid_python_code(text) and not self._has_obvious_python_runtime_risks(text)
        # Non-Python: apply lightweight generic validity checks.
        if len(text) < 8:
            return False
        if text.count("{") != text.count("}"):
            return False
        if text.count("(") != text.count(")"):
            return False
        return True

    @staticmethod
    def _has_obvious_python_runtime_risks(code: str) -> bool:
        """Returns True when simple static checks find unresolved names."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return True

        builtin_names = set(dir(builtins))
        module_defined: set[str] = set()

        def _collect_target_names(target: ast.AST, out: set[str]) -> None:
            if isinstance(target, ast.Name):
                out.add(target.id)
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    _collect_target_names(item, out)

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                module_defined.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_defined.add((alias.asname or alias.name.split(".")[0]).strip())
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    module_defined.add((alias.asname or alias.name).strip())
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    _collect_target_names(target, module_defined)
            elif isinstance(node, ast.AnnAssign):
                _collect_target_names(node.target, module_defined)
            elif isinstance(node, ast.AugAssign):
                _collect_target_names(node.target, module_defined)

        class FunctionRiskVisitor(ast.NodeVisitor):
            def __init__(self, known_module: set[str], known_builtins: set[str]) -> None:
                self.known_module = known_module
                self.known_builtins = known_builtins
                self.has_unresolved = False

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
                self._visit_function_like(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
                self._visit_function_like(node)

            def _visit_function_like(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                local_names: set[str] = set()
                global_names: set[str] = set()
                nonlocal_names: set[str] = set()

                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    local_names.add(arg.arg)
                if node.args.vararg is not None:
                    local_names.add(node.args.vararg.arg)
                if node.args.kwarg is not None:
                    local_names.add(node.args.kwarg.arg)

                for child in ast.walk(node):
                    if isinstance(child, ast.Global):
                        global_names.update(child.names)
                    elif isinstance(child, ast.Nonlocal):
                        nonlocal_names.update(child.names)
                    elif isinstance(child, ast.Assign):
                        for target in child.targets:
                            _collect_target_names(target, local_names)
                    elif isinstance(child, ast.AnnAssign):
                        _collect_target_names(child.target, local_names)
                    elif isinstance(child, ast.AugAssign):
                        _collect_target_names(child.target, local_names)
                    elif isinstance(child, ast.For):
                        _collect_target_names(child.target, local_names)
                    elif isinstance(child, ast.With):
                        for item in child.items:
                            if item.optional_vars is not None:
                                _collect_target_names(item.optional_vars, local_names)
                    elif isinstance(child, ast.comprehension):
                        _collect_target_names(child.target, local_names)
                    elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child is not node:
                        local_names.add(child.name)

                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        name = child.id
                        if (
                            name in local_names
                            or name in global_names
                            or name in nonlocal_names
                            or name in self.known_module
                            or name in self.known_builtins
                        ):
                            continue
                        self.has_unresolved = True
                        return

        visitor = FunctionRiskVisitor(module_defined, builtin_names)
        visitor.visit(tree)
        return visitor.has_unresolved

    @staticmethod
    def _has_recursive_fibonacci_pattern(code: str) -> bool:
        pattern = re.compile(
            r"def\s+fibonacci\s*\([^)]*\)\s*:[\s\S]*?return\s+fibonacci\s*\([^)]*\)\s*\+\s*fibonacci\s*\([^)]*\)",
            flags=re.IGNORECASE,
        )
        return bool(pattern.search(code))

    @staticmethod
    def _has_bubble_sort_range_n_pattern(code: str) -> bool:
        has_range_n = re.search(r"for\s+j\s+in\s+range\(\s*n\s*\)\s*:", code) is not None
        has_next_index = re.search(r"arr\s*\[\s*j\s*\+\s*1\s*\]", code) is not None
        return has_range_n and has_next_index

    def _requires_python_optimization(self, code: str) -> bool:
        return self._has_recursive_fibonacci_pattern(code) or self._has_bubble_sort_range_n_pattern(code)

    def _passes_python_optimization_gate(self, original_code: str, fixed_code: str) -> bool:
        original_has_recursive_fib = self._has_recursive_fibonacci_pattern(original_code)
        fixed_has_recursive_fib = self._has_recursive_fibonacci_pattern(fixed_code)
        if original_has_recursive_fib and fixed_has_recursive_fib:
            return False

        original_has_bad_bubble = self._has_bubble_sort_range_n_pattern(original_code)
        fixed_has_bad_bubble = self._has_bubble_sort_range_n_pattern(fixed_code)
        if original_has_bad_bubble and fixed_has_bad_bubble:
            return False

        return True

    @staticmethod
    def _rule_based_fixed_code(original_code: str, language: str) -> str:
        """Applies conservative heuristic fixes when LLM is unavailable."""
        if language != "python":
            return original_code
        fixed = original_code

        # Bubble-sort loop bound fix: for j in range(n) -> for j in range(n - i - 1)
        fixed = re.sub(
            r"for\s+j\s+in\s+range\(\s*n\s*\)\s*:",
            "for j in range(n - i - 1):",
            fixed,
        )

        # Guard find_max against empty input where it directly reads arr[0].
        find_max_pattern = re.compile(
            r"(def\s+find_max\s*\(\s*arr\s*\)\s*:\s*\n)([ \t]+)(max_val\s*=\s*arr\[0\])",
            flags=re.MULTILINE,
        )
        fixed = find_max_pattern.sub(
            r"\1\2if not arr:\n\2    return None\n\2\3",
            fixed,
        )

        # Fibonacci optimization fallback: recursive O(2^n) -> iterative O(n).
        fibonacci_pattern = re.compile(
            r"def\s+fibonacci\s*\(\s*n\s*\)\s*:\s*[\s\S]*?return\s+fibonacci\s*\(\s*n\s*-\s*1\s*\)\s*\+\s*fibonacci\s*\(\s*n\s*-\s*2\s*\)",
            flags=re.MULTILINE,
        )
        fibonacci_replacement = (
            "def fibonacci(n):\n"
            "    if n < 0:\n"
            "        raise ValueError(\"n must be non-negative\")\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b"
        )
        fixed = fibonacci_pattern.sub(fibonacci_replacement, fixed)

        return fixed

    @staticmethod
    def _debug_print_safe(text: str) -> None:
        try:
            print(text)
            return
        except UnicodeEncodeError:
            pass

        encoding = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)

    def _load_failure_memory(self) -> list[dict[str, str]]:
        try:
            if not self._failure_memory_path.exists():
                return []
            data = json.loads(self._failure_memory_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)][-30:]
        except (OSError, ValueError, TypeError):
            pass
        return []

    def _save_failure_memory(self) -> None:
        try:
            self._failure_memory_path.parent.mkdir(parents=True, exist_ok=True)
            self._failure_memory_path.write_text(
                json.dumps(self._failure_memory[-30:], indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _record_failure(self, user_input: str, bad_output: str, reason: str) -> None:
        self._failure_memory.append(
            {
                "input": user_input[:500],
                "bad_output": bad_output[:800],
                "failure_reason": reason[:200],
            }
        )
        self._save_failure_memory()

    def _failure_context_snippet(self) -> str:
        if not self._failure_memory:
            return "none"
        recent = self._failure_memory[-3:]
        lines = []
        for item in recent:
            lines.append(
                f"- input={item.get('input','')[:80]} | reason={item.get('failure_reason','')[:80]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _detect_language(code: str) -> str:
        text = code.strip()
        lower = text.lower()
        if "def " in lower or "import " in lower or "if __name__ ==" in lower:
            return "python"
        if "console.log" in lower or "function " in lower or ("=>" in lower and "{" in lower):
            return "javascript"
        if "public static void main" in lower or "system.out.println" in lower:
            return "java"
        if "#include" in lower or "printf(" in lower or "scanf(" in lower:
            return "c"
        if "cout <<" in lower or "std::" in lower:
            return "cpp"
        if "package main" in lower or "fmt.println" in lower:
            return "go"
        if "select " in lower and " from " in lower:
            return "sql"
        return "code"

    def _build_code_review_prompt(self, code: str, language: str) -> str:
        failure_context = self._failure_context_snippet()
        return f"""
You are a senior software engineer performing a deep and practical {language} code review.

Analyze the following code carefully and provide actionable insights.

CODE:
{code}

RECENT FAILURE MEMORY (use this to avoid repeating weak behavior):
{failure_context}

---

INSTRUCTIONS:
- Be specific to THIS code
- Identify real issues even if small
- Reference patterns or lines when possible
- Explain WHY something is a problem
- Suggest HOW to fix it
- Provide improved code snippets where applicable

---

OUTPUT FORMAT (STRICT):

CODE DEBUG:
- Identify bugs, edge cases, missing validation
- Mention exact problematic patterns

CODE IMPROVEMENTS:
- Refactoring suggestions
- Code structure improvements
- Naming issues
- Show improved snippets if possible

PERFORMANCE:
- Inefficient operations
- Memory or loop optimization

SECURITY:
- Unsafe practices
- Input validation issues

FIX SUGGESTIONS:
- Provide corrected code snippets
- Show before -> after where possible

FIXED CODE:
Return full corrected version of the {language} code.

FINAL SUMMARY:
- Most important issue to fix first
- One-line actionable advice

CONFIDENCE:
- Give realistic percentage (60–95)

---

IMPORTANT:
- DO NOT return generic text
- DO NOT say "no issues found" without explanation
- ALWAYS provide at least 2 concrete improvements
- Prefer short, practical explanations over long theory
"""

    @staticmethod
    def _parse_code_review_sections(text: str) -> dict[str, Any]:
        json_sections = RealityAnalyzer._try_parse_review_json(text)
        if json_sections is not None:
            return json_sections

        header_aliases = {
            "CODE DEBUG": "code_debug",
            "DEBUG ANALYSIS": "code_debug",
            "CODE IMPROVEMENTS": "code_improvements",
            "IMPROVEMENTS": "code_improvements",
            "PERFORMANCE": "performance",
            "SECURITY (IF APPLICABLE)": "security",
            "SECURITY": "security",
            "FIX SUGGESTIONS": "fix_suggestions",
            "FINAL SUMMARY": "final_summary",
            "CONFIDENCE": "confidence",
        }
        sections: dict[str, list[str]] = {
            "code_debug": [],
            "code_improvements": [],
            "performance": [],
            "security": [],
            "fix_suggestions": [],
            "final_summary": [],
            "confidence": [],
        }

        def _normalize_header(value: str) -> str:
            cleaned = value.strip()
            cleaned = re.sub(r"^[#>\-\*\s`]+", "", cleaned)
            cleaned = cleaned.replace("**", "").replace("__", "").strip()
            cleaned = cleaned.rstrip(":").strip().upper()
            return cleaned

        current: str | None = None
        in_code_block = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            if re.fullmatch(r"[-=_]{3,}", line):
                continue
            normalized_header = _normalize_header(line)
            if normalized_header in header_aliases:
                current = header_aliases[normalized_header]
                continue
            if current is None:
                continue
            item = re.sub(r"^[>\-\*\u2022]+\s*", "", line).strip()
            item = item.strip("*").strip()
            item = item.replace("**", "").strip()
            item = re.sub(r"^\d+\.\s*", "", item)
            if item:
                sections[current].append(item)

        confidence_score = 70
        for item in sections["confidence"]:
            match = re.search(r"(\d{1,3})\s*%?", item)
            if match:
                confidence_score = max(0, min(100, int(match.group(1))))
                break

        if not sections["code_debug"]:
            sections["code_debug"] = ["No critical bugs were explicitly identified."]
        if not sections["code_improvements"]:
            sections["code_improvements"] = ["Apply incremental refactoring for readability and maintainability."]
        if not sections["performance"]:
            sections["performance"] = ["LLM output parsing incomplete - check raw output"]
        if not sections["security"]:
            sections["security"] = ["No immediate security issue was explicitly identified."]
        if not sections["final_summary"]:
            sections["final_summary"] = ["Prioritize correctness issues first, then refactor and optimize."]

        return {
            "code_debug": sections["code_debug"],
            "code_improvements": sections["code_improvements"],
            "performance": sections["performance"],
            "security": sections["security"],
            "fix_suggestions": sections["fix_suggestions"],
            "final_summary": sections["final_summary"],
            "confidence_score": confidence_score,
        }

    @staticmethod
    def _try_parse_review_json(text: str) -> dict[str, Any] | None:
        payload: dict[str, Any] | None = None
        raw = text.strip()
        candidates: list[str] = [raw]
        fenced = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            candidates.append(brace_match.group(0))

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break

        if payload is None:
            return None

        def _normalize_items(value: Any) -> list[str]:
            out: list[str] = []
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # Support {"issue": "...", "description": "..."} and similar.
                        desc = str(item.get("description", "")).strip()
                        issue = str(item.get("issue", "")).strip()
                        code = str(item.get("code", "")).strip()
                        if desc and issue:
                            out.append(f"{issue}: {desc}")
                        elif desc:
                            out.append(desc)
                        elif code:
                            out.append(code)
                        elif issue:
                            out.append(issue)
                    else:
                        text_item = str(item).strip()
                        if text_item:
                            out.append(text_item)
            elif isinstance(value, dict):
                for key, val in value.items():
                    line = f"{str(key).strip()}: {str(val).strip()}".strip(": ").strip()
                    if line:
                        out.append(line)
            elif isinstance(value, str):
                for row in value.splitlines():
                    row = row.strip()
                    if row:
                        out.append(row)
            return out

        confidence_score = 70
        confidence_candidates = _normalize_items(payload.get("confidence", []))
        if not confidence_candidates and "score" in payload:
            confidence_candidates = _normalize_items(payload.get("score"))
        for item in confidence_candidates:
            match = re.search(r"(\d{1,3})\s*%?", item)
            if match:
                confidence_score = max(0, min(100, int(match.group(1))))
                break

        code_debug = _normalize_items(payload.get("code_debug", [])) or ["No critical bugs were explicitly identified."]
        code_improvements = _normalize_items(payload.get("code_improvements", [])) or ["Apply incremental refactoring for readability and maintainability."]
        performance = _normalize_items(payload.get("performance", [])) or ["LLM output parsing incomplete - check raw output"]
        security = _normalize_items(payload.get("security", [])) or ["No immediate security issue was explicitly identified."]
        fix_suggestions = _normalize_items(payload.get("fix_suggestions", []))
        final_summary = _normalize_items(payload.get("final_summary", [])) or ["Prioritize correctness issues first, then refactor and optimize."]

        return {
            "code_debug": code_debug,
            "code_improvements": code_improvements,
            "performance": performance,
            "security": security,
            "fix_suggestions": fix_suggestions,
            "final_summary": final_summary,
            "confidence_score": confidence_score,
        }

    @staticmethod
    def _fallback_code_review_sections(code: str) -> dict[str, Any]:
        code_lower = code.lower()
        code_debug: list[str] = []
        improvements: list[str] = []
        performance: list[str] = []
        security: list[str] = []

        lines = code.splitlines()
        for idx, line in enumerate(lines, start=1):
            line_lower = line.lower()
            stripped = line.strip()
            if "except:" in line_lower:
                code_debug.append(f"Line {idx}: bare except hides real failures and complicates debugging.")
            if re.search(r"\bfor\s+\w+\s+in\s+range\(\s*n\s*\)\s*:", line_lower):
                code_debug.append(f"Line {idx}: full-range inner loop can cause index errors in adjacent access patterns.")
                improvements.append(f"Line {idx}: use range(n - i - 1) for bubble-sort style loops.")
                performance.append(f"Line {idx}: reduce unnecessary iterations by shrinking loop bounds.")
            if re.search(r"\[[^\]]*\+\s*1\]", line):
                code_debug.append(f"Line {idx}: adjacent index access may overflow at list boundaries.")
                improvements.append(f"Line {idx}: add bounds-safe loop limits before using index+1.")
            if "eval(" in line_lower:
                security.append(f"Line {idx}: eval() is unsafe with untrusted input.")
            if "exec(" in line_lower:
                security.append(f"Line {idx}: exec() is unsafe and should be avoided for dynamic execution.")
            if "open(" in line_lower and "encoding=" not in line_lower:
                improvements.append(f"Line {idx}: file open call should specify encoding for portability.")
            if "print(" in line_lower and not stripped.startswith("#"):
                improvements.append(f"Line {idx}: consider structured logging instead of print in core flow.")
            if ".append(" in line_lower and "for " in line_lower:
                performance.append(f"Line {idx}: list accumulation in loops may be optimized with generators.")
            if "requests.get(" in line_lower and "timeout=" not in line_lower:
                security.append(f"Line {idx}: network call without timeout can hang indefinitely.")
            if "subprocess." in line_lower and "shell=true" in line_lower:
                security.append(f"Line {idx}: subprocess with shell=True can introduce command-injection risk.")

        if "re.compile(" not in code_lower and "re.search(" in code_lower:
            performance.append("If regex is reused frequently, precompile it for efficiency.")

        if not code_debug:
            code_debug.append("No high-confidence critical bug detected by fallback scan; use smaller targeted review for deeper findings.")
        if not improvements:
            improvements.append("Improve function decomposition and naming consistency for maintainability.")
        if not performance:
            performance.append("No obvious hot-path inefficiency detected in fallback rule scan.")
        if not security:
            security.append("No high-confidence security issue detected in fallback rule scan.")

        final_summary = [
            "Fix correctness and security findings first, then apply maintainability and performance improvements."
        ]

        return {
            "code_debug": code_debug,
            "code_improvements": improvements,
            "performance": performance,
            "security": security,
            "final_summary": final_summary,
            "confidence_score": 72,
        }

    @staticmethod
    def _compose_code_review_text(sections: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("CODE DEBUG:")
        for item in sections.get("code_debug", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("CODE IMPROVEMENTS:")
        for item in sections.get("code_improvements", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("PERFORMANCE:")
        for item in sections.get("performance", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("SECURITY:")
        for item in sections.get("security", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("FIX SUGGESTIONS:")
        for item in sections.get("fix_suggestions", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("FINAL SUMMARY:")
        for item in sections.get("final_summary", []):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("CONFIDENCE:")
        lines.append(f"- {int(sections.get('confidence_score', 70))}%")
        return "\n".join(lines)

    @staticmethod
    def _normalize_review_sections(sections: dict[str, Any]) -> dict[str, Any]:
        """Deduplicates and trims section items to keep output concise and stable."""
        normalized = dict(sections)
        for key in ("code_debug", "code_improvements", "performance", "security", "fix_suggestions", "final_summary"):
            value = normalized.get(key, [])
            lines: list[str] = []
            if isinstance(value, list):
                lines = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                lines = [value.strip()]

            deduped: list[str] = []
            seen: set[str] = set()
            for line in lines:
                canon = re.sub(r"\s+", " ", line.lower()).strip()
                if canon in seen:
                    continue
                seen.add(canon)
                deduped.append(line)
            normalized[key] = deduped[:8]

        if not normalized.get("fix_suggestions"):
            normalized["fix_suggestions"] = normalized.get("code_improvements", [])[:3]
        return normalized

    @staticmethod
    def _extract_fixed_code(response_text: str) -> str:
        json_payload = RealityAnalyzer._try_parse_review_json(response_text)
        if json_payload is not None:
            # Best effort JSON extraction for fixed code.
            # Parse from raw JSON to preserve full multiline code snippets.
            match = re.search(r"\{[\s\S]*\}", response_text)
            raw_json = match.group(0) if match else response_text.strip()
            try:
                parsed = json.loads(raw_json)
                fixed_value = parsed.get("fixed_code")
                if isinstance(fixed_value, str) and fixed_value.strip():
                    return fixed_value.strip()
                if isinstance(fixed_value, list):
                    chunks: list[str] = []
                    for item in fixed_value:
                        if isinstance(item, dict):
                            code = str(item.get("code", "")).strip()
                            if code:
                                chunks.append(code)
                        else:
                            text_item = str(item).strip()
                            if text_item:
                                chunks.append(text_item)
                    if chunks:
                        return "\n\n".join(chunks).strip()
            except ValueError:
                pass

        fenced = re.search(
            r"FIXED CODE:\s*```(?:python)?\s*([\s\S]*?)```",
            response_text,
            flags=re.IGNORECASE,
        )
        if fenced:
            return fenced.group(1).strip()

        plain = re.search(
            r"(?is)(?:^|\n)\s*#*\s*\**\s*FIXED CODE\s*\**\s*:\s*([\s\S]*?)(?:\n\s*#*\s*\**\s*(?:FINAL SUMMARY|CONFIDENCE|CODE DEBUG|CODE IMPROVEMENTS|PERFORMANCE|SECURITY|FIX SUGGESTIONS)\s*\**\s*:|\Z)",
            response_text,
        )
        if not plain:
            return ""

        code = plain.group(1).strip()
        if code.startswith("```") and code.endswith("```"):
            code = re.sub(r"^```(?:python)?\s*", "", code)
            code = re.sub(r"\s*```$", "", code)
        return code.strip()

    @staticmethod
    def _is_weak_code_review_response(response_text: str) -> bool:
        text = response_text.lower()
        weak_markers = (
            "no obvious",
            "no critical bugs were explicitly identified",
            "cannot determine",
            "not enough context",
        )
        return any(marker in text for marker in weak_markers)

    @staticmethod
    def _clarification_response() -> str:
        return (
            "I need a bit more context to debug this properly.\n"
            "Tell me what happened, where it failed, and what outcome you expected."
        )

    @staticmethod
    def _sanitize_identity(text: str) -> str:
        if not text:
            return text
        sanitized = re.sub(
            r"\b(qwen|ollama|llama|mistral|model|provider|runtime)\b",
            "assistant",
            text,
            flags=re.IGNORECASE,
        )
        return sanitized.strip()

    @staticmethod
    def parse_section_items(text: str, section_header: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        in_section = False
        items: list[str] = []

        for line in lines:
            if line == section_header:
                in_section = True
                continue
            if line.startswith("[") and line.endswith("]") and in_section:
                break
            if not in_section:
                continue

            item = line[1:].strip() if line.startswith("-") else line.strip()
            if item:
                items.append(item)

        if items:
            return items

        fallback = [line for line in lines if not (line.startswith("[") and line.endswith("]"))]
        cleaned = []
        for line in fallback:
            item = line[1:].strip() if line.startswith("-") else line
            if item:
                cleaned.append(item)
        return cleaned[:6]

    @staticmethod
    def _build_final_insight(
        debug_result: list[str],
        improve_result: list[str],
        predict_result: list[str],
    ) -> list[str]:
        debug_focus = debug_result[0] if debug_result else "Identify and isolate the primary blocker."
        improve_focus = improve_result[0] if improve_result else "Execute one high-impact improvement first."
        predict_focus = predict_result[0] if predict_result else "Monitor risk trends over the next cycles."
        return [
            f"Prioritize this action now: {improve_focus}",
            f"Reasoning chain: {debug_focus} -> {predict_focus}",
        ]

    def _build_chat_prompt(self, user_input: str) -> str:
        return (
            "You are TRX-AI, a friendly and professional assistant.\n"
            "Rules:\n"
            "- Speak naturally and helpfully.\n"
            "- Never mention internal models, runtimes, or infrastructure.\n"
            "- Keep responses concise and human-like.\n\n"
            f"User: {user_input}\n"
            "Assistant:"
        )

    def _build_debug_agent_prompt(self, user_input: str, context: list[dict[str, Any]]) -> str:
        compact_context = []
        for item in context[-self.config.context_window_size:]:
            compact_context.append(f"{item.get('mode', 'debug')}: {str(item.get('input', ''))[:100]}")
        context_line = " || ".join(compact_context) if compact_context else "none"

        return (
            "You are the Debug Agent in TRX-AI.\n"
            "Goal: find root causes and concrete problems.\n"
            f"Session Context: {context_line}\n"
            f"Input: {user_input}\n"
            "Output only:\n"
            "[DEBUG ANALYSIS]\n"
            "- ..."
        )

    def _build_improve_agent_prompt(self, user_input: str, debug_result: list[str]) -> str:
        debug_text = " | ".join(debug_result[:5]) if debug_result else "No debug findings"
        return (
            "You are the Improve Agent in TRX-AI.\n"
            "Goal: suggest actionable improvements and optimization strategies.\n"
            f"Input: {user_input}\n"
            f"Debug Findings: {debug_text}\n"
            "Output only:\n"
            "[IMPROVEMENTS]\n"
            "- ..."
        )

    def _build_predict_agent_prompt(
        self,
        user_input: str,
        debug_result: list[str],
        improve_result: list[str],
    ) -> str:
        debug_text = " | ".join(debug_result[:4]) if debug_result else "No debug findings"
        improve_text = " | ".join(improve_result[:4]) if improve_result else "No improvements"
        return (
            "You are the Predict Agent in TRX-AI.\n"
            "Goal: predict near-term outcomes and risk trajectory.\n"
            f"Input: {user_input}\n"
            f"Debug Findings: {debug_text}\n"
            f"Improvements: {improve_text}\n"
            "Output only:\n"
            "[PREDICTIONS]\n"
            "- ..."
        )

    def _build_graph_context(self, code: str) -> dict[str, Any]:
        files = self._extract_file_hints(code)
        symbols = self._extract_symbol_hints(code)
        summary = self._graph_adapter.summarize_context(files=files, symbols=symbols, max_items=5)
        summary["files"] = files
        summary["symbols"] = symbols
        return summary

    @staticmethod
    def _extract_file_hints(code: str) -> list[str]:
        matches = re.findall(r"^# FILE:\s*(.+)$", code, flags=re.MULTILINE)
        files = [item.strip() for item in matches if item.strip()]
        if files:
            return files[:8]
        return []

    @staticmethod
    def _extract_symbol_hints(code: str) -> list[str]:
        patterns = [
            r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b",
            r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        ]
        out: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for name in re.findall(pattern, code, flags=re.MULTILINE):
                key = str(name).strip()
                if not key or key in seen:
                    continue
                out.append(key)
                seen.add(key)
                if len(out) >= 10:
                    return out
        return out

    @staticmethod
    def _graph_context_text(graph_context: dict[str, Any]) -> str:
        if not graph_context:
            return "unavailable"
        lines: list[str] = []
        for title, key in (
            ("Dependency Chains", "dependencies"),
            ("Impacted Modules", "impacted_modules"),
            ("Call Relationships", "call_relationships"),
        ):
            values = graph_context.get(key, [])
            if isinstance(values, list) and values:
                lines.append(f"{title}:")
                for item in values[:8]:
                    lines.append(f"- {item}")
        if graph_context.get("degraded"):
            lines.append("MCP Graph Mode: degraded")
        else:
            lines.append("MCP Graph Mode: active")
        if not lines:
            return "unavailable"
        return "\n".join(lines)

    def _attach_graph_insights(self, result: dict[str, Any], pipeline: dict[str, Any]) -> None:
        graph_context = pipeline.get("graph_context", {})
        if not isinstance(graph_context, dict):
            return
        tags = result.get("system_status", [])
        if not isinstance(tags, list):
            tags = []
        degraded = bool(graph_context.get("degraded"))
        tags.append("mcp_query=degraded" if degraded else "mcp_query=active")
        elapsed_ms = int(graph_context.get("elapsed_ms", 0))
        if elapsed_ms > 0:
            tags.append(f"mcp_elapsed_ms={elapsed_ms}")
        result["system_status"] = self._dedupe([str(tag) for tag in tags])

        insights: list[str] = []
        for key, label in (
            ("dependencies", "Dependency chain"),
            ("impacted_modules", "Impacted module"),
            ("call_relationships", "Call relation"),
        ):
            values = graph_context.get(key, [])
            if isinstance(values, list):
                for item in values[:3]:
                    insights.append(f"{label}: {item}")
        errors = graph_context.get("errors", [])
        if degraded and isinstance(errors, list) and errors:
            insights.append(f"MCP degraded: {errors[0]}")

        existing_debug = result.get("debug_analysis", [])
        if isinstance(existing_debug, list):
            result["debug_analysis"] = self._dedupe(existing_debug + insights)
        else:
            result["debug_analysis"] = insights

    def _cache_key(self, user_input: str, mode: str, context: list[dict[str, Any]]) -> str:
        context_slice = context[-self.config.context_window_size:]
        raw = f"{mode}|{user_input.lower().strip()}|{context_slice}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if key not in self._response_cache:
            METRICS.inc("cache_miss")
            return None
        value = self._response_cache.pop(key)
        created_at = float(value.get("_cache_created_at", 0.0))
        ttl_seconds = int(value.get("_cache_ttl_seconds", self.config.cache_ttl_seconds))
        age_seconds = max(0.0, time.time() - created_at) if created_at > 0 else 0.0
        if created_at > 0 and age_seconds > ttl_seconds:
            METRICS.inc("cache_miss")
            return None
        self._response_cache[key] = value
        METRICS.inc("cache_hit")
        out = copy.deepcopy(value)
        statuses = out.get("system_status", [])
        if isinstance(statuses, list):
            if "cache_hit" not in statuses:
                statuses.append("cache_hit")
            out["system_status"] = statuses
        if self.config.debug_cache:
            ttl_remaining = max(0, int(ttl_seconds - age_seconds))
            out["cache_debug"] = {
                "status": "HIT",
                "ttl_remaining_s": ttl_remaining,
                "fingerprint": str(value.get("_cache_fingerprint", key[:12])),
            }
        return out

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        if key in self._response_cache:
            self._response_cache.pop(key)
        store = copy.deepcopy(value)
        store["_cache_created_at"] = time.time()
        store["_cache_ttl_seconds"] = int(self.config.cache_ttl_seconds)
        store["_cache_fingerprint"] = key[:12]
        self._response_cache[key] = store
        while len(self._response_cache) > self.config.cache_size:
            self._response_cache.popitem(last=False)

    def _review_cache_key(self, code: str, language: str) -> str:
        raw = f"{language}|{self.config.local_llm_model}|{code.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _review_cache_get(self, key: str) -> dict[str, Any] | None:
        if key not in self._review_response_cache:
            return None
        value = self._review_response_cache.pop(key)
        self._review_response_cache[key] = value
        return value

    def _review_cache_set(self, key: str, value: dict[str, Any]) -> None:
        if key in self._review_response_cache:
            self._review_response_cache.pop(key)
        self._review_response_cache[key] = copy.deepcopy(value)
        while len(self._review_response_cache) > max(20, self.config.cache_size // 2):
            self._review_response_cache.popitem(last=False)
        self._save_review_cache_disk()

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            value = str(item).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
        return out

    def _mcp_status_tags(self) -> list[str]:
        status = detect_mcp_graph_status()
        tags = [f"mcp_graph={status.state}"]
        if status.exists and status.size_bytes > 0:
            tags.append("mcp_graph_available")
        else:
            tags.append("mcp_graph_unavailable")
        return tags

    def _load_review_cache_disk(self) -> None:
        if not self.config.review_cache_disk_enabled:
            return
        try:
            if not self._review_cache_disk_path.exists():
                return
            payload = json.loads(self._review_cache_disk_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            items = payload.get("items", [])
            if not isinstance(items, list):
                return
            for item in items[-max(20, self.config.cache_size // 2):]:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key", "")).strip()
                value = item.get("value")
                if key and isinstance(value, dict):
                    self._review_response_cache[key] = value
        except (OSError, ValueError, TypeError):
            return

    def _save_review_cache_disk(self) -> None:
        if not self.config.review_cache_disk_enabled:
            return
        try:
            self._review_cache_disk_path.parent.mkdir(parents=True, exist_ok=True)
            max_entries = max(20, self.config.review_cache_disk_max_entries)
            items = [
                {"key": key, "value": value}
                for key, value in list(self._review_response_cache.items())[-max_entries:]
            ]
            payload = {"version": 1, "items": items}
            self._review_cache_disk_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            return

    def clear_caches(self) -> None:
        self._response_cache.clear()
        self._review_response_cache.clear()
        if self.config.review_cache_disk_enabled and self._review_cache_disk_path.exists():
            try:
                self._review_cache_disk_path.unlink()
            except OSError:
                pass
        if hasattr(self, "_graph_adapter") and self._graph_adapter is not None:
            try:
                self._graph_adapter.clear_cache()
            except Exception:
                pass

    def runtime_status(self) -> dict[str, Any]:
        graph = detect_mcp_graph_status()
        transport = "LOCAL"
        try:
            transport = self._graph_adapter.transport_label()
        except Exception:
            pass
        breaker_state = str(METRICS.snapshot().get("states", {}).get("circuit_breaker_state", "CLOSED"))
        return {
            "assistant_mode": str(getattr(self.config, "assistant_mode", "auto")).upper(),
            "graph_transport": transport,
            "mcp_active": graph.exists and graph.state == "ready",
            "cache_enabled": bool(self.config.review_cache_disk_enabled),
            "llm_connected": bool(self.config.use_local_llm),
            "cache_ttl_seconds": int(self.config.cache_ttl_seconds),
            "circuit_breaker_state": breaker_state,
        }

    def apply_feedback(self, text: str, correct_route: str) -> None:
        normalized_route = str(correct_route).strip().lower()
        if normalized_route not in {"code", "general"}:
            raise ValueError("correct_route must be 'code' or 'general'")
        self._router_memory.record(str(text), normalized_route)
