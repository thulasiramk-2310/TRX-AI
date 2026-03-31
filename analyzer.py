"""Core analysis engine for TRX-AI with local Ollama inference and rule fallback."""

from __future__ import annotations

import hashlib
import json
import sys
import re
import time
from collections import OrderedDict
from typing import Any

import requests

from config import AppConfig


TRX_GREETING = "Hey! I'm TRX-AI - your personal reasoning assistant. How can I help you today?"


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
    for _ in range(max(1, retries)):
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
                }
        except requests.RequestException:
            continue

    return {"ok": False, "text": "", "status_code": last_status}


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

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._response_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._agent_flags: dict[str, bool] = {
            "debug": True,
            "improve": True,
            "predict": True,
        }

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

        intent_data = self.detect_intent_hybrid(cleaned)
        intent = intent_data.get("intent", "chat")
        source = intent_data.get("source", "rule")
        confidence = float(intent_data.get("confidence", 0.5))

        if self.config.dev_mode:
            print(f"[Intent: {intent} | Source: {source} | Confidence: {confidence:.2f}]")

        if intent == "command":
            return {
                "response_mode": "chat",
                "chat_response": "Command detected. Use CLI command handling for this action.",
                "intent": intent,
                "intent_source": source,
                "intent_confidence": confidence,
            }

        if intent == "greeting":
            return {
                "response_mode": "chat",
                "chat_response": TRX_GREETING,
                "intent": intent,
                "intent_source": source,
                "intent_confidence": confidence,
            }

        if intent == "vague":
            return {
                "response_mode": "chat",
                "chat_response": self._clarification_response(),
                "intent": intent,
                "intent_source": source,
                "intent_confidence": confidence,
            }

        context = past_context or []
        cache_key = self._cache_key(cleaned, mode, context)
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        if intent == "problem":
            result = self._handle_problem(cleaned, mode, context)
        else:
            result = self._handle_chat(cleaned)

        result["intent"] = intent
        result["intent_source"] = source
        result["intent_confidence"] = confidence
        self._cache_set(cache_key, result)
        return result

    def detect_intent_hybrid(self, user_input: str) -> dict[str, Any]:
        """Priority order: command -> greeting -> problem -> vague -> chat."""
        normalized = re.sub(r"\s+", " ", user_input.strip().lower())
        if not normalized:
            return {"intent": "vague", "confidence": 1.0, "source": "rule"}

        if self._is_command(normalized):
            return {"intent": "command", "confidence": 1.0, "source": "rule"}

        if self._is_greeting(normalized):
            return {"intent": "greeting", "confidence": 1.0, "source": "rule"}

        if self._is_strong_problem_signal(normalized):
            return {"intent": "problem", "confidence": 0.85, "source": "rule"}

        if self._is_vague(normalized):
            return {"intent": "vague", "confidence": 0.75, "source": "rule"}

        llm_result = self._classify_intent_with_llm(normalized)
        if llm_result is not None:
            return llm_result

        return {"intent": "chat", "confidence": 0.5, "source": "rule"}

    def _is_command(self, normalized: str) -> bool:
        if normalized in {"help", "exit", "history"}:
            return True

        return (
            normalized.startswith("export ")
            or normalized.startswith("agents ")
            or normalized.startswith("mode ")
            or normalized.startswith("save ")
        )

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

    def _classify_intent_with_llm(self, user_input: str) -> dict[str, Any] | None:
        prompt = (
            "Classify intent for a CLI assistant.\n"
            "Allowed labels: greeting, chat, problem, vague, command\n"
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
        if intent not in {"greeting", "chat", "problem", "vague", "command"}:
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

    def _call_local(self, prompt: str, max_new_tokens: int | None = None) -> dict[str, Any]:
        if not self.config.use_local_llm:
            return {"ok": False, "text": "", "status_code": None}

        return call_local_llm(
            prompt,
            url=self.config.local_llm_url,
            model=self.config.local_llm_model,
            timeout=self.config.local_llm_timeout_seconds,
            max_new_tokens=(max_new_tokens if max_new_tokens is not None else self.config.local_llm_max_new_tokens),
            temperature=self.config.local_llm_temperature,
            retries=self.config.local_llm_retries,
        )

    def analyze_code_multi_agent(self, code: str) -> dict[str, Any]:
        """Runs structured multi-agent style code review on provided code content."""
        cleaned_code = code.strip()
        if not cleaned_code:
            raise ValueError("Code content is empty.")

        # Guard against oversized payloads that can degrade model quality.
        code_for_prompt = cleaned_code[:2500] if len(cleaned_code) > 2500 else cleaned_code
        prompt = self._build_code_review_prompt(code_for_prompt)
        print("[Code Review -> Using LLM]")
        started = time.perf_counter()
        response = self._call_local(prompt, max_new_tokens=600)
        elapsed = time.perf_counter() - started
        print(f"[LLM time: {elapsed:.2f}s]")

        if not response.get("ok"):
            return {
                "response_mode": "analysis",
                "analysis_text": "Error: LLM not responding",
                "system_status": ["code_review_failed"],
                "confidence": 0.0,
                "confidence_score": 0,
                "debug_analysis": ["Error: LLM not responding"],
                "improvements": ["Check local LLM service availability and retry review."],
                "predictions": ["No performance analysis available because LLM review failed."],
                "final_insight": ["Restore LLM connectivity first, then rerun code review."],
                "fixed_code": "",
                "analysis_source": "llm",
                "intent": "review",
                "intent_source": "llm",
            }

        response_text = str(response.get("text", ""))
        print("[RAW LLM OUTPUT]")
        self._debug_print_safe(response_text)
        if "no high-confidence" in response_text.lower():
            response_text += "\n\nNOTE: Response may be generic due to weak prompt or model limitations."
        fixed_code = self._extract_fixed_code(response_text)
        sections = self._parse_code_review_sections(response_text)
        truncated = str(response.get("done_reason", "")).lower() == "length"
        if not fixed_code:
            fallback_fixed = self._generate_fixed_code_only(cleaned_code)
            if fallback_fixed:
                fixed_code = fallback_fixed

        confidence_score = int(sections.get("confidence_score", 70))
        analysis_text = self._compose_code_review_text(sections)

        status = ["intent=review", "source=llm"]
        if len(cleaned_code) > 2500:
            status.append("code_truncated_for_review")
        if truncated:
            status.append("llm_output_truncated")
        if fixed_code:
            status.append("fixed_code_ready")
        else:
            status.append("fixed_code_missing")

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
            "analysis_source": "llm",
            "intent": "review",
            "intent_source": "llm",
        }

    def _generate_fixed_code_only(self, original_code: str) -> str:
        code_for_prompt = original_code[:2200] if len(original_code) > 2200 else original_code
        prompt = (
            "You are a senior Python engineer.\n"
            "Return a fully corrected version of this code.\n"
            "Rules:\n"
            "- Return only Python code\n"
            "- No markdown, no explanations\n"
            "- Preserve intent while fixing bugs, edge cases, and robustness\n\n"
            f"{code_for_prompt}\n"
        )
        response = self._call_local(prompt, max_new_tokens=900)
        if not response.get("ok"):
            return ""
        text = str(response.get("text", "")).strip()
        if not text:
            return ""

        fenced = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        cleaned = re.sub(r"^\s*```(?:python)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

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

    @staticmethod
    def _build_code_review_prompt(code: str) -> str:
        return f"""
You are a senior Python software engineer performing a deep and practical code review.

Analyze the following code carefully and provide actionable insights.

CODE:
{code}

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
Return full corrected version of the code.

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
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
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
    def _extract_fixed_code(response_text: str) -> str:
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

    def _cache_key(self, user_input: str, mode: str, context: list[dict[str, Any]]) -> str:
        context_slice = context[-self.config.context_window_size:]
        raw = f"{mode}|{user_input.lower().strip()}|{context_slice}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if key not in self._response_cache:
            return None
        value = self._response_cache.pop(key)
        self._response_cache[key] = value
        return value

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        if key in self._response_cache:
            self._response_cache.pop(key)
        self._response_cache[key] = value
        while len(self._response_cache) > self.config.cache_size:
            self._response_cache.popitem(last=False)
