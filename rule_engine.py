"""Deterministic reasoning layer used when the LLM is unavailable."""

from __future__ import annotations

import re
from typing import Any


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
