"""Semantic scoring helpers for TRX-AI evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


DEFAULT_SYNONYMS: dict[str, set[str]] = {
    "array": {"list", "vector"},
    "list": {"array", "vector"},
    "index": {"bounds", "out-of-range", "outofrange"},
    "error": {"exception", "failure", "crash"},
    "timeout": {"time out"},
    "iterative": {"loop", "dynamic-programming", "dp", "memoization"},
    "memoization": {"cache", "dynamic-programming", "dp"},
    "empty": {"null", "none"},
    "sql": {"database", "query"},
    "injection": {"unsafe", "untrusted"},
    "parameterized": {"prepared", "bind"},
    "missing": {"absent", "not"},
    "validation": {"guard", "check", "verify"},
    "check": {"guard", "validate", "validation"},
    "guard": {"check", "validation", "clause"},
    "clause": {"guard", "condition"},
}


# Connective words carry no signal in short labels like "use a set" or "check for empty input".
LABEL_STOPWORDS = {"a", "an", "the", "or", "and", "for", "to", "of", "on", "in", "with", "is", "be"}


def _normalize_text(text: str) -> str:
    # Split CamelCase identifiers (IndexError -> Index Error) before lowercasing.
    lowered = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()
    cleaned = re.sub(r"[^a-z0-9\s_\-]+", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _light_stem(token: str) -> str:
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


@dataclass(slots=True)
class SemanticMatcher:
    """Lightweight semantic scorer for benchmark signals."""

    synonyms: dict[str, set[str]] = field(default_factory=lambda: DEFAULT_SYNONYMS)
    max_window_tokens: int = 48

    def _tokenize(self, text: str) -> list[str]:
        return [_light_stem(part) for part in _normalize_text(text).split() if part]

    def _expand_tokens(self, tokens: list[str]) -> set[str]:
        expanded = set(tokens)
        for token in list(tokens):
            expanded.update(_light_stem(item) for item in self.synonyms.get(token, set()))
        return expanded

    @staticmethod
    def _bigrams(tokens: list[str]) -> set[str]:
        return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}

    def score(self, expected: str, actual: str) -> float:
        expected_norm = _normalize_text(expected)
        actual_norm = _normalize_text(actual)
        if not expected_norm or not actual_norm:
            return 0.0
        if expected_norm in actual_norm:
            return 1.0

        expected_tokens = self._tokenize(expected_norm)
        actual_tokens = self._tokenize(actual_norm)
        if not expected_tokens or not actual_tokens:
            return 0.0

        expected_bigrams = self._bigrams(expected_tokens)
        actual_text = " ".join(actual_tokens)

        # Phrase-level fuzzy match against bounded windows to avoid dilution on long text.
        window = max(self.max_window_tokens, len(expected_tokens) * 2)
        best_fuzzy = 0.0
        if len(actual_tokens) <= window:
            best_fuzzy = SequenceMatcher(None, expected_norm, actual_text).ratio()
        else:
            step = max(1, window // 3)
            for start in range(0, len(actual_tokens), step):
                chunk = " ".join(actual_tokens[start : start + window])
                if not chunk:
                    continue
                best_fuzzy = max(best_fuzzy, SequenceMatcher(None, expected_norm, chunk).ratio())
                if best_fuzzy >= 0.96:
                    break

        # Each expected content token counts once, satisfied by itself or any synonym.
        # (Synonyms must not inflate the denominator, or richer labels score lower.)
        actual_set = set(actual_tokens)
        content_tokens = [t for t in dict.fromkeys(expected_tokens) if t not in LABEL_STOPWORDS] or expected_tokens
        matched = sum(1 for t in content_tokens if t in actual_set or self._expand_tokens([t]) & actual_set)
        overlap = matched / len(content_tokens)
        actual_bigrams = self._bigrams(actual_tokens)
        bigram_overlap = len(expected_bigrams.intersection(actual_bigrams)) / max(1, len(expected_bigrams))

        # Weighted blend optimized for short expected labels and long model outputs.
        score = (0.5 * overlap) + (0.3 * best_fuzzy) + (0.2 * bigram_overlap)
        return max(0.0, min(1.0, score))
