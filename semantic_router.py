"""Semantic intent routing with optional sentence-transformers backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import embeddings


@dataclass(frozen=True)
class SemanticIntentResult:
    intent: str
    confidence: float
    source: str


CODE_EXAMPLES = [
    "fix bug in code",
    "debug function",
    "review file",
    "optimize algorithm",
    "make login faster",
    "improve performance of api",
]

GENERAL_EXAMPLES = [
    "what is data warehouse",
    "explain networking",
    "how does database work",
    "why is caching used",
    "difference between db and dw",
]


class SemanticRouter:
    """Embedding-based intent signal. Never blocks start-up: the shared model loads in the
    background, and until it is ready `semantic_intent` returns None (rule routing still works)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._util: Any = None
        self._code_embeddings: Any = None
        self._general_embeddings: Any = None
        self._available = False
        embeddings.warm_up()

    def _init_backend(self) -> bool:
        if self._available:
            return True
        model = embeddings.ready()
        if model is None:
            return False
        try:
            from sentence_transformers import util  # type: ignore

            self._util = util
            self._code_embeddings = model.encode(CODE_EXAMPLES, convert_to_tensor=True)
            self._general_embeddings = model.encode(GENERAL_EXAMPLES, convert_to_tensor=True)
            self._model = model
            self._available = True
        except Exception:
            self._available = False
        return self._available

    @property
    def available(self) -> bool:
        return self._init_backend()

    def semantic_intent(self, text: str) -> SemanticIntentResult | None:
        if not self._init_backend():
            return None
        try:
            emb = self._model.encode(text, convert_to_tensor=True)
            code_score = float(self._util.cos_sim(emb, self._code_embeddings).max().item())
            general_score = float(self._util.cos_sim(emb, self._general_embeddings).max().item())
            if code_score > general_score:
                return SemanticIntentResult(intent="code", confidence=code_score, source="semantic")
            return SemanticIntentResult(intent="general", confidence=general_score, source="semantic")
        except Exception:
            return None

