"""Semantic intent routing with optional sentence-transformers backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    def __init__(self) -> None:
        self._model: Any = None
        self._util: Any = None
        self._code_embeddings: Any = None
        self._general_embeddings: Any = None
        self._available = False
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore

            model = SentenceTransformer("all-MiniLM-L6-v2")
            self._model = model
            self._util = util
            self._code_embeddings = model.encode(CODE_EXAMPLES, convert_to_tensor=True)
            self._general_embeddings = model.encode(GENERAL_EXAMPLES, convert_to_tensor=True)
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def semantic_intent(self, text: str) -> SemanticIntentResult | None:
        if not self._available:
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

