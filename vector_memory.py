"""Semantic vector memory with graceful fallbacks and persistence."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class VectorMemory:
    def __init__(
        self,
        *,
        enabled: bool = True,
        path: str = "sessions/vector_memory.json",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path)
        self.embedding_model = embedding_model
        self.entries: list[dict[str, Any]] = []
        self._model = None
        self._index = None
        self._dim = 384
        self._load()
        self._bootstrap_index()

    def _load(self) -> None:
        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                rows = payload.get("entries", [])
                if isinstance(rows, list):
                    self.entries = [item for item in rows if isinstance(item, dict) and str(item.get("text", "")).strip()]
        except (OSError, ValueError):
            self.entries = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "entries": self.entries[-2000:]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _bootstrap_index(self) -> None:
        if not self.enabled or SentenceTransformer is None or faiss is None:
            return
        try:
            self._model = SentenceTransformer(self.embedding_model)
            self._index = faiss.IndexFlatL2(self._dim)
            texts = [str(item.get("text", "")) for item in self.entries]
            if texts:
                emb = self._model.encode(texts)
                self._index.add(np.array(emb).astype("float32"))
        except Exception:
            self._model = None
            self._index = None

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        lowered = str(text or "").lower()
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        return [tok for tok in cleaned.split() if tok]

    def _fallback_rank(self, query: str, pool: list[dict[str, Any]], k: int) -> list[str]:
        q_tokens = Counter(self._normalize_tokens(query))
        scored: list[tuple[float, str]] = []
        for item in pool:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            t_tokens = Counter(self._normalize_tokens(text))
            if not q_tokens or not t_tokens:
                score = 0.0
            else:
                dot = sum(q_tokens[t] * t_tokens[t] for t in set(q_tokens) | set(t_tokens))
                q_norm = math.sqrt(sum(v * v for v in q_tokens.values()))
                t_norm = math.sqrt(sum(v * v for v in t_tokens.values()))
                score = (dot / (q_norm * t_norm)) if q_norm and t_norm else 0.0
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [text for score, text in scored if score > 0.0][: max(1, int(k))]
        return out

    def add(self, text: str, *, user: str = "guest", personality: str = "friendly") -> None:
        value = str(text or "").strip()
        if not self.enabled or not value:
            return
        record = {
            "text": value,
            "user": str(user or "guest").strip().lower() or "guest",
            "personality": str(personality or "friendly").strip().lower() or "friendly",
        }
        self.entries.append(record)
        self.entries = self.entries[-2000:]
        self._save()

        if self._index is not None and self._model is not None:
            try:
                emb = self._model.encode([value])
                self._index.add(np.array(emb).astype("float32"))
            except Exception:
                pass

    def search(
        self,
        query: str,
        *,
        k: int = 3,
        user: str | None = None,
        personality: str | None = None,
    ) -> list[str]:
        if not self.enabled:
            return []

        user_key = (str(user).strip().lower() if user else "")
        personality_key = (str(personality).strip().lower() if personality else "")
        pool = [
            item
            for item in self.entries
            if (not user_key or str(item.get("user", "")).lower() == user_key)
            and (not personality_key or str(item.get("personality", "")).lower() == personality_key)
        ]
        if not pool:
            return []

        # If semantic dependencies are available, use embeddings.
        if self._index is not None and self._model is not None:
            try:
                # Build pool-local mapping by re-encoding pool for accuracy under filtering.
                pool_texts = [str(item.get("text", "")) for item in pool]
                pool_emb = self._model.encode(pool_texts)
                pool_index = faiss.IndexFlatL2(self._dim)
                pool_index.add(np.array(pool_emb).astype("float32"))
                q_emb = self._model.encode([str(query or "")])
                _, idx = pool_index.search(np.array(q_emb).astype("float32"), max(1, int(k)))
                out: list[str] = []
                for i in idx[0]:
                    if 0 <= int(i) < len(pool_texts):
                        out.append(pool_texts[int(i)])
                return out
            except Exception:
                pass

        # Graceful fallback when embeddings are unavailable.
        return self._fallback_rank(query, pool, k)

