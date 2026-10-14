"""Persistent adaptive routing memory for TRX."""

from __future__ import annotations

from guardrails import redact
import json
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


class RouterMemory:
    def __init__(self, path: str = "sessions/router_memory.json") -> None:
        self.path = Path(path)
        self.stats: dict[str, dict[str, int]] = defaultdict(lambda: {"code": 0, "general": 0})
        self._load()

    def record(self, text: str, route: str) -> None:
        if not str(text).strip():
            return
        key = self._key(text)
        target = "code" if route == "code" else "general"
        self.stats[key][target] = int(self.stats[key].get(target, 0)) + 1
        self._save()

    def get_bias(self, text: str) -> str | None:
        key = self._key(text)
        data = self.stats.get(key)
        if not data:
            return None
        return "code" if int(data.get("code", 0)) > int(data.get("general", 0)) else "general"

    def _load(self) -> None:
        try:
            if not self.path.exists():
                return
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            for key, value in payload.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                safe_key = key if re.fullmatch(r"[0-9a-f]{64}", key) else self._key(key)
                self.stats[safe_key] = {
                    "code": int(value.get("code", 0)),
                    "general": int(value.get("general", 0)),
                }
        except (OSError, ValueError, TypeError):
            return

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serializable: dict[str, Any] = dict(self.stats)
            self.path.write_text(json.dumps(redact(serializable), ensure_ascii=False), encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(str(text).lower().strip().encode("utf-8")).hexdigest()

