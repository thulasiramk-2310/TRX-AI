"""Persistent adaptive routing memory for TRX."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class RouterMemory:
    def __init__(self, path: str = "sessions/router_memory.json") -> None:
        self.path = Path(path)
        self.stats: dict[str, dict[str, int]] = defaultdict(lambda: {"code": 0, "general": 0})
        self._load()

    def record(self, text: str, route: str) -> None:
        key = str(text).lower().strip()
        if not key:
            return
        target = "code" if route == "code" else "general"
        self.stats[key][target] = int(self.stats[key].get(target, 0)) + 1
        self._save()

    def get_bias(self, text: str) -> str | None:
        key = str(text).lower().strip()
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
                self.stats[key] = {
                    "code": int(value.get("code", 0)),
                    "general": int(value.get("general", 0)),
                }
        except (OSError, ValueError, TypeError):
            return

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            serializable: dict[str, Any] = dict(self.stats)
            self.path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return

