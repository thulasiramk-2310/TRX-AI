"""Persistent long-term memory for TRX runtime preferences and patterns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LongTermMemory:
    def __init__(self, path: str = "sessions/long_term_memory.json") -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.data, dict):
                self.data = {}
        except (OSError, ValueError):
            self.data = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def store(self, key: str, value: Any) -> None:
        self.data[str(key)] = value
        self._save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(str(key), default)

    def store_user_value(self, username: str, key: str, value: Any) -> None:
        users = self.data.get("users", {})
        if not isinstance(users, dict):
            users = {}
        record = users.get(username, {})
        if not isinstance(record, dict):
            record = {}
        record[str(key)] = value
        users[username] = record
        self.data["users"] = users
        self._save()

    def get_user_value(self, username: str, key: str, default: Any = None) -> Any:
        users = self.data.get("users", {})
        if not isinstance(users, dict):
            return default
        record = users.get(username, {})
        if not isinstance(record, dict):
            return default
        return record.get(str(key), default)

