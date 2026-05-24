"""Persistent multi-user profile manager for TRX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class UserProfiles:
    def __init__(self, path: str = "sessions/users.json") -> None:
        self.path = Path(path)
        self.users: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    users = payload.get("users", payload)
                    if isinstance(users, dict):
                        self.users = {str(k).lower(): v for k, v in users.items() if isinstance(v, dict)}
                        return
        except (OSError, ValueError):
            pass
        self.users = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"users": self.users}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_users(self) -> list[str]:
        return sorted(self.users.keys())

    def get_user(self, username: str) -> dict[str, Any]:
        key = str(username or "").strip().lower()
        user = self.users.get(key, {})
        return dict(user) if isinstance(user, dict) else {}

    def set_user(self, username: str, data: dict[str, Any]) -> None:
        key = str(username or "").strip().lower()
        if not key:
            raise ValueError("username cannot be empty")
        self.users[key] = dict(data or {})
        self._save()

    def create_user(self, username: str, *, personality: str = "friendly") -> dict[str, Any]:
        key = str(username or "").strip().lower()
        if not key:
            raise ValueError("username cannot be empty")
        if key in self.users:
            raise ValueError("user already exists")
        profile = {
            "personality": str(personality).strip().lower() or "friendly",
            "preferences": {},
        }
        self.users[key] = profile
        self._save()
        return dict(profile)

