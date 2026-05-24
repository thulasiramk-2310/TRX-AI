"""Optional Supabase-backed cloud memory sync for TRX."""

from __future__ import annotations


try:
    from supabase import create_client  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    create_client = None


class CloudMemorySync:
    def __init__(
        self,
        *,
        enabled: bool = False,
        url: str = "",
        key: str = "",
        table: str = "memory",
    ) -> None:
        self.enabled = bool(enabled)
        self.url = str(url or "").strip()
        self.key = str(key or "").strip()
        self.table = str(table or "memory").strip()
        self.client = None
        if self.enabled and self.url and self.key and create_client is not None:
            try:
                self.client = create_client(self.url, self.key)
            except Exception:
                self.client = None

    def is_ready(self) -> bool:
        return bool(self.enabled and self.client is not None)

    def sync_memory(self, user: str, text: str, *, personality: str = "friendly") -> bool:
        if not self.is_ready():
            return False
        try:
            self.client.table(self.table).insert(
                {
                    "user": str(user or "guest"),
                    "text": str(text or ""),
                    "personality": str(personality or "friendly"),
                }
            ).execute()
            return True
        except Exception:
            return False

    def fetch_memory(self, user: str, *, limit: int = 5) -> list[str]:
        if not self.is_ready():
            return []
        try:
            res = self.client.table(self.table).select("*").eq("user", str(user or "guest")).limit(max(1, int(limit))).execute()
            rows = getattr(res, "data", []) or []
            out = [str(item.get("text", "")).strip() for item in rows if isinstance(item, dict) and str(item.get("text", "")).strip()]
            return out
        except Exception:
            return []

