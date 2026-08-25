"""Short conversation memory for contextual general Q&A prompts."""

from __future__ import annotations


class ConversationMemory:
    def __init__(self, max_turns: int = 5) -> None:
        self.memories: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self.max_turns = max(1, int(max_turns))

    @staticmethod
    def _key(personality: str, owner: str) -> tuple[str, str]:
        return (str(owner or "guest").strip().lower() or "guest",
                str(personality or "friendly").strip().lower() or "friendly")

    def add(self, personality: str, user: str, response: str, *, owner: str = "guest") -> None:
        key = self._key(personality, owner)
        if key not in self.memories:
            self.memories[key] = []
        self.memories[key].append((str(user), str(response)))
        self.memories[key] = self.memories[key][-self.max_turns :]

    def get_context(self, personality: str, *, owner: str = "guest") -> str:
        key = self._key(personality, owner)
        history = self.memories.get(key, [])
        chunks: list[str] = []
        for user, response in history:
            chunks.append(f"User: {user}\nAI: {response}")
        return "\n".join(chunks)
