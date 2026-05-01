"""Short conversation memory for contextual general Q&A prompts."""

from __future__ import annotations


class ConversationMemory:
    def __init__(self, max_turns: int = 5) -> None:
        self.history: list[tuple[str, str]] = []
        self.max_turns = max(1, int(max_turns))

    def add(self, user: str, response: str) -> None:
        self.history.append((str(user), str(response)))
        self.history = self.history[-self.max_turns :]

    def get_context(self) -> str:
        chunks: list[str] = []
        for user, response in self.history:
            chunks.append(f"User: {user}\nAI: {response}")
        return "\n".join(chunks)

