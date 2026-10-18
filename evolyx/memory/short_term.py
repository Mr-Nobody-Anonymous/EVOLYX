"""Short-term ephemeral working memory buffer."""

from typing import List
from evolyx.core.schemas import Message


class ShortTermMemory:
    """In-memory sliding window buffer for active working state."""

    def __init__(self, capacity: int = 15):
        self.capacity = capacity
        self._buffer: List[Message] = []

    def push(self, message: Message) -> None:
        self._buffer.append(message)
        if len(self._buffer) > self.capacity:
            self._buffer.pop(0)

    def get_recent(self, n: int = 5) -> List[Message]:
        return self._buffer[-n:]

    def clear(self) -> None:
        self._buffer.clear()

    def get_all(self) -> List[Message]:
        return list(self._buffer)
