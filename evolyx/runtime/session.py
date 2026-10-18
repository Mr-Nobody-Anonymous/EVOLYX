"""Conversational session and context window manager."""

import logging
from typing import List, Optional
from evolyx.core.schemas import Message

logger = logging.getLogger(__name__)


class SessionManager:
    """Tracks message history and manages context window boundaries."""

    def __init__(self, session_id: str, user_id: str = "default_user", max_turns: int = 20):
        self.session_id = session_id
        self.user_id = user_id
        self.max_turns = max_turns
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> Message:
        msg = Message(role=role, content=content, name=name)
        self.messages.append(msg)
        self._truncate_if_needed()
        return msg

    def get_messages(self) -> List[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _truncate_if_needed(self) -> None:
        # Keep system messages if present, and the most recent max_turns
        if len(self.messages) > self.max_turns:
            system_msgs = [m for m in self.messages if m.role == "system"]
            recent_msgs = [m for m in self.messages if m.role != "system"][-self.max_turns :]
            self.messages = system_msgs + recent_msgs
            logger.debug("Session truncated to %d messages", len(self.messages))
