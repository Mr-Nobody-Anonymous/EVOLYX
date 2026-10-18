"""Event bus for asynchronous event publishing and subscriptions."""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventBus:
    """Publish-subscribe event bus decoupling EVOLYX subsystems."""

    def __init__(self):
        self._listeners: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(handler)

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        logger.debug("Publishing event %s: %s", event_type, data)
        for handler in self._listeners.get(event_type, []):
            try:
                handler(data)
            except Exception as exc:
                logger.error("Error in event handler for %s: %s", event_type, exc)
