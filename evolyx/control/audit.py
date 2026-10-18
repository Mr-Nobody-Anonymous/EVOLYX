"""Structured audit logger recording sensitive operations and decisions."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AuditLogger:
    """Writes immutable JSON-lines audit records for governance."""

    def __init__(self, log_path: str = "artifacts/audit.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, actor: str, action: str, details: Dict[str, Any], status: str = "success") -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "status": status,
            "details": details,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("AUDIT: [%s] %s by %s (%s)", status.upper(), action, actor, details.get("summary", ""))
