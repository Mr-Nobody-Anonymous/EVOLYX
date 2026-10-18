"""Human-in-the-loop approval workflow for high-risk operations."""

import logging
import uuid
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ApprovalWorkflow:
    """Blocks critical changes (promotions, code modifications) until approved."""

    def __init__(self, require_approval: bool = True):
        self.require_approval = require_approval
        self._pending_requests: Dict[str, Dict] = {}

    def request_approval(self, action_type: str, description: str, payload: Dict) -> str:
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        self._pending_requests[req_id] = {
            "req_id": req_id,
            "action_type": action_type,
            "description": description,
            "payload": payload,
            "status": "pending" if self.require_approval else "auto_approved",
        }
        logger.info("Approval requested [%s]: %s (Status: %s)", req_id, description, self._pending_requests[req_id]["status"])
        return req_id

    def review_request(self, req_id: str, approved: bool, reviewer: str = "human_operator") -> bool:
        if req_id not in self._pending_requests:
            raise KeyError(f"Approval request {req_id} not found.")

        req = self._pending_requests[req_id]
        req["status"] = "approved" if approved else "rejected"
        req["reviewer"] = reviewer
        logger.info("Approval request %s %s by %s", req_id, req["status"], reviewer)
        return approved

    def is_approved(self, req_id: str) -> bool:
        req = self._pending_requests.get(req_id)
        if not req:
            return False
        return req["status"] in ["approved", "auto_approved"]
