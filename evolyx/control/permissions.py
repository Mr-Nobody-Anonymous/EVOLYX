"""Role-based access control and system authorization policies."""

from enum import Enum
from typing import Set


class ActionPermission(str, Enum):
    EXECUTE_READ_TOOLS = "execute_read_tools"
    EXECUTE_CODE = "execute_code"
    WRITE_MEMORY = "write_memory"
    TRIGGER_TRAINING = "trigger_training"
    PROMOTE_MODEL = "promote_model"
    MODIFY_SYSTEM_CODE = "modify_system_code"


class PermissionController:
    """Enforces authorization boundaries for autonomous agent actions."""

    DEFAULT_AGENT_PERMISSIONS: Set[ActionPermission] = {
        ActionPermission.EXECUTE_READ_TOOLS,
        ActionPermission.EXECUTE_CODE,
        ActionPermission.WRITE_MEMORY,
        ActionPermission.TRIGGER_TRAINING,
    }

    RESTRICTED_PERMISSIONS: Set[ActionPermission] = {
        ActionPermission.PROMOTE_MODEL,
        ActionPermission.MODIFY_SYSTEM_CODE,
    }

    def __init__(self, human_override: bool = False):
        self.human_override = human_override

    def can_perform(self, action: ActionPermission) -> bool:
        if self.human_override:
            return True
        if action in self.RESTRICTED_PERMISSIONS:
            return False
        return action in self.DEFAULT_AGENT_PERMISSIONS
