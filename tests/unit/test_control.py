"""Unit tests for permissions, resource limits, and approvals."""

import tempfile
from pathlib import Path
from evolyx.control.approval import ApprovalWorkflow
from evolyx.control.audit import AuditLogger
from evolyx.control.permissions import ActionPermission, PermissionController
from evolyx.control.resource_limits import ResourceGovernor


def test_permission_controller():
    ctrl = PermissionController(human_override=False)
    assert ctrl.can_perform(ActionPermission.EXECUTE_READ_TOOLS) is True
    assert ctrl.can_perform(ActionPermission.PROMOTE_MODEL) is False

    override_ctrl = PermissionController(human_override=True)
    assert override_ctrl.can_perform(ActionPermission.PROMOTE_MODEL) is True


def test_resource_governor():
    gov = ResourceGovernor(max_concurrent_training_jobs=1)
    acquired, _ = gov.acquire_training_slot()
    assert acquired is True

    acquired_second, _ = gov.acquire_training_slot()
    assert acquired_second is False

    gov.release_training_slot()
    acquired_third, _ = gov.acquire_training_slot()
    assert acquired_third is True


def test_audit_logger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "audit.log"
        logger = AuditLogger(log_path=str(log_file))
        logger.log_event("agent_runtime", "tool_call", {"tool": "web_search"})
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "web_search" in lines[0]
