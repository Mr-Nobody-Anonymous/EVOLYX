"""Unit tests for permissions, resource limits, token rate limiting, and approvals."""

import tempfile
import time
from pathlib import Path
import pytest
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
    acquired, _ = gov.acquire_training_slot(job_id="job_01")
    assert acquired is True

    acquired_second, reason = gov.acquire_training_slot(job_id="job_02")
    assert acquired_second is False
    assert "Max concurrent" in reason

    gov.release_training_slot(job_id="job_01")
    acquired_third, _ = gov.acquire_training_slot(job_id="job_03")
    assert acquired_third is True
    gov.release_training_slot(job_id="job_03")


def test_resource_governor_token_rate_limiting():
    gov = ResourceGovernor(max_tokens_per_minute=100)
    ok1, _ = gov.check_token_quota(60)
    assert ok1 is True

    ok2, _ = gov.check_token_quota(30)
    assert ok2 is True

    # Next request exceeds 100
    ok3, reason = gov.check_token_quota(20)
    assert ok3 is False
    assert "Token rate limit exceeded" in reason


def test_resource_governor_disk_space_rejection():
    # Set impossible disk requirement: 999999 GB
    gov = ResourceGovernor(min_free_disk_gb=999999.0)
    acquired, reason = gov.acquire_training_slot()
    assert acquired is False
    assert "Insufficient disk space" in reason


def test_resource_governor_job_context():
    gov = ResourceGovernor(max_concurrent_training_jobs=1)
    status_initial = gov.get_resource_status()
    assert status_initial["current_training_jobs"] == 0

    with gov.job_context(job_id="managed_job_1", timeout_seconds=10.0):
        status_inside = gov.get_resource_status()
        assert status_inside["current_training_jobs"] == 1
        assert "managed_job_1" in status_inside["active_job_ids"]

    status_after = gov.get_resource_status()
    assert status_after["current_training_jobs"] == 0
    assert "managed_job_1" not in status_after["active_job_ids"]


def test_audit_logger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_file = Path(tmp_dir) / "audit.log"
        logger = AuditLogger(log_path=str(log_file))
        logger.log_event("agent_runtime", "tool_call", {"tool": "web_search"})
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert "web_search" in lines[0]
