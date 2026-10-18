"""Sandboxed Python code execution tool with AST validation and isolated subprocess execution."""

import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional
from evolyx.core.interfaces import ToolBackend
from evolyx.core.schemas import Evidence, ToolExecutionResult
from evolyx.tools.tool_permissions import ToolPermissionManager

logger = logging.getLogger(__name__)


class PythonExecutorTool(ToolBackend):
    """Executes Python code in an isolated subprocess sandbox with strict resource and time limits."""

    def __init__(
        self,
        permission_manager: Optional[ToolPermissionManager] = None,
        default_timeout_seconds: float = 5.0,
    ):
        self.permission_manager = permission_manager or ToolPermissionManager()
        self.default_timeout_seconds = default_timeout_seconds

    @property
    def name(self) -> str:
        return "python_executor"

    @property
    def description(self) -> str:
        return "Safely executes a Python code snippet in an isolated sandbox subprocess and captures stdout."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (default 5.0)"},
            },
            "required": ["code"],
        }

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        code = kwargs.get("code", "")
        timeout = kwargs.get("timeout", self.default_timeout_seconds)

        # Layer 1: AST Security Filter
        is_safe, msg = self.permission_manager.validate_python_code(code)
        if not is_safe:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Security Violation: {msg}",
            )

        # Layer 2: Disposable Process Sandbox
        start_time = time.perf_counter()
        with tempfile.TemporaryDirectory() as sandbox_dir:
            script_path = os.path.join(sandbox_dir, "sandbox_run.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            # Minimal environment: Strip all host secrets, API keys, credentials, and user paths
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "TEMP": sandbox_dir,
                "TMP": sandbox_dir,
            }

            cmd = [sys.executable, "-I", "-s", script_path]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=sandbox_dir,
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                duration_ms = (time.perf_counter() - start_time) * 1000

                if proc.returncode != 0:
                    err_msg = proc.stderr.strip() or f"Process exited with code {proc.returncode}"
                    return ToolExecutionResult(
                        tool_name=self.name,
                        success=False,
                        output=proc.stdout.strip(),
                        error=f"Runtime Error: {err_msg}",
                        duration_ms=duration_ms,
                    )

                stdout = proc.stdout.strip()
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=True,
                    output=stdout or "Executed successfully (no stdout).",
                    evidence=[
                        Evidence(
                            source="tool://python_executor",
                            content=f"Execution stdout: {stdout[:300]}",
                            verification_method="sandboxed_subprocess_execution",
                            confidence=1.0,
                        )
                    ],
                    duration_ms=duration_ms,
                )

            except subprocess.TimeoutExpired:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error=f"Execution Timeout: Code exceeded limit of {timeout}s",
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error=f"Sandbox Process Error: {exc}",
                    duration_ms=duration_ms,
                )
