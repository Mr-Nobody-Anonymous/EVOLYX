"""Sandboxed Python code execution tool with AST validation, disposable filesystem sandbox, and process-level isolation."""

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

# Sandbox security preamble injected at the top of every isolated script
SANDBOX_PREAMBLE = """# EVOLYX Sandbox Security Shim
import sys, os

# 1. Disable network socket creation
try:
    import socket
    def _blocked_socket(*args, **kwargs):
        raise PermissionError("Network access is blocked inside EVOLYX execution sandbox.")
    socket.socket = _blocked_socket
    socket.create_connection = _blocked_socket
except Exception:
    pass

# 2. Prevent arbitrary process execution
for mod in ("subprocess", "pty"):
    if mod in sys.modules:
        del sys.modules[mod]

# 3. Restrict directory manipulation outside sandbox
_ORIG_CHDIR = os.chdir
_SANDBOX_DIR = os.getcwd()
def _safe_chdir(path):
    target = os.path.abspath(path)
    if not target.startswith(_SANDBOX_DIR):
        raise PermissionError(f"Cannot change directory outside sandbox root: {target}")
    return _ORIG_CHDIR(path)
os.chdir = _safe_chdir

# --- User Code Execution Begins Below ---
"""


class PythonExecutorTool(ToolBackend):
    """Executes Python code in an isolated disposable sandbox with enforced timeouts and network/filesystem boundaries."""

    def __init__(
        self,
        permission_manager: Optional[ToolPermissionManager] = None,
        default_timeout_seconds: float = 5.0,
        max_output_bytes: int = 65536,
    ):
        self.permission_manager = permission_manager or ToolPermissionManager()
        self.default_timeout_seconds = default_timeout_seconds
        self.max_output_bytes = max_output_bytes

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
        timeout = float(kwargs.get("timeout", self.default_timeout_seconds))

        # Layer 1: AST Security Filter (Static Analysis)
        is_safe, msg = self.permission_manager.validate_python_code(code)
        if not is_safe:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Security Violation: {msg}",
            )

        # Layer 2: Disposable Process Sandbox (Runtime Isolation)
        start_time = time.perf_counter()
        with tempfile.TemporaryDirectory() as sandbox_dir:
            script_path = os.path.join(sandbox_dir, "sandboxed_script.py")
            full_script = SANDBOX_PREAMBLE + code
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_script)

            # Minimal Environment: Strictly strip all host secrets, API keys, credentials, and user profiles
            clean_env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "TEMP": sandbox_dir,
                "TMP": sandbox_dir,
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
            }

            # Isolated invocation: -I (isolated mode), -s (no user site-packages), -E (ignore env), -B (no pyc)
            cmd = [sys.executable, "-I", "-s", "-E", "-B", script_path]

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

                # Cap stdout/stderr to max_output_bytes to prevent memory exhaustion
                stdout = proc.stdout[: self.max_output_bytes].strip()
                stderr = proc.stderr[: self.max_output_bytes].strip()

                if proc.returncode != 0:
                    err_msg = stderr or f"Process exited with code {proc.returncode}"
                    return ToolExecutionResult(
                        tool_name=self.name,
                        success=False,
                        output=stdout,
                        error=f"Runtime Error: {err_msg}",
                        duration_ms=duration_ms,
                    )

                return ToolExecutionResult(
                    tool_name=self.name,
                    success=True,
                    output=stdout,
                    error=None,
                    duration_ms=duration_ms,
                    evidence=[
                        Evidence(
                            source="python_executor",
                            content=stdout,
                            confidence=1.0,
                        )
                    ],
                )

            except subprocess.TimeoutExpired:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.warning("Code execution exceeded timeout of %.1fs", timeout)
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error=f"Execution Timeout: Code execution timed out after {timeout:.1f} seconds.",
                    duration_ms=duration_ms,
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=False,
                    output=None,
                    error=f"Sandbox Execution Failure: {exc}",
                    duration_ms=duration_ms,
                )
