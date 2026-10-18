"""Sandboxed Python code execution tool with AST security checks."""

import io
import sys
from typing import Any, Dict
from evolyx.core.interfaces import ToolBackend
from evolyx.core.schemas import Evidence, ToolExecutionResult
from evolyx.tools.tool_permissions import ToolPermissionManager


class PythonExecutorTool(ToolBackend):
    """Executes validated Python code in an isolated scope."""

    def __init__(self, permission_manager: ToolPermissionManager = None):
        self.permission_manager = permission_manager or ToolPermissionManager()

    @property
    def name(self) -> str:
        return "python_executor"

    @property
    def description(self) -> str:
        return "Safely executes a Python code snippet and captures printed output."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source code"}
            },
            "required": ["code"],
        }

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        code = kwargs.get("code", "")
        # Run AST security validation
        is_safe, msg = self.permission_manager.validate_python_code(code)
        if not is_safe:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Security Violation: {msg}",
            )

        # Execute in captured environment
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        local_scope: Dict[str, Any] = {}

        try:
            exec(code, {"__builtins__": __builtins__}, local_scope)
            output = redirected_output.getvalue().strip()
            return ToolExecutionResult(
                tool_name=self.name,
                success=True,
                output=output or "Executed successfully (no stdout).",
                evidence=[
                    Evidence(
                        source="tool://python_executor",
                        content=f"Execution stdout: {output[:300]}",
                        verification_method="runtime_execution",
                        confidence=1.0,
                    )
                ],
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                output=None,
                error=f"Runtime Exception: {exc}",
            )
        finally:
            sys.stdout = old_stdout
