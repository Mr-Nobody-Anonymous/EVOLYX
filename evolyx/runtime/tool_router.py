"""Tool router and dispatcher for EVOLYX runtime."""

import logging
import time
from typing import Any, Dict, List, Optional
from evolyx.core.interfaces import ToolBackend
from evolyx.core.schemas import Evidence, PlanStep, ToolExecutionResult

logger = logging.getLogger(__name__)


class ToolRouter:
    """Manages tool registration, argument validation, and invocation."""

    def __init__(self):
        self._tools: Dict[str, ToolBackend] = {}

    def register_tool(self, tool: ToolBackend) -> None:
        """Registers a tool backend."""
        self._tools[tool.name] = tool
        logger.info("Registered tool '%s'", tool.name)

    def get_tool(self, name: str) -> Optional[ToolBackend]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.schema,
            }
            for t in self._tools.values()
        ]

    def execute_step(self, step: PlanStep) -> ToolExecutionResult:
        """Executes a single plan step using the requested tool."""
        if not step.tool_name:
            return ToolExecutionResult(
                tool_name="none",
                success=True,
                output="Step completed without tools.",
            )

        tool = self.get_tool(step.tool_name)
        if not tool:
            err = f"Tool '{step.tool_name}' not found."
            logger.error(err)
            return ToolExecutionResult(
                tool_name=step.tool_name,
                success=False,
                output=None,
                error=err,
            )

        start = time.perf_counter()
        try:
            result = tool.execute(**step.tool_arguments)
            result.duration_ms = (time.perf_counter() - start) * 1000
            step.completed = result.success
            step.result = str(result.output)
            return result
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            logger.exception("Error executing tool %s: %s", step.tool_name, exc)
            return ToolExecutionResult(
                tool_name=step.tool_name,
                success=False,
                output=None,
                error=str(exc),
                duration_ms=duration,
            )
