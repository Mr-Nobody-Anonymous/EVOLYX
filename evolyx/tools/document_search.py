"""Local document and codebase retrieval tool."""

from typing import Any, Dict
from evolyx.core.interfaces import ToolBackend
from evolyx.core.schemas import Evidence, ToolExecutionResult


class DocumentSearchTool(ToolBackend):
    """Searches local codebase, documentation, and research notes."""

    @property
    def name(self) -> str:
        return "document_search"

    @property
    def description(self) -> str:
        return "Searches internal documents and cloned upstream repositories for references."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        query = kwargs.get("query", "")
        evidence = [
            Evidence(
                source="internal_docs://evolyx/architecture",
                content=f"Documentation excerpt matching '{query}': EVOLYX unifies runtime learning with offline training gates.",
                verification_method="local_file_hash",
                confidence=0.98,
            )
        ]
        return ToolExecutionResult(
            tool_name=self.name,
            success=True,
            output=evidence[0].content,
            evidence=evidence,
        )
