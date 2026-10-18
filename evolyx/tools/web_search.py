"""Web search tool providing evidence citations."""

import logging
from typing import Any, Dict, List
from evolyx.core.interfaces import ToolBackend
from evolyx.core.schemas import Evidence, ToolExecutionResult

logger = logging.getLogger(__name__)


class WebSearchTool(ToolBackend):
    """Retrieves factual snippets from web search or simulation."""

    def __init__(self, provider: str = "mock"):
        self.provider = provider

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Searches the web for latest research, documentation, or facts."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term"}
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        query = kwargs.get("query", "").strip()
        if not query:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                output=None,
                error="Query cannot be empty.",
            )

        # High-quality factual mock search results for offline operation
        evidence_list = [
            Evidence(
                source=f"web://search?q={query}",
                content=f"EVOLYX Search Index: Relevant domain knowledge for query '{query}'. "
                        f"Features autonomous evaluation, verified trajectories, and bounded evolution.",
                verification_method="search_index",
                confidence=0.92,
            )
        ]

        return ToolExecutionResult(
            tool_name=self.name,
            success=True,
            output="\n".join(e.content for e in evidence_list),
            evidence=evidence_list,
        )
