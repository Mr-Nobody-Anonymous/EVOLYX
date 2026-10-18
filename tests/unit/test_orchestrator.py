"""Unit tests for runtime orchestrator, intent, and planner."""

from evolyx.core.model_registry import ModelRegistry
from evolyx.core.schemas import IntentType
from evolyx.runtime.orchestrator import EvolyxAgent
from evolyx.runtime.tool_router import ToolRouter
from evolyx.tools.web_search import WebSearchTool
import integrations.model_backends.mock_backend  # noqa: F401


def test_agent_run_direct_answer():
    model = ModelRegistry.get_instance("mock")
    agent = EvolyxAgent(model=model)

    result = agent.run(user_id="test_user", message="Hello there!")
    assert result.success is True
    assert result.intent == IntentType.DIRECT_ANSWER
    assert len(result.answer) > 0


def test_agent_run_with_tools():
    model = ModelRegistry.get_instance("mock")
    tool_router = ToolRouter()
    tool_router.register_tool(WebSearchTool())

    agent = EvolyxAgent(model=model, tool_router=tool_router)
    result = agent.run(user_id="test_user", message="Search for latest AI papers")
    assert result.success is True
    assert result.intent == IntentType.RESEARCH_AND_RETRIEVE
    assert len(result.evidence) > 0
