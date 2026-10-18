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


def test_orchestrator_fail_closed_assessment():
    from evolyx.core.schemas import Assessment, EvaluationStatus
    from evolyx.learning.trajectory_store import TrajectoryStore
    import tempfile

    model = ModelRegistry.get_instance("mock")

    # 1. Unverified draft (no evaluator): Never eligible for training
    with tempfile.TemporaryDirectory() as tmp_dir:
        t_store = TrajectoryStore(db_path=f"{tmp_dir}/store.db")
        agent_unverified = EvolyxAgent(model=model, experience_store=t_store)
        res = agent_unverified.run(user_id="user_1", message="Write an essay")
        assert res.success is True
        eligible = t_store.get_training_eligible()
        assert len(eligible) == 0, "Unverified draft must NOT enter training store"

    # 2. Failing evaluator that raises exception: Fails closed to success=False
    class CrashingEvaluator:
        def evaluate_answer(self, **kwargs):
            raise ValueError("Evaluator service connection timed out")

    agent_crashing = EvolyxAgent(model=model, evaluator=CrashingEvaluator())
    res_crash = agent_crashing.run(user_id="user_2", message="Execute code")
    assert res_crash.success is False

    # 3. Verified passing evaluator
    class PassingEvaluator:
        def evaluate_answer(self, **kwargs):
            return Assessment(passed=True, score=0.95, feedback="Flawless")

    with tempfile.TemporaryDirectory() as tmp_dir:
        t_store = TrajectoryStore(db_path=f"{tmp_dir}/store.db")
        agent_passing = EvolyxAgent(model=model, evaluator=PassingEvaluator(), experience_store=t_store)
        res_pass = agent_passing.run(user_id="user_3", message="Prove math problem")
        assert res_pass.success is True
        eligible = t_store.get_training_eligible()
        assert len(eligible) == 1
