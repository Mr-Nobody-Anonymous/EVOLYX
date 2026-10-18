"""EVOLYX HTTP REST API."""

import logging
from typing import Any, Dict, List, Optional
from evolyx.core.model_registry import ModelRegistry
from evolyx.evaluation.factuality import FactualityEvaluator
from evolyx.evolution.experiment import ExperimentRunner
from evolyx.evolution.hypothesis import HypothesisGenerator
from evolyx.learning.trajectory_store import TrajectoryStore
from evolyx.memory.long_term import LongTermMemory
from evolyx.runtime.orchestrator import EvolyxAgent
from evolyx.runtime.tool_router import ToolRouter
from evolyx.tools.document_search import DocumentSearchTool
from evolyx.tools.python_executor import PythonExecutorTool
from evolyx.tools.web_search import WebSearchTool

import integrations.model_backends.mock_backend  # noqa: F401

logger = logging.getLogger(__name__)

# Initialize single global agent
model = ModelRegistry.get_instance("mock")
memory = LongTermMemory(db_path="artifacts/evolyx.db")
tool_router = ToolRouter()
tool_router.register_tool(WebSearchTool())
tool_router.register_tool(DocumentSearchTool())
tool_router.register_tool(PythonExecutorTool())
evaluator = FactualityEvaluator()
experience_store = TrajectoryStore(db_path="artifacts/evolyx.db")

agent = EvolyxAgent(
    model=model,
    memory=memory,
    tool_router=tool_router,
    evaluator=evaluator,
    experience_store=experience_store,
)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(
        title="EVOLYX AI Platform API",
        version="0.1.0",
        description="Autonomous Self-Improving AI Research Platform API",
    )

    class TaskRequest(BaseModel):
        user_id: str = "api_user"
        message: str

    class TaskResponse(BaseModel):
        answer: str
        intent: str
        success: bool
        latency_ms: float
        evidence_count: int

    @app.get("/")
    def root():
        return {
            "platform": "EVOLYX",
            "version": "0.1.0",
            "status": "online",
            "backend": "mock",
        }

    @app.get("/v1/status")
    def get_status():
        return {
            "model_healthy": agent.model.health_check(),
            "tools_registered": len(agent.tool_router.list_tools()),
            "active_sessions": len(agent.sessions),
        }

    @app.post("/v1/tasks", response_model=TaskResponse)
    def execute_task(req: TaskRequest):
        res = agent.run(user_id=req.user_id, message=req.message)
        return TaskResponse(
            answer=res.answer,
            intent=res.intent.value,
            success=res.success,
            latency_ms=round(res.latency_ms, 2),
            evidence_count=len(res.evidence),
        )

    @app.post("/v1/experiments/run")
    def run_experiment():
        hypo = HypothesisGenerator().analyze_failures([])
        result = ExperimentRunner().run_experiment(hypo, baseline_version="v001")
        return {"proposal": hypo.__dict__, "result": result}

except ImportError:
    # If fastapi is not installed yet, provide fallback placeholder
    app = None
