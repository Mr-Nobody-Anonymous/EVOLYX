"""Integration test validating the complete EVOLYX 5-stage self-improvement loop."""

import tempfile
from evolyx.core.model_registry import ModelRegistry
from evolyx.evaluation.benchmarks import BenchmarkRunner
from evolyx.evaluation.factuality import FactualityEvaluator
from evolyx.evaluation.promotion import PromotionManager
from evolyx.evaluation.regression import RegressionChecker
from evolyx.learning.dataset_builder import DatasetBuilder
from evolyx.learning.trajectory_store import TrajectoryStore
from evolyx.memory.long_term import LongTermMemory
from evolyx.runtime.orchestrator import EvolyxAgent
from evolyx.runtime.tool_router import ToolRouter
from evolyx.tools.python_executor import PythonExecutorTool
from evolyx.training.checkpoints import CheckpointManager
from evolyx.training.sft import SFTPipeline
import integrations.model_backends.mock_backend  # noqa: F401


def test_full_evolyx_self_improvement_cycle():
    with tempfile.TemporaryDirectory() as work_dir:
        db_path = f"{work_dir}/evolyx_test.db"
        models_dir = f"{work_dir}/models"
        datasets_dir = f"{work_dir}/datasets"
        baseline_path = f"{work_dir}/baseline.json"

        # 1. Setup EVOLYX Subsystems
        model = ModelRegistry.get_instance("mock")
        memory = LongTermMemory(db_path=db_path)
        tool_router = ToolRouter()
        tool_router.register_tool(PythonExecutorTool())
        evaluator = FactualityEvaluator()
        trajectory_store = TrajectoryStore(db_path=db_path)

        agent = EvolyxAgent(
            model=model,
            memory=memory,
            tool_router=tool_router,
            evaluator=evaluator,
            experience_store=trajectory_store,
        )

        # 2. Stage 1: Runtime Execution (Observe & Act)
        task_query = "Calculate factorial of 5 using python code"
        task_res = agent.run(user_id="integration_user", message=task_query)
        assert task_res.success is True

        # 3. Stage 2: Verify Trajectory Recorded
        eligible = trajectory_store.get_training_eligible()
        assert len(eligible) >= 1

        # 4. Stage 3: Dataset Builder
        builder = DatasetBuilder(trajectory_store=trajectory_store, output_dir=datasets_dir)
        dataset_meta = builder.build_sft_dataset(version="v002")
        assert "train_file" in dataset_meta

        # 5. Stage 4: Fine-Tuning Pipeline
        sft_pipe = SFTPipeline()
        candidate = sft_pipe.run_sft(
            base_model="Qwen2.5-0.5B-Instruct",
            dataset_manifest=dataset_meta,
            training_config={"hyperparameters": {"learning_rate": 2e-4}},
        )
        assert candidate.model_id is not None

        # 6. Stage 5: Evaluation Battery & Promotion Gate
        benchmark_runner = BenchmarkRunner()
        regression_checker = RegressionChecker(baseline_path=baseline_path)
        checkpoint_manager = CheckpointManager(root_dir=models_dir)

        promoter = PromotionManager(
            benchmark_runner=benchmark_runner,
            regression_checker=regression_checker,
            checkpoint_manager=checkpoint_manager,
            min_overall_score=0.85,
        )

        report = promoter.evaluate_candidate(candidate)
        assert report.passed_release_gates is True

        promoted = promoter.decide_promotion(candidate, report, version_tag="v002_production")
        assert promoted is True
        assert checkpoint_manager.get_active_production_version() == "v002_production"
