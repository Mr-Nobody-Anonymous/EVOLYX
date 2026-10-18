"""Verified synthetic data generation with automated execution checks."""

import logging
from typing import Any, Dict, List, Optional
from evolyx.core.interfaces import ToolBackend
from evolyx.tools.python_executor import PythonExecutorTool

logger = logging.getLogger(__name__)


class VerifiedSyntheticGenerator:
    """Generates synthetic reasoning and coding solutions and verifies them via execution."""

    def __init__(self, python_executor: Optional[ToolBackend] = None):
        self.executor = python_executor or PythonExecutorTool()

    def generate_and_verify_coding_example(
        self,
        task_prompt: str,
        solution_code: str,
        test_assertions: str,
    ) -> Optional[Dict[str, Any]]:
        """Executes code + assertions in the sandbox. Admitted ONLY if test passes with zero errors."""
        full_script = f"{solution_code}\n\n# Verification Test Assertions\n{test_assertions}"
        res = self.executor.execute(code=full_script)

        if not res.success:
            logger.warning("Synthetic solution rejected: Verification failed: %s", res.error)
            return None

        logger.info("Synthetic example passed sandboxed verification.")
        return {
            "task_prompt": task_prompt,
            "verified_solution": solution_code,
            "verification_method": "sandboxed_unit_test",
            "status": "PASSED",
            "eligible_for_training": True,
        }
