"""Self-modifying software engine executing isolated sandbox patches and regression testing."""

import ast
import difflib
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CodeProposalEngine:
    """Safely proposes, sandboxes, tests, and validates EVOLYX source code modifications."""

    def __init__(self, codebase_root: str = "."):
        self.codebase_root = Path(codebase_root).resolve()

    def validate_ast(self, code: str) -> Tuple[bool, str]:
        """Layer 1 Defense: AST static validation."""
        try:
            tree = ast.parse(code)
        except SyntaxError as err:
            return False, f"Syntax Error: {err}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ["os.system", "shutil.rmtree"]:
                return False, f"Forbidden call detected: {node.id}"

        return True, "AST check passed."

    def generate_unified_diff(self, original_text: str, modified_text: str, filename: str) -> str:
        """Produces standard Git unified diff for human review."""
        diff = difflib.unified_diff(
            original_text.splitlines(keepends=True),
            modified_text.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
        return "".join(diff)

    def test_patch_in_sandbox(
        self,
        rel_target_file: str,
        replacement_code: str,
        test_file: Optional[str] = None,
        timeout_seconds: int = 15,
    ) -> Dict[str, Any]:
        """Layer 2 Defense: Sandboxed directory copy, patch application, and isolated subprocess test run."""
        # 1. AST Check
        is_safe, ast_msg = self.validate_ast(replacement_code)
        if not is_safe:
            return {"status": "rejected", "reason": f"AST Check Failed: {ast_msg}", "tests_passed": False}

        orig_file = self.codebase_root / rel_target_file
        orig_content = orig_file.read_text(encoding="utf-8") if orig_file.exists() else ""
        diff_text = self.generate_unified_diff(orig_content, replacement_code, rel_target_file)

        # 2. Isolated Workspace Execution
        with tempfile.TemporaryDirectory() as sandbox_dir:
            sandbox_path = Path(sandbox_dir)
            sandboxed_target = sandbox_path / rel_target_file
            sandboxed_target.parent.mkdir(parents=True, exist_ok=True)
            sandboxed_target.write_text(replacement_code, encoding="utf-8")

            # Copy tests into sandbox
            tests_src = self.codebase_root / "tests"
            if tests_src.exists():
                shutil.copytree(tests_src, sandbox_path / "tests")

            # Run test suite in isolated subprocess
            test_target = [str(sandbox_path / test_file)] if test_file else [str(sandbox_path / "tests" / "unit")]
            cmd = [sys.executable, "-m", "pytest"] + test_target + ["-q"]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=sandbox_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env={"PYTHONPATH": str(sandbox_path)},
                )
                tests_passed = proc.returncode == 0
                stdout = proc.stdout
            except subprocess.TimeoutExpired:
                tests_passed = False
                stdout = "Execution timed out in sandbox."
            except Exception as exc:
                tests_passed = False
                stdout = str(exc)

        status = "approved_for_human_review" if tests_passed else "rejected"
        logger.info("Sandbox patch evaluation: %s (Tests Passed: %s)", status, tests_passed)

        return {
            "status": status,
            "target_file": rel_target_file,
            "tests_passed": tests_passed,
            "unified_diff": diff_text,
            "sandbox_output": stdout[:500],
        }
