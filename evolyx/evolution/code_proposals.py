"""Self-modification code proposal engine."""

import ast
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CodeProposalAgent:
    """Safely formulates code changes, validates syntax, and creates candidate patches."""

    def __init__(self, codebase_root: str = "."):
        self.codebase_root = codebase_root

    def validate_patch(self, patch_code: str) -> Tuple[bool, str]:
        """Validates that proposed patch contains valid Python syntax and no banned calls."""
        try:
            tree = ast.parse(patch_code)
        except SyntaxError as err:
            return False, f"Syntax Error in patch: {err}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in ["os.system", "shutil.rmtree"]:
                return False, f"Dangerous operation {node.id} detected in proposed patch."

        return True, "Patch syntax and static checks passed."

    def create_proposal(
        self,
        target_file: str,
        description: str,
        replacement_code: str,
    ) -> Dict[str, str]:
        """Generates a structured candidate change proposal."""
        valid, msg = self.validate_patch(replacement_code)
        if not valid:
            raise ValueError(f"Code proposal rejected: {msg}")

        proposal = {
            "target_file": target_file,
            "description": description,
            "code": replacement_code,
            "status": "ready_for_review",
        }
        logger.info("Created code modification proposal for %s: %s", target_file, description)
        return proposal
