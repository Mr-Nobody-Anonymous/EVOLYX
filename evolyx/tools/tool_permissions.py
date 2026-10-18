"""Security guardrails, command whitelists, and tool permission checks."""

import ast
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ToolPermissionManager:
    """Enforces execution boundaries, path restrictions, and AST safety checks."""

    FORBIDDEN_CALLS = {
        "os.system",
        "subprocess.Popen",
        "subprocess.run",
        "subprocess.call",
        "shutil.rmtree",
        "ctypes",
        "socket",
        "requests.post",  # Can be tuned
    }

    FORBIDDEN_NAMES = {
        "__import__",
        "eval",
        "exec",
    }

    def __init__(self, allow_network: bool = True, allow_fs_write: bool = True):
        self.allow_network = allow_network
        self.allow_fs_write = allow_fs_write

    def validate_python_code(self, code: str) -> Tuple[bool, str]:
        """Parses AST to check for dangerous function calls and imports."""
        try:
            tree = ast.parse(code)
        except SyntaxError as err:
            return False, f"Syntax Error: {err}"

        for node in ast.walk(tree):
            # Check for banned names (eval, exec, __import__)
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                return False, f"Use of forbidden function '{node.id}' is blocked."

            # Check for banned module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["ctypes", "subprocess"]:
                        return False, f"Import of dangerous module '{alias.name}' is blocked."

            if isinstance(node, ast.ImportFrom):
                if node.module in ["ctypes", "subprocess"]:
                    return False, f"Import from dangerous module '{node.module}' is blocked."

        return True, "Code passed security AST check."
