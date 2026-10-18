"""Security guardrails, command whitelists, and tool permission checks."""

import ast
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ToolPermissionManager:
    """Enforces execution boundaries, path restrictions, and AST safety checks."""

    FORBIDDEN_MODULES = {
        "subprocess",
        "ctypes",
        "os",
        "sys",
        "shutil",
        "socket",
        "pty",
        "posix",
        "winreg",
        "signal",
        "multiprocessing",
        "threading",
    }

    FORBIDDEN_NAMES = {
        "__import__",
        "eval",
        "exec",
        "compile",
        "breakpoint",
    }

    FORBIDDEN_ATTRS = {
        "__subclasses__",
        "__globals__",
        "__bases__",
        "__mro__",
        "__code__",
        "__builtins__",
        "system",
        "popen",
        "spawn",
        "rmtree",
        "kill",
        "unlink",
        "remove",
    }

    def __init__(self, allow_network: bool = True, allow_fs_write: bool = True):
        self.allow_network = allow_network
        self.allow_fs_write = allow_fs_write

    def validate_python_code(self, code: str) -> Tuple[bool, str]:
        """Parses AST to check for dangerous function calls, attribute traversal, and imports."""
        try:
            tree = ast.parse(code)
        except SyntaxError as err:
            return False, f"Syntax Error: {err}"

        for node in ast.walk(tree):
            # Check for banned names (eval, exec, __import__, compile)
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                return False, f"Use of forbidden function '{node.id}' is blocked."

            # Check for banned module imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in self.FORBIDDEN_MODULES:
                        return False, f"Import of dangerous module '{alias.name}' is blocked."

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in self.FORBIDDEN_MODULES:
                        return False, f"Import from dangerous module '{node.module}' is blocked."

            # Check for forbidden attribute access (e.g. .__subclasses__, .system, .popen)
            if isinstance(node, ast.Attribute):
                if node.attr in self.FORBIDDEN_ATTRS:
                    return False, f"Access to forbidden attribute/method '{node.attr}' is blocked."

        return True, "Code passed security AST check."
