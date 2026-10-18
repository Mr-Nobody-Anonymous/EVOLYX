"""Model backend registry for EVOLYX."""

import logging
from typing import Callable, Dict, Optional, Type
from evolyx.core.interfaces import ModelBackend

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for LLM inference backends."""

    _backends: Dict[str, Type[ModelBackend]] = {}
    _instances: Dict[str, ModelBackend] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[ModelBackend]], Type[ModelBackend]]:
        """Decorator to register a new model backend class."""
        def decorator(subclass: Type[ModelBackend]) -> Type[ModelBackend]:
            cls._backends[name.lower()] = subclass
            logger.info("Registered model backend '%s'", name.lower())
            return subclass
        return decorator

    @classmethod
    def get_backend_class(cls, name: str) -> Type[ModelBackend]:
        """Gets backend class by registered name."""
        key = name.lower()
        if key not in cls._backends:
            raise ValueError(
                f"Unknown model backend '{name}'. Available: {list(cls._backends.keys())}"
            )
        return cls._backends[key]

    @classmethod
    def get_instance(
        cls,
        name: str,
        config: Optional[Dict] = None,
        reuse: bool = True
    ) -> ModelBackend:
        """Instantiates or retrieves cached instance of a backend."""
        key = name.lower()
        if reuse and key in cls._instances:
            return cls._instances[key]

        backend_cls = cls.get_backend_class(key)
        config = config or {}
        instance = backend_cls(**config) if config else backend_cls()
        if reuse:
            cls._instances[key] = instance
        return instance

    @classmethod
    def clear(cls) -> None:
        """Clears cached backend instances."""
        cls._instances.clear()
