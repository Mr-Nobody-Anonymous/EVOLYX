"""Unit tests for EVOLYX interfaces and model registry."""

from evolyx.core.interfaces import ModelBackend
from evolyx.core.model_registry import ModelRegistry
from evolyx.core.schemas import Message
import integrations.model_backends.mock_backend  # noqa: F401


def test_mock_backend_protocol():
    backend = ModelRegistry.get_instance("mock")
    assert isinstance(backend, ModelBackend)
    assert backend.health_check() is True

    messages = [Message(role="user", content="Hello EVOLYX")]
    resp = backend.generate(messages)
    assert "EVOLYX automated response" in resp


def test_model_registry_unknown_backend():
    try:
        ModelRegistry.get_backend_class("non_existent_backend")
        assert False, "Should have raised ValueError"
    except ValueError:
        assert True
