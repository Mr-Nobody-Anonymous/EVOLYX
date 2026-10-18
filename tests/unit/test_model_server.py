"""Unit tests for production ModelServer and LocalCausalLMServer."""

from pathlib import Path
from evolyx.serving.model_server import LocalCausalLMServer, ModelServer
from evolyx.storage.registry import ExperimentRegistry


def test_local_causal_lm_server_generation():
    server = LocalCausalLMServer()
    out = server.generate("Hello world", max_tokens=8)
    assert isinstance(out, str)


def test_model_server_reload_and_health_check(tmp_path: Path):
    db_file = tmp_path / "reg.db"
    artifacts_dir = tmp_path / "artifacts"
    registry = ExperimentRegistry(db_path=str(db_file), artifacts_root=str(artifacts_dir))

    pointer_file = artifacts_dir / "models" / "ACTIVE_VERSION.txt"
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    pointer_file.write_text("v001_prod", encoding="utf-8")

    server = ModelServer(registry=registry, artifacts_root=str(artifacts_dir))
    assert server.active_version == "v001_prod"

    health = server.health_check()
    assert health["status"] == "healthy"
    assert health["active_version"] == "v001_prod"
    assert "latency_ms" in health
    assert health["latency_ms"] >= 0.0
