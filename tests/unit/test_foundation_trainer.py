"""Unit tests for FoundationAdapterTrainer and LoRA adapter layer."""

import hashlib
import json
from pathlib import Path
import pytest
import torch
from evolyx.training.foundation_trainer import FoundationAdapterTrainer, LoRAAdapterLayer


def test_lora_adapter_layer_forward():
    d_in = 32
    d_out = 32
    rank = 4
    alpha = 8.0
    layer = LoRAAdapterLayer(in_features=d_in, out_features=d_out, rank=rank, alpha=alpha)

    assert layer.scaling == 2.0
    assert layer.lora_A.shape == (rank, d_in)
    assert layer.lora_B.shape == (d_out, rank)

    x = torch.randn(2, 5, d_in)
    out = layer(x)
    assert out.shape == (2, 5, d_out)

    loss = out.sum()
    loss.backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None


def test_foundation_adapter_trainer_training_and_artifacts(tmp_path: Path):
    dataset_path = tmp_path / "train.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for _ in range(5):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Write a python function for fibonacci"},
                    {"role": "assistant", "content": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"}
                ]
            }) + "\n")

    out_dir = tmp_path / "candidates"
    trainer = FoundationAdapterTrainer(output_dir=str(out_dir))

    candidate = trainer.train_candidate(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        dataset_path=str(dataset_path),
        config={"rank": 4, "alpha": 8.0, "epochs": 5, "lr": 0.05, "d_model": 32},
    )

    assert candidate.status == "trained"
    adapter_path = Path(candidate.adapter_path)
    assert adapter_path.exists()

    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.pt"
    assert config_path.exists()
    assert weights_path.exists()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["peft_type"] == "LORA"
    assert cfg["r"] == 4
    assert len(cfg["sha256"]) == 64
    assert cfg["training_metrics"]["epochs"] == 5
    assert cfg["training_metrics"]["final_loss"] <= cfg["training_metrics"]["initial_loss"]

    # Verify sha256 matches actual weights file
    hasher = hashlib.sha256()
    with open(weights_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    assert hasher.hexdigest() == cfg["sha256"]

    # Test load_adapter
    loaded = trainer.load_adapter(str(adapter_path))
    assert loaded["status"] == "loaded"
    assert loaded["verified"] is True
    assert "lora_A" in loaded["state_dict"]
    assert "lora_B" in loaded["state_dict"]


def test_load_adapter_missing_artifacts(tmp_path: Path):
    empty_dir = tmp_path / "empty_adapter"
    empty_dir.mkdir()
    trainer = FoundationAdapterTrainer(output_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        trainer.load_adapter(str(empty_dir))
