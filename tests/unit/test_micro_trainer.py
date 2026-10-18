"""Unit tests for MicroTransformerLM and in-memory neural training engine."""

import pytest
import torch
from pathlib import Path
from evolyx.training.micro_trainer import MicroTransformerLM, MicroNeuralTrainer


def test_micro_transformer_architecture():
    """Verifies parameter count and forward pass shapes for in-memory micro model."""
    vocab_size = 128
    d_model = 64
    n_layers = 2
    model = MicroTransformerLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)

    params = model.count_parameters()
    assert 50_000 < params < 300_000, f"Expected micro model params, got {params}"

    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(dummy_input)

    assert logits.shape == (batch_size, seq_len, vocab_size)


def test_micro_training_and_checkpoint(tmp_path: Path):
    """Verifies that in-memory training reduces loss and saves valid checkpoint with SHA256."""
    trainer = MicroNeuralTrainer(vocab_size=128, d_model=32, n_layers=1, lr=1e-2)
    sample_texts = [
        "def add(a, b): return a + b",
        "def sub(a, b): return a - b",
        "def mul(a, b): return a * b",
    ]

    metrics = trainer.train_step(sample_texts, epochs=4)
    assert metrics["final_loss"] <= metrics["initial_loss"]
    assert len(metrics["loss_history"]) == 4

    ckpt_dir = tmp_path / "model_ckpt"
    saved_meta = trainer.save_checkpoint(ckpt_dir, model_id="test_m1")
    assert Path(saved_meta["weights_path"]).exists()
    assert len(saved_meta["checksum"]) == 64

    # Verify reload
    reloaded = MicroNeuralTrainer(vocab_size=128, d_model=32, n_layers=1)
    reloaded.load_checkpoint(Path(saved_meta["weights_path"]))
    assert reloaded.model.count_parameters() == trainer.model.count_parameters()
