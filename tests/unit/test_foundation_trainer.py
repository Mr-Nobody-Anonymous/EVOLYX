"""Unit tests for FoundationAdapterTrainer, CausalTransformerLM, SimpleChatTokenizer, and LoRA injection."""

import hashlib
import json
from pathlib import Path
import pytest
import torch
from evolyx.training.foundation_trainer import (
    CausalTransformerLM,
    FoundationAdapterTrainer,
    LoRALinear,
    SimpleChatTokenizer,
)


def test_simple_chat_tokenizer():
    tokenizer = SimpleChatTokenizer()
    messages = [
        {"role": "user", "content": "What is EVOLYX?"},
        {"role": "assistant", "content": "EVOLYX is an autonomous platform."},
    ]
    formatted = tokenizer.format_chat(messages)
    assert "<|im_start|>user\nWhat is EVOLYX?<|im_end|>\n" in formatted
    assert "<|im_start|>assistant\nEVOLYX is an autonomous platform.<|im_end|>\n" in formatted

    input_ids, labels = tokenizer.prepare_training_sample(messages)
    assert len(input_ids) > 0
    assert len(labels) == len(input_ids)

    # Prompt tokens must be masked with -100
    assert labels[0].item() == -100
    # Assistant tokens must not be masked
    assert any(l.item() != -100 for l in labels)

    decoded = tokenizer.decode(input_ids.tolist())
    assert "What is EVOLYX?" in decoded


def test_lora_linear_layer():
    d_in = 32
    d_out = 32
    rank = 4
    alpha = 8.0
    layer = LoRALinear(in_features=d_in, out_features=d_out, rank=rank, alpha=alpha)

    assert layer.scaling == 2.0
    assert layer.weight.requires_grad is False  # Frozen base weights
    assert layer.lora_A.requires_grad is True   # Trainable adapter A
    assert layer.lora_B.requires_grad is True   # Trainable adapter B

    x = torch.randn(2, 5, d_in)
    out = layer(x)
    assert out.shape == (2, 5, d_out)

    loss = out.sum()
    loss.backward()
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.weight.grad is None


def test_causal_transformer_clm_forward_and_generate():
    model = CausalTransformerLM(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
        lora_rank=4,
        lora_alpha=8.0,
    )

    input_ids = torch.randint(0, 128, (2, 16))
    labels = input_ids.clone()
    labels[:, :8] = -100  # Mask first 8 prompt tokens

    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (2, 16, 128)
    assert loss is not None
    assert loss.item() > 0.0

    loss.backward()
    # Base embeddings should have no gradients
    assert model.tok_embeddings.weight.grad is None
    # Trainable LoRA parameters in attention blocks must have gradients
    block0_lora = model.blocks[0].attn.q_proj.lora_A
    assert block0_lora.grad is not None

    # Autoregressive generation
    prompt_ids = torch.tensor([[2, 40, 50, 60]])
    gen_tokens = model.generate(prompt_ids, max_new_tokens=8)
    assert gen_tokens.shape[1] > prompt_ids.shape[1]


def test_foundation_adapter_trainer_e2e(tmp_path: Path):
    dataset_path = tmp_path / "train.jsonl"
    with open(dataset_path, "w", encoding="utf-8") as f:
        for _ in range(3):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Reverse string test"},
                    {"role": "assistant", "content": "s[::-1]"}
                ]
            }) + "\n")

    out_dir = tmp_path / "candidates"
    trainer = FoundationAdapterTrainer(output_dir=str(out_dir))

    candidate = trainer.train_candidate(
        base_model="Qwen/Qwen2.5-0.5B",
        dataset_path=str(dataset_path),
        config={"rank": 4, "alpha": 8.0, "epochs": 3, "lr": 0.01, "d_model": 32, "n_layers": 1, "n_heads": 2},
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
    assert cfg["training_metrics"]["epochs"] == 3

    # Cryptographic reload verification
    loaded = trainer.load_adapter(str(adapter_path))
    assert loaded["verified"] is True
    assert "blocks.0.attn.q_proj.lora_A" in loaded["state_dict"]

    # Generate with adapter
    gen_text = trainer.generate_with_adapter("Reverse string test", adapter_dir=str(adapter_path), max_new_tokens=8)
    assert isinstance(gen_text, str)
