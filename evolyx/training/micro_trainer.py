"""Micro-Neural Engine: In-memory PyTorch neural model and training pipeline.

Enables real gradient descent training, backpropagation, and weight checkpointing
without downloading any large external models or files.
"""

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class MicroTransformerLM(nn.Module):
    """A lightweight, fully functional Transformer Language Model for local verification.

    Features multi-head self-attention, feedforward networks, layer normalization,
    and causal masking. Entire model is ~50,000 parameters and initializes in milliseconds.
    """

    def __init__(self, vocab_size: int = 128, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask so tokens cannot attend to future tokens
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MicroNeuralTrainer:
    """Manages real model initialization, training loops, evaluation, and artifact saving."""

    def __init__(self, d_model: int = 64, n_layers: int = 2, vocab_size: int = 128, lr: float = 1e-3):
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.lr = lr
        self.model = MicroTransformerLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers).to(self.device)

    def text_to_tensor(self, text: str, max_len: int = 64) -> torch.Tensor:
        """Converts ASCII characters directly to token IDs (0-127)."""
        tokens = [ord(c) % 128 for c in text[:max_len]]
        if not tokens:
            tokens = [0]
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    def train_step(self, text_samples: List[str], epochs: int = 5, lr: float = 1e-3) -> Dict[str, Any]:
        """Executes actual gradient descent backpropagation over training samples."""
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        loss_history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for text in text_samples:
                if len(text) < 2:
                    continue
                input_ids = self.text_to_tensor(text)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]

                optimizer.zero_grad()
                logits = self.model(inputs)
                loss = criterion(logits.view(-1, 128), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(text_samples))
            loss_history.append(avg_loss)
            logger.debug("Epoch %d: loss = %.4f", epoch + 1, avg_loss)

        return {
            "initial_loss": round(loss_history[0], 4) if loss_history else 0.0,
            "final_loss": round(loss_history[-1], 4) if loss_history else 0.0,
            "loss_history": [round(l, 4) for l in loss_history],
            "parameters_count": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }

    def save_checkpoint(self, output_dir: Path, model_id: str) -> Dict[str, str]:
        """Saves weights and SHA256 integrity checksum."""
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / "micro_model.pt"
        torch.save(self.model.state_dict(), weights_file)

        # Compute SHA256 checksum
        hasher = hashlib.sha256()
        with open(weights_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        checksum = hasher.hexdigest()

        manifest = {
            "model_id": model_id,
            "architecture": "MicroTransformerLM",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "sha256": checksum,
            "weights_file": str(weights_file.name),
        }
        manifest_file = output_dir / "model_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {"weights_path": str(weights_file), "checksum": checksum, "manifest_path": str(manifest_file)}

    def load_checkpoint(self, weights_path: Path) -> None:
        """Loads state dict into model."""
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 20) -> str:
        """Performs autoregressive token generation."""
        self.model.eval()
        input_ids = self.text_to_tensor(prompt)
        generated = list(input_ids[0].tolist())

        with torch.no_grad():
            for _ in range(max_new_tokens):
                cur_tensor = torch.tensor([generated[-64:]], device=self.device)
                logits = self.model(cur_tensor)
                next_token = torch.argmax(logits[0, -1, :]).item()
                generated.append(next_token)

        chars = [chr(t) if 32 <= t <= 126 else " " for t in generated]
        return "".join(chars)
