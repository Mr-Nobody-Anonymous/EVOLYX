"""Model configuration generator, parameter estimator, and structural scaling explorer."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureSpec:
    name: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    vocab_size: int
    num_experts: int = 1
    expert_top_k: int = 1
    adapter_rank: Optional[int] = None
    total_params: int = 0
    estimated_vram_mb: float = 0.0


class ArchitectureExplorer:
    """Generates valid candidate architectures and estimates hardware requirements."""

    def __init__(self, base_d_model: int = 768, base_n_layers: int = 12, max_vram_mb: float = 4096.0):
        self.base_d_model = base_d_model
        self.base_n_layers = base_n_layers
        self.max_vram_mb = max_vram_mb

    def get_specs(self) -> Dict[str, Any]:
        params = self.calculate_params(num_layers=self.base_n_layers, hidden_dim=self.base_d_model)
        vram = self.estimate_vram_mb(params)
        return {
            "d_model": self.base_d_model,
            "n_layers": self.base_n_layers,
            "total_parameters": params,
            "vram_estimate_mb": vram,
        }

    def propose_layer_expansion(self, add_layers: int = 2) -> Dict[str, Any]:
        new_layers = self.base_n_layers + add_layers
        params = self.calculate_params(num_layers=new_layers, hidden_dim=self.base_d_model)
        return {
            "type": "layer_expansion",
            "n_layers": new_layers,
            "total_parameters": params,
            "vram_estimate_mb": self.estimate_vram_mb(params),
        }

    def propose_moe_expansion(self, num_experts: int = 4, top_k: int = 2) -> Dict[str, Any]:
        params = self.calculate_params(num_layers=self.base_n_layers, hidden_dim=self.base_d_model, num_experts=num_experts)
        return {
            "type": "moe_expansion",
            "is_moe": True,
            "num_experts": num_experts,
            "top_k": top_k,
            "total_parameters": params,
            "vram_estimate_mb": self.estimate_vram_mb(params),
        }

    @staticmethod
    def calculate_params(num_layers: int, hidden_dim: int, vocab_size: int = 32000, num_experts: int = 1) -> int:
        """Calculates approximate Transformer parameter count."""
        # Embeddings: vocab_size * hidden_dim
        embed_params = vocab_size * hidden_dim
        # Attention per layer: 4 * hidden_dim^2 (q, k, v, o)
        attn_params = 4 * (hidden_dim ** 2)
        # Feedforward per layer: 8 * hidden_dim^2 (gate/up/down) * num_experts
        ffn_params = 8 * (hidden_dim ** 2) * num_experts
        layer_params = attn_params + ffn_params
        total = embed_params + (num_layers * layer_params)
        return total

    @staticmethod
    def estimate_vram_mb(total_params: int, precision_bytes: int = 2) -> float:
        """Estimates model weights + KV cache footprint in MB for fp16/bf16."""
        weights_mb = (total_params * precision_bytes) / (1024 * 1024)
        overhead_mb = weights_mb * 0.35  # Activation / KV cache overhead
        return round(weights_mb + overhead_mb, 2)

    def generate_candidate_architectures(self) -> List[Dict[str, Any]]:
        """Generates concrete scaled architectures that fit within hardware limits."""
        candidates = []

        # 1. Base Dense Micro-Model
        c1_params = self.calculate_params(num_layers=12, hidden_dim=768, vocab_size=32000)
        candidates.append({
            "spec_id": "arch_dense_12x768",
            "type": "dense_transformer",
            "layers": 12,
            "hidden_dim": 768,
            "num_heads": 12,
            "num_experts": 1,
            "total_params": c1_params,
            "estimated_vram_mb": self.estimate_vram_mb(c1_params),
            "fits_in_budget": self.estimate_vram_mb(c1_params) <= self.max_vram_mb,
        })

        # 2. Sparse Mixture-of-Experts (MoE) Architecture
        c2_params = self.calculate_params(num_layers=12, hidden_dim=768, vocab_size=32000, num_experts=4)
        candidates.append({
            "spec_id": "arch_moe_12x768_e4",
            "type": "moe_transformer",
            "layers": 12,
            "hidden_dim": 768,
            "num_heads": 12,
            "num_experts": 4,
            "expert_top_k": 2,
            "total_params": c2_params,
            "estimated_vram_mb": self.estimate_vram_mb(c2_params),
            "fits_in_budget": self.estimate_vram_mb(c2_params) <= self.max_vram_mb,
        })

        # 3. Layer Replicated / Deep Expanded Model
        c3_params = self.calculate_params(num_layers=16, hidden_dim=768, vocab_size=32000)
        candidates.append({
            "spec_id": "arch_expanded_16x768",
            "type": "layer_expanded",
            "layers": 16,
            "hidden_dim": 768,
            "num_heads": 12,
            "num_experts": 1,
            "total_params": c3_params,
            "estimated_vram_mb": self.estimate_vram_mb(c3_params),
            "fits_in_budget": self.estimate_vram_mb(c3_params) <= self.max_vram_mb,
        })

        return candidates
