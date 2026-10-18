"""Unit tests for ArchitectureExplorer bounded architecture search and parameter counting."""

from evolyx.evolution.model_growth import ArchitectureExplorer


def test_architecture_explorer_growth_and_scaling():
    explorer = ArchitectureExplorer(base_d_model=64, base_n_layers=2)

    # 1. Base architecture specs
    base_specs = explorer.get_specs()
    assert base_specs["total_parameters"] > 0
    assert base_specs["vram_estimate_mb"] > 0.0

    # 2. Expand layers
    layer_expanded = explorer.propose_layer_expansion(add_layers=2)
    assert layer_expanded["n_layers"] == 4
    assert layer_expanded["total_parameters"] > base_specs["total_parameters"]

    # 3. MoE expansion
    moe_spec = explorer.propose_moe_expansion(num_experts=4, top_k=2)
    assert moe_spec["num_experts"] == 4
    assert moe_spec["top_k"] == 2
    assert moe_spec["is_moe"] is True
    assert moe_spec["total_parameters"] > base_specs["total_parameters"]
