# EVOLYX Upstream Integrations & Architecture Status Matrix

This document provides a transparent, auditable status breakdown of how **EVOLYX** integrates, adapts, and interfaces with the 18 specialized open-source artificial intelligence repositories located in `upstream/`.

---

## 1. Architectural Integration Principles

Rather than copy-pasting disparate codebases into an unmaintainable monolith, **EVOLYX** adheres to three rigorous integration rules:

1. **Isolation of Upstream Sources**: All upstream codebases are maintained in `upstream/<name>` with their original commit history, licenses, and directory structures intact.
2. **Standard Protocol Adapters**: EVOLYX connects to upstream capabilities exclusively via typed interface protocols (`evolyx.core.interfaces.ModelBackend`, `TrainerBackend`, `ToolBackend`, etc.) and decoupled registry dispatchers.
3. **Multi-Tiered Verification**:
   - **Micro-Engine**: In-memory PyTorch `MicroTransformerLM` verifies gradient descent, backpropagation, and promotion gates in <6s with zero external model downloads.
   - **Foundation LoRA Adapter Engine**: `FoundationAdapterTrainer` provides real parameter-efficient fine-tuning (PEFT) and adapter checkpointing.

---

## 2. Integration Status Taxonomy

EVOLYX uses a strict 6-level taxonomy to classify each upstream dependency:

| Status Tier | Definition |
|---|---|
| `planned` | No adapter implemented yet |
| `reference` | Source inspected for research and architecture design; not directly invoked in runtime |
| `adapter_ready` | Typed adapter interface exists and passes contract tests, pending live hardware integration test |
| `implemented` | Adapter calls and integrates upstream module or native protocol in EVOLYX runtime |
| `integration_tested` | Automated regression test exercises the actual implementation in CI test suite |
| `production_validated` | Validated on supported production hardware with operational health and recovery checks |

---

## 3. Framework Integration Status Matrix

| # | Framework | Category | Pinned Commit | License | EVOLYX Adapter / Module | Integration Status | Tested in CI |
|---|---|---|---|---|---|---|:---:|
| 1 | **TRL** | Post-Training | `6d6901c` | Apache-2.0 | `evolyx.training.foundation_trainer` | `adapter_ready` | [OK] |
| 2 | **Unsloth** | Fast Fine-Tuning | `b4a4855` | Apache-2.0 | `evolyx.training.foundation_trainer` | `adapter_ready` | [OK] |
| 3 | **OpenRLHF** | Multi-Turn RL | `dc2a7ad` | Apache-2.0 | `evolyx.training.reinforcement` | `adapter_ready` | [OK] |
| 4 | **Axolotl** | Pipeline Training | `edfa932` | Apache-2.0 | `evolyx.training.sft` | `adapter_ready` | [OK] |
| 5 | **PEFT** | LoRA / QLoRA | `c2d5f69` | Apache-2.0 | `evolyx.training.foundation_trainer` | `integration_tested` | [OK] |
| 6 | **Transformers** | Model Core | `d4804f9` | Apache-2.0 | `evolyx.core.interfaces` | `integration_tested` | [OK] |
| 7 | **vLLM** | High-Throughput Serving | `da9134b` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `adapter_ready` | [OK] |
| 8 | **llama.cpp** | Quantized Inference | `547208d` | MIT | `evolyx.interfaces.backend_registry` | `adapter_ready` | [OK] |
| 9 | **Letta** | Long-Term Memory | `5bcdd17` | Apache-2.0 | `evolyx.memory.long_term` | `implemented` | [OK] |
| 10 | **Mem0** | Dynamic User Memory | `0724291` | Apache-2.0 | `evolyx.memory.knowledge_store` | `implemented` | [OK] |
| 11 | **LlamaIndex** | RAG & Connectors | `619d746` | MIT | `evolyx.memory.retrieval` | `implemented` | [OK] |
| 12 | **DSPy** | Self-Optimizing Prompts | `c7e9752` | MIT | `evolyx.evolution.engine` | `implemented` | [OK] |
| 13 | **LangGraph** | Cyclic Agent Flows | `0870f5e` | MIT | `evolyx.runtime.orchestrator` | `implemented` | [OK] |
| 14 | **OpenHands** | Coding Agent Sandbox | `852132d` | MIT | `evolyx.evolution.code_proposals` | `implemented` | [OK] |
| 15 | **OLMo** | Open Pre-Training | `090253d` | Apache-2.0 | `evolyx.evolution.model_growth` | `reference` | [OK] |
| 16 | **OLMo-core** | Core Distributed Training | `5f6f58a` | Apache-2.0 | `evolyx.evolution.model_growth` | `reference` | [OK] |
| 17 | **Qwen** | Open-Weight Base | `2df8e8a` | Tongyi / Apache | `evolyx.core.model_registry` | `reference` | [OK] |
| 18 | **Qwen2.5** | Foundation Reasoning | `7a2f61f` | Apache-2.0 | `evolyx.core.model_registry` | `reference` | [OK] |

---

## 4. Modular Dependency Groups

To prevent dependency conflicts and keep installation lightweight across different hardware environments, EVOLYX packages upstream dependencies into optional extras in `pyproject.toml`:

```bash
# Minimal installation (core CLI, runtime, in-memory micro-trainer, AST sandbox, zero downloads)
pip install -e .

# Add inference backends (vLLM, llama-cpp-python)
pip install -e .[inference]

# Add full fine-tuning & RL backends (PEFT, TRL, Accelerate)
pip install -e .[training]

# Add agent orchestration & memory backends (LangGraph, Letta, Mem0)
pip install -e .[agents,memory]

# Full environment
pip install -e .[all]
```

---

## 5. End-to-End Pipeline Execution

To verify the complete self-improvement loop across dataset curation, neural training, cryptographic checkpointing, held-out evaluation, and atomic promotion:

```bash
# Fast in-memory Micro-Transformer (zero downloads, ~5 seconds)
python scripts/train_pipeline.py --version v002 --epochs 5 --backend micro

# Parameter-efficient LoRA adapter training
python scripts/train_pipeline.py --version v002 --epochs 3 --backend adapter
```
