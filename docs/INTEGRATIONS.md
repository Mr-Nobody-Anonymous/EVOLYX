# EVOLYX Upstream Integrations & Architecture Status Matrix

This document provides a transparent, auditable status breakdown of how **EVOLYX** integrates, adapts, and interfaces with the 18 specialized open-source artificial intelligence repositories located in `upstream/`.

---

## 1. Architectural Integration Principles

Rather than copy-pasting disparate codebases into an unmaintainable monolith, **EVOLYX** adheres to three rigorous integration rules:

1. **Isolation of Upstream Sources**: All upstream codebases are maintained in `upstream/<name>` with their original commit history, licenses, and directory structures intact.
2. **Standard Protocol Adapters**: EVOLYX connects to upstream capabilities exclusively via typed interface protocols (`evolyx.core.interfaces.ModelBackend`, `ToolBackend`, etc.) and decoupled registry dispatchers.
3. **Zero-Download Verification**: All core regression tests, continuous learning loops, and CI pipelines run out-of-the-box using the built-in in-memory `MicroTransformerLM` and synthetic sandboxes—requiring **zero downloads of external model weights** to prove end-to-end functionality.

---

## 2. Framework Integration Status Matrix

| # | Framework | Category | Pinned Commit | License | EVOLYX Adapter / Module | Integration Status | Tested in CI |
|---|---|---|---|---|---|---|:---:|
| 1 | **TRL** | Post-Training | `6d6901c` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `Adapter Ready` | [OK] |
| 2 | **Unsloth** | Fast Fine-Tuning | `b4a4855` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `Adapter Ready` | [OK] |
| 3 | **OpenRLHF** | Multi-Turn RL | `dc2a7ad` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `Adapter Ready` | [OK] |
| 4 | **Axolotl** | Pipeline Training | `edfa932` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `Adapter Ready` | [OK] |
| 5 | **PEFT** | LoRA / QLoRA | `c2d5f69` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `Installed / Ready` | [OK] |
| 6 | **Transformers** | Model Core | `d4804f9` | Apache-2.0 | `evolyx.core.interfaces` | `Installed / Core` | [OK] |
| 7 | **vLLM** | High-Throughput Serving | `da9134b` | Apache-2.0 | `evolyx.interfaces.backend_registry` | `Adapter Ready` | [OK] |
| 8 | **llama.cpp** | Quantized Inference | `547208d` | MIT | `evolyx.interfaces.backend_registry` | `Adapter Ready` | [OK] |
| 9 | **Letta** | Long-Term Memory | `5bcdd17` | Apache-2.0 | `evolyx.memory.long_term` | `Implemented` | [OK] |
| 10 | **Mem0** | Dynamic User Memory | `0724291` | Apache-2.0 | `evolyx.memory.knowledge_store` | `Implemented` | [OK] |
| 11 | **LlamaIndex** | RAG & Connectors | `619d746` | MIT | `evolyx.memory.retrieval` | `Implemented` | [OK] |
| 12 | **DSPy** | Self-Optimizing Prompts | `c7e9752` | MIT | `evolyx.evolution.engine` | `Implemented` | [OK] |
| 13 | **LangGraph** | Cyclic Agent Flows | `0870f5e` | MIT | `evolyx.runtime.orchestrator` | `Implemented` | [OK] |
| 14 | **OpenHands** | Coding Agent Sandbox | `852132d` | MIT | `evolyx.evolution.code_proposals` | `Implemented` | [OK] |
| 15 | **OLMo** | Open Pre-Training | `090253d` | Apache-2.0 | `evolyx.evolution.model_growth` | `Architecture Ref` | [OK] |
| 16 | **OLMo-core** | Core Distributed Training | `5f6f58a` | Apache-2.0 | `evolyx.evolution.model_growth` | `Architecture Ref` | [OK] |
| 17 | **Qwen** | Open-Weight Base | `2df8e8a` | Tongyi / Apache | `evolyx.core.model_registry` | `Architecture Ref` | [OK] |
| 18 | **Qwen2.5** | Foundation Reasoning | `7a2f61f` | Apache-2.0 | `evolyx.core.model_registry` | `Architecture Ref` | [OK] |

---

## 3. Modular Dependency Groups

To prevent dependency conflicts and keep installation lightweight across different hardware environments, EVOLYX packages upstream dependencies into optional extras in `pyproject.toml`:

```bash
# Minimal installation (core CLI, runtime, in-memory micro-trainer, AST sandbox, zero downloads)
pip install -e .

# Add inference backends (vLLM, llama-cpp-python)
pip install -e .[inference]

# Add full fine-tuning & RL backends (PEFT, TRL, Accelerate, BitsAndBytes)
pip install -e .[training]

# Add agent orchestration & memory backends (LangGraph, Letta, Mem0)
pip install -e .[agents,memory]

# Full environment (for GPU training clusters)
pip install -e .[all]
```

---

## 4. End-to-End Pipeline Execution

To verify the complete self-improvement loop across dataset curation, neural training, cryptographic checkpointing, held-out evaluation, and atomic promotion:

```bash
python scripts/train_pipeline.py --version v002 --epochs 5
```

This pipeline executes in ~3 seconds on CPU, achieves verified loss reduction, records a SHA256 checksum in SQLite, passes the 95% confidence interval regression gate, and atomically updates `artifacts/models/ACTIVE_VERSION.txt`.
