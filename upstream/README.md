# EVOLYX Upstream Reference Ecosystem

This directory contains the original, pinned open-source foundation repositories that power the EVOLYX ecosystem.

These upstream frameworks are intentionally preserved in their native structures to avoid dependency conflicts, preserve upstream Git histories, and permit modular updates:

| Framework | Role in EVOLYX | Integration Adapter in EVOLYX |
| :--- | :--- | :--- |
| **Hugging Face Transformers** | Model architecture, weights loading, tokenizers | `integrations/model_backends/transformers_backend.py` |
| **PEFT** | Parameter-Efficient Fine-Tuning (LoRA, QLoRA) | `integrations/training_backends/peft_adapter.py` |
| **Unsloth** | Ultra-fast low-memory fine-tuning & kernel optimizations | `integrations/training_backends/unsloth_adapter.py` |
| **TRL** | SFT, DPO, ORPO, and reinforcement learning | `integrations/training_backends/trl_adapter.py` |
| **Axolotl** | Configuration-driven multi-turn training pipeline | `integrations/training_backends/` |
| **OpenRLHF** | Distributed RLHF/RLAIF and agentic reinforcement learning | `integrations/training_backends/` |
| **Letta** (MemGPT) | Persistent stateful agent memory and recall | `integrations/memory_backends/letta_adapter.py` |
| **Mem0** | Personalized memory graph & entity extraction | `integrations/memory_backends/mem0_adapter.py` |
| **DSPy** | Algorithmic prompt and pipeline optimization | `evolyx/evaluation/` & `evolyx/evolution/` |
| **LangGraph** | Multi-agent cyclical state graphs and workflows | `integrations/agent_backends/langgraph_adapter.py` |
| **OpenHands** | Autonomous software engineering agent sandbox | `integrations/agent_backends/openhands_adapter.py` |
| **LlamaIndex** | Document retrieval, indexing, and RAG pipelines | `integrations/retrieval_backends/llamaindex_adapter.py` |
| **llama.cpp** | Local quantized GGUF inference on consumer hardware | `integrations/model_backends/llama_cpp_backend.py` |
| **vLLM** | Production PagedAttention high-throughput serving | `integrations/model_backends/vllm_backend.py` |
| **Qwen / Qwen 2.5** | Alibaba flagship open-weight foundation models | Base Model Architecture |
| **OLMo / OLMo Core** | AllenAI pretraining, checkpointing, and MoE research | Architecture Evolution Research Track |

---

### Automated Upstream Setup

To clone or update all 18 upstream repositories locally:

```powershell
powershell -ExecutionPolicy Bypass -File .\clone_repos.ps1
```
*(All clones use shallow `--depth 1` to conserve disk space and speed up synchronization).*
