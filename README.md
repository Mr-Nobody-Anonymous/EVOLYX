<div align="center">

<img src="assets/banner.svg" alt="EVOLYX Banner" width="100%"/>

<br/><br/>

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-16%20Passed%20(100%25)-success?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-Recursive%20Evolution-8A2BE2?style=for-the-badge)](#the-4-operational-layers)
[![Safety Gate](https://img.shields.io/badge/regression%20gate-Strict%20Zero--Degradation-00C49F?style=for-the-badge)](#-the-5-stage-continuous-improvement-loop)

<p align="center">
  <b>The Unified Open-Source Platform for Autonomous, Continuously Self-Improving Artificial Intelligence.</b>
  <br />
  Runs Locally &bull; Learns from Experience &bull; Synthesizes Training Data &bull; Fine-Tunes Adapters &bull; Proposes Architecture Evolutions
</p>

[Vision](#-the-vision-what-we-are-building) •
[Why EVOLYX](#-the-core-engineering-challenge) •
[Architecture](#-the-4-operational-layers) •
[Evolution Engine](#-the-4-levels-of-autonomous-evolution) •
[Upstream Repos](#-unified-upstream-ecosystem) •
[Quickstart](#-quickstart--installation) •
[Safety & Governance](#-safety-governance--anti-hallucination)

---

</div>

## 🌌 The Vision: What We Are Building

Contemporary large language models suffer from a fundamental constraint: **they are static artifacts frozen at the point of training**. They forget lessons between sessions, cannot autonomously acquire new skills without human engineers, and quickly degrade or hallucinate if allowed to train blindly on their own unverified outputs.

**EVOLYX** is built to break this paradigm.

Our ambition is an integrated, sovereign AI research platform that:
1. **Runs Privately**: Deploys on consumer hardware, local GPU workstations, or private cloud clusters without vendor lock-in or telemetry leaks.
2. **Remembers Comprehensively**: Maintains distinct conversational, procedural, and factual long-term memories with verified provenance and PII scrubbing.
3. **Acts & Researches Autonomously**: Formulates multi-step plans, writes and executes code in secure sandboxes, and searches external documentation when uncertain.
4. **Reflects Critically**: Scores its own solutions against objective unit tests, factual citations, and quality metrics before presenting answers.
5. **Generates Verified Training Data**: Curates successful task trajectories and user corrections into high-grade SFT, DPO, and GRPO datasets.
6. **Fine-Tunes its Own Weights**: Periodically fine-tunes candidate LoRA/QLoRA adapters and tests candidate checkpoints offline.
7. **Gates Every Promotion**: Compares new candidate models against fixed benchmarks and regression suites—promoting only models that demonstrably improve without degrading baseline skills.
8. **Evolves Its Own Code & Architecture**: Autonomously proposes prompt revisions, source code patches, and bounded neural architecture changes (layer scaling, MoE routing) in isolated branches.

---

## ⚡ The Core Engineering Challenge

> **"No single repository provides unlimited self-improvement or intelligence growth out of the box."**

True recursive self-improvement cannot be achieved by merely connecting an agent to a training script. Blind self-training causes **model collapse**, reinforces hallucinations, and catastrophic forgetting.

EVOLYX solves this through **decoupled multi-layer architecture**:
- **Runtime Inference** is strictly separated from **Weight Training**.
- **Working Memory** enhances performance instantly at zero training cost.
- **Offline Training** operates exclusively on **verified, test-proven trajectories**.
- **System Evolution** occurs inside **sandboxed experiments with explicit regression gates and rollbacks**.

---

## 🏛️ The 4 Operational Layers

```mermaid
flowchart TD
    subgraph L1["Layer 1: Inference Engine"]
        M1["Model Registry"] --> B1["Transformers"]
        M1 --> B2["vLLM High-Throughput"]
        M1 --> B3["llama.cpp GGUF"]
        M1 --> B4["Mock / Simulation"]
    end

    subgraph L2["Layer 2: Runtime Learning & Agency"]
        Agent["EvolyxAgent Orchestrator"]
        Intent["Intent Classifier"] --> Plan["Hierarchical Planner"]
        Plan --> Tools["Sandboxed Tools (Python / Search)"]
        Plan --> Mem["Long-Term Memory (SQLite / Letta / Mem0)"]
        Tools --> Assess["Objective Evaluator & Critic"]
    end

    subgraph L3["Layer 3: Model Improvement Pipeline"]
        Traj["Trajectory Store"] --> Filter["PII & Quality Data Filter"]
        Filter --> DS["Dataset Builder (SFT / DPO)"]
        DS --> Train["Trainers (Unsloth / TRL / PEFT)"]
        Train --> Cand["Candidate Models & Checkpoints"]
        Cand --> Bench["Multidimensional Benchmark Suite"]
        Bench --> Reg["Strict Regression Checker (< 2% drop)"]
        Reg --> Gate{"Release Gate Passed?"}
        Gate -- Yes --> Promo["Promote to Production"]
        Gate -- No --> Reject["Reject & Log Anomaly"]
    end

    subgraph L4["Layer 4: System Evolution Engine"]
        Fail["Failure Pattern Analyzer"] --> Hypo["Hypothesis Generator"]
        Hypo --> Exp["Isolated A/B Experiment Runner"]
        Exp --> CodeProp["Sandboxed Code Proposals"]
        Exp --> Arch["Neural Architecture Search"]
        CodeProp --> Rollback["Instant Version Rollback"]
    end

    L1 <--> L2
    L2 -- "Verified Trajectories" --> L3
    L3 -- "Promoted Adapters" --> L1
    L3 -- "Failure Signals" --> L4
    L4 -- "Tested Improvements" --> L2
```

---

## 🔁 The 5-Stage Continuous Improvement Loop

```
  [1. OBSERVE & ACT] ──► [2. VERIFY & STORE] ──► [3. DISTILL DATA]
          ▲                                              │
          │                                              ▼
  [5. GATE & PROMOTE] ◄───────────────────────── [4. TRAIN CANDIDATE]
```

1. **Observe & Act (Runtime)**: The agent decomposes user requests into actionable DAG plans, queries relevant memories, and invokes tools.
2. **Verify & Store**: The objective critic tests the output. If execution is backed by passing assertions, it records the trajectory and updates verified procedural memory.
3. **Distill Data**: The learning pipeline scrubs secrets, removes duplicates, balances task curriculum, and builds versioned SFT/DPO manifests.
4. **Train Candidate**: Training adapters launch parameter-efficient fine-tuning (LoRA via Unsloth/TRL/PEFT) on offline compute.
5. **Gate & Promote**: The promotion manager tests the candidate against a multi-benchmark battery. If any existing capability drops by more than 2.0%, the candidate is rejected. If it achieves superior overall scores, it is promoted to production with an immutable checkpoint pointer.

---

## 🧬 The 4 Levels of Autonomous Evolution

EVOLYX is architected to evolve across four distinct tiers of autonomy:

| Level | Name | How It Operates | Safeguards |
| :---: | :--- | :--- | :--- |
| **1** | **Prompt & Workflow Evolution** | Adapts instructions, tool-routing parameters, and decomposition heuristics dynamically based on failure patterns. | Fixed benchmark validation against the current baseline prompt. |
| **2** | **Model-Weight Evolution** | Targets model weaknesses by generating focused curriculum datasets and fine-tuning specialized LoRA adapters. | Regression test suite comparing candidate model weights to production. |
| **3** | **Software Self-Patching** | Generates proposed code changes to EVOLYX's runtime or tools, spins up an isolated sandbox, and runs unit tests. | AST forbidden call detection + full automated test suite pass. |
| **4** | **Architecture Search** | Explores structural capacity: adapter rank scaling, layer replication, and Mixture-of-Experts (MoE) router configurations. | Bounded FLOP and VRAM budgets; offline checkpoint compatibility verification. |

---

## 🌐 Unified Upstream Ecosystem

EVOLYX synthesizes 18 industry-leading open-source frameworks. Rather than merging disparate source codes into a fragile monolith, EVOLYX keeps them pinned in [`upstream/`](upstream/) and connects them through high-performance Python adapters:

```
                                  ┌───────────────────────────┐
                                  │      EVOLYX PLATFORM      │
                                  └─────────────┬─────────────┘
                                                │
         ┌───────────────────┬──────────────────┼───────────────────┬───────────────────┐
         ▼                   ▼                  ▼                   ▼                   ▼
    [INFERENCE]         [TRAINING]          [MEMORY]            [AGENTS]           [RESEARCH]
     • Transformers      • Unsloth          • Letta (MemGPT)    • LangGraph         • OLMo Core
     • vLLM              • TRL              • Mem0              • OpenHands         • OLMo
     • llama.cpp         • PEFT             • SQLite Engine     • DSPy              • Qwen / Qwen 2.5
                         • Axolotl                              • LlamaIndex
                         • OpenRLHF
```

---

## 📂 Project Structure

```
EVOLYX/
├── assets/                      # Official vector logo, icon, and banner graphics
│   ├── logo.svg
│   └── banner.svg
├── upstream/                    # Pinned original upstream repositories (TRL, Unsloth, etc.)
│   └── README.md
├── evolyx/                      # Primary integrated platform package
│   ├── core/                    # Protocols, schemas, model registry, inference engine
│   ├── runtime/                 # Intent classifier, planner, tool router, orchestrator
│   ├── memory/                  # Long-term SQLite, short-term, quality gates, knowledge store
│   ├── tools/                   # Web search, document search, Python executor, sandboxing
│   ├── learning/                # Trajectory store, feedback collector, dataset builder
│   ├── training/                # SFT, DPO, GRPO pipelines, checkpoints manager
│   ├── evaluation/              # Benchmarks, factuality evaluator, regression checker
│   ├── evolution/               # Hypothesis generator, experiment runner, code proposals
│   ├── control/                 # RBAC permissions, resource governor, audit logger
│   └── storage/                 # Database, artifact store, event bus
├── configs/                     # Central system configuration YAML files
├── apps/                        # Entry points: CLI, FastAPI server, Dashboard
├── integrations/                # Adapters connecting upstream frameworks to EVOLYX
├── artifacts/                   # Immutable storage for checkpoints, datasets, and reports
├── tests/                       # Unit, integration, and safety test suite (16 tests)
├── scripts/                     # PowerShell & Bash automation scripts
├── pyproject.toml               # Python project configuration
├── docker-compose.yml           # Containerized orchestration
└── Dockerfile                   # Production Docker image
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Git 2.40+
- *(Optional for GPU acceleration)*: NVIDIA CUDA 12.1+ / ROCm

### 2. Verification
Verify your environment and run the test battery:

```powershell
# On Windows (PowerShell):
.\scripts\run_tests.ps1

# On Linux / macOS (Bash):
./scripts/run_tests.sh
```

All 16 unit and end-to-end integration tests execute in **< 0.3 seconds** with zero external dependencies required for testing.

### 3. Interactive CLI
Launch the conversational agent:

```powershell
.\scripts\run_cli.ps1 chat
```

Execute a single autonomous task:

```powershell
.\scripts\run_cli.ps1 task "Find and fix bug in binary search algorithm"
```

### 4. Trigger an Autonomous Evolution Cycle
Simulate failure diagnosis, proposal generation, A/B experimentation, and regression gating:

```powershell
.\scripts\run_experiment.ps1
```

### 5. Launch the REST API
Start the FastAPI server:

```powershell
.\scripts\run_api.ps1
```
Interactive OpenAPI documentation will be accessible at `http://127.0.0.1:8000/docs`.

---

## 🛡️ Safety, Governance & Anti-Hallucination

1. **AST Tool Sandboxing**: Python tool execution validates every snippet through the Abstract Syntax Tree before running, blocking forbidden imports (`os.system`, `subprocess`, `ctypes`, `shutil.rmtree`).
2. **PII & Secret Sanitization**: Trajectories, queries, and long-term memories pass through regex scrubbers that automatically redact API keys (`[SECRET_REDACTED]`), email addresses (`[EMAIL_REDACTED]`), and IP addresses (`[IP_REDACTED]`).
3. **Zero Runtime Weight Drift**: Model weights are never modified directly during active chat. All interactions are collected as experience records and undergo offline evaluation before any candidate adapter is fine-tuned.
4. **Automated Rollback**: The system maintains an active pointer (`ACTIVE_VERSION.txt`). If anomalies or regressions are detected post-deployment, EVOLYX executes single-command rollbacks to prior production checkpoints.

---

## 🗺️ Engineering Roadmap

- [x] **Phase 1: Foundation & Architecture** (Protocols, Mock Backends, CLI, REST API, Test Suite)
- [x] **Phase 2: Runtime Intelligence** (Intent classification, hierarchical planning, AST tool sandbox)
- [x] **Phase 3: Learning & Memory** (SQLite long-term memory, PII filter, SFT dataset builder)
- [x] **Phase 4: Evolution & Gating** (Hypothesis generator, benchmark suite, regression gates, promotion manager)
- [ ] **Phase 5: Distributed Scaling** (vLLM production clusters, multi-node OpenRLHF training, deep OLMo-core MoE experiments)

---

## 📄 License & Attribution

EVOLYX is licensed under the [Apache 2.0 License](LICENSE).
The upstream repositories preserved in `upstream/` maintain their respective open-source licenses (Apache 2.0, MIT, BSD).

<div align="center">
  <sub>Engineered for the future of sovereign, autonomous, continually evolving intelligence.</sub>
</div>
