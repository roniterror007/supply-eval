# PlanFlow AI 2.0: Supply-Eval

[![Tests](https://img.shields.io/badge/pytest-39%20passed-success)](https://github.com/roniterror007/supply-eval)
[![Compliance](https://img.shields.io/badge/Pydantic%20V2-100%25%20Verified-blue)](https://github.com/roniterror007/supply-eval)
[![Solver](https://img.shields.io/badge/OR%20Optimizer-HiGHS%20MILP-orange)](https://github.com/roniterror007/supply-eval)
[![Security](https://img.shields.io/badge/Governance-Ed25519%20Signed-green)](https://github.com/roniterror007/supply-eval)
[![UI](https://img.shields.io/badge/Cockpit-Streamlit-red)](https://github.com/roniterror007/supply-eval)

> **Transforming Volatile Demand Forecasting into Solver-Certified, Physical-Constraint-Feasible Enterprise Supply Chain Planning.**  
> *Architecture: Specialized Peer Agents ➔ HELICASE Evidence Graph ➔ GPU Digital Twin ➔ Multi-Site HiGHS MILP Master Optimizer ➔ Two-Part Judge Verification Gate ➔ Risk-Adaptive Autonomy ➔ Executive Boss Cryptographic Governance (Ed25519).*

Repository: **[https://github.com/roniterror007/supply-eval.git](https://github.com/roniterror007/supply-eval.git)**

---

## 📑 Table of Contents

- [Executive Overview](#-executive-overview)
- [System Architecture (SPORD & Multi-Tier Governance)](#-system-architecture-spord--multi-tier-governance)
- [Core Engines & Key Technical Innovations](#-core-engines--key-technical-innovations)
  - [1. Centralized Cost Engine (Single Source of Truth)](#1-centralized-cost-engine-single-source-of-truth)
  - [2. Operations Research Master Optimizer (HiGHS MILP)](#2-operations-research-master-optimizer-highs-milp)
  - [3. Closed-Loop Agent Recommendations](#3-closed-loop-agent-recommendations)
  - [4. HELICASE Temporal Evidence Knowledge Graph](#4-helicase-temporal-evidence-knowledge-graph)
  - [5. Two-Part Deterministic Judge Verification Gate](#5-two-part-deterministic-judge-verification-gate)
  - [6. Tier 3 Risk-Adaptive Governance & Executive Boss (Ed25519)](#6-tier-3-risk-adaptive-governance--executive-boss-ed25519)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
  - [1. Run Complete Test Suite (39 Passing Tests)](#1-run-complete-test-suite-39-passing-tests)
  - [2. Launch Interactive Streamlit Cockpit](#2-launch-interactive-streamlit-cockpit)
  - [3. Run Single Planning Cycle CLI Demo](#3-run-single-planning-cycle-cli-demo)
  - [4. Run Automated Governance Evaluation Benchmark](#4-run-automated-governance-evaluation-benchmark)
  - [5. Run 200-Scenario Adversarial Stress Test](#5-run-200-scenario-adversarial-stress-test)
  - [6. Run Long-Duration Soak & Memory Stability Benchmark](#6-run-long-duration-soak--memory-stability-benchmark)
  - [7. Generate 10,000-Record Enterprise Data Lake](#7-generate-10000-record-enterprise-data-lake)
  - [8. Train PyTorch GPU Neural Surrogate](#8-train-pytorch-gpu-neural-surrogate)
- [Empirical Benchmarks & Verification Results](#-empirical-benchmarks--verification-results)
- [License](#-license)

---

## 🎯 Executive Overview

Modern supply chain management suffers from a disconnect between **predictive AI** (which forecasts demand with high uncertainty) and **operational execution** (which must satisfy hard factory capacities, multi-level bills of materials, transit lead times, labor agreements, and service-level agreements).

**PlanFlow AI 2.0 (Supply-Eval)** bridges this gap by replacing unconstrained LLM decision-making with a **hybrid AI + Operations Research (OR)** architecture:
1. **Agents Diagnose & Structure Facts:** 8 specialist peer agents ingest ERP, CRM, MES, and WMS data, critique pipeline estimates, and populate a temporal evidence graph.
2. **Operations Research Solves the Network:** An integer linear programming (MILP) solver computes provably optimal, constraint-adherent production allocations, overtime shifts, inter-plant transfers, and supplier expedites.
3. **The Optimizer Drives Recommendations:** Downstream agent recommendations are directly synthesized from the optimizer's solution rather than isolated guesses.
4. **Cryptographic Executive Governance:** Plans pass through a two-part deterministic verification gate (separating physical conservation laws from soft quality scores) and receive an **Ed25519 cryptographic signature** before deployment.

---

## 🏛️ System Architecture: SPORD & Multi-Tier Governance

```mermaid
flowchart TD
    classDef data fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#fff;
    classDef agents fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef engine fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef gov fill:#451a03,stroke:#fbbf24,stroke-width:2px,color:#fff;

    subgraph Tier 1: Sensing, Diagnosis & Peer Dialogue
        D[(Enterprise Data Lake<br/>SAP, Salesforce, MES, WMS)] ::: data --> EG[HELICASE Evidence Graph<br/>Provenance DAG & Staleness] ::: data
        EG --> PA[Specialist Peer Agents<br/>Forecast, Demand, Factory, Supply] ::: agents
        PA <--> DIALOGUE[Peer Collaborative Review<br/>Mutual Error Correction + Red Team] ::: agents
    end

    subgraph Tier 2: Operations Research Core
        DIALOGUE --> COST[Centralized Cost Engine<br/>Single Source of Truth] ::: engine
        COST --> MILP{Master MILP Planner<br/>HiGHS Linear Optimizer<br/>Multi-Site Network Allocation} ::: engine
        MILP --> ORACLE[Feasibility Oracle<br/>Flow & Capacity Conservation] ::: engine
        MILP --> TWIN[GPU Digital Twin<br/>Monte Carlo Shock Propagation] ::: engine
    end

    subgraph Tier 3: Verification & Cryptographic Governance
        MILP --> REC[Recommendation Agent<br/>MILP-Driven Ranked Actions] ::: agents
        ORACLE --> JUDGE[Judge Verification Gate<br/>Hard Invariants + Soft Scoring] ::: gov
        TWIN --> JUDGE
        REC --> JUDGE
        JUDGE --> AUTO[Risk-Adaptive Autonomy Engine<br/>Spend & Uncertainty Thresholds] ::: gov
        AUTO --> BOSS[Executive Boss Agent<br/>Ed25519 Asymmetric Signature] ::: gov
        BOSS --> EXEC((Approved Executable Plan<br/>Signed Decision Memo)) ::: data
    end
```

---

## 🔬 Core Engines & Key Technical Innovations

### 1. Centralized Cost Engine (Single Source of Truth)
Located at [`src/core/cost_engine.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/core/cost_engine.py).  
Loads all parameters directly from [`configs/business_rules.yaml`](file:///c:/Users/Ronit/Desktop/Forecasting/configs/business_rules.yaml). Eliminates cost discrepancies across the optimizer, objective functions, and agent recommendations:
- **Factory Production Rates & Overtime Multipliers** for all 5 enterprise plants (Stuttgart, Detroit, Yokohama, Wroclaw, Monterrey).
- **Inter-Plant Logistics Matrix:** Exact point-to-point transfer costs.
- **Critical Material Profiles:** Standard lead times, compressed expedite lead times, unit expedite costs, and minimum order quantities.
- **EPA/ESG Benchmark Carbon Cost:** Converted into a dollar penalty (\$50/metric ton = \$0.05/kg $\text{CO}_2\text{e}$) directly penalizing freight and expediting in the MILP objective function.

### 2. Operations Research Master Optimizer (HiGHS MILP)
Located at [`src/optimization/master_planner.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/optimization/master_planner.py).  
Uses SciPy's HiGHS solver to find provably optimal integer programming solutions under physical laws:
- **`solve(...)`:** Optimized 2-factory allocation with backward-compatible support for legacy calling conventions.
- **`solve_network(...)`:** Simultaneous $K$-plant network optimization across all compatible global facilities.
- **Physical Bounds Enforced:**
  $$\sum_{f} x_{f} + x_{\text{unmet}} = \text{Target Demand}$$
  $$x_{f} - \lambda_{\text{ot}} \cdot h_{f} \le \text{Capacity}_{f}$$
  $$\sum_{f} \text{BOM}_{m} \cdot x_{f} - x_{\text{expedite}, m} - x_{\text{spot}, m} \le \text{Usable Stock}_{m}$$
- **Cryptographic Constraint Fingerprinting:** Computes a SHA-256 hash over the constraint matrix $A$, boundary vectors $l, u$, and cost vector $c$, returning an `OptimizationCertificateContract`.

### 3. Closed-Loop Agent Recommendations
Located at [`src/agents/peer_agents.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/agents/peer_agents.py#L469-L570).  
The `RecommendationAgent` consumes `milp_plan` directly from the orchestrator workflow:
- Translates solver allocations into specific, prioritized business recommendations (e.g. `SCHEDULE_OVERTIME`, `SHIFT_FACTORY_TRANSFER`, `EXPEDITE_CRITICAL_SUPPLIER`, `EMERGENCY_SPOT_SOURCING`).
- Ensures reported ROI, protected revenue, and mitigation expenditures match solver economics to the exact cent.

### 4. HELICASE Temporal Evidence Knowledge Graph
Located at [`src/knowledge_graph/evidence_graph.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/knowledge_graph/evidence_graph.py).  
- **Directed Acyclic Graph (DAG):** Tracks data lineage from upstream enterprise sources (ERP, CRM, MES, WMS) to intermediate agent claims.
- **Compounded Trajectory Uncertainty:**
  $$U_{\text{trajectory}} = 1 - \prod_{i=1}^n (1 - u_i)$$
- **Temporal Replay:** Replays historical planning cycles to detect stale or falsified operational assertions.

### 5. Two-Part Deterministic Judge Verification Gate
Located at [`src/verification/math_verifier.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/verification/math_verifier.py) and [`src/verification/grounding_checker.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/verification/grounding_checker.py).  
Separates non-negotiable physical laws from soft heuristics:
- **Hard Constraint Audit:** Flow conservation, machine capacity ceilings, and material balances under a strict **7-day transit delay cutoff** (delayed shipments cannot be used for current cycle demand). A single hard violation yields an immediate `FAIL` verdict.
- **Enterprise Grounding:** Verifies that all referenced business lines, factories, and materials exist in enterprise master records to eliminate hallucination.

### 6. Tier 3 Risk-Adaptive Governance & Executive Boss (Ed25519)
Located at [`src/governance/autonomy_policy.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/governance/autonomy_policy.py) and [`src/agents/executive_boss.py`](file:///c:/Users/Ronit/Desktop/Forecasting/src/agents/executive_boss.py).  
- Evaluates actions against corporate financial exposure limits, uncertainty scores, and customer SLA risk.
- Signs approved decision memos with asymmetric cryptography (**Ed25519**), outputting an audit-ready, tamper-proof hex signature.

---

## 📂 Repository Structure

```text
├── configs/
│   └── business_rules.yaml              # Master enterprise data, factory capacities, costs, BOMs
├── src/
│   ├── core/
│   │   ├── contracts.py                 # Pydantic V2 immutable data contracts with validation
│   │   ├── cost_engine.py               # Centralized cost engine (Single Source of Truth)
│   │   ├── gpu_utils.py                 # GPU memory & CUDA telemetry manager
│   │   └── logger.py                    # Rich logging & audit trail handlers
│   ├── models/
│   │   ├── neural_surrogate.py          # PyTorch multi-quantile neural forecasting surrogate
│   │   └── trainer.py                   # PyTorch GPU training routine
│   ├── data_generator/
│   │   └── generator.py                 # 10,000-record synthetic enterprise data lake generator
│   ├── optimization/
│   │   ├── master_planner.py            # HiGHS MILP solver (solve & solve_network)
│   │   ├── objectives.py                # Multi-objective cost, CVaR, and carbon formulations
│   │   ├── feasibility_oracle.py        # Independent post-solve mathematical verifier
│   │   ├── robust_planner.py            # Monte Carlo disruption sampler & CVaR evaluator
│   │   └── candidate_generator.py       # Strategic Pareto frontier candidate generator
│   ├── knowledge_graph/
│   │   ├── evidence_graph.py            # HELICASE temporal evidence DAG
│   │   ├── provenance_graph.py          # Lineage & audit trail tracker
│   │   ├── truth_maintenance.py         # Contradiction resolution & root-cause tracing
│   │   └── uncertainty.py               # Uncertainty compounding calculator
│   ├── digital_twin/
│   │   ├── simulator.py                 # Closed-loop digital twin shock simulator
│   │   ├── event_simulator.py           # Discrete-event simulation routines
│   │   ├── learned_residual.py          # Sim-to-real residual error calibrator
│   │   └── sim_to_real_calibration.py   # Statistical parameter calibration
│   ├── governance/
│   │   ├── autonomy_policy.py           # Risk-adaptive autonomy engine
│   │   └── approval_simulator.py        # Tiered approval policy simulator
│   ├── verification/
│   │   ├── math_verifier.py             # Deterministic conservation & arithmetic verifier
│   │   └── grounding_checker.py         # Enterprise catalog grounding checker
│   ├── agents/
│   │   ├── base_agent.py                # Base agent class with contract enforcement
│   │   ├── peer_agents.py               # 8 Specialist peer agents
│   │   ├── peer_dialogue.py             # Collaborative critique & mutual correction engine
│   │   ├── red_team_challenger.py       # Adversarial falsification agent
│   │   ├── judge_manager.py             # Tier 2 Judge / Manager verification agent
│   │   ├── executive_boss.py            # Tier 3 Executive Boss governance agent (Ed25519)
│   │   ├── orchestrator.py              # End-to-end 3-tier planning orchestrator
│   │   └── llm_client.py                # Fast probe & offline-resilient local LLM client
│   ├── evaluation/
│   │   ├── metrics.py                   # MAE, RMSE, WAPE, Precision, Recall, F1
│   │   ├── calibration.py               # Expected Calibration Error (ECE), Brier score
│   │   ├── decision_metrics.py          # Regret, stability, and resilience metrics
│   │   └── resilience_benchmark.py      # 4-track OOD stress test benchmark runner
│   └── dashboard/
│       └── app.py                       # Modern 5-tab Streamlit planning cockpit
├── tests/
│   ├── test_adversarial_data.py         # Stress test, two-part gate, Ed25519 signing
│   ├── test_autonomy_policy.py          # Autonomy policy limits & escalation checks
│   ├── test_contracts.py                # Pydantic contract conservation checks
│   ├── test_emergency_sourcing.py       # Bounded spot sourcing & governance property tests
│   ├── test_evaluation.py               # Metric calculation & calibration tests
│   ├── test_evidence_graph.py           # Provenance DAG, temporal replay, staleness
│   ├── test_gpu_models.py               # PyTorch GPU inference & CUDA tests
│   ├── test_knowledge_graph.py          # Knowledge graph contradiction tests
│   ├── test_optimizer.py                # HiGHS MILP solver & certificate tests
│   ├── test_peer_dialogue.py            # Mutual peer correction & blackboard tests
│   ├── test_red_team.py                 # Red-team challenger falsification tests
│   ├── test_resilience_benchmark.py     # 4-track resilience metrics tests
│   ├── test_robust_planner.py           # CVaR Monte Carlo scenario tests
│   ├── test_soak_reliability.py         # Long-duration soak & memory stability test
│   ├── test_uncertainty_propagation.py  # HELICASE trajectory uncertainty tests
│   └── test_verification.py             # Math verifier & grounding tests
├── scripts/
│   ├── generate_data.py                 # CLI to generate 10,000-record dataset
│   ├── train_model.py                   # CLI to train neural surrogate on GPU
│   ├── run_agent_pipeline.py            # CLI single planning cycle demo
│   ├── run_evaluation.py                # CLI automated benchmark runner
│   ├── run_soak_test.py                 # CLI soak & RSS memory benchmark
│   └── stress_test_suite.py             # 200-scenario adversarial stress test
├── data/                                # Parquet datasets and JSON benchmark reports
├── checkpoints/                         # PyTorch model weights (best_surrogate_model.pt)
├── pytest.ini                           # Pytest configuration
├── conftest.py                          # Sys.path configuration
├── requirements.txt                     # Production dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 💻 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/roniterror007/supply-eval.git
cd supply-eval
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify GPU Acceleration (Optional)
PlanFlow AI automatically detects and uses CUDA GPUs (fallback to CPU is automatic):
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🚀 How to Run

### 1. Run Complete Test Suite (39 Passing Tests)
Executes all unit, contract, optimizer, emergency sourcing, cryptographic, and resilience tests:
```bash
pytest -v
```

### 2. Launch Interactive Streamlit Cockpit
Starts the web application dashboard:
```bash
streamlit run src/dashboard/app.py
```
Open **`http://localhost:8501`** in your browser.  
Features:
- **Tab 1: Planning Cockpit & Digital Twin** — Live hardware telemetry, scenario sliders, shared blackboard, 6 agent cards, and Section 7 executive approval flow.
- **Tab 2: OR Optimizer & Feasibility (MILP)** — Optimal decision variable table, centralized costs, and strategic Pareto candidate frontier.
- **Tab 3: Evidence Knowledge Graph (HELICASE)** — Provenance DAG, temporal replay, and financial exposure human review rankings.
- **Tab 4: 4-Track Resilience Benchmark** — Stress-test metrics and tail risk analysis.
- **Tab 5: 10,000 Dataset Explorer** — Data lake filtering and exploration.

### 3. Run Single Planning Cycle CLI Demo
Executes a single end-to-end planning cycle across all 8 peer agents, Red-Team challenger, Judge verification gate, HiGHS MILP solver, and Executive Boss sign-off with terminal visualization:
```bash
python scripts/run_agent_pipeline.py
```

### 4. Run Automated Governance Evaluation Benchmark
Runs evaluation across ground-truth test data and audits contract compliance, conservation laws, and approval rates:
```bash
python scripts/run_evaluation.py --agent-samples 30
```

### 5. Run 200-Scenario Adversarial Stress Test
Evaluates performance across 4 disruption regimes (Favorable, Challenging, Severe OOD Disruption, and Poisoned Data):
```bash
python scripts/stress_test_suite.py
```

### 6. Run Long-Duration Soak & Memory Stability Benchmark
Audits host RSS memory drift, CUDA handle cleanup, and MILP p95 solve latency over repeated cycles:
```bash
python scripts/run_soak_test.py --iterations 25
```

### 7. Generate 10,000-Record Enterprise Data Lake
*(Pre-generated data is already included in `data/`, but can be regenerated at any time):*
```bash
python scripts/generate_data.py
```

### 8. Train PyTorch GPU Neural Surrogate
*(Pre-trained weights are already included in `checkpoints/best_surrogate_model.pt`):*
```bash
python scripts/train_model.py
```

---

## 📊 Empirical Benchmarks & Verification Results

### Pytest Verification Suite
```text
============================= test session starts =============================
tests/test_adversarial_data.py ..                                        [  5%]
tests/test_autonomy_policy.py .                                          [  7%]
tests/test_contracts.py ...                                              [ 15%]
tests/test_emergency_sourcing.py .......                                 [ 33%]
tests/test_evaluation.py ...                                             [ 41%]
tests/test_evidence_graph.py ....                                        [ 51%]
tests/test_gpu_models.py ...                                             [ 58%]
tests/test_knowledge_graph.py ..                                         [ 64%]
tests/test_optimizer.py ..                                               [ 69%]
tests/test_peer_dialogue.py .                                            [ 71%]
tests/test_red_team.py .                                                 [ 74%]
tests/test_resilience_benchmark.py ...                                   [ 82%]
tests/test_robust_planner.py .                                           [ 84%]
tests/test_soak_reliability.py .                                         [ 87%]
tests/test_uncertainty_propagation.py ..                                 [ 92%]
tests/test_verification.py ...                                           [100%]

============================= 39 passed in 30.49s =============================
```

### End-to-End Governance Audit Metrics
| Metric | Benchmark Result | Specification Target | Status |
|---|:---:|:---:|:---:|
| **Pydantic Contract Compliance** | **100.0%** | 100.0% | ✅ PASS |
| **Deterministic Verification Score** | **100.0 / 100.0** | $\ge 90.0$ | ✅ PASS |
| **Unit Conservation & Flow Balance** | **0.0% error** | $\le 0.5\%$ | ✅ PASS |
| **Machine Capacity Ceiling Adherence** | **100.0%** | 100.0% | ✅ PASS |
| **Material Balance & Cutoff Accuracy** | **100.0%** | 100.0% | ✅ PASS |
| **Financial ROI Math Consistency** | **Exact** | Exact | ✅ PASS |
| **Enterprise Master Data Grounding** | **100.0%** | 100.0% | ✅ PASS |
| **Executive Boss Approval Rate** | **100.0%** | $\ge 95.0\%$ | ✅ PASS |
| **Cryptographic Audit Signature** | **Ed25519 Valid** | Asymmetric | ✅ PASS |

---

## 📜 License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
