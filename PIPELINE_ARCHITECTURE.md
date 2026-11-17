# 8-Layer Federated Learning + MORL Pipeline Architecture (`fl_pipeline/`)

## Overview

`fl_pipeline/` is a modular, reproducible 8-layer Python/NumPy package for Multi-Objective Reinforcement Learning (MORL) client scheduling in heterogeneous Federated Learning (FL), optimizing the trade-off between **Model Accuracy**, **Round Latency**, **Carbon Emissions ($\text{gCO}_2$)**, and **Participation Fairness (Gini)**.

> **Data Provenance Note:** Only `PC1` (`0.065 kWh` over `190 min`, $\approx 20.526\text{ W}$ mean power) is anchored to a physical wall-meter reading. All `PC2–PC4` hardware ratios, network transfer times, coordinator draw, and the 10,000-row dataset (`data/calibrated_telemetry_10k.csv`) are explicitly labeled synthetic/calibrated research scenarios (`SYNTHETIC_CALIBRATED_FROM_PC1_METER`).

---

## The 8 Layers

| Layer | Module | Responsibility | Key Inputs | Key Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Layer 1: Data Partitioning** | `fl_pipeline/data.py` | Synthetic Gaussian classification generation, IID uniform split, and configurable Dirichlet($\alpha$) non-IID partitioning with fixed seeds, zero empty partitions, and disjoint test set. | `n_clients`, `mode` (`iid` / `dirichlet`), `alpha`, `n_train`, `n_test`, `seed` | `partitions: List[(X_i, y_i)]`, `test_data: (X_test, y_test)`, `class_distribution` |
| **Layer 2: Client Hardware & Local Training** | `fl_pipeline/clients.py` | `FederatedClient` profile (`ram_gb`, `gpu_vram_gb`, `relative_speed`, `active_power_w`, `idle_power_w`, `bandwidth_mbps`, `grid_kgco2_per_kwh`, `availability`) and numerically stable full-batch NumPy softmax gradient descent. | `global_weights`, `epochs`, `lr`, client partition | `LocalTrainResult(weights, num_samples, initial_loss, final_loss)` |
| **Layer 3: Network, Energy & Carbon Accounting** | `fl_pipeline/network_energy.py` | Synchronous parallel round latency, active/idle/coordinator energy (Wh & kWh), and per-client carbon intensity accounting. | `clients`, `selected_ids`, `base_train_seconds`, `aggregation_seconds`, `coordinator_power_w`, `payload_mb` | `RoundTelemetry` with `round_latency_s`, `total_wh`, `total_co2_g`, and `per_client` breakdown |
| **Layer 4: Sample-Weighted FedAvg** | `fl_pipeline/fedavg.py` | Verifiable sample-weighted Federated Averaging ($w_k = n_k / \sum_j n_j$, $\sum_k w_k = 1.0$). | `local_weights`, `sample_counts` | `(aggregated_weights, normalized_weights)` |
| **Layer 5: Scheduling Policies & MORL Sweep** | `fl_pipeline/schedulers.py` | Client subset schedulers (`Random`, `RoundRobin`, `FastestClient`, `LowestCarbon`, `WeightedHeuristic`, `FairnessGini`, `TabularQLearning`) and multi-preference MORL scalarization sweep. | `clients`, `round_idx`, `participation_counts`, `acc` | `selected_ids: Tuple[int, ...]`, `List[MORLSweepPoint]` |
| **Layer 6: RL Environment** | `fl_pipeline/rl_env.py` | Gym-style `FederatedRLEnv` (`reset`, `step`, `validate_action`, `discretize_state`) with 15-D continuous state vector, 4-D vector reward, and latency/carbon budget constraint tracking. | Action index or client tuple `(i, j, ...)` | `(obs, scalar_reward, done, info)` including `vector_reward` & `constraints` |
| **Layer 7: Experiment Runner & Telemetry Loader** | `fl_pipeline/experiment.py` | Loads `data/calibrated_telemetry_10k.csv`, calibrates client profiles, trains RL agents, evaluates all 7 policies, and computes participation Gini. | `mode`, `alpha`, `rounds`, `q_episodes`, `use_calibrated_telemetry` | Structured experiment dictionary & `PolicyEvaluationResult` records |
| **Layer 8: Pareto Frontier & Verification** | `fl_pipeline/schedulers.py`, `tests_pipeline.py` | Extracts non-dominated 3D Pareto frontiers ($\max \text{Accuracy}, \min \text{Latency}, \min \text{CO}_2$) across policies and MORL preference weights. | `Sequence[MORLSweepPoint]` or `Sequence[PolicyEvaluationResult]` | Pareto-optimal subset with `pareto_optimal=True` flags |

---

## Core Mathematical Formulations

1. **Parallel Round Latency:**
   $$T_{\text{round}} = \max_{i \in S} \left( T_{\text{download}, i} + \frac{T_{\text{base}} \cdot \xi_i}{\text{speed}_i} + T_{\text{upload}, i} \right) + T_{\text{agg}}$$
2. **Active, Idle, and Coordinator Energy (Wh):**
   $$E_i = \frac{P_{\text{active}, i} \cdot T_{\text{train}, i} + P_{\text{idle}, i} \cdot (T_{\text{round}} - T_{\text{train}, i})}{3600}, \quad E_{\text{coord}} = \frac{P_{\text{coord}} \cdot T_{\text{round}}}{3600}$$
3. **Per-Client Carbon Accounting ($\text{gCO}_2$):**
   $$\text{CO}_{2, \text{total}} = E_{\text{coord}} \cdot \text{CI}_{\text{coord}} + \sum_{i=1}^{N} E_i \cdot \text{CI}_i \quad \left(\text{where } E_i\text{ in kWh} \times \text{CI}_i\text{ in kgCO}_2/\text{kWh} \times 1000 = \text{Wh}_i \times \text{CI}_i\right)$$
4. **4D Vector Reward & Linear Scalarization:**
   $$\mathbf{r}_t = \begin{bmatrix} \Delta \text{Acc}_t \\ -T_{\text{round}, t} \\ -\text{CO}_{2, t} \\ -\text{Gini}_t \end{bmatrix}, \quad R_t = w_{\text{acc}} \Delta \text{Acc}_t - w_{\text{lat}} T_{\text{round}, t} - w_{\text{co2}} \text{CO}_{2, t} - w_{\text{fair}} \text{Gini}_t - 0.6 \max(0, \text{Acc}_{\text{floor}} - \text{Acc}_t)$$

---

## CLI Usage

```bash
# Run the 8-layer pipeline unit and integration test suite
.venv/bin/python tests_pipeline.py

# Run the end-to-end 8-layer experiment runner from CLI
.venv/bin/python -m fl_pipeline.experiment --mode dirichlet --alpha 0.5 --rounds 15 --episodes 40 --seed 42
```
