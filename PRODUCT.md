# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Faculty evaluators, peer reviewers, ML systems researchers, and distributed computing engineers inspecting, benchmarking, and stress-testing Multi-Objective Reinforcement Learning (MORL) client scheduling policies in Federated Learning. They use this both during live academic/technical defenses and for self-directed offline artifact inspection (`dashboard.html`).

## Product Purpose

Federated Computing Lab is an interactive research workbench and live Python simulation console for optimizing the Latency–Carbon–Accuracy–Fairness trade-off in heterogeneous Federated Learning (`FedAvg`). Success means an evaluator or researcher can immediately understand the physical vs. modeled provenance boundary, run or replay live 4-policy and 8-layer MORL experiments, inspect per-round client scheduling behavior, and verify every mathematical formula, sensitivity distribution, and seed.

## Positioning

Unlike black-box FL simulators or static paper plots, Federated Computing Lab anchors its multi-node simulator to a real physical wall-meter measurement on `PC1` (`0.065 kWh` over `190 min` = `20.53 W` mean power on Intel Core i7-12700) while explicitly separating calibrated synthetic telemetry (`10,000` rows across `PC1`–`PC4`), synchronous parallel straggler + idle power accounting, and 8-layer MORL Pareto frontier extraction.

## Operating Context

- **Live Local Console (`demo_server.py`):** Runs on `127.0.0.1:8765`, backed by `run_project.py` and `fl_pipeline/`, streaming live progress and stdout logs to the browser.
- **Standalone Offline Artifact (`dashboard.html`):** Compiled via `vite-plugin-singlefile` into `web/index.html` and hydrated with embedded JSON results via `report.py` so completed runs (`results/dashboard.html`, `examples/quick_demo/dashboard.html`, and exported ZIP archives) remain fully interactive without a running Python server.
- **Five Core Surfaces:**
  1. `/` (`landing`): Project brief, interactive Naive vs. Carbon-Aware RL node comparison, 6 Research Questions (`RQ1`–`RQ6`), and walkthrough entry points.
  2. `/overview` (`overview`): Step 01 — Problem & Policy Overview: Command Deck for running simulations, KPI summary, interactive Pareto & Trajectory SVG charts, Matplotlib PNG figure viewer, and 4-policy comparison matrix.
  3. `/pipeline` (`pipeline`): Step 02 — Software-Driven 8-Layer Pipeline (`fl_pipeline/`): Interactive 8-layer inspector (`L1` Data & Partitions through `L8` Pareto & Fairness), live Dirichlet($\alpha$) / IID runner, 7-policy evaluation table with Gini fairness, and MORL preference scalarization sweep chart.
  4. `/replay` (`replay`): Step 03 — Federated Topology & Round Replay: Hub-and-spoke synchronous `FedAvg` topology visualizer (`PC1`–`PC4` + `30 W` Coordinator) with round-by-round playback, active/idle power states, and per-round telemetry.
  5. `/specs` (`specs`): Step 04 — Calibration, Parameters & Run Log: Full hyperparameter record, `PC1` physical meter anchor vs. `PC2`–`PC4` scenario ratios, 5th/50th/95th percentile scenario sensitivity, exact latency/energy/carbon/reward equations, and live Python stdout log.

## Capabilities and Constraints

- **Single-File Build Constraint:** Frontend in `frontend/` compiles via `npm run build` (`tsc -b && vite build` with `vite-plugin-singlefile`) into `web/index.html`, containing `<script id="bootstrap" type="application/json">__BOOTSTRAP__</script>`.
- **Automated Test Guardrails (`tests.py`, `tests_demo.py`, `tests_pipeline.py`):**
  - Root page must contain `Federated Computing Lab`.
  - Generated `dashboard.html` must not contain unreplaced `__BOOTSTRAP__`, nor banned legacy strings `19 NOVEMBER` or `Evidence boundary`.
  - All API routes (`/api/state`, `/api/results`, `/api/run`, `/api/cancel`, `/api/pipeline/*`, `/api/export`, `/artifacts/*`) and parameter bounds must remain intact.
- **Data Honesty & Provenance:** Never claim `PC2`–`PC4` or simulated round CO₂ are physical meter measurements; `PC1` (`0.065 kWh` over `190 min`) is the sole physical meter anchor, and `PC2`–`PC4` are modeled scenario ratios.

## Evidence on Hand

- Physical measurement: `PC1` (`8 GB RAM`, `Intel Core i7-12700`), `0.065 kWh` over `190 minutes` (`20.526 W` mean power), CEA grid factor `0.675 kg CO₂/kWh` (`config.json`, `results/actual_meter_reading_PC1.csv`).
- Calibrated dataset: `data/calibrated_telemetry_10k.csv` (`10,000` rows calibrated from `PC1` meter anchor).
- Baseline simulation artifacts in `results/` and `examples/quick_demo/`.
- Full 8-layer Python package in `fl_pipeline/` with unit and integration tests (`26/26` checks across `tests.py`, `tests_demo.py`, `tests_pipeline.py`).

## Product Principles

1. **Scientific Honesty as Visual Architecture:** Measured truth (`PC1` physical meter reading, real NumPy/PyTorch model training) and modeled scenarios (`PC2`–`PC4` hardware ratios, grid carbon intensity, sensitivity bounds) must be visually distinct at a glance.
2. **Simultaneous Mechanism & Proof:** Every view should expose the underlying physical/mathematical mechanism (parallel straggler completion + idle power draw + coordinator overhead) right next to the empirical result.
3. **Zero-Friction Dual Mode:** Every surface must work seamlessly both as a live experiment controller connected to `demo_server.py` and as a self-contained offline research artifact (`dashboard.html`).
4. **High-Density Precision:** Quantitative telemetry, tabular numbers, units (`s`, `Wh`, `g CO₂`, `kg CO₂/kWh`, `Gini`), and Pareto frontiers must read with laboratory-grade legibility.
