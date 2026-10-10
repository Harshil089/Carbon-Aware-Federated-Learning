# Carbon-Aware-Federated-Learning

Research prototype for multi-objective, carbon-aware client scheduling in federated learning.
See [`first-steps-morl-federated-computing.md`](first-steps-morl-federated-computing.md) for the plan.

## Status

| Component | State |
|---|---|
| MNIST / Fashion-MNIST loading | Done |
| IID and Dirichlet non-IID partitions (saved per run, class-distribution heatmap) | Done |
| Small CNN and MLP models | Done |
| Five simulated clients with configurable profiles (`configs/clients.yaml`) | Done; profile values are placeholders and are only logged for now |
| FedAvg (sample-size weighted), synchronous rounds | Done |
| Round- and client-level CSV logging, config snapshot, checkpoints | Done |
| Energy and carbon tracking of the running machine (GPU via NVML, CPU via RAPL or a TDP estimate) | Done; carbon intensity is an assumed constant |
| Per-client latency and energy/carbon models | Not started |
| Baseline schedulers (random is the only selection rule so far) | Not started |
| RL / MORL scheduler | Not started |

## Quick start

### Google Colab

Open [`notebooks/fedavg_baseline_colab.ipynb`](notebooks/fedavg_baseline_colab.ipynb) in Colab
(File → Open notebook → GitHub, paste this repo's URL) and run all cells. It clones the repo,
installs requirements, runs the tests and an IID and a non-IID baseline, then plots the results.

### Local

```bash
pip install -r requirements.txt
python -m pytest -q tests
python experiments/baseline/run_baseline.py                                  # IID, 20 rounds
python experiments/baseline/run_baseline.py --partition dirichlet --alpha 0.1
python experiments/baseline/run_baseline.py --clients-per-round 3 --rounds 30
```

Defaults are in `configs/experiment.yaml`; command-line flags override them.

### Energy and carbon

Enabled by default (`--no-energy` turns it off). A background thread samples power every second and integrates it:
GPU from NVML (measured), CPU from Intel RAPL if the VM exposes it (measured), otherwise `cpu_tdp_watts * (idle + (1 - idle) * utilisation)`
(an estimate; Colab normally hides RAPL, so expect this path). Carbon is `energy_kWh * carbon_intensity`; the default 475 gCO2e/kWh is an
assumed global average, so pass `--carbon-intensity <gCO2e/kWh>` for your region. Scope: CPU and GPU of the whole machine only, with no DRAM,
storage, network, cooling or embodied carbon. Treat the numbers as estimates until validated against a wall-power meter.

## Outputs

Each run writes `results/<experiment_id>/`:

- `rounds.csv`: per round, selected clients, global test accuracy and loss, accuracy improvement, mean local loss, wall time.
- `clients.csv`: per round and client, selection, dataset size, local loss, training wall time, participation count and profile fields.
- `config.json`, `client_profiles.json`, `summary.json`: configuration snapshot and final metrics.
- `class_distribution.csv` / `.png`, `partitions.npz`: exact client data split.
- `energy.json`: total energy (Wh), CPU/GPU split, average power, gCO2e, and which measurement source was used. `rounds.csv` also carries per-round and cumulative energy and carbon.
- `accuracy_curve.png`, `final_model.pt`, `best_model.pt`.

## Layout

```text
configs/            experiment.yaml, clients.yaml
src/datasets/       loader.py, iid_partition.py, noniid_partition.py
src/models/         cnn.py, mlp.py
src/federated/      client.py, server.py, fedavg.py, training.py
src/simulation/     client_profiles.py, energy_tracker.py, carbon.py
src/utils/          config.py, logger.py, seeds.py
experiments/baseline/run_baseline.py
notebooks/fedavg_baseline_colab.ipynb
tests/              test_fedavg.py, test_partitions.py, test_energy.py
```
