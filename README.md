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
| Latency, energy and carbon models | Not started |
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

## Outputs

Each run writes `results/<experiment_id>/`:

- `rounds.csv`: per round, selected clients, global test accuracy and loss, accuracy improvement, mean local loss, wall time.
- `clients.csv`: per round and client, selection, dataset size, local loss, training wall time, participation count and profile fields.
- `config.json`, `client_profiles.json`, `summary.json`: configuration snapshot and final metrics.
- `class_distribution.csv` / `.png`, `partitions.npz`: exact client data split.
- `accuracy_curve.png`, `final_model.pt`, `best_model.pt`.

## Layout

```text
configs/            experiment.yaml, clients.yaml
src/datasets/       loader.py, iid_partition.py, noniid_partition.py
src/models/         cnn.py, mlp.py
src/federated/      client.py, server.py, fedavg.py, training.py
src/simulation/     client_profiles.py
src/utils/          config.py, logger.py, seeds.py
experiments/baseline/run_baseline.py
notebooks/fedavg_baseline_colab.ipynb
tests/              test_fedavg.py, test_partitions.py
```
