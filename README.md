# Carbon-Aware Federated Learning

This repository is implementing a multi-objective scheduling project in small,
validated stages. The current implementation covers **Step 3: the baseline
federated-learning setup** from
`first-steps-morl-federated-computing.md`.

## Implemented baseline

- Five IID MNIST clients
- Deterministic round-robin client selection
- Private local training from a copy of the global model
- Sample-count-weighted Federated Averaging (FedAvg)
- Global loss and accuracy evaluation after every round
- Reproducible JSON configuration and random seed
- CSV metrics and an SVG convergence plot

Carbon, latency, heterogeneous client profiles, non-IID partitions, additional
schedulers, and reinforcement learning remain deliberately out of scope for this
step.

## Run the baseline

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/run_fedavg_baseline.py
```

The first run downloads MNIST into `data/raw/`. Generated files are placed in
`results/fedavg_baseline/`:

- `resolved_config.json`
- `round_metrics.csv`
- `convergence.svg`

Change round count, client count, local epochs, batch size, learning rate, or
sample limits in `configs/fedavg_baseline.json`.

## Validate the implementation

```bash
python -m unittest discover -s tests -v
```

The tests verify the sample-weighted aggregation formula, deterministic client
selection, and improvement of the global model over multiple federated rounds.
