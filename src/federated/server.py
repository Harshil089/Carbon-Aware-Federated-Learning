import time

import numpy as np
from torch import nn

from src.datasets.loader import TensorDataset
from src.federated.client import FederatedClient
from src.federated.fedavg import fedavg
from src.federated.training import evaluate


class FederatedServer:
    """Synchronous FedAvg server. Client selection is uniform random for now;
    carbon-aware and RL schedulers will replace ``select_clients`` later."""

    def __init__(self, model: nn.Module, clients: list[FederatedClient], test_data: TensorDataset, seed: int):
        self.model = model
        self.clients = clients
        self.test_data = test_data
        self.rng = np.random.default_rng(seed)

    def select_clients(self, k: int) -> list[FederatedClient]:
        eligible = [c for c in self.clients if c.profile.availability]
        if k >= len(eligible):
            return eligible
        picks = self.rng.choice(len(eligible), size=k, replace=False)
        return [eligible[i] for i in sorted(picks)]

    def run_round(self, cfg: dict) -> dict:
        selected = self.select_clients(cfg["clients_per_round"])
        # Every selected client starts from the same global weights.
        results = {c.client_id: c.fit(self.model, cfg) for c in selected}
        agg_start = time.perf_counter()
        new_state = fedavg(
            [r["state"] for r in results.values()],
            [r["num_samples"] for r in results.values()],
        )
        self.model.load_state_dict(new_state)
        aggregation_time = time.perf_counter() - agg_start
        accuracy, loss = evaluate(self.model, self.test_data)
        return {
            "selected": [c.client_id for c in selected],
            "client_results": results,
            "global_accuracy": accuracy,
            "global_loss": loss,
            "aggregation_time_s": aggregation_time,
        }
