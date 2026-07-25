"""Multi-round server loop shared by this and future schedulers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.federated.client import FederatedClient
from src.federated.fedavg import aggregate_weighted
from src.models.softmax import SoftmaxClassifier
from src.schedulers.round_robin import RoundRobinScheduler


@dataclass(frozen=True)
class RoundMetrics:
    round: int
    selected_clients: list[int]
    selected_samples: int
    mean_local_loss: float
    global_loss: float
    global_accuracy: float


def make_iid_clients(
    features: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    seed: int,
) -> list[FederatedClient]:
    rng = np.random.default_rng(seed)
    partitions = np.array_split(rng.permutation(labels.size), num_clients)
    return [
        FederatedClient(client_id, features[indices], labels[indices])
        for client_id, indices in enumerate(partitions)
    ]


def run_fedavg(
    clients: list[FederatedClient],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    *,
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[SoftmaxClassifier, list[dict[str, object]]]:
    if not clients:
        raise ValueError("At least one federated client is required")
    num_classes = int(max(test_labels.max(), max(client.labels.max() for client in clients))) + 1
    model = SoftmaxClassifier(test_features.shape[1], num_classes, seed=seed)
    scheduler = RoundRobinScheduler(len(clients), clients_per_round)
    history: list[dict[str, object]] = []

    for round_index in range(1, rounds + 1):
        selected_ids = scheduler.select()
        global_state = model.state_dict()
        updates = [
            clients[client_id].train(
                global_state,
                num_classes=num_classes,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed + round_index * 1000 + client_id,
            )
            for client_id in selected_ids
        ]
        model.load_state_dict(aggregate_weighted(updates))
        global_loss, global_accuracy = model.evaluate(test_features, test_labels)
        metrics = RoundMetrics(
            round=round_index,
            selected_clients=selected_ids,
            selected_samples=sum(update.num_samples for update in updates),
            mean_local_loss=float(np.mean([update.mean_train_loss for update in updates])),
            global_loss=global_loss,
            global_accuracy=global_accuracy,
        )
        history.append(asdict(metrics))
        print(
            f"Round {round_index:02d} | clients={selected_ids} | "
            f"loss={global_loss:.4f} | accuracy={global_accuracy:.2%}"
        )
    return model, history

