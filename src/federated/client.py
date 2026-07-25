"""Local training performed by one simulated federated client."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.softmax import ModelState, SoftmaxClassifier


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    num_samples: int
    state: ModelState
    mean_train_loss: float


class FederatedClient:
    def __init__(self, client_id: int, features: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != labels.shape[0] or labels.size == 0:
            raise ValueError("A client needs the same non-zero number of features and labels")
        self.client_id = client_id
        self.features = features
        self.labels = labels

    @property
    def num_samples(self) -> int:
        return int(self.labels.size)

    def train(
        self,
        global_state: ModelState,
        *,
        num_classes: int,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> ClientUpdate:
        """Train a private model copy; never mutate the supplied global state."""
        model = SoftmaxClassifier(self.features.shape[1], num_classes, seed=seed)
        model.load_state_dict(global_state)
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        for _ in range(local_epochs):
            indices = rng.permutation(self.num_samples)
            for start in range(0, self.num_samples, batch_size):
                batch = indices[start : start + batch_size]
                losses.append(model.train_batch(self.features[batch], self.labels[batch], learning_rate))
        return ClientUpdate(
            client_id=self.client_id,
            num_samples=self.num_samples,
            state=model.state_dict(),
            mean_train_loss=float(np.mean(losses)),
        )

