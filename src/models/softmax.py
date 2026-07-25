"""A small softmax classifier suitable for quick federated sanity checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ModelState = dict[str, np.ndarray]


class SoftmaxClassifier:
    def __init__(self, input_dim: int, num_classes: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=(input_dim, num_classes)).astype(np.float32)
        self.bias = np.zeros(num_classes, dtype=np.float32)

    def state_dict(self) -> ModelState:
        return {"weights": self.weights.copy(), "bias": self.bias.copy()}

    def load_state_dict(self, state: Mapping[str, np.ndarray]) -> None:
        self.weights = np.asarray(state["weights"], dtype=np.float32).copy()
        self.bias = np.asarray(state["bias"], dtype=np.float32).copy()

    def _probabilities(self, features: np.ndarray) -> np.ndarray:
        logits = features @ self.weights + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def train_batch(self, features: np.ndarray, labels: np.ndarray, learning_rate: float) -> float:
        probabilities = self._probabilities(features)
        count = labels.shape[0]
        loss = -np.log(probabilities[np.arange(count), labels] + 1e-12).mean()
        probabilities[np.arange(count), labels] -= 1.0
        probabilities /= count
        self.weights -= learning_rate * (features.T @ probabilities)
        self.bias -= learning_rate * probabilities.sum(axis=0)
        return float(loss)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        probabilities = self._probabilities(features)
        loss = -np.log(probabilities[np.arange(labels.size), labels] + 1e-12).mean()
        predictions = probabilities.argmax(axis=1)
        return float(loss), float((predictions == labels).mean())

