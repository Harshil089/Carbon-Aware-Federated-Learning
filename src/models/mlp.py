"""A compact NumPy MLP that provides a sustained, portable CPU workload."""

from __future__ import annotations

import numpy as np


class MLPClassifier:
    """Two-layer MNIST classifier trained with mini-batch SGD.

    The model is intentionally small in memory. Repeating mini-batches until a
    wall-clock deadline makes experiment duration independent of model size.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.weights_one = (
            rng.standard_normal((input_dim, hidden_dim), dtype=np.float32)
            * np.float32(np.sqrt(2.0 / input_dim))
        )
        self.bias_one = np.zeros(hidden_dim, dtype=np.float32)
        self.weights_two = (
            rng.standard_normal((hidden_dim, num_classes), dtype=np.float32)
            * np.float32(np.sqrt(2.0 / hidden_dim))
        )
        self.bias_two = np.zeros(num_classes, dtype=np.float32)

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = features @ self.weights_one + self.bias_one
        np.maximum(hidden, 0.0, out=hidden)
        logits = hidden @ self.weights_two + self.bias_two
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return hidden, probabilities

    def train_batch(self, features: np.ndarray, labels: np.ndarray, learning_rate: float) -> float:
        hidden, probabilities = self._forward(features)
        count = labels.size
        loss = -np.log(probabilities[np.arange(count), labels] + 1e-12).mean()

        probabilities[np.arange(count), labels] -= 1.0
        probabilities /= count
        gradient_two = hidden.T @ probabilities
        gradient_bias_two = probabilities.sum(axis=0)
        gradient_hidden = probabilities @ self.weights_two.T
        gradient_hidden[hidden <= 0.0] = 0.0

        self.weights_two -= learning_rate * gradient_two
        self.bias_two -= learning_rate * gradient_bias_two
        self.weights_one -= learning_rate * (features.T @ gradient_hidden)
        self.bias_one -= learning_rate * gradient_hidden.sum(axis=0)
        return float(loss)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        _, probabilities = self._forward(features)
        loss = -np.log(probabilities[np.arange(labels.size), labels] + 1e-12).mean()
        predictions = probabilities.argmax(axis=1)
        return float(loss), float((predictions == labels).mean())
