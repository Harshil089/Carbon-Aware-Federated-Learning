"""Sample-count-weighted Federated Averaging."""

from __future__ import annotations

import numpy as np

from src.federated.client import ClientUpdate
from src.models.softmax import ModelState


def aggregate_weighted(updates: list[ClientUpdate]) -> ModelState:
    if not updates:
        raise ValueError("FedAvg requires at least one client update")
    total_samples = sum(update.num_samples for update in updates)
    if total_samples <= 0:
        raise ValueError("FedAvg requires a positive total sample count")

    keys = set(updates[0].state)
    if any(set(update.state) != keys for update in updates):
        raise ValueError("All client states must contain the same parameters")

    aggregated: ModelState = {}
    for key in keys:
        reference_shape = updates[0].state[key].shape
        if any(update.state[key].shape != reference_shape for update in updates):
            raise ValueError(f"Client parameter shapes differ for {key}")
        value = np.zeros(reference_shape, dtype=np.float64)
        for update in updates:
            value += update.state[key] * (update.num_samples / total_samples)
        aggregated[key] = value.astype(np.float32)
    return aggregated

