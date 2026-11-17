"""Layer 4: Sample-Weighted Federated Averaging (FedAvg) Aggregation."""
from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np

from .clients import FederatedClient, LocalTrainResult


def compute_aggregation_weights(sample_counts: Sequence[int]) -> np.ndarray:
    """Compute normalized FedAvg aggregation weights w_i = n_i / sum_j(n_j) summing to 1.0."""
    counts = np.asarray(sample_counts, dtype=float)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("sample_counts must be a non-empty 1D sequence.")
    if np.any(counts <= 0):
        raise ValueError("All selected client sample counts must be strictly positive.")
    total = float(np.sum(counts))
    weights = counts / total
    return weights


def fedavg_aggregate(
    local_weights: Sequence[np.ndarray],
    sample_counts: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate client weight matrices using sample-weighted averaging.

    Returns:
        (aggregated_weights, normalized_aggregation_weights)
    """
    if len(local_weights) == 0:
        raise ValueError("local_weights cannot be empty.")
    if len(local_weights) != len(sample_counts):
        raise ValueError("local_weights and sample_counts must have the same length.")

    norm_weights = compute_aggregation_weights(sample_counts)
    stacked = np.stack([ np.asarray(w, dtype=float) for w in local_weights ], axis=0)
    aggregated = np.tensordot(norm_weights, stacked, axes=(0, 0))
    return aggregated, norm_weights


def run_fedavg_round(
    global_weights: np.ndarray,
    clients: Sequence[FederatedClient],
    selected_ids: Sequence[int],
    epochs: int = 2,
    lr: float = 0.23,
) -> Tuple[np.ndarray, np.ndarray, List[LocalTrainResult]]:
    """Train selected clients locally and aggregate their updates via sample-weighted FedAvg."""
    if len(selected_ids) == 0:
        raise ValueError("At least one client must be selected for FedAvg.")
    results: List[LocalTrainResult] = []
    for idx in selected_ids:
        res = clients[int(idx)].train_local(global_weights, epochs=epochs, lr=lr)
        results.append(res)

    agg_w, norm_w = fedavg_aggregate(
        [r.weights for r in results],
        [r.num_samples for r in results],
    )
    return agg_w, norm_w, results
