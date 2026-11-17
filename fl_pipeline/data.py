"""Layer 1: Reproducible IID and Dirichlet non-IID Federated Data Partitioning."""
from __future__ import annotations

from typing import List, Tuple
import numpy as np

Partition = Tuple[np.ndarray, np.ndarray]

DEFAULT_CLASS_CENTRES = np.array([[-1.8, -1.3], [1.7, -0.9], [0.1, 1.8]], dtype=float)


def generate_classification_data(
    n_samples: int,
    n_classes: int = 3,
    feature_noise_sd: float = 1.3,
    include_bias: bool = True,
    seed: int = 42,
) -> Partition:
    """Generate synthetic multi-class Gaussian cluster data with optional bias column."""
    if n_samples < n_classes:
        raise ValueError("n_samples must be at least n_classes.")
    rng = np.random.default_rng(seed)
    if n_classes == 3:
        centres = DEFAULT_CLASS_CENTRES
    else:
        angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
        centres = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])

    base_counts = np.full(n_classes, n_samples // n_classes, dtype=int)
    base_counts[: n_samples % n_classes] += 1
    y = np.repeat(np.arange(n_classes, dtype=int), base_counts)
    rng.shuffle(y)
    x = centres[y] + rng.normal(0.0, feature_noise_sd, size=(n_samples, centres.shape[1]))
    if include_bias:
        x = np.column_stack([x, np.ones(n_samples, dtype=float)])
    return x, y


def partition_iid(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int = 4,
    seed: int = 42,
) -> List[Partition]:
    """Partition dataset uniformly at random (IID) across clients with no empty partitions."""
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1.")
    if len(y) < n_clients:
        raise ValueError("Dataset size must be >= n_clients to prevent empty partitions.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    splits = np.array_split(indices, n_clients)
    return [(x[idx], y[idx]) for idx in splits]


def partition_dirichlet(
    x: np.ndarray,
    y: np.ndarray,
    n_clients: int = 4,
    alpha: float = 0.5,
    min_samples_per_client: int = 10,
    seed: int = 42,
) -> List[Partition]:
    """Partition dataset across clients using a Dirichlet(alpha) label distribution.

    Guarantees:
    - Fixed seed reproducibility.
    - No empty partitions (each client receives >= min_samples_per_client samples).
    - Exact sample conservation (disjoint union equals input dataset).
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >= 1.")
    if alpha <= 0:
        raise ValueError("Dirichlet concentration parameter alpha must be > 0.")
    min_req = max(1, min(min_samples_per_client, len(y) // n_clients))
    if len(y) < n_clients * min_req:
        raise ValueError("Not enough samples to satisfy min_samples_per_client.")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    class_indices = [rng.permutation(np.flatnonzero(y == c)) for c in classes]

    for _ in range(200):
        client_lists: List[List[int]] = [[] for _ in range(n_clients)]
        for c_idx in class_indices:
            proportions = rng.dirichlet(np.repeat(alpha, n_clients))
            # Balance slightly away from over-full clients while keeping Dirichlet skew
            counts = np.floor(proportions * len(c_idx)).astype(int)
            remainder = len(c_idx) - int(counts.sum())
            if remainder > 0:
                top_clients = np.argsort(proportions * len(c_idx) - counts)[::-1][:remainder]
                counts[top_clients] += 1
            pos = 0
            for i in range(n_clients):
                cnt = int(counts[i])
                if cnt > 0:
                    client_lists[i].extend(c_idx[pos : pos + cnt].tolist())
                    pos += cnt
        if all(len(lst) >= min_req for lst in client_lists):
            return [
                (x[np.array(lst, dtype=int)], y[np.array(lst, dtype=int)])
                for lst in client_lists
            ]

    # Deterministic fallback rebalance if extreme alpha left a client below min_req
    all_lists = [list(lst) for lst in client_lists]
    while any(len(lst) < min_req for lst in all_lists):
        smallest = int(np.argmin([len(lst) for lst in all_lists]))
        largest = int(np.argmax([len(lst) for lst in all_lists]))
        if len(all_lists[largest]) <= min_req:
            break
        all_lists[smallest].append(all_lists[largest].pop())

    return [
        (x[np.array(lst, dtype=int)], y[np.array(lst, dtype=int)])
        for lst in all_lists
    ]


def make_federated_dataset(
    n_clients: int = 4,
    mode: str = "dirichlet",
    alpha: float = 0.5,
    n_train: int = 1000,
    n_test: int = 1500,
    n_classes: int = 3,
    min_samples_per_client: int = 10,
    seed: int = 42,
) -> Tuple[List[Partition], Partition]:
    """Create federated training partitions and an independent held-out test set."""
    x_train, y_train = generate_classification_data(
        n_samples=n_train, n_classes=n_classes, seed=seed
    )
    x_test, y_test = generate_classification_data(
        n_samples=n_test, n_classes=n_classes, seed=seed + 10_000
    )
    mode_norm = mode.strip().lower()
    if mode_norm == "iid":
        parts = partition_iid(x_train, y_train, n_clients=n_clients, seed=seed + 1)
    elif mode_norm in ("dirichlet", "non-iid", "non_iid"):
        parts = partition_dirichlet(
            x_train,
            y_train,
            n_clients=n_clients,
            alpha=alpha,
            min_samples_per_client=min_samples_per_client,
            seed=seed + 1,
        )
    else:
        raise ValueError(f"Unsupported partition mode: {mode}")
    return parts, (x_test, y_test)


def class_distribution(partitions: List[Partition], n_classes: int = 3) -> np.ndarray:
    """Return a (n_clients, n_classes) matrix of class counts per client."""
    dist = np.zeros((len(partitions), n_classes), dtype=int)
    for i, (_, y) in enumerate(partitions):
        for c in range(n_classes):
            dist[i, c] = int(np.sum(y == c))
    return dist
