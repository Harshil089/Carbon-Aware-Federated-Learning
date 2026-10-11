import numpy as np


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int = 10,
    max_retries: int = 100,
) -> list[np.ndarray]:
    """Split indices across clients with per-class proportions p_k ~ Dir(alpha * 1_N).

    Smaller alpha gives more skewed class distributions. Resamples until every client
    holds at least ``min_samples_per_client`` examples so no partition is empty.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_retries):
        shards: list[list[int]] = [[] for _ in range(num_clients)]
        for k in classes:
            idx_k = rng.permutation(np.flatnonzero(labels == k))
            p_k = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(p_k)[:-1] * len(idx_k)).astype(int)
            for client, part in enumerate(np.split(idx_k, cuts)):
                shards[client].extend(part.tolist())
        if min(len(s) for s in shards) >= min_samples_per_client:
            return [np.sort(np.array(s, dtype=np.int64)) for s in shards]
    raise RuntimeError(
        f"Could not give every client >= {min_samples_per_client} samples with alpha={alpha} "
        f"after {max_retries} tries; raise alpha or lower min_samples_per_client."
    )


def class_distribution(labels: np.ndarray, shards: list[np.ndarray], num_classes: int) -> np.ndarray:
    """(num_clients, num_classes) matrix of per-client class counts."""
    return np.stack([np.bincount(labels[s], minlength=num_classes) for s in shards])
