import numpy as np


def iid_partition(labels: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    """Shuffle all indices and split them into near-equal client shards."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    return [np.sort(shard) for shard in np.array_split(perm, num_clients)]
