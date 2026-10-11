import numpy as np
import pytest

from src.datasets.iid_partition import iid_partition
from src.datasets.noniid_partition import class_distribution, dirichlet_partition

LABELS = np.repeat(np.arange(10), 600)


def _assert_exact_cover(shards, n):
    allidx = np.concatenate(shards)
    assert len(allidx) == n
    assert len(np.unique(allidx)) == n  # no duplicates, nothing dropped


def test_iid_covers_and_balances():
    shards = iid_partition(LABELS, 5, seed=0)
    _assert_exact_cover(shards, len(LABELS))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
    dist = class_distribution(LABELS, shards, 10)
    assert (dist > 0).all()


@pytest.mark.parametrize("alpha", [0.1, 0.5, 10.0])
def test_dirichlet_covers_and_nonempty(alpha):
    shards = dirichlet_partition(LABELS, 5, alpha, seed=0, min_samples_per_client=10)
    _assert_exact_cover(shards, len(LABELS))
    assert min(map(len, shards)) >= 10


def test_partitions_are_deterministic():
    a = dirichlet_partition(LABELS, 5, 0.5, seed=3)
    b = dirichlet_partition(LABELS, 5, 0.5, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_small_alpha_is_more_skewed():
    def skew(alpha):
        dist = class_distribution(LABELS, dirichlet_partition(LABELS, 5, alpha, seed=1), 10)
        p = dist / dist.sum(1, keepdims=True)
        return p.max(1).mean()
    assert skew(0.1) > skew(10.0)
