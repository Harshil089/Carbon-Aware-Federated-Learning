from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from src.federated.client import ClientUpdate
from src.federated.fedavg import aggregate_weighted
from src.federated.server import make_iid_clients, run_fedavg
from src.schedulers.round_robin import RoundRobinScheduler


class FedAvgTests(unittest.TestCase):
    def test_aggregation_is_weighted_by_sample_count(self) -> None:
        updates = [
            ClientUpdate(0, 1, {"weights": np.array([2.0]), "bias": np.array([4.0])}, 0.0),
            ClientUpdate(1, 3, {"weights": np.array([6.0]), "bias": np.array([8.0])}, 0.0),
        ]
        result = aggregate_weighted(updates)
        np.testing.assert_allclose(result["weights"], np.array([5.0]))
        np.testing.assert_allclose(result["bias"], np.array([7.0]))

    def test_round_robin_selects_only_requested_clients(self) -> None:
        scheduler = RoundRobinScheduler(num_clients=5, clients_per_round=3)
        self.assertEqual(scheduler.select(), [0, 1, 2])
        self.assertEqual(scheduler.select(), [3, 4, 0])

    def test_unselected_clients_do_not_train(self) -> None:
        features = np.array(
            [[-1.0, -1.0], [-0.8, -1.2], [1.0, 1.0], [0.8, 1.2]] * 5,
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1] * 5, dtype=np.int64)
        clients = make_iid_clients(features, labels, num_clients=5, seed=3)
        with (
            patch.object(clients[0], "train", wraps=clients[0].train) as train_zero,
            patch.object(clients[1], "train", wraps=clients[1].train) as train_one,
            patch.object(clients[2], "train", wraps=clients[2].train) as train_two,
            patch.object(clients[3], "train", wraps=clients[3].train) as train_three,
            patch.object(clients[4], "train", wraps=clients[4].train) as train_four,
        ):
            run_fedavg(
                clients,
                features,
                labels,
                rounds=1,
                clients_per_round=2,
                local_epochs=1,
                batch_size=4,
                learning_rate=0.1,
                seed=3,
            )
        self.assertEqual([mock.call_count for mock in (train_zero, train_one)], [1, 1])
        self.assertEqual([mock.call_count for mock in (train_two, train_three, train_four)], [0, 0, 0])

    def test_multi_round_training_improves_accuracy(self) -> None:
        rng = np.random.default_rng(7)
        class_zero = rng.normal(-1.0, 0.25, size=(100, 2)).astype(np.float32)
        class_one = rng.normal(1.0, 0.25, size=(100, 2)).astype(np.float32)
        features = np.vstack([class_zero, class_one])
        labels = np.array([0] * 100 + [1] * 100, dtype=np.int64)
        clients = make_iid_clients(features, labels, num_clients=5, seed=7)
        _, history = run_fedavg(
            clients,
            features,
            labels,
            rounds=3,
            clients_per_round=3,
            local_epochs=1,
            batch_size=16,
            learning_rate=0.1,
            seed=7,
        )
        self.assertGreater(history[-1]["global_accuracy"], 0.95)
        self.assertEqual(len(history), 3)


if __name__ == "__main__":
    unittest.main()
