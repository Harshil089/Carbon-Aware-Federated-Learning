from __future__ import annotations

import unittest

import numpy as np

from src.models.mlp import MLPClassifier
from src.training.duration import train_for_duration


class AdvancingClock:
    def __init__(self, seconds_per_call: float) -> None:
        self.now = -seconds_per_call
        self.seconds_per_call = seconds_per_call

    def __call__(self) -> float:
        self.now += self.seconds_per_call
        return self.now


class DurationTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        class_zero = rng.normal(-1.0, 0.25, size=(100, 2)).astype(np.float32)
        class_one = rng.normal(1.0, 0.25, size=(100, 2)).astype(np.float32)
        self.features = np.vstack([class_zero, class_one])
        self.labels = np.array([0] * 100 + [1] * 100, dtype=np.int64)

    def test_training_stops_at_duration_and_reports_progress(self) -> None:
        reports = []
        result = train_for_duration(
            MLPClassifier(2, 16, 2, seed=7),
            self.features,
            self.labels,
            self.features,
            self.labels,
            duration_seconds=5.0,
            batch_size=20,
            learning_rate=0.1,
            report_interval_seconds=2.0,
            seed=7,
            on_report=reports.append,
            clock=AdvancingClock(1.0),
        )
        self.assertGreaterEqual(result.elapsed_seconds, 5.0)
        self.assertEqual(result.steps, 5)
        self.assertEqual(len(reports), 3)

    def test_training_learns_separable_data(self) -> None:
        result = train_for_duration(
            MLPClassifier(2, 16, 2, seed=7),
            self.features,
            self.labels,
            self.features,
            self.labels,
            duration_seconds=20.0,
            batch_size=20,
            learning_rate=0.1,
            report_interval_seconds=20.0,
            seed=7,
            clock=AdvancingClock(1.0),
        )
        self.assertGreater(result.test_accuracy, 0.95)


if __name__ == "__main__":
    unittest.main()
