"""Run model training until a wall-clock duration has elapsed."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.models.mlp import MLPClassifier


@dataclass(frozen=True)
class TrainingProgress:
    elapsed_seconds: float
    steps: int
    samples_processed: int
    completed_epochs: float
    samples_per_second: float
    mean_training_loss: float
    test_loss: float
    test_accuracy: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "steps": self.steps,
            "samples_processed": self.samples_processed,
            "completed_epochs": round(self.completed_epochs, 3),
            "samples_per_second": round(self.samples_per_second, 3),
            "mean_training_loss": self.mean_training_loss,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
        }


def train_for_duration(
    model: MLPClassifier,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    duration_seconds: float,
    batch_size: int,
    learning_rate: float,
    report_interval_seconds: float,
    seed: int,
    on_report: Callable[[TrainingProgress], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> TrainingProgress:
    """Train continuously and return only after the requested duration.

    The deadline is checked after every mini-batch. Consequently, the run can
    exceed the requested duration by at most roughly one training step.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if report_interval_seconds <= 0:
        raise ValueError("report_interval_seconds must be positive")
    if train_y.size == 0 or test_y.size == 0:
        raise ValueError("training and test data must not be empty")

    rng = np.random.default_rng(seed)
    order = rng.permutation(train_y.size)
    position = 0
    steps = 0
    samples_processed = 0
    accumulated_loss = 0.0
    interval_steps = 0
    start = clock()
    next_report = report_interval_seconds
    last_progress: TrainingProgress | None = None

    while True:
        if position + batch_size > train_y.size:
            order = rng.permutation(train_y.size)
            position = 0
        indices = order[position : position + batch_size]
        position += indices.size
        loss = model.train_batch(train_x[indices], train_y[indices], learning_rate)
        steps += 1
        interval_steps += 1
        samples_processed += indices.size
        accumulated_loss += loss

        elapsed = clock() - start
        finished = elapsed >= duration_seconds
        if elapsed >= next_report or finished:
            test_loss, test_accuracy = model.evaluate(test_x, test_y)
            last_progress = TrainingProgress(
                elapsed_seconds=elapsed,
                steps=steps,
                samples_processed=samples_processed,
                completed_epochs=samples_processed / train_y.size,
                samples_per_second=samples_processed / max(elapsed, 1e-12),
                mean_training_loss=accumulated_loss / interval_steps,
                test_loss=test_loss,
                test_accuracy=test_accuracy,
            )
            if on_report is not None:
                on_report(last_progress)
            accumulated_loss = 0.0
            interval_steps = 0
            while next_report <= elapsed:
                next_report += report_interval_seconds
        if finished:
            assert last_progress is not None
            return last_progress
