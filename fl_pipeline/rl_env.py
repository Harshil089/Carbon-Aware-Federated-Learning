"""Layer 6: Multi-Objective Reinforcement Learning Environment (FederatedRLEnv)."""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .clients import FederatedClient, cross_entropy_loss, evaluate_accuracy
from .data import Partition
from .fedavg import run_fedavg_round
from .network_energy import simulate_round_telemetry

MODEL_CHOICES = ("numpy", "torch_mlp", "torch_cnn")


def compute_gini(counts: Sequence[float]) -> float:
    """Compute the Gini coefficient of client participation counts in [0, 1)."""
    arr = np.asarray(counts, dtype=float)
    if len(arr) <= 1 or np.all(arr == 0):
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Participation counts cannot be negative.")
    n = len(arr)
    diffs = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diffs / (2.0 * n * np.sum(arr)))


class FederatedRLEnv:
    """Multi-Objective Federated Learning Environment with constraint tracking.

    val_data: split used for RL state and reward; None falls back to test_data (legacy).
        Reported "accuracy"/"loss" in step() info always come from test_data.
    measure_train_time: when True, each trained client's measured local train wall-clock
        time scales its simulated train time (ratio to the first measured time, clipped to
        [0.5, 2.0]). Default False keeps runs deterministic under a seed.
    """

    def __init__(
        self,
        clients: Sequence[FederatedClient],
        test_data: Partition,
        max_rounds: int = 20,
        local_epochs: int = 2,
        lr: float = 0.23,
        allowed_subset_sizes: Sequence[int] = (2, 3, 4),
        base_train_seconds: float = 16.0,
        aggregation_seconds: float = 0.45,
        coordinator_power_w: float = 30.0,
        speed_jitter_fraction: float = 0.10,
        accuracy_weight: float = 15.0,
        latency_weight: float = 0.018,
        carbon_weight: float = 0.32,
        fairness_weight: float = 0.05,
        accuracy_floor: float = 0.78,
        latency_budget_s: Optional[float] = None,
        carbon_budget_g: Optional[float] = None,
        seed: int = 42,
        measure_train_time: bool = False,
        val_data: Optional[Partition] = None,
        model: str = "numpy",
        torch_partitions: Optional[Sequence[Any]] = None,
    ) -> None:
        if model not in MODEL_CHOICES:
            raise ValueError(f"model must be one of {MODEL_CHOICES}, got {model!r}.")
        self.model = model
        self.clients = list(clients)
        self.n_clients = len(self.clients)
        self.test_data = test_data
        # RL state/reward use validation; None falls back to test data (legacy behaviour).
        self.val_data = val_data if val_data is not None else test_data
        self.max_rounds = int(max_rounds)
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)
        self.base_train_seconds = float(base_train_seconds)
        self.aggregation_seconds = float(aggregation_seconds)
        self.coordinator_power_w = float(coordinator_power_w)
        self.speed_jitter_fraction = float(speed_jitter_fraction)

        self.accuracy_weight = float(accuracy_weight)
        self.latency_weight = float(latency_weight)
        self.carbon_weight = float(carbon_weight)
        self.fairness_weight = float(fairness_weight)
        self.accuracy_floor = float(accuracy_floor)
        self.latency_budget_s = latency_budget_s
        self.carbon_budget_g = carbon_budget_g

        self.actions: List[Tuple[int, ...]] = [
            tuple(comb)
            for k in allowed_subset_sizes
            if 1 <= k <= self.n_clients
            for comb in itertools.combinations(range(self.n_clients), k)
        ]
        if not self.actions:
            raise ValueError("Action space is empty; check allowed_subset_sizes.")

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._torch: Optional[Any] = None
        if model == "numpy":
            self.n_features = int(test_data[0].shape[1])
            self.n_classes = int(np.max(test_data[1]) + 1)
            self.weights = np.zeros((self.n_features, self.n_classes), dtype=float)
        else:
            if torch_partitions is None:
                raise ValueError(f"model={model!r} requires torch_partitions (tensor (x, y) pairs).")
            try:
                from .torch_backend import TorchBackend  # lazy: keeps torch optional
            except ImportError as exc:
                raise RuntimeError(
                    f"PyTorch is required for model={model}; "
                    "pip install -r requirements-optional.txt"
                ) from exc
            # Torch data are tensors, so numpy shape/max is not used: n_features is unused (0),
            # n_classes is fixed at MNIST's 10. self.weights is an unused (1, 1) placeholder.
            self.n_features = 0
            self.n_classes = 10
            self._torch = TorchBackend(model, torch_partitions, self.val_data, test_data, seed=self.seed)
            self.weights = np.zeros((1, 1), dtype=float)
        self.round_idx = 0
        self.current_acc = 0.0  # validation accuracy (RL state and reward)
        self.current_loss = 0.0  # validation loss
        self.test_acc = 0.0  # reported metric only; never used for decisions
        self.test_loss = 0.0
        self.cum_latency_s = 0.0
        self.cum_wh = 0.0
        self.cum_co2_g = 0.0
        self.participation_counts = np.zeros(self.n_clients, dtype=int)
        self.constraint_violations = 0
        self.prev_gini = 0.0
        self.measure_train_time = bool(measure_train_time)
        self._ref_train_s: Optional[float] = None

    def set_scalarization_weights(
        self,
        accuracy_weight: float,
        latency_weight: float,
        carbon_weight: float,
        fairness_weight: float = 0.0,
    ) -> None:
        """Update MORL linear scalarization weights."""
        self.accuracy_weight = float(accuracy_weight)
        self.latency_weight = float(latency_weight)
        self.carbon_weight = float(carbon_weight)
        self.fairness_weight = float(fairness_weight)

    def _get_obs(self) -> np.ndarray:
        progress = self.round_idx / max(1, self.max_rounds)
        total_part = max(1, int(np.sum(self.participation_counts)))
        part_rates = self.participation_counts.astype(float) / total_part
        speeds = np.array([c.relative_speed for c in self.clients], dtype=float)
        cis = np.array([c.grid_kgco2_per_kwh for c in self.clients], dtype=float)
        return np.concatenate(
            [
                np.array([progress, self.current_acc, self.current_loss], dtype=float),
                part_rates,
                speeds,
                cis,
            ]
        )

    def discretize_state(
        self, acc: Optional[float] = None, round_idx: Optional[int] = None
    ) -> Tuple[int, int]:
        """Map continuous environment state into discrete (phase, accuracy_bucket)."""
        a = self.current_acc if acc is None else float(acc)
        r = self.round_idx if round_idx is None else int(round_idx)
        phase = min(3, int(4 * r / max(1, self.max_rounds)))
        bucket = 0 if a < 0.50 else 1 if a < 0.68 else 2 if a < 0.76 else 3 if a < 0.81 else 4
        return phase, bucket

    def validate_action(self, action: Union[int, Sequence[int]]) -> Tuple[int, Tuple[int, ...]]:
        """Validate action index or client tuple and return (action_index, selected_tuple)."""
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            if idx < 0 or idx >= len(self.actions):
                raise ValueError(f"Action index {idx} out of range [0, {len(self.actions) - 1}].")
            return idx, self.actions[idx]

        chosen = tuple(sorted(int(i) for i in action))
        if len(chosen) == 0 or len(set(chosen)) != len(chosen):
            raise ValueError(f"Invalid client selection {action}: must be non-empty unique IDs.")
        for i in chosen:
            if i < 0 or i >= self.n_clients:
                raise ValueError(f"Client ID {i} out of bounds [0, {self.n_clients - 1}].")
        idx = self.actions.index(chosen) if chosen in self.actions else -1
        return idx, chosen

    def _evaluate_global(self) -> None:
        """Refresh validation (RL) and test (reported) metrics of the global model."""
        if self._torch is None:
            self.current_acc = evaluate_accuracy(self.weights, self.val_data)
            self.current_loss = cross_entropy_loss(self.weights, self.val_data)
            self.test_acc = evaluate_accuracy(self.weights, self.test_data)
            self.test_loss = cross_entropy_loss(self.weights, self.test_data)
        else:
            self.current_acc, self.current_loss = self._torch.evaluate(self.val_data)
            self.test_acc, self.test_loss = self._torch.evaluate(self.test_data)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment state to round 0 and return initial observation vector."""
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        if self._torch is None:
            self.weights = np.zeros((self.n_features, self.n_classes), dtype=float)
        else:
            self._torch.reset()
        self.round_idx = 0
        self._evaluate_global()
        self.cum_latency_s = 0.0
        self.cum_wh = 0.0
        self.cum_co2_g = 0.0
        self.participation_counts = np.zeros(self.n_clients, dtype=int)
        self.constraint_violations = 0
        self.prev_gini = 0.0
        return self._get_obs()

    def step(
        self,
        action: Union[int, Sequence[int]],
        speed_multipliers: Optional[Sequence[float]] = None,
        power_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one federated training round and return (obs, scalar_reward, done, info)."""
        if self.round_idx >= self.max_rounds:
            raise RuntimeError("Episode already finished; call reset() before step().")

        action_idx, selected_ids = self.validate_action(action)
        prev_acc = self.current_acc

        if speed_multipliers is None:
            if self.speed_jitter_fraction > 0:
                speed_multipliers = self.rng.uniform(
                    1.0 - self.speed_jitter_fraction,
                    1.0 + self.speed_jitter_fraction,
                    size=self.n_clients,
                )
            else:
                speed_multipliers = np.ones(self.n_clients, dtype=float)

        # Availability is sampled once per client, before training, in selected order.
        trained_ids = tuple(i for i in selected_ids if self.clients[i].is_available(self.rng))
        skipped_ids = tuple(i for i in selected_ids if i not in trained_ids)

        # Per-client train seconds, aligned with trained_ids (torch mode has no LocalTrainResult).
        if self._torch is not None:
            if trained_ids:
                train_secs, agg_weights = self._torch.train_round(
                    trained_ids, self.local_epochs, self.lr
                )
            else:
                train_secs, agg_weights = [], np.array([])
            local_results = []
        elif trained_ids:
            self.weights, agg_weights, local_results = run_fedavg_round(
                self.weights,
                self.clients,
                trained_ids,
                epochs=self.local_epochs,
                lr=self.lr,
            )
            train_secs = [res.train_seconds for res in local_results]
        else:
            agg_weights = np.array([])
            local_results = []
            train_secs = []
        self._evaluate_global()

        telemetry_speeds = speed_multipliers
        measured_scale: Dict[int, float] = {}
        if self.measure_train_time and trained_ids:
            # Copy so the caller's speed_multipliers array is not mutated.
            telemetry_speeds = np.array(speed_multipliers, dtype=float)
            for i, secs in zip(trained_ids, train_secs):
                if secs <= 0:
                    continue
                if self._ref_train_s is None:
                    self._ref_train_s = secs
                scale = float(np.clip(secs / self._ref_train_s, 0.5, 2.0))
                measured_scale[i] = scale
                telemetry_speeds[i] *= scale

        telemetry = simulate_round_telemetry(
            self.clients,
            trained_ids,
            base_train_seconds=self.base_train_seconds,
            aggregation_seconds=self.aggregation_seconds,
            coordinator_power_w=self.coordinator_power_w,
            speed_multipliers=telemetry_speeds,
            power_scales=power_scales,
            allow_empty=True,
        )

        for i in trained_ids:
            self.participation_counts[i] += 1

        self.round_idx += 1
        self.cum_latency_s += telemetry.round_latency_s
        self.cum_wh += telemetry.total_wh
        self.cum_co2_g += telemetry.total_co2_g

        acc_gain = self.current_acc - prev_acc
        gini = compute_gini(self.participation_counts)
        gini_delta = gini - self.prev_gini
        self.prev_gini = gini
        shortfall = max(0.0, self.accuracy_floor - self.current_acc)

        # 4D Multi-Objective Vector Reward: [accuracy_gain, -latency_s, -co2_g, -fairness_gini_delta]
        vector_reward = np.array(
            [acc_gain, -telemetry.round_latency_s, -telemetry.total_co2_g, -gini_delta],
            dtype=float,
        )
        scalar_reward = float(
            self.accuracy_weight * acc_gain
            - self.latency_weight * telemetry.round_latency_s
            - self.carbon_weight * telemetry.total_co2_g
            - self.fairness_weight * gini_delta
            - 0.6 * shortfall
        )

        lat_exceeded = (
            self.latency_budget_s is not None and self.cum_latency_s > self.latency_budget_s
        )
        co2_exceeded = (
            self.carbon_budget_g is not None and self.cum_co2_g > self.carbon_budget_g
        )
        unavailable_selected = len(skipped_ids) > 0
        if lat_exceeded or co2_exceeded or unavailable_selected:
            self.constraint_violations += 1

        done = self.round_idx >= self.max_rounds
        info: Dict[str, Any] = {
            "model": self.model,
            "round": self.round_idx,
            "action_index": action_idx,
            "selected_ids": selected_ids,
            "trained_ids": trained_ids,
            "skipped_ids": skipped_ids,
            "measured_train_scale": measured_scale,
            "selected_names": tuple(self.clients[i].name for i in selected_ids),
            # "accuracy"/"loss" are TEST metrics (reported); "val_*" and accuracy_gain drive RL.
            "accuracy": self.test_acc,
            "loss": self.test_loss,
            "val_accuracy": self.current_acc,
            "val_loss": self.current_loss,
            "accuracy_gain": acc_gain,
            "round_latency_s": telemetry.round_latency_s,
            "round_wh": telemetry.total_wh,
            "round_co2_g": telemetry.total_co2_g,
            "cum_latency_s": self.cum_latency_s,
            "cum_wh": self.cum_wh,
            "cum_co2_g": self.cum_co2_g,
            "aggregation_weights": agg_weights,
            "participation_counts": self.participation_counts.copy(),
            "participation_gini": gini,
            "gini_delta": gini_delta,
            "vector_reward": vector_reward,
            "reward_components": {
                "accuracy_gain": acc_gain,
                "neg_latency_s": -telemetry.round_latency_s,
                "neg_co2_g": -telemetry.total_co2_g,
                "neg_gini": -gini_delta,
                "shortfall_penalty": -0.6 * shortfall,
            },
            "constraints": {
                "latency_budget_exceeded": bool(lat_exceeded),
                "carbon_budget_exceeded": bool(co2_exceeded),
                "accuracy_floor_shortfall": float(shortfall),
                "unavailable_selected": bool(unavailable_selected),
                "total_violations": self.constraint_violations,
            },
            "telemetry": telemetry,
            "local_results": local_results,
        }
        return self._get_obs(), scalar_reward, done, info
