"""Layer 5: Client Selection Schedulers, Tabular Q-Learning, and MORL Scalarization Sweep."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .clients import FederatedClient
from .rl_env import FederatedRLEnv, compute_gini


def estimate_client_latency(
    client: FederatedClient,
    base_train_seconds: float = 16.0,
    payload_mb: float = 0.0,
) -> float:
    transfer_s = (2.0 * payload_mb * 8.0) / client.bandwidth_mbps if payload_mb > 0 else 0.0
    return base_train_seconds / client.relative_speed + client.base_network_latency_s + transfer_s


def estimate_client_active_co2_g(
    client: FederatedClient,
    base_train_seconds: float = 16.0,
) -> float:
    train_s = base_train_seconds / client.relative_speed
    active_wh = (client.active_power_w * train_s) / 3600.0
    return active_wh * client.grid_kgco2_per_kwh


class BaseScheduler:
    """Abstract base interface for federated client selection schedulers."""

    name: str = "BaseScheduler"

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: int = 2,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        raise NotImplementedError


class RandomScheduler(BaseScheduler):
    """Selects k clients uniformly at random using a seeded generator."""

    name = "Random"

    def __init__(self, k: int = 2, seed: int = 42) -> None:
        self.k = int(k)
        self.rng = np.random.default_rng(seed)

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        num_select = min(len(clients), self.k if k is None else int(k))
        chosen = self.rng.choice(len(clients), size=num_select, replace=False)
        return tuple(sorted(int(i) for i in chosen))


class RoundRobinScheduler(BaseScheduler):
    """Cycles through clients deterministically across rounds."""

    name = "RoundRobin"

    def __init__(self, k: int = 2) -> None:
        self.k = int(k)

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        n = len(clients)
        num_select = min(n, self.k if k is None else int(k))
        start = (round_idx * num_select) % n
        chosen = [(start + offset) % n for offset in range(num_select)]
        return tuple(sorted(chosen))


class FastestClientScheduler(BaseScheduler):
    """Selects the k clients with the lowest estimated per-round latency."""

    name = "FastestClient"

    def __init__(self, k: int = 2, base_train_seconds: float = 16.0) -> None:
        self.k = int(k)
        self.base_train_seconds = float(base_train_seconds)

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        num_select = min(len(clients), self.k if k is None else int(k))
        ranked = sorted(
            range(len(clients)),
            key=lambda i: estimate_client_latency(clients[i], self.base_train_seconds),
        )
        return tuple(sorted(ranked[:num_select]))


class LowestCarbonScheduler(BaseScheduler):
    """Selects the k clients with the lowest estimated active carbon footprint."""

    name = "LowestCarbon"

    def __init__(self, k: int = 2, base_train_seconds: float = 16.0) -> None:
        self.k = int(k)
        self.base_train_seconds = float(base_train_seconds)

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        num_select = min(len(clients), self.k if k is None else int(k))
        ranked = sorted(
            range(len(clients)),
            key=lambda i: estimate_client_active_co2_g(clients[i], self.base_train_seconds),
        )
        return tuple(sorted(ranked[:num_select]))


class WeightedHeuristicScheduler(BaseScheduler):
    """Selects k clients minimizing w_lat * norm_latency + w_co2 * norm_co2 - w_data * norm_samples."""

    name = "WeightedHeuristic"

    def __init__(
        self,
        k: int = 2,
        latency_weight: float = 0.45,
        carbon_weight: float = 0.45,
        data_weight: float = 0.10,
        base_train_seconds: float = 16.0,
    ) -> None:
        self.k = int(k)
        self.latency_weight = float(latency_weight)
        self.carbon_weight = float(carbon_weight)
        self.data_weight = float(data_weight)
        self.base_train_seconds = float(base_train_seconds)

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        num_select = min(len(clients), self.k if k is None else int(k))
        lats = np.array([estimate_client_latency(c, self.base_train_seconds) for c in clients])
        co2s = np.array([estimate_client_active_co2_g(c, self.base_train_seconds) for c in clients])
        samples = np.array([max(1, c.num_samples) for c in clients], dtype=float)

        norm_lat = lats / max(1e-9, float(np.max(lats)))
        norm_co2 = co2s / max(1e-9, float(np.max(co2s)))
        norm_samples = samples / max(1e-9, float(np.max(samples)))

        scores = (
            self.latency_weight * norm_lat
            + self.carbon_weight * norm_co2
            - self.data_weight * norm_samples
        )
        ranked = np.argsort(scores)[:num_select]
        return tuple(sorted(int(i) for i in ranked))


class FairnessGiniScheduler(BaseScheduler):
    """Selects k under-participating clients to minimize participation Gini combined with carbon."""

    name = "FairnessGini"

    def __init__(
        self,
        k: int = 2,
        fairness_priority: float = 0.75,
        base_train_seconds: float = 16.0,
    ) -> None:
        self.k = int(k)
        self.fairness_priority = float(fairness_priority)
        self.base_train_seconds = float(base_train_seconds)
        self._internal_counts: Optional[np.ndarray] = None

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        n = len(clients)
        num_select = min(n, self.k if k is None else int(k))
        if round_idx == 0 or self._internal_counts is None or len(self._internal_counts) != n:
            self._internal_counts = np.zeros(n, dtype=float)
        counts = (
            np.asarray(participation_counts, dtype=float)
            if participation_counts is not None
            else self._internal_counts
        )
        norm_counts = counts / max(1.0, float(np.max(counts)))
        co2s = np.array([estimate_client_active_co2_g(c, self.base_train_seconds) for c in clients])
        norm_co2 = co2s / max(1e-9, float(np.max(co2s)))

        cost = self.fairness_priority * norm_counts + (1.0 - self.fairness_priority) * norm_co2
        chosen = sorted(int(i) for i in np.argsort(cost)[:num_select])
        for i in chosen:
            self._internal_counts[i] += 1.0
        return tuple(chosen)


class TabularQLearningScheduler(BaseScheduler):
    """Tabular epsilon-greedy Q-learning scheduler over combinatorial client subsets."""

    name = "TabularQLearning"

    def __init__(
        self,
        actions: Sequence[Tuple[int, ...]],
        alpha: float = 0.24,
        gamma: float = 0.86,
        epsilon_start: float = 0.80,
        epsilon_min: float = 0.06,
        seed: int = 42,
    ) -> None:
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.seed = int(seed)
        self.q_table: Dict[Tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.actions), dtype=float)
        )

    def train(
        self,
        env: FederatedRLEnv,
        episodes: int = 80,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        envs: Optional[Sequence[FederatedRLEnv]] = None,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """Train tabular Q-values on FederatedRLEnv for the specified number of episodes.

        envs: optional pool of environments built on different data seeds; episode ep runs
            on envs[ep % len(envs)] while sharing one Q table. None trains on `env` only.
        """
        pool = [env] if envs is None else list(envs)
        if envs is not None and (not pool or any(len(e.actions) != len(self.actions) for e in pool)):
            raise ValueError("envs must be non-empty and share the scheduler's action list.")
        rng = np.random.default_rng(self.seed)
        for ep in range(episodes):
            cur = pool[ep % len(pool)]
            cur.reset(seed=self.seed + ep)
            eps = max(self.epsilon_min, self.epsilon_start * (1.0 - ep / max(1, episodes)))
            for r in range(cur.max_rounds):
                s = cur.discretize_state()
                if rng.random() < eps:
                    a_idx = int(rng.integers(len(self.actions)))
                else:
                    q_vals = self.q_table[s]
                    best = np.flatnonzero(q_vals == q_vals.max())
                    a_idx = int(rng.choice(best))
                _, reward, done, _ = cur.step(self.actions[a_idx])
                sp = cur.discretize_state()
                target = reward + (0.0 if done else self.gamma * float(np.max(self.q_table[sp])))
                self.q_table[s][a_idx] += self.alpha * (target - self.q_table[s][a_idx])
                if done:
                    break
            if progress_callback is not None:
                progress_callback(ep + 1, episodes)
        return self.q_table

    def select(
        self,
        clients: Sequence[FederatedClient],
        round_idx: int = 0,
        k: Optional[int] = None,
        participation_counts: Optional[Sequence[int]] = None,
        acc: float = 0.0,
        max_rounds: int = 20,
    ) -> Tuple[int, ...]:
        phase = min(3, int(4 * round_idx / max(1, max_rounds)))
        bucket = (
            0 if acc < 0.50 else 1 if acc < 0.68 else 2 if acc < 0.76 else 3 if acc < 0.81 else 4
        )
        q_vals = self.q_table[(phase, bucket)]
        a_idx = int(np.argmax(q_vals))
        return self.actions[a_idx]


@dataclass
class MORLSweepPoint:
    preference_name: str
    accuracy_weight: float
    latency_weight: float
    carbon_weight: float
    fairness_weight: float
    final_accuracy: float
    total_latency_s: float
    total_wh: float
    total_co2_g: float
    participation_gini: float
    pareto_optimal: bool = False


def extract_pareto_frontier(points: Sequence[MORLSweepPoint]) -> List[MORLSweepPoint]:
    """Identify non-dominated Pareto-optimal points maximizing accuracy & minimizing latency, CO2."""
    frontier: List[MORLSweepPoint] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            no_worse = (
                q.final_accuracy >= p.final_accuracy
                and q.total_latency_s <= p.total_latency_s
                and q.total_co2_g <= p.total_co2_g
            )
            strictly_better = (
                q.final_accuracy > p.final_accuracy
                or q.total_latency_s < p.total_latency_s
                or q.total_co2_g < p.total_co2_g
            )
            if no_worse and strictly_better:
                dominated = True
                break
        p.pareto_optimal = not dominated
        if not dominated:
            frontier.append(p)
    return frontier


def morl_scalarization_sweep(
    env: FederatedRLEnv,
    preference_grid: Optional[Sequence[Tuple[str, float, float, float, float]]] = None,
    episodes: int = 40,
    eval_seed: int = 999,
) -> List[MORLSweepPoint]:
    """Train and evaluate Tabular Q-Learning across multiple MORL preference vectors."""
    if preference_grid is None:
        preference_grid = [
            ("Accuracy_First", 25.0, 0.005, 0.05, 0.02),
            ("Balanced_MORL", 15.0, 0.018, 0.32, 0.05),
            ("Latency_First", 10.0, 0.080, 0.05, 0.02),
            ("Low_Carbon_First", 10.0, 0.010, 0.90, 0.02),
            ("Fairness_Carbon", 12.0, 0.015, 0.45, 0.40),
        ]

    saved = (env.accuracy_weight, env.latency_weight, env.carbon_weight, env.fairness_weight)
    results: List[MORLSweepPoint] = []
    try:
        for idx, (name, w_acc, w_lat, w_co2, w_fair) in enumerate(preference_grid):
            env.set_scalarization_weights(w_acc, w_lat, w_co2, w_fair)
            agent = TabularQLearningScheduler(env.actions, seed=env.seed + 100 * idx)
            agent.train(env, episodes=episodes)

            env.reset(seed=eval_seed)
            info = {}
            for r in range(env.max_rounds):
                chosen = agent.select(
                    env.clients,
                    round_idx=r,
                    acc=env.current_acc,
                    max_rounds=env.max_rounds,
                )
                _, _, done, info = env.step(chosen)
                if done:
                    break

            results.append(
                MORLSweepPoint(
                    preference_name=name,
                    accuracy_weight=w_acc,
                    latency_weight=w_lat,
                    carbon_weight=w_co2,
                    fairness_weight=w_fair,
                    final_accuracy=float(info["accuracy"]),
                    total_latency_s=float(info["cum_latency_s"]),
                    total_wh=float(info["cum_wh"]),
                    total_co2_g=float(info["cum_co2_g"]),
                    participation_gini=float(info["participation_gini"]),
                )
            )

    finally:
        env.set_scalarization_weights(*saved)
    extract_pareto_frontier(results)
    return results
