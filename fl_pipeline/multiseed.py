"""Multi-Seed Federated Learning + MORL Experiment Runner and 2D Pareto Hypervolume."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .clients import default_clients_from_config
from .data import make_federated_dataset
from .experiment import (
    PolicyEvaluationResult,
    build_clients_from_telemetry,
    evaluate_scheduler,
)
from .rl_env import FederatedRLEnv
from .schedulers import (
    BaseScheduler,
    FairnessGiniScheduler,
    FastestClientScheduler,
    LowestCarbonScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    TabularQLearningScheduler,
    WeightedHeuristicScheduler,
)
from .torch_fl import DQNScheduler

Point2DInput = Union[
    Tuple[float, float],
    Sequence[float],
    Mapping[str, Any],
    PolicyEvaluationResult,
    Any,
]


def _extract_scalar(val: Any) -> float:
    """Extract a float scalar from a number or a {'mean': ...} mapping."""
    if isinstance(val, Mapping) and "mean" in val:
        return float(val["mean"])
    return float(val)


def _extract_co2_acc(point: Point2DInput) -> Tuple[float, float]:
    """Extract (co2_g, accuracy) from a tuple, dict, or policy evaluation object."""
    if isinstance(point, Mapping):
        co2_val = None
        for k in ("total_co2_g_mean", "total_co2_g", "co2_g", "co2", "max_co2"):
            if k in point:
                co2_val = _extract_scalar(point[k])
                break
        acc_val = None
        for k in ("final_accuracy_mean", "final_accuracy", "accuracy", "acc", "min_acc"):
            if k in point:
                acc_val = _extract_scalar(point[k])
                break
        if co2_val is None or acc_val is None:
            raise ValueError(f"Mapping point {point!r} must contain CO2 and accuracy keys.")
        return co2_val, acc_val

    for co2_attr in ("total_co2_g_mean", "total_co2_g", "co2_g"):
        if hasattr(point, co2_attr):
            for acc_attr in ("final_accuracy_mean", "final_accuracy", "accuracy"):
                if hasattr(point, acc_attr):
                    return (
                        _extract_scalar(getattr(point, co2_attr)),
                        _extract_scalar(getattr(point, acc_attr)),
                    )

    if isinstance(point, (tuple, list, np.ndarray)) and len(point) >= 2:
        return float(point[0]), float(point[1])

    raise TypeError(f"Cannot extract (co2_g, accuracy) from point: {point!r}")


def compute_hypervolume_2d(
    points: Iterable[Point2DInput],
    ref_point: Union[Tuple[float, float], Sequence[float], Mapping[str, Any]],
) -> float:
    """Compute 2D Pareto hypervolume for (co2_g minimized, accuracy maximized).

    Args:
        points: Iterable of candidate points `(co2_g, accuracy)` (or dicts/objects with
            `total_co2_g` and `final_accuracy`).
        ref_point: Explicit reference point `(max_co2, min_acc)`. Only points strictly
            dominating `ref_point` (`co2_g < max_co2` and `accuracy > min_acc`) contribute
            to the hypervolume.

    Returns:
        Non-negative 2D Lebesgue measure (area) dominated by the Pareto frontier.
    """
    ref_co2, ref_acc = _extract_co2_acc(ref_point)
    if not (np.isfinite(ref_co2) and np.isfinite(ref_acc)):
        raise ValueError(f"ref_point must contain finite values, got {ref_point!r}.")

    valid: List[Tuple[float, float]] = []
    for pt in points:
        co2, acc = _extract_co2_acc(pt)
        if not (np.isfinite(co2) and np.isfinite(acc)):
            continue
        if co2 < ref_co2 and acc > ref_acc:
            valid.append((co2, acc))

    if not valid:
        return 0.0

    # Sort by CO2 ascending; break ties by accuracy descending
    valid.sort(key=lambda item: (item[0], -item[1]))

    hypervolume = 0.0
    best_acc = ref_acc
    for co2, acc in valid:
        if acc > best_acc:
            hypervolume += (ref_co2 - co2) * (acc - best_acc)
            best_acc = acc

    return float(max(0.0, hypervolume))


class MetricSummary(dict):
    """Dictionary {'mean': float, 'std': float, 'values': List[float]} with attribute access."""

    def __init__(self, mean: float, std: float, values: Sequence[float]) -> None:
        vals = [float(v) for v in values]
        super().__init__(mean=float(mean), std=float(std), values=vals)
        self.mean = float(mean)
        self.std = float(std)
        self._values = vals

    def __float__(self) -> float:
        return self.mean


class PolicyMultiSeedSummary(dict):
    """Per-policy multi-seed summary supporting both flat keys, nested dicts, and attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _summarize_metric(values: Sequence[float]) -> MetricSummary:
    arr = np.asarray(values, dtype=float)
    return MetricSummary(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        values=[float(v) for v in arr],
    )


def run_multiseed_experiment(
    seeds: Sequence[int] = (42, 43, 44),
    mode: str = "dirichlet",
    alpha: float = 0.5,
    rounds: int = 8,
    q_episodes: int = 15,
    include_dqn: bool = True,
    use_calibrated_telemetry: bool = True,
    telemetry_path: Optional[Path] = None,
    ref_point: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Run FederatedRLEnv and all schedulers (including DQNScheduler) across multiple seeds.

    Args:
        seeds: Sequence of random seeds to evaluate.
        mode: Partition mode (`"dirichlet"` or `"iid"`).
        alpha: Dirichlet concentration parameter.
        rounds: Federated communication rounds per episode.
        q_episodes: Training episodes for TabularQLearningScheduler and DQNScheduler.
        include_dqn: Whether to train and evaluate PyTorch `DQNScheduler`.
        use_calibrated_telemetry: Whether to load client hardware profiles from the 10k CSV.
        telemetry_path: Optional override path for the calibrated telemetry CSV.
        ref_point: Optional explicit `(max_co2, min_acc)` reference point for 2D hypervolume.

    Returns:
        Dictionary containing:
        - `config`: experiment configuration including `seeds` and `include_dqn`
        - `policies`: dict mapping `policy_name -> PolicyMultiSeedSummary`
        - `policy_list`: ordered list of `PolicyMultiSeedSummary` objects
        - `per_policy`: alias for `policies` dict (`policy_name -> PolicyMultiSeedSummary`)
        - `pareto_policies`: list of Pareto-optimal policy names on mean (Accuracy, Latency, CO2)
        - `ref_point`: explicit `(max_co2, min_acc)` tuple
        - `hypervolume`: 2D Pareto hypervolume of mean policy outcomes relative to `ref_point`
        - `per_seed_hypervolumes`: list of per-seed 2D hypervolumes relative to `ref_point`
        - `hypervolume_mean` / `hypervolume_std`: mean and std of per-seed hypervolumes
    """
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("seeds cannot be empty.")

    per_policy_runs: Dict[str, List[PolicyEvaluationResult]] = {}
    per_seed_results: Dict[int, List[PolicyEvaluationResult]] = {}
    policy_order: List[str] = []

    for seed in seed_list:
        partitions, test_data = make_federated_dataset(
            n_clients=4,
            mode=mode,
            alpha=alpha,
            seed=seed,
        )
        if use_calibrated_telemetry:
            clients = build_clients_from_telemetry(csv_path=telemetry_path, partitions=partitions)
        else:
            clients = default_clients_from_config(partitions=partitions)

        env = FederatedRLEnv(
            clients=clients,
            test_data=test_data,
            max_rounds=rounds,
            seed=seed,
        )

        q_scheduler = TabularQLearningScheduler(env.actions, seed=seed + 100)
        q_scheduler.train(env, episodes=q_episodes)

        schedulers: List[BaseScheduler] = [
            RandomScheduler(k=2, seed=seed),
            RoundRobinScheduler(k=2),
            FastestClientScheduler(k=2),
            LowestCarbonScheduler(k=2),
            WeightedHeuristicScheduler(k=2),
            FairnessGiniScheduler(k=2),
            q_scheduler,
        ]
        if include_dqn:
            dqn_scheduler = DQNScheduler(obs_dim=15, actions=env.actions, seed=seed + 200)
            dqn_scheduler.train(env, episodes=q_episodes)
            schedulers.append(dqn_scheduler)

        seed_evals: List[PolicyEvaluationResult] = []
        for sched in schedulers:
            res = evaluate_scheduler(env, sched, eval_seed=seed + 500)
            seed_evals.append(res)
            if sched.name not in per_policy_runs:
                per_policy_runs[sched.name] = []
                policy_order.append(sched.name)
            per_policy_runs[sched.name].append(res)

        per_seed_results[seed] = seed_evals

    # Determine explicit reference point (max_co2, min_acc) if not provided
    all_co2 = [r.total_co2_g for runs in per_policy_runs.values() for r in runs]
    if ref_point is None:
        max_co2_ref = round(float(max(all_co2) * 1.15 + 0.25), 6)
        min_acc_ref = 0.0
        explicit_ref_point: Tuple[float, float] = (max_co2_ref, min_acc_ref)
    else:
        explicit_ref_point = (float(ref_point[0]), float(ref_point[1]))

    policies_by_name: Dict[str, PolicyMultiSeedSummary] = {}
    policy_list: List[PolicyMultiSeedSummary] = []

    for name in policy_order:
        runs = per_policy_runs[name]
        acc_stat = _summarize_metric([r.final_accuracy for r in runs])
        lat_stat = _summarize_metric([r.total_latency_s for r in runs])
        co2_stat = _summarize_metric([r.total_co2_g for r in runs])
        gini_stat = _summarize_metric([r.participation_gini for r in runs])
        wh_stat = _summarize_metric([r.total_wh for r in runs])
        loss_stat = _summarize_metric([r.final_loss for r in runs])

        summary = PolicyMultiSeedSummary(
            policy_name=name,
            n_seeds=len(runs),
            seeds=list(seed_list),
            # Nested {'mean': ..., 'std': ..., 'values': ...}
            final_accuracy=acc_stat,
            total_latency_s=lat_stat,
            total_co2_g=co2_stat,
            participation_gini=gini_stat,
            total_wh=wh_stat,
            final_loss=loss_stat,
            # Flat convenience keys
            final_accuracy_mean=acc_stat.mean,
            final_accuracy_std=acc_stat.std,
            total_latency_s_mean=lat_stat.mean,
            total_latency_s_std=lat_stat.std,
            total_co2_g_mean=co2_stat.mean,
            total_co2_g_std=co2_stat.std,
            participation_gini_mean=gini_stat.mean,
            participation_gini_std=gini_stat.std,
            total_wh_mean=wh_stat.mean,
            total_wh_std=wh_stat.std,
            final_loss_mean=loss_stat.mean,
            final_loss_std=loss_stat.std,
            pareto_optimal=False,
        )
        policies_by_name[name] = summary
        policy_list.append(summary)

    # Mark 3D Pareto optimality across mean (Accuracy maximized, Latency & CO2 minimized)
    pareto_policies: List[str] = []
    for i, p in enumerate(policy_list):
        dominated = False
        for j, q in enumerate(policy_list):
            if i == j:
                continue
            no_worse = (
                q["final_accuracy_mean"] >= p["final_accuracy_mean"]
                and q["total_latency_s_mean"] <= p["total_latency_s_mean"]
                and q["total_co2_g_mean"] <= p["total_co2_g_mean"]
            )
            strictly_better = (
                q["final_accuracy_mean"] > p["final_accuracy_mean"]
                or q["total_latency_s_mean"] < p["total_latency_s_mean"]
                or q["total_co2_g_mean"] < p["total_co2_g_mean"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        p["pareto_optimal"] = not dominated
        if not dominated:
            pareto_policies.append(p["policy_name"])

    mean_points = [
        (p["total_co2_g_mean"], p["final_accuracy_mean"]) for p in policy_list
    ]
    hypervolume = compute_hypervolume_2d(mean_points, explicit_ref_point)

    per_seed_hvs = [
        compute_hypervolume_2d(
            [(r.total_co2_g, r.final_accuracy) for r in per_seed_results[s]],
            explicit_ref_point,
        )
        for s in seed_list
    ]
    hv_stat = _summarize_metric(per_seed_hvs)

    return {
        "config": {
            "seeds": seed_list,
            "mode": mode,
            "alpha": alpha,
            "rounds": rounds,
            "q_episodes": q_episodes,
            "include_dqn": include_dqn,
            "use_calibrated_telemetry": use_calibrated_telemetry,
        },
        "policies": policies_by_name,
        "per_policy": policies_by_name,
        "policy_list": policy_list,
        "pareto_policies": pareto_policies,
        "ref_point": explicit_ref_point,
        "ref_point_dict": {
            "max_co2": explicit_ref_point[0],
            "min_acc": explicit_ref_point[1],
        },
        "hypervolume": hypervolume,
        "per_seed_hypervolumes": per_seed_hvs,
        "hypervolume_mean": hv_stat.mean,
        "hypervolume_std": hv_stat.std,
    }
