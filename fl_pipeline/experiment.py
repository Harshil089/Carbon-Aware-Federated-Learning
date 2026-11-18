"""Layer 7 & 8: End-to-End Federated Pipeline Runner, Telemetry Calibration, and Pareto Analysis."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .clients import FederatedClient, default_clients_from_config
from .data import Partition, make_federated_splits
from .rl_env import MODEL_CHOICES, FederatedRLEnv, compute_gini
from .schedulers import (
    BaseScheduler,
    FairnessGiniScheduler,
    FastestClientScheduler,
    LowestCarbonScheduler,
    MORLSweepPoint,
    RandomScheduler,
    RoundRobinScheduler,
    TabularQLearningScheduler,
    WeightedHeuristicScheduler,
    extract_pareto_frontier,
    morl_scalarization_sweep,
)

DEFAULT_TELEMETRY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "calibrated_telemetry_10k.csv"
)


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    final_accuracy: float
    final_loss: float
    total_latency_s: float
    total_wh: float
    total_co2_g: float
    participation_counts: List[int]
    participation_gini: float
    constraint_violations: int
    pareto_optimal: bool = False
    rounds: List[Dict[str, Any]] = field(default_factory=list)


def load_calibrated_telemetry_csv(
    csv_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Load and summarize the 10k calibrated telemetry CSV dataset per node."""
    path = DEFAULT_TELEMETRY_PATH if csv_path is None else Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibrated telemetry CSV not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Telemetry CSV at {path} is empty.")

    by_node: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_node.setdefault(r["node"], []).append(r)

    summary: Dict[str, Dict[str, float]] = {}
    for node, nrows in sorted(by_node.items()):
        first = nrows[0]
        summary[node] = {
            "count": float(len(nrows)),
            "ram_gb": float(first["ram_gb"]),
            "gpu_vram_gb": float(first["gpu_vram_gb"]),
            "active_power_w": float(np.mean([float(x["active_power_w"]) for x in nrows])),
            "idle_power_w": float(np.mean([float(x["idle_power_w"]) for x in nrows])),
            "relative_speed": float(np.mean([float(x["relative_speed"]) for x in nrows])),
            "network_s": float(np.mean([float(x["network_s"]) for x in nrows])),
            "bandwidth_mbps": float(np.mean([float(x["bandwidth_mbps"]) for x in nrows])),
            "grid_kgco2_per_kwh": float(np.mean([float(x["grid_kgco2_per_kwh"]) for x in nrows])),
            "availability": float(np.mean([float(x["availability"]) for x in nrows])),
        }
    return rows, summary


def build_clients_from_telemetry(
    csv_path: Optional[Path] = None,
    partitions: Optional[List[Partition]] = None,
) -> List[FederatedClient]:
    """Build FederatedClient profiles calibrated from data/calibrated_telemetry_10k.csv."""
    path = DEFAULT_TELEMETRY_PATH if csv_path is None else Path(csv_path)
    if not path.exists():
        return default_clients_from_config(partitions=partitions)

    _, summary = load_calibrated_telemetry_csv(path)
    clients: List[FederatedClient] = []
    for idx, node_name in enumerate(sorted(summary.keys())):
        stats = summary[node_name]
        part = partitions[idx] if partitions is not None and idx < len(partitions) else None
        clients.append(
            FederatedClient(
                client_id=idx,
                name=node_name,
                ram_gb=int(round(stats["ram_gb"])),
                gpu_vram_gb=int(round(stats["gpu_vram_gb"])),
                relative_speed=float(stats["relative_speed"]),
                active_power_w=float(stats["active_power_w"]),
                idle_power_w=float(stats["idle_power_w"]),
                bandwidth_mbps=float(stats["bandwidth_mbps"]),
                grid_kgco2_per_kwh=float(stats["grid_kgco2_per_kwh"]),
                availability=float(stats["availability"]),
                base_network_latency_s=float(stats["network_s"]),
                provenance="SYNTHETIC_CALIBRATED_FROM_PC1_METER",
                partition=part,
            )
        )
    return clients


def evaluate_scheduler(
    env: FederatedRLEnv,
    scheduler: BaseScheduler,
    eval_seed: int = 42,
) -> PolicyEvaluationResult:
    """Evaluate a scheduler policy over one full episode of FederatedRLEnv."""
    env.reset(seed=eval_seed)
    round_records: List[Dict[str, Any]] = []
    last_info: Dict[str, Any] = {}

    for r in range(env.max_rounds):
        chosen = scheduler.select(
            env.clients,
            round_idx=r,
            participation_counts=env.participation_counts,
            acc=env.current_acc,
            max_rounds=env.max_rounds,
        )
        _, reward, done, info = env.step(chosen)
        last_info = info
        round_records.append(
            {
                "policy": scheduler.name,
                "round": info["round"],
                "selected_ids": info["selected_ids"],
                "selected_names": "+".join(info["selected_names"]),
                "accuracy": round(float(info["accuracy"]), 5),
                "loss": round(float(info["loss"]), 5),
                "round_latency_s": round(float(info["round_latency_s"]), 5),
                "round_wh": round(float(info["round_wh"]), 6),
                "round_co2_g": round(float(info["round_co2_g"]), 6),
                "cum_latency_s": round(float(info["cum_latency_s"]), 5),
                "cum_wh": round(float(info["cum_wh"]), 6),
                "cum_co2_g": round(float(info["cum_co2_g"]), 6),
                "participation_gini": round(float(info["participation_gini"]), 5),
                "scalar_reward": round(float(reward), 5),
            }
        )
        if done:
            break

    return PolicyEvaluationResult(
        policy_name=scheduler.name,
        final_accuracy=float(last_info["accuracy"]),
        final_loss=float(last_info["loss"]),
        total_latency_s=float(last_info["cum_latency_s"]),
        total_wh=float(last_info["cum_wh"]),
        total_co2_g=float(last_info["cum_co2_g"]),
        participation_counts=[int(x) for x in last_info["participation_counts"]],
        participation_gini=float(last_info["participation_gini"]),
        constraint_violations=int(last_info["constraints"]["total_violations"]),
        rounds=round_records,
    )


def mark_policy_pareto_frontier(
    policies: Sequence[PolicyEvaluationResult],
) -> List[PolicyEvaluationResult]:
    """Mark non-dominated policies on the 3D (Accuracy, Latency, Carbon) Pareto frontier."""
    frontier: List[PolicyEvaluationResult] = []
    for i, p in enumerate(policies):
        dominated = False
        for j, q in enumerate(policies):
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


def run_pipeline_experiment(
    mode: str = "dirichlet",
    alpha: float = 0.5,
    rounds: int = 15,
    q_episodes: int = 40,
    morl_episodes: int = 25,
    use_calibrated_telemetry: bool = True,
    telemetry_path: Optional[Path] = None,
    seed: int = 42,
    train_data_seeds: int = 3,
    n_val: int = 500,
    model: str = "numpy",
) -> Dict[str, Any]:
    """Run the full 8-layer federated pipeline and return structured evaluation metrics.

    RL state/reward use a validation split; reported accuracy uses the held-out test split.
    train_data_seeds: extra data seeds whose envs form the Q-learning training pool (the
        evaluation env is not in the pool). 0 trains Q-learning on the evaluation env itself.
    model: "numpy" (synthetic Gaussian clusters, softmax-linear) or "torch_mlp"/"torch_cnn"
        (MNIST via load_mnist_partitions). In torch mode n_val is ignored: validation is the
        first 25% of the MNIST test tensors and the remaining 75% is the reported test set.
    """
    if model not in MODEL_CHOICES:
        raise ValueError(f"model must be one of {MODEL_CHOICES}, got {model!r}.")
    is_torch = model != "numpy"

    def make_clients(parts: Optional[List[Partition]]) -> List[FederatedClient]:
        if use_calibrated_telemetry:
            return build_clients_from_telemetry(csv_path=telemetry_path, partitions=parts)
        return default_clients_from_config(partitions=parts)

    def make_splits(split_seed: int) -> Tuple[List[Partition], Partition, Partition, str]:
        if not is_torch:
            parts, val, test = make_federated_splits(
                n_clients=4, mode=mode, alpha=alpha, n_val=n_val, seed=split_seed
            )
            return parts, val, test, "SYNTHETIC_GAUSSIAN_CLUSTERS"
        from .torch_fl import load_mnist_partitions  # lazy: keeps torch optional

        bundle = load_mnist_partitions(n_clients=4, mode=mode, alpha=alpha, seed=split_seed)
        x_all, y_all = bundle.test_data
        n_val_t = int(0.25 * len(y_all))
        val = (x_all[:n_val_t], y_all[:n_val_t])
        test = (x_all[n_val_t:], y_all[n_val_t:])
        return bundle.partitions, val, test, bundle.provenance

    def make_env(split_seed: int) -> Tuple[FederatedRLEnv, List[Partition], str]:
        parts, val, test, provenance = make_splits(split_seed)
        env = FederatedRLEnv(
            # Torch mode: clients carry hardware/telemetry only; data lives in torch_partitions.
            clients=make_clients(None if is_torch else parts),
            test_data=test,
            val_data=val,
            max_rounds=rounds,
            seed=seed,
            model=model,
            torch_partitions=parts if is_torch else None,
        )
        return env, parts, provenance

    env, parts, provenance = make_env(seed)
    clients = env.clients
    train_envs = [make_env(seed + 1000 * (k + 1))[0] for k in range(train_data_seeds)]

    q_scheduler = TabularQLearningScheduler(env.actions, seed=seed + 100)
    if train_envs:
        q_scheduler.train(env, episodes=q_episodes, envs=train_envs)
    else:
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

    policy_results = [
        evaluate_scheduler(env, sched, eval_seed=seed + 500) for sched in schedulers
    ]
    pareto_policies = mark_policy_pareto_frontier(policy_results)

    morl_points = morl_scalarization_sweep(
        env,
        episodes=morl_episodes,
        eval_seed=seed + 500,
    )
    pareto_morl = [pt for pt in morl_points if pt.pareto_optimal]

    return {
        "config": {
            "mode": mode,
            "alpha": alpha,
            "rounds": rounds,
            "q_episodes": q_episodes,
            "morl_episodes": morl_episodes,
            "use_calibrated_telemetry": use_calibrated_telemetry,
            "seed": seed,
            "train_data_seeds": train_data_seeds,
            "n_val": n_val,
            "model": model,
            "data_provenance": provenance,
            "data_split": {
                "n_train": sum(len(p[1]) for p in parts) if is_torch else 1000,
                "n_val": len(env.val_data[1]) if is_torch else n_val,
                "n_test": len(env.test_data[1]) if is_torch else 1500,
                "rl_selection_metric": "validation_accuracy",
                "reported_metric": "test_accuracy",
            },
        },
        "clients": [
            {
                "client_id": c.client_id,
                "name": c.name,
                "ram_gb": c.ram_gb,
                "gpu_vram_gb": c.gpu_vram_gb,
                "relative_speed": round(c.relative_speed, 4),
                "active_power_w": round(c.active_power_w, 4),
                "idle_power_w": round(c.idle_power_w, 4),
                "bandwidth_mbps": round(c.bandwidth_mbps, 4),
                "grid_kgco2_per_kwh": round(c.grid_kgco2_per_kwh, 4),
                "num_samples": len(parts[c.client_id][1]) if is_torch else c.num_samples,
                "provenance": c.provenance,
            }
            for c in clients
        ],
        "policies": policy_results,
        "pareto_policies": [p.policy_name for p in pareto_policies],
        "morl_sweep": [asdict(pt) for pt in morl_points],
        "pareto_morl": [pt.preference_name for pt in pareto_morl],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8-layer Federated + MORL Pipeline")
    parser.add_argument("--mode", default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seeds", type=int, default=3)
    parser.add_argument("--model", default="numpy", choices=list(MODEL_CHOICES))
    args = parser.parse_args()

    out = run_pipeline_experiment(
        mode=args.mode,
        alpha=args.alpha,
        rounds=args.rounds,
        q_episodes=args.episodes,
        seed=args.seed,
        train_data_seeds=args.train_seeds,
        model=args.model,
    )
    print(f"8-Layer Federated Pipeline ({args.mode}, alpha={args.alpha}, rounds={args.rounds})")
    if args.model != "numpy":
        print(f"Model: {args.model} | data provenance: {out['config']['data_provenance']}")
    for p in out["policies"]:
        flag = "*" if p.pareto_optimal else " "
        print(
            f" {flag} {p.policy_name:18s} | acc={p.final_accuracy:.3f} | "
            f"lat={p.total_latency_s:6.1f}s | co2={p.total_co2_g:6.3f}g | "
            f"gini={p.participation_gini:.3f}"
        )


if __name__ == "__main__":
    main()
