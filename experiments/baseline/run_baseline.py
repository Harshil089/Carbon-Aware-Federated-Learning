"""FedAvg baseline: simulated clients, IID or Dirichlet non-IID MNIST / Fashion-MNIST / CIFAR-10, CSV logs.

Usage (from the repository root):
    python experiments/baseline/run_baseline.py
    python experiments/baseline/run_baseline.py --partition dirichlet --alpha 0.1 --rounds 30
    python experiments/baseline/run_baseline.py --config configs/experiment_cifar10.yaml --alpha 0.1
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import copy  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.datasets.iid_partition import iid_partition  # noqa: E402
from src.datasets.loader import load_dataset  # noqa: E402
from src.datasets.noniid_partition import class_distribution, dirichlet_partition  # noqa: E402
from src.federated.client import FederatedClient  # noqa: E402
from src.federated.server import FederatedServer  # noqa: E402
from src.federated.training import evaluate  # noqa: E402
from src.models import build_model, count_parameters  # noqa: E402
from src.simulation.client_profiles import load_client_profiles  # noqa: E402
from src.simulation.energy_tracker import EnergyTracker  # noqa: E402
from src.simulation.system_model import SystemModel  # noqa: E402
from src.utils.config import apply_overrides, load_yaml  # noqa: E402
from src.utils.logger import ExperimentLogger  # noqa: E402
from src.utils.seeds import resolve_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--dataset", choices=["mnist", "fashion_mnist", "cifar10"])
    p.add_argument("--model", choices=["cnn", "cifar_cnn", "mlp"])
    p.add_argument("--partition", choices=["iid", "dirichlet"])
    p.add_argument("--alpha", type=float, help="Dirichlet concentration (non-IID only)")
    p.add_argument("--rounds", type=int)
    p.add_argument("--clients-per-round", type=int)
    p.add_argument("--local-epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--clients-config", help="Client profile YAML, e.g. configs/clients_pool50.yaml")
    p.add_argument("--device", help="auto | cpu | cuda")
    p.add_argument("--carbon-intensity", type=float, help="Grid carbon intensity in gCO2e/kWh")
    p.add_argument("--no-energy", action="store_true", help="Disable energy/carbon tracking")
    p.add_argument("--results-dir")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config)
    return apply_overrides(config, {
        "dataset": args.dataset,
        "model": args.model,
        "partition.type": args.partition,
        "partition.alpha": args.alpha,
        "federated.num_rounds": args.rounds,
        "federated.clients_per_round": args.clients_per_round,
        "federated.local_epochs": args.local_epochs,
        "seed": args.seed,
        "clients_config": args.clients_config,
        "device": args.device,
        "energy.carbon_intensity_gco2e_kwh": args.carbon_intensity,
        "energy.enabled": False if args.no_energy else None,
        "results_dir": args.results_dir,
    })


def make_partitions(labels: np.ndarray, num_clients: int, part_cfg: dict, seed: int) -> list[np.ndarray]:
    if part_cfg["type"] == "iid":
        return iid_partition(labels, num_clients, seed)
    if part_cfg["type"] == "dirichlet":
        return dirichlet_partition(labels, num_clients, part_cfg["alpha"], seed, part_cfg["min_samples_per_client"])
    raise ValueError(f"Unknown partition type {part_cfg['type']!r}")


def save_plots(run_dir: Path, rounds: pd.DataFrame, dist: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    ax[0].plot(rounds["round"], rounds["global_accuracy"], marker="o")
    ax[0].set(xlabel="Round", ylabel="Test accuracy", title="Global accuracy")
    ax[1].plot(rounds["round"], rounds["global_loss"], marker="o", color="tab:red")
    ax[1].set(xlabel="Round", ylabel="Test loss", title="Global loss")
    ax[2].plot(rounds["sim_cumulative_latency_s"] / 60, rounds["global_accuracy"], marker="o", color="tab:purple")
    ax[2].set(xlabel="Simulated training time (min)", ylabel="Test accuracy", title="Accuracy vs simulated latency")
    ax[3].plot(rounds["sim_cumulative_carbon_gco2e"], rounds["global_accuracy"], marker="o", color="tab:green")
    ax[3].set(xlabel="Simulated emissions (gCO2e)", ylabel="Test accuracy", title="Accuracy vs simulated carbon")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_curve.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(dist, aspect="auto", cmap="viridis")
    ax.set(xlabel="Class", ylabel="Client", title="Client class distribution",
           xticks=range(dist.shape[1]), yticks=range(dist.shape[0]))
    for (i, j), v in np.ndenumerate(dist):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=7,
                color="k" if v > 0.6 * dist.max() else "w")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run_dir / "class_distribution.png", dpi=120)
    plt.close(fig)


def run(config: dict) -> Path:
    seed = config["seed"]
    fed = config["federated"]
    part = config["partition"]
    set_seed(seed)
    device = resolve_device(config["device"])

    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    train, test = load_dataset(config["dataset"], str(data_dir))
    profiles = load_client_profiles(str(ROOT / config["clients_config"]))
    num_clients = len(profiles)

    labels = train.y.numpy()
    shards = make_partitions(labels, num_clients, part, seed)
    num_classes = int(labels.max()) + 1
    dist = class_distribution(labels, shards, num_classes)

    part_tag = "iid" if part["type"] == "iid" else f"dir{part['alpha']}"
    experiment_id = f"{config['experiment_name']}_{config['dataset']}_{part_tag}_seed{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    results_dir = Path(config["results_dir"])
    run_dir = (results_dir if results_dir.is_absolute() else ROOT / results_dir) / experiment_id
    logger = ExperimentLogger(run_dir, experiment_id)

    clients = [FederatedClient(p, train.subset(s).to(device), seed) for p, s in zip(profiles, shards)]
    clients_by_id = {c.client_id: c for c in clients}
    model = build_model(config["model"], num_classes).to(device)
    server = FederatedServer(model, clients, test.to(device), seed)
    system = SystemModel.from_config(config.get("system_model"))
    model_bits = system.model_bits(model)

    init_acc, init_loss = evaluate(server.model, server.test_data)
    logger.save_json("config.json", {
        **config, "experiment_id": experiment_id, "device_used": str(device),
        "torch_version": torch.__version__, "model_parameters": count_parameters(model),
        "num_train_samples": len(train), "num_test_samples": len(test),
        "num_clients": num_clients, "model_size_mb": model_bits / 8e6,
    })
    logger.save_json("client_profiles.json", {"clients": [p.as_dict() for p in profiles]})
    pd.DataFrame(dist, columns=[f"class_{k}" for k in range(num_classes)]).rename_axis("client_id").to_csv(
        run_dir / "class_distribution.csv")
    np.savez_compressed(run_dir / "partitions.npz", **{f"client_{p.client_id}": s for p, s in zip(profiles, shards)})

    sizes = [len(s) for s in shards]
    print(f"Run {experiment_id} on {device}; {count_parameters(model):,} params; "
          f"{num_clients} clients, sizes min/median/max {min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")
    print(f"round  0  acc {init_acc:.4f}  loss {init_loss:.4f}  (initial model)")

    ecfg = config["energy"]
    tracker = EnergyTracker(
        ecfg["carbon_intensity_gco2e_kwh"], ecfg["sample_interval_s"],
        ecfg["cpu_tdp_watts"], ecfg["cpu_idle_fraction"],
        track_gpu=device.type == "cuda",
    ) if ecfg["enabled"] else None
    if tracker:
        tracker.start()
        print(f"Energy tracking: CPU={tracker.cpu_source}, GPU={tracker.gpu_source}, "
              f"intensity={ecfg['carbon_intensity_gco2e_kwh']} gCO2e/kWh (assumed)")
    prev_energy, prev_carbon = 0.0, 0.0

    prev_acc, cumulative = init_acc, 0.0
    sim_latency = sim_energy = sim_carbon = 0.0
    best_acc, best_state = init_acc, copy.deepcopy(model.state_dict())
    for rnd in range(1, fed["num_rounds"] + 1):
        start = time.perf_counter()
        out = server.run_round(fed)
        wall = time.perf_counter() - start
        cumulative += wall
        res = out["client_results"]
        costs = {cid: system.client_cost(clients_by_id[cid].profile, r["num_samples"], fed["local_epochs"], model_bits)
                 for cid, r in res.items()}
        sim = system.round_cost(costs, out["aggregation_time_s"])
        sim_latency += sim["latency_s"]
        sim_energy += sim["energy_kwh"]
        sim_carbon += sim["carbon_gco2e"]

        for c in clients:
            selected = c.client_id in res
            cost = costs.get(c.client_id)
            logger.log_client({
                "seed": seed, "round": rnd, "client_id": c.client_id, "tier": c.profile.tier,
                "selected": int(selected), "eligible": int(c.profile.availability),
                "dataset_size": c.num_samples, "local_epochs": fed["local_epochs"] if selected else 0,
                "train_wall_time_s": round(res[c.client_id]["train_wall_time_s"], 4) if selected else "",
                "local_loss": round(res[c.client_id]["local_loss"], 5) if selected else "",
                "participation_count": c.profile.participation_count,
                "compute_power_watts": c.profile.compute_power_watts,
                "compute_speed_factor": c.profile.compute_speed_factor,
                "carbon_intensity_gco2e_kwh": c.profile.carbon_intensity_gco2e_kwh,
                "bandwidth_mbps": c.profile.bandwidth_mbps, "upload_bandwidth_mbps": c.profile.upload_bandwidth,
                "latency_ms": c.profile.latency_ms,
                **({
                    "sim_download_s": round(cost["download_s"], 4), "sim_train_s": round(cost["train_s"], 4),
                    "sim_upload_s": round(cost["upload_s"], 4), "sim_total_s": round(cost["total_s"], 4),
                    "sim_compute_energy_kwh": f"{cost['compute_energy_kwh']:.8f}",
                    "sim_comm_energy_kwh": f"{cost['comm_energy_kwh']:.8f}",
                    "sim_energy_kwh": f"{cost['energy_kwh']:.8f}", "sim_carbon_gco2e": f"{cost['carbon_gco2e']:.6f}",
                } if cost else {}),
            })
        acc = out["global_accuracy"]
        energy_row = {}
        if tracker:
            snap = tracker.snapshot()
            energy_row = {
                "round_energy_kwh": f"{snap['energy_kwh'] - prev_energy:.8f}",
                "round_carbon_gco2e": f"{snap['carbon_gco2e'] - prev_carbon:.6f}",
                "cumulative_energy_kwh": f"{snap['energy_kwh']:.8f}",
                "cumulative_carbon_gco2e": f"{snap['carbon_gco2e']:.6f}",
            }
            prev_energy, prev_carbon = snap["energy_kwh"], snap["carbon_gco2e"]
        logger.log_round({
            **energy_row,
            "scheduler": "random" if fed["clients_per_round"] < num_clients else "all_clients",
            "round": rnd, "selected_client_ids": " ".join(map(str, out["selected"])),
            "num_participants": len(out["selected"]),
            "global_accuracy": round(acc, 5), "global_loss": round(out["global_loss"], 5),
            "accuracy_improvement": round(acc - prev_acc, 5),
            "mean_local_loss": round(float(np.mean([r["local_loss"] for r in res.values()])), 5),
            "round_wall_time_s": round(wall, 4), "aggregation_time_s": round(out["aggregation_time_s"], 5),
            "cumulative_wall_time_s": round(cumulative, 4),
            "sim_round_latency_s": round(sim["latency_s"], 4),
            "sim_straggler_client_id": sim["straggler_client_id"],
            "sim_cumulative_latency_s": round(sim_latency, 4),
            "sim_client_energy_kwh": f"{sim['client_energy_kwh']:.8f}",
            "sim_server_energy_kwh": f"{sim['server_energy_kwh']:.8f}",
            "sim_round_energy_kwh": f"{sim['energy_kwh']:.8f}",
            "sim_round_carbon_gco2e": f"{sim['carbon_gco2e']:.6f}",
            "sim_cumulative_energy_kwh": f"{sim_energy:.8f}",
            "sim_cumulative_carbon_gco2e": f"{sim_carbon:.6f}",
        })
        print(f"round {rnd:2d}  acc {acc:.4f}  loss {out['global_loss']:.4f}  "
              f"{len(out['selected'])} clients  sim {sim['latency_s']:.1f}s / {sim['carbon_gco2e']:.2f} gCO2e  "
              f"(machine {wall:.1f}s)")
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
        prev_acc = acc

    energy_summary = None
    if tracker:
        tracker.stop()
        energy_summary = tracker.summary()
        logger.save_json("energy.json", energy_summary)
        print(f"Energy: {energy_summary['energy_kwh'] * 1000:.3f} Wh "
              f"(CPU {energy_summary['cpu_energy_kwh'] * 1000:.3f} Wh [{energy_summary['cpu_source']}], "
              f"GPU {energy_summary['gpu_energy_kwh'] * 1000:.3f} Wh [{energy_summary['gpu_source']}]); "
              f"carbon {energy_summary['carbon_gco2e']:.3f} gCO2e @ {energy_summary['carbon_intensity_gco2e_kwh']} g/kWh")
    rounds = pd.read_csv(logger.round_path)
    save_plots(run_dir, rounds, dist, f"FedAvg {config['dataset']} ({part_tag}, {num_clients} clients)")
    torch.save(model.state_dict(), run_dir / "final_model.pt")
    torch.save(best_state, run_dir / "best_model.pt")
    summary = {
        "experiment_id": experiment_id, "dataset": config["dataset"], "model": config["model"],
        "partition": part["type"], "alpha": part["alpha"] if part["type"] == "dirichlet" else None,
        "seed": seed, "rounds": fed["num_rounds"], "initial_accuracy": init_acc,
        "final_accuracy": float(rounds["global_accuracy"].iloc[-1]), "best_accuracy": best_acc,
        "total_wall_time_s": cumulative, "device": str(device),
        "total_energy_kwh": energy_summary["energy_kwh"] if energy_summary else None,
        "total_carbon_gco2e": energy_summary["carbon_gco2e"] if energy_summary else None,
        "num_clients": num_clients, "clients_per_round": fed["clients_per_round"],
        "sim_total_latency_s": sim_latency, "sim_total_energy_kwh": sim_energy, "sim_total_carbon_gco2e": sim_carbon,
    }
    logger.save_json("summary.json", summary)
    print(f"Simulated federation: {sim_latency / 60:.1f} min, {sim_energy * 1000:.2f} Wh, {sim_carbon:.2f} gCO2e")
    print(f"Final accuracy {summary['final_accuracy']:.4f} (best {best_acc:.4f}); results in {run_dir}")
    return run_dir


if __name__ == "__main__":
    run(build_config(parse_args()))
