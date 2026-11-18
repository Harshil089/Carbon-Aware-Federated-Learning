"""Client node for the real-hardware FedAvg run (run on PC1/PC2/PC3). Requires flwr (lazy)."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fl_pipeline.data import make_federated_dataset

from .common import EnergyTracker, load_nodes_config
from .server import MISSING_FLWR, TORCH_NOT_IMPLEMENTED, WEIGHT_SHAPE


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-hardware FedAvg client (measured path).")
    p.add_argument("--name", required=True, help="Node name from config.json, e.g. PC1")
    p.add_argument("--server", required=True, help="Coordinator address, e.g. 192.168.1.10:8080")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.23)
    p.add_argument("--model", choices=["numpy", "torch_mlp"], default="numpy")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    nodes, _ = load_nodes_config()
    if args.name not in nodes:
        print(f"error: unknown node {args.name!r}; config.json has {sorted(nodes)}", file=sys.stderr)
        return 2
    if args.model != "numpy":
        print(TORCH_NOT_IMPLEMENTED, file=sys.stderr)
        return 2
    if args.local_epochs < 1 or args.lr <= 0:
        print("error: --local-epochs must be >= 1 and --lr must be > 0", file=sys.stderr)
        return 2

    try:
        import flwr as fl  # lazy: optional dependency
    except ImportError:
        print(MISSING_FLWR, file=sys.stderr)
        return 2

    from fl_pipeline.clients import default_clients_from_config

    # Same partitioning as the simulator: 4 nodes, node index = position in config.json.
    idx = list(nodes).index(args.name)
    partitions, _ = make_federated_dataset(n_clients=4, seed=args.seed)
    local = default_clients_from_config()[idx]
    local.partition = partitions[idx]

    class HardwareClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [np.zeros(WEIGHT_SHAPE)]

        def fit(self, parameters, config):
            t0 = time.perf_counter()
            with EnergyTracker() as tracker:
                result = local.train_local(parameters[0], epochs=args.local_epochs, lr=args.lr)
            seconds = time.perf_counter() - t0
            metrics = {
                "client_name": args.name,
                "train_seconds": float(seconds),
                "energy_wh": float(tracker.wh) if tracker.wh is not None else -1.0,
            }
            return [result.weights], result.num_samples, metrics

        def evaluate(self, parameters, config):
            return 0.0, 0, {}  # evaluation is done centrally by the server

    print(f"{args.name}: partition has {len(local.partition[1])} samples; connecting to {args.server}")
    fl.client.start_numpy_client(server_address=args.server, client=HardwareClient())
    return 0


if __name__ == "__main__":
    sys.exit(main())
