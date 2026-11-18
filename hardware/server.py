"""Coordinator for the real-hardware FedAvg run. Requires flwr (imported lazily in main)."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from fl_pipeline.clients import evaluate_accuracy
from fl_pipeline.data import make_federated_dataset

from .common import (
    EnergyTracker,
    build_round_row,
    load_nodes_config,
    parse_client_list,
    parse_wallmeter,
    write_per_round_csv,
)

WEIGHT_SHAPE = (3, 3)  # 3 features incl. bias, 3 classes (matches fl_pipeline softmax model)
MISSING_FLWR = "error: flwr is not installed. Run: pip install -r requirements-optional.txt"
TORCH_NOT_IMPLEMENTED = "error: --model torch_mlp is not implemented for the hardware path yet; use --model numpy"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-hardware FedAvg coordinator (measured path).")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--clients", required=True, help="Comma-separated node names, e.g. PC1,PC2,PC3")
    p.add_argument("--address", default="0.0.0.0:8080")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results_hardware/per_round_HARDWARE.csv")
    p.add_argument("--wallmeter", default=None,
                   help='Whole-run Wh per node, e.g. "PC1=65.0,PC2=20.0". Recorded on the final round row.')
    p.add_argument("--model", choices=["numpy", "torch_mlp"], default="numpy")
    return p


def _make_strategy(fl, args, n_clients, test_set, grid, wallmeter_total, rows):
    """Build the recording FedAvg strategy. Defined here because it needs flwr classes."""
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    class RecordingFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._t0 = None
            self._cum_seconds = 0.0

        def configure_fit(self, server_round, parameters, client_manager):
            self._t0 = time.perf_counter()
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            wall = time.perf_counter() - self._t0
            params, metrics = super().aggregate_fit(server_round, results, failures)
            if params is None:
                return params, metrics
            accuracy = evaluate_accuracy(parameters_to_ndarrays(params)[0], test_set)
            names = [str(res.metrics["client_name"]) for _, res in results]
            per_node = {str(res.metrics["client_name"]): float(res.metrics["train_seconds"])
                        for _, res in results}
            energies = [float(res.metrics.get("energy_wh", -1.0)) for _, res in results]
            codecarbon_wh = sum(energies) if energies and all(e >= 0 for e in energies) else None
            is_last = server_round == args.rounds
            rows.append(build_round_row(
                round_no=server_round,
                seed=args.seed,
                selected_names=sorted(names),
                accuracy=accuracy,
                wall_seconds=wall,
                per_node_seconds=per_node,
                codecarbon_wh=codecarbon_wh,
                wallmeter_wh=wallmeter_total if is_last else None,
                cum_seconds=self._cum_seconds,
                grid_kgco2_per_kwh=grid,
            ))
            self._cum_seconds += wall
            print(f"round {server_round}: clients={sorted(names)} acc={accuracy:.4f} wall={wall:.2f}s")
            return params, metrics

    return RecordingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # evaluation is server-side on the held-out test split
        min_fit_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=ndarrays_to_parameters([np.zeros(WEIGHT_SHAPE)]),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        names = parse_client_list(args.clients)
        wallmeter = parse_wallmeter(args.wallmeter)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.model != "numpy":
        print(TORCH_NOT_IMPLEMENTED, file=sys.stderr)
        return 2
    if args.rounds < 1:
        print("error: --rounds must be >= 1", file=sys.stderr)
        return 2

    try:
        import flwr as fl  # lazy: optional dependency
    except ImportError:
        print(MISSING_FLWR, file=sys.stderr)
        return 2

    grid = load_nodes_config()[1]
    _, test_set = make_federated_dataset(seed=args.seed)
    wallmeter_total = sum(wallmeter.values()) if wallmeter else None
    rows: list = []
    strategy = _make_strategy(fl, args, len(names), test_set, grid, wallmeter_total, rows)
    try:
        with EnergyTracker() as coordinator:
            fl.server.start_server(
                server_address=args.address,
                config=fl.server.ServerConfig(num_rounds=args.rounds),
                strategy=strategy,
            )
    finally:
        out = write_per_round_csv(args.output, rows)
        print(f"wrote {len(rows)} round(s) to {out}")
        if coordinator.wh is not None:
            print(f"coordinator process energy (CodeCarbon estimate): {coordinator.wh:.4f} Wh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
