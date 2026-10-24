import csv
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Writes round-level and client-level CSVs plus a JSON config snapshot for one run."""

    ROUND_FIELDS = [
        "experiment_id", "scheduler", "round", "selected_client_ids", "num_participants",
        "global_accuracy", "global_loss", "accuracy_improvement", "mean_local_loss",
        "round_wall_time_s", "aggregation_time_s", "cumulative_wall_time_s",
        "round_energy_kwh", "round_carbon_gco2e", "cumulative_energy_kwh", "cumulative_carbon_gco2e",
        # Simulated federated system (src/simulation/system_model.py); the four columns above are the
        # measured energy of the machine running the simulation.
        "sim_round_latency_s", "sim_straggler_client_id", "sim_cumulative_latency_s",
        "sim_client_energy_kwh", "sim_server_energy_kwh", "sim_round_energy_kwh",
        "sim_round_carbon_gco2e", "sim_cumulative_energy_kwh", "sim_cumulative_carbon_gco2e",
    ]
    CLIENT_FIELDS = [
        "experiment_id", "seed", "round", "client_id", "tier", "selected", "eligible",
        "dataset_size", "local_epochs", "train_wall_time_s", "local_loss",
        "participation_count", "compute_power_watts", "compute_speed_factor",
        "carbon_intensity_gco2e_kwh", "bandwidth_mbps", "upload_bandwidth_mbps", "latency_ms",
        "sim_download_s", "sim_train_s", "sim_upload_s", "sim_total_s",
        "sim_compute_energy_kwh", "sim_comm_energy_kwh", "sim_energy_kwh", "sim_carbon_gco2e",
    ]

    def __init__(self, run_dir: str | Path, experiment_id: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.round_path = self.run_dir / "rounds.csv"
        self.client_path = self.run_dir / "clients.csv"
        self._init_csv(self.round_path, self.ROUND_FIELDS)
        self._init_csv(self.client_path, self.CLIENT_FIELDS)

    @staticmethod
    def _init_csv(path: Path, fields: list[str]) -> None:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append(self, path: Path, fields: list[str], row: dict[str, Any]) -> None:
        row = {"experiment_id": self.experiment_id, **row}
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    def log_round(self, row: dict[str, Any]) -> None:
        self._append(self.round_path, self.ROUND_FIELDS, row)

    def log_client(self, row: dict[str, Any]) -> None:
        self._append(self.client_path, self.CLIENT_FIELDS, row)

    def save_json(self, name: str, payload: dict[str, Any]) -> None:
        with open(self.run_dir / name, "w") as f:
            json.dump(payload, f, indent=2, default=str)
