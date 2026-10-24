"""Simulated latency, energy and carbon of the *virtual* federated clients (brief steps 9-11).

The simulator trains clients one after another on a single machine, so the machine's wall time and
energy (``energy_tracker.py``) say nothing about the devices being modelled. Here each selected client is
treated as a separate remote device running in parallel, using its profile from ``client_profiles.py``:

  T_download = M_bits / B_down + network delay
  T_train    = samples * epochs / (base_samples_per_second * compute_speed_factor)
  T_upload   = M_bits / B_up + network delay
  T_round    = max_i (T_download + T_train + T_upload) + T_aggregation        (synchronous; slowest client)

  E_i = compute_power * T_train + comm_power * (T_download + T_upload)
  C_i = E_i * client grid intensity;  server: E = server_power * T_round at the server's grid intensity

Every value here is SIMULATED from placeholder profile numbers, not measured.
"""
from dataclasses import dataclass

from torch import nn

from src.simulation.carbon import J_PER_KWH, energy_to_carbon_g
from src.simulation.client_profiles import ClientProfile


@dataclass
class SystemModel:
    # Training throughput (samples/s) of a client with compute_speed_factor 1.0 for this model. Placeholder.
    base_samples_per_second: float = 1000.0
    # Network interface power while sending or receiving the model. Placeholder.
    comm_power_watts: float = 5.0
    # Aggregation server: powered for the whole round, on its own grid.
    server_power_watts: float = 200.0
    server_carbon_intensity_gco2e_kwh: float = 400.0
    bytes_per_param: int = 4  # float32 weights, no compression

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SystemModel":
        return cls(**(cfg or {}))

    def model_bits(self, model: nn.Module) -> float:
        return 8.0 * self.bytes_per_param * sum(p.numel() for p in model.parameters())

    @staticmethod
    def transfer_time_s(bits: float, bandwidth_mbps: float, latency_ms: float) -> float:
        return bits / (bandwidth_mbps * 1e6) + latency_ms / 1000.0

    def client_cost(self, profile: ClientProfile, num_samples: int, epochs: int, model_bits: float) -> dict:
        download = self.transfer_time_s(model_bits, profile.bandwidth_mbps, profile.latency_ms)
        upload = self.transfer_time_s(model_bits, profile.upload_bandwidth, profile.latency_ms)
        train = num_samples * epochs / (self.base_samples_per_second * profile.compute_speed_factor)
        compute_j = profile.compute_power_watts * train
        comm_j = self.comm_power_watts * (download + upload)
        energy_kwh = (compute_j + comm_j) / J_PER_KWH
        return {
            "download_s": download,
            "train_s": train,
            "upload_s": upload,
            "total_s": download + train + upload,
            "compute_energy_kwh": compute_j / J_PER_KWH,
            "comm_energy_kwh": comm_j / J_PER_KWH,
            "energy_kwh": energy_kwh,
            "carbon_gco2e": energy_to_carbon_g(energy_kwh, profile.carbon_intensity_gco2e_kwh),
        }

    def round_cost(self, client_costs: dict[int, dict], aggregation_time_s: float) -> dict:
        """Aggregate per-client costs (keyed by client id) into the synchronous round's latency/energy/carbon."""
        straggler = max(client_costs, key=lambda cid: client_costs[cid]["total_s"])
        latency = client_costs[straggler]["total_s"] + aggregation_time_s
        server_kwh = self.server_power_watts * latency / J_PER_KWH
        server_g = energy_to_carbon_g(server_kwh, self.server_carbon_intensity_gco2e_kwh)
        client_kwh = sum(c["energy_kwh"] for c in client_costs.values())
        client_g = sum(c["carbon_gco2e"] for c in client_costs.values())
        return {
            "latency_s": latency,
            "straggler_client_id": straggler,
            "client_energy_kwh": client_kwh,
            "server_energy_kwh": server_kwh,
            "energy_kwh": client_kwh + server_kwh,
            "client_carbon_gco2e": client_g,
            "server_carbon_gco2e": server_g,
            "carbon_gco2e": client_g + server_g,
        }
