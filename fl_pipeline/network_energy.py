"""Layer 3: Parallel Round Latency, Active/Idle/Coordinator Energy, and Per-Client Carbon."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

from .clients import FederatedClient


@dataclass
class ClientRoundTelemetry:
    client_id: int
    name: str
    selected: bool
    t_download_s: float
    t_train_s: float
    t_upload_s: float
    t_client_total_s: float
    idle_s: float
    active_wh: float
    idle_wh: float
    total_wh: float
    total_kwh: float
    grid_kgco2_per_kwh: float
    co2_g: float


@dataclass
class RoundTelemetry:
    selected_ids: Tuple[int, ...]
    round_latency_s: float
    aggregation_s: float
    coordinator_wh: float
    coordinator_co2_g: float
    total_active_wh: float
    total_idle_wh: float
    total_wh: float
    total_kwh: float
    total_co2_g: float
    total_co2_kg: float
    per_client: Dict[int, ClientRoundTelemetry] = field(default_factory=dict)


def simulate_round_telemetry(
    clients: Sequence[FederatedClient],
    selected_ids: Sequence[int],
    base_train_seconds: float = 16.0,
    aggregation_seconds: float = 0.45,
    coordinator_power_w: float = 30.0,
    coordinator_grid_kgco2_per_kwh: float = 0.675,
    payload_mb: float = 0.0,
    speed_multipliers: Optional[Sequence[float]] = None,
    power_scales: Optional[Sequence[float]] = None,
    grid_overrides: Optional[Sequence[float]] = None,
) -> RoundTelemetry:
    """Compute parallel round latency, active/idle/coordinator energy, and per-client CO2.

    Latency model:
        For each selected client i:
            T_download_i = 0.5 * base_network_latency_s + (payload_mb * 8 / bandwidth_mbps)
            T_train_i    = base_train_seconds * speed_multiplier_i / relative_speed_i
            T_upload_i   = 0.5 * base_network_latency_s + (payload_mb * 8 / bandwidth_mbps)
        Round latency = max_{i in selected}(T_download_i + T_train_i + T_upload_i) + aggregation_seconds

    Energy & Carbon model:
        Each client i runs active compute for T_train_i (0 if unselected) and idles for
        (round_latency - T_train_i). Coordinator draws coordinator_power_w for round_latency.
        Per-client CO2 (g) = E_i (kWh) * CI_i (kg CO2/kWh) * 1000 = Wh_i * CI_i.
    """
    n_clients = len(clients)
    selected_tuple = tuple(int(i) for i in selected_ids)
    if len(selected_tuple) == 0:
        raise ValueError("selected_ids must contain at least one client.")
    selected_set = set(selected_tuple)
    for i in selected_tuple:
        if i < 0 or i >= n_clients:
            raise ValueError(f"Invalid client index {i} for {n_clients} clients.")

    speed_mult = (
        np.ones(n_clients, dtype=float)
        if speed_multipliers is None
        else np.asarray(speed_multipliers, dtype=float)
    )
    pwr_scale = (
        np.ones(n_clients, dtype=float)
        if power_scales is None
        else np.asarray(power_scales, dtype=float)
    )

    t_dl: Dict[int, float] = {}
    t_tr: Dict[int, float] = {}
    t_ul: Dict[int, float] = {}
    t_tot: Dict[int, float] = {}

    for i, c in enumerate(clients):
        if i in selected_set:
            transfer_s = (payload_mb * 8.0) / c.bandwidth_mbps if payload_mb > 0 else 0.0
            dl_s = 0.5 * c.base_network_latency_s + transfer_s
            ul_s = 0.5 * c.base_network_latency_s + transfer_s
            tr_s = (base_train_seconds * float(speed_mult[i])) / c.relative_speed
            t_dl[i] = dl_s
            t_tr[i] = tr_s
            t_ul[i] = ul_s
            t_tot[i] = dl_s + tr_s + ul_s
        else:
            t_dl[i] = 0.0
            t_tr[i] = 0.0
            t_ul[i] = 0.0
            t_tot[i] = 0.0

    round_latency_s = max(t_tot[i] for i in selected_tuple) + float(aggregation_seconds)

    coord_wh = (float(coordinator_power_w) * round_latency_s) / 3600.0
    coord_co2_g = coord_wh * float(coordinator_grid_kgco2_per_kwh)

    per_client: Dict[int, ClientRoundTelemetry] = {}
    total_active_wh = 0.0
    total_idle_wh = 0.0
    clients_co2_g = 0.0

    for i, c in enumerate(clients):
        work_s = t_tr[i]
        idle_s = max(0.0, round_latency_s - work_s)
        active_w = c.active_power_w * float(pwr_scale[i])
        active_wh = (active_w * work_s) / 3600.0
        idle_wh = (c.idle_power_w * idle_s) / 3600.0
        node_wh = active_wh + idle_wh
        node_kwh = node_wh / 1000.0
        ci = float(grid_overrides[i]) if grid_overrides is not None else c.grid_kgco2_per_kwh
        # E_kWh * CI (kgCO2/kWh) * 1000 (g/kg) == node_wh * CI
        node_co2_g = node_kwh * ci * 1000.0

        total_active_wh += active_wh
        total_idle_wh += idle_wh
        clients_co2_g += node_co2_g

        per_client[i] = ClientRoundTelemetry(
            client_id=c.client_id,
            name=c.name,
            selected=(i in selected_set),
            t_download_s=t_dl[i],
            t_train_s=work_s,
            t_upload_s=t_ul[i],
            t_client_total_s=t_tot[i],
            idle_s=idle_s,
            active_wh=active_wh,
            idle_wh=idle_wh,
            total_wh=node_wh,
            total_kwh=node_kwh,
            grid_kgco2_per_kwh=ci,
            co2_g=node_co2_g,
        )

    total_wh = coord_wh + total_active_wh + total_idle_wh
    total_kwh = total_wh / 1000.0
    total_co2_g = coord_co2_g + clients_co2_g

    return RoundTelemetry(
        selected_ids=selected_tuple,
        round_latency_s=round_latency_s,
        aggregation_s=float(aggregation_seconds),
        coordinator_wh=coord_wh,
        coordinator_co2_g=coord_co2_g,
        total_active_wh=total_active_wh,
        total_idle_wh=total_idle_wh,
        total_wh=total_wh,
        total_kwh=total_kwh,
        total_co2_g=total_co2_g,
        total_co2_kg=total_co2_g / 1000.0,
        per_client=per_client,
    )
