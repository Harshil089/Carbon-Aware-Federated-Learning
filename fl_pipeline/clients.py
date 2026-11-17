"""Layer 2: Heterogeneous FederatedClient hardware/carbon profile and local NumPy training."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from .data import Partition

PC1_METER_KWH = 0.065
PC1_METER_MINUTES = 190.0
PC1_MEAN_ACTIVE_W = PC1_METER_KWH * 1000.0 / (PC1_METER_MINUTES / 60.0)


def softmax_probs(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(weights: np.ndarray, data: Partition) -> float:
    """Mean multi-class cross-entropy loss on (x, y)."""
    x, y = data
    probs = softmax_probs(x @ weights)
    eps = 1e-12
    chosen = np.clip(probs[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(chosen)))


def evaluate_accuracy(weights: np.ndarray, data: Partition) -> float:
    """Classification accuracy of linear softmax weights on (x, y)."""
    x, y = data
    preds = np.argmax(x @ weights, axis=1)
    return float(np.mean(preds == y))


@dataclass
class LocalTrainResult:
    client_id: int
    name: str
    weights: np.ndarray
    num_samples: int
    initial_loss: float
    final_loss: float


@dataclass
class FederatedClient:
    """Hardware, network, and carbon profile for a federated client node."""

    client_id: int
    name: str
    ram_gb: int
    gpu_vram_gb: int
    relative_speed: float
    active_power_w: float
    idle_power_w: float
    bandwidth_mbps: float
    grid_kgco2_per_kwh: float = 0.675
    availability: float = 1.0
    base_network_latency_s: float = 0.30
    provenance: str = "SYNTHETIC_SCENARIO"
    partition: Optional[Partition] = None

    def __post_init__(self) -> None:
        if self.relative_speed <= 0:
            raise ValueError("relative_speed must be > 0.")
        if self.active_power_w < 0 or self.idle_power_w < 0:
            raise ValueError("Power values must be non-negative.")
        if self.bandwidth_mbps <= 0:
            raise ValueError("bandwidth_mbps must be > 0.")
        if self.grid_kgco2_per_kwh < 0:
            raise ValueError("grid_kgco2_per_kwh must be non-negative.")
        if not 0.0 <= self.availability <= 1.0:
            raise ValueError("availability must be in [0, 1].")

    @property
    def num_samples(self) -> int:
        if self.partition is None:
            return 0
        return int(len(self.partition[1]))

    def is_available(self, rng: Optional[np.random.Generator] = None) -> bool:
        """Sample client availability for a round."""
        if self.availability >= 1.0:
            return True
        if self.availability <= 0.0:
            return False
        if rng is None:
            return self.availability >= 0.5
        return bool(rng.random() <= self.availability)

    def train_local(
        self,
        global_weights: np.ndarray,
        epochs: int = 2,
        lr: float = 0.23,
        partition: Optional[Partition] = None,
    ) -> LocalTrainResult:
        """Run local full-batch softmax gradient descent on the client's partition."""
        part = partition if partition is not None else self.partition
        if part is None:
            raise ValueError(f"Client {self.name} has no data partition assigned.")
        x, y = part
        if len(y) == 0:
            raise ValueError(f"Client {self.name} has an empty partition.")
        if epochs < 1:
            raise ValueError("epochs must be >= 1.")
        if lr <= 0:
            raise ValueError("lr must be > 0.")

        num_classes = global_weights.shape[1]
        w = global_weights.astype(float, copy=True)
        onehot = np.eye(num_classes, dtype=float)[y]
        init_loss = cross_entropy_loss(w, part)

        for _ in range(epochs):
            probs = softmax_probs(x @ w)
            grad = (x.T @ (probs - onehot)) / len(y)
            w -= lr * grad

        final_loss = cross_entropy_loss(w, part)
        return LocalTrainResult(
            client_id=self.client_id,
            name=self.name,
            weights=w,
            num_samples=int(len(y)),
            initial_loss=init_loss,
            final_loss=final_loss,
        )


def default_clients_from_config(
    config_path: Optional[Path] = None,
    partitions: Optional[List[Partition]] = None,
) -> List[FederatedClient]:
    """Load the 4 PC1..PC4 nodes from config.json with anchored PC1 active power."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    default_grid = float(cfg.get("grid_factor_kgco2_per_kwh", 0.675))
    meas = cfg["actual_measurement"]
    pc1_w = float(meas["meter_energy_kwh"]) * 1000.0 / (float(meas["meter_duration_minutes"]) / 60.0)

    default_bw = {"PC1": 45.0, "PC2": 32.0, "PC3": 25.0, "PC4": 32.0}
    clients: List[FederatedClient] = []
    for idx, node in enumerate(cfg["nodes"]):
        name = str(node["name"])
        ratio = float(node["active_power_ratio_to_PC1"])
        net_s = float(node.get("network_seconds_assumed", 0.30))
        bw = float(node.get("bandwidth_mbps", default_bw.get(name, max(10.0, 15.0 / max(net_s, 0.1)))))
        part = partitions[idx] if partitions is not None and idx < len(partitions) else None
        clients.append(
            FederatedClient(
                client_id=idx,
                name=name,
                ram_gb=int(node["ram_gb"]),
                gpu_vram_gb=int(node["gpu_vram_gb"]),
                relative_speed=float(node["relative_speed"]),
                active_power_w=pc1_w * ratio,
                idle_power_w=float(node["idle_power_w_assumed"]),
                bandwidth_mbps=bw,
                grid_kgco2_per_kwh=float(node.get("grid_kgco2_per_kwh", default_grid)),
                availability=float(node.get("availability", 1.0)),
                base_network_latency_s=net_s,
                provenance=(
                    "MEASURED_PC1_METER_ANCHOR"
                    if name == "PC1"
                    else "SYNTHETIC_CALIBRATED_SCENARIO"
                ),
                partition=part,
            )
        )
    return clients
