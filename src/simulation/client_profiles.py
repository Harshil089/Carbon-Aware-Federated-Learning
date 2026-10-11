from dataclasses import asdict, dataclass

import numpy as np

from src.utils.config import load_yaml


@dataclass
class ClientProfile:
    """Simulated hardware/network/grid profile. Values are placeholders, not measurements."""
    client_id: int
    tier: str
    compute_power_watts: float
    compute_speed_factor: float
    carbon_intensity_gco2e_kwh: float
    bandwidth_mbps: float
    latency_ms: float
    availability: bool = True
    participation_count: int = 0
    upload_bandwidth_mbps: float | None = None  # defaults to bandwidth_mbps (symmetric link)

    @property
    def upload_bandwidth(self) -> float:
        return self.upload_bandwidth_mbps if self.upload_bandwidth_mbps is not None else self.bandwidth_mbps

    def as_dict(self) -> dict:
        return asdict(self)


def generate_client_profiles(spec: dict) -> list[ClientProfile]:
    """Sample a client pool from per-tier ranges.

    ``spec`` holds ``num_clients``, ``seed``, ``tiers`` (name -> ``share`` plus ``[low, high]`` ranges for
    compute_power_watts, compute_speed_factor, bandwidth_mbps, upload_bandwidth_mbps, latency_ms) and a
    ``carbon_intensity_gco2e_kwh`` range that is drawn independently of the tier, since a device's grid does
    not depend on its hardware.
    """
    rng = np.random.default_rng(spec.get("seed", 0))
    n = spec["num_clients"]
    tiers = spec["tiers"]
    names = list(tiers)
    counts = [int(round(tiers[t]["share"] * n)) for t in names]
    counts[-1] = n - sum(counts[:-1])
    tier_of = rng.permutation(np.repeat(names, counts))
    lo, hi = spec["carbon_intensity_gco2e_kwh"]

    def draw(rng_range):
        return float(round(rng.uniform(*rng_range), 2))

    profiles = []
    for cid, tier in enumerate(tier_of):
        t = tiers[tier]
        profiles.append(ClientProfile(
            client_id=cid, tier=str(tier),
            compute_power_watts=draw(t["compute_power_watts"]),
            compute_speed_factor=draw(t["compute_speed_factor"]),
            carbon_intensity_gco2e_kwh=float(round(rng.uniform(lo, hi), 1)),
            bandwidth_mbps=draw(t["bandwidth_mbps"]),
            upload_bandwidth_mbps=draw(t["upload_bandwidth_mbps"]) if "upload_bandwidth_mbps" in t else None,
            latency_ms=draw(t["latency_ms"]),
            availability=bool(rng.random() < spec.get("availability_prob", 1.0)),
        ))
    return profiles


def load_client_profiles(path: str) -> list[ClientProfile]:
    """Either an explicit ``clients:`` list or a ``generate:`` spec (see ``generate_client_profiles``)."""
    cfg = load_yaml(path)
    if "generate" in cfg:
        return generate_client_profiles(cfg["generate"])
    return [ClientProfile(**c) for c in cfg["clients"]]
