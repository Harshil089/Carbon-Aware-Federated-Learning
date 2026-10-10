from dataclasses import asdict, dataclass

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

    def as_dict(self) -> dict:
        return asdict(self)


def load_client_profiles(path: str) -> list[ClientProfile]:
    return [ClientProfile(**c) for c in load_yaml(path)["clients"]]
