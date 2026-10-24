from pathlib import Path

import pytest
import torch

from src.models import build_model, count_parameters
from src.simulation.client_profiles import ClientProfile, generate_client_profiles, load_client_profiles
from src.simulation.system_model import SystemModel

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _profile(cid, speed=1.0, bw=10.0, up=None, lat=0.0, power=10.0, intensity=500.0):
    return ClientProfile(client_id=cid, tier="t", compute_power_watts=power, compute_speed_factor=speed,
                         carbon_intensity_gco2e_kwh=intensity, bandwidth_mbps=bw, latency_ms=lat,
                         upload_bandwidth_mbps=up)


def test_client_cost_matches_formulas():
    sm = SystemModel(base_samples_per_second=100.0, comm_power_watts=2.0)
    p = _profile(0, speed=2.0, bw=10.0, up=5.0, lat=50.0, power=36.0, intensity=1000.0)
    c = sm.client_cost(p, num_samples=400, epochs=3, model_bits=10e6)
    assert c["download_s"] == pytest.approx(1.0 + 0.05)
    assert c["upload_s"] == pytest.approx(2.0 + 0.05)
    assert c["train_s"] == pytest.approx(400 * 3 / (100.0 * 2.0))
    joules = 36.0 * 6.0 + 2.0 * (1.05 + 2.05)
    assert c["energy_kwh"] == pytest.approx(joules / 3.6e6)
    assert c["carbon_gco2e"] == pytest.approx(joules / 3.6e6 * 1000.0)


def test_round_latency_is_slowest_client_not_sum():
    sm = SystemModel(base_samples_per_second=100.0, server_power_watts=0.0)
    costs = {0: sm.client_cost(_profile(0, speed=1.0), 100, 1, 0.0),
             1: sm.client_cost(_profile(1, speed=0.25), 100, 1, 0.0)}
    r = sm.round_cost(costs, aggregation_time_s=0.5)
    assert r["straggler_client_id"] == 1
    assert r["latency_s"] == pytest.approx(4.0 + 0.5)
    assert r["energy_kwh"] == pytest.approx(costs[0]["energy_kwh"] + costs[1]["energy_kwh"])


def test_carbon_uses_each_clients_grid():
    sm = SystemModel()
    clean = sm.client_cost(_profile(0, intensity=50.0), 1000, 1, 1e6)
    dirty = sm.client_cost(_profile(1, intensity=800.0), 1000, 1, 1e6)
    assert clean["energy_kwh"] == pytest.approx(dirty["energy_kwh"])
    assert dirty["carbon_gco2e"] == pytest.approx(16 * clean["carbon_gco2e"])


def test_generated_pool_is_deterministic_and_in_range():
    spec = {"num_clients": 50, "seed": 0, "carbon_intensity_gco2e_kwh": [30, 800], "tiers": {
        "low": {"share": 0.4, "compute_power_watts": [5, 20], "compute_speed_factor": [0.2, 0.6],
                "bandwidth_mbps": [2, 20], "upload_bandwidth_mbps": [0.5, 5], "latency_ms": [50, 150]},
        "high": {"share": 0.6, "compute_power_watts": [80, 150], "compute_speed_factor": [1.5, 3.0],
                 "bandwidth_mbps": [100, 500], "latency_ms": [5, 20]},
    }}
    a, b = generate_client_profiles(spec), generate_client_profiles(spec)
    assert a == b
    assert [p.client_id for p in a] == list(range(50))
    assert sum(p.tier == "low" for p in a) == 20
    for p in a:
        assert 30 <= p.carbon_intensity_gco2e_kwh <= 800
        if p.tier == "low":
            assert 0.2 <= p.compute_speed_factor <= 0.6 and 0.5 <= p.upload_bandwidth <= 5
        else:
            assert p.upload_bandwidth == p.bandwidth_mbps  # no upload range => symmetric link


def test_repo_client_configs_load():
    assert len(load_client_profiles(str(CONFIGS / "clients.yaml"))) == 5
    pool = load_client_profiles(str(CONFIGS / "clients_pool50.yaml"))
    assert len(pool) == 50 and len({p.tier for p in pool}) == 3


def test_cifar_cnn_shapes():
    model = build_model("cifar_cnn")
    assert model(torch.zeros(4, 3, 32, 32)).shape == (4, 10)
    assert 1.0e6 < count_parameters(model) < 1.2e6
    assert count_parameters(build_model("cnn")) == 215370  # MNIST model unchanged
