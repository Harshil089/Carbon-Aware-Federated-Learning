import time

import pytest

from src.simulation import energy_tracker as et
from src.simulation.carbon import energy_to_carbon_g, joules_to_kwh


def test_unit_conversions():
    assert joules_to_kwh(3.6e6) == pytest.approx(1.0)
    assert energy_to_carbon_g(2.0, 500.0) == pytest.approx(1000.0)
    assert energy_to_carbon_g(0.0, 500.0) == 0.0


def test_constant_power_integrates_to_p_times_t(monkeypatch):
    """A fixed 100 W CPU estimate for ~1 s should integrate to about 100 J (+-)."""
    monkeypatch.setattr(et.EnergyTracker, "_cpu_power_estimate", lambda self: 100.0)
    t = et.EnergyTracker(475.0, sample_interval_s=0.05, track_gpu=False, use_rapl=False)
    t.start()
    time.sleep(1.0)
    t.stop()
    s = t.summary()
    assert s["cpu_energy_kwh"] * 3.6e6 == pytest.approx(100.0 * s["elapsed_s"], rel=0.05)
    assert s["gpu_energy_kwh"] == 0.0
    assert s["carbon_gco2e"] == pytest.approx(s["energy_kwh"] * 475.0)
    assert s["cpu_source"] == "tdp_model_estimate"


def test_snapshot_is_monotonic(monkeypatch):
    monkeypatch.setattr(et.EnergyTracker, "_cpu_power_estimate", lambda self: 50.0)
    with et.EnergyTracker(100.0, sample_interval_s=0.05, track_gpu=False, use_rapl=False) as t:
        a = t.snapshot()["energy_kwh"]
        time.sleep(0.2)
        b = t.snapshot()["energy_kwh"]
    assert b > a >= 0


def test_rapl_counter_wraparound(tmp_path, monkeypatch):
    dom = tmp_path / "intel-rapl:0"
    dom.mkdir()
    (dom / "max_energy_range_uj").write_text("1000000")  # wraps at 1 J
    (dom / "energy_uj").write_text("900000")
    monkeypatch.setattr(et, "_RAPL_ROOT", tmp_path)
    m = et._RaplMeter()
    assert m.available
    (dom / "energy_uj").write_text("100000")  # wrapped: 0.1 J + 0.1 J consumed
    assert m.update() == pytest.approx(0.2)
    (dom / "energy_uj").write_text("350000")
    assert m.update() == pytest.approx(0.45)
