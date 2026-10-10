"""Energy tracking for the machine running the experiment (e.g. a Colab VM).

Power is sampled by a background thread and integrated over time (trapezoid rule).
Sources, best available first, per component:

  GPU: NVML board power (``pynvml``)                    -> measured
  CPU: Intel RAPL package energy counters               -> measured
       else TDP model: tdp * (idle + (1 - idle) * util)  -> ESTIMATE

Cloud VMs usually hide RAPL, so on Colab the CPU number is normally a model-based
estimate; the GPU number comes from the driver. DRAM, disk, networking, cooling and the
embodied carbon of the hardware are NOT counted. ``summary()`` reports which source was used.
"""
import threading
import time
from pathlib import Path

from src.simulation.carbon import J_PER_KWH, energy_to_carbon_g, joules_to_kwh

try:  # optional; only present on machines with an NVIDIA driver
    import pynvml
except ImportError:  # pragma: no cover
    pynvml = None

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

_RAPL_ROOT = Path("/sys/class/powercap")


class _GpuMeter:
    def __init__(self):
        self.handles = []
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self.handles = []

    @property
    def available(self) -> bool:
        return bool(self.handles)

    def power_watts(self) -> float:
        return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000.0


class _RaplMeter:
    """Cumulative CPU package energy from powercap; handles counter wraparound."""

    def __init__(self):
        self.domains = []  # (energy_path, max_range_uj)
        try:
            for d in sorted(_RAPL_ROOT.glob("intel-rapl:[0-9]")):  # top-level packages only
                e = d / "energy_uj"
                e.read_text()  # raises PermissionError if unreadable
                self.domains.append((e, int((d / "max_energy_range_uj").read_text())))
        except (OSError, ValueError):
            self.domains = []
        self._last = self._read_raw() if self.domains else None
        self.total_j = 0.0

    @property
    def available(self) -> bool:
        return bool(self.domains)

    def _read_raw(self) -> list[int]:
        return [int(p.read_text()) for p, _ in self.domains]

    def update(self) -> float:
        """Accumulate energy since the last call; returns total joules so far."""
        now = self._read_raw()
        for (_, max_range), prev, cur in zip(self.domains, self._last, now):
            self.total_j += (cur - prev if cur >= prev else cur + max_range - prev) / 1e6
        self._last = now
        return self.total_j


class EnergyTracker:
    def __init__(
        self,
        carbon_intensity_gco2e_kwh: float,
        sample_interval_s: float = 1.0,
        cpu_tdp_watts: float = 65.0,
        cpu_idle_fraction: float = 0.3,
        track_gpu: bool = True,
        use_rapl: bool = True,
    ):
        self.intensity = carbon_intensity_gco2e_kwh
        self.interval = sample_interval_s
        self.cpu_tdp = cpu_tdp_watts
        self.cpu_idle = cpu_idle_fraction
        self.gpu = _GpuMeter()
        if not track_gpu:
            self.gpu.handles = []
        self.rapl = _RaplMeter() if use_rapl else None
        self._rapl_ok = bool(self.rapl and self.rapl.available)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_j = 0.0
        self.gpu_j = 0.0
        self._t0 = self._t_last = 0.0
        self._cpu_w_last = self._gpu_w_last = 0.0
        if psutil is not None:
            psutil.cpu_percent(None)  # prime the utilisation counter

    @property
    def cpu_source(self) -> str:
        return "rapl_measured" if self._rapl_ok else "tdp_model_estimate"

    @property
    def gpu_source(self) -> str:
        return "nvml_measured" if self.gpu.available else "none_detected"

    def _cpu_power_estimate(self) -> float:
        util = (psutil.cpu_percent(None) / 100.0) if psutil is not None else 1.0
        return self.cpu_tdp * (self.cpu_idle + (1.0 - self.cpu_idle) * util)

    def _sample(self) -> None:
        now = time.perf_counter()
        gpu_w = self.gpu.power_watts() if self.gpu.available else 0.0
        with self._lock:
            dt = now - self._t_last
            if self._rapl_ok:
                self.cpu_j = self.rapl.update()
                cpu_w = self.cpu_j / max(now - self._t0, 1e-9)
            else:
                cpu_w = self._cpu_power_estimate()
                self.cpu_j += 0.5 * (self._cpu_w_last + cpu_w) * dt
            self.gpu_j += 0.5 * (self._gpu_w_last + gpu_w) * dt
            self._cpu_w_last, self._gpu_w_last, self._t_last = cpu_w, gpu_w, now

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:  # never let a metering glitch kill the experiment
                pass

    def start(self) -> "EnergyTracker":
        self._t0 = self._t_last = time.perf_counter()
        self._cpu_w_last = self._cpu_power_estimate() if not self._rapl_ok else 0.0
        self._gpu_w_last = self.gpu.power_watts() if self.gpu.available else 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self._sample()  # close the final interval

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def snapshot(self) -> dict:
        """Cumulative totals so far; call at round boundaries and diff for per-round energy."""
        if self._thread is not None:
            self._sample()
        with self._lock:
            total_j = self.cpu_j + self.gpu_j
            kwh = joules_to_kwh(total_j)
            return {
                "elapsed_s": time.perf_counter() - self._t0,
                "cpu_energy_kwh": joules_to_kwh(self.cpu_j),
                "gpu_energy_kwh": joules_to_kwh(self.gpu_j),
                "energy_kwh": kwh,
                "carbon_gco2e": energy_to_carbon_g(kwh, self.intensity),
            }

    def summary(self) -> dict:
        s = self.snapshot()
        s.update({
            "carbon_intensity_gco2e_kwh": self.intensity,
            "avg_power_w": (s["energy_kwh"] * J_PER_KWH / s["elapsed_s"]) if s["elapsed_s"] > 0 else 0.0,
            "cpu_source": self.cpu_source,
            "gpu_source": self.gpu_source,
            "cpu_tdp_watts_assumed": None if self._rapl_ok else self.cpu_tdp,
            "scope": "whole machine CPU + GPU only; excludes DRAM, storage, network, cooling, embodied carbon",
        })
        return s

