"""Pure-Python helpers for the real-hardware path. No flwr import here."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.json"

# First 11 columns mirror results/per_round_SIMULATION.csv exactly; the rest are measured extras.
SIMULATOR_COLUMNS = [
    "policy",
    "scenario_seed",
    "round",
    "clients_selected",
    "test_accuracy_REAL_MODEL_SYNTHETIC_DATA",
    "round_seconds_MODELED",
    "round_Wh_MODELED",
    "round_CO2_g_MODELED",
    "cum_seconds_MODELED",
    "cum_Wh_MODELED",
    "cum_CO2_g_MODELED",
]
MEASURED_COLUMNS = [
    "round_seconds_MEASURED",
    "round_Wh_MEASURED_CODECARBON",
    "round_Wh_MEASURED_WALLMETER",
    "round_CO2_g_MEASURED",
    "cum_seconds_MEASURED",
    "node_wall_seconds_MEASURED",
]
PER_ROUND_COLUMNS = SIMULATOR_COLUMNS + MEASURED_COLUMNS
POLICY_LABEL = "MEASURED_HARDWARE"


def write_per_round_csv(path, rows: Iterable[Mapping]) -> Path:
    """Write rows with PER_ROUND_COLUMNS order; missing keys become empty strings."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_ROUND_COLUMNS, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def build_round_row(
    round_no: int,
    seed: int,
    selected_names: Iterable[str],
    accuracy: float,
    wall_seconds: float,
    per_node_seconds: Mapping[str, float],
    codecarbon_wh: Optional[float] = None,
    wallmeter_wh: Optional[float] = None,
    cum_seconds: float = 0.0,
    grid_kgco2_per_kwh: float = 0.675,
) -> Dict[str, object]:
    """Build one CSV row. `cum_seconds` is the running total BEFORE this round.

    CO2 uses wall-meter Wh when given, else CodeCarbon Wh, else stays empty.
    """
    if wallmeter_wh is not None:
        measured_wh = float(wallmeter_wh)
    elif codecarbon_wh is not None:
        measured_wh = float(codecarbon_wh)
    else:
        measured_wh = None
    co2_g = "" if measured_wh is None else measured_wh * float(grid_kgco2_per_kwh)
    return {
        "policy": POLICY_LABEL,
        "scenario_seed": seed,
        "round": round_no,
        "clients_selected": "+".join(selected_names),
        "test_accuracy_REAL_MODEL_SYNTHETIC_DATA": accuracy,
        # Modeled columns are not produced on real hardware.
        "round_seconds_MODELED": "",
        "round_Wh_MODELED": "",
        "round_CO2_g_MODELED": "",
        "cum_seconds_MODELED": "",
        "cum_Wh_MODELED": "",
        "cum_CO2_g_MODELED": "",
        "round_seconds_MEASURED": wall_seconds,
        "round_Wh_MEASURED_CODECARBON": "" if codecarbon_wh is None else codecarbon_wh,
        "round_Wh_MEASURED_WALLMETER": "" if wallmeter_wh is None else wallmeter_wh,
        "round_CO2_g_MEASURED": co2_g,
        "cum_seconds_MEASURED": cum_seconds + wall_seconds,
        "node_wall_seconds_MEASURED": json.dumps(dict(per_node_seconds), sort_keys=True),
    }


def load_nodes_config(config_path=None) -> Tuple[Dict[str, dict], float]:
    """Return ({node_name: node_dict}, grid_kgco2_per_kwh) from config.json."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG
    cfg = json.loads(path.read_text(encoding="utf-8"))
    nodes = {str(n["name"]): dict(n) for n in cfg["nodes"]}
    grid = float(cfg.get("grid_factor_kgco2_per_kwh", 0.675))
    return nodes, grid


def parse_client_list(spec: str, config_path=None) -> List[str]:
    """Parse 'PC1,PC2,PC3' into validated node names (non-empty, unique, known)."""
    if spec is None or not str(spec).strip():
        raise ValueError("client list is empty; expected e.g. PC1,PC2,PC3")
    names = [item.strip() for item in str(spec).split(",")]
    if any(not n for n in names):
        raise ValueError(f"client list has an empty entry: {spec!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"client list has duplicate names: {spec!r}")
    known, _ = load_nodes_config(config_path)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError(f"unknown node name(s) {unknown}; config.json has {sorted(known)}")
    return names


def parse_wallmeter(spec) -> Dict[str, float]:
    """Parse 'PC1=65.0,PC2=20' into {name: Wh for the whole run}. Empty/None gives {}."""
    if spec is None or not str(spec).strip():
        return {}
    out: Dict[str, float] = {}
    for item in str(spec).split(","):
        if "=" not in item:
            raise ValueError(f"malformed wall-meter entry {item!r}; expected NAME=Wh")
        name, _, value = item.partition("=")
        name = name.strip()
        if not name:
            raise ValueError(f"wall-meter entry {item!r} has no node name")
        if name in out:
            raise ValueError(f"duplicate wall-meter entry for {name}")
        try:
            wh = float(value)
        except ValueError as exc:
            raise ValueError(f"wall-meter value for {name} is not a number: {value!r}") from exc
        if not math.isfinite(wh) or wh < 0:
            raise ValueError(f"wall-meter value for {name} must be a finite non-negative number")
        out[name] = wh
    return out


class EnergyTracker:
    """Context manager around CodeCarbon. `.wh` stays None if codecarbon is missing or fails."""

    def __init__(self) -> None:
        self.wh: Optional[float] = None
        self._tracker = None

    def __enter__(self) -> "EnergyTracker":
        try:
            from codecarbon import EmissionsTracker  # lazy: optional dependency

            self._tracker = EmissionsTracker(save_to_file=False, log_level="error")
            self._tracker.start()
        except Exception:
            self._tracker = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._tracker is not None:
            try:
                self._tracker.stop()
                data = getattr(self._tracker, "final_emissions_data", None)
                kwh = getattr(data, "energy_consumed", None)
                if kwh is None:
                    kwh = getattr(getattr(self._tracker, "_total_energy", None), "kWh", None)
                if kwh is not None:
                    self.wh = float(kwh) * 1000.0
            except Exception:
                self.wh = None
        return False
