#!/usr/bin/env python3
"""
scripts/generate_telemetry.py

Generates a reproducible dataset of 10,000 unique calibrated synthetic telemetry rows
grounded in the user's physical PC1 meter reading and the PC1-PC4 scenario specs from config.json.

Provenance: SYNTHETIC_CALIBRATED_FROM_PC1_METER
"""

import csv
import os
import numpy as np

# Seed for complete reproducibility
SEED = 42

# PC1 Meter Ground Truth
# 0.065 kWh over 190 min = (0.065 * 1000) / (190 / 60) ≈ 20.526315789473685 W
PC1_METER_KWH = 0.065
PC1_METER_MINUTES = 190.0
PC1_MEAN_ACTIVE_POWER_W = (PC1_METER_KWH * 1000.0) / (PC1_METER_MINUTES / 60.0)

GRID_FACTOR_KGCO2_PER_KWH = 0.675
PROVENANCE_TAG = "SYNTHETIC_CALIBRATED_FROM_PC1_METER"

# Node specifications matching config.json & prompt
NODE_SPECS = {
    "PC1": {
        "ram_gb": 8,
        "gpu_vram_gb": 0,
        "relative_speed": 1.00,
        "active_power_w": PC1_MEAN_ACTIVE_POWER_W * 1.0,  # ~20.526 W
        "idle_power_w": 7.0,
        "network_s": 0.30,
    },
    "PC2": {
        "ram_gb": 16,
        "gpu_vram_gb": 0,
        "relative_speed": 1.10,
        "active_power_w": 86.22,  # Physical meter: 0.276 kWh / 192.075 min
        "idle_power_w": 9.0,
        "network_s": 0.45,
    },
    "PC3": {
        "ram_gb": 32,
        "gpu_vram_gb": 8,
        "relative_speed": 1.54,  # 39.66% faster than PC2 (2.75s vs 3.84s per round)
        "active_power_w": 209.4,  # Physical meter: 0.480 kWh / 137.528 min
        "idle_power_w": 13.0,
        "network_s": 0.60,
    },
    "PC4": {
        "ram_gb": 16,
        "gpu_vram_gb": 4,
        "relative_speed": 1.35,
        "active_power_w": PC1_MEAN_ACTIVE_POWER_W * 1.75,  # ~35.921 W
        "idle_power_w": 11.0,
        "network_s": 0.45,
    },
}

COLUMNS = [
    "sample_id",
    "node",
    "ram_gb",
    "gpu_vram_gb",
    "duration_min",
    "active_power_w",
    "idle_power_w",
    "energy_kwh",
    "grid_kgco2_per_kwh",
    "co2_g",
    "relative_speed",
    "network_s",
    "bandwidth_mbps",
    "availability",
    "provenance",
]


def generate_dataset(output_path: str, rows_per_node: int = 2500):
    rng = np.random.default_rng(SEED)

    records = []
    global_sample_id = 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for node_name, spec in NODE_SPECS.items():
        base_active_p = spec["active_power_w"]
        base_idle_p = spec["idle_power_w"]
        base_speed = spec["relative_speed"]
        base_net_s = spec["network_s"]
        ram_gb = spec["ram_gb"]
        gpu_vram_gb = spec["gpu_vram_gb"]

        # 1. duration_min: lognormal centered around 15 minutes, bounded between 1.0 and 190.0 min
        # PC1 anchor test was 190 minutes, typical FL task rounds or subtasks vary from 1 min to a few hours
        durations = rng.lognormal(mean=np.log(15.0), sigma=0.5, size=rows_per_node)
        durations = np.clip(durations, 1.0, 190.0)

        # 2. active_power_w: Gaussian variation around base_active_p with 8% std dev, clipped to realistic hardware envelope
        active_power = rng.normal(loc=base_active_p, scale=base_active_p * 0.08, size=rows_per_node)
        active_power = np.clip(active_power, base_idle_p + 1.0, base_active_p * 1.8)

        # 3. idle_power_w: slight variation around base idle (std dev ~0.4W)
        idle_power = rng.normal(loc=base_idle_p, scale=0.4, size=rows_per_node)
        idle_power = np.clip(idle_power, base_idle_p * 0.75, base_idle_p * 1.35)

        # 4. relative_speed: variation around base_speed with ~5% std dev, clipped
        speed = rng.normal(loc=base_speed, scale=base_speed * 0.05, size=rows_per_node)
        speed = np.clip(speed, base_speed * 0.7, base_speed * 1.3)

        # 5. network_s: network latency/transmission per round (lognormal ~0.15 sigma)
        net_s = rng.lognormal(mean=np.log(base_net_s), sigma=0.15, size=rows_per_node)
        net_s = np.clip(net_s, base_net_s * 0.5, base_net_s * 2.5)

        # 6. bandwidth_mbps: realistic client bandwidths (e.g. 50 - 500 Mbps)
        # Higher-spec or lab nodes might have slightly higher base bandwidth
        bw_base = 100.0 if node_name in ["PC1", "PC2"] else 250.0
        bandwidth = rng.normal(loc=bw_base, scale=bw_base * 0.2, size=rows_per_node)
        bandwidth = np.clip(bandwidth, 15.0, 950.0)

        # 7. availability: probability of node availability / uptime fraction in [0.85, 1.00]
        # Beta distribution skewed towards high availability
        avail = 0.80 + 0.20 * rng.beta(a=9.0, b=1.5, size=rows_per_node)
        avail = np.clip(avail, 0.80, 1.00)

        # 8. grid_kgco2_per_kwh: Anchor is 0.675 (CEA Indian grid baseline), with slight regional/hourly diurnal fluctuation (std dev 0.015)
        grid_co2 = rng.normal(loc=GRID_FACTOR_KGCO2_PER_KWH, scale=0.012, size=rows_per_node)
        grid_co2 = np.clip(grid_co2, 0.600, 0.750)

        for i in range(rows_per_node):
            dur_m = round(float(durations[i]), 4)
            p_active = round(float(active_power[i]), 4)
            p_idle = round(float(idle_power[i]), 4)

            # Energy in kWh: (active_power in W * duration in hours) / 1000
            # duration in hours = dur_m / 60.0
            energy = (p_active * (dur_m / 60.0)) / 1000.0
            energy = round(energy, 6)

            g_co2_factor = round(float(grid_co2[i]), 5)
            # co2_g = energy_kwh * grid_factor (kg/kWh) * 1000 (g/kg)
            co2_emissions_g = round(energy * g_co2_factor * 1000.0, 4)

            rel_spd = round(float(speed[i]), 4)
            n_s = round(float(net_s[i]), 4)
            bw = round(float(bandwidth[i]), 2)
            av = round(float(avail[i]), 4)

            row = {
                "sample_id": global_sample_id,
                "node": node_name,
                "ram_gb": ram_gb,
                "gpu_vram_gb": gpu_vram_gb,
                "duration_min": dur_m,
                "active_power_w": p_active,
                "idle_power_w": p_idle,
                "energy_kwh": energy,
                "grid_kgco2_per_kwh": g_co2_factor,
                "co2_g": co2_emissions_g,
                "relative_speed": rel_spd,
                "network_s": n_s,
                "bandwidth_mbps": bw,
                "availability": av,
                "provenance": PROVENANCE_TAG,
            }
            records.append(row)
            global_sample_id += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} rows written to {output_path}")


if __name__ == "__main__":
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "calibrated_telemetry_10k.csv")
    generate_dataset(out_csv)
