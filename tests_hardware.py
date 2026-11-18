"""Tests for the real-hardware path. Runs WITHOUT flwr or codecarbon installed."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from hardware.common import (
    PER_ROUND_COLUMNS,
    SIMULATOR_COLUMNS,
    EnergyTracker,
    build_round_row,
    load_nodes_config,
    parse_client_list,
    parse_wallmeter,
    write_per_round_csv,
)

REPO_ROOT = Path(__file__).resolve().parent
SIM_CSV = REPO_ROOT / "results" / "per_round_SIMULATION.csv"


def _row(**overrides):
    base = dict(round_no=1, seed=42, selected_names=["PC1", "PC2"], accuracy=0.8,
                wall_seconds=2.5, per_node_seconds={"PC1": 1.0, "PC2": 2.0})
    base.update(overrides)
    return build_round_row(**base)


class WritePerRoundCsvTests(unittest.TestCase):
    def test_columns_order_and_simulator_prefix(self):
        self.assertEqual(PER_ROUND_COLUMNS[:11], SIMULATOR_COLUMNS)
        self.assertEqual(
            SIMULATOR_COLUMNS,
            ["policy", "scenario_seed", "round", "clients_selected",
             "test_accuracy_REAL_MODEL_SYNTHETIC_DATA", "round_seconds_MODELED",
             "round_Wh_MODELED", "round_CO2_g_MODELED", "cum_seconds_MODELED",
             "cum_Wh_MODELED", "cum_CO2_g_MODELED"],
        )
        if SIM_CSV.exists():  # the simulator's header must start with the simulator schema
            with SIM_CSV.open(encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh))
                self.assertEqual(header[:len(SIMULATOR_COLUMNS)], SIMULATOR_COLUMNS)

    def test_round_trip_and_cumulative_values(self):
        rows = [
            _row(round_no=1, wall_seconds=2.0, cum_seconds=0.0),
            _row(round_no=2, wall_seconds=3.0, cum_seconds=2.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write_per_round_csv(Path(tmp) / "nested" / "out.csv", rows)
            self.assertTrue(path.exists())
            with path.open(encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, PER_ROUND_COLUMNS)
                got = list(reader)
        self.assertEqual([r["round"] for r in got], ["1", "2"])
        self.assertEqual(got[0]["cum_seconds_MEASURED"], "2.0")
        self.assertEqual(got[1]["cum_seconds_MEASURED"], "5.0")
        self.assertEqual(got[0]["policy"], "MEASURED_HARDWARE")
        self.assertEqual(got[0]["round_seconds_MODELED"], "")
        self.assertEqual(json.loads(got[0]["node_wall_seconds_MEASURED"]), {"PC1": 1.0, "PC2": 2.0})

    def test_missing_keys_become_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_per_round_csv(Path(tmp) / "x.csv", [{"policy": "P", "round": 1}])
            with path.open(encoding="utf-8-sig", newline="") as fh:
                got = list(csv.DictReader(fh))[0]
        self.assertEqual(got["policy"], "P")
        self.assertEqual(got["round_Wh_MEASURED_WALLMETER"], "")


class BuildRoundRowTests(unittest.TestCase):
    def test_co2_from_wallmeter_with_grid_factor(self):
        row = _row(wallmeter_wh=100.0, grid_kgco2_per_kwh=0.675)
        self.assertAlmostEqual(row["round_CO2_g_MEASURED"], 67.5)
        self.assertEqual(row["round_Wh_MEASURED_WALLMETER"], 100.0)
        self.assertEqual(row["clients_selected"], "PC1+PC2")

    def test_wallmeter_preferred_over_codecarbon(self):
        row = _row(wallmeter_wh=10.0, codecarbon_wh=50.0, grid_kgco2_per_kwh=1.0)
        self.assertAlmostEqual(row["round_CO2_g_MEASURED"], 10.0)

    def test_codecarbon_used_when_no_wallmeter(self):
        row = _row(codecarbon_wh=50.0, grid_kgco2_per_kwh=0.5)
        self.assertAlmostEqual(row["round_CO2_g_MEASURED"], 25.0)
        self.assertEqual(row["round_Wh_MEASURED_WALLMETER"], "")

    def test_co2_empty_when_neither(self):
        row = _row()
        self.assertEqual(row["round_CO2_g_MEASURED"], "")
        self.assertEqual(row["round_Wh_MEASURED_CODECARBON"], "")
        self.assertEqual(row["round_Wh_MEASURED_WALLMETER"], "")

    def test_default_grid_factor_from_config(self):
        _, grid = load_nodes_config()
        self.assertAlmostEqual(grid, 0.675)


class ParseClientListTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_client_list("PC1,PC2,PC3"), ["PC1", "PC2", "PC3"])
        self.assertEqual(parse_client_list(" PC1 , PC4 "), ["PC1", "PC4"])

    def test_unknown_name(self):
        with self.assertRaises(ValueError):
            parse_client_list("PC1,PC9")

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            parse_client_list("PC1,PC1")

    def test_empty(self):
        for spec in ["", "   ", None, "PC1,,PC2"]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_client_list(spec)


class ParseWallmeterTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_wallmeter("PC1=65.0,PC2=20"), {"PC1": 65.0, "PC2": 20.0})

    def test_empty_gives_empty_dict(self):
        self.assertEqual(parse_wallmeter(None), {})
        self.assertEqual(parse_wallmeter(""), {})

    def test_malformed(self):
        for spec in ["PC1", "=5", "PC1=abc", "PC1=5,PC1=6"]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_wallmeter(spec)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            parse_wallmeter("PC1=-1")


class EnergyTrackerTests(unittest.TestCase):
    def test_works_without_codecarbon(self):
        with mock.patch.dict(sys.modules, {"codecarbon": None}):  # forces ImportError
            with EnergyTracker() as tracker:
                total = sum(range(10))
        self.assertEqual(total, 45)
        self.assertIsNone(tracker.wh)

    def test_converts_kwh_to_wh_when_codecarbon_present(self):
        class FakeTracker:
            def __init__(self, **kwargs):
                self.final_emissions_data = None

            def start(self):
                pass

            def stop(self):
                self.final_emissions_data = SimpleNamespace(energy_consumed=0.002)

        fake = types.ModuleType("codecarbon")
        fake.EmissionsTracker = FakeTracker
        with mock.patch.dict(sys.modules, {"codecarbon": fake}):
            with EnergyTracker() as tracker:
                pass
        self.assertAlmostEqual(tracker.wh, 2.0)

    def test_exception_inside_block_propagates(self):
        with self.assertRaises(RuntimeError):
            with EnergyTracker():
                raise RuntimeError("boom")


class ImportWithoutFlwrTests(unittest.TestCase):
    def test_modules_import_without_flwr(self):
        import importlib

        importlib.import_module("hardware.common")
        for name in ["hardware.server", "hardware.client"]:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertTrue(callable(module.main))
        self.assertNotIn("flwr", sys.modules)


if __name__ == "__main__":
    unittest.main()
