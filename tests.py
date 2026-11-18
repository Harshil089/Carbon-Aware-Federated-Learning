"""Built-in sanity checks. Run: python tests.py"""
import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_project as p
from fl_pipeline.clients import default_clients_from_config
from fl_pipeline.network_energy import simulate_round_telemetry

CFG = SimpleNamespace(rounds=20, local_epochs=2, learning_rate=.23, latency_weight=.018, carbon_weight=.32)


def _telemetry(selected, grids=None):
    return simulate_round_telemetry(default_clients_from_config(), selected,
        base_train_seconds=p.ASSUM['synthetic_pc1_train_seconds_per_round'],
        aggregation_seconds=p.ASSUM['aggregation_seconds_per_round'],
        coordinator_power_w=p.ASSUM['coordinator_power_w'], grid_overrides=grids, allow_empty=True)


class Tests(unittest.TestCase):
    def test_measured_carbon(self):
        self.assertAlmostEqual(p.MEAS['meter_energy_kwh']*p.GRID*1000, 43.875, places=4)
    def test_power(self):
        self.assertAlmostEqual(p.AVG_W_PC1, 20.526315789, places=5)
    def test_profile_pc1_consistency(self):
        self.assertAlmostEqual(p.base_profiles()[0]['energy_kWh'], .065, places=6)
    def test_provenance_from_config(self):
        prof = p.base_profiles()
        self.assertEqual(prof[0]['provenance'], 'MEASURED: 0.065 kWh over 190 minutes; CO2 CALCULATED')
        self.assertTrue(prof[1]['provenance'].startswith('MEASURED BY USER'))
        self.assertEqual(prof[-1]['provenance'], 'SYNTHETIC SCENARIO — NOT MEASURED')
    def test_make_data_split(self):
        parts, val, test = p.make_data(42)
        self.assertEqual(len(parts), len(p.NODES))
        self.assertEqual(parts[0][0].shape, (250, 3))
        self.assertEqual(val[0].shape, (500, 3))
        self.assertEqual(test[0].shape, (1500, 3))
    def test_make_data_matches_original_generation(self):
        # Reference: the original script's generation order (client parts, then test).
        rng = np.random.default_rng(42)
        cent = np.array([[-1.8,-1.3],[1.7,-.9],[0.1,1.8]])
        mixes = ([.80,.10,.10],[.10,.80,.10],[.10,.10,.80],[.34,.33,.33])
        ref_parts = []
        for mix in mixes:
            y = rng.choice(3, size=250, p=mix)
            ref_parts.append((np.column_stack([cent[y]+rng.normal(0,1.3,size=(250,2)), np.ones(250)]), y))
        ty = np.repeat(np.arange(3), 500); rng.shuffle(ty)
        tx = np.column_stack([cent[ty]+rng.normal(0,1.3,size=(1500,2)), np.ones(1500)])
        parts, _, test = p.make_data(42)
        for (rx, ry), (px, py) in zip(ref_parts, parts):
            np.testing.assert_array_equal(rx, px); np.testing.assert_array_equal(ry, py)
        np.testing.assert_array_equal(test[0], tx); np.testing.assert_array_equal(test[1], ty)
    def test_make_data_deterministic_per_seed(self):
        a = p.make_data(1)[0][0][0]; b = p.make_data(1)[0][0][0]; c = p.make_data(2)[0][0][0]
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
    def test_fl_gradient_updates(self):
        env = p.build_env(CFG, 42); env.reset()
        w0 = env.weights.copy()
        _, _, _, info = env.step((0, 1, 2, 3))
        self.assertFalse(np.array_equal(env.weights, w0))
        self.assertTrue(0 <= info['accuracy'] <= 1)
    def test_idle_and_coordinator_included(self):
        env = p.build_env(CFG, 42); env.reset()
        _, _, _, info = env.step((2, 3))
        self.assertGreater(info['round_wh'], 0)
        self.assertAlmostEqual(info['round_co2_g'], info['round_wh']*p.GRID, places=6)
    def test_action_count(self):
        self.assertEqual(len(p.build_env(CFG, 42).actions), 11)
    def test_policy_valid(self):
        clients = default_clients_from_config()
        for pol in p.POLICIES[:-1]:
            self.assertTrue(2 <= len(p.heuristic(pol, clients)) <= 4)
        self.assertEqual(p.heuristic('Fastest2', clients), (2, 3))
    def test_per_client_grid_backward_compat(self):
        t0 = _telemetry((0,1,2,3))
        t1 = _telemetry((0,1,2,3), grids=[p.GRID]*4)
        self.assertAlmostEqual(t0.round_latency_s, t1.round_latency_s, places=8)
        self.assertAlmostEqual(t0.total_wh, t1.total_wh, places=8)
        self.assertAlmostEqual(t0.total_co2_g, t0.total_wh*p.GRID, places=8)
        self.assertAlmostEqual(t1.total_co2_g, t1.total_wh*p.GRID, places=8)
    def test_per_client_grid_heterogeneous(self):
        gf = [0.2, 0.8, 0.5, 0.5]
        self.assertLess(_telemetry((0,2), gf).total_co2_g, _telemetry((1,2), gf).total_co2_g)
        a = _telemetry((0,2), gf); b = _telemetry((0,2), [0.8, 0.2, 0.5, 0.5])
        self.assertAlmostEqual(a.total_wh, b.total_wh, places=8)
        self.assertLess(a.total_co2_g, b.total_co2_g)
    def test_per_client_grid_coordinator_and_idle(self):
        t = _telemetry((0,1), grids=[0.0, 0.0, 0.0, 0.0])
        expected_coord = p.ASSUM['coordinator_power_w']*t.round_latency_s/3600.0*p.GRID
        self.assertGreater(t.total_co2_g, 0)
        self.assertAlmostEqual(t.total_co2_g, expected_coord, places=8)
        t2 = _telemetry((0,1), grids=[0.0, 0.0, 1.0, 1.0])
        expected_idle = (p.NODES[2]['idle_power_w_assumed']+p.NODES[3]['idle_power_w_assumed'])*t2.round_latency_s/3600.0
        self.assertAlmostEqual(t2.total_co2_g, expected_coord+expected_idle, places=8)
    def test_end_to_end_small_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run([sys.executable, str(p.BASE/'run_project.py'), '--rounds', '3', '--episodes', '4',
                                  '--repeats', '3', '--sensitivity', '2', '--output', folder],
                                 cwd=p.BASE, capture_output=True, text=True, timeout=300)
            self.assertEqual(run.returncode, 0, run.stderr)
            out = Path(folder)
            with (out/'per_round_SIMULATION.csv').open(encoding='utf-8-sig') as f: rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3*3*4)
            fedavg_final = {r['test_accuracy_REAL_MODEL_SYNTHETIC_DATA'] for r in rows
                            if r['policy'] == 'FedAvg_all4' and r['round'] == '3'}
            self.assertGreater(len(fedavg_final), 1)
            with (out/'comparison_SIMULATION.csv').open(encoding='utf-8-sig') as f:
                self.assertIn('final_accuracy_ci95_low', csv.DictReader(f).fieldnames)
            params = json.loads((out/'run_parameters.json').read_text(encoding='utf-8'))
            self.assertEqual(params['paired_rl_vs_fedavg_all4']['n_pairs'], 3)
            self.assertTrue((out/'dashboard.html').exists())


if __name__ == '__main__':
    unittest.main()
