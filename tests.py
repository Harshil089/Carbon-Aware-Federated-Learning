"""Built-in sanity checks. Run: python tests.py"""
import unittest
import numpy as np
import run_project as p

class Tests(unittest.TestCase):
    def test_measured_carbon(self):
        self.assertAlmostEqual(p.MEAS['meter_energy_kwh']*p.GRID*1000, 43.875, places=4)
    def test_power(self):
        self.assertAlmostEqual(p.AVG_W_PC1, 20.526315789, places=5)
    def test_profile_pc1_consistency(self):
        self.assertAlmostEqual(p.base_profiles()[0]['energy_kWh'],.065,places=6)
    def test_fl_gradient_updates(self):
        data,test=p.make_data(seed=42)
        weights=np.zeros((3,3))
        new=p.fedavg(weights,data,[0,1,2,3])
        self.assertFalse(np.array_equal(new,weights))
        self.assertTrue(0<=p.eval_acc(new,test)<=1)
    def test_idle_and_coordinator_included(self):
        sec,wh,carbon=p.round_model((2,3),np.ones(4))
        self.assertGreater(sec,0)
        self.assertGreater(wh,0)
        self.assertAlmostEqual(wh*p.GRID,carbon,places=8)
    def test_action_count(self):
        self.assertEqual(len(p.ACTIONS),11)
    def test_policy_valid(self):
        for pol in p.POLICIES[:-1]:
            self.assertTrue(2<=len(p.heuristic(pol))<=4)
    def test_per_client_grid_backward_compat(self):
        sec0, wh0, co2_0 = p.round_model((0,1,2,3), np.ones(4))
        sec1, wh1, co2_1 = p.round_model((0,1,2,3), np.ones(4), grid_factors=[0.675]*4)
        self.assertAlmostEqual(sec0, sec1, places=8)
        self.assertAlmostEqual(wh0, wh1, places=8)
        self.assertAlmostEqual(co2_0, wh0*p.GRID, places=8)
        self.assertAlmostEqual(co2_1, wh1*p.GRID, places=8)
    def test_per_client_grid_heterogeneous(self):
        gf = [0.2, 0.8, 0.5, 0.5]
        _, _, co2_low = p.round_model((0,2), np.ones(4), grid_factors=gf)
        _, _, co2_high = p.round_model((1,2), np.ones(4), grid_factors=gf)
        self.assertLess(co2_low, co2_high)
        _, wh_a, co2_a = p.round_model((0,2), np.ones(4), grid_factors=gf)
        _, wh_b, co2_b = p.round_model((0,2), np.ones(4), grid_factors=[0.8, 0.2, 0.5, 0.5])
        self.assertAlmostEqual(wh_a, wh_b, places=8)
        self.assertLess(co2_a, co2_b)
    def test_per_client_grid_coordinator_and_idle(self):
        sec, _, coord_only = p.round_model((0,1), np.ones(4), grid_factors=[0.0, 0.0, 0.0, 0.0])
        expected_coord = p.ASSUM['coordinator_power_w']*sec/3600.0*p.GRID
        self.assertGreater(coord_only, 0)
        self.assertAlmostEqual(coord_only, expected_coord, places=8)
        sec2, _, with_idle = p.round_model((0,1), np.ones(4), grid_factors=[0.0, 0.0, 1.0, 1.0])
        expected_idle = (p.NODES[2]['idle_power_w_assumed']+p.NODES[3]['idle_power_w_assumed'])*sec2/3600.0*1.0
        self.assertGreater(with_idle, coord_only)
        self.assertAlmostEqual(with_idle, expected_coord+expected_idle, places=8)

if __name__=='__main__':unittest.main()
