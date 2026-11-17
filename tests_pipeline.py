"""Comprehensive unit and integration test suite for the 8-layer fl_pipeline package.

Run: .venv/bin/python tests_pipeline.py
"""
from __future__ import annotations

import unittest
import numpy as np

from fl_pipeline import (
    FairnessGiniScheduler,
    FastestClientScheduler,
    FederatedClient,
    FederatedRLEnv,
    LowestCarbonScheduler,
    MORLSweepPoint,
    RandomScheduler,
    RoundRobinScheduler,
    TabularQLearningScheduler,
    WeightedHeuristicScheduler,
    build_clients_from_telemetry,
    class_distribution,
    compute_aggregation_weights,
    compute_gini,
    default_clients_from_config,
    evaluate_accuracy,
    extract_pareto_frontier,
    fedavg_aggregate,
    load_calibrated_telemetry_csv,
    make_federated_dataset,
    morl_scalarization_sweep,
    partition_dirichlet,
    partition_iid,
    run_fedavg_round,
    run_pipeline_experiment,
    simulate_round_telemetry,
    softmax_probs,
)
from fl_pipeline.multiseed import compute_hypervolume_2d, run_multiseed_experiment
from fl_pipeline.torch_fl import (
    DQNScheduler,
    MNISTCNN,
    MNISTMLP,
    eval_torch,
    fedavg_torch,
    load_mnist_partitions,
    train_client_torch,
)


class TestPipelineLayers(unittest.TestCase):
    """Unit and integration tests across all 8 layers of fl_pipeline."""

    # --- Layer 1: Data Partitioning ---
    def test_layer1_iid_and_dirichlet_partitions(self) -> None:
        parts_iid, test_iid = make_federated_dataset(
            n_clients=4, mode="iid", n_train=800, n_test=400, seed=42
        )
        self.assertEqual(len(parts_iid), 4)
        self.assertEqual(sum(len(y) for _, y in parts_iid), 800)
        self.assertTrue(all(len(y) == 200 for _, y in parts_iid))
        self.assertEqual(len(test_iid[1]), 400)

        # Dirichlet non-IID: reproducibility, no empty partitions, sample conservation
        parts_d1, _ = make_federated_dataset(
            n_clients=4, mode="dirichlet", alpha=0.2, n_train=800, min_samples_per_client=20, seed=42
        )
        parts_d2, _ = make_federated_dataset(
            n_clients=4, mode="dirichlet", alpha=0.2, n_train=800, min_samples_per_client=20, seed=42
        )
        self.assertEqual(sum(len(y) for _, y in parts_d1), 800)
        for (x1, y1), (x2, y2) in zip(parts_d1, parts_d2):
            self.assertGreaterEqual(len(y1), 20)
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)

        # Lower alpha should create higher label skew than high alpha
        parts_high_alpha, _ = make_federated_dataset(
            n_clients=4, mode="dirichlet", alpha=50.0, n_train=800, seed=42
        )
        dist_low = class_distribution(parts_d1)
        dist_high = class_distribution(parts_high_alpha)
        prop_low = dist_low / dist_low.sum(axis=1, keepdims=True)
        prop_high = dist_high / dist_high.sum(axis=1, keepdims=True)
        self.assertGreater(float(np.std(prop_low)), float(np.std(prop_high)))

    # --- Layer 2: Client Profiles & Local Softmax Training ---
    def test_layer2_clients_and_local_training(self) -> None:
        parts, _ = make_federated_dataset(n_clients=4, mode="iid", n_train=400, seed=42)
        clients = default_clients_from_config(partitions=parts)
        self.assertEqual(len(clients), 4)
        self.assertAlmostEqual(clients[0].active_power_w, 20.5263158, places=5)
        self.assertEqual(clients[0].ram_gb, 8)
        self.assertEqual(clients[2].gpu_vram_gb, 8)

        probs = softmax_probs(np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-8)

        w0 = np.zeros((3, 3), dtype=float)
        res = clients[0].train_local(w0, epochs=4, lr=0.25)
        self.assertEqual(res.num_samples, len(parts[0][1]))
        self.assertLess(res.final_loss, res.initial_loss)
        self.assertFalse(np.allclose(res.weights, w0))

    # --- Layer 3: Parallel Latency, Energy & Per-Client Carbon ---
    def test_layer3_parallel_latency_and_per_client_carbon(self) -> None:
        clients = default_clients_from_config()
        # Select PC1 (0) and PC3 (2) with zero payload and unit multipliers
        tel = simulate_round_telemetry(
            clients,
            selected_ids=(0, 2),
            base_train_seconds=16.0,
            aggregation_seconds=0.45,
            coordinator_power_w=30.0,
            payload_mb=0.0,
        )
        # PC1 total = 16.0/1.0 + 0.30 = 16.30s; PC3 total = 16.0/1.8 + 0.60 = 9.4889s
        expected_round_s = max(16.0 / 1.0 + 0.30, 16.0 / 1.8 + 0.60) + 0.45
        self.assertAlmostEqual(tel.round_latency_s, expected_round_s, places=6)
        self.assertAlmostEqual(
            tel.total_wh, tel.coordinator_wh + tel.total_active_wh + tel.total_idle_wh, places=8
        )
        # Unselected client PC2 (1) has 0 active_wh and idle_s == round_latency_s
        self.assertAlmostEqual(tel.per_client[1].active_wh, 0.0, places=8)
        self.assertAlmostEqual(tel.per_client[1].idle_s, expected_round_s, places=8)

        # Per-client carbon intensity test: E_kWh * CI * 1000 == co2_g
        tel_low = simulate_round_telemetry(
            clients, selected_ids=(0, 2), grid_overrides=[0.15, 0.5, 0.20, 0.5]
        )
        tel_high = simulate_round_telemetry(
            clients, selected_ids=(0, 2), grid_overrides=[0.90, 0.5, 0.85, 0.5]
        )
        self.assertLess(tel_low.total_co2_g, tel_high.total_co2_g)
        for cid, c_tel in tel_low.per_client.items():
            self.assertAlmostEqual(
                c_tel.total_kwh * c_tel.grid_kgco2_per_kwh * 1000.0, c_tel.co2_g, places=8
            )

    # --- Layer 4: Sample-Weighted FedAvg Aggregation ---
    def test_layer4_fedavg_weighted_aggregation(self) -> None:
        w = compute_aggregation_weights([100, 300, 600])
        np.testing.assert_allclose(w, np.array([0.1, 0.3, 0.6]), atol=1e-9)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=9)

        m1 = np.ones((3, 3)) * 2.0
        m2 = np.ones((3, 3)) * 6.0
        agg, norm_w = fedavg_aggregate([m1, m2], [100, 300])
        np.testing.assert_allclose(norm_w, np.array([0.25, 0.75]), atol=1e-9)
        np.testing.assert_allclose(agg, np.ones((3, 3)) * 5.0, atol=1e-9)

        parts, test_data = make_federated_dataset(n_clients=4, mode="iid", seed=42)
        clients = default_clients_from_config(partitions=parts)
        w0 = np.zeros((3, 3), dtype=float)
        w1, agg_w, results = run_fedavg_round(w0, clients, selected_ids=(0, 1, 2, 3), epochs=3)
        self.assertAlmostEqual(float(agg_w.sum()), 1.0, places=9)
        self.assertEqual(len(results), 4)
        self.assertGreater(evaluate_accuracy(w1, test_data), evaluate_accuracy(w0, test_data))

    # --- Layer 5: Schedulers & MORL Sweep ---
    def test_layer5_all_schedulers_and_pareto(self) -> None:
        parts, test_data = make_federated_dataset(n_clients=4, mode="dirichlet", alpha=0.5, seed=42)
        clients = default_clients_from_config(partitions=parts)

        fastest = FastestClientScheduler(k=2)
        # PC3 (idx 2, speed 1.8) and PC4 (idx 3, speed 1.35) are the fastest
        self.assertEqual(fastest.select(clients), (2, 3))

        lowest_co2 = LowestCarbonScheduler(k=2)
        # PC1 (idx 0, 20.53W) and PC4 (idx 3, 35.92W) have the lowest active energy/CO2 per round
        self.assertEqual(lowest_co2.select(clients), (0, 3))

        rr = RoundRobinScheduler(k=2)
        self.assertEqual(rr.select(clients, round_idx=0), (0, 1))
        self.assertEqual(rr.select(clients, round_idx=1), (2, 3))

        rnd = RandomScheduler(k=2, seed=42)
        sel_rnd = rnd.select(clients, round_idx=0)
        self.assertEqual(len(sel_rnd), 2)

        wh = WeightedHeuristicScheduler(k=2)
        self.assertEqual(len(wh.select(clients)), 2)

        fg = FairnessGiniScheduler(k=2)
        counts = np.zeros(4, dtype=int)
        for r in range(8):
            chosen = fg.select(clients, round_idx=r, participation_counts=counts)
            for i in chosen:
                counts[i] += 1
        self.assertLess(compute_gini(counts), 0.15)

        pts = [
            MORLSweepPoint("Dom", 1, 1, 1, 0, final_accuracy=0.70, total_latency_s=200, total_wh=5, total_co2_g=4.0, participation_gini=0.2),
            MORLSweepPoint("OptA", 1, 1, 1, 0, final_accuracy=0.82, total_latency_s=180, total_wh=4, total_co2_g=3.5, participation_gini=0.1),
            MORLSweepPoint("OptB", 1, 1, 1, 0, final_accuracy=0.76, total_latency_s=150, total_wh=3, total_co2_g=2.8, participation_gini=0.1),
        ]
        frontier = extract_pareto_frontier(pts)
        self.assertEqual([p.preference_name for p in frontier], ["OptA", "OptB"])

    # --- Layer 6: FederatedRLEnv ---
    def test_layer6_rl_environment_and_constraints(self) -> None:
        parts, test_data = make_federated_dataset(n_clients=4, mode="dirichlet", alpha=0.5, seed=42)
        clients = default_clients_from_config(partitions=parts)
        env = FederatedRLEnv(
            clients=clients,
            test_data=test_data,
            max_rounds=5,
            latency_budget_s=20.0,
            carbon_budget_g=0.5,
            seed=42,
        )
        obs = env.reset()
        self.assertEqual(obs.shape, (3 + 3 * 4,))
        self.assertEqual(len(env.actions), 11)

        with self.assertRaises(ValueError):
            env.validate_action(99)
        with self.assertRaises(ValueError):
            env.validate_action((0, 0))

        next_obs, rew, done, info = env.step((0, 2))
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(rew, float)
        self.assertFalse(done)
        self.assertEqual(info["vector_reward"].shape, (4,))
        self.assertIn("latency_budget_exceeded", info["constraints"])

        # Step until done and verify TabularQLearning trains cleanly
        q_sched = TabularQLearningScheduler(env.actions, seed=42)
        q_sched.train(env, episodes=10)
        self.assertGreater(len(q_sched.q_table), 0)

    # --- Layer 7 & 8: Calibrated Telemetry CSV + End-to-End Experiment ---
    def test_layer7_8_calibrated_telemetry_and_end_to_end_runner(self) -> None:
        rows, summary = load_calibrated_telemetry_csv()
        self.assertEqual(len(rows), 10000)
        self.assertEqual(set(summary.keys()), {"PC1", "PC2", "PC3", "PC4"})

        cal_clients = build_clients_from_telemetry()
        self.assertEqual(len(cal_clients), 4)
        self.assertEqual(cal_clients[0].provenance, "SYNTHETIC_CALIBRATED_FROM_PC1_METER")
        self.assertAlmostEqual(cal_clients[0].active_power_w, 20.526, delta=0.25)

        exp = run_pipeline_experiment(
            mode="dirichlet",
            alpha=0.5,
            rounds=6,
            q_episodes=12,
            morl_episodes=8,
            use_calibrated_telemetry=True,
            seed=42,
        )
        self.assertEqual(len(exp["policies"]), 7)
        self.assertGreaterEqual(len(exp["pareto_policies"]), 1)
        self.assertEqual(len(exp["morl_sweep"]), 5)
        self.assertGreaterEqual(len(exp["pareto_morl"]), 1)

    # --- PyTorch MNIST Models & FedAvg Aggregation ---
    def test_torch_mnist_models_and_fedavg_weights(self) -> None:
        partitions, (x_test, y_test), provenance = load_mnist_partitions(
            n_clients=4,
            mode="dirichlet",
            alpha=0.5,
            seed=42,
            max_train_samples=320,
            max_test_samples=80,
        )
        self.assertEqual(len(partitions), 4)
        self.assertTrue(all(len(yi) > 0 for _, yi in partitions))
        self.assertIn(provenance, ("TORCHVISION_MNIST", "SYNTHETIC_28x28_FALLBACK"))

        for model_cls in (MNISTMLP, MNISTCNN):
            model = model_cls()
            logits = model(x_test[:8])
            self.assertEqual(tuple(logits.shape), (8, 10))

            client_states = []
            sample_counts = []
            for xi, yi in partitions[:3]:
                state = train_client_torch(model, xi, yi, epochs=1, lr=0.05)
                client_states.append(state)
                sample_counts.append(len(yi))

            agg_state, norm_weights = fedavg_torch(client_states, sample_counts)
            self.assertAlmostEqual(float(np.sum(norm_weights)), 1.0, places=7)
            expected_w = np.asarray(sample_counts, dtype=float) / float(sum(sample_counts))
            np.testing.assert_allclose(norm_weights, expected_w, atol=1e-7)

            model.load_state_dict(agg_state)
            acc = eval_torch(model, x_test, y_test)
            self.assertGreaterEqual(float(acc), 0.0)
            self.assertLessEqual(float(acc), 1.0)

    # --- PyTorch DQN Scheduler Action Validity & Training Step ---
    def test_dqn_scheduler_action_validity_and_train_step(self) -> None:
        parts, test_data = make_federated_dataset(n_clients=4, mode="dirichlet", alpha=0.5, seed=42)
        clients = default_clients_from_config(partitions=parts)
        env = FederatedRLEnv(clients=clients, test_data=test_data, max_rounds=5, seed=42)
        obs = env.reset(seed=42)

        dqn = DQNScheduler(obs_dim=15, actions=env.actions, seed=42, batch_size=4)
        for eps in (0.0, 0.5, 1.0):
            act = dqn.select(obs, epsilon=eps)
            idx, validated = env.validate_action(act)
            self.assertGreaterEqual(idx, 0)
            self.assertEqual(act, validated)

        act_from_clients = dqn.select(
            env.clients,
            round_idx=0,
            participation_counts=env.participation_counts,
            acc=env.current_acc,
            max_rounds=env.max_rounds,
        )
        self.assertIn(act_from_clients, env.actions)

        next_obs, reward, done, _ = env.step(act_from_clients)
        loss = dqn.train_step(obs, act_from_clients, reward, next_obs, done)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(len(dqn.replay_buffer), 1)
        self.assertEqual(dqn.train_steps, 1)

        ep_losses = dqn.train(env, episodes=3)
        self.assertEqual(len(ep_losses), 3)
        self.assertTrue(all(np.isfinite(l) and l >= 0.0 for l in ep_losses))

    # --- 2D Pareto Hypervolume Dominance Sanity ---
    def test_compute_hypervolume_2d_dominance_sanity(self) -> None:
        ref = (5.0, 0.20)
        self.assertEqual(compute_hypervolume_2d([], ref), 0.0)
        self.assertEqual(compute_hypervolume_2d([(5.5, 0.80), (3.0, 0.10)], ref), 0.0)

        # Single point: (co2=2.0, acc=0.80) -> (5.0 - 2.0) * (0.80 - 0.20) = 1.80
        hv_single = compute_hypervolume_2d([(2.0, 0.80)], ref)
        self.assertAlmostEqual(hv_single, 1.80, places=8)

        # Adding a dominated point (higher CO2=3.5, lower acc=0.65) must not change hypervolume
        hv_with_dominated = compute_hypervolume_2d([(2.0, 0.80), (3.5, 0.65)], ref)
        self.assertAlmostEqual(hv_with_dominated, hv_single, places=8)

        # Adding a non-dominated tradeoff point (higher CO2=3.0, higher acc=0.90) increases HV
        hv_tradeoff = compute_hypervolume_2d([(2.0, 0.80), (3.0, 0.90)], ref)
        self.assertAlmostEqual(hv_tradeoff, 1.80 + (5.0 - 3.0) * (0.90 - 0.80), places=8)
        self.assertGreater(hv_tradeoff, hv_single)

        # Strictly dominating point (lower CO2=1.5, higher acc=0.85) yields strictly greater HV
        hv_dominating = compute_hypervolume_2d([(1.5, 0.85)], ref)
        self.assertGreater(hv_dominating, hv_single)

    # --- Multi-Seed Experiment Mean/Std & Hypervolume Output ---
    def test_run_multiseed_experiment_mean_std_output(self) -> None:
        res = run_multiseed_experiment(
            seeds=(42, 43, 44),
            mode="dirichlet",
            alpha=0.5,
            rounds=6,
            q_episodes=8,
            include_dqn=True,
        )
        self.assertEqual(len(res["policies"]), 8)
        self.assertIn("DQN", res["policies"])
        self.assertEqual(len(res["ref_point"]), 2)
        self.assertGreater(res["hypervolume"], 0.0)
        self.assertEqual(len(res["per_seed_hypervolumes"]), 3)

        metrics = ("final_accuracy", "total_latency_s", "total_co2_g", "participation_gini")
        for policy_name, pstats in res["policies"].items():
            self.assertEqual(pstats["n_seeds"], 3)
            for m in metrics:
                self.assertIn("mean", pstats[m])
                self.assertIn("std", pstats[m])
                self.assertEqual(len(pstats[m]["values"]), 3)
                self.assertTrue(np.isfinite(pstats[f"{m}_mean"]))
                self.assertGreaterEqual(pstats[f"{m}_std"], 0.0)
                self.assertAlmostEqual(
                    pstats[f"{m}_mean"],
                    float(np.mean(pstats[m]["values"])),
                    places=8,
                    msg=f"Mean mismatch for {policy_name}.{m}",
                )
                self.assertAlmostEqual(
                    pstats[f"{m}_std"],
                    float(np.std(pstats[m]["values"])),
                    places=8,
                    msg=f"Std mismatch for {policy_name}.{m}",
                )


if __name__ == "__main__":
    unittest.main()
