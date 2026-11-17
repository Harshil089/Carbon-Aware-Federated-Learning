# Physical validation after laboratory reopening

1. Inventory PC1–PC4 and coordinator (CPU exact model, GPU model, RAM, OS, power plan, software versions, cooling). Record whether code actually uses GPU.
2. Confirm meter units, error/accuracy class, kWh resolution and whether monitors/peripherals are inside the measurement boundary.
3. Record starting and ending kWh and precise timestamp for an idle baseline on **each** machine.
4. Define the dataset, identical train/test split and training settings. Record workload hash, batch size, local epochs, selected clients, accuracy threshold, background tasks and environmental conditions.
5. Measure the same benchmark separately on each PC, ideally >=3 runs, using the same meter moved between PCs if needed, and reserve the full federated LAN experiment for when machines are simultaneously available. For a simultaneous deployment measure each node with its own meter, or a verified synchronized power telemetry approach; do not attribute aggregate power to individual nodes otherwise.
6. Install network software and use actual IPv4 LAN addresses; record round start/end timestamps and network byte counts. Obtain proper administrative approval for university lab firewall/network changes.
7. Run four-policy experiments with controlled data and seed: FedAvg all-4, fastest-two, energy-aware-two and RL. Compare cumulative energy and time **to reach equal target accuracy**; report when the target is not reached.
8. Include coordinator and idle-worker energy in system totals. Report full-wall energy and an idle-subtracted sensitivity analysis separately, without mixing them.
9. Use the appropriate location-based electricity factor for the fiscal year and document whether grid electricity was used. Do not treat a static annual factor as marginal or real-time grid intensity.
10. Present medians/means with measured variability; preserve raw readings in read-only source files, and never overwrite original measurements with predicted values.
