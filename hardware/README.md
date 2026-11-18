# Real-hardware federated run (measured path)

This directory runs the same FedAvg softmax classifier as the simulator on real PCs, using Flower (`flwr`). It writes a per-round CSV with the simulator's columns first, then measured columns. The simulator (`results/per_round_SIMULATION.csv`) stays the modeled reference; the CSV from this path is the measured one. Measured and modeled columns are never mixed in one row: on real hardware the `*_MODELED` columns are empty.

Only PC1, PC2 and PC3 have hardware here. PC4 is a scenario profile and is not part of this run.

## Install

On the coordinator and on each client PC:

```
pip install -r requirements-optional.txt
```

This installs `flwr` and `codecarbon`. The simulator and the test suite do not need them. `flwr` is imported only when `server.py` or `client.py` runs.

## Run

On the coordinator machine:

```
python -m hardware.server --rounds 20 --clients PC1,PC2,PC3 --address 0.0.0.0:8080 --seed 42 --output results_hardware/per_round_HARDWARE.csv
```

On PC1, PC2 and PC3 (same command with the node name changed):

```
python -m hardware.client --name PC1 --server <coordinator-ip>:8080 --seed 42
python -m hardware.client --name PC2 --server <coordinator-ip>:8080 --seed 42
python -m hardware.client --name PC3 --server <coordinator-ip>:8080 --seed 42
```

Start the coordinator first, then the clients. The server waits until all listed clients have connected before round 1. Use the same `--seed` on every machine, since each client builds its data partition from the seed and the node's position in `config.json`.

## Wall-meter readings

A wall meter reads energy for a whole run, not per round. Pass one total per node, in Wh:

```
python -m hardware.server ... --wallmeter "PC1=65.0,PC2=20.0,PC3=31.5"
```

The server adds the supplied values and records the total in `round_Wh_MEASURED_WALLMETER` on the final round row only. `round_CO2_g_MEASURED` on that row is that Wh times the grid factor (0.675 kgCO2/kWh from `config.json`). Other rows use CodeCarbon Wh if available, or are empty.

## Output columns

- First 11 columns: same names and order as `results/per_round_SIMULATION.csv`. `policy` is `MEASURED_HARDWARE`, and the `*_MODELED` columns are empty.
- `round_seconds_MEASURED`: coordinator wall time from sending fit instructions to aggregating results.
- `round_Wh_MEASURED_CODECARBON`: sum of the clients' CodeCarbon estimates for the round. Empty if any client did not report one.
- `round_Wh_MEASURED_WALLMETER`: only on the final round, from `--wallmeter`.
- `round_CO2_g_MEASURED`: measured Wh times grid factor. Wall meter is preferred, then CodeCarbon, else empty.
- `cum_seconds_MEASURED`: running total of `round_seconds_MEASURED`.
- `node_wall_seconds_MEASURED`: JSON map of node name to local training seconds for that round.

## Network and firewall

The coordinator must accept inbound TCP on the port in `--address` (8080 above). Allow it through the OS firewall on the coordinator and make sure the client PCs can reach that IP.

## Caveats

- CodeCarbon numbers are software estimates. They are not a wall meter, and they depend on the OS and hardware counters available. Treat them as indicative.
- The wall-meter readings are whatever the user supplies. The meter reading for PC1 in `config.json` is flagged as needing re-verification.
- The `--model torch_mlp` option is not implemented on this path. It exits with an error. Use `--model numpy`.
- This path has not been run end to end. It is not in CI, and the flwr code was not executed in the environment where it was written (flwr is not installed there). Run a short 2-round test on the real machines before a full run.
