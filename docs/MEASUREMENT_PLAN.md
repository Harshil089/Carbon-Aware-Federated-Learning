# Measurement plan for the remaining hardware runs

This is the run sheet for the next lab sessions. [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) sets the general rules; this plan turns them into concrete runs, in order, with what to record and how to check each reading before moving on.

## Where things stand

| PC | CPU | Meter reading | Status |
|---|---|---|---|
| PC1 | Intel Core i7-12700 | 0.065 kWh / 190 min (20.53 W) | Needs re-verification. 20.5 W is below what a desktop of this class usually draws at idle, and the workload for this run is not recorded. |
| PC2 | Intel Core Ultra 5 235 | 0.276 kWh / 192.1 min (86.22 W) | Measured, CIFAR-10 CPU, 3000 rounds, one run. |
| PC3 | Intel Core Ultra 7 265K | 0.480 kWh / 137.5 min (209.4 W) | Measured, CIFAR-10 CPU, 3000 rounds, one run. |
| PC4 | Unknown | None | Not measured. |

What the existing readings show: on the same workload, PC3 finishes 28.4% sooner than PC2 but uses 73.9% more energy. That is a valid single-run hardware result.

What they do not show yet:

- **Baseline draw.** No idle reading exists, so training energy cannot be separated from what each PC draws anyway.
- **Repeatability.** Each PC has one run, so there is no spread to report.
- **Federated training across PCs.** Each PC ran the whole 50-client experiment on its own. The client-selection pipeline itself has not been measured.
- **Carbon over time.** One fixed factor (0.675 kg/kWh) was used, so carbon is energy times a constant.
- **Accuracy.** The 0.35-point gap between PC2 and PC3 is run-to-run noise, not a hardware effect. Hardware does not change what the model learns. Report it as "same accuracy within noise".

## Phase 0 — Set up once

Do this before any reading. Write the answers in `data/measurements/setup.md`.

1. **Meter.** Record make, model, accuracy class and display resolution (the current meter shows 0.001 kWh). If there is more than one meter, label them and always use the same meter on the same PC, or record which meter was used for each run.
2. **Measurement boundary.** Only the PC tower is on the meter. The monitor, speakers and network switch stay on a different socket. Write this down; it is the first question a reviewer asks.
3. **PC state.** Same OS power plan on every PC ("Balanced" or "High performance", the same for all). Turn off sleep, screen savers, Windows Update and antivirus scans for the session. Close every other program. Note the room temperature.
4. **Software.** Record OS version, Python version and `pip freeze` output for each PC.
5. **Benchmark script.** Commit the CIFAR-10 benchmark script used for the PC2 and PC3 runs into this repository (for example `benchmarks/cifar10_fedavg.py`). It is not in the repo yet. Without it, the runs cannot be reproduced or cited.
6. **Seeds.** The script must take a `--seed` argument and set the Python, NumPy and PyTorch seeds from it. Fix the number of CPU threads (`torch.set_num_threads`) and record it.

## Phase 1 — Idle baseline, every PC

**Answers:** How much of each measured watt is training, and how much is the PC being switched on?

For PC1, PC2, PC3, PC4 and the coordinator machine:

1. Boot, log in, close everything, wait 5 minutes.
2. Note the meter reading and the time.
3. Leave the PC untouched for **30 minutes**. At 0.001 kWh resolution, a 30 W idle draw gives 0.015 kWh, so 30 minutes keeps the rounding error under 7%. Use 60 minutes if the idle draw is below 20 W.
4. Note the meter reading and the time again.
5. Repeat on a second day.

**Check before moving on:** idle power must be clearly below the training power for the same PC. If PC1's idle comes out above 20.5 W, the PC1 training reading is wrong and must be redone in Phase 2.

## Phase 2 — Single-PC benchmark, with repeats

**Answers:** What are the energy and time per round for each PC, with a spread?

Use the same CIFAR-10 configuration as the PC2 and PC3 runs: FedAvg, 50 clients, 10 per round, Dirichlet α = 0.1, CPU only.

1. **Short-run check (once, on PC2).** Run 500 rounds and work out Wh per round. If it is within 5% of the 3000-round figure (0.092 Wh), 500-round runs are enough for the repeats below. If not, keep 3000 rounds.
2. **Runs.** For each of PC1, PC2, PC3 and PC4, do **3 runs** with seeds 1, 2 and 3. That is 12 runs in total. PC2 and PC3 already have one 3000-round run each; keep those as an extra data point, but do not count them as one of the three.
3. **During each run.** Note the meter reading and clock time at the start, every 15 minutes, and at the end. The 15-minute readings show whether power stays steady or drops when the CPU heats up.
4. **After each run.** Record final accuracy, final loss and total rounds from the script's output.

**Check before moving on:**

- Average power from (kWh ÷ hours) must match what the meter suggests between the 15-minute readings.
- The three runs on one PC should agree within about 5% on energy per round. If they do not, find out why before continuing (background task, thermal throttling, power plan).
- Accuracy should be the same across PCs for the same seed, within noise. If it differs by a lot, something other than the hardware changed.

**Report:** mean ± spread for time per round and Wh per round on each PC. Report full-wall energy and idle-subtracted energy (wall minus Phase 1 idle × duration) side by side, never mixed.

## Phase 3 — Federated run across the PCs

**Answers:** Do the simulated savings from choosing which PCs train hold up on real machines?

This uses `hardware/` (see [hardware/README.md](../hardware/README.md)). That path runs the NumPy model, not CIFAR-10, so its energy numbers are not directly comparable with Phase 2. That is fine; this phase compares the selection rules with each other.

1. **Smoke test.** Run 2 rounds with PC1, PC2 and PC3. Fix any network or firewall issues. Nothing from this run is reported.
2. **Fixed rules.** The server trains whichever PCs are listed in `--clients`, so each fixed rule is one list:

   | Rule | `--clients` |
   |---|---|
   | All PCs (FedAvg) | `PC1,PC2,PC3` (add `PC4` once it is set up) |
   | Fastest two | the two PCs with the lowest time per round from Phase 2 |
   | Lowest-power two | the two PCs with the lowest Wh per round from Phase 2 |

   Run each rule for 20 rounds, 3 times (seeds 1, 2, 3).
3. **Metering.** With one meter per PC, read all meters at the start and end of each run. With only one meter, repeat each run once per PC, moving the meter each time. The same seed gives the same training, so the readings can be combined. Never split one meter's total between PCs.
4. **Idle PCs count.** A PC that is switched on but not picked still draws idle power. Either switch it off for that rule or include its idle draw in the total. Write down which.
5. **RL picker.** The RL picker changes which PCs train from round to round. `hardware/server.py` cannot do that yet; it trains the same list every round. Running the RL picker on hardware needs a small addition: the server reads the round-by-round choices from a simulator run and follows them. Do the fixed rules first.

**Report:** for each rule, total energy and time to reach the same target accuracy, with the spread over the three runs.

## Phase 4 — Carbon intensity that changes over the day

**Answers:** Does choosing *when* to train reduce CO₂, on top of choosing *which* PCs train?

The energy a run uses does not depend on the time of day; only the CO₂ per kWh does. So no extra lab runs are needed for this phase. Phases 1 to 3 only need accurate timestamps.

1. Get an hourly carbon-intensity series for your grid region for the dates of the runs. Possible sources are Electricity Maps, or a series built from Grid-India hourly generation mix and CEA emission factors. Check the source's coverage and licence before using it, and cite it.
2. Recompute CO₂ for each run as the sum, over each 15-minute reading, of energy in that interval × intensity at that hour.
3. In the simulator, replace the fixed 0.675 factor with the hourly series and add start time as a choice the scheduler can make. Use the Phase 2 power figures, not assumed ones.

Keep the annual CEA factor as the main reported figure. The hourly series is an additional analysis, and the paper must say which one each number uses.

## Recording sheet

One row per reading in `data/measurements/readings.csv`. Never edit a row after it is written; add a correcting row with a note instead.

```
run_id,phase,pc,meter_id,seed,workload,rounds,timestamp_local,meter_kwh,note
```

- `run_id`: e.g. `P2-PC3-s2` (phase, PC, seed).
- `workload`: `idle`, `cifar10_fedavg`, or `hardware_numpy_<rule>`.
- `timestamp_local`: `2026-11-20T14:05` format, with the timezone written in `setup.md`.
- One row at the start, one every 15 minutes, one at the end.

Keep a photo of the meter display for the first and last reading of every run.

## Order and rough time

| Step | Sessions |
|---|---|
| Phase 0 setup | 1 short session |
| Phase 1 idle (5 machines × 2 days) | can run alongside other work, about 35 minutes per machine per day |
| Phase 2 (12 runs) | about 6–10 hours with 500-round runs, about 35 hours at 3000 rounds |
| Phase 3 (3 rules × 3 runs × 20 rounds) | 1–2 sessions once the network works |
| Phase 4 | analysis only, no lab time |

Phases 1 and 2 can start right away. Phase 3 needs all PCs on the same network at the same time. Phase 4 can start as soon as Phase 2 has timestamped readings.
