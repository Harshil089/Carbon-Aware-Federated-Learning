# Federated Computing Lab

**A Multi-Objective Reinforcement Learning Model for Optimizing the Latency–Carbon Trade-off in Federated Computing**

A working Python simulation and local browser console. Run experiments, compare four selection policies, replay client participation, inspect exact parameters, and export a completed run.

## Start on Windows

1. Extract the whole project folder.
2. Double-click **START_DEMO.bat**. The launcher reuses the installed `.venv`, or creates one using Python 3.10+ and installs NumPy and Matplotlib if needed.
3. Keep the console window open. The browser opens automatically, normally at **http://127.0.0.1:8765/**. If that port is busy, the console prints the selected port.
4. Choose **Quick demo**, then **Run Simulation**.

First-time dependency setup needs internet. After setup, the app and simulation work offline. `START_WINDOWS.bat` opens the same app. Linux/macOS: run `bash START_DEMO.sh` with Python 3.10+ installed.

## Demonstration flow

- **Run:** choose a preset. Progress and log messages come from the Python process.
- **Overview:** compare final accuracy, estimated latency, energy, carbon, and target hit rate.
- **Charts:** inspect learning progress and the latency–carbon trade-off.
- **Client replay:** select a policy and trial, then play its recorded client selections.
- **Parameters:** inspect exact settings for the displayed results.
- **Export run ZIP:** retain the CSVs, charts, parameter record, offline dashboard, and log.

## Default parameters

| Setting | Quick demo | Full experiment |
|---|---:|---:|
| Federated rounds | 12 | 20 |
| RL training episodes | 35 | 130 |
| Performance repeats | 3 | 5 |
| Sensitivity samples | 12 | 70 |
| Comparison target | 80% | 80% |
| Local epochs | 2 | 2 |
| Local learning rate | 0.23 | 0.23 |
| Dataset seed | 42 | 42 |
| Latency penalty | 0.018 | 0.018 |
| Carbon penalty | 0.32 | 0.32 |

Both presets use four clients, 250 training examples per client, 1,500 test examples, three classes, and two features plus a bias term. The model is a softmax linear classifier. Data are synthetic and non-IID; NumPy training runs on CPU.

Tabular Q-learning uses alpha 0.24, gamma 0.86, epsilon starting at 0.80 with a 0.06 floor, four training phases, and five accuracy buckets. Eleven actions select subsets of two, three, or four clients. The comparison target controls target-hit reporting. The reward has a separate 78% accuracy floor.

## Infrastructure parameters

| Client | RAM | GPU VRAM descriptor | Speed ratio | Power ratio | Idle W | Network s |
|---|---:|---:|---:|---:|---:|---:|
| PC1 | 8 GB | — | 1.00 | 1.00 | 7 | 0.30 |
| PC2 | 16 GB | — | 1.10 | 1.25 | 9 | 0.45 |
| PC3 | 32 GB | 8 GB | 1.80 | 2.45 | 13 | 0.60 |
| PC4 | 16 GB | 4 GB | 1.35 | 1.75 | 11 | 0.45 |

The model uses 16 seconds of base client work, 0.45 seconds of aggregation, a 30 W coordinator, ±10% performance jitter, and a configured grid factor of 0.675 kg CO₂/kWh. Reference input: 0.065 kWh over 190 minutes. PC1 uses its mean power anchor; other profiles use scenario ratios. Provenance files retain source details. GPU memory describes hardware profiles and does not enable GPU training.

Round latency is the longest selected client completion time plus aggregation. Energy includes active work, idle clients, and the coordinator. Wh multiplied by the grid factor in kg CO₂/kWh gives grams CO₂. Sensitivity scenarios vary power by ±35% and duration scaling by ±30% for clients 2–4.

Accuracy is computed on synthetic test data. Infrastructure latency, energy and carbon are modeled. Performance repeats vary timing draws while sharing data and the trained policy. The app reports the resulting trade-off without assuming RL beats every baseline.

## Files and reproducibility

- `results/`: full default experiment and offline `dashboard.html`.
- `runs/<run id>/`: app experiments, each with its log and `run_parameters.json`.
- `config.json`: infrastructure settings and input provenance.
- `DEMO_GUIDE.md`: short demonstration sequence.
- `reference/`: earlier reference material, separate from the current demo.

Command-line run:

```bash
python run_project.py --rounds 20 --episodes 130 --repeats 5 --sensitivity 70
```

Additional flags: `--target 0.80`, `--local-epochs 2`, `--learning-rate 0.23`, `--seed 42`, `--latency-weight 0.018`, `--carbon-weight 0.32`, and `--output PATH`.

Checks: `python tests.py` and `python tests_demo.py`.

Close the console or press Ctrl+C there to stop the app. A browser tab alone does not execute Python. The offline dashboard keeps completed results available after the engine stops.
