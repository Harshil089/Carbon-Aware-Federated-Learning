# Single-Device MNIST Carbon Workload

This repository now starts with a **single-device, duration-controlled training
experiment**. It keeps the model and dataset small enough to run on different
hardware, while repeatedly training until a wall-clock deadline so every device
can be kept busy for the same amount of time.

The default experiment runs for at least 10 hours. It uses a compact two-layer
NumPy neural network on MNIST and CodeCarbon to record energy and estimated
CO2-equivalent emissions. The earlier five-client federated baseline remains in
the repository for a later stage, but it is no longer the primary run path.

## Install and run

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/run_single_device.py
```

Before comparing machines, set `country_iso_code` in
`configs/single_device_10h.json` to the 3-letter ISO code for the machine's
location. The supplied value is `IND`. Keep the same config on every device for
a fair workload comparison.

For a quick smoke test, override the duration (0.01 hours is 36 seconds):

```bash
python experiments/run_single_device.py --duration-hours 0.01
```

Press `Ctrl+C` to stop early. Periodic metrics and carbon measurements already
written to disk are retained.

## Outputs

Results are written to `results/single_device_10h/` by default:

- `training_metrics.csv`: elapsed time, throughput, loss, and accuracy
- `emissions.csv`: CodeCarbon energy, power, and CO2-equivalent estimates
- `hardware.json`: machine, CPU, Python, and NumPy details
- `resolved_config.json`: exact experiment settings
- `summary.json`: final training and emissions summary after a complete run

CodeCarbon automatically uses an available hardware measurement method and may
fall back to a power estimate on unsupported hardware. Therefore, inspect the
CodeCarbon logs/method fields and treat cross-device emissions as estimates,
especially when the measurement methods differ.

## Configuration

Edit `configs/single_device_10h.json` to change duration, model width, batch
size, data limits, reporting frequency, or output location. Increasing
`hidden_dim` raises compute intensity and memory use. The default width of 256
keeps model parameters under 1 MB (temporary batch arrays use a few more MB).

## Validate

```bash
python -m unittest discover -s tests -v
```
