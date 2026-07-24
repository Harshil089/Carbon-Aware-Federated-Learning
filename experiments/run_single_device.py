#!/usr/bin/env python3
"""Run a duration-controlled single-device MNIST carbon experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402

from src.data.mnist import load_mnist  # noqa: E402
from src.models.mlp import MLPClassifier  # noqa: E402
from src.training.duration import TrainingProgress, train_for_duration  # noqa: E402


def _make_tracker(config: dict[str, Any], output_dir: Path):
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError as error:
        raise RuntimeError(
            "Carbon tracking requires CodeCarbon. Run: python -m pip install -r requirements.txt"
        ) from error
    return OfflineEmissionsTracker(
        project_name="single-device-mnist",
        country_iso_code=config["country_iso_code"],
        measure_power_secs=config["carbon_measure_interval_seconds"],
        output_dir=str(output_dir),
        output_file="emissions.csv",
        log_level="warning",
    )


def _hardware_metadata() -> dict[str, Any]:
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "logical_cpu_count": __import__("os").cpu_count(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/single_device_10h.json",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        help="Override duration for smoke tests, for example 0.01 for 36 seconds.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the result directory.")
    args = parser.parse_args()

    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if args.duration_hours is not None:
        config["duration_hours"] = args.duration_hours
    if config["duration_hours"] <= 0:
        parser.error("duration_hours must be positive")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else REPOSITORY_ROOT / config["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "hardware.json").write_text(
        json.dumps(_hardware_metadata(), indent=2) + "\n", encoding="utf-8"
    )

    train_x, train_y, test_x, test_y = load_mnist(
        REPOSITORY_ROOT / config["data_dir"],
        config["train_limit"],
        config["test_limit"],
    )
    model = MLPClassifier(784, config["hidden_dim"], 10, config["seed"])
    metrics_path = output_dir / "training_metrics.csv"
    tracker = _make_tracker(config, output_dir)

    with metrics_path.open("w", newline="", encoding="utf-8", buffering=1) as stream:
        fieldnames = list(TrainingProgress.__dataclass_fields__)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        def record(progress: TrainingProgress) -> None:
            writer.writerow(progress.as_dict())
            tracker.flush()
            print(
                f"{progress.elapsed_seconds / 3600:.2f}h | "
                f"{progress.samples_per_second:.0f} samples/s | "
                f"accuracy {progress.test_accuracy:.3f}",
                flush=True,
            )

        tracker.start()
        try:
            final = train_for_duration(
                model,
                train_x,
                train_y,
                test_x,
                test_y,
                duration_seconds=config["duration_hours"] * 3600,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                report_interval_seconds=config["report_interval_seconds"],
                seed=config["seed"],
                on_report=record,
            )
        except KeyboardInterrupt:
            print("\nStopped early; measurements collected so far were preserved.")
            return
        finally:
            emissions_kg = tracker.stop()

    summary = {
        **final.as_dict(),
        "target_duration_hours": config["duration_hours"],
        "emissions_kg_co2eq": emissions_kg,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed. Results: {output_dir}")


if __name__ == "__main__":
    main()
