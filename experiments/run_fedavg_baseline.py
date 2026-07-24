#!/usr/bin/env python3
"""Run Step 3: the minimum multi-round FedAvg baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.mnist import load_mnist  # noqa: E402
from src.federated.server import make_iid_clients, run_fedavg  # noqa: E402


def _write_svg(history: list[dict[str, object]], destination: Path) -> None:
    width, height, margin = 720, 420, 55
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points: list[str] = []
    for index, row in enumerate(history):
        x = margin + (index / max(1, len(history) - 1)) * plot_width
        y = margin + (1.0 - float(row["global_accuracy"])) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">FedAvg global accuracy by round</text>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>
<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif">Accuracy</text>
<text x="{width / 2}" y="{height-12}" text-anchor="middle" font-family="sans-serif">Round</text>
<polyline fill="none" stroke="#1572a1" stroke-width="3" points="{' '.join(points)}"/>
<text x="{margin-8}" y="{margin+5}" text-anchor="end" font-family="sans-serif" font-size="12">100%</text>
<text x="{margin-8}" y="{height-margin+5}" text-anchor="end" font-family="sans-serif" font-size="12">0%</text>
</svg>
"""
    destination.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/fedavg_baseline.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    train_x, train_y, test_x, test_y = load_mnist(
        REPOSITORY_ROOT / config["data_dir"],
        config["train_limit"],
        config["test_limit"],
    )
    clients = make_iid_clients(train_x, train_y, config["num_clients"], config["seed"])
    _, history = run_fedavg(
        clients,
        test_x,
        test_y,
        rounds=config["rounds"],
        clients_per_round=config["clients_per_round"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        seed=config["seed"],
    )

    output_dir = REPOSITORY_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        for row in history:
            writer.writerow({**row, "selected_clients": json.dumps(row["selected_clients"])})
    _write_svg(history, output_dir / "convergence.svg")
    print(f"Saved metrics and convergence plot to {output_dir}")


if __name__ == "__main__":
    main()

