"""Build the paper's figures from results/ (no new experiments). Run: python paper/make_figures.py"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

LABEL = {"FedAvg_all4": "FedAvg (all 4)", "Fastest2": "Fastest-2",
         "LowestModelEnergy2": "Lowest-energy-2", "Q_learning_RL": "Q-learning (ours)"}
STYLE = {"FedAvg_all4": ("#1f4e9c", "o"), "Fastest2": ("#b45309", "s"),
         "LowestModelEnergy2": ("#0f766e", "^"), "Q_learning_RL": ("#c82d14", "D")}


def rows(name):
    with open(ROOT / "results" / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# Fig. 1: measured single-PC benchmark (wall meter, CIFAR-10 FedAvg, 3000 rounds).
pcs = ["PC2", "PC3"]
minutes = [192.075, 137.528]
kwh = [0.276, 0.480]
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55))
for ax, vals, ylabel, fmt in [(axes[0], minutes, "Training time (min)", "{:.1f}"),
                              (axes[1], kwh, "Wall energy (kWh)", "{:.3f}")]:
    bars = ax.bar(pcs, vals, color=["#7a8aa8", "#c82d14"], width=0.55)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals) * 1.22)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("$-$28.4% time", fontsize=8)
axes[1].set_title("+73.9% energy", fontsize=8)
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "measured_benchmark.pdf")
plt.close(fig)

# Per-round data, averaged over the 5 paired trial seeds.
per_round = rows("per_round_SIMULATION.csv")
curves = defaultdict(lambda: defaultdict(list))
for r in per_round:
    curves[r["policy"]][int(r["round"])].append(
        (float(r["test_accuracy_REAL_MODEL_SYNTHETIC_DATA"]), float(r["cum_CO2_g_MODELED"])))

# Fig. 2: final accuracy vs total CO2, mean with 95% CI over seeds.
summary = {r["policy"]: r for r in rows("comparison_SIMULATION.csv")}
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for pol, s in summary.items():
    c, m = STYLE[pol]
    acc = float(s["final_accuracy_mean"]) * 100
    lo, hi = float(s["final_accuracy_ci95_low"]) * 100, float(s["final_accuracy_ci95_high"]) * 100
    co2 = float(s["mean_total_CO2_g_MODELED"])
    co2_sd = float(s["mean_total_CO2_g_std"])
    ax.errorbar(co2, acc, yerr=[[acc - lo], [hi - acc]], xerr=1.96 * co2_sd / np.sqrt(5),
                fmt=m, color=c, ms=5, capsize=2, lw=0.8, label=LABEL[pol])
ax.set_xlabel(r"Total CO$_2$ over 20 rounds (g, modeled)")
ax.set_ylabel("Final test accuracy (%)")
ax.set_xlim(0, 21)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "accuracy_vs_co2.pdf")
plt.close(fig)

# Fig. 3: mean cumulative CO2 and mean test accuracy per round.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9))
for pol in summary:
    c, m = STYLE[pol]
    rs = sorted(curves[pol])
    acc = [np.mean([a for a, _ in curves[pol][k]]) * 100 for k in rs]
    co2 = [np.mean([g for _, g in curves[pol][k]]) for k in rs]
    axes[0].plot(rs, co2, color=c, marker=m, ms=2.5, lw=1.0, label=LABEL[pol])
    axes[1].plot(rs, acc, color=c, marker=m, ms=2.5, lw=1.0, label=LABEL[pol])
axes[0].set_ylabel(r"Cumulative CO$_2$ (g, modeled)")
axes[1].set_ylabel("Test accuracy (%)")
for ax in axes:
    ax.set_xlabel("Federated round")
    ax.grid(alpha=0.25, lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout(w_pad=2)
fig.savefig(OUT / "rounds.pdf")
plt.close(fig)

# Numbers quoted in the text: RL accuracy at peak vs final round, averaged over seeds.
rl = curves["Q_learning_RL"]
rl_mean = {k: np.mean([a for a, _ in v]) for k, v in rl.items()}
peak = max(rl_mean, key=rl_mean.get)
print(f"RL mean accuracy peak round {peak}: {rl_mean[peak]*100:.2f}%  final: {rl_mean[max(rl_mean)]*100:.2f}%")
for pol in summary:
    m = {k: np.mean([a for a, _ in v]) for k, v in curves[pol].items()}
    print(pol, "round1", f"{m[1]*100:.2f}", "final", f"{m[max(m)]*100:.2f}", "max", f"{max(m.values())*100:.2f}")
print("figures written to", OUT)
