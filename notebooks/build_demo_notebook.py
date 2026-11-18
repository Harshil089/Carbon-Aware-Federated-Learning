"""Writes notebooks/Pipeline_Demo_Colab.ipynb. Run: python notebooks/build_demo_notebook.py"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


md("""
# Carbon-aware federated learning — pipeline demo

This notebook runs the full pipeline on four **virtual** PCs, because the physical machines are not available today.

**What you will see, step by step:**
1. The four PCs and where each number comes from (measured with a wall meter, or assumed).
2. Each PC's private training data. The data never leaves its PC.
3. The RL picker learning which PCs to train each round.
4. One complete federated training run, replayed round by round.
5. A comparison of the RL picker with three simple rules, over 5 independent runs.

**What is real and what is simulated:**
- The model training is real: each picked PC trains a model on its own data, and the coordinator averages the results (FedAvg).
- Round time, energy and CO₂ are calculated from each PC's profile. PC2 and PC3 use wall-meter measurements, PC1 uses a reading still to be re-verified, and PC4 is assumed.

**To run:** *Runtime → Run all*. It takes about a minute and needs no GPU.
""")

code("""
# Get the project code (skipped if it is already here).
import os, sys, subprocess
REPO = 'https://github.com/Harshil089/Carbon-Aware-Federated-Learning.git'
if os.path.exists('run_project.py'):
    pass
elif os.path.exists('../run_project.py'):
    os.chdir('..')
else:
    if not os.path.exists('Carbon-Aware-Federated-Learning'):
        subprocess.run(['git', 'clone', '-q', '--depth', '1', REPO], check=True)
    os.chdir('Carbon-Aware-Federated-Learning')
sys.path.insert(0, os.getcwd())

from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_project as rp
from fl_pipeline import TabularQLearningScheduler
%matplotlib inline
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
pd.set_option('display.width', 160)

PCS = [n['name'] for n in rp.NODES]
COLORS = {'FedAvg_all4': '#1f4e9c', 'Fastest2': '#b45309', 'LowestModelEnergy2': '#0f766e', 'Q_learning_RL': '#c82d14'}
NAMES = {'FedAvg_all4': 'All 4 PCs (FedAvg)', 'Fastest2': 'Fastest 2', 'LowestModelEnergy2': 'Lowest-energy 2',
         'Q_learning_RL': 'RL picker'}
print('Project loaded from', os.getcwd())
""")

md("""
## Step 1 — The four PCs

Every number has a source. **Meter** means it came from a physical electricity meter; **assumed** means it is a modelling input.
""")

code("""
source = {'PC1': 'meter (to re-verify)', 'PC2': 'meter', 'PC3': 'meter', 'PC4': 'assumed'}
pcs = pd.DataFrame([{
    'PC': n['name'],
    'Processor': 'not recorded (assumed profile)' if n['processor'].startswith('UNKNOWN') else n['processor'],
    'RAM (GB)': n['ram_gb'],
    'Speed vs PC1': n['relative_speed'],
    'Power while training (W)': round(rp.AVG_W_PC1 * n['active_power_ratio_to_PC1'], 1),
    'Idle power (W, assumed)': n['idle_power_w_assumed'],
    'Power source': source[n['name']],
} for n in rp.NODES]).set_index('PC')
pcs
""")

md("""
### The measured result behind the project

PC2 and PC3 ran the **same** CIFAR-10 federated training job (3000 rounds, CPU only) on a wall meter.
""")

code("""
measured = pd.DataFrame({
    'Time (min)': [192.1, 137.5],
    'Energy (kWh)': [0.276, 0.480],
    'Average power (W)': [86.2, 209.4],
    'Final accuracy (%)': [72.11, 72.46],
}, index=['PC2 (Core Ultra 5 235)', 'PC3 (Core Ultra 7 265K)'])
display(measured)

fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
for a, col, c in [(ax[0], 'Time (min)', '#7a8aa8'), (ax[1], 'Energy (kWh)', '#c82d14')]:
    bars = a.bar(['PC2', 'PC3'], measured[col], color=[c, c], alpha=0.85)
    a.bar_label(bars, fmt='%.3g'); a.set_title(col)
fig.suptitle('PC3 is 28.4% faster but uses 73.9% more energy for the same job', y=1.05)
plt.show()
""")

md("""
## Step 2 — Each PC's private data

Each PC holds 250 samples of a 3-class problem. The data is deliberately uneven (non-IID): PC1, PC2 and PC3 each have mostly one class, and PC4 has a balanced mix. This is what makes *which* PCs train together matter.
""")

code("""
parts, val, test = rp.make_data(seed=42)
class_colors = ['#4c72b0', '#dd8452', '#55a868']
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharex=True, sharey=True)
for ax, name, (X, y) in zip(axes, PCS, parts):
    for k in range(3):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=6, color=class_colors[k], label=f'class {k}')
    share = np.bincount(y, minlength=3) / len(y) * 100
    ax.set_title(f'{name}: ' + ' / '.join(f'{s:.0f}%' for s in share), fontsize=9)
axes[0].legend(fontsize=7, loc='lower left')
fig.suptitle('Private data on each PC (share of class 0 / 1 / 2)', y=1.04)
plt.show()
""")

md("""
## Step 3 — The RL picker learns which PCs to train

Each round, the picker chooses 2, 3 or all 4 PCs (11 possible groups). Its reward goes up when accuracy improves and down for time and CO₂. It learns by practising on separate training data, then plays greedily. The test data is never shown to it.
""")

code("""
cfg = SimpleNamespace(rounds=20, episodes=130, repeats=5, local_epochs=2, learning_rate=0.23,
                      latency_weight=0.018, carbon_weight=0.32, target=0.8, seed=42, train_seeds=3)
train_envs = [rp.build_env(cfg, cfg.seed + 2000 + k) for k in range(cfg.train_seeds)]
sched = TabularQLearningScheduler(train_envs[0].actions, seed=cfg.seed + 1984)

def on_episode(n, total):
    if n % 26 == 0 or n == total:
        print(f'  practice run {n:>3}/{total}')
print('Training the RL picker...')
sched.train(train_envs[0], episodes=cfg.episodes, progress_callback=on_episode, envs=train_envs)

# What it learned: the group it picks in each situation.
phases = ['rounds 1-5', 'rounds 6-10', 'rounds 11-15', 'rounds 16-20']
buckets = ['<50%', '50-68%', '68-76%', '76-81%', '>81%']
learned = pd.DataFrame(index=phases, columns=buckets)
for p in range(4):
    for b in range(5):
        q = sched.q_table.get((p, b))
        learned.iloc[p, b] = '-' if q is None or not np.any(q) else '+'.join(PCS[i] for i in sched.actions[int(np.argmax(q))])
print('\\nGroup the RL picker chooses (rows: stage of training, columns: current validation accuracy; "-" = situation never reached):')
learned
""")

md("""
## Step 4 — One federated training run, round by round

In every round: the coordinator sends the current model to the picked PCs, each trains on its own data, sends back only its model update, and the coordinator averages them.
""")

code("""
trial_seed = cfg.seed + 1158
env = rp.build_env(cfg, trial_seed)
run = pd.DataFrame(rp.evaluate('Q_learning_RL', env, sched, trial_seed))
for _, r in run.iterrows():
    print(f"Round {r['round']:>2}: coordinator -> {r['clients_selected']:<15} train -> average  | "
          f"accuracy {r['test_accuracy_REAL_MODEL_SYNTHETIC_DATA']*100:5.1f}%  "
          f"time {r['round_seconds_MODELED']:5.1f}s  energy {r['round_Wh_MODELED']:.3f} Wh  CO2 {r['round_CO2_g_MODELED']:.3f} g")
print(f"\\nTotal: {run['cum_seconds_MODELED'].iloc[-1]:.0f} s, {run['cum_Wh_MODELED'].iloc[-1]:.2f} Wh, "
      f"{run['cum_CO2_g_MODELED'].iloc[-1]:.2f} g CO2, final accuracy {run['test_accuracy_REAL_MODEL_SYNTHETIC_DATA'].iloc[-1]*100:.1f}%")
""")

code("""
# The same run as a picture: green PCs trained in that round, grey PCs waited.
pos = {'PC1': (-1, 1), 'PC2': (1, 1), 'PC3': (-1, -1), 'PC4': (1, -1)}
fig, axes = plt.subplots(4, 5, figsize=(13, 10))
for ax, (_, r) in zip(axes.flat, run.iterrows()):
    picked = r['clients_selected'].split('+')
    for name, (x, y) in pos.items():
        on = name in picked
        ax.plot([0, x * 0.75], [0, y * 0.75], color='#0f766e' if on else '#d0d0d0', lw=2 if on else 1,
                ls='-' if on else ':')
        ax.scatter(x, y, s=650, color='#0f766e' if on else '#e8e4da', edgecolor='#333', zorder=3)
        ax.text(x, y, name, ha='center', va='center', fontsize=7, color='white' if on else '#555', zorder=4)
    ax.scatter(0, 0, s=500, marker='s', color='#121316', zorder=3)
    ax.set_title(f"Round {r['round']}  ·  {r['test_accuracy_REAL_MODEL_SYNTHETIC_DATA']*100:.1f}%", fontsize=9)
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.axis('off')
fig.suptitle('RL picker, round by round (black square = coordinator)', fontsize=12)
plt.tight_layout(); plt.show()
""")

md("""
## Step 5 — RL picker vs. simple rules, over 5 independent runs

Each run uses fresh data and fresh timing noise, shared by all four rules so the comparison is fair. Brackets are 95% confidence intervals.
""")

code("""
rows = []
for t in range(cfg.repeats):
    s = cfg.seed + 1158 + t
    env = rp.build_env(cfg, s)
    for pol in rp.POLICIES:
        rows.extend(rp.evaluate(pol, env, sched, s))
summary = pd.DataFrame(rp.summarize(rows, cfg.rounds, cfg.target)).set_index('policy')
table = pd.DataFrame({
    'Final accuracy (%)': [f"{float(r.final_accuracy_mean)*100:.1f} [{float(r.final_accuracy_ci95_low)*100:.1f}, {float(r.final_accuracy_ci95_high)*100:.1f}]"
                           for r in summary.itertuples()],
    'Total time (s)': summary['mean_total_seconds_MODELED'].astype(float).round(1).values,
    'Total energy (Wh)': summary['mean_total_Wh_MODELED'].astype(float).round(2).values,
    'Total CO2 (g)': summary['mean_total_CO2_g_MODELED'].astype(float).round(2).values,
}, index=[NAMES[p] for p in summary.index])
table
""")

code("""
per = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for pol in rp.POLICIES:
    g = per[per.policy == pol].groupby('round').mean(numeric_only=True)
    ax[0].plot(g.index, g['cum_CO2_g_MODELED'], color=COLORS[pol], label=NAMES[pol], marker='o', ms=3)
    ax[1].plot(g.index, g['test_accuracy_REAL_MODEL_SYNTHETIC_DATA'] * 100, color=COLORS[pol], marker='o', ms=3)
    s = summary.loc[pol]
    ax[2].scatter(float(s.mean_total_CO2_g_MODELED), float(s.final_accuracy_mean) * 100, s=90, color=COLORS[pol])
    ax[2].annotate(NAMES[pol], (float(s.mean_total_CO2_g_MODELED), float(s.final_accuracy_mean) * 100),
                   textcoords='offset points', xytext=(6, 4), fontsize=8)
ax[0].set(title='Cumulative CO2 (g)', xlabel='Round'); ax[0].legend(fontsize=8)
ax[1].set(title='Test accuracy (%)', xlabel='Round')
ax[2].set(title='Trade-off after 20 rounds', xlabel='Total CO2 (g)', ylabel='Final accuracy (%)')
plt.tight_layout(); plt.show()

paired = rp.paired_comparison(rows)
co2 = paired['total_CO2_g_MODELED']; secs = paired['total_seconds_MODELED']; acc = paired['final_accuracy']
base = summary.loc['FedAvg_all4']
print(f"RL picker vs all 4 PCs (paired over {paired['n_pairs']} runs):")
print(f"  CO2:      {co2['mean_difference']:+.2f} g ({co2['mean_difference']/float(base.mean_total_CO2_g_MODELED)*100:+.1f}%), 95% CI {co2['ci95_low']:+.2f} to {co2['ci95_high']:+.2f}")
print(f"  Time:     {secs['mean_difference']:+.1f} s ({secs['mean_difference']/float(base.mean_total_seconds_MODELED)*100:+.1f}%), 95% CI {secs['ci95_low']:+.1f} to {secs['ci95_high']:+.1f}")
print(f"  Accuracy: {acc['mean_difference']*100:+.2f} points, 95% CI {acc['ci95_low']*100:+.2f} to {acc['ci95_high']*100:+.2f}")
""")

md("""
## Takeaways

- **Faster is not greener.** On the meter, PC3 finished the same job 28% sooner but used 74% more energy.
- **The RL picker** cuts CO₂ by about 44% and time by about 8% compared with training on all four PCs, for about 1.3 points of accuracy.
- **Simple rules stay competitive.** Training only the two lowest-energy PCs emits even less CO₂ at similar accuracy. Once each PC's energy is measured, a simple energy-ranked rule captures most of the saving; RL earns its place when conditions change over time, such as grid carbon intensity.
- **Limits:** client selection here is simulated with measured PC profiles, not run live across the four machines; PC1's reading is being re-verified and PC4 is assumed; CO₂ uses a fixed grid factor (0.675 kg/kWh, CEA FY 2025–26).
""")

nb = {"cells": cells, "metadata": {"colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).resolve().parent / "Pipeline_Demo_Colab.ipynb"
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
