"""Run a real NumPy FedAvg + tabular Q-learning simulation on ONE laptop.

DATA HONESTY: only PC1 0.065 kWh for 190 minutes is a physical meter reading.
All PC2-4 times/power, networking, coordinator draw and simulated round energy
are assumed research scenarios; these are NOT measured values or validation.
Training, round latency/energy and carbon all run through the shared fl_pipeline engine.
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fl_pipeline import FederatedRLEnv, TabularQLearningScheduler, default_clients_from_config

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
GRID = float(CONFIG['grid_factor_kgco2_per_kwh'])
MEAS = CONFIG['actual_measurement']
NODES = CONFIG['nodes']
ASSUM = CONFIG['scenario_settings']
AVG_W_PC1 = MEAS['meter_energy_kwh'] * 1000 / (MEAS['meter_duration_minutes'] / 60)
POLICIES = ('FedAvg_all4', 'Fastest2', 'LowestModelEnergy2', 'Q_learning_RL')
NUM_CLASSES = 3
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}


def write_rows(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        dw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        dw.writeheader(); dw.writerows(rows)


def base_profiles():
    profiles = []
    for n in NODES:
        active_w = AVG_W_PC1 * n['active_power_ratio_to_PC1']
        runtime_min = MEAS['meter_duration_minutes'] / n['relative_speed']
        scenario_energy_kwh = active_w * (runtime_min / 60) / 1000
        node_grid = float(n.get('grid_kgco2_per_kwh', GRID))
        if n['name'] == MEAS['node'] and 'measured_energy_kwh' in n:
            prov = f"MEASURED: {n['measured_energy_kwh']} kWh over {n['measured_duration_min']:g} minutes; CO2 CALCULATED"
        elif n['status'].startswith('MEASURED'):
            prov = n['status'] + ' — PROFILE INPUT; ROUND ENERGY SIMULATED'
        else:
            prov = 'SYNTHETIC SCENARIO — NOT MEASURED'
        profiles.append({
            'node':n['name'], 'ram_gb':n['ram_gb'], 'gpu_vram_gb':n['gpu_vram_gb'],
            'cpu_or_gpu_model': n['processor'],
            'provenance': prov,
            'relative_speed_ASSUMED': n['relative_speed'],
            'power_ratio_ASSUMED':n['active_power_ratio_to_PC1'],
            'mean_power_W': round(active_w,3),
            'duration_min':round(runtime_min,2),
            'energy_kWh':round(scenario_energy_kwh,6),
            'estimated_grid_CO2_g': round(scenario_energy_kwh*node_grid*1000,3)
        })
    return profiles


def _draw(rng, cent, y):
    x = cent[y] + rng.normal(0, 1.3, size=(len(y), 2))
    return np.column_stack([x, np.ones(len(y))]), y


def make_data(seed=42):
    """Non-IID client partitions, then test, then validation (RL state/reward), all from one rng."""
    rng = np.random.default_rng(seed)
    cent = np.array([[-1.8,-1.3],[1.7,-.9],[0.1,1.8]])
    mixes = ([.80,.10,.10],[.10,.80,.10],[.10,.10,.80],[.34,.33,.33])
    parts = [_draw(rng, cent, rng.choice(NUM_CLASSES, size=250, p=mixes[i % len(mixes)])) for i in range(len(NODES))]
    test_y = np.repeat(np.arange(NUM_CLASSES), 500); rng.shuffle(test_y)
    test = _draw(rng, cent, test_y)
    val_y = np.repeat(np.arange(NUM_CLASSES), [167, 167, 166]); rng.shuffle(val_y)
    return parts, _draw(rng, cent, val_y), test


def build_env(cfg, seed):
    """One dataset (seed) with its own clients and engine env; fairness_weight=0 keeps the documented reward."""
    parts, val, test = make_data(seed)
    return FederatedRLEnv(
        default_clients_from_config(partitions=parts), test_data=test, val_data=val, max_rounds=cfg.rounds,
        local_epochs=cfg.local_epochs, lr=cfg.learning_rate,
        base_train_seconds=ASSUM['synthetic_pc1_train_seconds_per_round'],
        aggregation_seconds=ASSUM['aggregation_seconds_per_round'],
        coordinator_power_w=ASSUM['coordinator_power_w'], speed_jitter_fraction=ASSUM['speed_jitter_fraction'],
        latency_weight=cfg.latency_weight, carbon_weight=cfg.carbon_weight, fairness_weight=0.0, seed=seed)


def heuristic(pol, clients):
    n, base = len(clients), ASSUM['synthetic_pc1_train_seconds_per_round']
    if pol == 'FedAvg_all4': return tuple(range(n))
    if pol == 'Fastest2': key = lambda i: base / clients[i].relative_speed + clients[i].base_network_latency_s
    elif pol == 'LowestModelEnergy2': key = lambda i: clients[i].active_power_w * base / clients[i].relative_speed
    else: raise ValueError(pol)
    return tuple(sorted(sorted(range(n), key=key)[:2]))


def evaluate(pol, env, sched, trial_seed, power_scales=None, speed_scales=None):
    env.reset(seed=trial_seed)
    jit = np.random.default_rng(trial_seed); n = len(env.clients); j = env.speed_jitter_fraction
    rows = []
    for r in range(env.max_rounds):
        if pol == 'Q_learning_RL': chosen = sched.select(env.clients, round_idx=r, acc=env.current_acc, max_rounds=env.max_rounds)
        else: chosen = heuristic(pol, env.clients)
        mult = None if speed_scales is None else jit.uniform(1-j, 1+j, size=n) * np.asarray(speed_scales)
        _, _, _, info = env.step(chosen, speed_multipliers=mult, power_scales=power_scales)
        rows.append({
            'policy':pol,'scenario_seed':trial_seed,'round':r+1,
            'clients_selected':'+'.join(info['selected_names']),
            'test_accuracy_REAL_MODEL_SYNTHETIC_DATA':round(info['accuracy'],5),
            'round_seconds_MODELED':round(info['round_latency_s'],5),
            'round_Wh_MODELED':round(info['round_wh'],6),
            'round_CO2_g_MODELED':round(info['round_co2_g'],6),
            'cum_seconds_MODELED':round(info['cum_latency_s'],5),
            'cum_Wh_MODELED':round(info['cum_wh'],6),
            'cum_CO2_g_MODELED':round(info['cum_co2_g'],6),
            'val_accuracy_REAL_MODEL_SYNTHETIC_DATA':round(info['val_accuracy'],5)
        })
    return rows


def ci95(vals):
    v = np.asarray(vals, dtype=float); n = len(v); m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    half = T95.get(n-1, 2.0) * sd / math.sqrt(n) if n > 1 else 0.0
    return m, sd, m-half, m+half


def summarize(rows,rounds,target):
    out=[]
    for pol in POLICIES:
        seeds=sorted({r['scenario_seed'] for r in rows if r['policy']==pol})
        fins=[]; reached=[]
        for seed in seeds:
            rs=sorted((r for r in rows if r['policy']==pol and r['scenario_seed']==seed),key=lambda x:x['round'])
            fins.append(rs[-1]); matching=[r for r in rs if r['test_accuracy_REAL_MODEL_SYNTHETIC_DATA']>=target]
            reached.append(matching[0] if matching else None)
        def avg(key): return float(np.mean([r[key] for r in fins]))
        def std(key): return float(np.std([r[key] for r in fins],ddof=1)) if len(fins)>1 else 0.0
        _,acc_sd,acc_lo,acc_hi=ci95([r['test_accuracy_REAL_MODEL_SYNTHETIC_DATA'] for r in fins])
        success=[r for r in reached if r is not None]
        out.append({
            'policy':pol,'repeat_count':len(fins),'rounds':rounds,
            'final_accuracy_mean':round(avg('test_accuracy_REAL_MODEL_SYNTHETIC_DATA'),5),
            'mean_total_seconds_MODELED':round(avg('cum_seconds_MODELED'),3),
            'sd_total_seconds_MODELED':round(std('cum_seconds_MODELED'),3),
            'mean_total_Wh_MODELED':round(avg('cum_Wh_MODELED'),5),
            'mean_total_CO2_g_MODELED':round(avg('cum_CO2_g_MODELED'),5),
            'target_accuracy':target,'target_hit_rate':round(len(success)/len(reached),3),
            'mean_seconds_to_target_if_reached_MODELED':round(float(np.mean([r['cum_seconds_MODELED'] for r in success])),3) if success else '',
            'mean_CO2_g_to_target_if_reached_MODELED':round(float(np.mean([r['cum_CO2_g_MODELED'] for r in success])),4) if success else '',
            'evidence_label':'REAL NUMPY MODEL TRAINING; VIRTUAL CLIENTS; SIMULATED TIME/ENERGY; EMISSIONS ESTIMATED',
            'final_accuracy_std':round(acc_sd,5),'final_accuracy_ci95_low':round(acc_lo,5),'final_accuracy_ci95_high':round(acc_hi,5),
            'mean_total_CO2_g_std':round(std('cum_CO2_g_MODELED'),5)
        })
    return out


def _diff_stats(d):
    if not d: return {'mean_difference':None,'sd':None,'ci95_low':None,'ci95_high':None}
    m, sd, lo, hi = ci95(d)
    if len(d) < 2: return {'mean_difference':round(m,6),'sd':None,'ci95_low':None,'ci95_high':None}
    return {'mean_difference':round(m,6),'sd':round(sd,6),'ci95_low':round(lo,6),'ci95_high':round(hi,6)}


def paired_comparison(rows, base='FedAvg_all4', rl='Q_learning_RL'):
    last = {}
    for r in rows:
        k = (r['policy'], r['scenario_seed'])
        if k not in last or r['round'] > last[k]['round']: last[k] = r
    seeds = sorted({s for p_, s in last if p_ == rl} & {s for p_, s in last if p_ == base})
    out = {'comparison':f'{rl} minus {base}','n_pairs':len(seeds)}
    for name,key in [('final_accuracy','test_accuracy_REAL_MODEL_SYNTHETIC_DATA'),('total_seconds_MODELED','cum_seconds_MODELED'),('total_CO2_g_MODELED','cum_CO2_g_MODELED')]:
        out[name] = _diff_stats([last[(rl,s)][key] - last[(base,s)][key] for s in seeds])
    return out


def make_figures(rows, profiles, out):
    labels={'FedAvg_all4':'FedAvg · All 4','Fastest2':'Fastest 2','LowestModelEnergy2':'Lowest Energy 2','Q_learning_RL':'Q-Learning RL'}
    colors={'FedAvg_all4':'#1D4ED8','Fastest2':'#B45309','LowestModelEnergy2':'#0F766E','Q_learning_RL':'#D9381E'}
    plt.rcParams.update({
        'figure.facecolor':'#FAF8F3',
        'axes.facecolor':'#F5F2EB',
        'axes.edgecolor':'#121316',
        'axes.labelcolor':'#121316',
        'xtick.color':'#3A3832',
        'ytick.color':'#3A3832',
        'text.color':'#121316',
        'grid.color':'#D5CFC2',
        'font.size':10,
    })
    specs=[('test_accuracy_REAL_MODEL_SYNTHETIC_DATA','accuracy_by_round.png','Held-Out Accuracy by Federated Round','Classification Accuracy'),
           ('cum_seconds_MODELED','cumulative_latency.png','Modeled Synchronous End-to-End Latency','Cumulative Seconds (Modeled)'),
           ('cum_CO2_g_MODELED','cumulative_carbon.png','Modeled Cluster Carbon Emissions','Cumulative Grams CO₂ (Modeled)')]
    for col,filename,title,ytitle in specs:
        fig,ax=plt.subplots(figsize=(9,5))
        for pol in POLICIES:
            rs=sorted({r['round'] for r in rows if r['policy']==pol})
            yy=[np.mean([r[col] for r in rows if r['round']==rd and r['policy']==pol]) for rd in rs]
            lw=2.8 if pol=='Q_learning_RL' else 2.0
            ax.plot(rs,yy,marker='o',markersize=4,lw=lw,color=colors[pol],label=labels[pol])
        ax.set(title=title,xlabel='Federated Rounds',ylabel=ytitle)
        ax.grid(alpha=.55,linestyle='--'); ax.legend(fontsize=9,frameon=True,facecolor='#FAF8F3',edgecolor='#121316')
        fig.tight_layout(); fig.savefig(out/filename,dpi=150,facecolor='#FAF8F3'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for pol in POLICIES:
        records=sorted([r for r in rows if r['policy']==pol and r['scenario_seed']==min(x['scenario_seed'] for x in rows)],key=lambda x:x['round'])
        lw=2.8 if pol=='Q_learning_RL' else 2.0
        ax.plot([r['cum_CO2_g_MODELED'] for r in records],[r['test_accuracy_REAL_MODEL_SYNTHETIC_DATA'] for r in records],'-o',markersize=4,lw=lw,color=colors[pol],label=labels[pol])
    ax.set(xlabel='Modeled Cumulative CO₂ (g)',ylabel='Held-Out Accuracy',title='Accuracy–Carbon Trajectory Trade-Off (Single Trial Seed)\nModeled Cluster Scenarios Calibrated from PC1 Meter Anchor'); ax.legend(fontsize=9,frameon=True,facecolor='#FAF8F3',edgecolor='#121316');ax.grid(alpha=.55,linestyle='--')
    fig.tight_layout(); fig.savefig(out/'tradeoff.png',dpi=150,facecolor='#FAF8F3');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    names=[r['node'] for r in profiles]
    energ=[r['energy_kWh']*1000 for r in profiles]
    bar_colors=['#D9381E','#1D4ED8','#B45309','#0F766E']
    b=ax.bar(names,energ,color=bar_colors,edgecolor='#121316',linewidth=1.2)
    ax.set(ylabel='Comparable Benchmark Energy (Wh)',title='PC1 Physical Wall-Meter Anchor (65.0 Wh) vs. Three Modeled Client Scenarios\nPC2–PC4 Are Calibrated Research Scenarios')
    for bar,value in zip(b,energ):ax.text(bar.get_x()+bar.get_width()/2,value+1.2,f'{value:.1f} Wh',ha='center',fontweight='bold')
    ax.set_ylim(0,max(energ)*1.22); ax.grid(axis='y',alpha=.55,linestyle='--')
    fig.tight_layout();fig.savefig(out/'node_energy_scenarios.png',dpi=150,facecolor='#FAF8F3');plt.close(fig)


def sensitivity(env, sched, cfg):
    rng=np.random.default_rng(cfg.seed+849); n=len(env.clients)
    pu, su = ASSUM['power_uncertainty_fraction'], ASSUM['speed_uncertainty_fraction']
    results=[]
    for i in range(cfg.sensitivity):
        # power and speed independently uncertain; PC1 average-power anchor is held fixed.
        pw=np.ones(n); sp=np.ones(n)
        pw[1:]=rng.uniform(1-pu,1+pu,size=n-1)
        sp[1:]=rng.uniform(1-su,1+su,size=n-1)
        for pol in POLICIES:
            r=evaluate(pol,env,sched,10000+i,power_scales=pw,speed_scales=sp)[-1]
            results.append({'policy':pol,'sample':i,'total_seconds_MODELED':r['cum_seconds_MODELED'],
                            'total_CO2_g_MODELED':r['cum_CO2_g_MODELED']})
    summary=[]
    for pol in POLICIES:
        rr=[r for r in results if r['policy']==pol]
        summary.append({'policy':pol, 'scenario_samples':cfg.sensitivity,
                        'co2_g_p05_ASSUMED':round(float(np.quantile([r['total_CO2_g_MODELED'] for r in rr],.05)),4),
                        'co2_g_median_ASSUMED':round(float(np.median([r['total_CO2_g_MODELED'] for r in rr])),4),
                        'co2_g_p95_ASSUMED':round(float(np.quantile([r['total_CO2_g_MODELED'] for r in rr],.95)),4),
                        'seconds_p05_ASSUMED':round(float(np.quantile([r['total_seconds_MODELED'] for r in rr],.05)),2),
                        'seconds_p95_ASSUMED':round(float(np.quantile([r['total_seconds_MODELED'] for r in rr],.95)),2)})
    return summary


def main():
    p=argparse.ArgumentParser(description='Federated compute simulation')
    for name,default in [('rounds',20),('episodes',130),('repeats',5),('sensitivity',70),('local-epochs',2),('seed',42),('train-seeds',3)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('target',.8),('learning-rate',.23),('latency-weight',.018),('carbon-weight',.32)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--output',type=Path,default=BASE/'results')
    a=p.parse_args()
    if min(a.rounds,a.episodes,a.repeats,a.sensitivity,a.local_epochs,a.train_seeds)<1 or a.seed<0:p.error('Counts must be positive; seed must be nonnegative.')
    if not 0<a.target<=1 or not 0<a.learning_rate<=2:p.error('Invalid target or local learning rate.')
    if not all(math.isfinite(v) and v>=0 for v in (a.latency_weight,a.carbon_weight)):p.error('Reward weights must be finite and nonnegative.')
    cfg=SimpleNamespace(**vars(a))
    started=time.perf_counter();out=cfg.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def progress(n,msg):print(f'PROGRESS {n} {msg}',flush=True)
    progress(0,'Preparing client data')
    profiles=base_profiles();write_rows(out/'node_benchmark_MEASURED_PLUS_SCENARIOS.csv',profiles)
    write_rows(out/'actual_meter_reading_PC1.csv',[{'node':'PC1','energy_kWh_MEASURED':MEAS['meter_energy_kwh'],
        'duration_min_MEASURED':MEAS['meter_duration_minutes'],'average_meter_power_W_CALCULATED':AVG_W_PC1,
        'grid_factor_kg_CO2_per_kWh_CEA':GRID,'grid_CO2_g_CALCULATED':MEAS['meter_energy_kwh']*GRID*1000,
        'evidence':'User-reported meter input'}])
    train_envs=[build_env(cfg,cfg.seed+2000+k) for k in range(cfg.train_seeds)]
    env0=train_envs[0]
    sched=TabularQLearningScheduler(env0.actions,seed=cfg.seed+1984)
    def on_episode(n,total):
        if n%max(1,total//10)==0 or n==total: progress(int(50*n/total),f'Training scheduler: episode {n}/{total}')
    sched.train(env0,episodes=cfg.episodes,progress_callback=on_episode,envs=train_envs)
    rows=[]; eval_env=None
    for trial in range(cfg.repeats):
        s=cfg.seed+1158+trial
        env=build_env(cfg,s)  # each repeat: new dataset + jitter seed, shared by all policies (paired)
        if trial==0: eval_env=env
        for j,policy in enumerate(POLICIES):
            rows.extend(evaluate(policy,env,sched,s))
            progress(50+int(25*(trial*len(POLICIES)+j+1)/(cfg.repeats*len(POLICIES))),f'Evaluating {policy}: trial {trial+1}/{cfg.repeats}')
    write_rows(out/'per_round_SIMULATION.csv',rows)
    summary=summarize(rows,cfg.rounds,cfg.target);write_rows(out/'comparison_SIMULATION.csv',summary)
    progress(78,'Evaluating scenario sensitivity')
    sens=sensitivity(eval_env,sched,cfg)
    write_rows(out/'sensitivity_ASSUMPTIONS.csv',sens)
    progress(90,'Rendering charts')
    make_figures(rows,profiles,out)
    params={
      'run':{'rounds':cfg.rounds,'episodes':cfg.episodes,'repeats':cfg.repeats,'sensitivity':cfg.sensitivity,
             'target':cfg.target,'local_epochs':cfg.local_epochs,'learning_rate':cfg.learning_rate,
             'seed':cfg.seed,'latency_weight':cfg.latency_weight,'carbon_weight':cfg.carbon_weight,'train_seeds':cfg.train_seeds},
      'training':{'clients':len(NODES),'classes':3,'features':2,'bias':True,'model':'Softmax linear classifier',
        'aggregation':'Sample-weighted FedAvg','samples_per_client':250,'training_samples':250*len(NODES),
        'validation_samples':500,'test_samples':1500,
        'local_epochs':cfg.local_epochs,'learning_rate':cfg.learning_rate,'feature_noise_sd':1.3,
        'class_centres':[[-1.8,-1.3],[1.7,-.9],[.1,1.8]],
        'client_class_probabilities':[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.34,.33,.33]],
        'data':'Synthetic non-IID classification, regenerated for every repeat seed','execution':'CPU / NumPy'},
      'rl':{'algorithm':'Tabular Q-learning','alpha':.24,'gamma':.86,'epsilon_start':.8,'epsilon_min':.06,
        'epsilon_schedule':'max(0.06, 0.80 * (1 - episode / episodes))','actions':len(env0.actions),'clients_per_action':[2,3,4],
        'training_phases':4,'accuracy_bucket_edges':[.50,.68,.76,.81],
        'accuracy_gain_weight':15,'latency_weight':cfg.latency_weight,'carbon_weight':cfg.carbon_weight,
        'accuracy_shortfall_weight':.6,'reward_accuracy_floor':.78,'fairness_weight':0.0,
        'state_accuracy_source':'validation split (not test)',
        'reward':'15 * accuracy_gain - latency_weight * seconds - carbon_weight * CO2_grams - 0.6 * max(0, 0.78 - accuracy)',
        'target_note':'Comparison target controls target-hit reporting; the reward accuracy floor remains 0.78.'},
      'seeds':{'data':cfg.seed,'q_learning':cfg.seed+1984,'rl_training_data':[cfg.seed+2000+k for k in range(cfg.train_seeds)],
        'performance_trials':[cfg.seed+1158+i for i in range(cfg.repeats)],
        'performance_trial_data':'each repeat regenerates data with the trial seed',
        'sensitivity_profiles':cfg.seed+849,'sensitivity_performance_trials':[10000+i for i in range(cfg.sensitivity)]},
      'infrastructure':{'grid_kg_co2_per_kwh':GRID,'reference_energy_kwh':MEAS['meter_energy_kwh'],
        'reference_duration_min':MEAS['meter_duration_minutes'],'reference_mean_power_w':AVG_W_PC1,
        'settings':ASSUM,'clients':NODES,
        'latency_formula':'max(client work + client network delay) + aggregation delay',
        'energy_formula':'(sum(active_W * work_s + idle_W * remaining_s) + coordinator_W * round_s) / 3600 = Wh',
        'carbon_formula':'Wh * grid_kg_CO2_per_kWh = grams CO2'},
      'runtime':{'engine':'Python / NumPy (fl_pipeline shared engine)','python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__},
      'paired_rl_vs_fedavg_all4':paired_comparison(rows),
      'execution_wall_seconds':round(time.perf_counter()-started,3),
      'interpretation':'Synthetic client data. Infrastructure latency, energy and carbon are modeled.'}
    (out/'run_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    measured=[n['name'] for n in NODES if n['status'].startswith('MEASURED')]
    unmeasured=[n['name'] for n in NODES if not n['status'].startswith('MEASURED')]
    provenance={'reference_meter_input':MEAS,'reference_mean_power_w':AVG_W_PC1,'source':CONFIG['grid_factor_source'],
        'scenario_note':f"Nodes with user-reported measurements: {', '.join(measured)}. Unmeasured scenario nodes: {', '.join(unmeasured) or 'none'}. All round latency/energy is simulated; NumPy training uses the CPU.",
        'node_status':{n['name']:n['status'] for n in NODES},
        'interpretation':params['interpretation']}
    (out/'evidence_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    from report import write_report
    write_report(out,summary,sens,profiles,params)
    print('Policy comparison | modeled infrastructure metrics')
    for r in summary:print(f'{r["policy"]:23} accuracy={r["final_accuracy_mean"]:.3f}±{(r["final_accuracy_ci95_high"]-r["final_accuracy_ci95_low"])/2:.3f} latency={r["mean_total_seconds_MODELED"]:.1f}s CO2={r["mean_total_CO2_g_MODELED"]:.3f}g target_hits={r["target_hit_rate"]:.0%}')
    progress(100,'Simulation complete')

if __name__=='__main__':main()
