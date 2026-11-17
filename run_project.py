"""Run a real NumPy FedAvg + tabular Q-learning simulation on ONE laptop.

DATA HONESTY: only PC1 0.065 kWh for 190 minutes is a physical meter reading.
All PC2-4 times/power, networking, coordinator draw and simulated round energy
are assumed research scenarios; these are NOT measured values or validation.
"""
from __future__ import annotations
import argparse, csv, itertools, json, math, time, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
GRID = float(CONFIG['grid_factor_kgco2_per_kwh'])
MEAS = CONFIG['actual_measurement']
NODES = CONFIG['nodes']
ASSUM = CONFIG['scenario_settings']
AVG_W_PC1 = MEAS['meter_energy_kwh'] * 1000 / (MEAS['meter_duration_minutes'] / 60)
ACTIONS = [tuple(a) for k in (2,3,4) for a in itertools.combinations(range(4), k)]
POLICIES = ('FedAvg_all4', 'Fastest2', 'LowestModelEnergy2', 'Q_learning_RL')
NUM_CLASSES = 3
LOCAL_EPOCHS = 2
LOCAL_RATE = .23
LATENCY_WEIGHT = .018
CARBON_WEIGHT = .32


def write_rows(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        dw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        dw.writeheader(); dw.writerows(rows)


def base_profiles():
    duration_hr = MEAS['meter_duration_minutes']/60
    profiles = []
    for n in NODES:
        active_w = AVG_W_PC1 * n['active_power_ratio_to_PC1']
        runtime_min = MEAS['meter_duration_minutes'] / n['relative_speed']
        scenario_energy_kwh = active_w * (runtime_min / 60)/1000
        node_grid = float(n.get('grid_kgco2_per_kwh', GRID))
        profiles.append({
            'node':n['name'], 'ram_gb':n['ram_gb'], 'gpu_vram_gb':n['gpu_vram_gb'],
            'cpu_or_gpu_model': n['processor'],
            'provenance': 'MEASURED: 0.065 kWh over 190 minutes; CO2 CALCULATED' if n['name']=='PC1' else 'SYNTHETIC SCENARIO — NOT MEASURED',
            'relative_speed_ASSUMED': n['relative_speed'],
            'power_ratio_ASSUMED':n['active_power_ratio_to_PC1'],
            'mean_power_W': round(active_w,3),
            'duration_min':round(runtime_min,2),
            'energy_kWh':round(scenario_energy_kwh,6),
            'estimated_grid_CO2_g': round(scenario_energy_kwh*node_grid*1000,3)
        })
    return profiles


def make_data(seed=42):
    """Local client data are heterogeneous/non-IID, generated entirely on laptop."""
    rng=np.random.default_rng(seed)
    cent=np.array([[-1.8,-1.3],[1.7,-.9],[0.1,1.8]])
    mixes=([.80,.10,.10],[.10,.80,.10],[.10,.10,.80],[.34,.33,.33])
    parts=[]
    for mix in mixes:
        y=rng.choice(NUM_CLASSES,size=250,p=mix)
        x=cent[y]+rng.normal(0,1.3,size=(len(y),2))
        parts.append((np.column_stack([x,np.ones(len(y))]), y))
    test_y=np.repeat(np.arange(NUM_CLASSES),500)
    rng.shuffle(test_y)
    test_x=cent[test_y]+rng.normal(0,1.3,size=(len(test_y),2))
    return parts,(np.column_stack([test_x,np.ones(len(test_y))]),test_y)


def eval_acc(w, test):
    x,y=test
    return float(np.mean(np.argmax(x@w,axis=1)==y))


def train_local(w, part, epochs=None, lr=None):
    epochs=LOCAL_EPOCHS if epochs is None else epochs
    lr=LOCAL_RATE if lr is None else lr
    x,y=part; ww=w.copy(); onehot=np.eye(NUM_CLASSES)[y]
    for _ in range(epochs):
        z=x@ww; z-=z.max(axis=1,keepdims=True)
        p=np.exp(z); p/=p.sum(axis=1,keepdims=True)
        ww-=lr*(x.T@(p-onehot))/len(y)
    return ww


def fedavg(w, parts, selected):
    vals=[]; weights=[]
    for i in selected:
        vals.append(train_local(w,parts[i])); weights.append(len(parts[i][1]))
    return np.average(np.stack(vals),axis=0,weights=np.array(weights))


def round_model(selected, multiplier, power_scales=None, speed_scales=None, grid_factors=None):
    """Hypothetical PARALLEL round on 4 physical nodes + powered coordinator.

    During active work use assumed power. After finishing and if not selected,
    each machine consumes its assumed idle power until round end.
    """
    if power_scales is None: power_scales=np.ones(4)
    if speed_scales is None: speed_scales=np.ones(4)
    if grid_factors is None: grid_factors=[float(n.get('grid_kgco2_per_kwh', GRID)) for n in NODES]
    base_train=ASSUM['synthetic_pc1_train_seconds_per_round']
    exec_s={i:base_train*multiplier[i]*speed_scales[i]/NODES[i]['relative_speed'] for i in selected}
    round_s=max(exec_s[i]+NODES[i]['network_seconds_assumed'] for i in selected)+ASSUM['aggregation_seconds_per_round']
    coord_wh=(ASSUM['coordinator_power_w']*round_s)/3600.0
    total_wh=coord_wh
    total_co2_g=coord_wh*GRID
    for i,n in enumerate(NODES):
        work=exec_s.get(i,0.0)
        remaining=max(0.0,round_s-work)
        active_w=AVG_W_PC1*n['active_power_ratio_to_PC1']*power_scales[i]
        node_wh=(active_w*work+n['idle_power_w_assumed']*remaining)/3600.0
        total_wh+=node_wh
        total_co2_g+=node_wh*grid_factors[i]
    return round_s,total_wh,total_co2_g


def state(acc, rnd, total):
    phase=min(3,int(4*rnd/total))
    bucket=0 if acc<.50 else 1 if acc<.68 else 2 if acc<.76 else 3 if acc<.81 else 4
    return phase,bucket


def heuristic(pol):
    if pol=='FedAvg_all4': return (0,1,2,3)
    if pol=='Fastest2':
        return tuple(sorted(range(4),key=lambda i:ASSUM['synthetic_pc1_train_seconds_per_round']/NODES[i]['relative_speed']+NODES[i]['network_seconds_assumed'])[:2])
    if pol=='LowestModelEnergy2':
        # Simple heuristic, active-only, ignores data distribution and idle power.
        return tuple(sorted(range(4),key=lambda i:AVG_W_PC1*NODES[i]['active_power_ratio_to_PC1']*ASSUM['synthetic_pc1_train_seconds_per_round']/NODES[i]['relative_speed'])[:2])
    raise ValueError(pol)


def learn_q(parts,test,rounds,episodes,seed,progress=None):
    rng=np.random.default_rng(seed)
    q=defaultdict(lambda:np.zeros(len(ACTIONS)))
    for ep in range(episodes):
        w=np.zeros((3,NUM_CLASSES)); acc=eval_acc(w,test)
        eps=max(.06,.80*(1-ep/episodes))
        for r in range(rounds):
            s=state(acc,r,rounds)
            if rng.random()<eps: action=int(rng.integers(len(ACTIONS)))
            else: action=int(rng.choice(np.flatnonzero(q[s]==q[s].max())))
            chosen=ACTIONS[action]
            factors=rng.uniform(1-ASSUM['speed_jitter_fraction'],1+ASSUM['speed_jitter_fraction'],size=4)
            time, wh, co2=round_model(chosen,factors)
            wnew=fedavg(w,parts,chosen); anew=eval_acc(wnew,test)
            # Reward is a DESIGN CHOICE, not objectively validated weights.
            # Accuracy improvement and acceptable accuracy are encouraged.
            rew=15*(anew-acc)-LATENCY_WEIGHT*time-CARBON_WEIGHT*co2-.6*max(0,.78-anew)
            sp=state(anew,r+1,rounds)
            target=rew+(0 if r==rounds-1 else .86*q[sp].max())
            q[s][action]+=.24*(target-q[s][action])
            w,acc=wnew,anew
        if progress and ((ep+1)%max(1,episodes//10)==0 or ep+1==episodes):progress(ep+1,episodes)
    return q


def evaluate(pol,parts,test,q,rounds,seed,power_scales=None,speed_scales=None):
    rng=np.random.default_rng(seed)
    factors=rng.uniform(1-ASSUM['speed_jitter_fraction'],1+ASSUM['speed_jitter_fraction'],size=(rounds,4))
    w=np.zeros((3,NUM_CLASSES)); totals=np.zeros(3)
    rows=[]
    for r in range(rounds):
        prev=eval_acc(w,test)
        chosen=ACTIONS[int(np.argmax(q[state(prev,r,rounds)]))] if pol=='Q_learning_RL' else heuristic(pol)
        w=fedavg(w,parts,chosen)
        accuracy=eval_acc(w,test)
        t,wh,grams=round_model(chosen,factors[r],power_scales,speed_scales)
        totals+=np.array([t,wh,grams])
        rows.append({
            'policy':pol,'scenario_seed':seed,'round':r+1,
            'clients_selected':'+'.join(NODES[i]['name'] for i in chosen),
            'test_accuracy_REAL_MODEL_SYNTHETIC_DATA':round(accuracy,5),
            'round_seconds_MODELED':round(t,5),
            'round_Wh_MODELED':round(wh,6),
            'round_CO2_g_MODELED':round(grams,6),
            'cum_seconds_MODELED':round(totals[0],5),
            'cum_Wh_MODELED':round(totals[1],6),
            'cum_CO2_g_MODELED':round(totals[2],6)
        })
    return rows


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
            'evidence_label':'REAL NUMPY MODEL TRAINING; VIRTUAL CLIENTS; SIMULATED TIME/ENERGY; EMISSIONS ESTIMATED'
        })
    return out


def make_figures(rows, summary, out):
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
    names=[r['node'] for r in base_profiles()]
    energ=[r['energy_kWh']*1000 for r in base_profiles()]
    bar_colors=['#D9381E','#1D4ED8','#B45309','#0F766E']
    b=ax.bar(names,energ,color=bar_colors,edgecolor='#121316',linewidth=1.2)
    ax.set(ylabel='Comparable Benchmark Energy (Wh)',title='PC1 Physical Wall-Meter Anchor (65.0 Wh) vs. Three Modeled Client Scenarios\nPC2–PC4 Are Calibrated Research Scenarios')
    for bar,value in zip(b,energ):ax.text(bar.get_x()+bar.get_width()/2,value+1.2,f'{value:.1f} Wh',ha='center',fontweight='bold')
    ax.set_ylim(0,max(energ)*1.22); ax.grid(axis='y',alpha=.55,linestyle='--')
    fig.tight_layout();fig.savefig(out/'node_energy_scenarios.png',dpi=150,facecolor='#FAF8F3');plt.close(fig)


def sensitivity(parts,test,q,rounds,seed,rep=70):
    rng=np.random.default_rng(seed)
    results=[]
    for i in range(rep):
        # power and speed independently uncertain, and PC1 average-power anchor
        # is held fixed for comparability but not validated for FL rounds.
        pw=np.ones(4); sp=np.ones(4)
        pw[1:]=rng.uniform(1-ASSUM['power_uncertainty_fraction'],1+ASSUM['power_uncertainty_fraction'],size=3)
        sp[1:]=rng.uniform(1-ASSUM['speed_uncertainty_fraction'],1+ASSUM['speed_uncertainty_fraction'],size=3)
        for pol in POLICIES:
            r=evaluate(pol,parts,test,q,rounds,10000+i,power_scales=pw,speed_scales=sp)[-1]
            results.append({'policy':pol,'sample':i,'total_seconds_MODELED':r['cum_seconds_MODELED'],
                            'total_CO2_g_MODELED':r['cum_CO2_g_MODELED']})
    summary=[]
    for pol in POLICIES:
        rr=[r for r in results if r['policy']==pol]
        summary.append({'policy':pol, 'scenario_samples':rep,
                        'co2_g_p05_ASSUMED':round(float(np.quantile([r['total_CO2_g_MODELED'] for r in rr],.05)),4),
                        'co2_g_median_ASSUMED':round(float(np.median([r['total_CO2_g_MODELED'] for r in rr])),4),
                        'co2_g_p95_ASSUMED':round(float(np.quantile([r['total_CO2_g_MODELED'] for r in rr],.95)),4),
                        'seconds_p05_ASSUMED':round(float(np.quantile([r['total_seconds_MODELED'] for r in rr],.05)),2),
                        'seconds_p95_ASSUMED':round(float(np.quantile([r['total_seconds_MODELED'] for r in rr],.95)),2)})
    return summary


def main():
    global LOCAL_EPOCHS, LOCAL_RATE, LATENCY_WEIGHT, CARBON_WEIGHT
    p=argparse.ArgumentParser(description='Federated compute simulation')
    for name,default in [('rounds',20),('episodes',130),('repeats',5),('sensitivity',70),('local-epochs',2),('seed',42)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('target',.8),('learning-rate',.23),('latency-weight',.018),('carbon-weight',.32)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--output',type=Path,default=BASE/'results')
    a=p.parse_args()
    if min(a.rounds,a.episodes,a.repeats,a.sensitivity,a.local_epochs)<1 or a.seed<0:p.error('Counts must be positive; seed must be nonnegative.')
    if not 0<a.target<=1 or not 0<a.learning_rate<=2:p.error('Invalid target or local learning rate.')
    if not all(math.isfinite(v) and v>=0 for v in (a.latency_weight,a.carbon_weight)):p.error('Reward weights must be finite and nonnegative.')
    LOCAL_EPOCHS=a.local_epochs;LOCAL_RATE=a.learning_rate;LATENCY_WEIGHT=a.latency_weight;CARBON_WEIGHT=a.carbon_weight
    started=time.perf_counter();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def progress(n,msg):print(f'PROGRESS {n} {msg}',flush=True)
    progress(0,'Preparing client data')
    profiles=base_profiles();write_rows(out/'node_benchmark_MEASURED_PLUS_SCENARIOS.csv',profiles)
    write_rows(out/'actual_meter_reading_PC1.csv',[{'node':'PC1','energy_kWh_MEASURED':MEAS['meter_energy_kwh'],
        'duration_min_MEASURED':MEAS['meter_duration_minutes'],'average_meter_power_W_CALCULATED':AVG_W_PC1,
        'grid_factor_kg_CO2_per_kWh_CEA':GRID,'grid_CO2_g_CALCULATED':MEAS['meter_energy_kwh']*GRID*1000,
        'evidence':'User-reported meter input'}])
    parts,test=make_data(seed=a.seed)
    q=learn_q(parts,test,a.rounds,a.episodes,seed=a.seed+1984,
        progress=lambda n,total:progress(int(50*n/total),f'Training scheduler: episode {n}/{total}'))
    rows=[]
    for trial in range(a.repeats):
        for j,policy in enumerate(POLICIES):
            rows.extend(evaluate(policy,parts,test,q,a.rounds,a.seed+1158+trial))
            progress(50+int(25*(trial*4+j+1)/(a.repeats*4)),f'Evaluating {policy}: trial {trial+1}/{a.repeats}')
    write_rows(out/'per_round_SIMULATION.csv',rows)
    summary=summarize(rows,a.rounds,a.target);write_rows(out/'comparison_SIMULATION.csv',summary)
    progress(78,'Evaluating scenario sensitivity')
    sens=sensitivity(parts,test,q,a.rounds,seed=a.seed+849,rep=a.sensitivity)
    write_rows(out/'sensitivity_ASSUMPTIONS.csv',sens)
    progress(90,'Rendering charts')
    make_figures(rows,summary,out)
    params={
      'run':{'rounds':a.rounds,'episodes':a.episodes,'repeats':a.repeats,'sensitivity':a.sensitivity,
             'target':a.target,'local_epochs':a.local_epochs,'learning_rate':a.learning_rate,
             'seed':a.seed,'latency_weight':a.latency_weight,'carbon_weight':a.carbon_weight},
      'training':{'clients':4,'classes':3,'features':2,'bias':True,'model':'Softmax linear classifier',
        'aggregation':'Sample-weighted FedAvg','samples_per_client':250,'training_samples':1000,'test_samples':1500,
        'local_epochs':a.local_epochs,'learning_rate':a.learning_rate,'feature_noise_sd':1.3,
        'class_centres':[[-1.8,-1.3],[1.7,-.9],[.1,1.8]],
        'client_class_probabilities':[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.34,.33,.33]],
        'data':'Synthetic non-IID classification','execution':'CPU / NumPy'},
      'rl':{'algorithm':'Tabular Q-learning','alpha':.24,'gamma':.86,'epsilon_start':.8,'epsilon_min':.06,
        'epsilon_schedule':'max(0.06, 0.80 * (1 - episode / episodes))','actions':11,'clients_per_action':[2,3,4],
        'training_phases':4,'accuracy_bucket_edges':[.50,.68,.76,.81],
        'accuracy_gain_weight':15,'latency_weight':a.latency_weight,'carbon_weight':a.carbon_weight,
        'accuracy_shortfall_weight':.6,'reward_accuracy_floor':.78,
        'reward':'15 * accuracy_gain - latency_weight * seconds - carbon_weight * CO2_grams - 0.6 * max(0, 0.78 - accuracy)',
        'target_note':'Comparison target controls target-hit reporting; the reward accuracy floor remains 0.78.'},
      'seeds':{'data':a.seed,'q_learning':a.seed+1984,'performance_trials':[a.seed+1158+i for i in range(a.repeats)],
        'sensitivity_profiles':a.seed+849,'sensitivity_performance_trials':[10000+i for i in range(a.sensitivity)]},
      'infrastructure':{'grid_kg_co2_per_kwh':GRID,'reference_energy_kwh':MEAS['meter_energy_kwh'],
        'reference_duration_min':MEAS['meter_duration_minutes'],'reference_mean_power_w':AVG_W_PC1,
        'settings':ASSUM,'clients':NODES,
        'latency_formula':'max(client work + client network delay) + aggregation delay',
        'energy_formula':'(sum(active_W * work_s + idle_W * remaining_s) + coordinator_W * round_s) / 3600 = Wh',
        'carbon_formula':'Wh * grid_kg_CO2_per_kWh = grams CO2'},
      'runtime':{'engine':'Python / NumPy','python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__},
      'execution_wall_seconds':round(time.perf_counter()-started,3),
      'interpretation':'Synthetic client data. Infrastructure latency, energy and carbon are modeled.'}
    (out/'run_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    provenance={'reference_meter_input':MEAS,'reference_mean_power_w':AVG_W_PC1,'source':CONFIG['grid_factor_source'],
        'scenario_note':'PC2-4 speed and power are unvalidated scenario assumptions; NumPy training uses the CPU.',
        'interpretation':params['interpretation']}
    (out/'evidence_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    from report import write_report
    write_report(out,summary,sens,profiles,params)
    print('Policy comparison | modeled infrastructure metrics')
    for r in summary:print(f'{r["policy"]:23} accuracy={r["final_accuracy_mean"]:.3f} latency={r["mean_total_seconds_MODELED"]:.1f}s CO2={r["mean_total_CO2_g_MODELED"]:.3f}g target_hits={r["target_hit_rate"]:.0%}')
    progress(100,'Simulation complete')

if __name__=='__main__':main()
