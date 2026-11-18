"""Helpers shared by the local demo server and the Vercel serverless entrypoint. Stdlib only."""
from __future__ import annotations
import csv
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

BASE=Path(__file__).resolve().parent
DEFAULTS={'rounds':20,'episodes':130,'repeats':5,'sensitivity':70,'target':80,
          'local_epochs':2,'learning_rate':.23,'seed':42,'latency_weight':.018,'carbon_weight':.32}
FILES={'dashboard.html','run_parameters.json','evidence_provenance.json','comparison_SIMULATION.csv',
 'per_round_SIMULATION.csv','sensitivity_ASSUMPTIONS.csv','actual_meter_reading_PC1.csv',
 'node_benchmark_MEASURED_PLUS_SCENARIOS.csv','accuracy_by_round.png','cumulative_latency.png',
 'cumulative_carbon.png','tradeoff.png','node_energy_scenarios.png','run.log'}
MODELS=('numpy','torch_mlp','torch_cnn')

def validate_settings(raw):
    if not isinstance(raw,dict):raise ValueError('Settings must be a JSON object.')
    if set(raw)-set(DEFAULTS):raise ValueError('Unrecognized setting.')
    values={**DEFAULTS,**raw}
    bounds={'rounds':(1,100),'episodes':(1,500),'repeats':(1,20),'sensitivity':(1,100),
            'local_epochs':(1,20),'seed':(0,2**31-1)}
    for name,(low,high) in bounds.items():
        value=values[name]
        if isinstance(value,bool):raise ValueError(name+' must be a whole number.')
        if isinstance(value,str) and re.fullmatch(r'\d+',value):value=int(value)
        if not isinstance(value,int) or not low<=value<=high:raise ValueError(f'{name}: enter a whole number from {low} to {high}.')
        values[name]=value
    for name,low,high in [('target',0,100),('learning_rate',0,2),('latency_weight',0,100),('carbon_weight',0,100)]:
        value=values[name]
        if isinstance(value,bool):raise ValueError(name+' must be a number.')
        try:value=float(value)
        except (TypeError,ValueError):raise ValueError(name+' must be a number.') from None
        if not math.isfinite(value) or not low<=value<=high or (name in ('target','learning_rate') and value==0):
            raise ValueError(name+' is outside its supported range.')
        values[name]=value
    return values

def command_for(values,out):
    cmd=[sys.executable,'-u',str(BASE/'run_project.py')]
    for name in ('rounds','episodes','repeats','sensitivity','local_epochs','learning_rate','seed','latency_weight','carbon_weight'):
        cmd.extend(['--'+name.replace('_','-'),str(values[name])])
    return cmd+['--target',str(values['target']/100),'--output',str(out)]

def result_dir(result_id,tmp_root=None):
    if result_id=='baseline':return BASE/'results'
    if not re.fullmatch('[a-f0-9]{12}',result_id):raise ValueError('Invalid run identifier.')
    return (BASE/'runs' if tmp_root is None else Path(tmp_root))/result_id

def read_results_from(folder,result_id):
    def rows(name):
        with (folder/name).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))
    return {'summary':rows('comparison_SIMULATION.csv'),'rows':rows('per_round_SIMULATION.csv'),
        'sensitivity':rows('sensitivity_ASSUMPTIONS.csv'),'profiles':rows('node_benchmark_MEASURED_PLUS_SCENARIOS.csv'),
        'parameters':json.loads((folder/'run_parameters.json').read_text(encoding='utf-8')),
        'artifact_prefix':'/artifacts/'+result_id+'/','result_id':result_id,'offline':False}

def run_and_cache_pipeline(raw=None):
    cfg=raw or {}
    mode=cfg.get('mode','dirichlet')
    if mode not in ('iid','dirichlet'):raise ValueError('mode must be iid or dirichlet.')
    model=cfg.get('model','numpy')
    if model not in MODELS:raise ValueError('model must be numpy, torch_mlp or torch_cnn.')
    if model.startswith('torch') and importlib.util.find_spec('torch') is None:
        raise ValueError('PyTorch is not installed on this server; install requirements-optional.txt to use torch models.')
    alpha=float(cfg.get('alpha',0.5))
    rounds=int(cfg.get('rounds',15))
    q_ep=int(cfg.get('q_episodes',35))
    m_ep=int(cfg.get('morl_episodes',20))
    use_cal=bool(cfg.get('use_calibrated_telemetry',True))
    seed=int(cfg.get('seed',42))
    if not (0.01<=alpha<=100 and 1<=rounds<=50 and 1<=q_ep<=200 and 1<=m_ep<=100):
        raise ValueError('Pipeline parameter out of range.')
    from fl_pipeline.experiment import run_pipeline_experiment
    res=run_pipeline_experiment(mode=mode,alpha=alpha,rounds=rounds,q_episodes=q_ep,
                                morl_episodes=m_ep,use_calibrated_telemetry=use_cal,seed=seed,model=model)
    res['policies']=[asdict(p) for p in res['policies']]
    return res

def prune_runs(runs_dir,keep=None):
    if keep is None:
        # Opt-in: deleting old local runs needs FCL_KEEP_RUNS set explicitly.
        try:keep=int(os.environ['FCL_KEEP_RUNS'])
        except (KeyError,ValueError):return []
    keep=max(1,keep)
    runs_dir=Path(runs_dir)
    if not runs_dir.is_dir():return []
    folders=[p for p in runs_dir.iterdir() if p.is_dir() and re.fullmatch('[a-f0-9]{12}',p.name)]
    folders.sort(key=lambda p:p.stat().st_mtime,reverse=True)
    protected=set()
    try:protected.add(json.loads((runs_dir/'latest.json').read_text(encoding='utf-8'))['result_id'])
    except (OSError,ValueError,KeyError,TypeError):pass
    removed=[]
    for index,folder in enumerate(folders):
        if index<keep or folder.name in protected:continue
        shutil.rmtree(folder,ignore_errors=True);removed.append(folder.name)
    return removed
