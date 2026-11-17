"""Local browser console backed by the real Python simulation. No web dependencies."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import uuid
import webbrowser
import zipfile

BASE=Path(__file__).resolve().parent
DEFAULTS={'rounds':20,'episodes':130,'repeats':5,'sensitivity':70,'target':80,
          'local_epochs':2,'learning_rate':.23,'seed':42,'latency_weight':.018,'carbon_weight':.32}
FILES={'dashboard.html','run_parameters.json','evidence_provenance.json','comparison_SIMULATION.csv',
 'per_round_SIMULATION.csv','sensitivity_ASSUMPTIONS.csv','actual_meter_reading_PC1.csv',
 'node_benchmark_MEASURED_PLUS_SCENARIOS.csv','accuracy_by_round.png','cumulative_latency.png',
 'cumulative_carbon.png','tradeoff.png','node_energy_scenarios.png','run.log'}
LOCK=threading.Lock()
STATE={'status':'idle','progress':0,'message':'Ready to run','run_id':None,'result_id':'baseline','log':[],
       'cancel_requested':False,'process':None}
PIPELINE_CACHE=None
PIPELINE_LOCK=threading.Lock()

def run_and_cache_pipeline(raw=None):
    from fl_pipeline.experiment import run_pipeline_experiment
    cfg=raw or {}
    mode=cfg.get('mode','dirichlet')
    if mode not in ('iid','dirichlet'):raise ValueError('mode must be iid or dirichlet.')
    alpha=float(cfg.get('alpha',0.5))
    rounds=int(cfg.get('rounds',15))
    q_ep=int(cfg.get('q_episodes',35))
    m_ep=int(cfg.get('morl_episodes',20))
    use_cal=bool(cfg.get('use_calibrated_telemetry',True))
    seed=int(cfg.get('seed',42))
    if not (0.01<=alpha<=100 and 1<=rounds<=50 and 1<=q_ep<=200 and 1<=m_ep<=100):
        raise ValueError('Pipeline parameter out of range.')
    res=run_pipeline_experiment(mode=mode,alpha=alpha,rounds=rounds,q_episodes=q_ep,
                                morl_episodes=m_ep,use_calibrated_telemetry=use_cal,seed=seed)
    res['policies']=[asdict(p) for p in res['policies']]
    return res

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

def result_dir(result_id):
    if result_id=='baseline':return BASE/'results'
    if not re.fullmatch('[a-f0-9]{12}',result_id):raise ValueError('Invalid run identifier.')
    return BASE/'runs'/result_id

def read_results(result_id):
    folder=result_dir(result_id)
    def rows(name):
        with (folder/name).open(encoding='utf-8-sig',newline='') as file:return list(csv.DictReader(file))
    return {'summary':rows('comparison_SIMULATION.csv'),'rows':rows('per_round_SIMULATION.csv'),
        'sensitivity':rows('sensitivity_ASSUMPTIONS.csv'),'profiles':rows('node_benchmark_MEASURED_PLUS_SCENARIOS.csv'),
        'parameters':json.loads((folder/'run_parameters.json').read_text(encoding='utf-8')),
        'artifact_prefix':'/artifacts/'+result_id+'/','result_id':result_id,'offline':False}

def run_worker(values,run_id):
    folder=result_dir(run_id);folder.mkdir(parents=True)
    process=None
    try:
        env=os.environ.copy();env.update({'MPLCONFIGDIR':str(BASE/'.matplotlib'),'PYTHONIOENCODING':'utf-8'})
        with LOCK:
            if STATE['cancel_requested']:
                STATE.update(status='cancelled',message='Run stopped');return
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        process=subprocess.Popen(command_for(values,folder),cwd=BASE,env=env,stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=flags)
        with LOCK:
            STATE['process']=process
            if STATE['cancel_requested']:process.terminate()
        with process.stdout as stream, (folder/'run.log').open('w',encoding='utf-8') as log:
            for line in stream:
                log.write(line);log.flush()
                with LOCK:
                    STATE['log']=(STATE['log']+[line.rstrip()])[-500:]
                    match=re.match(r'PROGRESS (\d+) (.*)',line)
                    if match:STATE.update(progress=int(match[1]),message=match[2])
        code=process.wait()
        with LOCK:
            if STATE['cancel_requested']:STATE.update(status='cancelled',message='Run stopped. Previous results retained.')
            elif code==0:
                read_results(run_id)
                STATE.update(status='complete',progress=100,message='Simulation complete',result_id=run_id)
                (BASE/'runs'/'latest.json').write_text(json.dumps({'result_id':run_id}),encoding='utf-8')
            else:STATE.update(status='failed',message=f'Simulation exited with code {code}. See the run log.')
    except Exception as exc:
        with LOCK:STATE.update(status='failed',message='Run failed: '+str(exc))
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        with LOCK:
            if STATE['run_id']==run_id:STATE['process']=None

def get_pipeline_cached():
    global PIPELINE_CACHE
    with PIPELINE_LOCK:
        if PIPELINE_CACHE is None:PIPELINE_CACHE=run_and_cache_pipeline()
        return PIPELINE_CACHE

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):pass
    def send(self,code,body,content_type='application/json; charset=utf-8',headers=None):
        if isinstance(body,(dict,list)):body=json.dumps(body).encode()
        elif isinstance(body,str):body=body.encode('utf-8')
        self.send_response(code);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        for name,value in (headers or {}).items():self.send_header(name,value)
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path)
        try:
            if path.path in ('/','/index.html','/overview','/pipeline','/replay','/specs'):
                html=(BASE/'web'/'index.html').read_text(encoding='utf-8').replace('__BOOTSTRAP__','null')
                return self.send(200,html,'text/html; charset=utf-8')
            if path.path=='/api/state':
                with LOCK:data={k:v for k,v in STATE.items() if k not in ('process','cancel_requested')}
                return self.send(200,data)
            if path.path=='/api/results':
                with LOCK:result_id=STATE['result_id']
                return self.send(200,read_results(result_id))
            if path.path=='/api/overview':
                with LOCK:result_id=STATE['result_id']
                res=read_results(result_id)
                return self.send(200,{'summary':res['summary'],'run':res['parameters']['run'],
                    'execution_wall_seconds':res['parameters']['execution_wall_seconds'],
                    'artifact_prefix':res['artifact_prefix'],'result_id':res['result_id']})
            if path.path=='/api/replay':
                with LOCK:result_id=STATE['result_id']
                res=read_results(result_id);q=parse_qs(path.query)
                rows=res['rows']
                if 'policy' in q:rows=[r for r in rows if r['policy']==q['policy'][0]]
                if 'seed' in q:rows=[r for r in rows if str(r['scenario_seed'])==q['seed'][0]]
                return self.send(200,{'rows':rows,'clients':res['parameters']['infrastructure']['clients'],
                    'rounds':res['parameters']['run']['rounds'],'grid_kg_co2_per_kwh':res['parameters']['infrastructure']['grid_kg_co2_per_kwh']})
            if path.path=='/api/specs':
                with LOCK:result_id=STATE['result_id']
                res=read_results(result_id)
                return self.send(200,{'parameters':res['parameters'],'sensitivity':res['sensitivity'],
                    'profiles':res['profiles'],'artifact_prefix':res['artifact_prefix']})
            if path.path=='/api/telemetry/summary':
                from fl_pipeline.experiment import load_calibrated_telemetry_csv
                _,summary=load_calibrated_telemetry_csv()
                return self.send(200,{'source':'data/calibrated_telemetry_10k.csv','nodes':summary})
            if path.path=='/api/pipeline':
                return self.send(200,get_pipeline_cached())
            if path.path=='/api/pipeline/layers':
                p=get_pipeline_cached()
                return self.send(200,{'config':p['config'],'clients':p['clients'],'pareto_policies':p['pareto_policies'],'pareto_morl':p['pareto_morl']})
            if path.path=='/api/pipeline/policies':
                p=get_pipeline_cached()
                return self.send(200,{'policies':p['policies'],'pareto_policies':p['pareto_policies']})
            if path.path=='/api/pipeline/morl':
                p=get_pipeline_cached()
                return self.send(200,{'morl_sweep':p['morl_sweep'],'pareto_morl':p['pareto_morl']})
            if path.path=='/api/export':
                result_id=parse_qs(path.query).get('run',['baseline'])[0];folder=result_dir(result_id)
                if not (folder/'run_parameters.json').exists():return self.send(404,{'error':'No completed run found.'})
                data=io.BytesIO()
                with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as archive:
                    for file in folder.iterdir():
                        if file.name in FILES and file.is_file():archive.write(file,file.name)
                return self.send(200,data.getvalue(),'application/zip',{'Content-Disposition':'attachment; filename="Simulation_Results.zip"'})
            parts=path.path.strip('/').split('/')
            if len(parts)==3 and parts[0]=='artifacts':
                folder=result_dir(parts[1]);name=parts[2]
                if name not in FILES:return self.send(404,{'error':'File not found.'})
                file=folder/name
                content_type='image/png' if name.endswith('.png') else 'text/html; charset=utf-8' if name.endswith('.html') else 'application/json' if name.endswith('.json') else 'text/plain; charset=utf-8'
                return self.send(200,file.read_bytes(),content_type)
            return self.send(404,{'error':'Route not found.'})
        except (ValueError,FileNotFoundError):return self.send(404,{'error':'Result not found.'})
        except BrokenPipeError:pass
    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
            return self.send(403,{'error':'This console accepts local requests only.'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send(415,{'error':'Use JSON settings.'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<12000:return self.send(400,{'error':'Invalid request size.'})
            raw=json.loads(self.rfile.read(length))
            if self.path=='/api/run':
                values=validate_settings(raw)
                with LOCK:
                    if STATE['status']=='running':return self.send(409,{'error':'A simulation is already running.'})
                    run_id=uuid.uuid4().hex[:12]
                    STATE.update(status='running',progress=0,message='Starting simulation',run_id=run_id,log=[],cancel_requested=False)
                threading.Thread(target=run_worker,args=(values,run_id),daemon=True).start()
                return self.send(202,{'run_id':run_id})
            if self.path=='/api/cancel':
                with LOCK:
                    STATE['cancel_requested']=True;process=STATE['process']
                    if process and process.poll() is None:process.terminate()
                return self.send(200,{'ok':True})
            if self.path in ('/api/pipeline','/api/pipeline/run'):
                global PIPELINE_CACHE
                with PIPELINE_LOCK:
                    PIPELINE_CACHE=run_and_cache_pipeline(raw)
                    return self.send(200,PIPELINE_CACHE)
            return self.send(404,{'error':'Route not found.'})
        except (ValueError,TypeError,json.JSONDecodeError) as exc:return self.send(400,{'error':str(exc)})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    latest=BASE/'runs'/'latest.json'
    if latest.exists():
        try:
            value=json.loads(latest.read_text())['result_id'];read_results(value);STATE['result_id']=value
        except (ValueError,KeyError,OSError):pass
    server=None
    for port in range(args.port,args.port+20):
        try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler);break
        except OSError:continue
    if server is None:raise SystemExit('No available local port. Close an older demo window and retry.')
    url=f'http://127.0.0.1:{server.server_port}/';print('Federated Computing Lab: '+url,flush=True)
    if not args.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        with LOCK:
            process=STATE['process']
            if process and process.poll() is None:process.terminate()
        server.server_close()

if __name__=='__main__':main()
