"""Input, execution, and reproducibility checks for the demonstration interface."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import threading
import time
import urllib.request
import urllib.error
import zipfile
import io
import demo_server as d

class SettingsTests(unittest.TestCase):
    def test_rejects_invalid_values(self):
        for raw in [{'rounds':True},{'rounds':0},{'rounds':1.5},{'seed':-1},{'target':float('nan')},
                    {'learning_rate':0},{'carbon_weight':float('inf')},{'target':101},{'command':'anything'}]:
            with self.subTest(raw=raw),self.assertRaises(ValueError):d.validate_settings(raw)
    def test_reporting_target_conversion(self):
        values=d.validate_settings({'target':87.5,'rounds':'12','carbon_weight':0})
        command=d.command_for(values,Path('example'))
        self.assertEqual(command[command.index('--target')+1],'0.875')
        self.assertEqual(command[command.index('--rounds')+1],'12')
        self.assertEqual(command[command.index('--carbon-weight')+1],'0.0')
    def test_run_path_cannot_escape(self):
        for value in ('../config.json','baseline/../','a'*13,'-example'):
            with self.subTest(value=value),self.assertRaises(ValueError):d.result_dir(value)

class ExecutionTests(unittest.TestCase):
    def test_custom_settings_reach_engine_and_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            values=d.validate_settings({'rounds':3,'episodes':4,'repeats':2,'sensitivity':2,'target':75,
                'local_epochs':3,'learning_rate':.11,'seed':7,'latency_weight':.07,'carbon_weight':.9})
            execution=subprocess.run(d.command_for(values,folder),cwd=d.BASE,capture_output=True,text=True,timeout=60)
            self.assertEqual(execution.returncode,0,execution.stderr)
            out=Path(folder);p=json.loads((out/'run_parameters.json').read_text())
            for key in values:
                self.assertEqual(p['run'][key],values[key]/100 if key=='target' else values[key])
            self.assertEqual(p['training']['local_epochs'],3)
            self.assertEqual(p['rl']['latency_weight'],.07)
            self.assertEqual(p['rl']['carbon_weight'],.9)
            self.assertEqual(p['seeds']['q_learning'],7+1984)
            with (out/'per_round_SIMULATION.csv').open(encoding='utf-8-sig') as file:rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),3*2*4)
            self.assertEqual({r['policy'] for r in rows},set(d.read_results('baseline')['summary'][i]['policy'] for i in range(4)))
            html=(out/'dashboard.html').read_text()
            self.assertNotIn('__BOOTSTRAP__',html)
            self.assertNotIn('19 NOVEMBER',html)
            self.assertNotIn('Evidence boundary',html)
            self.assertIn('PROGRESS 100 Simulation complete',execution.stdout)
            for name in ('accuracy_by_round.png','tradeoff.png','cumulative_carbon.png','cumulative_latency.png','node_energy_scenarios.png'):
                self.assertGreater((out/name).stat().st_size,1000)

class HttpTests(unittest.TestCase):
    def test_live_run_export_and_cancellation(self):
        with d.LOCK:d.STATE.update(status='idle',progress=0,message='Ready',run_id=None,result_id='baseline',log=[],cancel_requested=False,process=None)
        server=d.ThreadingHTTPServer(('127.0.0.1',0),d.Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        base='http://127.0.0.1:'+str(server.server_port)
        def get(path):
            with urllib.request.urlopen(base+path,timeout=5) as response:return response.read()
        def post(path,body,origin=None):
            headers={'Content-Type':'application/json'}
            if origin:headers['Origin']=origin
            request=urllib.request.Request(base+path,json.dumps(body).encode(),headers,method='POST')
            try:
                with urllib.request.urlopen(request,timeout=5) as response:return response.status,json.load(response)
            except urllib.error.HTTPError as response:return response.code,json.load(response)
        def finished():
            for _ in range(300):
                state=json.loads(get('/api/state'))
                if state['status']!='running':return state
                time.sleep(.1)
            self.fail('Simulation did not finish within the test timeout.')
        try:
            self.assertIn(b'Federated Computing Lab',get('/'))
            self.assertEqual(post('/api/run',{'rounds':True})[0],400)
            self.assertEqual(post('/api/run',{},'https://example.com')[0],403)
            code,job=post('/api/run',{'rounds':3,'episodes':4,'repeats':2,'sensitivity':2,'seed':7,'carbon_weight':.9})
            self.assertEqual(code,202)
            self.assertEqual(post('/api/run',{})[0],409)
            self.assertEqual(finished()['status'],'complete')
            results=json.loads(get('/api/results'))
            self.assertEqual(len(results['rows']),24)
            self.assertEqual(results['parameters']['rl']['carbon_weight'],.9)
            self.assertTrue(get(results['artifact_prefix']+'accuracy_by_round.png').startswith(b'\x89PNG'))
            with zipfile.ZipFile(io.BytesIO(get('/api/export?run='+job['run_id']))) as archive:
                self.assertTrue({'run_parameters.json','dashboard.html','run.log'}<=set(archive.namelist()))
            self.assertEqual(post('/api/run',{'rounds':100,'episodes':500})[0],202)
            self.assertEqual(post('/api/cancel',{})[0],200)
            self.assertEqual(finished()['status'],'cancelled')
        finally:server.shutdown();server.server_close()

if __name__=='__main__':unittest.main()
