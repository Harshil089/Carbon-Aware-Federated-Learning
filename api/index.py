"""Vercel Serverless entrypoint for Federated Computing Lab."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import re
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import zipfile

BASE = Path(__file__).resolve().parent.parent
TMP_RUNS = Path('/tmp/fcl_runs')
# Quick-preset bounds: the hosted demo has a time limit, so full-size runs stay local.
HOSTED_LIMITS = {'rounds': 12, 'episodes': 35, 'repeats': 3, 'sensitivity': 12}
HOSTED_LIMIT_MESSAGE = (
    'Hosted runs are limited to rounds ≤ 12, episodes ≤ 35, repeats ≤ 3 and what-if samples ≤ 12 '
    'so they finish within the time limit. Run the project locally for full-size runs.'
)

LOCK = threading.Lock()
STATE = {
    'status': 'complete',
    'progress': 100,
    'message': 'Baseline results loaded (Ready to run)',
    'run_id': None,
    'result_id': 'baseline',
    'hosted_limits': HOSTED_LIMITS,
    'log': [
        'PROGRESS 0 Preparing client data',
        'PROGRESS 50 Training scheduler: episode 130/130',
        'PROGRESS 75 Evaluating Q_learning_RL: trial 5/5',
        'PROGRESS 78 Evaluating scenario sensitivity',
        'PROGRESS 90 Rendering charts',
        'PROGRESS 100 Simulation complete',
    ],
}
PIPELINE_CACHE = None
PIPELINE_LOCK = threading.Lock()


def result_dir(result_id: str) -> Path:
    if result_id == 'baseline':
        return BASE / 'results'
    if re.fullmatch(r'[a-f0-9]{12}', result_id):
        candidate = TMP_RUNS / result_id
        if (candidate / 'run_parameters.json').exists():
            return candidate
    return BASE / 'results'


def read_results(result_id: str):
    from server_common import read_results_from

    folder = result_dir(result_id)
    actual_id = 'baseline' if folder == (BASE / 'results') else result_id
    return read_results_from(folder, actual_id)


def get_pipeline_cached():
    from server_common import run_and_cache_pipeline

    global PIPELINE_CACHE
    with PIPELINE_LOCK:
        if PIPELINE_CACHE is None:
            PIPELINE_CACHE = run_and_cache_pipeline()
        return PIPELINE_CACHE


def execute_serverless_run(raw: dict) -> str:
    import sys
    import uuid
    import server_common as ds

    values = ds.validate_settings(raw)
    if any(values[name] > limit for name, limit in HOSTED_LIMITS.items()):
        raise ValueError(HOSTED_LIMIT_MESSAGE)
    run_id = uuid.uuid4().hex[:12]
    out = TMP_RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = '/tmp/.matplotlib'

    cmd = ds.command_for(values, out)
    import subprocess

    with LOCK:
        STATE.update(
            status='running',
            progress=5,
            message='Starting simulation…',
            run_id=run_id,
            log=['PROGRESS 0 Preparing client data'],
        )

    proc = subprocess.Popen(
        cmd,
        cwd=BASE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    lines = []
    if proc.stdout is not None:
        for line in proc.stdout:
            s = line.rstrip()
            if s:
                lines.append(s)
            m = re.match(r'PROGRESS (\d+) (.*)', s)
            if m:
                with LOCK:
                    STATE.update(progress=int(m[1]), message=m[2], log=lines[-500:])
    returncode = proc.wait(timeout=55)
    (out / 'run.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with LOCK:
        if returncode == 0 and (out / 'run_parameters.json').exists():
            STATE.update(
                status='complete',
                progress=100,
                message='Simulation complete',
                run_id=run_id,
                result_id=run_id,
                log=lines[-500:],
            )
            try:
                ds.prune_runs(TMP_RUNS,keep=20)
            except Exception:
                pass
        else:
            STATE.update(
                status='failed',
                message=f'Simulation exited with code {returncode}',
                log=lines[-500:],
            )
    return run_id


class handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send(self, code: int, body, content_type='application/json; charset=utf-8', headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode('utf-8')
        elif isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _resolve_request(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        req_path = parsed.path
        if req_path in ('/api/index.py', '/api/index') and '__route' in q:
            req_path = re.sub(r'/+', '/', q['__route'][0]) or '/'
        return req_path, q

    def do_GET(self):
        req_path, q = self._resolve_request()
        try:
            if req_path in ('/', '/index.html', '/overview', '/pipeline', '/replay', '/specs'):
                html = (BASE / 'web' / 'index.html').read_text(encoding='utf-8').replace('__BOOTSTRAP__', 'null')
                return self.send(200, html, 'text/html; charset=utf-8')
            if req_path == '/api/state':
                with LOCK:
                    data = dict(STATE)
                return self.send(200, data)
            if req_path == '/api/results':
                with LOCK:
                    rid = STATE['result_id']
                return self.send(200, read_results(rid))
            if req_path == '/api/overview':
                with LOCK:
                    rid = STATE['result_id']
                res = read_results(rid)
                return self.send(
                    200,
                    {
                        'summary': res['summary'],
                        'run': res['parameters']['run'],
                        'execution_wall_seconds': res['parameters']['execution_wall_seconds'],
                        'artifact_prefix': res['artifact_prefix'],
                        'result_id': res['result_id'],
                    },
                )
            if req_path == '/api/replay':
                with LOCK:
                    rid = STATE['result_id']
                res = read_results(rid)
                rows = res['rows']
                if 'policy' in q:
                    rows = [r for r in rows if r['policy'] == q['policy'][0]]
                if 'seed' in q:
                    rows = [r for r in rows if str(r['scenario_seed']) == q['seed'][0]]
                return self.send(
                    200,
                    {
                        'rows': rows,
                        'clients': res['parameters']['infrastructure']['clients'],
                        'rounds': res['parameters']['run']['rounds'],
                        'grid_kg_co2_per_kwh': res['parameters']['infrastructure']['grid_kg_co2_per_kwh'],
                    },
                )
            if req_path == '/api/specs':
                with LOCK:
                    rid = STATE['result_id']
                res = read_results(rid)
                return self.send(
                    200,
                    {
                        'parameters': res['parameters'],
                        'sensitivity': res['sensitivity'],
                        'profiles': res['profiles'],
                        'artifact_prefix': res['artifact_prefix'],
                    },
                )
            if req_path == '/api/telemetry/summary':
                from fl_pipeline.experiment import load_calibrated_telemetry_csv

                _, summary = load_calibrated_telemetry_csv()
                return self.send(200, {'source': 'data/calibrated_telemetry_10k.csv', 'nodes': summary})
            if req_path == '/api/pipeline':
                return self.send(200, get_pipeline_cached())
            if req_path == '/api/pipeline/layers':
                p = get_pipeline_cached()
                return self.send(
                    200,
                    {
                        'config': p['config'],
                        'clients': p['clients'],
                        'pareto_policies': p['pareto_policies'],
                        'pareto_morl': p['pareto_morl'],
                    },
                )
            if req_path == '/api/pipeline/policies':
                p = get_pipeline_cached()
                return self.send(200, {'policies': p['policies'], 'pareto_policies': p['pareto_policies']})
            if req_path == '/api/pipeline/morl':
                p = get_pipeline_cached()
                return self.send(200, {'morl_sweep': p['morl_sweep'], 'pareto_morl': p['pareto_morl']})
            if req_path == '/api/export':
                from server_common import FILES

                rid = q.get('run', ['baseline'])[0]
                folder = result_dir(rid)
                if not (folder / 'run_parameters.json').exists():
                    return self.send(404, {'error': 'No completed run found.'})
                data = io.BytesIO()
                with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for file in folder.iterdir():
                        if file.name in FILES and file.is_file():
                            archive.write(file, file.name)
                return self.send(
                    200,
                    data.getvalue(),
                    'application/zip',
                    {'Content-Disposition': 'attachment; filename="Simulation_Results.zip"'},
                )
            parts = req_path.strip('/').split('/')
            if len(parts) == 3 and parts[0] == 'artifacts':
                from server_common import FILES

                folder = result_dir(parts[1])
                name = parts[2]
                if name not in FILES:
                    return self.send(404, {'error': 'File not found.'})
                file = folder / name
                if not file.exists():
                    file = BASE / 'results' / name
                content_type = (
                    'image/png'
                    if name.endswith('.png')
                    else 'text/html; charset=utf-8'
                    if name.endswith('.html')
                    else 'application/json'
                    if name.endswith('.json')
                    else 'text/plain; charset=utf-8'
                )
                return self.send(200, file.read_bytes(), content_type)
            return self.send(404, {'error': 'Route not found.'})
        except (ValueError, FileNotFoundError):
            return self.send(404, {'error': 'Result not found.'})

    def do_POST(self):
        req_path, _ = self._resolve_request()
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            return self.send(415, {'error': 'Use JSON settings.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length < 12000:
                return self.send(400, {'error': 'Invalid request size.'})
            raw = json.loads(self.rfile.read(length))
            if req_path == '/api/run':
                run_id = execute_serverless_run(raw)
                return self.send(202, {'run_id': run_id})
            if req_path == '/api/cancel':
                with LOCK:
                    STATE.update(status='cancelled', message='Run stopped. Previous results retained.')
                return self.send(200, {'ok': True})
            if req_path in ('/api/pipeline', '/api/pipeline/run'):
                from server_common import run_and_cache_pipeline

                if isinstance(raw, dict) and raw.get('model', 'numpy') != 'numpy':
                    raise ValueError('Torch models are not available on the hosted demo.')
                global PIPELINE_CACHE
                with PIPELINE_LOCK:
                    PIPELINE_CACHE = run_and_cache_pipeline(raw)
                    return self.send(200, PIPELINE_CACHE)
            return self.send(404, {'error': 'Route not found.'})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self.send(400, {'error': str(exc)})
