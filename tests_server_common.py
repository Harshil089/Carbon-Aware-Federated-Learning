"""Run retention, path safety, pipeline validation and hosted caps for the two servers."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock
import server_common as sc

def make_run(root,name,mtime):
    folder=Path(root)/name;folder.mkdir(parents=True)
    os.utime(folder,(mtime,mtime));return folder

class PruneTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.names=[f'{i:012x}' for i in range(6)]
        now=time.time()
        for index,name in enumerate(self.names):make_run(self.root,name,now-index*100)  # index 0 is newest
    def tearDown(self):self.tmp.cleanup()
    def remaining(self):return sorted(p.name for p in self.root.iterdir() if p.is_dir())
    def test_keeps_newest_n_by_mtime(self):
        removed=sc.prune_runs(self.root,keep=3)
        self.assertEqual(self.remaining(),sorted(self.names[:3]))
        self.assertEqual(sorted(removed),sorted(self.names[3:]))
    def test_keeps_latest_json_target_even_if_old(self):
        (self.root/'latest.json').write_text(json.dumps({'result_id':self.names[5]}))
        sc.prune_runs(self.root,keep=2)
        self.assertEqual(self.remaining(),sorted([self.names[0],self.names[1],self.names[5]]))
    def test_ignores_non_run_entries(self):
        for name in ('results','notes'):make_run(self.root,name,0)
        (self.root/'abcdef123456').write_text('a file, not a run folder')
        sc.prune_runs(self.root,keep=1)
        self.assertEqual(self.remaining(),sorted([self.names[0],'notes','results']))
        self.assertTrue((self.root/'abcdef123456').is_file())
    def test_env_var_respected_and_unset_or_invalid_keeps_all(self):
        before=len(self.remaining())
        with mock.patch.dict(os.environ,{},clear=True):self.assertEqual(sc.prune_runs(self.root),[])
        with mock.patch.dict(os.environ,{'FCL_KEEP_RUNS':'banana'}):self.assertEqual(sc.prune_runs(self.root),[])
        self.assertEqual(len(self.remaining()),before)
        with mock.patch.dict(os.environ,{'FCL_KEEP_RUNS':'2'}):sc.prune_runs(self.root)
        self.assertEqual(len(self.remaining()),2)
    def test_missing_dir_is_noop(self):
        self.assertEqual(sc.prune_runs(self.root/'nope',keep=1),[])

class PathTests(unittest.TestCase):
    def test_strict_result_dir(self):
        self.assertEqual(sc.result_dir('baseline'),sc.BASE/'results')
        self.assertEqual(sc.result_dir('0'*12,Path('/tmp/x')),Path('/tmp/x')/('0'*12))
        for value in ('../config.json','baseline/../','a'*13,'-example','ABCDEF123456'):
            with self.subTest(value=value),self.assertRaises(ValueError):sc.result_dir(value)

class PipelineTests(unittest.TestCase):
    def test_rejects_unknown_model(self):
        with self.assertRaises(ValueError):sc.run_and_cache_pipeline({'model':'gpt'})
    def test_rejects_torch_when_missing(self):
        with mock.patch('importlib.util.find_spec',return_value=None),self.assertRaises(ValueError) as caught:
            sc.run_and_cache_pipeline({'model':'torch_mlp'})
        self.assertIn('PyTorch is not installed',str(caught.exception))
    def test_tiny_numpy_run(self):
        res=sc.run_and_cache_pipeline({'rounds':2,'q_episodes':1,'morl_episodes':1})
        self.assertEqual(res['config']['model'],'numpy')
        self.assertIn('data_provenance',res['config'])

class HostedCapTests(unittest.TestCase):
    def load_api(self):
        path=Path(sc.BASE)/'api'/'index.py'
        spec=importlib.util.spec_from_file_location('fcl_api_index',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    def test_over_limit_rejected_before_any_launch(self):
        api=self.load_api();runs=Path(tempfile.mkdtemp())
        with mock.patch('subprocess.Popen',side_effect=AssertionError('must not launch')),\
             mock.patch.object(api,'TMP_RUNS',runs):
            with self.assertRaises(ValueError) as caught:api.execute_serverless_run({'rounds':50})
            self.assertIn('Hosted runs',str(caught.exception))
            self.assertEqual(list(runs.iterdir()),[])
    def test_hosted_limits_constant(self):
        api=self.load_api()
        self.assertEqual(api.HOSTED_LIMITS,{'rounds':12,'episodes':35,'repeats':3,'sensitivity':12})

if __name__=='__main__':unittest.main()
