"""Strict-phase policy checks, then the prospective fixed official AR task."""
import os
from pathlib import Path
import runpy
import sys
folder=Path(__file__).resolve().parent
out=Path(os.environ['SANDBOX_OUT'])
sys.argv=[str(folder/'compiler_policy_checks.py'),'--synthetic-denied-path',
 '/workspace/prometheus-v2-m0-architecture/private-compiler-policy-fixtures/.compiler-policy-synthetic-denied.txt']
runpy.run_path(str(folder/'compiler_policy_checks.py'),run_name='__main__')
sys.argv=[str(folder/'train_char_official_mamba2_alltoken.py'),
 '--data','/opt/prometheus-parameter-char-alltoken-mamba2-300-v1-inputs/data',
 '--out',str(out/'learn'),'--arm','mamba2','--seed','0','--device','cuda:0',
 '--max-seconds','730','--resource-overlap-note',
 'ROOT records actual concurrency; compilation and priming charged; no isolated throughput claim']
runpy.run_path(str(folder/'train_char_official_mamba2_alltoken.py'),run_name='__main__')
