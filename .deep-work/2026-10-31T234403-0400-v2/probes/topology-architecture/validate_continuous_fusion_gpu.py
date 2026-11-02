"""Strict 16 checks after GPU priming, then fixed-weight fusion audit."""
import os,runpy,sys
from pathlib import Path
folder=Path(__file__).resolve().parent;out=Path(os.environ['SANDBOX_OUT'])
sys.argv=[str(folder/'compiler_policy_checks.py'),'--synthetic-denied-path','/workspace/prometheus-v2-m0-architecture/private-compiler-policy-fixtures/.compiler-policy-synthetic-denied.txt']
runpy.run_path(str(folder/'compiler_policy_checks.py'),run_name='__main__')
sys.argv=[str(folder/'audit_continuous_fusion_gpu.py'),'--inputs','/opt/prometheus-parameter-continuous-fusion-gpu-audit-v1-inputs','--out',str(out/'audit'),'--max-seconds','730']
runpy.run_path(str(folder/'audit_continuous_fusion_gpu.py'),run_name='__main__')
