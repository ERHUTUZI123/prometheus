"""Fixed strict CPU policy checks then bounded CPU formula contract."""
import os,runpy,sys
from pathlib import Path
folder=Path(__file__).resolve().parent;out=Path(os.environ['SANDBOX_OUT'])
sys.argv=[str(folder/'compiler_policy_checks.py'),'--synthetic-denied-path','/workspace/prometheus-v2-m0-architecture/private-compiler-policy-fixtures/.compiler-policy-synthetic-denied.txt']
runpy.run_path(str(folder/'compiler_policy_checks.py'),run_name='__main__')
sys.argv=[str(folder/'check_continuous_fusion_cpu.py'),'--checkpoint','/opt/prometheus-parameter-continuous-fusion-cpu-contract-v1-inputs/checkpoint.pt','--out',str(out/'contract'),'--max-seconds','90']
runpy.run_path(str(folder/'check_continuous_fusion_cpu.py'),run_name='__main__')
