import pathlib,os,runpy,sys
folder=pathlib.Path(__file__).resolve().parent;out=pathlib.Path(os.environ['SANDBOX_OUT'])
sys.argv=[str(folder/'compiler_policy_checks.py'),'--synthetic-denied-path','/workspace/prometheus-v2-m0-architecture/private-compiler-policy-fixtures/.compiler-policy-synthetic-denied.txt']
runpy.run_path(str(folder/'compiler_policy_checks.py'),run_name='__main__')
sys.argv=[str(folder/'official_mamba2_smoke.py'),'--out',str(out/'smoke'),'--device','cuda:0','--max-seconds','60']
runpy.run_path(str(folder/'official_mamba2_smoke.py'),run_name='__main__')
