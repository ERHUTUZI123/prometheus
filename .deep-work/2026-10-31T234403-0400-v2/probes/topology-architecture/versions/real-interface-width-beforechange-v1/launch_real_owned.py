"""Trusted bounded real-feature development launcher; world stays offline."""
import argparse,ast,base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--dataset',choices=['speech','cifar'],required=True);p.add_argument('--arm',choices=['block_dynamic','block_static','block_detached','gru','transformer','cnn'],default='block_dynamic');p.add_argument('--revision',choices=['v1','v2'],default='v1');a=p.parse_args()
 name=f'parameter-real-{a.dataset}-data-v1' if a.prepare else f'parameter-real-{a.dataset}-{a.arm}-300-'+a.revision
 files=['real_feature_development.py','dev_dataset_readers.py']+([] if a.prepare else ['continuous_primitive_block.py']);version=ROOT/'versions'/('code-'+name);version.mkdir(exist_ok=False);encoded={};hashes={}
 for n in files:
  d=(ROOT/n).read_bytes();ast.parse(d);(version/n).write_bytes(d);encoded[n]=base64.b64encode(d).decode();hashes[n]=hashlib.sha256(d).hexdigest()
 protocol={'before_execution':True,'name':name,'scope':'first10 officialclasses REAL-DEVELOPMENT only; not full benchmark or frozenPrimary','args':vars(a),'source_sha256':hashes,'TRAIN_cap_per_class':128,'VALID_cap_per_class':64,'TEST_cap_per_class':128,'output_head':35 if a.dataset=='speech' else 100,'updates':0 if a.prepare else 300,'batch_size':64,'optimizer':'Adam.003','features':'same shared TRAIN-normalized logmel40 or64 RGBpatches48 for ALLmodels; no taskID/teacherroute','evaluation':'TRAIN/VALIDcurves, TESTafterfinaldiskfreeze; actualNNreload; noaccuracyvaliditygate','costs':'allP,datacache/inputadapter/memory, actualtraining/inference/proxyMAC, preprocessing separately; concurrentwallnotisolatedranking','outside_seconds':120 if a.prepare else 900,'inside_seconds':100 if a.prepare else 850}
 if a.arm=='block_detached':
  protocol['credit_ablation']={'hypothesis':'recurrent CALL input Jacobian may destabilize credit; not yet identified cause','rule':'CALL sees actual h/read values; detach ONLY h/read at CALL inputs; x and caller/expert/read/write gradients remain','parameters':'identical to dynamic/static','feedback_budget':'same cached examples/common minibatches/300updates; no teacher route or data change','interpretation':'targeted optimizer-credit ablation; not a bugfix or topology-specific advantage'}
 (ROOT/(name+'-protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n');cfg={'name':name,'files':encoded,'hashes':hashes,'prepare':a.prepare,'dataset':a.dataset,'arm':a.arm,'remote':REMOTE}
 code='''import pathlib,base64,hashlib,json,subprocess,sys,time,shutil
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);code.mkdir()
for n,b in c['files'].items():
 d=base64.b64decode(b);assert hashlib.sha256(d).hexdigest()==c['hashes'][n];p=code/n;p.write_bytes(d);p.chmod(0o444)
code.chmod(0o555);out=r/'outputs'/name
if c['prepare']:
 runner=r/'remote_runner_cpu_v2.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
 cmd=[sys.executable,str(runner),'--script',str(code/'real_feature_development.py'),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','prepare','--dataset',c['dataset'],'--raw-root','/opt/prometheus-dev-data','--out',str(out/'data'),'--seed','0','--max-seconds','100']
else:
 inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs');inputs.mkdir();shutil.copytree(r/'outputs'/('parameter-real-'+c['dataset']+'-data-v1')/'data',inputs/'data')
 for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
 inputs.chmod(0o555);runner=r/'remote_runner_gpu_authorized.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
 cmd=[sys.executable,str(runner),'--script',str(code/'real_feature_development.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','train','--data',str(inputs/'data'),'--out',str(out/'learn'),'--arm',c['arm'],'--seed','0','--device','cuda:0','--max-seconds','850']
print(json.dumps({'event':'frozen_real_launch','command':cmd}),flush=True);start=time.monotonic();x=subprocess.run(cmd,capture_output=True,text=True,timeout=930);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-start}
if out.exists():(out/'coordinator-exit.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt));raise SystemExit(x.returncode)
'''.replace('CONFIG',repr(cfg),1);ast.parse(code)
 try:
  x=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=960);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
 except subprocess.TimeoutExpired as e:receipt={'returncode':None,'error':'SSHwaittimeout, inspect actualremote checkpoint before assuming incomplete','stdout':str(e.stdout),'stderr':str(e.stderr)}
 (ROOT/(name+'-coordinator-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
if __name__=='__main__':main()
