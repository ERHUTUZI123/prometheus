"""Trusted bounded real-feature development launcher; world stays offline."""
import argparse,ast,base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--dataset',choices=['speech','cifar'],required=True);p.add_argument('--arm',choices=['block_dynamic','block_static','block_detached','gru','transformer','cnn'],default='block_dynamic');p.add_argument('--revision',choices=['v1','v2'],default='v1');p.add_argument('--token-width',type=int,choices=[16,64],default=16);a=p.parse_args()
 if a.token_width!=16 and (a.prepare or a.arm not in ('block_dynamic','block_static','block_detached')):raise ValueError('token64 is only an interface-width/capacity ablation for Block training; baselines/preparation cannot request it')
 name=f'parameter-real-{a.dataset}-data-v1' if a.prepare else f'parameter-real-{a.dataset}-{a.arm}-300-'+a.revision
 if not a.prepare and a.token_width==64:name+='-token64'
 if any(path.exists() for path in (ROOT/'versions'/('code-'+name),ROOT/'runs'/name,ROOT/(name+'-protocol.json'),ROOT/(name+'-coordinator-receipt.json'))):raise FileExistsError('new run namespace already exists; previous evidence is read-only')
 archive_dir=ROOT/'versions'/'real-interface-width-beforechange-v1'
 files=['real_feature_development.py','dev_dataset_readers.py']+([] if a.prepare else ['continuous_primitive_block.py']);version=ROOT/'versions'/('code-'+name);version.mkdir(exist_ok=False);encoded={};hashes={};origins={}
 for n in files:
  origin=archive_dir/n if a.token_width==64 and n=='continuous_primitive_block.py' else ROOT/n
  d=origin.read_bytes();ast.parse(d);(version/n).write_bytes(d);encoded[n]=base64.b64encode(d).decode();hashes[n]=hashlib.sha256(d).hexdigest();origins[n]=str(origin)
  if a.token_width==64 and n=='continuous_primitive_block.py' and hashes[n]!='f67e448dd393a8f7c7d0e988bab3812cbae21e0ff7b4ffe7691f87213f36b13f':raise ValueError('token64 must freeze original beforechange f67core; no shared API revision substituted')
 protocol={'before_execution':True,'name':name,'scope':'first10 officialclasses REAL-DEVELOPMENT only; not full benchmark or frozenPrimary','args':vars(a),'source_sha256':hashes,'TRAIN_cap_per_class':128,'VALID_cap_per_class':64,'TEST_cap_per_class':128,'output_head':35 if a.dataset=='speech' else 100,'updates':0 if a.prepare else 300,'batch_size':64,'optimizer':'Adam.003','features':'same shared TRAIN-normalized logmel40 or64 RGBpatches48 for ALLmodels; no taskID/teacherroute','evaluation':'TRAIN/VALIDcurves, TESTafterfinaldiskfreeze; actualNNreload; noaccuracyvaliditygate','costs':'allP,datacache/inputadapter/memory, actualtraining/inference/proxyMAC, preprocessing separately; concurrentwallnotisolatedranking','outside_seconds':120 if a.prepare else 900,'inside_seconds':100 if a.prepare else 850}
 if not a.prepare and a.arm.startswith('block_'):
  protocol['interface_width_ablation']={'token_width':a.token_width,'label':'interface_width_and_capacity' if a.token_width==64 else 'default_interface16','changes':'inputprojection output AND core token_dim; caller/expert/read/write input dimensions and P/MAC all change','fixed':'H64,rank8,K4,memory8,key16,slot1,300x64 batches,Adam.003,clip1,same feature cache','rationale':'architectural rank16 constraint plus existing TRAIN/VALID; static16 Speech success rules out rank as sole dynamic/static explanation','causal_scope':'not a pure frontend-rank causal test; no teacher operation/call','selection':'prospective bounded interface-width diagnostic; no TEST-based hyperparameter selection'}
 protocol['source_origins']=origins
 archive=archive_dir/'manifest.json'
 protocol['beforechange_source_archive']={'path':str(archive.parent),'manifest_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
 if not a.prepare and a.arm=='block_detached':
  protocol['credit_ablation']={'hypothesis':'recurrent CALL input Jacobian may destabilize credit; not yet identified cause','rule':'CALL sees actual h/read values; detach ONLY h/read at CALL inputs; x and caller/expert/read/write gradients remain','parameters':'identical to dynamic/static at SAME token-width; token64 has moreP than16','feedback_budget':'same cached examples/common minibatches/300updates; no teacher route or data change','interpretation':'targeted optimizer-credit ablation; not a bugfix or topology-specific advantage'}
 with (ROOT/(name+'-protocol.json')).open('x') as stream:stream.write(json.dumps(protocol,indent=2)+'\n')
 cfg={'name':name,'files':encoded,'hashes':hashes,'prepare':a.prepare,'dataset':a.dataset,'arm':a.arm,'token_width':a.token_width,'remote':REMOTE}
 code='''import pathlib,base64,hashlib,json,subprocess,sys,time,shutil
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name
if code.exists() or out.exists():raise FileExistsError('remote code/output namespace exists; no overwrite')
code.mkdir()
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
 cmd=[sys.executable,str(runner),'--script',str(code/'real_feature_development.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','train','--data',str(inputs/'data'),'--out',str(out/'learn'),'--arm',c['arm'],'--token-width',str(c['token_width']),'--seed','0','--device','cuda:0','--max-seconds','850']
print(json.dumps({'event':'frozen_real_launch','command':cmd}),flush=True);start=time.monotonic();x=subprocess.run(cmd,capture_output=True,text=True,timeout=930);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-start}
if out.exists():(out/'coordinator-exit.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt));raise SystemExit(x.returncode)
'''.replace('CONFIG',repr(cfg),1);ast.parse(code)
 try:
  x=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=960);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
 except subprocess.TimeoutExpired as e:receipt={'returncode':None,'error':'SSHwaittimeout, inspect actualremote checkpoint before assuming incomplete','stdout':str(e.stdout),'stderr':str(e.stderr)}
 with (ROOT/(name+'-coordinator-receipt.json')).open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
if __name__=='__main__':main()
