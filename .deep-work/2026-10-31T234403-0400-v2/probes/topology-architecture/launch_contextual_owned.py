"""Trusted bounded context-Block development coordinator; not a sandbox."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');p.add_argument('--arm',choices=['dynamic','static','static_expand','gru','transformer','ssm'],default='dynamic');p.add_argument('--states',type=int,default=8);p.add_argument('--seed',type=int,default=0);p.add_argument('--stages',type=int,default=1);p.add_argument('--steps',type=int,default=600);p.add_argument('--revision',choices=['v1','v2','v3'],default='v1');p.add_argument('--block-initialization',choices=['iid','carry'],default='iid');p.add_argument('--block-training',choices=['scheduled','soft_only'],default='scheduled');p.add_argument('--straight-through-backward',choices=['selected','dense'],default='selected');a=p.parse_args()
 if a.states not in (8,16,32) or a.stages not in (1,3) or not 1<=a.steps<=600:raise ValueError('finite development scope only')
 name=(f'parameter-context-data-n{a.states}-seed{a.seed}-v1' if a.data_only else f'parameter-context-n{a.states}-seed{a.seed}-{a.arm}-stage{a.stages}-'+a.revision)
 if not a.data_only and a.block_training!='scheduled':name+='-'+a.block_training
 if not a.data_only and a.straight_through_backward!='selected':name+='-'+a.straight_through_backward
 files=['contextual_task_data.py'] if a.data_only else ['train_contextual_primitives.py','contextual_primitive_block.py']
 if not a.data_only and a.arm=='ssm':files+=['selective_ssm_baseline.py']
 driver=files[0];version=ROOT/'versions'/('code-'+name);version.mkdir(exist_ok=False)
 encoded={};hashes={}
 for leaf in files:
  data=(ROOT/leaf).read_bytes();ast.parse(data,filename=leaf);(version/leaf).write_bytes(data);encoded[leaf]=base64.b64encode(data).decode();hashes[leaf]=hashlib.sha256(data).hexdigest()
 protocol={'before_execution':True,'scope':'developmental falsifiable context/working-memory acquisition, NOT Primary formal results','name':name,'args':vars(a),'source_sha256':hashes,'feedback':'identical initial state/raw tokens/terminal labels, no taskID/routes/intermediate or teacheroperations','shared_step_budget':a.steps,'proposal_config':'Block4slots/K4, learned general row transitions and GRU memory; soft200 thenST400 as predefined development setting, native separately','comparison':'dynamic; static sameP; static expanded2N; ordinaryGRU; real3layer4headTransformer64','budget':{'outside_seconds':120 if a.data_only else 900,'internal_seconds':60 if a.data_only else 850,'gpu_fraction':None if a.data_only else .3},'costs':'allP, optimizer/checkpoint/memory, actualupdatecalls/time, numericalwork/proxyFLOPs, inference; concurrentwallnotisolatedranking','training_runtime':'existing uid65534,Landlock4+,TSYNCseccomp,offline; authorizedGPUselftaskcommexception only','acceptance':'no accuracythreshold for baseline validity; all failures preserved; newcurrentarchitecture only after appropriate validation'}
 (ROOT/(name+'-protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
 cfg={'name':name,'files':encoded,'hashes':hashes,'driver':driver,'args':vars(a),'remote':REMOTE}
 code='''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;a=c['args'];r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);code.mkdir()
for n,b in c['files'].items():
 d=base64.b64decode(b);assert hashlib.sha256(d).hexdigest()==c['hashes'][n];p=code/n;p.write_bytes(d);p.chmod(0o444)
code.chmod(0o555);out=r/'outputs'/name
if a['data_only']:
 runner=r/'remote_runner_cpu_v2.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
 cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out/'data'),'--states',str(a['states']),'--seed',str(a['seed']),'--max-seconds','60']
else:
 source=r/'outputs'/('parameter-context-data-n'+str(a['states'])+'-seed'+str(a['seed'])+'-v1')/'data';inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs');inputs.mkdir();shutil.copytree(source,inputs/'data')
 for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
 inputs.chmod(0o555)
 runner=r/'remote_runner_gpu_authorized.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
 cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--data',str(inputs/'data'),'--out',str(out/'learn'),'--arm',a['arm'],'--states',str(a['states']),'--seed',str(a['seed']),'--stages',str(a['stages']),'--steps',str(a['steps']),'--slots','4','--block-initialization',a['block_initialization'],'--block-training',a['block_training'],'--straight-through-backward',a['straight_through_backward'],'--device','cuda:0','--max-seconds','850']
print(json.dumps({'event':'frozen_contextual_launch','command':cmd}),flush=True);beg=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=930);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-beg};print(x.stdout);print(x.stderr,file=sys.stderr)
except subprocess.TimeoutExpired as e:receipt={'returncode':None,'transport':'coordinator_timeout','wall_seconds':time.monotonic()-beg,'stdout':str(e.stdout),'stderr':str(e.stderr)}
if out.exists():(out/'coordinator-exit.json').write_text(json.dumps(receipt,indent=2)+'\\n')
raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''.replace('CONFIG',repr(cfg),1)
 ast.parse(code);started=time.monotonic()
 try:
  x=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=960);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-started}
 except subprocess.TimeoutExpired as e:receipt={'returncode':None,'error':'SSH timeout; remote state must be inspected, no rerun inferred','stdout':str(e.stdout),'stderr':str(e.stderr),'wall_seconds':time.monotonic()-started}
 (ROOT/(name+'-coordinator-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
if __name__=='__main__':main()
