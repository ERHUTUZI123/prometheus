"""Freeze/execute one fixed ordinary GRU32 four-stage baseline, never tune it."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
NAMES=dict(cpu='parameter-gru32-fourstage-cpu-v1',train='parameter-gru32-n8-seed0-fourstage-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
REMOTE_SCRIPT=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);code=r/('code-'+c['name']);out=r/'outputs'/c['name'];inputs=pathlib.Path('/opt/prometheus-'+c['name']+'-inputs')
status=out.parent/(out.name+'.runner-status.json');receipt=out.parent/(out.name+'.coordinator-exit.json')
if any(p.exists()or p.is_symlink()for p in(code,out,inputs,status,receipt)):raise FileExistsError('new namespace only; preserve all evidence')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if c['mode']=='train':
 proofpath=r/'outputs/parameter-gru32-fourstage-cpu-v1/check/result.json';rs_path=r/'outputs/parameter-gru32-fourstage-cpu-v1.runner-status.json'
 if not proofpath.is_file()or not rs_path.is_file():raise ValueError('actual same-source CPU shape/reload check required')
 proof=json.loads(proofpath.read_text());rs=json.loads(rs_path.read_text())
 if proof.get('status')!='CPU_GRU32_FOURSTAGE_SHAPE_RELOAD_PASS'or rs.get('status')!='completed'or proof.get('source_sha256')!=c['manifest']['source_sha256']:raise ValueError('CPU source/result mismatch')
code.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded,validate=True)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport mismatch')
 p=code/leaf
 with p.open('xb')as f:f.write(raw)
 p.chmod(0o444)
code.chmod(0o555)
if c['mode']=='cpu':
 runner=r/'remote_runner_cpu_v2.py'
 if sha(runner)!=c['CPU']:raise ValueError('unchanged CPU runner differs')
 cmd=[sys.executable,str(runner),'--script',str(code/'check_gru32_fourstage.py'),'--out',str(out),'--timeout','150','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out/'check')]
else:
 inputs.mkdir()
 for key,spec in c['manifest']['files'].items():
  source=r/'outputs'/spec['remote_relative'];target=inputs/spec['leaf']
  if source.is_symlink()or not source.is_file()or source.stat().st_size>100*1024*1024 or sha(source)!=spec['sha256']:raise ValueError('raw input mismatch: '+key)
  shutil.copyfile(source,target);target.chmod(0o444)
 p=inputs/'manifest.json'
 with p.open('x')as f:json.dump(c['manifest'],f,indent=2,sort_keys=True);f.write('\n')
 p.chmod(0o444);inputs.chmod(0o555);runner=r/'remote_runner_gpu_authorized.py'
 if sha(runner)!=c['GPU']:raise ValueError('unchanged GPU runner differs')
 cmd=[sys.executable,str(runner),'--script',str(code/'train_gru32_fourstage.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--inputs',str(inputs),'--out',str(out/'learn'),'--max-seconds','850']
started=time.monotonic()
try:
 x=subprocess.run(cmd,text=True,capture_output=True,timeout=180 if c['mode']=='cpu' else 930);rc=x.returncode;stdout=x.stdout;stderr=x.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or'');stderr=str(e.stderr or'');error='coordinator wait timeout; inspect existing bounded runner, no retry'
for suffix,text in(('coordinator-stdout.log',stdout),('coordinator-stderr.log',stderr)):
 with(out.parent/(out.name+'.'+suffix)).open('x')as f:f.write(text)
def load(p):return json.loads(p.read_text())if p.is_file()else None
rs=load(status);result=load(out/('check/result.json'if c['mode']=='cpu'else'learn/result.json'))
expected='CPU_GRU32_FOURSTAGE_SHAPE_RELOAD_PASS'if c['mode']=='cpu'else'ACTUAL_FOUR_STAGE_GRU32_DEVELOPMENT_COMPLETE'
passed=bool(rc==0 and rs and rs.get('status')=='completed'and result and result.get('status')==expected)
a=dict(name=c['name'],returncode=rc,protocol_completed=passed,error=error,whole_runner_wall_seconds=time.monotonic()-started,source_sha256=c['source_sha256'],command=cmd,root_status_path=str(status),result_path=str(out/('check/result.json'if c['mode']=='cpu'else'learn/result.json')),sandbox_stderr_path=str(out/'stderr.log'),actual_parameters=(result or{}).get('parameters',(result or{}).get('parameters_after')),actual_updates=(result or{}).get('actual_training_updates'),no_accuracy_gate=True)
with receipt.open('x')as f:json.dump(a,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(a,sort_keys=True));raise SystemExit(0 if passed else(rc if rc not in(None,0)else 1))
'''
def prepare(mode):
 name=NAMES[mode];version=ROOT/'versions'/('code-'+name);protocol=ROOT/(name+'-protocol.json')
 if version.exists()or protocol.exists()or(ROOT/(name+'-coordinator-receipt.json')).exists():raise FileExistsError('freeze only a new namespace')
 base=ROOT/'versions/code-parameter-factorized-value-anchor1-n8-seed0-fourstage-v3'
 common={'train_gru32_fourstage.py':(ROOT/'train_gru32_fourstage.py').read_bytes()}
 fixed={'owned_parent_driver.py':'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc','context_replay128.py':'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140','memory_live_stage_train.py':'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6'}
 for leaf,digest in fixed.items():
  path=base/leaf
  if sha(path)!=digest:raise ValueError('unmodified original source required: '+leaf)
  common[leaf]=path.read_bytes()
 common_hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in common.items()}
 sources=dict(common)
 if mode=='cpu':sources['check_gru32_fourstage.py']=(ROOT/'check_gru32_fourstage.py').read_bytes()
 for raw in sources.values():ast.parse(raw)
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 manifest=dict(schema='terminal-gru32-fourstage-inputs-v1',source_sha256=common_hashes,files={})
 if mode=='train':
  old=ROOT/'runs/parameter-context-data-n8-seed0-v1/data';new=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data';om=json.loads((old/'metadata.json').read_text());nm=json.loads((new/'metadata.json').read_text())
  manifest.update(old_metadata_sha256=sha(old/'metadata.json'),new_metadata_sha256=sha(new/'metadata.json'))
  for stage in range(4):
   for split in('train','valid','test_iid','test_long','test_unseen'):
    if stage<3:source=old/om['stages'][stage]['files'][split];digest=om['stages'][stage]['hashes'][split]
    else:entry=nm['files'][split];source=new/entry['file'];digest=entry['sha256']
    if split in('train','valid')and sha(source)!=digest:raise ValueError('original TRAIN/VALID hash changed')
    manifest['files'][f'{stage}:{split}']=dict(leaf=f'stage{stage}-{split}.npz',sha256=digest,remote_relative=str(source.relative_to(ROOT/'runs')))
 config=dict(name=name,mode=mode,remote=REMOTE,CPU=CPU,GPU=GPU,manifest=manifest,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()})
 remote=REMOTE_SCRIPT.replace('CONFIG',repr(config),1);ast.parse(remote);version.mkdir()
 for leaf,raw in sources.items():
  with(version/leaf).open('xb')as f:f.write(raw)
  (version/leaf).chmod(0o444)
 rp=version/'remote-launch.source';rp.write_text(remote);rp.chmod(0o444)
 put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
 doc=dict(name=name,before_execution=True,mode=mode,source_sha256=hashes,formal_Primary=False,model='unmodified cc1 TerminalGRU width32/layers2/dropout0',expected_P15=13992,expected_P16=14024,width_selection='parameter count only, no score-based selection',objective='scalar final terminal NLL/crossentropy only; no auxiliary/teacher objective',fresh_initial_NN=True,stages_rerun=0,stages=4,updates_per_stage=600,batch=64,seed=0,optimizer='Adam.003,clipnorm1',schedule='SeedSequence([0,8,stage,9027]),600x64; stage0 all64current, later first48current+16Replay128',data_manifest=manifest,maintenance='fsynced actual NN/Adam/RNG/replay/pending-indices checkpoint BEFORE boundary; only replay storage sees zero-copy128->48 view after allremovedzeros/length<=48; original training128 matrix remains unchanged',checkpoint_schema='terminal-gru32-fourstage-training-v1',test_policy='all2400 updates and all4stage checkpoint/reload freeze BEFORE any TEST load',no_accuracy_gate=True,no_extra_curriculum=True,no_tuning_or_retry=True,CPU_diagnostic='one synthetic CPU Adam step for shape/growth/reload/padding only; never reused as baseline init',outside_seconds=150 if mode=='cpu' else 900,inside_seconds=None if mode=='cpu' else 850,costs='actualP/Adam/RNG/replay/growth/checkpoints/reload/eval/storage plus trainingwall/CPU/CUDAstream; exacttrainingFLOPsunknown, forwardMACnotfullFLOPs',runner_sha256=CPU if mode=='cpu' else GPU,remote_launch_sha256=sha(rp),remote_launch_source=str(rp.relative_to(ROOT)))
 put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555)
 print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=name,source_sha256=hashes,remote_sha256=sha(rp),protocol_sha256=sha(protocol))))
def execute(mode):
 name=NAMES[mode];v=ROOT/'versions'/('code-'+name);config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(name+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']:raise ValueError('frozen remote source differs')
 for leaf,digest in config['source_sha256'].items():
  if sha(v/leaf)!=digest:raise ValueError('frozen source differs')
  ast.parse((v/leaf).read_bytes())
 literal=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));frozen=ast.literal_eval(literal.value)
 if frozen!=config or source!=REMOTE_SCRIPT.replace('CONFIG',repr(frozen),1):raise ValueError('frozen coordinator/config differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actual receipt; no rerun')
 began=time.monotonic()
 try:
  x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=210 if mode=='cpu' else 980);d=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-began)
 except subprocess.TimeoutExpired as e:d=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-began,error='SSH wait timeout; inspect existing bounded state, no retry')
 d['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,d);print(json.dumps(dict(name=name,returncode=d['returncode'],wall_seconds=d['wall_seconds'])));raise SystemExit(d['returncode']if d['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('cpu','train'));g.add_argument('--execute',choices=('cpu','train'));a=p.parse_args()
 if a.prepare:prepare(a.prepare)
 else:execute(a.execute)
