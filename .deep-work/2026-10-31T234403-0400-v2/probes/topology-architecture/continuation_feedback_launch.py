"""ROOT-only fixed matched600 continuation-feedback Block/GRU development."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
TEMPLATE=r'''import pathlib,json,base64,hashlib,subprocess,sys,time,shutil
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists()or out.exists()or inputs.exists():raise FileExistsError('new namespace only')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
code.mkdir();inputs.mkdir()
for leaf,raw in c['files'].items():
 b=base64.b64decode(raw)
 if hashlib.sha256(b).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport mismatch')
 p=code/leaf;p.write_bytes(b);p.chmod(0o444)
code.chmod(0o555)
for key,spec in c['manifest']['files'].items():
 src=r/'outputs'/spec['remote_relative']
 if src.is_symlink()or not src.is_file()or src.stat().st_size>100*1024*1024 or sha(src)!=spec['sha256']:raise ValueError('input hash mismatch: '+key)
 p=inputs/spec['leaf'];shutil.copyfile(src,p);p.chmod(0o444)
p=inputs/'manifest.json';p.write_text(json.dumps(c['manifest'],sort_keys=True,indent=2)+'\n');p.chmod(0o444);inputs.chmod(0o555)
runner=r/'remote_runner_gpu_authorized.py'
if sha(runner)!=c['GPU']:raise ValueError('unchanged runner hash differs')
command=[sys.executable,str(runner),'--script',str(code/'train_continuation_feedback.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--inputs',str(inputs),'--out',str(out/'learn'),'--family',c['family'],'--max-seconds','850']
print(json.dumps(dict(event='frozen_continuation_feedback_launch',command=command)),flush=True)
tick=time.monotonic();x=subprocess.run(command,text=True,capture_output=True,timeout=930)
print(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,runner_wall_seconds=time.monotonic()-tick)),flush=True)
raise SystemExit(x.returncode)
'''
def prepare(family):
 name=f'parameter-continuation-feedback-{family}-n8-seed0-v'+('2'if family=='gru'else'1');version=ROOT/'versions'/('code-'+name)
 if version.exists():raise FileExistsError('preserve frozen source')
 prior=json.loads((ROOT/'versions/code-parameter-factorized-value-anchor0-n8-seed0-fourstage-v3/launch-config.json').read_text())
 sources={leaf:(ROOT/'versions/code-parameter-factorized-value-anchor0-n8-seed0-fourstage-v3'/leaf).read_bytes()for leaf in('continuous_primitive_block.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')}
 sources['train_continuation_feedback.py']=(ROOT/'train_continuation_feedback.py').read_bytes()
 hashes={k:hashlib.sha256(v).hexdigest()for k,v in sources.items()}
 for raw in sources.values():ast.parse(raw)
 files={k:v for k,v in prior['manifest']['files'].items()if k[0].isdigit()and':'in k}
 if family=='block':
  parent='parameter-factorized-value-anchor0-n8-seed0-fourstage-v2';folder=ROOT/'runs'/parent/'learn'
  report=json.loads((folder/'completed-stage2-before-TEST.json').read_text())['report'];cp=report['checkpoint'];leaf=cp['file']
  source=json.loads((ROOT/(parent+'-protocol.json')).read_text())['source_sha256']
  paid=dict(scope='actual first three phases only; prior failed/discarded and noaux stage4 trials separate',reports=[json.loads((folder/f'completed-stage{s}-before-TEST.json').read_text())['report']for s in range(3)])
 else:
  parent=f'parameter-context-n8-seed0-{family}-stage3-v1-width'+('44'if family=='gru'else'32')+'-resume-replay128x16';folder=ROOT/'runs'/parent/'learn'
  finish=json.loads((folder/'completed-training-through-stage2.json').read_text());report=next(x for x in finish['reports']if x['stage']==2);cp=report['checkpoint'];leaf=cp['file']
  configurations=[json.loads(line)for line in(folder/'records.jsonl').open()if '"kind": "configuration"'in line]
  if len(configurations)!=1:raise ValueError('unique actual GRU configuration required')
  source=configurations[0]['source_sha256'];paid=dict(scope='actual OWN three-phase GRU44 lineage; phase0 linked in resume_lineage',reports=finish['reports'],resume_lineage=finish['resume_lineage'],full_cost_ledger='memory-live-stage-total-paid-training-cost-evidence-v1.json')
 checkpoint=folder/leaf
 if sha(checkpoint)!=cp['sha256']:raise ValueError('actual own NN checkpoint mismatch')
 files['resume']=dict(leaf='resume.pt',sha256=sha(checkpoint),remote_relative=str(checkpoint.relative_to(ROOT/'runs')))
 manifest=dict(schema='continuation-feedback-owned-inputs-v1',family=family,parent_run=parent,checkpoint_sha256=cp['sha256'],parent_source_sha256=source,parent_paid_lineage=paid,source_sha256=hashes,files=files)
 config=dict(name=name,remote=REMOTE,GPU=GPU,family=family,source_sha256=hashes,manifest=manifest,files={k:base64.b64encode(v).decode()for k,v in sources.items()})
 # Canonical key order prevents a representation-only transport mismatch.
 config=json.loads(json.dumps(config,sort_keys=True));remote=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(remote)
 protocol=dict(name=name,before_execution=True,family=family,source_sha256=hashes,formal_Primary=False,
  scope='new matched learning-rule diagnostic, not novel rule/Primary/broad advantage claim',
  hypothesis='finite old-program behavior may teach a reusable new-operation interface with more freedom than a pointwise latent anchor',
  comparison='Block and GRU receive same terminal-feedback/data/schedules/600updates/Replay128 and same4word selection+teacher algorithm; own teachers may have different errors',
  teacher='OWN trained pre-stage4 NN; first4 distinct phase0 TRAIN raw words length1or2,32query rows,256float cache; no true intermediate states or correct call sequence',
  objective='terminal NLL plus coefficient1 KL to own frozen continuation cache, only48new rows; cycle4words bystep',
  cost='all P, own paid parent, teacher32+32comparison forwards,1024Bcache, auxiliary suffix forward/backward,600SGD,replay/restore/2checkpoints/eval/inference; complete FLOPs unknown notzero',
  numerical_gate='beforeSGD: actual own prefix vs original and16rows×4 incrementalcarry vs fullconcat atol2e-4/rtol2e-5; preserve any failure',
  generalization='cache words are now TRAINED continuations; do not call their closure unseen; other words and longer compositions require separate frozen evaluation',
  fixed_budget='600x64 sameindices,48current+16old,Adam.003/clip1/seed0; no earlyaccuracy stopping or testselection',outside_seconds=900,inside_seconds=850,permissions='unchanged original offline GPU runner',
  decision='if terminal+heldout closure improves inspect tradeoff; if no gain keep negative, do not expand budget to force score')
 if family=='gru':protocol.update(repair='v1 stopped before any SGD: dense tail chose different cuDNN TF32 arithmetic; v2 uses same packed backend as original prefix/full reference, original precision unchanged',actual_diagnosis='parameter-gru-carry-numeric-diagnostic-v1: dense maxabs.00352/zeroargmaxdifference, packed maxabs0 on same frozen NN; no accuracy-based tuning')
 version.mkdir()
 for leaf,raw in sources.items():(version/leaf).write_bytes(raw)
 put(version/'launch-config.json',config);put(ROOT/(name+'-protocol.json'),protocol);(version/'remote-launch.source').write_text(remote)
 print(json.dumps(dict(name=name,status='FROZEN_UNEXECUTED',source_sha256=hashes,remote_sha256=sha(version/'remote-launch.source'))))
def execute(name):
 if name not in{f'parameter-continuation-feedback-{x}-n8-seed0-v1'for x in('block','gru','transformer')}|{'parameter-continuation-feedback-gru-n8-seed0-v2'}:raise ValueError('fixed names')
 version=ROOT/'versions'/('code-'+name);c=json.loads((version/'launch-config.json').read_text());source=TEMPLATE.replace('CONFIG',repr(c),1)
 if source!=(version/'remote-launch.source').read_text():raise ValueError('frozen coordinator differs')
 for leaf,digest in c['source_sha256'].items():
  if sha(version/leaf)!=digest:raise ValueError('frozen code differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve original attempt')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=980);v=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:v=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='inspect actual bounded process; no automatic retry')
 v['remote_launch_sha256']=sha(version/'remote-launch.source');put(receipt,v);print(json.dumps({k:v[k]for k in('returncode','wall_seconds')}));raise SystemExit(v['returncode']if v['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('block','gru','transformer'));g.add_argument('--execute');a=p.parse_args()
 if a.prepare:prepare(a.prepare)
 else:execute(a.execute)
