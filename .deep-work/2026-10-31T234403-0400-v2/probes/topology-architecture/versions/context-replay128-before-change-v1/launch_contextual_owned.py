"""Trusted bounded context coordinator. Import/static inspection does not run.

Resume reads frozen parent source/JSON and checkpoint bytes for hashes only.
Root copies the OWN checkpoint into read-only /opt; the sandboxed training UID
restores model content. The coordinator never imports Torch/unpickles weights.
"""
import argparse,ast,base64,hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
ORIGINAL_CONTINUOUS_SHA='b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f'
MAX_FILE=100*1024*1024

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def checked_name(value,label):
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}',value):raise ValueError(label+' must be a plain bounded name')
 return value

def parent_lineage(a,provided):
 """Read actual local frozen source/JSON/checkpoint BYTES; never torch.load."""
 parent=checked_name(a.resume_parent,'resume parent')
 frozen=ROOT/'versions'/('code-'+parent);learned=ROOT/'runs'/parent/'learn'
 if not frozen.is_dir():raise FileNotFoundError('missing parent frozen source: '+str(frozen))
 result=json.loads((learned/'result.json').read_text())
 if result.get('status')!='DECLARED_DEVELOPMENT_COMPLETED':raise ValueError('parent must have actually completed development')
 configuration=None
 with (learned/'records.jsonl').open() as stream:
  for line in stream:
   row=json.loads(line)
   if row.get('kind')=='configuration':configuration=row;break
 if configuration is None:raise ValueError('parent actual configuration missing')
 previous=configuration['args']
 if any(result.get(key)!=previous.get(key) for key in ('arm','states','seed')):raise ValueError('parent result/configuration identity differs')
 defaults=dict(block_state='categorical',block_initialization='iid',block_training='scheduled',straight_through_backward='selected',slots=4,baseline_width=64)
 for field in ('arm','states','seed','block_state','block_initialization','block_training','straight_through_backward','slots','baseline_width'):
  old=previous.get(field,defaults.get(field))
  if old is None:old=defaults.get(field)
  flag='--'+field.replace('_','-')
  if flag in provided and getattr(a,field)!=old:raise ValueError('explicit resume option differs from parent: '+flag)
  setattr(a,field,old)
 if '--stages' in provided and a.stages!=3:raise ValueError('resume requires --stages3')
 a.stages=3
 if a.steps!=600 or previous.get('steps')!=600:raise ValueError('resume preserves600 updates per stage')
 if previous.get('stages')!=1:raise ValueError('parent must be original one-stage phase0 run')
 leaf=a.resume_checkpoint or f'{a.arm}-stage0-step600.pt';checked_name(leaf,'resume checkpoint leaf')
 if not leaf.endswith('.pt') or Path(leaf).name!=leaf:raise ValueError('checkpoint must be one .pt leaf')
 checkpoint=learned/leaf
 if not checkpoint.is_file() or checkpoint.is_symlink() or checkpoint.stat().st_size>MAX_FILE:raise ValueError('regular owned <=100MiB checkpoint required')
 summaries=[row for row in result['training_reports'] if row.get('stage')==0]
 if len(summaries)!=1 or summaries[0]['costs'].get('SGD_updates')!=600:raise ValueError('parent did not record600 paid phase0 updates')
 summary=summaries[0];receipt=summary['checkpoint'];checkpoint_hash=digest(checkpoint)
 if receipt['file']!=leaf or receipt['sha256']!=checkpoint_hash or receipt['bytes']!=checkpoint.stat().st_size:raise ValueError('checkpoint differs from completed parent receipt')
 frozen_hashes={path.name:digest(path) for path in sorted(frozen.glob('*.py'))}
 for name,value in result['source_sha256'].items():
  if frozen_hashes.get(name)!=value:raise ValueError('parent frozen source differs from actual run: '+name)
 if configuration['source_sha256']!=result['source_sha256']:raise ValueError('parent configuration/final source hashes differ')
 if a.block_state=='continuous' and a.arm in ('dynamic','static','static_expand') and frozen_hashes.get('continuous_primitive_block.py')!=ORIGINAL_CONTINUOUS_SHA:
  raise ValueError('continuous continuation requires ORIGINAL b8a51 core, never shared8deca')
 a.resume_checkpoint=leaf
 return dict(parent_run=parent,parent_frozen_source_directory=str(frozen),parent_frozen_source_sha256=frozen_hashes,
  parent_actual_source_sha256=result['source_sha256'],parent_driver_sha256=frozen_hashes['train_contextual_primitives.py'],
  parent_checkpoint_leaf=leaf,parent_checkpoint_sha256=checkpoint_hash,parent_checkpoint_bytes=checkpoint.stat().st_size,
  parent_checkpoint_receipt=receipt,parent_actual_configuration=previous,parent_stage0_paid_updates=600,
  parent_stage0_checkpoint_costs=summary['costs'],historical_cost_scope='original parent retains post-freeze/inference/coordinator costs',
  first_new_stage=1,new_stages=[1,2],new_updates_per_stage=600,new_training_updates=1200,old_phase0_updates_repeated=False,replay_examples=0)

REMOTE_SCRIPT='''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;a=c['args'];r=pathlib.Path(c['remote']);name=c['name']
code_dir=r/('code-'+name);out_dir=r/'outputs'/name;inputs_dir=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code_dir.exists() or out_dir.exists() or (not a['data_only'] and inputs_dir.exists()):raise FileExistsError('run/code/input namespace exists; no overwrite')
code_dir.mkdir()
for leaf,encoded in c['files'].items():
 data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==c['hashes'][leaf]
 target=code_dir/leaf;target.write_bytes(data);target.chmod(0o444)
code_dir.chmod(0o555)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
if a['data_only']:
 runner=r/'remote_runner_cpu_v2.py';assert digest(runner)=='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
 cmd=[sys.executable,str(runner),'--script',str(code_dir/c['driver']),'--out',str(out_dir),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out_dir/'data'),'--states',str(a['states']),'--seed',str(a['seed']),'--max-seconds','60']
else:
 lineage=c['resume_lineage']
 source=(pathlib.Path('/opt/prometheus-'+lineage['parent_run']+'-inputs')/'data' if lineage else r/'outputs'/('parameter-context-data-n'+str(a['states'])+'-seed'+str(a['seed'])+'-v1')/'data')
 if not source.is_dir():raise FileNotFoundError('original frozen data input missing')
 inputs_dir.mkdir();shutil.copytree(source,inputs_dir/'data')
 if lineage:
  parent_code=r/('code-'+lineage['parent_run'])
  for leaf,expected in lineage['parent_actual_source_sha256'].items():
   if digest(parent_code/leaf)!=expected:raise ValueError('remote frozen parent source differs: '+leaf)
  original=r/'outputs'/lineage['parent_run']/'learn'/lineage['parent_checkpoint_leaf']
  if not original.is_file() or original.is_symlink():raise ValueError('remote checkpoint must be regular')
  if original.stat().st_size!=lineage['parent_checkpoint_bytes'] or digest(original)!=lineage['parent_checkpoint_sha256']:raise ValueError('remote checkpoint differs from synced receipt')
  # ROOT COPY/HASH ONLY: no Torch import/unpickling or model-content output.
  resume_input=inputs_dir/'resume.pt';shutil.copyfile(original,resume_input)
  if digest(resume_input)!=lineage['parent_checkpoint_sha256']:raise ValueError('resume copy mismatch')
  (inputs_dir/'resume-lineage.json').write_text(json.dumps(lineage,indent=2)+'\\n')
 for target in inputs_dir.rglob('*'):target.chmod(0o555 if target.is_dir() else 0o444)
 inputs_dir.chmod(0o555)
 runner=r/'remote_runner_gpu_authorized.py';assert digest(runner)=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
 policy=lineage['parent_actual_configuration'] if lineage else {}
 cmd=[sys.executable,str(runner),'--script',str(code_dir/c['driver']),'--out',str(out_dir),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--data',str(inputs_dir/'data'),'--out',str(out_dir/'learn'),'--arm',a['arm'],'--baseline-width',str(a['baseline_width']),'--states',str(a['states']),'--seed',str(a['seed']),'--stages',str(a['stages']),'--steps',str(a['steps']),'--primitives',str(policy.get('primitives',4)),'--lr',str(policy.get('lr',.003)),'--slots',str(a['slots']),'--block-state',a['block_state'],'--block-width',str(policy.get('block_width',64)),'--primitive-rank',str(policy.get('primitive_rank',8)),'--memory-slots',str(policy.get('memory_slots',8)),'--block-initialization',a['block_initialization'],'--block-training',a['block_training'],'--straight-through-backward',a['straight_through_backward'],'--device','cuda:0','--max-seconds','850']
 if lineage:cmd+=['--resume',str(inputs_dir/'resume.pt')]
print(json.dumps({'event':'frozen_contextual_launch','command':cmd,'resume_lineage':c['resume_lineage']}),flush=True);beg=time.monotonic()
try:
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=930)
 receipt={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'wall_seconds':time.monotonic()-beg,'resume_lineage':c['resume_lineage']}
 print(result.stdout);print(result.stderr,file=sys.stderr)
except subprocess.TimeoutExpired as error:receipt={'returncode':None,'transport':'coordinator_timeout','wall_seconds':time.monotonic()-beg,'stdout':str(error.stdout),'stderr':str(error.stderr),'resume_lineage':c['resume_lineage']}
if out_dir.exists():
 with (out_dir/'coordinator-exit.json').open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\\n')
raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true')
 p.add_argument('--arm',choices=['dynamic','static','static_expand','gru','transformer','ssm'],default='dynamic')
 p.add_argument('--states',type=int,default=8);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--stages',type=int,default=1);p.add_argument('--steps',type=int,default=600)
 p.add_argument('--revision',choices=['v1','v2','v3'],default='v1')
 p.add_argument('--block-initialization',choices=['iid','carry'],default='iid')
 p.add_argument('--block-training',choices=['scheduled','soft_only'],default='scheduled')
 p.add_argument('--straight-through-backward',choices=['selected','dense'],default='selected')
 p.add_argument('--block-state',choices=['categorical','continuous'],default='categorical')
 p.add_argument('--slots',type=int,default=4);p.add_argument('--baseline-width',type=int,default=64)
 p.add_argument('--resume-parent',default=None,help='actual completed phase0 RUNNAME; inherit unspecified policy/architecture')
 p.add_argument('--resume-checkpoint',default=None,help='optional actual checkpoint LEAF, default ARM-stage0-step600.pt')
 a=p.parse_args();provided={item.split('=',1)[0] for item in sys.argv[1:] if item.startswith('--')}
 if a.resume_checkpoint and not a.resume_parent:raise ValueError('--resume-checkpoint requires --resume-parent')
 if a.resume_parent and a.data_only:raise ValueError('data-only cannot resume neural training')
 lineage=parent_lineage(a,provided) if a.resume_parent else None
 if a.states not in (8,16,32) or a.stages not in (1,3) or not 1<=a.steps<=600:raise ValueError('finite development scope only')
 if not 1<=a.slots<=8 or not 1<=a.baseline_width<=256 or (a.arm=='transformer' and a.baseline_width%4):raise ValueError('bounded slots/width; Transformer width divisible4')
 name=(f'parameter-context-data-n{a.states}-seed{a.seed}-v1' if a.data_only else f'parameter-context-n{a.states}-seed{a.seed}-{a.arm}-stage{a.stages}-'+a.revision)
 if not a.data_only and a.block_training!='scheduled':name+='-'+a.block_training
 if not a.data_only and a.straight_through_backward!='selected':name+='-'+a.straight_through_backward
 if not a.data_only and a.block_state=='continuous':name+='-continuous'
 if not a.data_only and a.baseline_width!=64:name+='-width'+str(a.baseline_width)
 if lineage:name+='-resume'
 checked_name(name,'new run');version=ROOT/'versions'/('code-'+name)
 protected=[version,ROOT/'runs'/name,ROOT/(name+'-protocol.json'),ROOT/(name+'-coordinator-receipt.json')]
 if any(path.exists() for path in protected):raise FileExistsError('local new-run namespace exists; no overwrite')
 if a.data_only:files=['contextual_task_data.py']
 else:
  files=['train_contextual_primitives.py']
  if a.arm in ('dynamic','static','static_expand'):files+=['continuous_primitive_block.py' if a.block_state=='continuous' else 'contextual_primitive_block.py']
  if a.arm=='ssm':files+=['selective_ssm_baseline.py']
 driver=files[0];encoded={};hashes={};origins={}
 for leaf in files:
  origin=ROOT/leaf if not lineage or leaf==driver else Path(lineage['parent_frozen_source_directory'])/leaf
  data=origin.read_bytes();ast.parse(data,filename=leaf);hashes[leaf]=hashlib.sha256(data).hexdigest();encoded[leaf]=base64.b64encode(data).decode();origins[leaf]=str(origin)
  if lineage and leaf!=driver and hashes[leaf]!=lineage['parent_actual_source_sha256'].get(leaf):raise ValueError('selected parent execution dependency differs')
 if lineage:
  leaf='parent_train_contextual_primitives.py.source';origin=Path(lineage['parent_frozen_source_directory'])/'train_contextual_primitives.py'
  data=origin.read_bytes();ast.parse(data,filename=leaf);hashes[leaf]=hashlib.sha256(data).hexdigest();encoded[leaf]=base64.b64encode(data).decode();origins[leaf]=str(origin)
 version.mkdir(exist_ok=False)
 for leaf,value in encoded.items():(version/leaf).write_bytes(base64.b64decode(value))
 protocol=dict(before_execution=True,scope='same V2 development; NOT frozen Primary',name=name,args=vars(a),source_sha256=hashes,source_origins=origins,resume_lineage=lineage,
  feedback='identical observedinitial/rawtokens/terminal labels; no taskID/routes/intermediate or teacher operations',shared_step_budget=a.steps,
  previous_paid_training_updates=600 if lineage else 0,new_training_updates=1200 if lineage else a.steps*a.stages,replay_examples=0,
  proposal_config='actual declared categorical/continuous state and calling semantics remain distinct',comparison='actual matched baseline width/P; static differs only caller context; all failures retained',
  budget=dict(outside_seconds=120 if a.data_only else 900,internal_seconds=60 if a.data_only else 850,gpu_fraction=None if a.data_only else .3),
  costs='historical parent costs separate; new updates/time/P/optimizer/checkpoint/memory/proxyMAC/inference charged; no isolated-throughput claim',
  training_runtime='existing uid65534,Landlock4+,TSYNCseccomp,offline; unchanged authorized GPU exception',acceptance='no baseline score floor; no broad advantage or formal Primary claim')
 with (ROOT/(name+'-protocol.json')).open('x') as stream:stream.write(json.dumps(protocol,indent=2)+'\n')
 config=dict(name=name,files=encoded,hashes=hashes,driver=driver,args=vars(a),remote=REMOTE,resume_lineage=lineage)
 code=REMOTE_SCRIPT.replace('CONFIG',repr(config),1);ast.parse(code);started=time.monotonic()
 try:
  result=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=960)
  receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-started,resume_lineage=lineage)
 except subprocess.TimeoutExpired as error:receipt=dict(returncode=None,error='SSH timeout; inspect state, no rerun inferred',stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-started,resume_lineage=lineage)
 with (ROOT/(name+'-coordinator-receipt.json')).open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)

if __name__=='__main__':main()
