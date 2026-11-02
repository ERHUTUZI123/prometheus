"""ROOT-only, same-host bounded NEW extended closure packages.

Preparation reads OWN bytes/JSON and freezes sources; never imports a NN,
unpickles, runs a model, launches SSH, or changes old evidence. CPU generation
derives from the already generated closure once; GPU packages wait for sync.
"""
import argparse,ast,base64,hashlib,json,pickletools,subprocess,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
BASE='parameter-live-composition-data-v1'
DATA='parameter-extended-live-closure-data-v1'
BASE_SHA='3a6e98941c2465233e65546c875ba6035a5ec2304862fe994a42677e5e1db093'
KEYS=('factorized0','factorized1','profile_block','gru','transformer','profile_gru','profile_transformer')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
 with path.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
def checked(path,digest=None):
 if path.is_symlink()or not path.is_file()or path.stat().st_size>100*1024*1024:raise ValueError('bounded OWN regular file required: '+str(path))
 receipt=dict(bytes=path.stat().st_size,sha256=sha(path))
 if digest is not None and receipt['sha256']!=digest:raise ValueError('OWN SHA differs: '+str(path))
 return receipt

TEMPLATE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists()or out.exists()or inputs.exists():raise FileExistsError('new namespace only; preserve attempts')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
code.mkdir();inputs.mkdir()
for leaf,encoded in c['sources'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport mismatch')
 target=code/leaf;target.write_bytes(raw);target.chmod(0o444)
code.chmod(0o555)
for leaf,spec in c['inputs'].items():
 source=r/'outputs'/spec['remote_relative']
 if source.is_symlink()or not source.is_file()or source.stat().st_size!=spec['bytes']or sha(source)!=spec['sha256']:raise ValueError('exact OWN input differs: '+leaf)
 target=inputs/leaf;shutil.copyfile(source,target);target.chmod(0o444)
if c['mode']=='eval':
 for leaf,digest in c['parent_source_sha256'].items():
  if sha(r/('code-'+c['parent'])/leaf)!=digest:raise ValueError('parent execution source changed')
 raw=base64.b64decode(c['manifest_bytes'])
 if hashlib.sha256(raw).hexdigest()!=c['manifest_sha256']:raise ValueError('manifest transport differs')
 p=inputs/'manifest.json';p.write_bytes(raw);p.chmod(0o444)
inputs.chmod(0o555)
runner=r/('remote_runner_cpu_v2.py'if c['mode']=='data'else'remote_runner_gpu_authorized.py')
if sha(runner)!=(c['CPU']if c['mode']=='data'else c['GPU']):raise ValueError('unchanged bounded runner differs')
command=[sys.executable,str(runner),'--script',str(code/c['entry']),'--out',str(out),'--timeout','120'if c['mode']=='data'else'900','--address-space-gib','64'if c['mode']=='data'else'512','--file-limit-mib','100']
command+=['--cpu-only']if c['mode']=='data'else['--gpu-fraction','.3','--cuda-init-before-seccomp']
command+=['--','--inputs',str(inputs),'--out',str(out/c['result_subdir'])]
print(json.dumps(dict(event='frozen_extended_closure_launch',name=name,command=command)),flush=True)
tick=time.monotonic()
try:
 x=subprocess.run(command,capture_output=True,text=True,timeout=150 if c['mode']=='data'else 930)
 receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,runner_wall_seconds=time.monotonic()-tick)
except subprocess.TimeoutExpired as e:
 receipt=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),runner_wall_seconds=time.monotonic()-tick,error='inspect bounded existing process; no automatic retry')
receipt['original_inputs_unchanged']={leaf:sha(r/'outputs'/spec['remote_relative'])==spec['sha256']for leaf,spec in c['inputs'].items()}
if out.exists():
 with(out/'coordinator-exit.json').open('x')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt),flush=True);raise SystemExit(receipt['returncode']if receipt['returncode']is not None else 124)
'''

def commit(name,sources,protocol,config,manifest=None):
 version=ROOT/'versions'/('code-'+name)
 for p in(version,ROOT/'runs'/name,ROOT/(name+'-protocol.json'),ROOT/(name+'-coordinator-receipt.json')):
  if p.exists():raise FileExistsError('freeze new namespace only: '+str(p))
 hashes={}
 for leaf,raw in sources.items():
  if Path(leaf).name!=leaf or not leaf.endswith('.py'):raise ValueError('flat Python source only')
  ast.parse(raw,filename=leaf);hashes[leaf]=hashlib.sha256(raw).hexdigest()
 protocol=dict(protocol,name=name,before_execution=True,source_sha256=hashes,formal_Primary=False)
 config=dict(config,name=name,remote=REMOTE,CPU=CPU,GPU=GPU,source_sha256=hashes,sources={k:base64.b64encode(v).decode()for k,v in sources.items()},protocol=protocol)
 config=json.loads(json.dumps(config,sort_keys=True));remote=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(remote)
 version.mkdir()
 for leaf,raw in sources.items():(version/leaf).write_bytes(raw)
 if manifest is not None:(version/'input-manifest.json').write_bytes(manifest)
 put(version/'launch-config.json',config);(version/'remote-launch.source').write_text(remote);put(ROOT/(name+'-protocol.json'),protocol)
 return dict(status='FROZEN_UNEXECUTED_ROOT_ONLY',name=name,source_sha256=hashes,remote_sha256=sha(version/'remote-launch.source'))

def prepare_data():
 folder=ROOT/'runs'/BASE/'generated/data';path=folder/'closure.npz';receipt=checked(path,BASE_SHA)
 metadata=json.loads((folder/'metadata.json').read_text())
 if metadata['schema']!='live-composition-closure-data-v1'or metadata['rows']!=2048 or metadata['file']['sha256']!=BASE_SHA:raise ValueError('actual original closure schema/SHA required')
 for p in(ROOT/(BASE+'-coordinator-receipt.json'),ROOT/'runs'/BASE/'coordinator-exit.json'):
  if json.loads(p.read_text()).get('returncode')!=0:raise ValueError('actual original data completion receipt required')
 protocol=dict(scope='NEW postdesign DEVELOPMENT; derive seven fixed suffixes from exact already generated512base cases; preserve old data/results',
  original_data_run=BASE,original_closure=receipt,original_metadata_sha256=sha(folder/'metadata.json'),rows=3584,base_cases=512,
  suffixes=[[],[1,3],[2,3],[14],[1,11,3],[2,11,3],[1,3,2,3]],maximum_length=50,
  fixed_length_stratum='within_original48 assigned from raw length before model query; full and stratum denominators reported',
  overlap='profile arms TRAINED SET0_STEP and SET1_STEP; NOP/composed/CLEAR outside four-word teacher cache',
  labels='fixed old-task algebra derives3584 labels; zero new external labels; original teacher generation remains separately paid',
  NN_queries=0,SGD_updates=0,outside_seconds=120,script_internal_deadline_seconds=None,address_space_GiB=64,
  permissions='unchanged same-host CPU runner; generator unchanged; external120second termination bound')
 return commit(DATA,{'build_extended_closure_data.py':(ROOT/'build_extended_closure_data.py').read_bytes()},protocol,
  dict(mode='data',entry='build_extended_closure_data.py',result_subdir='generated/data',inputs={'closure.npz':dict(receipt,remote_relative=str(path.relative_to(ROOT/'runs')))}))

def registry(key):
 if key in('factorized0','factorized1'):
  return dict(parent='parameter-factorized-value-anchor'+key[-1]+'-n8-seed0-fourstage-v3',family='block',P=13397,checkpoint_schema='factorized-value-development-training-v1',profile_cache_trained=False,status='ACTUAL_FOUR_STAGE_FACTORIZED_DEVELOPMENT_COMPLETE')
 if key.startswith('profile_'):
  family=key[8:]
  revision='v2'if family=='gru'else'v1'
  return dict(parent='parameter-continuation-feedback-'+family+'-n8-seed0-'+revision,family=family,P={'block':13397,'gru':25616,'transformer':26760}[family],checkpoint_schema='continuation-feedback-stage4-training-v1',profile_cache_trained=True,status='ACTUAL_CONTINUATION_FEEDBACK600_COMPLETE')
 return dict(parent='parameter-memory-live-stage-n8-seed0-'+key+'-stage4-v1',family=key,P={'gru':25616,'transformer':26760}[key],checkpoint_schema='contextual-primitives-training-v1',profile_cache_trained=False,status='DECLARED_FOURTH_STAGE_DEVELOPMENT_COMPLETED')

def checkpoint_header(path):
 """Bounded literal header parse, not pickle loading or tensor execution."""
 with zipfile.ZipFile(path)as archive:
  entries=[n for n in archive.namelist()if n.endswith('/data.pkl')]
  if len(entries)!=1 or archive.getinfo(entries[0]).file_size>8*1024*1024:raise ValueError('OWN bounded Torch ZIP header required')
  raw=archive.read(entries[0])
 ops=[]
 for op,arg,_ in pickletools.genops(raw):
  if op.name not in('BINPUT','LONG_BINPUT','MEMOIZE'):ops.append((op.name,arg))
  if len(ops)>=256:break
 result={}
 for i,(name,arg)in enumerate(ops[:-1]):
  if name in('BINUNICODE','SHORT_BINUNICODE','UNICODE')and arg in('schema','stage','step','stage_complete')and arg not in result:
   op,value=ops[i+1]
   if op in('BINUNICODE','SHORT_BINUNICODE','UNICODE','BININT','BININT1','BININT2','INT','LONG1','LONG4'):result[arg]=value
   elif op in('NEWTRUE','NEWFALSE'):result[arg]=op=='NEWTRUE'
 return result

def cost_macs(cost):
 if 'base_forward_MAC_proxy'in cost:return cost['base_forward_MAC_proxy']+cost['continuation_forward_MAC_proxy']
 for key in('forward_MAC_proxy','training_forward_MAC_proxy'):
  if key in cost:return cost[key]
 raise ValueError('actual paid phase forward MAC accounting missing')

def paid_lineage(spec,result,protocol,folder):
 if spec['checkpoint_schema']=='factorized-value-development-training-v1':
  phases=[dict(stage=r['stage'],costs=r['costs'],checkpoint=r['checkpoint'])for r in result['reports']]
  extra=dict(scope='actual discarded old stage3 attempts, separately paid beyond retained four-phase recipe',
   SGD_updates=result['recovery']['original_stage3_updates_paid_before_failure'],
   training_wall_seconds=result['recovery']['original_stage3_training_seconds'],
   training_forward_MAC_proxy=result['recovery']['original_stage3_forward_MAC'],
   recovery=result['recovery'],cost_boundary=result['cost_boundary'])
 elif spec['profile_cache_trained']:
  history=result['parent_paid_lineage'];reports=history['reports']
  phases=[dict(stage=r['stage'],costs=r['costs'],checkpoint=r['checkpoint'])for r in reports]
  if sorted(p['stage']for p in phases)==[1,2]:
   phases.insert(0,dict(stage=0,costs=history['resume_lineage']['historical_checkpoint_costs'],scope='actual OWN saved phase0 checkpoint costs'))
  phases.append(dict(stage=3,costs=result['costs'],checkpoint=result['checkpoint']))
  extra=dict(scope='OWN teacher cache formation/comparison and restore charged separately from SGD forwards',teacher=result['teacher'],restore=result['restore'],
   maintenance=result.get('maintenance'),reload=result.get('reload'),inference_total_seconds=result.get('inference_total_seconds'),
   parent_paid_lineage_scope=history.get('scope'),shared_parent_warning='same actual early phases shared with no-anchor arm; never add shared histories as independent trials')
 else:
  history=protocol['parent_lineage']
  phases=[dict(stage=0,costs=history['historical_resume_lineage']['historical_checkpoint_costs'],scope='actual OWN saved phase0 checkpoint costs')]
  phases +=[dict(stage=r['stage'],costs=r['costs'],checkpoint=r['checkpoint'])for r in history['historical_training_reports']]
  phases.append(dict(stage=3,costs=result['costs'],checkpoint=result['checkpoint']))
  extra=dict(scope='original four-phase trained model; no continuation-cache auxiliary training',parent_run=result['parent_run'],parent_checkpoint_sha256=history['checkpoint_sha256'])
 if sorted(p['stage']for p in phases)!=[0,1,2,3]or any(p['costs']['SGD_updates']!=600 for p in phases):raise ValueError('actual distinct four600 phase lineage required')
 # Preserve all measured phase counters. Forward MAC is a proxy and does not
 # cover backward, anchor loss, or auxiliary backward; measured times do.
 lineage=dict(phases=phases,retained_lineage_SGD_updates=2400,retained_lineage_training_wall_seconds=sum(p['costs']['training_wall_seconds']for p in phases),
  retained_lineage_training_forward_MAC_proxy=sum(cost_macs(p['costs'])for p in phases),other_paid_costs=extra,
  current_parent_result_sha256=sha(folder/'result.json'),current_parent_records_sha256=sha(folder/'records.jsonl'),
  exact_full_training_FLOPs=None,scope='four retained phases and explicitly separate discarded/teacher costs; no pooled performance or independent-seed claim; shared stages paid once')
 return lineage

def parent_facts(key):
 if key not in KEYS:raise ValueError('fixed extended allowlist only')
 spec=registry(key);parent=spec['parent'];folder=ROOT/'runs'/parent/'learn'
 result=json.loads((folder/'result.json').read_text())
 if result.get('status')!=spec['status']or result.get('formal_Primary')is not False or result['resources']['parameters']!=spec['P']:raise ValueError('actual OWN completed model/status/P required: '+key)
 protocol_path=ROOT/(parent+'-protocol.json');protocol=json.loads(protocol_path.read_text())
 sources=result['source_sha256'];version=ROOT/'versions'/('code-'+parent)
 if sources!=protocol['source_sha256']or{p.name for p in version.glob('*.py')}!=set(sources):raise ValueError('actual/frozen NN sources differ')
 original={}
 for leaf,digest in sources.items():checked(version/leaf,digest);original[leaf]=(version/leaf).read_bytes();ast.parse(original[leaf],filename=leaf)
 if key.startswith('factorized'):
  if result['actual_new_updates_this_process']!=600 or result['actual_training_updates']!=2400:raise ValueError('actual four-phase recipe required')
  report=next(r for r in result['reports']if r['stage']==3);cp=report['checkpoint']
 else:cp=result['checkpoint']
 checkpoint=folder/cp['file'];receipt=checked(checkpoint,cp['sha256'])
 if receipt['bytes']!=cp['bytes']:raise ValueError('OWN checkpoint byte count differs')
 header=checkpoint_header(checkpoint)
 if header!=dict(schema=spec['checkpoint_schema'],stage=3,step=600,stage_complete=True):raise ValueError('actual checkpoint literal header differs: '+str(header))
 updates=[];configs=[]
 for line in(folder/'records.jsonl').open():
  if'"kind": "actual_update"'in line:updates.append(json.loads(line))
  elif'"kind": "configuration"'in line:configs.append(json.loads(line))
 if len(configs)!=1 or configs[0]['source_sha256']!=sources or[r['step']for r in updates]!=list(range(1,601)):raise ValueError('actual600 committed updates/source required')
 block_header={}
 if spec['family']=='block':
  cls=next(node for node in ast.parse(original['continuous_primitive_block.py']).body if isinstance(node,ast.ClassDef)and node.name=='ContinuousPrimitiveBlock')
  constants={target.id:ast.literal_eval(node.value)for node in cls.body if isinstance(node,ast.Assign)for target in node.targets if isinstance(target,ast.Name)and target.id in('checkpoint_version','organization_policy')}
  block_header=dict(version=constants['checkpoint_version'],organization_policy=constants['organization_policy'])
  if block_header!=dict(version='continuous-primitive-block-control-value-v1',organization_policy='learned-control-history/value-identity-write/group8-carry-v1'):raise ValueError('exact new control/value NN policy required')
 return dict(spec=spec,original=original,checkpoint=checkpoint,checkpoint_receipt=receipt,parent_sources=sources,
  checkpoint_header=header,block_header=block_header,resources=result['resources'],cost_lineage=paid_lineage(spec,result,protocol,folder))

def generated_data_facts():
 folder=ROOT/'runs'/DATA/'generated/data';meta=json.loads((folder/'metadata.json').read_text())
 data_files={leaf:checked(folder/leaf)for leaf in('closure.npz','metadata.json')}
 if meta.get('schema')!='extended-live-closure-development-v1'or meta['rows']!=3584 or meta['base_cases']!=512 or meta['maximum_length']>50 or meta['source_sha256']!=BASE_SHA or meta['file_sha256']!=data_files['closure.npz']['sha256']:raise ValueError('actual once-generated extended data differs')
 version=ROOT/'versions'/('code-'+DATA);config=json.loads((version/'launch-config.json').read_text())
 checked(version/'build_extended_closure_data.py',config['source_sha256']['build_extended_closure_data.py'])
 local_path=ROOT/(DATA+'-coordinator-receipt.json');local=json.loads(local_path.read_text());remote_path=ROOT/'runs'/DATA/'coordinator-exit.json';remote=json.loads(remote_path.read_text())
 if local.get('returncode')!=0 or remote.get('returncode')!=0 or local['remote_launch_sha256']!=sha(version/'remote-launch.source')or not all(remote['original_inputs_unchanged'].values()):raise ValueError('actual bounded CPU data completion/hash receipt required')
 old_meta=ROOT/'runs'/BASE/'generated/data/metadata.json';old=json.loads(old_meta.read_text());old_receipt=ROOT/(BASE+'-coordinator-receipt.json');old_remote=ROOT/'runs'/BASE/'coordinator-exit.json'
 paid=dict(extended_generation=dict(algorithm_seconds=meta['generation_wall_seconds'],complete_CPU_runner_wall_seconds=remote['runner_wall_seconds'],complete_SSH_coordinator_wall_seconds=local['wall_seconds'],CPU_peak_RSS_bytes=meta['cpu_peak_RSS_bytes'],metadata_sha256=sha(folder/'metadata.json'),coordinator_receipt_sha256=sha(local_path)),
  original_base_generation=dict(metadata_sha256=sha(old_meta),teacher_costs=old.get('teacher_costs'),algorithm_seconds=old['generation_wall_seconds'],complete_CPU_runner_wall_seconds=json.loads(old_remote.read_text())['wall_seconds'],complete_SSH_coordinator_wall_seconds=json.loads(old_receipt.read_text())['wall_seconds']),
  new_external_labels=0,new_derived_terminal_labels=3584,scope='two shared once-only data jobs; algorithm/runner/SSH scopes nested, never sum timers or repeat costs per NN')
 return folder,data_files,paid

def prepare_eval(key):
 # Scientific checkpoint/source/header/committed updates are checked BEFORE
 # any new extended metadata or labels are accessed in local preparation.
 facts=parent_facts(key);spec=facts['spec'];folder,data_files,paid=generated_data_facts()
 name='parameter-extended-live-closure-'+key+'-v1'
 sources=dict(facts['original']);sources['evaluate_extended_closure.py']=(ROOT/'evaluate_extended_closure.py').read_bytes()
 source_hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 manifest=dict(schema='extended-live-closure-evaluator-inputs-v1',model=key,family=spec['family'],P=spec['P'],checkpoint_sha256=facts['checkpoint_receipt']['sha256'],
  checkpoint_schema=spec['checkpoint_schema'],block_checkpoint_header=facts['block_header'],source_sha256=source_hashes,parent_source_sha256=facts['parent_sources'],
  data_sha256=data_files['closure.npz']['sha256'],metadata_sha256=data_files['metadata.json']['sha256'],profile_cache_trained=spec['profile_cache_trained'],
  parent_resources=facts['resources'],training_lineage=facts['cost_lineage'],generation_paid_lineage=paid)
 raw=(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 inputs={leaf:dict(receipt,remote_relative=str((folder/leaf).relative_to(ROOT/'runs')))for leaf,receipt in data_files.items()}
 inputs['resume.pt']=dict(facts['checkpoint_receipt'],remote_relative=str(facts['checkpoint'].relative_to(ROOT/'runs')))
 protocol=dict(scope='NEW postdesign DEVELOPMENT frozen SOFT closure; no training/search/oldTEST replacement or universal computation claim',
  fixed_model_registry_version='extended-live-closure-seven-owned-v2-profile-gru-v2'if key=='profile_gru'else'extended-live-closure-seven-owned-v1',model=key,parent=spec['parent'],family=spec['family'],P=spec['P'],
  checkpoint=facts['checkpoint_receipt'],checkpoint_schema=spec['checkpoint_schema'],block_checkpoint_header=facts['block_header'],original_source_sha256=facts['parent_sources'],
  data_run=DATA,data_files=data_files,rows=3584,base_cases=512,maximum_length=50,
  metrics='MIX/READ separate; all256 denominator and fixed <=48/>48 strata, joint absolute correctness, prediction consistency, own-base-correct conditional with numerator/denominator/exact source indices',
  overlap='profile arms TRAINED SET0_STEP/SET1_STEP; NOP/composed/CLEAR not in teacher four-word cache; empty tautological/CLEAR degenerate',
  information='exact source/hash/checkpoint schema/version/policy/NN restored before metadata/label access; NN gets raw initial/tokens/lengths only',
  modes=['soft'],SGD_updates=0,training_lineage=facts['cost_lineage'],generation_paid_lineage=paid,
  outside_seconds=900,inside_seconds=800,gpu_fraction=.3,address_space_GiB=512,permissions='unchanged existing bounded same-host GPU runner; no additional framework or permissions')
 return commit(name,sources,protocol,dict(mode='eval',entry='evaluate_extended_closure.py',result_subdir='evaluate',parent=spec['parent'],parent_source_sha256=facts['parent_sources'],inputs=inputs,manifest_bytes=base64.b64encode(raw).decode(),manifest_sha256=hashlib.sha256(raw).hexdigest()),raw)

def execute(name):
 if name not in{DATA}|{'parameter-extended-live-closure-'+key+'-v1'for key in KEYS}:raise ValueError('fixed new namespaces only')
 version=ROOT/'versions'/('code-'+name);config=json.loads((version/'launch-config.json').read_text());source=(version/'remote-launch.source').read_text()
 if source!=TEMPLATE.replace('CONFIG',repr(config),1):raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():checked(version/leaf,digest)
 for spec in config['inputs'].values():
  actual=checked(ROOT/'runs'/spec['remote_relative'],spec['sha256'])
  if actual['bytes']!=spec['bytes']:raise ValueError('frozen input byte count differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actual attempt; no automatic retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=180 if config['mode']=='data'else 980)
  result=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:
  result=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='inspect actual bounded process; no automatic retry')
 result['remote_launch_sha256']=sha(version/'remote-launch.source');put(receipt,result)
 print(json.dumps({k:result[k]for k in('returncode','wall_seconds')}));raise SystemExit(result['returncode']if result['returncode']is not None else 124)

if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-data',action='store_true');g.add_argument('--prepare-eval',choices=KEYS);g.add_argument('--execute');a=p.parse_args()
 if a.prepare_data:print(json.dumps(prepare_data()))
 elif a.prepare_eval:print(json.dumps(prepare_eval(a.prepare_eval)))
 else:execute(a.execute)
