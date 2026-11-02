"""Freeze one baseline CPU admission or one of four prospectively fixed tails.

ROOT alone invokes --execute. Existing owned CPU/GPU runners remain unchanged.
"""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335';GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
CPUNAME='parameter-paired-baseline-resume-cpu-v1'
COMMON=('paired_baseline_resume.py','train_paired_baseline_stage3.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,value):
 with p.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
def name_for(mode,arm,sampling):return CPUNAME if mode=='cpu'else f'parameter-paired-tail-{arm}-{sampling.replace("_","-")}-n8-seed0-600-v1'
REMOTE_SCRIPT=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG
r=pathlib.Path(c['remote']);code=r/('code-'+c['name']);out=r/'outputs'/c['name'];inputs=pathlib.Path('/opt/prometheus-'+c['name']+'-inputs');status=out.parent/(out.name+'.runner-status.json');receipt=out.parent/(out.name+'.coordinator-exit.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())if p.is_file()else None
if any(p.exists()or p.is_symlink()for p in(code,out,inputs,status,receipt)):raise FileExistsError('new namespace only; preserve evidence')
if c['mode']=='train':
 proofpath=r/'outputs'/c['CPU_name']/'check/result.json';rs=load(r/'outputs'/(c['CPU_name']+'.runner-status.json'));proof=load(proofpath)
 if not proof or sha(proofpath)!=c['CPU_proof_sha256']or not rs or rs.get('status')!='completed'or proof.get('status')!='CPU_PAIRED_BASELINE_OWN_PARENT_RESUME_PASS'or proof.get('source_sha256')!=c['manifest']['source_sha256']:raise ValueError('actual same-source CPU admission required')
 for flag in c['CPU_required_flags']:
  if proof.get(flag)is not True:raise ValueError('CPU semantic check missing: '+flag)
code.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded,validate=True)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source bytes differ')
 p=code/leaf
 with p.open('xb')as f:f.write(raw)
 p.chmod(0o444)
code.chmod(0o555);inputs.mkdir()
for key,spec in c['manifest']['files'].items():
 source=r/'outputs'/spec['remote_relative'];target=inputs/spec['leaf']
 if source.is_symlink()or not source.is_file()or source.stat().st_size>100*1024*1024 or sha(source)!=spec['sha256']:raise ValueError('actual owned input differs: '+key)
 shutil.copyfile(source,target);target.chmod(0o444)
p=inputs/'manifest.json'
with p.open('x')as f:json.dump(c['manifest'],f,sort_keys=True,indent=2);f.write('\n')
p.chmod(0o444);inputs.chmod(0o555)
if c['mode']=='cpu':
 runner=r/'remote_runner_cpu_v2.py';expected='CPU_PAIRED_BASELINE_OWN_PARENT_RESUME_PASS'
 if sha(runner)!=c['CPU']:raise ValueError('unchanged CPU runner')
 cmd=[sys.executable,str(runner),'--script',str(code/'check_paired_baseline_resume_cpu.py'),'--out',str(out),'--timeout','150','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--inputs',str(inputs),'--out',str(out/'check')]
else:
 runner=r/'remote_runner_gpu_authorized.py';expected='ACTUAL_PAIRED_BASELINE_STAGE3_600_COMPLETE'
 if sha(runner)!=c['GPU']:raise ValueError('unchanged GPU runner')
 cmd=[sys.executable,str(runner),'--script',str(code/'train_paired_baseline_stage3.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--inputs',str(inputs),'--out',str(out/'learn'),'--arm',c['arm'],'--sampling',c['sampling'],'--max-seconds','850']
began=time.monotonic()
try:x=subprocess.run(cmd,text=True,capture_output=True,timeout=180 if c['mode']=='cpu'else 930);rc=x.returncode;stdout=x.stdout;stderr=x.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or'');stderr=str(e.stderr or'');error='coordinator wait timeout, inspect existing bounded runner; no retry'
for suffix,value in(('coordinator-stdout.log',stdout),('coordinator-stderr.log',stderr)):
 with(out.parent/(out.name+'.'+suffix)).open('x')as f:f.write(value)
rs=load(status);result=load(out/('check/result.json'if c['mode']=='cpu'else'learn/result.json'));passed=bool(rc==0 and rs and rs.get('status')=='completed'and result and result.get('status')==expected)
if c['mode']=='train':passed=bool(passed and result.get('arm')==c['arm']and result.get('sampling')==c['sampling']and result.get('actual_new_updates')==600 and result.get('old_updates_repeated')==0 and result.get('NN_unchanged_after_TEST')and result.get('source_sha256')==c['manifest']['source_sha256'])
record=dict(name=c['name'],returncode=rc,protocol_completed=passed,error=error,whole_runner_wall_seconds=time.monotonic()-began,source_sha256=c['source_sha256'],command=cmd,result_path=str(out/('check/result.json'if c['mode']=='cpu'else'learn/result.json')),root_status_path=str(status),actual_updates=(result or{}).get('actual_new_updates'),no_accuracy_gate=True,full_training_FLOPs=None,provider_effective_compute=None)
with receipt.open('x')as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(record,sort_keys=True));raise SystemExit(0 if passed else(rc if rc not in(None,0)else 1))
'''
FLAGS=['both_OWN_stage2_NN_Adam_available_RNG_replay_restored','original_growth_preserved_old_forward_and_Adam','same_start_both_samplers','exact_original_Replay_schedule','discarded_synthetic_NN_never_GPU_parent']
def prepare(mode,arm,sampling):
 name=name_for(mode,arm,sampling);version=ROOT/'versions'/('code-'+name);protocol=ROOT/(name+'-protocol.json')
 if version.exists()or protocol.exists()or(ROOT/(name+'-coordinator-receipt.json')).exists():raise FileExistsError('new freeze only')
 fixed={'owned_parent_driver.py':'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc','context_replay128.py':'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140','memory_live_stage_train.py':'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6'};base=ROOT/'versions/code-parameter-gru32-n8-seed0-fourstage-v1';raws={}
 for leaf in COMMON:
  path=base/leaf if leaf in fixed else ROOT/leaf
  if leaf in fixed and sha(path)!=fixed[leaf]:raise ValueError('unchanged original source: '+leaf)
  raws[leaf]=path.read_bytes()
 common={k:hashlib.sha256(v).hexdigest()for k,v in raws.items()}
 if mode=='cpu':raws['check_paired_baseline_resume_cpu.py']=(ROOT/'check_paired_baseline_resume_cpu.py').read_bytes()
 for b in raws.values():ast.parse(b)
 hashes={k:hashlib.sha256(v).hexdigest()for k,v in raws.items()};inventory=json.loads((ROOT/'paired-tail-stage2-parent-inventory-v1.json').read_text());manifest=dict(schema='paired-baseline-stage3-inputs-v1',source_sha256=common,parents={},files={})
 def add(key,path):
  leaf=key.replace(':','-')+path.suffix;manifest['files'][key]=dict(leaf=leaf,sha256=sha(path),remote_relative=str(path.relative_to(ROOT/'runs')))
 for model in('gru32','transformer32'):
  entry=inventory['parents'][model];path=ROOT/entry['checkpoint_relative']
  if sha(path)!=entry['checkpoint_sha256']:raise ValueError('exact actual OWN parent')
  manifest['parents'][model]=dict(schema=entry['schema_from_frozen_writer_source'],checkpoint_sha256=entry['checkpoint_sha256'],source_sha256=entry['source_sha256'],model_spec=entry['model_spec_source_inference'],replay_digest=entry['replay_digest']);add('parent:'+model,path)
  for stage in range(3):
   parentrun=entry['run']if model=='gru32'or stage else'parameter-context-n8-seed0-transformer-stage1-v1-width32';add(f'oldindices:{model}:{stage}',ROOT/'runs'/parentrun/'learn'/f'stage{stage}-common-indices.npz')
 olddata=ROOT/'runs/parameter-context-data-n8-seed0-v1/data';newdata=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data';om=json.loads((olddata/'metadata.json').read_text());nm=json.loads((newdata/'metadata.json').read_text())
 for stage in range(4):
  for split in('train','valid','test_iid','test_long','test_unseen'):
   path=(olddata/om['stages'][stage]['files'][split])if stage<3 else(newdata/nm['files'][split]['file']);digest=om['stages'][stage]['hashes'][split]if stage<3 else nm['files'][split]['sha256']
   if sha(path)!=digest:raise ValueError('original raw data')
   add(f'data:{stage}:{split}',path)
 schedule=ROOT/'runs/parameter-paired-train-indices-n8-stage3-seed0-v1/generate';result=json.loads((schedule/'result.json').read_text());path=schedule/'paired-stage3-indices.npz'
 if result['status']!='CPU_PAIRED_TRAIN_INDICES_GENERATED_PASS'or sha(path)!=result['schedule']['sha256']or not result['same_pool_row_exposures_exact']:raise ValueError('actual fixed paired/control schedule required')
 add('paired_schedule',path);add('replay_draws',ROOT/'runs/parameter-gru32-n8-seed0-fourstage-v1/learn/stage3-replay-draws.npz');manifest['paired_schedule_actual_result_sha256']=sha(schedule/'result.json');manifest['paired_schedule_actual_protocol_sha256']=sha(ROOT/'parameter-paired-train-indices-n8-stage3-seed0-v1-protocol.json')
 cpuproofdigest=None
 if mode=='train':
  proof=ROOT/'runs'/CPUNAME/'check/result.json';value=json.loads(proof.read_text())
  if value['status']!='CPU_PAIRED_BASELINE_OWN_PARENT_RESUME_PASS'or value['source_sha256']!=common or not all(value[x]is True for x in FLAGS):raise ValueError('actual same-source CPU admission')
  cpuproofdigest=sha(proof)
 config=dict(name=name,mode=mode,arm=arm if mode=='train'else None,sampling=sampling if mode=='train'else None,remote=REMOTE,CPU=CPU,GPU=GPU,CPU_name=CPUNAME,CPU_proof_sha256=cpuproofdigest,CPU_required_flags=FLAGS,manifest=manifest,source_sha256=hashes,files={k:base64.b64encode(v).decode()for k,v in raws.items()});remote=REMOTE_SCRIPT.replace('CONFIG',repr(config),1);ast.parse(remote);version.mkdir()
 for leaf,raw in raws.items():(version/leaf).write_bytes(raw);(version/leaf).chmod(0o444)
 (version/'remote-launch.source').write_text(remote);(version/'remote-launch.source').chmod(0o444);put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
 doc=dict(name=name,before_execution=True,mode=mode,arm=config['arm'],sampling=config['sampling'],source_sha256=hashes,common_source_sha256=common,manifest=manifest,CPU_proof_sha256=cpuproofdigest,old_effective_updates=1800,old_updates_repeated=0,new_updates=0 if mode=='cpu'else 600,only_prospective_tail_variation='paired vssameglobalpoolshuffle, sameNN/Adam/RNG/growth/Replay/labels/budget; changesbatchcovariance ANDtemporalorder',six_global_tails='typed/GRU32/TF32 xpaired/same_pool_shuffle; thislauncheronly4baseline tails',optimizer='originalAdam.003/clip1/terminalcrossentropyonly',no_accuracy_gate=True,no_extra_NN_class_or_teacher=True,CPU_scope='two discarded synthetic steps only; noCUDAcalls, GPUrownotforcedsameasCPU',GPU_scope='ordinaryunmodifiedclasses/originalbatchcropping, noartificialpadding/Graph/compiler; nativealias softsameNN',outside_seconds=150 if mode=='cpu'else 900,inside_seconds=None if mode=='cpu'else 850,AS_GiB=64 if mode=='cpu'else 512,file_limit_MiB=100,GPU_fraction=None if mode=='cpu'else.3,runner_sha256=CPU if mode=='cpu'else GPU,remote_launch_sha256=sha(version/'remote-launch.source'),formal_Primary=False,full_training_FLOPs=None,provider_effective_compute=None)
 put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=name,source_sha256=hashes,protocol_sha256=sha(protocol),remote_sha256=doc['remote_launch_sha256'])))
def execute(mode,arm,sampling):
 name=name_for(mode,arm,sampling);v=ROOT/'versions'/('code-'+name);config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(name+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text()
 literal=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));frozen=ast.literal_eval(literal.value)
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or frozen!=config or source!=REMOTE_SCRIPT.replace('CONFIG',repr(frozen),1):raise ValueError('frozen source/config differs')
 for k,h in config['source_sha256'].items():
  if sha(v/k)!=h:raise ValueError('frozen code differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('one actual attempt only; preserve receipt')
 begin=time.monotonic()
 try:x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=210 if mode=='cpu'else 980);r=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-begin)
 except subprocess.TimeoutExpired as e:r=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-begin,error='SSH wait timeout; inspect existingboundedrunner noautomaticretry')
 r['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,r);print(json.dumps(dict(name=name,returncode=r['returncode'],wall_seconds=r['wall_seconds'])));raise SystemExit(r['returncode']if r['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('cpu','train'));g.add_argument('--execute',choices=('cpu','train'));p.add_argument('--arm',choices=('gru32','transformer32'));p.add_argument('--sampling',choices=('paired','same_pool_shuffle'));a=p.parse_args();mode=a.prepare or a.execute
 if mode=='train'and(a.arm is None or a.sampling is None):p.error('fixed arm and sampling required for train')
 (prepare if a.prepare else execute)(mode,a.arm,a.sampling)
