"""Freeze or ROOT-execute ONLY the one-scalar bridge CPU admission."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-value-bridge-graph-cpu-v1'
REMOTE='/workspace/prometheus-v2-m0-architecture'
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
FIXED={'owned_fcba_core.py':'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158','factorized_masked_graph_fb.py':'f2e3e16d75c02ce68b62189ab808debe71c5ed43e0f12107c212fcd0341af4f5','owned_parent_driver.py':'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc','context_replay128.py':'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140','memory_live_stage_train.py':'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
TEMPLATE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);code=r/('code-'+c['name']);out=r/'outputs'/c['name'];inputs=pathlib.Path('/opt/prometheus-'+c['name']+'-inputs');status=out.parent/(out.name+'.runner-status.json');receipt=out.parent/(out.name+'.coordinator-exit.json')
if any(p.exists()or p.is_symlink()for p in(code,out,inputs,status,receipt)):raise FileExistsError('newnamespace only; preserve evidence')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
runner=r/'remote_runner_cpu_v2.py'
if sha(runner)!=c['runner_SHA']:raise ValueError('unchanged CPUrunner differs')
code.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded,validate=True)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport differs')
 with(code/leaf).open('xb')as f:f.write(raw)
 (code/leaf).chmod(0o444)
code.chmod(0o555);inputs.mkdir()
for leaf,spec in c['input_files'].items():
 source=r/'outputs'/spec['remote_relative']
 if source.is_symlink()or not source.is_file()or sha(source)!=spec['sha256']or source.stat().st_size>100*1024*1024:raise ValueError('ownedinput differs: '+leaf)
 shutil.copyfile(source,inputs/leaf);(inputs/leaf).chmod(0o444)
with(inputs/'manifest.json').open('x')as f:json.dump(c['manifest'],f,indent=2,sort_keys=True);f.write('\n')
(inputs/'manifest.json').chmod(0o444);inputs.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'check_value_bridge_cpu.py'),'--out',str(out),'--timeout','150','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--inputs',str(inputs),'--out',str(out/'check')];started=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=180);rc=x.returncode;stdout=x.stdout;stderr=x.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or'');stderr=str(e.stderr or'');error='coordinatorwaittimeout, no retry'
for suffix,value in(('coordinator-stdout.log',stdout),('coordinator-stderr.log',stderr)):
 with(out.parent/(out.name+'.'+suffix)).open('x')as f:f.write(value)
def load(p):return json.loads(p.read_text())if p.is_file()else None
rs=load(status);result=load(out/'check/result.json');passed=bool(rc==0 and rs and rs.get('status')=='completed'and result and result.get('status')=='CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS')
value=dict(name=c['name'],returncode=rc,protocol_completed=passed,error=error,whole_runner_wall_seconds=time.monotonic()-started,command=cmd,result_path=str(out/'check/result.json'),root_status_path=str(status),source_sha256=c['source_sha256'],common_source_sha256=c['manifest']['source_sha256'],actual_P=(result or{}).get('P'),research_updates=0)
with receipt.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(value,sort_keys=True));raise SystemExit(0 if passed else(rc if rc not in(None,0)else 1))
'''
def prepare():
 version=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if version.exists()or protocol.exists():raise FileExistsError('newfreeze only')
 old=ROOT/'versions/code-parameter-factorized-value-anchor0-n8-seed0-fourstage-v3';oldconfig=json.loads((old/'launch-config.json').read_text());common={}
 for leaf,digest in FIXED.items():
  source=(old/'continuous_primitive_block.py')if leaf=='owned_fcba_core.py'else(ROOT/leaf)if leaf=='factorized_masked_graph_fb.py'else(old/leaf)
  if sha(source)!=digest:raise ValueError('exact original/helper source differs: '+leaf)
  common[leaf]=source.read_bytes()
 common['continuous_primitive_block.py']=(ROOT/'continuous_primitive_block_value_bridge.py').read_bytes();sources=dict(common);sources['check_value_bridge_cpu.py']=(ROOT/'check_value_bridge_cpu.py').read_bytes()
 for raw in sources.values():ast.parse(raw)
 common_sha={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in common.items()};sources_sha={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 files={'resume.pt':dict(remote_relative='parameter-factorized-value-anchor0-n8-seed0-fourstage-v2/learn/factorized-anchor0-stage2-step600.pt',sha256='9bb47e660e4ea67d0bf1a06e93e418f0c3fa50760fa76ff067d3a0c954a6c40f'),'fixed-cases.npz':dict(remote_relative='parameter-factorized-graph-fb-cpu-admission-v1/fixed-cases.npz',sha256=sha(ROOT/'runs/parameter-factorized-graph-fb-cpu-admission-v1/fixed-cases.npz'))}
 for spec in files.values():
  source=ROOT/'runs'/spec['remote_relative']
  if sha(source)!=spec['sha256']:raise ValueError('exactlocalOWNinput changed')
  spec['bytes']=source.stat().st_size
 manifest=dict(schema='soft-value-bridge-CPU-admission-inputs-v1',source_sha256=common_sha,parent_source_sha256=oldconfig['manifest']['resume']['source_sha256'],parent_checkpoint_sha256=files['resume.pt']['sha256'],cases_sha256=files['fixed-cases.npz']['sha256'])
 config=dict(name=NAME,remote=REMOTE,runner_SHA=CPU,source_sha256=sources_sha,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest)
 source=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(source);version.mkdir()
 for leaf,raw in sources.items():
  with(version/leaf).open('xb')as f:f.write(raw)
  (version/leaf).chmod(0o444)
 rp=version/'remote-launch.source';rp.write_text(source);rp.chmod(0o444);put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',before_execution=True,formal_Primary=False,source_sha256=sources_sha,common_source_sha256=common_sha,input_files=files,expected_P=13398,new_P=1,scalar='value_bridge_alpha init0, tanhstrength maynegative',bridge='afteroldG8/beforeunchangedWRITE; normpreserving continuous learnedreadoutposterior→sharednormalizedencoderbasis residual; nocontroller/memoryreset/noargmax/nointermediatetruth',checkpoint_version='continuous-primitive-block-soft-value-bridge-v1',policy='learned-control-history/value-identity-write/group8-carry+soft-value-bridge/v1',CPU_gate='old27NN/Adam/RNG/replay+zeroalpha factoryneutral migration, a0 exactsoft/native, originaltokenrow, 1discardedsyntheticAdam scalarflow+realreload, 3existingfixedTRAINcases original/masked/repeat all28gradients+output at fixedf2etolerances; no researchSGD',SGD_updates=0,synthetic_optimizer_updates=1,CPU_is_not_CUDA=True,anchor0_scope='fixedhelper retains originaltargetlookup/MSE*0 physicalwork, coefficient0 so optimizationNLL-only; never auxiliaryfit or additionalteacher',costs='all synthetic+TRAIN0SGDFB/serialization/CPU/RSS/storage paid; fullFLOPs/provider-effective unknown',outside_seconds=150,AS_GiB=64,file_limit_MiB=100,runner_sha256=CPU,remote_launch_sha256=sha(rp),no_new_permissions=True)
 put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=sources_sha,common_source_sha256=common_sha,remote_launch_sha256=sha(rp),protocol_sha256=sha(protocol))))
def execute():
 version=ROOT/'versions'/('code-'+NAME);config=json.loads((version/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(version/'remote-launch.source').read_text()
 if sha(version/'remote-launch.source')!=protocol['remote_launch_sha256']:raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():
  if sha(version/leaf)!=digest:raise ValueError('frozen source differs')
  ast.parse((version/leaf).read_bytes())
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));frozen=ast.literal_eval(node.value)
 if frozen!=config or source!=TEMPLATE.replace('CONFIG',repr(frozen),1):raise ValueError('frozen config/template differs')
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actualreceipt, no retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspectboundedprocess, no retry')
 value['remote_launch_sha256']=sha(version/'remote-launch.source');put(receipt,value);print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=parser.parse_args()
 if a.prepare:prepare()
 else:execute()
