"""Freeze or ROOT-execute one fixed zero-SGD CPU suffix memory intervention."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-suffix-memory-intervention-cpu-v1'
REMOTE='/workspace/prometheus-v2-m0-architecture'
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
AUDIT='audit_typed_suffix_memory_intervention_cpu.py'
TEMPLATE=next(ast.literal_eval(n.value)for n in ast.parse((ROOT/'launch_value_bridge_cpu_owned.py').read_text()).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='TEMPLATE'for t in n.targets))
TEMPLATE=TEMPLATE.replace('check_value_bridge_cpu.py',AUDIT).replace("out/'check'","out/'audit'").replace("out/'check/result.json'","out/'audit/result.json'").replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','COMPLETED_TYPED_SUFFIX_MEMORY_INTERVENTION_CPU')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
 with path.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def prepare():
 version=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if version.exists()or protocol.exists():raise FileExistsError('preserve namespace')
 old='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2'
 oldversion=ROOT/'versions'/('code-'+old)
 sources={leaf:(oldversion/leaf).read_bytes()for leaf in('continuous_primitive_block.py','owned_fcba_core.py')}
 sources[AUDIT]=(ROOT/AUDIT).read_bytes()
 expected={'continuous_primitive_block.py':'efabc8cbd2754a4a1d60f52b907cb5842f11fcf716c52646f8f935db5170d3f2','owned_fcba_core.py':'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158',AUDIT:'016a7ec9fba33ecfad1e87b513f9732e42a2aa4c45e607b3fda69a33b84b0a74'}
 for leaf,raw in sources.items():
  ast.parse(raw)
  if hashlib.sha256(raw).hexdigest()!=expected[leaf]:raise ValueError('reviewed source differs: '+leaf)
 files={}
 def owned(leaf,relative,digest=None):
  source=ROOT/'runs'/relative
  if source.is_symlink()or not source.is_file():raise ValueError('OWN regular input required')
  actual=sha(source)
  if digest is not None and actual!=digest:raise ValueError('exact input differs: '+leaf)
  files[leaf]=dict(remote_relative=relative,sha256=actual,bytes=source.stat().st_size)
 owned('valid.npz','parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz','7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b')
 models=[]
 specs=[('paired','parameter-typed-paired-tail-paired-n8-seed0-600-v1',3,'paired','typed-paired-tail-stage3-step600.pt','5bb678b87a688cb373b001b5054a1cacbc36b5cd2f3c49012bc97658345a6bc8',10991),('control','parameter-typed-paired-tail-same-pool-shuffle-n8-seed0-600-v1',3,'same_pool_shuffle','typed-paired-tail-stage3-step600.pt','cbaf9828486a17458e45e561a7ff2e47c17da049cafd84e809f9563b27280abb',10991)]
 for key,name,stage,sampling,checkpoint,digest,P in specs:
  result=json.loads((ROOT/'runs'/name/'learn/result.json').read_text())
  leaf=key+'.pt';owned(leaf,name+'/learn/'+checkpoint,digest)
  item=dict(key=key,stage=stage,sampling=sampling,leaf=leaf,sha256=digest,bytes=files[leaf]['bytes'],P=P,checkpoint_schema='typed-memory-actions-fourstage-graph-training-v1'if stage==2 else'typed-paired-tail-stage3-training-v1',parent_source_sha256=result['source_sha256'])
  event=key+'-saved-events.npz';owned(event,'parameter-typed-three-nn-valid-actions-cpu-v1/audit/'+key+'-NN-events-before-truth.npz');item.update(event_leaf=event,event_sha256=files[event]['sha256'])
  models.append(item)
 manifest=dict(schema='typed-suffix-memory-intervention-CPU-v1',source_sha256=expected,models=models,donor_group_offset=127)
 config=dict(name=NAME,remote=REMOTE,runner_SHA=CPU,source_sha256=expected,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest)
 config=json.loads(json.dumps(config,sort_keys=True))
 source=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(source)
 # Reused coordinator initializes all paths before any filesystem operation.
 prefix=ast.parse(source).body
 assert isinstance(prefix[1],ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in prefix[1].targets)
 version.mkdir()
 for leaf,raw in sources.items():
  with(version/leaf).open('xb')as f:f.write(raw)
  (version/leaf).chmod(0o444)
 rp=version/'remote-launch.source';rp.write_text(source);rp.chmod(0o444);put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',source_sha256=expected,input_files=files,models=models,runner_sha256=CPU,remote_launch_sha256=sha(rp),outside_seconds=150,science_seconds=120,AS_GiB=64,file_limit_MiB=100,CPU_only=True,SGD_updates=0,NNs=2,conditions=['sham','zero_V','donor_V'],NN_forward_transitions=12288,all_raw_before_targets=True,privileged_truth_loaded=False,donor_group_offset=127,donor_rule='first-row-encounter public prefixgroup (initial,tokens[:-4]); donor=(group+127)%256 first row; same for both query members; no labels',entry='saved event4 beforecarry, exact within-prefix equality required',sham_gate=dict(atol=2e-5,rtol=2e-5,argmax_differences=0),no_accuracy_gate=True,checkpoint_unchanged_required=True,full_FLOPs=None,provider_effective_compute=None,cost_scope='all six 4-token suffix passes, checkpoint/eventraw copies/setup/CPU/RSS/rawIO; no optimizer or training; zeroV and mismatched donorV are OOD interventions')
 put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555)
 print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=expected,remote_launch_sha256=sha(rp),protocol_sha256=sha(protocol)),sort_keys=True))
def execute():
 version=ROOT/'versions'/('code-'+NAME);config=json.loads((version/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(version/'remote-launch.source').read_text()
 if sha(version/'remote-launch.source')!=protocol['remote_launch_sha256']:raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():
  if sha(version/leaf)!=digest:raise ValueError('frozen source differs')
  ast.parse((version/leaf).read_bytes())
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));frozen=ast.literal_eval(node.value)
 if frozen!=config or source!=TEMPLATE.replace('CONFIG',repr(frozen),1):raise ValueError('frozen config/template differs')
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actual receipt, no retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as error:value=dict(returncode=None,stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspect bounded actual state, no retry')
 value.update(name=NAME,remote_launch_sha256=sha(version/'remote-launch.source'),source_sha256=config['source_sha256']);put(receipt,value);print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');args=parser.parse_args()
 if args.prepare:prepare()
 else:execute()
