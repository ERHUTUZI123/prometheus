"""Freeze/ROOT-execute one0SGDCPUjob on exactly Graphbase7d8 andbridge19fb."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
from extended_closure_launch import checkpoint_header
ROOT=Path(__file__).resolve().parent
NAME='parameter-value-interface-two-owned-cpu-v1'
REGISTRY={
 'baseline':dict(parent='parameter-factorized-graph-stage3-anchor0-n8-seed0-600-v1',checkpoint='factorized-graph-stage3-step600.pt',sha256='7d8a75f960bd55fedbe4afa68eccf59b18d05de759b2eb1e0724818c3acedd5d',schema='factorized-graph-stage3-training-v1',P=13397,status='ACTUAL_FACTORIZED_GRAPH_STAGE3_600_COMPLETE'),
 'bridge':dict(parent='parameter-value-bridge-graph-n8-seed0-stage3-v3',checkpoint='value-bridge-graph-stage3-step600.pt',sha256='19fb4fc6c07326c494a44d7f573066515751a969efcc5e8776b6d96efb085772',schema='soft-value-bridge-graph-stage3-training-v1',P=13398,status='ACTUAL_SOFT_VALUE_BRIDGE_GRAPH_STAGE3_600_COMPLETE')}
CORE='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
BRIDGE='f1ccdc83fbae14754e4271da9cdfb949f4f656c9704e35df283f899806c95da3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
def template():
 s=transport.TEMPLATE.replace('check_value_bridge_cpu.py','audit_value_interface_owned_cpu.py').replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','COMPLETED_TWO_OWNED_VALUE_INTERFACE_CPU_AUDIT')
 needle='code.mkdir()\n'
 guard="for spec in c['manifest']['models'].values():\n for leaf,digest in spec['parent_source_sha256'].items():\n  if sha(r/('code-'+spec['parent'])/leaf)!=digest:raise ValueError('actual parent source changed')\n"
 if s.count(needle)!=1:raise ValueError('exact coordinator injection boundary missing')
 return s.replace(needle,guard+needle)

def prepare():
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespace only')
 models={};input_files={}
 for key,expected in REGISTRY.items():
  parent=expected['parent'];folder=ROOT/'runs'/parent/'learn';result=json.loads((folder/'result.json').read_text());original=ROOT/'versions'/('code-'+parent);frozen=json.loads((ROOT/(parent+'-protocol.json')).read_text())
  if result.get('status')!=expected['status']or result['resources']['parameters']!=expected['P']or result['checkpoint']['file']!=expected['checkpoint']or result['checkpoint']['sha256']!=expected['sha256']or result.get('actual_new_updates_this_process')!=600:raise ValueError('actual fixed NN result mismatch')
  if result['source_sha256']!=frozen['source_sha256']:raise ValueError('actual/frozenNNsource mismatch')
  for leaf,digest in result['source_sha256'].items():
   if sha(original/leaf)!=digest:raise ValueError('actualoriginalsource changed')
  cp=folder/expected['checkpoint']
  if cp.is_symlink()or cp.stat().st_size!=result['checkpoint']['bytes']or sha(cp)!=expected['sha256']or checkpoint_header(cp)!=dict(schema=expected['schema'],stage=3,step=600,stage_complete=True):raise ValueError('actualNNbytes/header mismatch')
  models[key]=dict(parent=parent,leaf=key+'.pt',sha256=expected['sha256'],bytes=cp.stat().st_size,schema=expected['schema'],P=expected['P'],parent_source_sha256=result['source_sha256'],parent_result_sha256=sha(folder/'result.json'),historical_resources=result['resources'],historical_cost_scope='actual NN/Adam/replay/training costs retained in parent, not zeroed by this0SGDjob')
  input_files[key+'.pt']=dict(remote_relative=str(cp.relative_to(ROOT/'runs')),sha256=expected['sha256'],bytes=cp.stat().st_size)
 base=ROOT/'versions/code-parameter-factorized-graph-stage3-anchor0-n8-seed0-600-v1/continuous_primitive_block.py';bridge=ROOT/'versions/code-parameter-value-bridge-graph-n8-seed0-stage3-v3/continuous_primitive_block.py'
 if sha(base)!=CORE or sha(bridge)!=BRIDGE:raise ValueError('exactactualNNclasssource mismatch')
 sources={'owned_fcba_core.py':base.read_bytes(),'value_bridge_core.py':bridge.read_bytes(),'audit_value_interface_owned_cpu.py':(ROOT/'audit_value_interface_owned_cpu.py').read_bytes()}
 for raw in sources.values():ast.parse(raw)
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};manifest=dict(schema='two-owned-value-interface-CPU-v1',models=models,source_sha256=hashes)
 config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=transport.CPU,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=input_files,manifest=manifest)
 config=json.loads(json.dumps(config,sort_keys=True));remote=template().replace('CONFIG',repr(config),1);ast.parse(remote);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 (v/'remote-launch.source').write_text(remote);(v/'remote-launch.source').chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',before_execution=True,source_sha256=hashes,models=models,SGD_updates=0,task_data_labels_or_oracles=False,only_two_actual_NNs=True,scope='publicinitialvalues0..7→actualsavedindependentreadout; all64logp/argmax/lognormalization+prototypeGram/norms andhypotheticalR/R² saved; learnedinterfaceproperty nottaskorfullcarryaccuracy',calculation='E=encode_state(arange8); logp=logsoftmax(readout(E)); R(v)=||v||normalize(softmax(readout(v))@E), norms/dot/geometryonly; no_transition/token/taskinputs',raw_evidence='perNN NPZ includesall64logp, E, readoutW/b, Gram/norms, R(E), R²(E), fixedpoint/idempotenceresiduals; resultJSONalsoincludesall64logp',costs='oneCPUjob,2readoutB8calls+2B8weightedbasisproducts+Gram=10240linear/dotMACproxy perNN; nonlinearity/norm/cdist/IO/wholewall/RSS/NN/checkpoint/historicalAdam/replay storagecharged; fullFLOPs/provider-effective unknown',no_alpha_or_temperature_tuning=True,no_fresh_model_or_training=True,formal_Primary=False,outside_seconds=150,address_space_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,remote_launch_sha256=sha(v/'remote-launch.source'))
 put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=hashes,remote_launch_sha256=doc['remote_launch_sha256'],protocol_sha256=sha(protocol))))
def execute():
 v=ROOT/'versions'/('code-'+NAME);config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=template().replace('CONFIG',repr(config),1):raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():
  if sha(v/leaf)!=digest:raise ValueError('frozen source differs')
 for spec in config['input_files'].values():
  path=ROOT/'runs'/spec['remote_relative']
  if sha(path)!=spec['sha256']or path.stat().st_size!=spec['bytes']:raise ValueError('actualNNinput differs')
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserveattempt/no automaticretry')
 tick=time.monotonic()
 try:
  x=subprocess.run(transport.SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspectboundedrun, no retry')
 value['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,value);print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=parser.parse_args()
 if a.prepare:prepare()
 else:execute()
