"""Freeze/ROOT-execute one persistent input-address CPU admission, no task data."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAME='parameter-persistent-input-address-cpu-v1'
FIXED={'owned_fcba_core.py':'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158','persistent_address_masked_graph_fb.py':'f3e56ead0e54c39c718e9c551dd8026e78be28bc800235aa2a9b74fb5db70598','owned_parent_driver.py':'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc','context_replay128.py':'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140','memory_live_stage_train.py':'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
def template():return transport.TEMPLATE.replace('check_value_bridge_cpu.py','check_persistent_address_cpu.py').replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','CPU_PERSISTENT_INPUT_ADDRESS_GRAPH_ADMISSION_PASS').replace(".get('P')",".get('P16')")
def prepare():
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespace only')
 old=ROOT/'versions/code-parameter-factorized-value-anchor0-n8-seed0-fourstage-v3';common={}
 for leaf,digest in FIXED.items():
  source=old/'continuous_primitive_block.py'if leaf=='owned_fcba_core.py'else ROOT/leaf if leaf=='persistent_address_masked_graph_fb.py'else old/leaf
  if sha(source)!=digest:raise ValueError('exact trustedsource changed: '+leaf)
  common[leaf]=source.read_bytes()
 common['continuous_primitive_block.py']=(ROOT/'continuous_primitive_block_persistent_address.py').read_bytes();sources=dict(common);sources['check_persistent_address_cpu.py']=(ROOT/'check_persistent_address_cpu.py').read_bytes()
 for raw in sources.values():ast.parse(raw)
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};common_hashes={leaf:hashes[leaf]for leaf in common};manifest=dict(schema='persistent-input-address-CPU-v1',source_sha256=common_hashes,no_historical_NN_or_task_data=True)
 config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=transport.CPU,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files={},manifest=manifest)
 config=json.loads(json.dumps(config,sort_keys=True));remote=template().replace('CONFIG',repr(config),1);ast.parse(remote);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 (v/'remote-launch.source').write_text(remote);(v/'remote-launch.source').chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,before_execution=True,status='FROZEN_UNEXECUTED',source_sha256=hashes,common_source_sha256=common_hashes,seed=0,expectedP15=10550,expectedP16=10566,parameter_tensors=25,logical_carry_scalars=264,architecture='RAWvalue/probmemory/cache; input-onlyGRU; persistentz16 fromrawtokenproposal/scalargatebias-2; query=z/no bypass; sharedREADWRITEattention; fixedtrainablekeys',checkpoint_version='continuous-primitive-block-persistent-input-address-v1',policy='input-gated/persistent-address/static-episode-keys/probvalue-cache/group8/v1',CPU_checks='retainedinitializationsubset/RNG; all25 originalgather/maskedgrad16/48; controllerindependenth/addressindependenth+c; sharedattention; fixedkeysandnonzerograd; cachedpublicAPI/causality/ragged/empty; discardedAdam+growth/reload/replay; old/wrongpolicyreject',output_tolerance=[2e-5,2e-5],gradient_tolerance=[2e-5,2e-4],SGD_updates=0,research_updates=0,synthetic_optimizer_updates='checker-only/discarded; actualcountinresult',no_task_labels_or_teacher=True,fresh_GPU_NN_must_not_load_CPU_synthetic_checkpoint=True,no_accuracy_gate=True,permissions='unchanged c9 boundedCPUrunner; ROOT-onlyexecute',outside_seconds=150,address_space_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,costs='allsyntheticF-B/Adam/growth/reload/replay/CPU/RSS/source/raw paid; nofullFLOPs/provider-effective claim',scientific_scope='one persistentrawinputgatedregister repair within RAW/probmemory/staticK Block; no novelty/guaranteedbinding',remote_launch_sha256=sha(v/'remote-launch.source'))
 put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=hashes,common_source_sha256=common_hashes,remote_launch_sha256=doc['remote_launch_sha256'],protocol_sha256=sha(protocol))))
def execute():
 v=ROOT/'versions'/('code-'+NAME);config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=template().replace('CONFIG',repr(config),1):raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():
  if sha(v/leaf)!=digest:raise ValueError('frozen source differs')
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
