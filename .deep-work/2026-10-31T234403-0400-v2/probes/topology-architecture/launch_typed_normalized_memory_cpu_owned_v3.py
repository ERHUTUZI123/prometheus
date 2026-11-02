"""Freeze/ROOT-execute one fixed three-mode A/B normalized-memory CPU admission, no task data."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-normalized-memory-cpu-v3'
FIXED={'owned_fcba_core.py':'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158','typed_normalized_memory_masked_graph_fb.py':'c322fb82f1095ae63463c9c584cf9cba05345514bdf71cb0a9a25b32b0f29294','owned_typed_memory_core.py':'efabc8cbd2754a4a1d60f52b907cb5842f11fcf716c52646f8f935db5170d3f2','owned_parent_driver.py':'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc','context_replay128.py':'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140','memory_live_stage_train.py':'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
def template():return transport.TEMPLATE.replace('check_value_bridge_cpu.py','check_typed_normalized_memory_cpu.py').replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','CPU_TYPED_NORMALIZED_MEMORY_GRAPH_ADMISSION_PASS').replace(".get('P')",".get('P16')")
def prepare():
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespace only')
 old=ROOT/'versions/code-parameter-factorized-value-anchor0-n8-seed0-fourstage-v3';common={}
 for leaf,digest in FIXED.items():
  source=old/'continuous_primitive_block.py'if leaf=='owned_fcba_core.py'else ROOT/'versions/code-parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2/continuous_primitive_block.py'if leaf=='owned_typed_memory_core.py'else ROOT/leaf if leaf=='typed_normalized_memory_masked_graph_fb.py'else old/leaf
  if sha(source)!=digest:raise ValueError('exact trustedsource changed: '+leaf)
  common[leaf]=source.read_bytes()
 common['continuous_primitive_block.py']=(ROOT/'continuous_primitive_block_typed_normalized_memory.py').read_bytes();sources=dict(common);sources['check_typed_normalized_memory_cpu.py']=(ROOT/'check_typed_normalized_memory_cpu.py').read_bytes()
 for raw in sources.values():ast.parse(raw)
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};common_hashes={leaf:hashes[leaf]for leaf in common};manifest=dict(schema='typed-normalized-memory-CPU-v1',source_sha256=common_hashes,no_historical_NN_or_task_data=True)
 config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=transport.CPU,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files={},manifest=manifest)
 config=json.loads(json.dumps(config,sort_keys=True));remote=template().replace('CONFIG',repr(config),1);ast.parse(remote);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 (v/'remote-launch.source').write_text(remote);(v/'remote-launch.source').chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,before_execution=True,status='FROZEN_UNEXECUTED',source_sha256=hashes,common_source_sha256=common_hashes,seed=0,modes={'A':dict(input_only_address=True,coverage_normalization=False,P15=10335,P16=10351,carry=320),'B':dict(input_only_address=False,coverage_normalization=True,P15=10975,P16=10991,carry=328),'AB':dict(input_only_address=True,coverage_normalization=True,P15=10335,P16=10351,carry=328)},parameter_tensors=20,coverage_epsilon=1e-4,checkpoint_version='continuous-primitive-block-typed-normalized-memory-v1',CPU_checks='newArawxproposal/fullcontextalpha; BexactV+coverage/empty/singlewrite/eps/multiwrite; originalvsmaskedall20grad/fullcarry; strictflags/checkpoint/growth; commonfactory/RNG/subset',SGD_updates=0,research_updates=0,synthetic_optimizer_updates=1,no_task_labels_or_teacher=True,fresh_GPU_NN_must_not_load_CPU_synthetic_checkpoint=True,no_accuracy_gate=True,permissions='unchanged c9 boundedCPUrunner; ROOT-onlyexecute',outside_seconds=150,address_space_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,costs='allsynthetic/F-B/oneABdiscardedAdam/growth/reload/CPU/RSS/source/raw paid; fullFLOPs/provider-effective unknown',scientific_scope='newpercellcoverage zstate withV/max(z,eps), notposthocscalarreadnormalization; Aprunes640P; Badds32B/example, noP; no bindingguarantee/noveltyclaim',remote_launch_sha256=sha(v/'remote-launch.source'))
 doc['prior_attempts']=[dict(name='parameter-typed-normalized-memory-cpu-v1',status='UNEXECUTED_import_name_race_preserved'),dict(name='parameter-typed-normalized-memory-cpu-v2',status='ACTUAL_CHECKER_SCOPE_FAILURE_NO_RESEARCH_SGD',SSH_seconds=9.983725708,runner_seconds=8.862430614,A_mode_seconds=4.354358299,failure_scope='detachedcarry/directraw bypass exactlyencoder/tokenembedding; v3namedNonepatternfix only, fullsequenceall20 unchanged',receipt_sha256=sha(ROOT/'parameter-typed-normalized-memory-cpu-v2-coordinator-receipt.json'))]
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
