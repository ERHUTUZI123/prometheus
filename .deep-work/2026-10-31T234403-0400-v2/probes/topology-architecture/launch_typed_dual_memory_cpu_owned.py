"""Freeze/ROOT-execute one zero-optimizer CPU dual-memory admission."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-dual-memory-cpu-v1'
PARENT='parameter-typed-normalized-memory-ab-n8-seed0-fourstage-graph-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def template():return transport.TEMPLATE.replace('check_value_bridge_cpu.py','check_typed_dual_memory_cpu.py').replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','CPU_TYPED_DUAL_MEMORY_ADMISSION_PASS')
def prepare():
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespace only')
 parent=ROOT/'versions'/('code-'+PARENT);result=json.loads((ROOT/'runs'/PARENT/'learn/result.json').read_text());cp=result['checkpoint'];checkpoint=ROOT/'runs'/PARENT/'learn'/cp['file']
 if cp['sha256']!='912abc79925e9014146fafdec7f538282198bbddcc44034a8d7852021ebfbf85'or sha(checkpoint)!=cp['sha256']or checkpoint.stat().st_size!=cp['bytes']:raise ValueError('exact OWN AB final checkpoint required')
 common={'owned_fcba_core.py':(parent/'owned_fcba_core.py').read_bytes(),'owned_ab_core.py':(parent/'continuous_primitive_block.py').read_bytes(),'continuous_primitive_block.py':(ROOT/'continuous_primitive_block_typed_dual_memory.py').read_bytes(),'typed_dual_memory_masked_fb.py':(ROOT/'typed_dual_memory_masked_fb.py').read_bytes()}
 if hashlib.sha256(common['owned_ab_core.py']).hexdigest()!='93bdfea778c2f079ec04a6696ab8df77dc8bcc52b7f3fabdc57c05cbcad5250f'or hashlib.sha256(common['owned_fcba_core.py']).hexdigest()!='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158':raise ValueError('frozen AB/base source differs')
 sources=dict(common);sources['check_typed_dual_memory_cpu.py']=(ROOT/'check_typed_dual_memory_cpu.py').read_bytes()
 for raw in sources.values():ast.parse(raw)
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};common_hashes={leaf:hashes[leaf]for leaf in common};files={'ab.pt':dict(remote_relative=PARENT+'/learn/'+cp['file'],sha256=cp['sha256'],bytes=cp['bytes'])}
 manifest=dict(schema='typed-dual-memory-CPU-v1',source_sha256=common_hashes,parent_source_sha256=result['source_sha256'],checkpoint_sha256=cp['sha256'],checkpoint_bytes=cp['bytes'],no_task_data=True,no_optimizer=True)
 config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=transport.CPU,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest);config=json.loads(json.dumps(config,sort_keys=True));remote=template().replace('CONFIG',repr(config),1);ast.parse(remote);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 rp=v/'remote-launch.source';rp.write_text(remote);rp.chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',formal_Primary=False,source_sha256=hashes,common_source_sha256=common_hashes,input_files=files,parent_source_sha256=result['source_sha256'],parent='actualfrozenABmodelthetaONLY, neveroptimizer/RNG/replay/episodeV/zinheritance',P=10351,parameter_tensors=20,carry_scalars=384,carry_bytes_FP32=1536,additional_episode_state_vs_AB_bytes=224,shared_nonpersistent_identity_FP32_bytes=256,initial_C='exactzerosineverynewepisode; oldcoveragecannotbeinterpretedasC',C_rule='Cj+=w_j(k-Cj), w=alphaSTORE*k; OLDhVwriteunchanged',read_rule='z=C.sum(-1); Cbar/Vbar dividedbymax(z,1e-4); solve(CbarCbarT+1e-4I)b=Cbark; r=bTVbar; signedretrieval',CPU_checks='actualtwooverlappingindependentkeys/consistentvalues/ridgebias/residual/condition/dualnormbound; identicalkeycounterexample; exactABtheta20/P; finitefullNNF/Ball20/raggedmasked/reload/newcheckpoint; noVALIDorTEST',SGD_updates=0,optimizer_updates=0,synthetic_optimizer_updates=0,new_training_budget=0,completed_three_arm_training_updates_remain=7200,not_a_new_trained_architecture_arm=True,no_score_gate=True,no_training_permission_added=True,outside_seconds=150,science_seconds=120,address_space_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,remote_launch_sha256=sha(rp),full_training_FLOPs=None,provider_effective_compute=None,cost_scope='alltheta-onlyload/initialization/syntheticFFB/solveforwardbackward/equations/reload/source/checkpoint/rawIO/CPU/RSS measured; dense solver internals/workspace/fullFLOPs unknown; historicalABtrainingcostretainedseparately',failure_predictions='erasedidenticalkeys/rankcollapse/conditioning/noise/ridgebias/conflictingrepeatedkeyEMAoverwrite; finiteCPUchecknotlearning/binding/GPUcaptureevidence',GPU_status='notadmitted; solve synchronizes/errorstatus/capture/runtimecost requireseparatefutureauthorizationandactualcheck',novelty='ordinaryridge/dualassociativememoryprecedent, no originalmechanismclaim')
 put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=hashes,remote_launch_sha256=sha(rp),protocol_sha256=sha(protocol)),sort_keys=True))
def execute():
 v=ROOT/'versions'/('code-'+NAME);config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=template().replace('CONFIG',repr(config),1):raise ValueError('frozen coordinator differs')
 for leaf,h in config['source_sha256'].items():
  if sha(v/leaf)!=h:raise ValueError('frozen source differs')
  ast.parse((v/leaf).read_bytes())
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve everyattempt, no automatic retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(transport.SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspectboundedstate, no retry')
 value['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,value);print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=parser.parse_args()
 if a.prepare:prepare()
 else:execute()
