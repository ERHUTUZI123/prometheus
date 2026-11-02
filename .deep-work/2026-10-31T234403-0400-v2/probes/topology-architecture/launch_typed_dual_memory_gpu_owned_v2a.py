"""Freeze or ROOT-execute one bounded dual-memory GPU execution diagnostic."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-dual-memory-gpu-execution-v2a'
RUNNER='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
PARENT='parameter-typed-normalized-memory-ab-n8-seed0-fourstage-graph-v1'
CPU='parameter-typed-dual-memory-cpu-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def template():
 s=transport.TEMPLATE.replace('remote_runner_cpu_v2.py','remote_runner_gpu_authorized.py').replace('check_value_bridge_cpu.py','check_typed_dual_memory_gpu_v2.py').replace("out/'check'","out/'admit'").replace("out/'check/result.json'","out/'admit/result.json'").replace("'--address-space-gib','64'","'--address-space-gib','512'").replace("'--cpu-only'","'--gpu-fraction','.3','--cuda-init-before-seccomp'").replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','GPU_DUAL_MEMORY_EXECUTION_REPAIR_V2_COMPLETE')
 s=s.replace("actual_P=(result or{}).get('P'),research_updates=0)","actual_P=(result or{}).get('P'),research_updates=0,capture_supported=(result or{}).get('capture',{}).get('supported'),numerical_GPU_admission=(result or{}).get('numerical_GPU_admission'),GPU_training_admitted=False)")
 return s
def prepare():
 if sha(ROOT/'remote_runner_gpu_authorized.py')!=RUNNER:raise ValueError('verifiedGPUrunnerhashdiffers')
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespaceonly/preserveallattempts')
 cpudir=ROOT/'versions'/('code-'+CPU);cpuconfig=json.loads((cpudir/'launch-config.json').read_text());proof=ROOT/'runs'/CPU/'check/result.json';CPUresult=json.loads(proof.read_text())
 if sha(proof)!='ec6ac98cbf62480645f50e44f0d4dbc7a4c7897afcbed56263292c7ce95dc970'or CPUresult['status']!='CPU_TYPED_DUAL_MEMORY_ADMISSION_PASS':raise ValueError('fixedactualCPUadmissionrequired')
 oldcommon={leaf:(cpudir/leaf).read_bytes()for leaf in('continuous_primitive_block.py','owned_ab_core.py','owned_fcba_core.py','typed_dual_memory_masked_fb.py')}
 if {leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in oldcommon.items()}!=CPUresult['source_sha256']:raise ValueError('preservedv1CPUsource differs')
 failed_path=ROOT/'runs/parameter-typed-dual-memory-gpu-execution-v1a/admit/result.json';failed=json.loads(failed_path.read_text())
 if sha(failed_path)!='3281571511bb71691c39911f90f5b9c339742e32111836c2da298eb04374a394'or failed['capture']['supported']is not False or failed['evidence']['CPU_GPU_eager']['fixed_tolerances_pass']is not False:raise ValueError('actualv1failure requiredbeforeONEexecutionrepair')
 common={leaf:raw for leaf,raw in oldcommon.items()if leaf!='continuous_primitive_block.py'};common['owned_dual_v1_core.py']=oldcommon['continuous_primitive_block.py'];common['continuous_primitive_block.py']=(ROOT/'continuous_primitive_block_typed_dual_memory_fp64_solve_ex.py').read_bytes();sources=dict(common);sources['check_typed_dual_memory_gpu_v2.py']=(ROOT/'check_typed_dual_memory_gpu_v2.py').read_bytes()
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};common_hashes={leaf:hashes[leaf]for leaf in common}
 for raw in sources.values():ast.parse(raw)
 parent=json.loads((ROOT/'runs'/PARENT/'learn/result.json').read_text());files={}
 for leaf,relative,digest in [('ab.pt',PARENT+'/learn/'+parent['checkpoint']['file'],'912abc79925e9014146fafdec7f538282198bbddcc44034a8d7852021ebfbf85'),('CPU-proof.json',CPU+'/check/result.json','ec6ac98cbf62480645f50e44f0d4dbc7a4c7897afcbed56263292c7ce95dc970'),('CPU-reference.pt',CPU+'/check/dual-NN-B64-L16-all20-gradients.pt','285ed48bb8ecbcb829356b55ae5fd732f6a54ee4dbd3dc8675942e0cd223e188'),('old-v1-checkpoint.pt',CPU+'/check/dual-CPU-no-optimizer-checkpoint.pt','63793fe5b1d43d206f856839a96f49543a02e1752605951fb92144990c82da2e'),('v1-GPU-result.json','parameter-typed-dual-memory-gpu-execution-v1a/admit/result.json','3281571511bb71691c39911f90f5b9c339742e32111836c2da298eb04374a394')]:
  path=ROOT/'runs'/relative
  if path.is_symlink()or sha(path)!=digest:raise ValueError('exactOWNinputchanged')
  files[leaf]=dict(remote_relative=relative,sha256=digest,bytes=path.stat().st_size)
 manifest=dict(schema='typed-dual-memory-GPU-execution-v2',source_sha256=common_hashes|{'check_typed_dual_memory_gpu_v2.py':hashes['check_typed_dual_memory_gpu_v2.py']},parent_source_sha256=parent['source_sha256'],input_sha256={leaf:spec['sha256']for leaf,spec in files.items()})
 if manifest['schema']!='typed-dual-memory-GPU-execution-v2'or set(manifest['input_sha256'])!={'ab.pt','CPU-proof.json','CPU-reference.pt','old-v1-checkpoint.pt','v1-GPU-result.json'}:raise ValueError('v2driver manifest/input schema differs')
 config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=RUNNER,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest);config=json.loads(json.dumps(config,sort_keys=True));source=template().replace('CONFIG',repr(config),1);ast.parse(source);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 rp=v/'remote-launch.source';rp.write_text(source);rp.chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',formal_Primary=False,source_sha256=hashes,common_source_sha256=common_hashes,input_files=files,CPU_proof_sha256=sha(proof),runner_sha256=RUNNER,remote_launch_sha256=sha(rp),P=10351,parameter_tensors=20,carry_scalars=384,shared_identity_bytes=512,checkpoint_restore='exactOWNABthetaonly/newzeroCepisodes',data='exactactualCPU B64L16syntheticbatch inSHA-boundraw; noVALID/TEST/teacher; NEWCPUhybridprecisionreferencegeneratedbeforeGPU',optimizer_instantiated=False,optimizer_updates=0,SGD_updates=0,CPU_reference_reused=False,old_FP32_CPU_reference_supplies='samefixedsyntheticbatch+historicaldifferenceONLY',new_CPU_hybrid_NN_FB=1,new_CPU_reload_forward_only=1,targeted_new_execution_checkpoint_reload=True,training_budget_extension=0,state_update_rule_unchanged=True,execution_precision_changed=True,current_API='privateFP64normalization/Gram/RHS/ridge/solve_exFalse/mix, returnreadF32; allforwardinfoscheckedoutsidecapture',execution_policy='private-read-FP64/solve_ex-check_errors_false/all-info-checked-outside-capture/v2',theta_state_objective_tolerances_unchanged=True,forward_infos_per_NN_FB=1024,captured_forward_info_tensor_refs=16,info_logical_bytes_per_NN_FB=4096,explicit_named_FP64_temporary_logical_bytes_per_example_read=6208,NN_eager_reference_FB=1,NN_warmup_FB=3,NN_eager_timing_FB=10,solver_only_reference_FB=1,solver_only_warmup_FB=3,solver_only_timing_FB=10,capture_attempts=1,if_capture_success='onegraphreference+10timings, all20raw/CPUcomparison',if_capture_failure='saveexceptionchain+allprecapturedata, nobackgroundretry/postfailureNN; numericalGPUadmissionfalse',CPU_GPU_tolerances=dict(output=[2e-5,2e-5],gradients=[2e-5,2e-4]),fixed_numeric_failure='reportedactualoutcome, neverrelaxed; finitecases maystillcompleteindependentcapture/costdiagnostic',outside_seconds=150,science_seconds=120,address_space_GiB=512,file_limit_MiB=100,gpu_fraction=.3,existing_cuda_bootstrap_before_seccomp=True,no_new_permissions=True,no_install_or_compiler_exception=True,GPU_training_admitted=False,cost_scope='newCPUhybridFB/reload+allGPUeagerFB/warmups/timings/detachedFP64formedsolverFB/onecapture+conditionalreplays/mandatoryinfoD2Hchecks/model+grad/FP64temporaries/input/poolallocator/CPU+wall/rawIO; latenciesarecheckedexecutionincludinginfoD2HandCPUreduction; CUDAeventstreamelapsednotGPUbusytime; solverinternals/backward/workspace/drivercache/fullFLOPsunknown',full_training_FLOPs=None,provider_effective_compute=None)
 doc['unexecuted_preparation_preserved']=dict(name='parameter-typed-dual-memory-gpu-execution-v1',reason='runnerSHAtranscriptionmissingonehexcharacter; caughtlocalbeforeREADY/noSSH/noNN; samebee54driver/core')
 doc['actual_v1a_failure_preserved']=dict(result_sha256=sha(failed_path),source_core='1374fd3ca62f64faec0daf6982a632f1ff8cd447e3c11d43a39aaacf3e712c6c',capture_supported=False,CPU_GPU_fixed_tols_pass=False,cold_wall_seconds=json.loads((ROOT/'parameter-typed-dual-memory-gpu-execution-v1a-coordinator-receipt.json').read_text())['wall_seconds'],scope='actualFP32numericandcapturefailuresremainfailures; neverreclassifiedbyv2')
 doc['failed_v2_packaging_preserved']=dict(name='parameter-typed-dual-memory-gpu-execution-v2',scope='actualmanifest-v1-v2mismatchbeforeanyNN/evidenceempty, core+driverunchanged',cold_wall_seconds=json.loads((ROOT/'parameter-typed-dual-memory-gpu-execution-v2-coordinator-receipt.json').read_text())['wall_seconds'],failure_sha256=sha(ROOT/'runs/parameter-typed-dual-memory-gpu-execution-v2/admit/failure.json'))
 put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=hashes,remote_launch_sha256=sha(rp),protocol_sha256=sha(protocol)),sort_keys=True))
def execute():
 v=ROOT/'versions'/('code-'+NAME);config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=template().replace('CONFIG',repr(config),1):raise ValueError('frozencoordinatordiffers')
 for leaf,h in config['source_sha256'].items():
  if sha(v/leaf)!=h:raise ValueError('frozensourcediffers')
  ast.parse((v/leaf).read_bytes())
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserveactual/noautomaticretry')
 tick=time.monotonic()
 try:
  x=subprocess.run(transport.SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspectboundedstate/no retry')
 value['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,value);print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=parser.parse_args()
 if a.prepare:prepare()
 else:execute()
