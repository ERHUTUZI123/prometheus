"""Freeze or ROOT-execute one bounded dual-memory GPU execution diagnostic."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-dual-memory-gpu-execution-v1a'
RUNNER='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
PARENT='parameter-typed-normalized-memory-ab-n8-seed0-fourstage-graph-v1'
CPU='parameter-typed-dual-memory-cpu-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def template():
 s=transport.TEMPLATE.replace('remote_runner_cpu_v2.py','remote_runner_gpu_authorized.py').replace('check_value_bridge_cpu.py','check_typed_dual_memory_gpu.py').replace("out/'check'","out/'admit'").replace("out/'check/result.json'","out/'admit/result.json'").replace("'--address-space-gib','64'","'--address-space-gib','512'").replace("'--cpu-only'","'--gpu-fraction','.3','--cuda-init-before-seccomp'").replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','GPU_DUAL_MEMORY_EXECUTION_DIAGNOSTIC_COMPLETE')
 s=s.replace("actual_P=(result or{}).get('P'),research_updates=0)","actual_P=(result or{}).get('P'),research_updates=0,capture_supported=(result or{}).get('capture',{}).get('supported'),numerical_GPU_admission=(result or{}).get('numerical_GPU_admission'),GPU_training_admitted=False)")
 return s
def prepare():
 if sha(ROOT/'remote_runner_gpu_authorized.py')!=RUNNER:raise ValueError('verifiedGPUrunnerhashdiffers')
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespaceonly/preserveallattempts')
 cpudir=ROOT/'versions'/('code-'+CPU);cpuconfig=json.loads((cpudir/'launch-config.json').read_text());proof=ROOT/'runs'/CPU/'check/result.json';CPUresult=json.loads(proof.read_text())
 if sha(proof)!='ec6ac98cbf62480645f50e44f0d4dbc7a4c7897afcbed56263292c7ce95dc970'or CPUresult['status']!='CPU_TYPED_DUAL_MEMORY_ADMISSION_PASS':raise ValueError('fixedactualCPUadmissionrequired')
 common={leaf:(cpudir/leaf).read_bytes()for leaf in('continuous_primitive_block.py','owned_ab_core.py','owned_fcba_core.py','typed_dual_memory_masked_fb.py')};sources=dict(common);sources['check_typed_dual_memory_gpu.py']=(ROOT/'check_typed_dual_memory_gpu.py').read_bytes()
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};common_hashes={leaf:hashes[leaf]for leaf in common}
 if common_hashes!=CPUresult['source_sha256']:raise ValueError('UNCHANGEDCPU/GPUcore/helper required')
 for raw in sources.values():ast.parse(raw)
 parent=json.loads((ROOT/'runs'/PARENT/'learn/result.json').read_text());files={}
 for leaf,relative,digest in [('ab.pt',PARENT+'/learn/'+parent['checkpoint']['file'],'912abc79925e9014146fafdec7f538282198bbddcc44034a8d7852021ebfbf85'),('CPU-proof.json',CPU+'/check/result.json','ec6ac98cbf62480645f50e44f0d4dbc7a4c7897afcbed56263292c7ce95dc970'),('CPU-reference.pt',CPU+'/check/dual-NN-B64-L16-all20-gradients.pt','285ed48bb8ecbcb829356b55ae5fd732f6a54ee4dbd3dc8675942e0cd223e188')]:
  path=ROOT/'runs'/relative
  if path.is_symlink()or sha(path)!=digest:raise ValueError('exactOWNinputchanged')
  files[leaf]=dict(remote_relative=relative,sha256=digest,bytes=path.stat().st_size)
 manifest=dict(schema='typed-dual-memory-GPU-execution-v1',source_sha256=common_hashes|{'check_typed_dual_memory_gpu.py':hashes['check_typed_dual_memory_gpu.py']},parent_source_sha256=parent['source_sha256'],input_sha256={leaf:spec['sha256']for leaf,spec in files.items()})
 config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=RUNNER,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest);config=json.loads(json.dumps(config,sort_keys=True));source=template().replace('CONFIG',repr(config),1);ast.parse(source);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 rp=v/'remote-launch.source';rp.write_text(source);rp.chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',formal_Primary=False,source_sha256=hashes,common_source_sha256=common_hashes,input_files=files,CPU_proof_sha256=sha(proof),runner_sha256=RUNNER,remote_launch_sha256=sha(rp),P=10351,parameter_tensors=20,carry_scalars=384,shared_identity_bytes=256,checkpoint_restore='exactOWNABthetaonly/newzeroCepisodes',data='exactactualCPU B64L16syntheticbatch inSHA-boundraw; noVALID/TEST/teacher',optimizer_instantiated=False,optimizer_updates=0,SGD_updates=0,CPU_reference_reused=True,training_budget_extension=0,core_rule_unchanged=True,current_API='torch.linalg.solve only, no solve_ex/kernel/compilerrepair beforeactualfailure',NN_eager_reference_FB=1,NN_warmup_FB=3,NN_eager_timing_FB=10,solver_only_reference_FB=1,solver_only_warmup_FB=3,solver_only_timing_FB=10,capture_attempts=1,if_capture_success='onegraphreference+10timings, all20raw/CPUcomparison',if_capture_failure='saveexceptionchain+allprecapturedata, nobackgroundretry/postfailureNN; numericalGPUadmissionfalse',CPU_GPU_tolerances=dict(output=[2e-5,2e-5],gradients=[2e-5,2e-4]),fixed_numeric_failure='reportedactualoutcome, neverrelaxed; finitecases maystillcompleteindependentcapture/costdiagnostic',outside_seconds=150,science_seconds=120,address_space_GiB=512,file_limit_MiB=100,gpu_fraction=.3,existing_cuda_bootstrap_before_seccomp=True,no_new_permissions=True,no_install_or_compiler_exception=True,GPU_training_admitted=False,cost_scope='allGPUeagerFB/warmups/timings/detachedformedsolverFB/onecapture+conditionalreplays/model+grad/input/poolallocator/CPU+wall/rawIO; CUDAeventstreamelapsednotGPUbusytime; solverinternals/backward/workspace/drivercache/fullFLOPsunknown',full_training_FLOPs=None,provider_effective_compute=None)
 doc['unexecuted_preparation_preserved']=dict(name='parameter-typed-dual-memory-gpu-execution-v1',reason='runnerSHAtranscriptionmissingonehexcharacter; caughtlocalbeforeREADY/noSSH/noNN; samebee54driver/core')
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
