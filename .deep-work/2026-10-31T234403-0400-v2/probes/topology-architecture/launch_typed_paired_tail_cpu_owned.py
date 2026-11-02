"""Freeze/ROOT-execute one OWN typed-stage2 resume admission, zero SGD."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-paired-tail-resume-cpu-v1'
PARENT='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2'
PAIR_RUN='parameter-paired-train-indices-n8-stage3-seed0-v1'
COMMON=('continuous_primitive_block.py','owned_fcba_core.py','typed_memory_actions_masked_graph_fb.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
def template():return transport.TEMPLATE.replace('check_value_bridge_cpu.py','check_typed_paired_tail_resume_cpu.py').replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','CPU_TYPED_PAIRED_TAIL_RESUME_PASS').replace(".get('P')",".get('P16')")
def prepare():
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists():raise FileExistsError('newnamespace only')
 parent_version=ROOT/'versions'/('code-'+PARENT);parent_config=json.loads((parent_version/'launch-config.json').read_text());parent_result=json.loads((ROOT/'runs'/PARENT/'learn/result.json').read_text());cp=parent_result['reports'][2]['checkpoint'];cp_path=ROOT/'runs'/PARENT/'learn'/cp['file']
 if sha(cp_path)!=cp['sha256']or cp_path.stat().st_size!=cp['bytes']or cp['sha256']!='c5a3cbe14ebc1e592a0f0e462f6bd07062870c8551f345a2b0c64bf92a3a908e':raise ValueError('exact OWN typedstage2')
 pair_folder=ROOT/'runs'/PAIR_RUN/'generate';pair_path=pair_folder/'paired-stage3-indices.npz';pair_result_path=pair_folder/'result.json';pair_result=json.loads(pair_result_path.read_text())
 if sha(pair_path)!=pair_result['schedule']['sha256']or pair_result['status']!='CPU_PAIRED_TRAIN_INDICES_GENERATED_PASS'or not pair_result['same_pool_row_exposures_exact']or pair_result['target_array_reads']!=0:raise ValueError('actual publicTRAIN pair/control generator required')
 common={leaf:(parent_version/leaf).read_bytes()for leaf in COMMON};common_hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in common.items()}
 if any(parent_result['source_sha256'][leaf]!=digest for leaf,digest in common_hashes.items()):raise ValueError('actualparent/common source differs')
 sources=dict(common);sources['check_typed_paired_tail_resume_cpu.py']=(ROOT/'check_typed_paired_tail_resume_cpu.py').read_bytes()
 for raw in sources.values():ast.parse(raw)
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};files={};data={}
 for s in range(4):
  spec=dict(parent_config['manifest']['files'][f'{s}:train']);source=ROOT/'runs'/spec['remote_relative']
  if sha(source)!=spec['sha256']:raise ValueError('original TRAIN changed')
  spec['bytes']=source.stat().st_size;data[f'{s}:train']=spec;files[spec['leaf']]=dict(remote_relative=spec['remote_relative'],sha256=spec['sha256'],bytes=spec['bytes'])
 resume=dict(leaf='resume.pt',sha256=cp['sha256'],bytes=cp['bytes'],parent_run=PARENT,source_sha256=parent_result['source_sha256'])
 pairs=dict(leaf='paired-stage3-indices.npz',sha256=sha(pair_path),bytes=pair_path.stat().st_size,result_leaf='pair-generation-result.json',result_sha256=sha(pair_result_path),result_bytes=pair_result_path.stat().st_size,parent_run=PAIR_RUN)
 files['resume.pt']=dict(remote_relative=str(cp_path.relative_to(ROOT/'runs')),sha256=cp['sha256'],bytes=cp['bytes']);files[pairs['leaf']]=dict(remote_relative=str(pair_path.relative_to(ROOT/'runs')),sha256=pairs['sha256'],bytes=pairs['bytes']);files[pairs['result_leaf']]=dict(remote_relative=str(pair_result_path.relative_to(ROOT/'runs')),sha256=pairs['result_sha256'],bytes=pairs['result_bytes'])
 manifest=dict(schema='typed-paired-tail-resume-CPU-v1',source_sha256=common_hashes,resume=resume,pairs=pairs,files=data)
 c=dict(name=NAME,remote=transport.REMOTE,runner_SHA=transport.CPU,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},input_files=files,manifest=manifest);c=json.loads(json.dumps(c,sort_keys=True));remote=template().replace('CONFIG',repr(c),1);ast.parse(remote);v.mkdir()
 for leaf,raw in sources.items():(v/leaf).write_bytes(raw);(v/leaf).chmod(0o444)
 (v/'remote-launch.source').write_text(remote);(v/'remote-launch.source').chmod(0o444);put(v/'launch-config.json',c);(v/'launch-config.json').chmod(0o444)
 doc=dict(name=NAME,status='FROZEN_UNEXECUTED',before_execution=True,source_sha256=hashes,common_source_sha256=common_hashes,resume=resume,pairs=pairs,input_files=files,expectedP15=10975,expectedP16=10991,parameter_tensors=20,SGD_updates=0,research_updates=0,optimizer_steps=0,CPU_scope='restore actualOWNstage2NN/Adam1800/CPU+NumPy+Python RNG/Replay; retainCUDA RNGpayloadbytes only notCUDAexecution;originalonegrowth;2samplerfirstTRAINbatches/all20gather-vsmasked/stablegrad; realroundtrip; CPUnewrownotGPUidentityproof',pair_scope='sameactual immutable NPZ+global currentIDmultiplicities;paired24x2 vs same_pool_shuffle, untouched originalReplaydraws; noformationlabels/newtasklabels/teacher',no_quality_or_gradient_norm_gate=True,CPU_checkpoint_not_GPU_parent=True,permissions='unchangedCPUrunner c9/150s/AS64GiB/file100MiB; ROOT-onlyexecute',outside_seconds=150,AS_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,remote_launch_sha256=sha(v/'remote-launch.source'),costs='allNNrestore/Adam/RNG/Replay/source+TRAIN+schedule/CPUF-B/serialization/RSS paid;0SGD;existing1800notrerun;pairformationactualcostseparate/shared6tails')
 put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=hashes,protocol_sha256=sha(protocol),remote_launch_sha256=sha(v/'remote-launch.source'))))
def execute():
 v=ROOT/'versions'/('code-'+NAME);c=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());remote=(v/'remote-launch.source').read_text()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or remote!=template().replace('CONFIG',repr(c),1):raise ValueError('frozen coordinator differs')
 for leaf,h in c['source_sha256'].items():
  if sha(v/leaf)!=h:raise ValueError('frozen source differs')
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actualattempt; noautomaticretry')
 started=time.monotonic()
 try:
  x=subprocess.run(transport.SSH,input=remote,text=True,capture_output=True,timeout=210);d=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
 except subprocess.TimeoutExpired as e:d=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-started,error='SSHwaittimeout; inspect boundedrunner/noautomaticretry')
 d['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,d);print(json.dumps(dict(name=NAME,returncode=d['returncode'],wall_seconds=d['wall_seconds'])));raise SystemExit(d['returncode']if d['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 else:execute()
