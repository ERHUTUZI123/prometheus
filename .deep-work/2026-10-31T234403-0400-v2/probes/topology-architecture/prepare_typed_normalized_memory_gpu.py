"""Stdlib freeze only: three fixed A/B fresh arms after actual CPU admission.

No SSH/model/training/compile execution in this preparer. ROOT executes the
resulting frozen launchers separately and serially. Historical none is reused.
"""
import argparse,ast,base64,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
CPU_NAME='parameter-typed-normalized-memory-cpu-v3'
POLICY='typed-memory-actions/input-address+per-cell-coverage/raw-value/group8/v1'
RUNNER='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
REQUIRED=['all20_gradients_checked','all_carry_fields_masked','exact_historical_common_initialization_and_RNG','strict_mode_flags_and_checkpoint_rejection_checked','CPU_only_not_CUDA_result']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=('A','B','AB'),required=True);a=parser.parse_args();flags=dict(input_only_address=a.arm in('A','AB'),coverage_normalization=a.arm in('B','AB'))
 proof_path=P/'runs'/CPU_NAME/'check/result.json';proof=json.loads(proof_path.read_text())
 if proof.get('status')!='CPU_TYPED_NORMALIZED_MEMORY_GRAPH_ADMISSION_PASS'or proof.get('SGD_updates')!=0 or any(proof.get(k)is not True for k in REQUIRED):raise ValueError('actual CPU result required, no inferred admission')
 cpuconfig=json.loads((P/'versions'/('code-'+CPU_NAME)/'launch-config.json').read_text())
 if proof['source_sha256']!=cpuconfig['manifest']['source_sha256']:raise ValueError('actual CPU/frozen source differs')
 old=P/'versions/code-parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2';oldconfig=json.loads((old/'launch-config.json').read_text())
 sources={leaf:(old/leaf).read_bytes()for leaf in('owned_fcba_core.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')}
 sources.update({'continuous_primitive_block.py':(P/'continuous_primitive_block_typed_normalized_memory.py').read_bytes(),'typed_normalized_memory_masked_graph_fb.py':(P/'typed_normalized_memory_masked_graph_fb.py').read_bytes(),'typed_normalized_memory_graph_engine.py':(P/'typed_normalized_memory_graph_engine.py').read_bytes(),'train_typed_normalized_memory_fourstage_graph.py':(P/'train_typed_normalized_memory_fourstage_graph.py').read_bytes()})
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()};common={k:v for k,v in hashes.items()if k not in('typed_normalized_memory_graph_engine.py','train_typed_normalized_memory_fourstage_graph.py')}
 for leaf,digest in common.items():
  if proof['source_sha256'].get(leaf)!=digest:raise ValueError('actualCPU/GPU common source differs: '+leaf)
 for raw in sources.values():ast.parse(raw)
 files=oldconfig['manifest']['files']
 for spec in files.values():
  path=P/'runs'/spec['remote_relative']
  if path.is_symlink()or sha(path)!=spec['sha256']:raise ValueError('fixed historical data differs')
 name='parameter-typed-normalized-memory-'+a.arm.lower()+'-n8-seed0-fourstage-graph-v1';version=P/'versions'/('code-'+name);protocol=P/(name+'-protocol.json');launcher=P/('launch_typed_normalized_memory_'+a.arm.lower()+'_owned.py')
 if any(x.exists()or x.is_symlink()for x in(version,protocol,launcher)):raise FileExistsError('new fixed namespace only, preserve all attempts')
 manifest=dict(schema='typed-normalized-memory-fourstage-inputs-v1',architecture_policy=POLICY,arm=a.arm,architecture_flags=flags,files=files,source_sha256=hashes)
 config=dict(name=name,arm=a.arm,architecture_flags=flags,expected_P16=10351 if flags['input_only_address']else 10991,runner_sha256=RUNNER,CPU_gate_run=CPU_NAME,CPU_proof_sha256=sha(proof_path),CPU_required_true_fields=REQUIRED,CPU_mode_P={'A':[10335,10351],'B':[10975,10991],'AB':[10335,10351]},architecture_policy=POLICY,source_sha256=hashes,common_source_sha256=common,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()},manifest=manifest)
 config=json.loads(json.dumps(config,sort_keys=True));remote=(P/'typed-normalized-memory-gpu-remote-template.source').read_text().replace('CONFIG',repr(config),1);ast.parse(remote)
 version.mkdir()
 for leaf,raw in sources.items():
  with(version/leaf).open('xb')as f:f.write(raw)
  (version/leaf).chmod(0o444)
 put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444);rp=version/'remote-launch.source';rp.write_text(remote);rp.chmod(0o444)
 doc=dict(name=name,status='FROZEN_UNEXECUTED',formal_Primary=False,arm=a.arm,architecture_flags=flags,architecture_policy=POLICY,source_sha256=hashes,common_source_sha256=common,CPU_gate_run=CPU_NAME,CPU_proof_sha256=sha(proof_path),CPU_checker_sha256=cpuconfig['source_sha256']['check_typed_normalized_memory_cpu.py'],runner_sha256=RUNNER,remote_launch_sha256=sha(rp),expected_P15=config['expected_P16']-16,expected_P16=config['expected_P16'],live_parameter_tensors=20,carry_scalars=328 if flags['coverage_normalization']else 320,coverage_epsilon=1e-4,NN_updates=2400,stages=[0,1,2,3],steps_per_stage=600,batch=64,replay_capacity=128,replay_rows=16,lr=.003,clip=1.,seed=0,fresh_NN=True,parent_NN_loaded=False,historical_updates_rerun=False,old_none_control='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2',prospective_family='three arms A/B/AB only, old none retained; no extra seeds/ranks/configs, no score gate',data='exact original TRAIN/VALID sampler9027 and Replay128; no paired-tail schedule; TEST only after all2400 frozen',source_rule='Arawlearnedx16 proposal only, fullcontextalphaA; BpercellV/clamp_min(z,1e-4), z updated withsameSTOREweight; OLDh Vwrite/CALL/G8/oldlquery unchanged',initialization='fulloriginaltypedfactory/RNG thenA exactproposalcolumns48:64 copy, no additionaldraws; Bstateonly',captures=2,buckets=[16,48],all20_current_NN_checks='fixed firstTRAIN/currentNN at admission andeach100, same predeclared tolerances; all NN/gradstorage stable',outside_seconds=900,science_seconds=850,gpu_fraction=.3,address_space_GiB=512,file_limit_MiB=100,no_new_permissions=True,no_compiler_exception=True,no_accuracy_or_speed_gate=True,cost_scope='allfresh2400,0SGDsetup/warmups/2captures/all20diagnostics/eagerclipAdam/source/data/coverage32B+divisionsclamps/NNAdamReplay/cache/checkpoints/safety/reload/maintenance/eval/inference/fullCPUwall; fullFLOPs/provider-effective unknown',limitations='normalized weakwrites maybroadcastsamevalue insofttails; no guaranteed keyseparation/closure/retention; epsilonboundedgain and noisy/stalecontent amplification; both factors tested independently',full_training_FLOPs=None,provider_effective_compute=None)
 doc['shared_CPU_admission_attempts']=[dict(name=n,receipt_sha256=sha(P/(n+'-coordinator-receipt.json')),local_cold_wall_seconds=json.loads((P/(n+'-coordinator-receipt.json')).read_text())['wall_seconds'])for n in('parameter-typed-normalized-memory-cpu-v2',CPU_NAME)]
 doc['shared_CPU_cost_scope']='count both actual CPU attempts once for the three-arm family; v1 import-race freeze unexecuted, v2 checker-scopefailure0researchSGD, v3 syntheticadmission; never multiply shared admissionfees bythree'
 put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555)
 tree=ast.parse((P/'launch_typed_memory_actions_graph_owned_v2.py').read_text());updates={'NAME':name,'SOURCE_SHA256':hashes,'PROTOCOL_SHA256':sha(protocol),'REMOTE_SHA256':sha(rp)}
 tree.body[0].value=ast.Constant('ROOT-only one fixed normalized-memory fresh A/B arm; no automatic retry or parallel NN.')
 for n in tree.body:
  if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in updates:n.value=ast.parse(repr(updates[n.targets[0].id]),mode='eval').body
 code=ast.unparse(ast.fix_missing_locations(tree))+'\n';ast.parse(code);launcher.write_text(code)
 print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=name,launcher=str(launcher),launcher_sha256=sha(launcher),protocol_sha256=sha(protocol),remote_launch_sha256=sha(rp),source_sha256=hashes),sort_keys=True))
if __name__=='__main__':main()
