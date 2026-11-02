"""PREP/ROOT-execute exactly one actualGraph baseline and one bridge closure.

No model import/deserialization/execution during preparation. Final binding
requires the ACTUAL locally synced completed NN bytes/source/result, never an
invented future checkpoint hash. No old completed closure is rerun.
"""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import extended_closure_launch as old
ROOT=Path(__file__).resolve().parent
DATA_SHA='a72d1a6a335f2c42edde553c92457996b723784a37cdd84e61b6a39f3fd54009'
METADATA_SHA='9225cbe615986e50982b9f33d97d9a8c401e0695d818535d172dcd9c0252cb3f'
PARENT_STAGE2='9bb47e660e4ea67d0bf1a06e93e418f0c3fa50760fa76ff067d3a0c954a6c40f'
REGISTRY={
 'baseline':dict(parent='parameter-factorized-graph-stage3-anchor0-n8-seed0-600-v1',name='parameter-extended-live-closure-factorized_graph-v1',P=13397,status='ACTUAL_FACTORIZED_GRAPH_STAGE3_600_COMPLETE',schema='factorized-graph-stage3-training-v1',block_header=dict(version='continuous-primitive-block-control-value-v1',organization_policy='learned-control-history/value-identity-write/group8-carry-v1'),core_SHA='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158',checkpoint_SHA='7d8a75f960bd55fedbe4afa68eccf59b18d05de759b2eb1e0724818c3acedd5d'),
 'bridge':dict(parent='parameter-value-bridge-graph-n8-seed0-stage3-v3',name='parameter-extended-live-closure-value_bridge_graph-v1',P=13398,status='ACTUAL_SOFT_VALUE_BRIDGE_GRAPH_STAGE3_600_COMPLETE',schema='soft-value-bridge-graph-stage3-training-v1',block_header=dict(version='continuous-primitive-block-soft-value-bridge-v1',organization_policy='learned-control-history/value-identity-write/group8-carry+soft-value-bridge/v1'),core_SHA='f1ccdc83fbae14754e4271da9cdfb949f4f656c9704e35df283f899806c95da3',checkpoint_SHA=None)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def facts(key):
 spec=REGISTRY[key];parent=spec['parent'];folder=ROOT/'runs'/parent/'learn';rf=folder/'result.json'
 if not rf.is_file():raise ValueError('actual completed parent not locally synced; no future SHA binding: '+parent)
 d=json.loads(rf.read_text())
 if d.get('status')!=spec['status']or d.get('formal_Primary')is not False or d['resources']['parameters']!=spec['P']or d['actual_new_updates_this_process']!=600 or d['historical_updates_shared']!=1800 or d['parent_stage2_checkpoint_sha256']!=PARENT_STAGE2 or d.get('NN_unchanged_by_evaluation')is not True:raise ValueError('exact actual completed retained lineage required')
 cp=d['checkpoint'];path=folder/cp['file'];receipt=old.checked(path,cp['sha256'])
 if receipt['bytes']!=cp['bytes']or(spec['checkpoint_SHA']is not None and receipt['sha256']!=spec['checkpoint_SHA']):raise ValueError('actual checkpoint bytes differ')
 if old.checkpoint_header(path)!=dict(schema=spec['schema'],stage=3,step=600,stage_complete=True):raise ValueError('literal completedNN header differs')
 sources=d['source_sha256'];v=ROOT/'versions'/('code-'+parent);protocol=json.loads((ROOT/(parent+'-protocol.json')).read_text())
 if protocol['source_sha256']!=sources or sources['continuous_primitive_block.py']!=spec['core_SHA']:raise ValueError('frozen/actual source maps differ')
 original={}
 for leaf,digest in sources.items():old.checked(v/leaf,digest);original[leaf]=(v/leaf).read_bytes();ast.parse(original[leaf])
 records=folder/'records.jsonl';rows=[json.loads(x)for x in records.read_text().splitlines()];updates=[x for x in rows if x['kind']=='actual_update']
 if [x['step']for x in updates]!=list(range(1,601))or sum(x['training_wall_seconds']for x in updates)!=d['costs']['training_wall_seconds']:raise ValueError('actual600 paid stream differs')
 phases=[dict(stage=r['stage'],costs=r['costs'],checkpoint=r['checkpoint'])for r in d['reports']]
 if [x['stage']for x in phases]!=list(range(4))or any(x['costs']['SGD_updates']!=600 for x in phases):raise ValueError('actual fourphase lineage required')
 physical=lambda c:c['padded_forward_MAC_proxy']if'padded_forward_MAC_proxy'in c else c['forward_MAC_proxy']
 active=lambda c:c['original_active_forward_MAC_proxy']if'original_active_forward_MAC_proxy'in c else c['forward_MAC_proxy']
 lineage=dict(phases=phases,retained_lineage_SGD_updates=2400,retained_lineage_training_wall_seconds=sum(x['costs']['training_wall_seconds']for x in phases),retained_lineage_training_forward_MAC_proxy=sum(physical(x['costs'])for x in phases),retained_lineage_original_active_forward_MAC_proxy=sum(active(x['costs'])for x in phases),current_parent_result_sha256=sha(rf),current_parent_records_sha256=sha(records),other_paid_costs=dict(setup=d['setup'],restore_and_growth=d['restore_and_growth'],reload=d['reload'],maintenance=d['maintenance'],inference_total_seconds=d['inference_total_seconds'],driver_wall_seconds=d['driver_wall_seconds'],driver_CPU_seconds=d['driver_CPU_seconds'],premaintenance_safety_checkpoint=d['premaintenance_safety_checkpoint'],scope='Graphsetup/capture/padding/diagnostics/eagerAdam/eval/checkpoint/restore costs paid; nested timers not additive; other pastcampaign failures/profile/operand arms separately retained'),exact_full_training_FLOPs=None,scope='actual shared early1800 paidonce acrossNNs; distinct Graphbaseline600 and bridge600 independently paid, not independentseeds or pooledperformance')
 return spec,d,original,path,receipt,lineage

def prepare(key):
 # Establish actual NN/source before reading any closure metadata.
 spec,d,sources,checkpoint,cp,lineage=facts(key);name=spec['name'];v=ROOT/'versions'/('code-'+name)
 for path in(v,ROOT/(name+'-protocol.json'),ROOT/(name+'-coordinator-receipt.json'),ROOT/'runs'/name):
  if path.exists():raise FileExistsError('newclosure namespace only; preserveattempt: '+str(path))
 folder,data_files,paid=old.generated_data_facts()
 if data_files['closure.npz']['sha256']!=DATA_SHA or data_files['metadata.json']['sha256']!=METADATA_SHA:raise ValueError('same actual3584closure data required')
 sources=dict(sources);sources['evaluate_extended_closure.py']=(ROOT/'evaluate_graph_value_closure.py').read_bytes();hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 for raw in sources.values():ast.parse(raw)
 manifest=dict(schema='extended-live-closure-evaluator-inputs-v1',model='factorized_graph'if key=='baseline'else'value_bridge_graph',family='block',P=spec['P'],checkpoint_sha256=cp['sha256'],checkpoint_schema=spec['schema'],block_checkpoint_header=spec['block_header'],source_sha256=hashes,parent_source_sha256=d['source_sha256'],data_sha256=DATA_SHA,metadata_sha256=METADATA_SHA,profile_cache_trained=False,parent_resources=d['resources'],training_lineage=lineage,generation_paid_lineage=paid)
 raw=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode();inputs={leaf:dict(receipt,remote_relative=str((folder/leaf).relative_to(ROOT/'runs')))for leaf,receipt in data_files.items()};inputs['resume.pt']=dict(cp,remote_relative=str(checkpoint.relative_to(ROOT/'runs')))
 protocol=dict(name=name,before_execution=True,status='FROZEN_UNEXECUTED',formal_Primary=False,scope='fixedactualNN seven-suffix developmentclosure; noSGD/search/tuning/newdata or priorclosure rerun',parent=spec['parent'],source_sha256=hashes,parent_source_sha256=d['source_sha256'],P=spec['P'],checkpoint=cp,checkpoint_schema=spec['schema'],block_checkpoint_header=spec['block_header'],same_data_sha256=DATA_SHA,metadata_sha256=METADATA_SHA,rows=3584,base_cases=512,suffix_count=7,maximum_length=50,metric_math='unchanged originalsubset_metrics; MIX/READ separately, full/within48/above48, joint/consistency/ownbaseconditionaldenominators andrawperexampleoutputs',carry_export='sameoneforward forward_value returns alreadycomputedh/controller/keys/values, rawfinalcarry+logicalpayload+CPUownposteriorR(h)geometry; no terminaly→fullcarry identification',value_bridge_alpha=d.get('value_bridge_alpha'),value_bridge_strength=d.get('value_bridge_strength'),modes=['soft'],SGD_updates=0,training_lineage=lineage,generation_paid_lineage=paid,costs='forwardwall/CUDAstream/H2D+rawcarryD2H/CPUgeometry/wholewall/RSS/allocator/input/raw/NN/checkpoint/Adam/replay storage allpaid; no fullFLOPs/provider-effective claims; sharedgeneration/earlytraining paidonce',outside_seconds=900,inside_seconds=800,gpu_fraction=.3,address_space_GiB=512,runner_sha256=old.GPU,no_new_permissions=True)
 config=dict(name=name,remote=old.REMOTE,CPU=old.CPU,GPU=old.GPU,mode='eval',entry='evaluate_extended_closure.py',result_subdir='evaluate',parent=spec['parent'],parent_source_sha256=d['source_sha256'],inputs=inputs,source_sha256=hashes,sources={leaf:base64.b64encode(value).decode()for leaf,value in sources.items()},manifest_bytes=base64.b64encode(raw).decode(),manifest_sha256=hashlib.sha256(raw).hexdigest())
 config=json.loads(json.dumps(config,sort_keys=True));remote=old.TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(remote);v.mkdir()
 for leaf,value in sources.items():
  with(v/leaf).open('xb')as f:f.write(value)
  (v/leaf).chmod(0o444)
 with(v/'input-manifest.json').open('xb')as f:f.write(raw)
 (v/'input-manifest.json').chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 (v/'remote-launch.source').write_text(remote);(v/'remote-launch.source').chmod(0o444);protocol['remote_launch_sha256']=sha(v/'remote-launch.source');put(ROOT/(name+'-protocol.json'),protocol);(ROOT/(name+'-protocol.json')).chmod(0o444);v.chmod(0o555)
 print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=name,parent_checkpoint_sha256=cp['sha256'],remote_launch_sha256=protocol['remote_launch_sha256'],protocol_sha256=sha(ROOT/(name+'-protocol.json')),evaluator_sha256=hashes['evaluate_extended_closure.py'])))

def execute(key):
 name=REGISTRY[key]['name'];v=ROOT/'versions'/('code-'+name);config=json.loads((v/'launch-config.json').read_text());source=(v/'remote-launch.source').read_text();protocol=json.loads((ROOT/(name+'-protocol.json')).read_text())
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=old.TEMPLATE.replace('CONFIG',repr(config),1):raise ValueError('frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():old.checked(v/leaf,digest)
 for spec in config['inputs'].values():
  actual=old.checked(ROOT/'runs'/spec['remote_relative'],spec['sha256'])
  if actual['bytes']!=spec['bytes']:raise ValueError('frozeninput bytecount differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserveactualattempt; no retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(old.SSH,input=source,text=True,capture_output=True,timeout=980);value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspectboundedrun, no retry')
 value['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,value);print(json.dumps(dict(name=name,returncode=value['returncode'],wall_seconds=value['wall_seconds'])));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=REGISTRY);g.add_argument('--execute',choices=REGISTRY);a=parser.parse_args()
 if a.prepare:prepare(a.prepare)
 else:execute(a.execute)
