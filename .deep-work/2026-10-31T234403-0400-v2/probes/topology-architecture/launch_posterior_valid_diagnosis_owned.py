"""PREP/ROOT-run privileged CPUtruth then frozen4NN VALID diagnosis only."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport
ROOT=Path(__file__).resolve().parent
NAMES=dict(truth='parameter-posterior-value-valid-prefix-truth-v1',audit='parameter-posterior-value-four-nn-valid-diagnosis-v1')
VALID_REL='parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz'
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
ORACLE_SHA='8943cd6dd4d638224dcc7be62df82e505b871b32bbcf1be5de3be3afadb92726'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d):
 with p.open('x')as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
def truth_template():
 return transport.TEMPLATE.replace('check_value_bridge_cpu.py','build_posterior_valid_prefix_truth.py').replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS','PRIVILEGED_VALID_PREFIX_TRUTH_COMPLETE').replace("out/'check'","out/'truth'").replace("out/'check/result.json'","out/'truth/result.json'")

def freeze(mode,sources,config,doc,template):
 name=NAMES[mode];v=ROOT/'versions'/('code-'+name);protocol=ROOT/(name+'-protocol.json')
 if any(f.exists()for f in(v,protocol,ROOT/(name+'-coordinator-receipt.json'))):raise FileExistsError('new namespace only; preserveattempts')
 hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 for raw in sources.values():ast.parse(raw)
 config.update(name=name,source_sha256=hashes,files={leaf:base64.b64encode(raw).decode()for leaf,raw in sources.items()});config=json.loads(json.dumps(config,sort_keys=True));source=template.replace('CONFIG',repr(config),1);ast.parse(source);v.mkdir()
 for leaf,raw in sources.items():
  with(v/leaf).open('xb')as f:f.write(raw)
  (v/leaf).chmod(0o444)
 (v/'remote-launch.source').write_text(source);(v/'remote-launch.source').chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 doc.update(name=name,mode=mode,status='FROZEN_UNEXECUTED',before_execution=True,formal_Primary=False,source_sha256=hashes,remote_launch_sha256=sha(v/'remote-launch.source'));put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555)
 print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=name,source_sha256=hashes,remote_launch_sha256=doc['remote_launch_sha256'],protocol_sha256=sha(protocol))))

def prepare_truth():
 oracle=ROOT/'memory_live_stage_data.py';valid=ROOT/'runs'/VALID_REL
 if sha(oracle)!=ORACLE_SHA or sha(valid)!=VALID_SHA:raise ValueError('exact originalVALID/oracle sources required')
 sources={'official_memory_oracle.py':oracle.read_bytes(),'build_posterior_valid_prefix_truth.py':(ROOT/'build_posterior_valid_prefix_truth.py').read_bytes()};hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 config=dict(remote=transport.REMOTE,runner_SHA=transport.CPU,input_files={'valid.npz':dict(remote_relative=VALID_REL,sha256=VALID_SHA,bytes=valid.stat().st_size)},manifest=dict(schema='privileged-VALID-prefix-truth-inputs-v1',source_sha256=hashes))
 doc=dict(scope='PRIVILEGEDofflineaudittruth ONLY, neverNN/training/routing input',VALID_sha256=VALID_SHA,official_oracle_sha256=ORACLE_SHA,rows=512,events=['write_first','write_second','after_read','before_mix_after_key_other','at_mix'],rule='instrument alloriginal6statevariables; crosscheckEVERYprefixq againstexactofficialterminal; originalterminaltargets512 checked',NN_queries=0,SGD_updates=0,new_learner_labels=0,outside_seconds=150,AS_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,permissions='unchangedofflineCPUrunner; noNNimports or model execution',costs='allprivilegedoraclecalls/CPU/wall/RSS/truthstorage andsource/data bytes counted; no freeaudit')
 freeze('truth',sources,config,doc,truth_template())

AUDIT_TEMPLATE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG
r=pathlib.Path(c['remote'])
name=c['name']
code=r/('code-'+name)
out=r/'outputs'/name
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
status=out.parent/(name+'.runner-status.json')
receipt=out.parent/(name+'.coordinator-exit.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())if p.is_file()else None
if any(p.exists()or p.is_symlink()for p in(code,out,inputs,status,receipt)):raise FileExistsError('newnamespace only; preserveattempt')
runner=r/'remote_runner_gpu_authorized.py'
if sha(runner)!=c['runner_SHA']:raise ValueError('unchangedGPUrunner differs')
for spec in c['manifest']['models']:
 for leaf,digest in spec['parent_source_sha256'].items():
  if sha(r/('code-'+spec['parent'])/leaf)!=digest:raise ValueError('actualNNparent source changed')
for leaf,spec in c['input_files'].items():
 source=r/'outputs'/spec['remote_relative']
 if source.is_symlink()or not source.is_file()or source.stat().st_size!=spec['bytes']or sha(source)!=spec['sha256']or spec['bytes']>100*1024*1024:raise ValueError('exactOWNinput mismatch: '+leaf)
code.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded,validate=True)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport differs')
 with(code/leaf).open('xb')as f:f.write(raw)
 (code/leaf).chmod(0o444)
code.chmod(0o555);inputs.mkdir()
for leaf,spec in c['input_files'].items():shutil.copyfile(r/'outputs'/spec['remote_relative'],inputs/leaf);(inputs/leaf).chmod(0o444)
with(inputs/'manifest.json').open('x')as f:json.dump(c['manifest'],f,indent=2,sort_keys=True);f.write('\n')
(inputs/'manifest.json').chmod(0o444);inputs.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'audit_posterior_valid_four_nn.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--inputs',str(inputs),'--out',str(out/'audit'),'--max-seconds','850'];started=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=930);rc=x.returncode;stdout=x.stdout;stderr=x.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or'');stderr=str(e.stderr or'');error='coordinatorwaittimeout; inspectexistingboundedrun, no retry'
for suffix,value in(('coordinator-stdout.log',stdout),('coordinator-stderr.log',stderr)):
 with(out.parent/(name+'.'+suffix)).open('x')as f:f.write(value)
rs=load(status);result=load(out/'audit/result.json');passed=bool(rc==0 and rs and rs.get('status')=='completed'and result and result.get('status')=='COMPLETED_FOUR_OWNED_NN_PRIVILEGED_VALID_DIAGNOSIS')
value=dict(name=name,returncode=rc,protocol_completed=passed,error=error,whole_runner_wall_seconds=time.monotonic()-started,source_sha256=c['source_sha256'],command=cmd,result_path=str(out/'audit/result.json'),root_status_path=str(status),SGD_updates=0,NNs=4,truth_used_only_after_all_NN_raw_outputs=True)
with receipt.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(value,sort_keys=True));raise SystemExit(0 if passed else(rc if rc not in(None,0)else 1))
'''
def audit_template():return AUDIT_TEMPLATE

def prepare_audit():
 from extended_closure_launch import checkpoint_header
 truthfolder=ROOT/'runs'/NAMES['truth']/'truth';truthpath=truthfolder/'prefix-truth.npz';metapath=truthfolder/'result.json';teacher=json.loads(metapath.read_text());receipt=json.loads((ROOT/(NAMES['truth']+'-coordinator-receipt.json')).read_text())
 if receipt.get('returncode')!=0 or teacher['status']!='PRIVILEGED_VALID_PREFIX_TRUTH_COMPLETE'or teacher['VALID_sha256']!=VALID_SHA or teacher['official_oracle_sha256']!=ORACLE_SHA or sha(truthpath)!=teacher['truth_sha256']:raise ValueError('actualcompletedCPUtruth required')
 models=[];inputs={'valid.npz':dict(remote_relative=VALID_REL,sha256=VALID_SHA,bytes=(ROOT/'runs'/VALID_REL).stat().st_size),'prefix-truth.npz':dict(remote_relative=str(truthpath.relative_to(ROOT/'runs')),sha256=sha(truthpath),bytes=truthpath.stat().st_size),'truth-result.json':dict(remote_relative=str(metapath.relative_to(ROOT/'runs')),sha256=sha(metapath),bytes=metapath.stat().st_size)}
 for mode in('posterior','raw'):
  parent='parameter-posterior-value-'+mode+'-n8-seed0-fourstage-graph-v2';folder=ROOT/'runs'/parent/'learn';result=json.loads((folder/'result.json').read_text());protocol=json.loads((ROOT/(parent+'-protocol.json')).read_text());version=ROOT/'versions'/('code-'+parent)
  if result['status']!='ACTUAL_FOUR_STAGE_POSTERIOR_VALUE_GRAPH_COMPLETE'or result['value_projection']!=mode or result['actual_new_updates_this_process']!=2400 or result['source_sha256']!=protocol['source_sha256']:raise ValueError('exactactualfreshNNresult required')
  for leaf,digest in result['source_sha256'].items():
   if sha(version/leaf)!=digest:raise ValueError('actualNNsource changed')
  for stage in(2,3):
   cp=result['reports'][stage]['checkpoint'];path=folder/cp['file'];P=result['reports'][stage]['resources']['parameters']
   if sha(path)!=cp['sha256']or path.stat().st_size!=cp['bytes']or P!=(13381 if stage==2 else 13397)or checkpoint_header(path)!=dict(schema='posterior-value-fourstage-graph-training-v1',stage=stage,step=600,stage_complete=True):raise ValueError('actualNNcheckpoint stage/header/bytes mismatch')
   key=mode+'-stage'+str(stage);leaf=key+'.pt';models.append(dict(key=key,mode=mode,stage=stage,P=P,parent=parent,leaf=leaf,sha256=cp['sha256'],bytes=cp['bytes'],parent_source_sha256=result['source_sha256']));inputs[leaf]=dict(remote_relative=str(path.relative_to(ROOT/'runs')),sha256=cp['sha256'],bytes=cp['bytes'])
 canonical=ROOT/'versions/code-parameter-posterior-value-posterior-n8-seed0-fourstage-graph-v2';sources={leaf:(canonical/leaf).read_bytes()for leaf in('continuous_primitive_block.py','owned_fcba_core.py')};sources['audit_posterior_valid_four_nn.py']=(ROOT/'audit_posterior_valid_four_nn.py').read_bytes();hashes={leaf:hashlib.sha256(raw).hexdigest()for leaf,raw in sources.items()}
 if hashes['continuous_primitive_block.py']!='d25a0ddda846ab0858873eca37e1ccb8269a083b312c2af2dc9ad6923b61af80':raise ValueError('exactactualposterior/raw rule source required')
 manifest=dict(schema='four-owned-posterior-VALID-audit-v1',models=models,source_sha256=hashes,truth_sha256=sha(truthpath),truth_result_sha256=sha(metapath),official_oracle_sha256=ORACLE_SHA,VALID_sha256=VALID_SHA)
 config=dict(remote=transport.REMOTE,runner_SHA=GPU,input_files=inputs,manifest=manifest)
 doc=dict(scope='ONEfrozenfourcheckpointPRIVILEGEDVALIDdiagnosis,0SGD,notcapacityproof/newrepair',models=models,VALID_sha256=VALID_SHA,privileged_truth_sha256=sha(truthpath),privileged_truth_result_sha256=sha(metapath),teacher_source_sha256=ORACLE_SHA,teacher_generation_costs=teacher,rows_per_NN=512,events=['write_first','write_second','after_read','before_mix_after_key_other','at_mix'],NNinput_fields=['initial','tokens','lengths'],NNscope='restoreeachtrueNN thenexactactive-row_transition once/position; capturesbefore/aftercarry248/read_p/write_p/candidate/attention/gates withouttruth; saveALL4NNrawfiles+markerBEFOREtruthload; no seconddecoder or teacherNNarguments',privileged_scope='posthoctrueq/selectedmemorycomparisons ONLY; sameUIDcanreadtruth, sourceorderingnotOSblindness',stage2_scope='derivecopy lengthminus1 andreplaceinactivefinal15 withPAD0, analyzeoldprefix4events, markMIXunexecuted; preserveoriginalVALID',SGD_updates=0,new_learner_labels=0,logical_NNpasses=4,outside_seconds=900,inside_seconds=850,AS_GiB=512,gpu_fraction=.3,file_limit_MiB=100,runner_sha256=GPU,costs='all4NNrestores/initialcache/active_transition/eventstores/D2H/raw+truthstorage/CPU/GPUstream/wholewall/P/carry/checkpoint/historicalAdam+replay charged; CUDAstreamnotGPUbusy/fullFLOPs unknown')
 freeze('audit',sources,config,doc,audit_template())

def execute(mode):
 name=NAMES[mode];v=ROOT/'versions'/('code-'+name);c=json.loads((v/'launch-config.json').read_text());protocol=json.loads((ROOT/(name+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text();template=truth_template()if mode=='truth'else audit_template()
 if sha(v/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=template.replace('CONFIG',repr(c),1):raise ValueError('frozen coordinator differs')
 for leaf,digest in c['source_sha256'].items():
  if sha(v/leaf)!=digest:raise ValueError('frozen source differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserveactualattempt; no retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(transport.SSH,input=source,text=True,capture_output=True,timeout=210 if mode=='truth'else 980);d=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:d=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSHwaittimeout; inspectboundedrun, no retry')
 d['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,d);print(json.dumps(dict(name=name,returncode=d['returncode'],wall_seconds=d['wall_seconds'])));raise SystemExit(d['returncode']if d['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('truth','audit'));g.add_argument('--execute',choices=('truth','audit'));a=p.parse_args()
 if a.prepare:
  if a.prepare=='truth':prepare_truth()
  else:prepare_audit()
 else:execute(a.execute)
