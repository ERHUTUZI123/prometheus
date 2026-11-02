"""ROOT-only SAMEV2 packages: one CPU interface check, one fixed AR300 job.

Preparation is source/metadata hashing only. Never starts SSH, imports Torch,
reads TEST labels, executes a NN, changes prior results or trains a model.
"""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
CORE='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
OLD_DRIVER='567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02'
META='f157d58e9246f59f78dbd883b4ccc80284278b9eb270365cb13556dc8703e792'
BATCH='538c1157cacea82a51bf5be950d451768f013ee7178f5aabc100e339b966f615'
DATA='parameter-char-alltoken-data-v1'
CHECK='parameter-char-alltoken-factorized-interface-cpu-v1'
TRAIN='parameter-char-alltoken-factorized-300-v1'
DRIVER='train_char_factorized_alltoken.py'
CHECKER='check_char_factorized_alltoken.py'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
 with path.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
def checked(path,expected=None):
 if path.is_symlink()or not path.is_file()or path.stat().st_size>100*1024*1024:raise ValueError('OWN bounded regular file required: '+str(path))
 receipt=dict(bytes=path.stat().st_size,sha256=sha(path))
 if expected is not None and receipt['sha256']!=expected:raise ValueError('OWN exact SHA differs: '+str(path))
 return receipt

TEMPLATE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists()or out.exists()or inputs.exists():raise FileExistsError('new namespace only; preserve all attempts')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if c['mode']=='train':
 check=r/'outputs'/c['CPU_check_run']/'check'/'result.json';report=json.loads(check.read_text())
 if report.get('status')!='CPU_FACTORIZED_ALLTOKEN_INTERFACE_PASS'or report.get('SGD_updates')!=0 or report.get('P')!=20721 or report.get('common_batch_indices_sha256')!=c['BATCH']:raise ValueError('actual SAME-source CPU interface/reload/causality/check required before training')
 for leaf,digest in c['source_sha256'].items():
  if report['source_sha256'].get(leaf)!=digest:raise ValueError('CPU/GPU source differs')
code.mkdir();inputs.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport differs')
 target=code/leaf;target.write_bytes(raw);target.chmod(0o444)
code.chmod(0o555)
data_inputs=inputs/'data'if c['mode']=='train'else inputs;data_inputs.mkdir(exist_ok=True)
for leaf,spec in c['data_files'].items():
 source=r/'outputs'/c['DATA']/'data'/leaf
 if source.is_symlink()or not source.is_file()or source.stat().st_size!=spec['bytes']or sha(source)!=spec['sha256']:raise ValueError('same original AR data differs: '+leaf)
 target=data_inputs/leaf;shutil.copyfile(source,target);target.chmod(0o444)
if c['mode']=='check':
 raw=base64.b64decode(c['manifest_bytes'])
 if hashlib.sha256(raw).hexdigest()!=c['manifest_sha256']:raise ValueError('CPU manifest differs')
 target=inputs/'manifest.json';target.write_bytes(raw);target.chmod(0o444)
for target in inputs.rglob('*'):target.chmod(0o555 if target.is_dir()else 0o444)
inputs.chmod(0o555)
runner=r/('remote_runner_cpu_v2.py'if c['mode']=='check'else'remote_runner_gpu_authorized.py')
if sha(runner)!=(c['CPU']if c['mode']=='check'else c['GPU']):raise ValueError('unchanged existing runner required')
command=[sys.executable,str(runner),'--script',str(code/c['entry']),'--out',str(out),'--timeout','120'if c['mode']=='check'else'900','--address-space-gib','64'if c['mode']=='check'else'512','--file-limit-mib','100']
if c['mode']=='check':command+=['--cpu-only','--','--inputs',str(inputs),'--out',str(out/'check'),'--max-seconds','90']
else:command+=['--gpu-fraction','.3','--cuda-init-before-seccomp','--','--data',str(inputs/'data'),'--out',str(out/'learn'),'--arm','dynamic','--seed','0','--device','cuda:0','--max-seconds','850','--resource-overlap-note','ROOT coordinator records actual concurrency; no isolated throughput claim']
print(json.dumps(dict(event='frozen_factorized_alltoken_launch',command=command,name=name)),flush=True);tick=time.monotonic()
try:
 x=subprocess.run(command,capture_output=True,text=True,timeout=150 if c['mode']=='check'else 930)
 receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,complete_runner_wall_seconds=time.monotonic()-tick)
except subprocess.TimeoutExpired as e:
 receipt=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),complete_runner_wall_seconds=time.monotonic()-tick,error='inspect actual bounded existing process; no automatic retry')
receipt['original_data_unchanged']={leaf:sha(r/'outputs'/c['DATA']/'data'/leaf)==spec['sha256']for leaf,spec in c['data_files'].items()}
if out.exists():
 with(out/'coordinator-exit.json').open('x')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt),flush=True);raise SystemExit(receipt['returncode']if receipt['returncode']is not None else 124)
'''

def data_facts(mode):
 folder=ROOT/'runs'/DATA/'data';checked(folder/'metadata.json',META);meta=json.loads((folder/'metadata.json').read_text())
 if meta['schema']!='char-alltoken-ar-development-v1'or meta['n_tokens']!=100 or meta['n_states']!=100:raise ValueError('actual frozen AR vocabulary100 required')
 # Common batches are checked against an actual completed old300 model, not
 # an invented schedule/name. Scores/TEST vectors are never inspected here.
 oldrun=ROOT/'runs/parameter-char-alltoken-dynamic-300-v1/learn';summary=json.loads((oldrun/'summary.json').read_text())
 if summary['completed_updates']!=300 or summary['metadata_sha256']!=META or summary['batch_indices_sha256']!=BATCH or summary['source_sha256']['train_char_alltoken.py']!=OLD_DRIVER:raise ValueError('actual original fixed300 schedule/source differs')
 checked(oldrun/'common-batch-indices.npz',BATCH)
 files={'metadata.json':checked(folder/'metadata.json',META)}
 for split,entry in meta['files'].items():
  if mode=='check'and split!='train':continue
  for leaf,digest,size in((entry['file'],entry['sha256'],entry['bytes']),(entry['endpoint_file'],entry['endpoint_sha256'],None)):
   # Only TRAIN/VALID archives are byte-read at preparation. TEST identities
   # are copied from the original frozen metadata; remote verifies each hash.
   path=folder/leaf
   if split in('train','valid'):
    receipt=checked(path,digest)
   else:
    if path.is_symlink()or not path.is_file():raise ValueError('actual synchronized OWN data file required')
    receipt=dict(bytes=path.stat().st_size,sha256=digest)
   if size is not None and receipt['bytes']!=size:raise ValueError('metadata byte count differs')
   files[leaf]=receipt
 return files,meta

def prepare(mode):
 name=CHECK if mode=='check'else TRAIN;version=ROOT/'versions'/('code-'+name)
 for path in(version,ROOT/'runs'/name,ROOT/(name+'-protocol.json'),ROOT/(name+'-coordinator-receipt.json')):
  if path.exists():raise FileExistsError('preserve frozen/new namespace only: '+str(path))
 checked(ROOT/'continuous_primitive_block_factorized.py',CORE);checked(ROOT/'train_char_alltoken.py',OLD_DRIVER)
 source={'continuous_primitive_block.py':(ROOT/'continuous_primitive_block_factorized.py').read_bytes(),DRIVER:(ROOT/DRIVER).read_bytes()}
 if mode=='check':source[CHECKER]=(ROOT/CHECKER).read_bytes()
 hashes={}
 for leaf,raw in source.items():ast.parse(raw,filename=leaf);hashes[leaf]=hashlib.sha256(raw).hexdigest()
 data,meta=data_facts(mode)
 diff=(ROOT/'train_char_factorized_alltoken-v1-source-diff.patch').read_bytes()
 protocol=dict(name=name,before_execution=True,mode=mode,source_sha256=hashes,original_driver_sha256=OLD_DRIVER,exact_core_sha256=CORE,source_diff_sha256=hashlib.sha256(diff).hexdigest(),
  scope='NEW single-seed DEVELOPMENT architecture transfer; no Primary, novelty or capability/resource advantage assumption',
  architecture='explicitH32/state32/memory32/c16/K4/r8/token16/M8/key16/G8; learned control/value and identity value write; dynamic only',
  checkpoint_version='continuous-primitive-block-control-value-v1',organization_policy='learned-control-history/value-identity-write/group8-carry-v1',
  transfer='new AR initialization from seed0, not a restore of task-trained NN; same OWN fcba architecture; no anchor/profile auxiliary loss',
  exact_diff='constructor H64→32 and explicit c16/G8; wrapper alltoken report replaces terminal readout withB*L*H*V; encoderP/carry/version metadata corrected; reject other arms/seeds; training/evaluation/loss/batches unchanged',
  data_run=DATA,data_metadata_sha256=META,data_files=data,common_batch_indices_sha256=BATCH,
  updates=0 if mode=='check'else 300,batch_size=32,TRAIN_context=64,TRAIN_windows=8192,VALID_windows=512,TEST_windows=512,TEST_contexts=[64,128,256],
  training_label_observations=0 if mode=='check'else 614400,optimizer='Adam.003/clipnorm1/seed0',feedback='all nextcharacters, exact original target shift; no intermediate states/operation hints',
  evaluation='unchanged alltoken and same32768 paired last64 labels; soft/native VALID+postfreeze TEST; no TEST selection, no baseline changes',
  CPU_contract='actual TRAIN-input-only shape/normalized probability/streamed-step/terminal-alignment/future-causality/NNreload/MAC/P/carry/commonbatch + synthetic scalar backward, zeroSGD',
  CPU_check_run=CHECK,CPU_must_pass_before_GPU=True,expected_all_P=20721,expected_initial_encoder_P=3200,expected_logical_carry_scalars=432,expected_logical_carry_bytes_float32=1728,
  cost_scope='all P/buffers/Adam/data/checkpoints/causality/VALID/soft-native TEST/reload; alltoken readout each position; actual backward included in wall not forward MAC; cold imports/process/runner/SSH nested and separately reported; no fullFLOPs/energy/serving-cache claim',
  inference_scope='original fullwindow eval forwardwall/MAC for64/128/256 plus reload; no separate resident latency/throughput or cached sequential-generation benchmark',
  outside_seconds=120 if mode=='check'else 900,inside_seconds=90 if mode=='check'else 850,address_space_GiB=64 if mode=='check'else 512,gpu_fraction=None if mode=='check'else .3,
  isolation='unchanged original runners, same host/config, no network/install/framework/permission growth',formal_Primary=False,original_baselines_untouched=True)
 manifest=dict(schema='char-factorized-alltoken-interface-inputs-v1',source_sha256=hashes,data_files=data,common_batch_indices_sha256=BATCH)
 raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
 config=dict(name=name,mode=mode,remote=REMOTE,CPU=CPU,GPU=GPU,CORE=CORE,BATCH=BATCH,DATA=DATA,CPU_check_run=CHECK,entry=CHECKER if mode=='check'else DRIVER,
  files={k:base64.b64encode(v).decode()for k,v in source.items()},source_sha256=hashes,data_files=data,manifest_bytes=base64.b64encode(raw).decode(),manifest_sha256=hashlib.sha256(raw).hexdigest())
 config=json.loads(json.dumps(config,sort_keys=True));remote=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(remote)
 version.mkdir()
 for leaf,body in source.items():(version/leaf).write_bytes(body)
 (version/'source-diff.patch').write_bytes(diff);(version/'input-manifest.json').write_bytes(raw);put(version/'launch-config.json',config);put(ROOT/(name+'-protocol.json'),protocol);(version/'remote-launch.source').write_text(remote)
 return dict(status='FROZEN_UNEXECUTED_ROOT_ONLY',name=name,source_sha256=hashes,remote_sha256=sha(version/'remote-launch.source'))

def execute(name):
 if name not in(CHECK,TRAIN):raise ValueError('two fixed packages only')
 version=ROOT/'versions'/('code-'+name);config=json.loads((version/'launch-config.json').read_text());remote=(version/'remote-launch.source').read_text()
 if remote!=TEMPLATE.replace('CONFIG',repr(config),1):raise ValueError('exact frozen coordinator differs')
 for leaf,digest in config['source_sha256'].items():checked(version/leaf,digest)
 if config['mode']=='train':
  report=json.loads((ROOT/'runs'/CHECK/'check/result.json').read_text())
  if report.get('status')!='CPU_FACTORIZED_ALLTOKEN_INTERFACE_PASS'or report.get('SGD_updates')!=0 or report.get('P')!=20721 or report.get('common_batch_indices_sha256')!=BATCH:raise ValueError('actual synchronized CPU pass required before ROOT GPU launch')
  for leaf,digest in config['source_sha256'].items():
   if report['source_sha256'].get(leaf)!=digest:raise ValueError('actual same-source CPU/GPU proof differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actual attempt; no rerun')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=180 if config['mode']=='check'else 980)
  value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,complete_SSH_coordinator_wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:
  value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),complete_SSH_coordinator_wall_seconds=time.monotonic()-tick,error='inspect actual bounded existing process; no automatic retry')
 value.update(name=name,remote_launch_sha256=sha(version/'remote-launch.source'),protocol_sha256=sha(ROOT/(name+'-protocol.json')),source_sha256=config['source_sha256'])
 put(receipt,value);print(json.dumps({k:value[k]for k in('name','returncode','complete_SSH_coordinator_wall_seconds')}));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('check','train'));g.add_argument('--execute');a=p.parse_args()
 if a.prepare:print(json.dumps(prepare(a.prepare)))
 else:execute(a.execute)
