"""ROOT-only fixed-width GRU34/TF24 AR packages; no execution in preparation."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
CORE='58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497'
OLD_DRIVER='567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02'
META='f157d58e9246f59f78dbd883b4ccc80284278b9eb270365cb13556dc8703e792'
BATCH='538c1157cacea82a51bf5be950d451768f013ee7178f5aabc100e339b966f615'
DATA='parameter-char-alltoken-data-v1'
CHECK='parameter-char-alltoken-parameter-matched-interface-cpu-v1'
DRIVER='train_char_parameter_matched_alltoken.py'
CHECKER='check_char_parameter_matched_alltoken.py'
COUNTS='parameter-char-alltoken-parameter-matched-counts-v1.json'
ARMS={'gru':dict(name='parameter-char-alltoken-gru34-300-v1',P=21180,width=34,layers=2,feedforward=None),
      'transformer':dict(name='parameter-char-alltoken-transformer24-300-v1',P=19564,width=24,layers=3,feedforward=48)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def checked(p,expected=None):
 if p.is_symlink()or not p.is_file()or p.stat().st_size>100*1024*1024:raise ValueError('OWN bounded regular file required: '+str(p))
 result=dict(bytes=p.stat().st_size,sha256=sha(p))
 if expected is not None and result['sha256']!=expected:raise ValueError('OWN exact SHA differs: '+str(p))
 return result

TEMPLATE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists()or out.exists()or inputs.exists():raise FileExistsError('new namespace only; preserve all attempts')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if not c['is_check']:
 report=json.loads((r/'outputs'/c['CPU_check_run']/'check/result.json').read_text())
 if report.get('status')!='CPU_PARAMETER_MATCHED_ALLTOKEN_CONTROLS_PASS'or report.get('SGD_updates')!=0 or report['models'][c['arm']]['P']!=c['expected_P']or report['common_batch_indices_sha256']!=c['BATCH']:raise ValueError('actual same-source CPU/P/causality/reload admission required')
 for leaf,digest in c['source_sha256'].items():
  if report['source_sha256'].get(leaf)!=digest:raise ValueError('CPU/GPU scientific source differs')
code.mkdir();inputs.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('frozen scientific source transport differs')
 target=code/leaf;target.write_bytes(raw);target.chmod(0o444)
code.chmod(0o555)
data_inputs=inputs if c['is_check']else inputs/'data';data_inputs.mkdir(exist_ok=True)
for leaf,spec in c['data_files'].items():
 source=r/'outputs'/c['DATA']/'data'/leaf
 if source.is_symlink()or not source.is_file()or source.stat().st_size!=spec['bytes']or sha(source)!=spec['sha256']:raise ValueError('same original AR data differs: '+leaf)
 target=data_inputs/leaf;shutil.copyfile(source,target);target.chmod(0o444)
if c['is_check']:
 raw=base64.b64decode(c['manifest_bytes'])
 if hashlib.sha256(raw).hexdigest()!=c['manifest_sha256']:raise ValueError('CPU admission manifest differs')
 target=inputs/'manifest.json';target.write_bytes(raw);target.chmod(0o444)
for target in inputs.rglob('*'):target.chmod(0o555 if target.is_dir()else 0o444)
inputs.chmod(0o555)
runner=r/('remote_runner_cpu_v2.py'if c['is_check']else'remote_runner_gpu_authorized.py')
if sha(runner)!=(c['CPU']if c['is_check']else c['GPU']):raise ValueError('unchanged existing bounded runner required')
command=[sys.executable,str(runner),'--script',str(code/c['entry']),'--out',str(out),'--timeout','120'if c['is_check']else'900','--address-space-gib','64'if c['is_check']else'512','--file-limit-mib','100']
if c['is_check']:command+=['--cpu-only','--','--inputs',str(inputs),'--out',str(out/'check'),'--max-seconds','90']
else:command+=['--gpu-fraction','.3','--cuda-init-before-seccomp','--','--data',str(inputs/'data'),'--out',str(out/'learn'),'--arm',c['arm'],'--seed','0','--device','cuda:0','--max-seconds','850','--resource-overlap-note','ROOT records actual concurrency; not isolated throughput or matched quality']
print(json.dumps(dict(event='frozen_parameter_matched_alltoken_launch',name=name,command=command)),flush=True);tick=time.monotonic()
try:
 x=subprocess.run(command,capture_output=True,text=True,timeout=150 if c['is_check']else 930)
 receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,complete_runner_wall_seconds=time.monotonic()-tick)
except subprocess.TimeoutExpired as e:
 receipt=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),complete_runner_wall_seconds=time.monotonic()-tick,error='inspect actual bounded existing process; no automatic retry')
receipt['original_data_unchanged']={leaf:sha(r/'outputs'/c['DATA']/'data'/leaf)==spec['sha256']for leaf,spec in c['data_files'].items()}
if out.exists():
 with(out/'coordinator-exit.json').open('x')as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True);raise SystemExit(receipt['returncode']if receipt['returncode']is not None else 124)
'''

def data_facts(is_check):
 # The requested dependent architecture check has ACTUALLY passed. Do not
 # continue baseline preparation from a failed/incomplete factorized contract.
 fact=json.loads((ROOT/'runs/parameter-char-alltoken-factorized-interface-cpu-v1/check/result.json').read_text())
 if fact.get('status')!='CPU_FACTORIZED_ALLTOKEN_INTERFACE_PASS'or fact['P']!=20721:raise ValueError('actual factorized architecture CPU contract required')
 folder=ROOT/'runs'/DATA/'data';checked(folder/'metadata.json',META);meta=json.loads((folder/'metadata.json').read_text())
 if meta['schema']!='char-alltoken-ar-development-v1'or meta['n_tokens']!=100 or meta['n_states']!=100:raise ValueError('same actual original AR vocabulary100 required')
 old=ROOT/'runs/parameter-char-alltoken-dynamic-300-v1/learn';summary=json.loads((old/'summary.json').read_text())
 if summary['completed_updates']!=300 or summary['metadata_sha256']!=META or summary['batch_indices_sha256']!=BATCH or summary['source_sha256']['train_char_alltoken.py']!=OLD_DRIVER:raise ValueError('actual original300 schedule/source differs')
 checked(old/'common-batch-indices.npz',BATCH)
 files={'metadata.json':checked(folder/'metadata.json',META)}
 for split,entry in meta['files'].items():
  if is_check and split!='train':continue
  for leaf,digest,size in((entry['file'],entry['sha256'],entry['bytes']),(entry['endpoint_file'],entry['endpoint_sha256'],None)):
   path=folder/leaf
   if split in('train','valid'):receipt=checked(path,digest)
   else:
    if path.is_symlink()or not path.is_file():raise ValueError('actual synchronized OWN data file required')
    receipt=dict(bytes=path.stat().st_size,sha256=digest)
   if size is not None and receipt['bytes']!=size:raise ValueError('frozen metadata byte count differs')
   files[leaf]=receipt
 return files

def analytic_counts():
 path=ROOT/COUNTS;value=json.loads(path.read_text())
 if value['schema']!='analytic-frozen-parameter-matched-AR-controls-v1'or value['source_sha256']!=OLD_DRIVER or value['V']!=100:raise ValueError('pre-model analytic width selection evidence required')
 if value['rows']['gru34']['P']!=21180 or value['rows']['transformer24']['P']!=19564:raise ValueError('one fixed analytic width per family only')
 return value

def prepare(mode):
 is_check=mode=='check';arm=None if is_check else mode;name=CHECK if is_check else ARMS[arm]['name'];version=ROOT/'versions'/('code-'+name)
 for p in(version,ROOT/'runs'/name,ROOT/(name+'-protocol.json'),ROOT/(name+'-coordinator-receipt.json')):
  if p.exists():raise FileExistsError('freeze new namespace only; preserve prepared/actual artifacts: '+str(p))
 checked(ROOT/'train_char_alltoken.py',OLD_DRIVER)
 original_core=ROOT/'versions/code-parameter-char-alltoken-dynamic-300-v1/continuous_primitive_block.py';checked(original_core,CORE)
 source={'continuous_primitive_block.py':original_core.read_bytes(),DRIVER:(ROOT/DRIVER).read_bytes()}
 if is_check:source[CHECKER]=(ROOT/CHECKER).read_bytes()
 hashes={}
 for leaf,raw in source.items():ast.parse(raw,filename=leaf);hashes[leaf]=hashlib.sha256(raw).hexdigest()
 counts=analytic_counts();data=data_facts(is_check);diff=(ROOT/'train_char_parameter_matched_alltoken-v1-source-diff.patch').read_bytes()
 protocol=dict(name=name,before_execution=True,is_CPU_check=is_check,arm=arm,source_sha256=hashes,original_driver_sha256=OLD_DRIVER,source_diff_sha256=hashlib.sha256(diff).hexdigest(),
  analytic_counts=counts,analytic_count_evidence_sha256=sha(ROOT/COUNTS),selection='one fixed width per family from source-derived parameter count before new model/labels; no search/accuracy gate',
  architecture='GRU34/2layers/zero hidden; Transformer24/3layers/4heads/FF48/dropout0/normfirst/sinusoidalPE256/finalLN; original classes otherwise unchanged',
  exact_diff='width44→34 and36→24 with TF FF72→48 and corresponding PE/norm/head dimensions; proxy shape constants; fixed arms/seed guard and carry metadata; original forward/loss/train/eval/checkpoint loop unchanged',
  expected_all_P={'gru':21180,'transformer':19564}if is_check else ARMS[arm]['P'],data_run=DATA,data_metadata_sha256=META,data_files=data,common_batch_indices_sha256=BATCH,
  updates=0 if is_check else 300,batch_size=32,TRAIN_context=64,TRAIN_windows=8192,VALID_windows=512,TEST_windows=512,TEST_contexts=[64,128,256],training_label_observations=0 if is_check else 614400,
  optimizer='same Adam.003/clipnorm1/seed0',objective='same alltoken nextcharacter NLL only; no auxiliary/profile/teacher loss',
  evaluation='same original VALID/after300freeze TEST alltokens and same32768pairedlast64 targets; no new sampling or score-targeted retries',
  CPU_check_run=CHECK,CPU_must_pass_before_GPU=True,CPU_contract='actual P/buffer counts, probabilities, future-causality/prefix alignment, NN+buffer+output checkpoint reload, forward-product proxy, commonbatch, scalar backward with zeroSGD; TRAIN inputs only',
  cost_scope='allP/Adam/buffers/data/checkpoints/causality/VALID/TEST/reload/carry; original forwardMAC proxy excludes backward+optimizer but wall includes them; complete runner/SSH cold scopes separately measured, never sum nested timers; no fullFLOPs/energy or cached-serving claim',
  parameter_scope='similar P only: +459P GRU and−1157P TF vs20721P Block, not matched quality or an advantage finding',
  inference_scope='same original full-window eval forwardwall/MAC64/128/256 and reload; no extra resident latency or sequential cached generation measurement',
  outside_seconds=120 if is_check else 900,inside_seconds=90 if is_check else 850,address_space_GiB=64 if is_check else 512,gpu_fraction=None if is_check else .3,
  isolation='same host/config and unchanged existing bounded offline runners; no install/framework/API/model-route/permission expansion',formal_Primary=False,old_results_untouched=True)
 manifest=dict(schema='parameter-matched-alltoken-controls-CPU-inputs-v1',source_sha256=hashes,data_files=data,common_batch_indices_sha256=BATCH,analytic_counts=counts)
 raw=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
 config=dict(name=name,is_check=is_check,arm=arm,expected_P=None if is_check else ARMS[arm]['P'],remote=REMOTE,CPU=CPU,GPU=GPU,BATCH=BATCH,DATA=DATA,CPU_check_run=CHECK,
  entry=CHECKER if is_check else DRIVER,files={leaf:base64.b64encode(body).decode()for leaf,body in source.items()},source_sha256=hashes,data_files=data,manifest_bytes=base64.b64encode(raw).decode(),manifest_sha256=hashlib.sha256(raw).hexdigest())
 config=json.loads(json.dumps(config,sort_keys=True));remote=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(remote)
 version.mkdir()
 for leaf,body in source.items():(version/leaf).write_bytes(body)
 (version/'source-diff.patch').write_bytes(diff);(version/'input-manifest.json').write_bytes(raw);put(version/'analytic-counts.json',counts);put(version/'launch-config.json',config);put(ROOT/(name+'-protocol.json'),protocol);(version/'remote-launch.source').write_text(remote)
 return dict(status='FROZEN_UNEXECUTED_ROOT_ONLY',name=name,source_sha256=hashes,remote_sha256=sha(version/'remote-launch.source'))

def execute(name):
 if name not in{CHECK}|{v['name']for v in ARMS.values()}:raise ValueError('one CPU and two fixed conventional300 packages only')
 version=ROOT/'versions'/('code-'+name);config=json.loads((version/'launch-config.json').read_text());remote=(version/'remote-launch.source').read_text()
 if remote!=TEMPLATE.replace('CONFIG',repr(config),1):raise ValueError('frozen trusted coordinator differs')
 for leaf,digest in config['source_sha256'].items():checked(version/leaf,digest)
 if not config['is_check']:
  report=json.loads((ROOT/'runs'/CHECK/'check/result.json').read_text())
  if report.get('status')!='CPU_PARAMETER_MATCHED_ALLTOKEN_CONTROLS_PASS'or report.get('SGD_updates')!=0 or report['models'][config['arm']]['P']!=config['expected_P']or report['common_batch_indices_sha256']!=BATCH:raise ValueError('actual synchronized CPU technical admission required, no accuracy gate')
  for leaf,digest in config['source_sha256'].items():
   if report['source_sha256'].get(leaf)!=digest:raise ValueError('actual CPU/GPU source differs')
 receipt=ROOT/(name+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('preserve actual attempt; no automatic retry')
 tick=time.monotonic()
 try:
  x=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=180 if config['is_check']else 980)
  value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,complete_SSH_coordinator_wall_seconds=time.monotonic()-tick)
 except subprocess.TimeoutExpired as e:
  value=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),complete_SSH_coordinator_wall_seconds=time.monotonic()-tick,error='inspect actual bounded existing process; no automatic retry')
 value.update(name=name,remote_launch_sha256=sha(version/'remote-launch.source'),protocol_sha256=sha(ROOT/(name+'-protocol.json')),source_sha256=config['source_sha256'])
 put(receipt,value);print(json.dumps({k:value[k]for k in('name','returncode','complete_SSH_coordinator_wall_seconds')}));raise SystemExit(value['returncode']if value['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('check','gru','transformer'));g.add_argument('--execute');a=p.parse_args()
 if a.prepare:print(json.dumps(prepare(a.prepare)))
 else:execute(a.execute)
