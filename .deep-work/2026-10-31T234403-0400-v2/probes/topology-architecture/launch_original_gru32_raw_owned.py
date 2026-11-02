"""Freeze one0SGD original GRU32 four-split raw-output extraction.

--prepare is local only. ROOT alone invokes --execute after the three new
architecture jobs, under the existing unchanged offline GPU runner/resources.
"""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-original-gru32-4split-raw-eval-v1'
REMOTE='/workspace/prometheus-v2-m0-architecture'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
DRIVER='audit_original_gru32_saved_predictions.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
    with p.open('x')as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
REMOTE_SCRIPT=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG
r=pathlib.Path(c['remote'])
code=r/('code-'+c['name'])
out=r/'outputs'/c['name']
inputs=pathlib.Path('/opt/prometheus-'+c['name']+'-inputs')
status=out.parent/(out.name+'.runner-status.json')
receipt=out.parent/(out.name+'.coordinator-exit.json')
runner=r/'remote_runner_gpu_authorized.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())if p.is_file()else None
if any(p.exists()or p.is_symlink()for p in(code,out,inputs,status,receipt)):raise FileExistsError('one new namespace; preserve evidence')
if sha(runner)!=c['runner_sha256']:raise ValueError('unchanged old85 runner required')
for spec in c['manifest']['files'].values():
 source=r/'outputs'/spec['remote_relative']
 if source.is_symlink()or not source.is_file()or source.stat().st_size!=spec['bytes']or sha(source)!=spec['sha256']:raise ValueError('fixed owned input changed before writes')
code.mkdir();inputs.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded,validate=True)
 if pathlib.Path(leaf).name!=leaf or hashlib.sha256(raw).hexdigest()!=c['manifest']['source_sha256'][leaf]:raise ValueError('source SHA')
 path=code/leaf
 with path.open('xb')as f:f.write(raw)
 path.chmod(0o444)
for spec in c['manifest']['files'].values():
 target=inputs/spec['leaf'];shutil.copyfile(r/'outputs'/spec['remote_relative'],target);target.chmod(0o444)
with(inputs/'manifest.json').open('x')as f:json.dump(c['manifest'],f,indent=2,sort_keys=True);f.write('\n')
(inputs/'manifest.json').chmod(0o444);inputs.chmod(0o555);code.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--inputs',str(inputs),'--out',str(out/'audit')]
start=time.monotonic()
try:x=subprocess.run(cmd,text=True,capture_output=True,timeout=930);rc=x.returncode;stdout=x.stdout;stderr=x.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or'');stderr=str(e.stderr or'');error='wait timeout; inspect bounded existing runner; no retry'
for suffix,value in [('coordinator-stdout.log',stdout),('coordinator-stderr.log',stderr)]:
 with(out.parent/(out.name+'.'+suffix)).open('x')as f:f.write(value)
rs=load(status);result=load(out/'audit/result.json')
passed=bool(rc==0 and rs and rs.get('status')=='completed'and result and result.get('status')=='ORIGINAL_GRU32_FROZEN_4SPLIT_RAW_EXTRACTION_COMPLETE'and result.get('P')==14024 and result.get('forward_examples')==2048 and result.get('forward_calls')==32 and result.get('SGD_updates')==0 and result.get('NN_unchanged')is True and result.get('all_four_raw_saved_before_targets')is True and result.get('source_sha256')==c['manifest']['source_sha256'])
record=dict(name=c['name'],returncode=rc,protocol_completed=passed,error=error,command=cmd,whole_runner_wall_seconds=time.monotonic()-start,result_path=str(out/'audit/result.json'),source_sha256=c['manifest']['source_sha256'],no_accuracy_gate=True,SGD_updates=0)
with receipt.open('x')as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(record));raise SystemExit(0 if passed else(rc if rc not in(None,0)else 1))
'''
def prepare():
    version=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
    if any(p.exists()for p in(version,protocol,ROOT/(NAME+'-coordinator-receipt.json'),ROOT/'runs'/NAME)):raise FileExistsError('new freeze only')
    parent='parameter-gru32-n8-seed0-fourstage-v1';folder=ROOT/'runs'/parent/'learn';result=json.loads((folder/'result.json').read_text())
    records=[json.loads(x)for x in(folder/'records.jsonl').read_text().splitlines()]
    if [r['step']for r in records if r.get('kind')=='actual_update'and r['stage']==3]!=list(range(1,601))or result['actual_training_updates']!=2400:raise ValueError('original completed2400 required')
    checkpoint=result['reports'][-1]['checkpoint'];cp=folder/checkpoint['file']
    if checkpoint['file']!='gru32-stage3-step600.pt'or checkpoint['sha256']!='66b60a19ac387af47a810f61fa8c4768570e94a46e2ce58906cb28abaa6f3e01'or sha(cp)!=checkpoint['sha256']or cp.stat().st_size!=checkpoint['bytes']:raise ValueError('exact OWN original finalNN')
    old=ROOT/'versions'/('code-'+parent)/'owned_parent_driver.py'
    if sha(old)!='cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc':raise ValueError('original packed class/backend')
    raws={DRIVER:(ROOT/DRIVER).read_bytes(),'owned_parent_driver.py':old.read_bytes()}
    for leaf,raw in raws.items():ast.parse(raw,filename=leaf)
    sources={k:hashlib.sha256(v).hexdigest()for k,v in raws.items()}
    manifest=dict(schema='original-gru32-frozen-4split-raw-v1',source_sha256=sources,parent_source_sha256=result['source_sha256'],files={},data={},original_scores={})
    def add(key,path):
        entry=dict(leaf=key+path.suffix,remote_relative=str(path.relative_to(ROOT/'runs')),sha256=sha(path),bytes=path.stat().st_size)
        manifest['files'][key]=entry;return entry
    manifest['checkpoint']=add('checkpoint',cp)
    metadata_path=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/metadata.json';metadata=json.loads(metadata_path.read_text())
    for split in('valid','test_iid','test_long','test_unseen'):
        path=metadata_path.parent/metadata['files'][split]['file']
        if sha(path)!=metadata['files'][split]['sha256']:raise ValueError('same original data')
        manifest['data'][split]=add(split,path)
        manifest['original_scores'][split]=result['reports'][-1]['curves'][-1]['valid']['3']['soft']if split=='valid'else result['tests']['3'][split]['soft']
    config=dict(name=NAME,remote=REMOTE,runner_sha256=GPU,driver=DRIVER,manifest=manifest,files={k:base64.b64encode(v).decode()for k,v in raws.items()})
    remote=REMOTE_SCRIPT.replace('CONFIG',repr(config),1);ast.parse(remote)
    version.mkdir()
    for leaf,raw in raws.items():(version/leaf).write_bytes(raw);(version/leaf).chmod(0o444)
    (version/'remote-launch.source').write_text(remote);(version/'remote-launch.source').chmod(0o444);put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
    doc=dict(name=NAME,before_execution=True,schema=manifest['schema'],source_sha256=sources,manifest=manifest,runner_sha256=GPU,remote_launch_sha256=sha(version/'remote-launch.source'),
        original_NN_checkpoint=checkpoint,original_effective_training_updates=2400,NN_updates_this_job=0,optimizer_steps=0,
        purpose='Fill missing perexample predictions of existing originalGRU32, not retrain/tune/change configuration or choose a favorable baseline.',
        forwards='Exactly4x512 rows, original cc1 batch/TerminalGRU/packed backend, one soft forward each; native is documented same-function alias, no duplicate forward.',
        raw_before_labels='logits/logp/predictions/rowindex/publicinitial/lengths four files before target array access; no privileged truth source copied.',
        outside_seconds=900,science_seconds=120,AS_GiB=512,file_limit_MiB=100,GPU_fraction=.3,
        isolation='old unchanged GPUrunner; existing trustedCUDAbootstrap; strict science execution; no compiler or newpermissions/dependencies.',
        no_accuracy_gate=True,reproduction='Original aggregate comparison descriptive only; no tuning/retry based on quality.',
        original_cost_history=result['reports'],formal_Primary=False,ROOT_serial_only_after_A_B_AB=True)
    put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555)
    print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=sources,protocol_sha256=sha(protocol),remote_launch_sha256=doc['remote_launch_sha256'])))
def execute():
    version=ROOT/'versions'/('code-'+NAME);protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());config=json.loads((version/'launch-config.json').read_text());remote=(version/'remote-launch.source').read_text()
    literal=next(n for n in ast.parse(remote).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));frozen=ast.literal_eval(literal.value)
    if frozen!=config or remote!=REMOTE_SCRIPT.replace('CONFIG',repr(frozen),1)or sha(version/'remote-launch.source')!=protocol['remote_launch_sha256']:raise ValueError('frozen launcher/config changed')
    for leaf,digest in protocol['source_sha256'].items():
        if sha(version/leaf)!=digest:raise ValueError('frozen science source changed')
    receipt=ROOT/(NAME+'-coordinator-receipt.json')
    if receipt.exists():raise FileExistsError('preserve one actual attempt')
    start=time.monotonic()
    try:x=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=980);record=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-start)
    except subprocess.TimeoutExpired as e:record=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-start,error='wait timeout; inspect bounded existing runner, no automatic retry')
    record['remote_launch_sha256']=protocol['remote_launch_sha256'];put(receipt,record);print(json.dumps(dict(name=NAME,returncode=record['returncode'],wall_seconds=record['wall_seconds'])));raise SystemExit(record['returncode']if record['returncode']is not None else 124)
if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');args=p.parse_args();(prepare if args.prepare else execute)()
