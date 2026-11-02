"""Trusted ROOT coordinator, fixed new architecture and two loss arms.

No preparation executes a model, SSH or training. New namespaces only.
All scientific execution uses unchanged existing bounded offline UID/LL/seccomp
runners. This coordinator transports OWN inspected sources/data, not secrets.
"""
import argparse,ast,base64,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
GPU='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
OLD_DATA='parameter-context-data-n8-seed0-v1'
NEW_DATA='parameter-memory-live-stage-data-n8-seed0-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(path,value):
    with path.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')

REMOTE_SCRIPT=r'''import pathlib,json,base64,hashlib,subprocess,sys,time,shutil
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists()or out.exists()or inputs.exists():raise FileExistsError('new namespace only, preserve all attempts')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
code.mkdir()
for leaf,raw in c['files'].items():
 b=base64.b64decode(raw)
 if hashlib.sha256(b).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('source transport mismatch')
 p=code/leaf;p.write_bytes(b);p.chmod(0o444)
code.chmod(0o555)
if c['mode']=='cpu':
 runner=r/'remote_runner_cpu_v2.py'
 if sha(runner)!=c['CPU']:raise ValueError('unchanged CPU runner differs')
 command=[sys.executable,str(runner),'--script',str(code/'check_factorized_value_block.py'),'--out',str(out),'--timeout','150','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out/'check')]
else:
 inputs.mkdir()
 for key,spec in c['manifest']['files'].items():
  source=r/'outputs'/spec['remote_relative']
  if source.is_symlink()or not source.is_file()or source.stat().st_size>100*1024*1024 or sha(source)!=spec['sha256']:raise ValueError('owned input mismatch: '+key)
  target=inputs/spec['leaf'];shutil.copyfile(source,target);target.chmod(0o444)
 p=inputs/'manifest.json';p.write_text(json.dumps(c['manifest'],sort_keys=True,indent=2)+'\n');p.chmod(0o444);inputs.chmod(0o555)
 runner=r/'remote_runner_gpu_authorized.py'
 if sha(runner)!=c['GPU']:raise ValueError('unchanged GPU runner differs')
 command=[sys.executable,str(runner),'--script',str(code/'train_factorized_value_block.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--inputs',str(inputs),'--resume',str(inputs/'resume.pt'),'--out',str(out/'learn'),'--anchor',str(c['anchor']),'--max-seconds','850']
print(json.dumps(dict(event='actual_frozen_launch',name=name,command=command)),flush=True)
tick=time.monotonic();x=subprocess.run(command,text=True,capture_output=True,timeout=930)
print(json.dumps(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,whole_runner_wall_seconds=time.monotonic()-tick)),flush=True)
raise SystemExit(x.returncode)
'''

def prepare(mode,anchor):
    name='parameter-factorized-value-cpu-v1'if mode=='cpu'else f'parameter-factorized-value-anchor{anchor}-n8-seed0-fourstage-v3'
    version=ROOT/'versions'/('code-'+name)
    if version.exists()or(ROOT/(name+'-protocol.json')).exists():raise FileExistsError('freeze once; no overwrite')
    sources={'continuous_primitive_block.py':(ROOT/'continuous_primitive_block_factorized.py').read_bytes()}
    if mode=='cpu':sources['check_factorized_value_block.py']=(ROOT/'check_factorized_value_block.py').read_bytes()
    else:
        originals={'owned_parent_driver.py':(ROOT/'versions/code-parameter-context-n8-seed0-dynamic-stage3-v1-soft_only-continuous-resume-replay128x16/train_contextual_primitives.py','cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc'),
                   'context_replay128.py':(ROOT/'context_replay128.py','cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140'),
                   'memory_live_stage_train.py':(ROOT/'versions/code-parameter-memory-live-stage-n8-seed0-dynamic-stage4-v1/memory_live_stage_train.py','dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6')}
        for leaf,(p,digest)in originals.items():
            if sha(p)!=digest:raise ValueError('inspected original source changed')
            sources[leaf]=p.read_bytes()
        sources['train_factorized_value_block.py']=(ROOT/'train_factorized_value_block.py').read_bytes()
    hashes={k:hashlib.sha256(v).hexdigest()for k,v in sources.items()}
    for value in sources.values():ast.parse(value)
    manifest=dict(schema='factorized-value-fourstage-inputs-v1',source_sha256=hashes,files={})
    if mode!='cpu':
        olddir=ROOT/'runs'/OLD_DATA/'data';meta=json.loads((olddir/'metadata.json').read_text())
        newdir=ROOT/'runs'/NEW_DATA/'generated/data';newmeta=json.loads((newdir/'metadata.json').read_text())
        for stage in range(4):
            for split in ('train','valid','test_iid','test_long','test_unseen'):
                if stage<3:
                    src=olddir/meta['stages'][stage]['files'][split];digest=meta['stages'][stage]['hashes'][split]
                else:
                    spec=newmeta['files'][split];src=newdir/spec['file'];digest=spec['sha256']
                # Metadata fixes TEST digests; only TRAIN/VALID byte-checks at preparation.
                if split in ('train','valid')and sha(src)!=digest:raise ValueError('actual original TRAIN/VALID changed')
                manifest['files'][f'{stage}:{split}']=dict(leaf=f'stage{stage}-{split}.npz',sha256=digest,remote_relative=str(src.relative_to(ROOT/'runs')))
        parent=f'parameter-factorized-value-anchor{anchor}-n8-seed0-fourstage-v2'
        folder=ROOT/'runs'/parent/'learn';prior=json.loads((folder/'completed-stage2-before-TEST.json').read_text())
        if prior.get('test_data_read')is not False or prior['report']['stage']!=2:raise ValueError('actual saved stage2 boundary required')
        cp=prior['report']['checkpoint'];checkpoint=folder/cp['file']
        if sha(checkpoint)!=cp['sha256']:raise ValueError('saved actual stage2 checkpoint SHA differs')
        original=json.loads((ROOT/(parent+'-protocol.json')).read_text())
        manifest['resume']=dict(parent_run=parent,checkpoint_sha256=cp['sha256'],source_sha256=original['source_sha256'])
        extra={'resume':(checkpoint,'resume.pt'),'oldrecords':(folder/'records.jsonl','original-paid-records.jsonl')}
        extra.update({f'report{s}':(folder/f'completed-stage{s}-before-TEST.json',f'prior-report{s}.json')for s in range(3)})
        for key,(src,leaf)in extra.items():manifest['files'][key]=dict(leaf=leaf,sha256=sha(src),remote_relative=str(src.relative_to(ROOT/'runs')))
    config=dict(name=name,mode=mode,anchor=anchor,remote=REMOTE,CPU=CPU,GPU=GPU,manifest=manifest,
        files={k:base64.b64encode(v).decode()for k,v in sources.items()},source_sha256=hashes)
    protocol=dict(name=name,before_execution=True,mode=mode,source_sha256=hashes,formal_Primary=False,
        architecture='H32 value/c16 learned GRU history/K4 rank8/M8 key16/identity value write/G8 carry; normalized learned value encoding',
        target_hypothesis='same observable terminal value obtains a reusable computational interface, while removing large write-value projection reduces P; not guaranteed',
        objective='NLL + anchor*mean((hT-stopgrad(encode_state(terminal_y)))^2)',anchor=anchor,
        information='same raw initial/tokens/terminal y; no intermediate states/routes/program/teacher operator, no new task samples',
        fixed_budget='four600-step learning lineage; v3 restores OWN stage2 and executes only stage3 600; old failed600 or stopped8 counted separately; Adam.003/clip1/seed0 unchanged',
        repair='posttraining replay storage slices zero padding128to48 without changing any training input/label/forward/update; old negative/interrupted evidence preserved',
        comparator='anchor0 vs1 SAME new architecture; historical original, GRU44 and Transformer32 same source data/batch/replay/updates; versions separate',
        measurements='all VALID curves; freeze all training then all old/new TEST splits; NN/Adam reload; joint P+forwardMAC+actual full training time/eval/reload/maintenance/storage/inference',
        boundaries='one seed development, no Primary/no universal novel rule; TEST not used to choose configuration or extend budget',
        stopping='per-process external150CPU/900GPU; internal850GPU; nonfinite/source/restore errors preserve failure and stop that process; no accuracy gate')
    version.mkdir()
    for leaf,raw in sources.items():(version/leaf).write_bytes(raw)
    put(version/'launch-config.json',config);put(ROOT/(name+'-protocol.json'),protocol)
    remote=REMOTE_SCRIPT.replace('CONFIG',repr(config),1);ast.parse(remote);(version/'remote-launch.source').write_text(remote)
    print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=name,source_sha256=hashes,remote_sha256=sha(version/'remote-launch.source'))))

def execute(name):
    allowed={'parameter-factorized-value-cpu-v1'}|{f'parameter-factorized-value-anchor{x}-n8-seed0-fourstage-v3'for x in (0,1)}
    if name not in allowed:raise ValueError('fixed names only')
    v=ROOT/'versions'/('code-'+name);c=json.loads((v/'launch-config.json').read_text())
    for leaf,digest in c['source_sha256'].items():
        if sha(v/leaf)!=digest:raise ValueError('frozen source differs')
    if c['mode']!='cpu':
        check=ROOT/'runs/parameter-factorized-value-cpu-v1/check/result.json'
        result=json.loads(check.read_text())
        if result.get('status')!='CPU_FACTOR_VALUE_RULE_CHECK_PASSED'or result['source_sha256']!=c['source_sha256']['continuous_primitive_block.py']:raise ValueError('actual same-source CPU check required')
    source=(v/'remote-launch.source').read_text()
    frozen_tree=ast.parse(source)
    frozen_assignment=next(n for n in frozen_tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets))
    frozen_config=ast.literal_eval(frozen_assignment.value)
    # JSON sorting changes dict insertion order, not configuration semantics.
    # Keep the ORIGINAL frozen script bytes and compare its literal value and
    # unchanged coordinator template; never rebuild it with reordered repr.
    if frozen_config!=c or source!=REMOTE_SCRIPT.replace('CONFIG',repr(frozen_config),1):raise ValueError('frozen coordinator differs')
    receipt=ROOT/(name+'-coordinator-receipt.json')
    if receipt.exists():raise FileExistsError('preserve actual receipt; no rerun')
    tick=time.monotonic()
    try:
        x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=980)
        result=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-tick)
    except subprocess.TimeoutExpired as e:
        result=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-tick,error='SSH timeout; inspect bounded existing process, no automatic retry')
    result['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,result)
    print(json.dumps({k:result[k]for k in ('returncode','wall_seconds')}));raise SystemExit(result['returncode']if result['returncode']is not None else 124)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',choices=('cpu','train'));g.add_argument('--execute');p.add_argument('--anchor',type=int,choices=(0,1),default=1);a=p.parse_args()
    if a.prepare:prepare(a.prepare,a.anchor)
    else:execute(a.execute)
