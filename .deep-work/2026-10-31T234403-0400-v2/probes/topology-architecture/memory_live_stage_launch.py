"""Trusted SAMEV2 coordinator; ROOT alone invokes SSH/CPU tests/training.

No import-time execution and no Torch/unpickling in this trusted coordinator.
--prepare freezes concrete code/protocol without SSH; --execute NAME executes
that exact frozen package. Original code, inputs, checkpoints/results remain
immutable. External resource/isolation settings are unchanged.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REMOTE="/workspace/prometheus-v2-m0-architecture"
SSH=["ssh","-o","BatchMode=yes","-p","24343","-i","/Users/xiaoyangliu/.ssh/runpod_prometheus","root@195.26.232.180","python3","-"]
DATA_NAME="parameter-memory-live-stage-data-n8-seed0-v1"
OLD_DATA_NAME="parameter-context-data-n8-seed0-v1"
PARENTS={
 "dynamic":"parameter-context-n8-seed0-dynamic-stage3-v1-soft_only-continuous-resume-replay128x16",
 "static":"parameter-context-n8-seed0-static-stage3-v1-soft_only-continuous-resume-replay128x16",
 "gru":"parameter-context-n8-seed0-gru-stage3-v1-width44-resume-replay128x16",
 "transformer":"parameter-context-n8-seed0-transformer-stage3-v1-width32-resume-replay128x16"}
CORE_SHA="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
CPU_RUNNER_SHA="c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335"
GPU_RUNNER_SHA="85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0"
MAX_FILE=100*1024*1024


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}",value):raise ValueError("plain bounded run name required")
    return value


def checked(path,expected=None):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_FILE:raise ValueError("regular owned bounded file required: "+str(path))
    digest=sha(path)
    if expected is not None and digest!=expected:raise ValueError("owned actual SHA differs: "+str(path))
    return digest


def parent_manifest(arm):
    parent=PARENTS[arm];learn=ROOT/"runs"/parent/"learn";frozen=ROOT/"versions"/("code-"+parent)
    result=json.loads((learn/"result.json").read_text())
    if result.get("status")!="DECLARED_DEVELOPMENT_COMPLETED" or result.get("arm")!=arm or result.get("seed")!=0:
        raise ValueError("OWN completed seed0 parent result required")
    rows=[json.loads(line) for line in (learn/"records.jsonl").read_text().splitlines()]
    configurations=[r for r in rows if r.get("kind")=="configuration"]
    if len(configurations)!=1:raise ValueError("actual parent configuration missing/ambiguous")
    config=configurations[0];policy=config["args"]
    if config["source_sha256"]!=result["source_sha256"]:raise ValueError("parent actual start/end source hashes differ")
    for leaf,expected in result["source_sha256"].items():checked(frozen/leaf,expected)
    if policy["states"]!=8 or policy["seed"]!=0 or policy["steps"]!=600 or policy["stages"]!=3:
        raise ValueError("frozen actual N8/seed0/three-phase parent required")
    if policy.get("replay_capacity")!=128 or policy.get("replay_per_batch")!=16:
        raise ValueError("actual Replay128x16 parent required")
    if arm in ("dynamic","static"):
        if policy["block_state"]!="continuous" or policy["block_training"]!="soft_only" or checked(frozen/"continuous_primitive_block.py")!=CORE_SHA:
            raise ValueError("original actual b8a51 continuous soft-only parent required")
    elif policy["baseline_width"]!={"gru":44,"transformer":32}[arm]:raise ValueError("actual matched baseline width differs")
    reports={r["stage"]:r for r in result["training_reports"]}
    if set(reports)!={1,2}:raise ValueError("actual parent phase1/2 reports required")
    for stage in (1,2):
        updates=[r["step"] for r in rows if r.get("kind")=="actual_update" and r.get("stage")==stage]
        if updates!=list(range(1,601)) or reports[stage]["costs"]["SGD_updates"]!=600:
            raise ValueError("actual complete 600-update phase history required")
    cp=reports[2]["checkpoint"];checkpoint=learn/cp["file"]
    checked(checkpoint,cp["sha256"])
    if checkpoint.stat().st_size!=cp["bytes"] or cp["file"]!=arm+"-stage2-step600.pt":raise ValueError("actual phase2 receipt differs")
    original=result["resume_lineage"]
    phase0=original["parent_run"];phase0learn=ROOT/"runs"/phase0/"learn"
    history={}
    for stage in range(3):
        origin=(phase0learn/"stage0-common-indices.npz") if stage==0 else learn/f"stage{stage}-common-indices.npz"
        expected=(original["replay_intervention"]["bootstrap"]["parent_indices_file_sha256"] if stage==0 else
                  next(r["indices_sha256"] for r in rows if r.get("kind")=="stage_begin" and r.get("stage")==stage))
        checked(origin,expected)
        history[str(stage)]=dict(file=f"history-stage{stage}-indices.npz",sha256=expected,bytes=origin.stat().st_size,
                                remote_relative=str(origin.relative_to(ROOT/"runs")),local_source=str(origin))
    olddir=ROOT/"runs"/OLD_DATA_NAME/"data";oldmeta=json.loads((olddir/"metadata.json").read_text())
    if {k:oldmeta[k] for k in ("n_states","n_tokens","max_length")}!=dict(n_states=8,n_tokens=15,max_length=48):raise ValueError("original old metadata differs")
    oldfiles={}
    for stage in range(3):
        for split in ("train","valid","test_iid","test_long","test_unseen"):
            leaf=oldmeta["stages"][stage]["files"][split]
            expected=oldmeta["stages"][stage]["hashes"][split];checked(olddir/leaf,expected)
            oldfiles[f"{stage}:{split}"]=dict(file=leaf,sha256=expected)
    newdir=ROOT/"runs"/DATA_NAME/"generated"/"data"
    newmeta=json.loads((newdir/"metadata.json").read_text())
    generator=ROOT/"versions"/("code-"+DATA_NAME)/"memory_live_stage_data.py"
    checked(generator,newmeta["source_sha256"])
    if newmeta["labels_created"]!=6144 or newmeta["counts"]!=dict(train=4096,valid=512,test_iid=512,test_long=512,test_unseen=512):
        raise ValueError("prospective balanced fourth-task counts differ")
    for entry in newmeta["files"].values():checked(newdir/entry["file"],entry["sha256"])
    truth=ROOT/"runs"/DATA_NAME/"generated"/"counterfactual-truth.npz"
    checked(truth,newmeta["counterfactual_truth_sha256"])
    return dict(schema="memory-live-stage-owned-lineage-v1",arm=arm,parent_run=parent,
        parent_actual_configuration=policy,parent_actual_source_sha256=result["source_sha256"],
        parent_frozen_source_directory=str(frozen),checkpoint_leaf=cp["file"],checkpoint_sha256=cp["sha256"],checkpoint_bytes=cp["bytes"],
        replay_payload_sha256=reports[2]["replay_boundary"]["snapshot"]["buffer_payload_sha256"],history_files=history,
        old_metadata_sha256=sha(olddir/"metadata.json"),old_files=oldfiles,
        data_run=DATA_NAME,data_metadata_sha256=sha(newdir/"metadata.json"),data_source_sha256=newmeta["source_sha256"],
        counterfactual_truth_sha256=newmeta["counterfactual_truth_sha256"],truth_excluded_from_training_inputs=True,
        historical_training_reports=result["training_reports"],historical_resume_lineage=original,
        parent_result_sha256=sha(learn/"result.json"),parent_records_sha256=sha(learn/"records.jsonl"),
        previous_actual_paid_updates=1800,new_actual_updates=600,old_training_repeated=False)


REMOTE_SCRIPT=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];mode=c['mode']
code_dir=r/('code-'+name);out_dir=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code_dir.exists() or out_dir.exists() or (mode!='data' and inputs.exists()):raise FileExistsError('new namespace only; no overwrite/retry')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
code_dir.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('frozen source transport hash differs')
 target=code_dir/leaf;target.write_bytes(raw);target.chmod(0o444)
code_dir.chmod(0o555)
if mode=='data':
 runner=r/'remote_runner_cpu_v2.py'
 if sha(runner)!=c['CPU_runner_sha256']:raise ValueError('unchanged CPU runner hash differs')
 cmd=[sys.executable,str(runner),'--script',str(code_dir/'memory_live_stage_data.py'),'--out',str(out_dir),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out_dir/'generated'),'--seed','0','--max-seconds','60']
else:
 m=c['manifest'];parent=m['parent_run'];parentcode=r/('code-'+parent);parentlearn=r/'outputs'/parent/'learn'
 for leaf,expected in m['parent_actual_source_sha256'].items():
  if sha(parentcode/leaf)!=expected:raise ValueError('actual remote parent source differs: '+leaf)
 original=parentlearn/m['checkpoint_leaf']
 if original.is_symlink() or not original.is_file() or original.stat().st_size!=m['checkpoint_bytes'] or sha(original)!=m['checkpoint_sha256']:raise ValueError('actual remote OWN phase2 checkpoint differs')
 oldsource=pathlib.Path('/opt/prometheus-'+parent+'-inputs')/'data'
 newsource=r/'outputs'/m['data_run']/'generated'/'data'
 if sha(oldsource/'metadata.json')!=m['old_metadata_sha256'] or sha(newsource/'metadata.json')!=m['data_metadata_sha256']:raise ValueError('actual raw metadata differs')
 for entry in m['old_files'].values():
  if sha(oldsource/entry['file'])!=entry['sha256']:raise ValueError('actual old raw bytes differ')
 newmeta=json.loads((newsource/'metadata.json').read_text())
 for entry in newmeta['files'].values():
  if sha(newsource/entry['file'])!=entry['sha256']:raise ValueError('actual new raw bytes differ')
 inputs.mkdir();shutil.copytree(oldsource,inputs/'old-data');shutil.copytree(newsource,inputs/'data')
 # Root copies/hashes bytes ONLY. Unpickling is confined to the training UID.
 shutil.copyfile(original,inputs/'resume.pt')
 if sha(inputs/'resume.pt')!=m['checkpoint_sha256']:raise ValueError('resume copy hash differs')
 for proof in m['history_files'].values():
  source=r/'outputs'/proof['remote_relative']
  if source.stat().st_size!=proof['bytes'] or sha(source)!=proof['sha256']:raise ValueError('actual committed index proof differs')
  shutil.copyfile(source,inputs/proof['file'])
 (inputs/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
 for target in inputs.rglob('*'):target.chmod(0o555 if target.is_dir() else 0o444)
 inputs.chmod(0o555)
 runner=r/('remote_runner_cpu_v2.py' if mode=='selftest' else 'remote_runner_gpu_authorized.py')
 expected=c['CPU_runner_sha256'] if mode=='selftest' else c['GPU_runner_sha256']
 if sha(runner)!=expected:raise ValueError('unchanged external runner hash differs')
 cmd=[sys.executable,str(runner),'--script',str(code_dir/'memory_live_stage_train.py'),'--out',str(out_dir),'--timeout','150' if mode=='selftest' else '900','--address-space-gib','64' if mode=='selftest' else '512','--file-limit-mib','100']
 cmd+=['--cpu-only'] if mode=='selftest' else ['--gpu-fraction','.3','--cuda-init-before-seccomp']
 cmd+=['--','--data',str(inputs/'data'),'--old-data',str(inputs/'old-data'),'--resume',str(inputs/'resume.pt'),'--manifest',str(inputs/'manifest.json'),'--out',str(out_dir/'learn'),'--device','cpu' if mode=='selftest' else 'cuda:0','--max-seconds','120' if mode=='selftest' else '850']
 if mode=='selftest':cmd+=['--self-test']
print(json.dumps({'event':'frozen_memory_live_stage_launch','command':cmd,'protocol':c['protocol']}),flush=True)
began=time.monotonic()
try:
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=180 if mode in ('data','selftest') else 930)
 receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-began)
 print(result.stdout);print(result.stderr,file=sys.stderr)
except subprocess.TimeoutExpired as error:
 receipt=dict(returncode=None,status='coordinator_timeout; inspect actual state, no rerun inferred',stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
if out_dir.exists():
 with (out_dir/'coordinator-exit.json').open('x') as stream:stream.write(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''


def prepare(mode,arm):
    runname=DATA_NAME if mode=="data" else f"parameter-memory-live-stage-n8-seed0-{arm}-"+("growth-selftest-v1" if mode=="selftest" else "stage4-v1")
    name(runname);version=ROOT/"versions"/("code-"+runname)
    for path in (version,ROOT/"runs"/runname,ROOT/(runname+"-protocol.json"),ROOT/(runname+"-coordinator-receipt.json")):
        if path.exists():raise FileExistsError("new namespace required: "+str(path))
    manifest=None;origins={"memory_live_stage_data.py":ROOT/"memory_live_stage_data.py"}
    if mode!="data":
        manifest=parent_manifest(arm);parent=Path(manifest["parent_frozen_source_directory"])
        origins={"memory_live_stage_train.py":ROOT/"memory_live_stage_train.py",
                 "owned_parent_driver.py":parent/"train_contextual_primitives.py",
                 "context_replay128.py":parent/"context_replay128.py"}
        if arm in ("dynamic","static"):origins["continuous_primitive_block.py"]=parent/"continuous_primitive_block.py"
    encoded={};hashes={}
    for leaf,origin in origins.items():
        raw=origin.read_bytes();ast.parse(raw,filename=leaf);hashes[leaf]=hashlib.sha256(raw).hexdigest();encoded[leaf]=base64.b64encode(raw).decode()
    if manifest:
        if hashes["owned_parent_driver.py"]!=manifest["parent_actual_source_sha256"]["train_contextual_primitives.py"]:
            raise ValueError("imported OWN parent driver differs from executed class bytes")
        manifest["execution_sources"]=hashes
    protocol=dict(before_execution=True,name=runname,mode=mode,arm=arm if mode!="data" else None,
        scope="SAMEV2 new fourth-task development only; no Primary or old-measurement repair claim",
        source_sha256=hashes,source_origins={k:str(v) for k,v in origins.items()},
        source_class_policy="exact OWN frozen parent driver/core/replay bytes; new wrapper only",
        new_token="MIXREAD=15, learned random append only; correct rule excluded from learner code/inputs",
        prospectively_fixed=dict(TRAIN=4096,VALID=512,TEST_IID=512,TEST_LONG=512,TEST_UNSEEN=512,
            n_states=8,n_tokens=16,public_max_length=128,train_max_length=48,long_length=[64,112],
            stage=3,updates=600,batch=64,current_per_batch=48,old_per_batch=16,replay_rows=128,
            no_extra_updates=True,no_test_selection=True,query_write_orders_and_targets_balanced=True,
            values_query_write_order_grid="8x8x2x2=256 per template"),
        causal_scope="variable-write-order two-key binding: both stored values independently change terminal label at exact same query suffix/write-order/distractors; held-out distractor composition; not arbitrary-key addressing",
        controls="ordinary TRAIN majority + raw last4-token lookup; expected balanced chance1/8; no teacher parsing",
        parent_lineage=manifest,budget=dict(outside_seconds=120 if mode=="data" else 150 if mode=="selftest" else 900,
            inside_seconds=60 if mode=="data" else 120 if mode=="selftest" else 850,
            address_space_GiB=64 if mode in ("data","selftest") else 512,gpu_fraction=.3 if mode=="train" else None,file_MiB=100),
        costs="parameters AND restore/growth/data/Adam/model/buffer/replay/training/eval/inference/checkpoint paid; exact unmeasured FLOPs unknown",
        ongoing_constraints="current human objective; TODO3 priority; Monodromy optional; Online USER-PARK; no new workflow/resources/permissions",
        historical_boundary="all old datasets/metrics/source/checkpoints retained unchanged; old unseen overwritten READ does not establish two-address composition")
    config=dict(name=runname,mode=mode,remote=REMOTE,files=encoded,source_sha256=hashes,manifest=manifest,
                CPU_runner_sha256=CPU_RUNNER_SHA,GPU_runner_sha256=GPU_RUNNER_SHA,protocol=protocol)
    # Canonical key insertion order makes the saved/loaded repr exactly equal.
    config=json.loads(json.dumps(config,sort_keys=True))
    remote=REMOTE_SCRIPT.replace("CONFIG",repr(config),1);ast.parse(remote)
    version.mkdir()
    for leaf,value in encoded.items():(version/leaf).write_bytes(base64.b64decode(value))
    (version/"launch-config.json").write_text(json.dumps(config,sort_keys=True,indent=2)+"\n")
    (version/"remote-launch.py.source").write_text(remote)
    (ROOT/(runname+"-protocol.json")).write_text(json.dumps(protocol,sort_keys=True,indent=2)+"\n")
    print(json.dumps(dict(status="FROZEN_FOR_ROOT_REVIEW_NO_SSH",name=runname,protocol=str(ROOT/(runname+"-protocol.json")),
                          exact_remote_script_sha256=sha(version/"remote-launch.py.source"))))


def execute(runname):
    name(runname);version=ROOT/"versions"/("code-"+runname);config=json.loads((version/"launch-config.json").read_text())
    if config["name"]!=runname:raise ValueError("frozen identity differs")
    for leaf,expected in config["source_sha256"].items():checked(version/leaf,expected)
    remote=REMOTE_SCRIPT.replace("CONFIG",repr(config),1);ast.parse(remote)
    if remote!=(version/"remote-launch.py.source").read_text():raise ValueError("coordinator changed after prospective freeze")
    receipt_path=ROOT/(runname+"-coordinator-receipt.json")
    if receipt_path.exists():raise FileExistsError("no execution receipt overwrite; inspect failed run")
    began=time.monotonic()
    try:
        result=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=210 if config["mode"] in ("data","selftest") else 960)
        receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-began,
                     exact_remote_script_sha256=sha(version/"remote-launch.py.source"))
    except subprocess.TimeoutExpired as error:
        receipt=dict(returncode=None,status="SSH timeout; inspect actual process, no rerun inferred",stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
    with receipt_path.open("x") as stream:stream.write(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt));raise SystemExit(receipt["returncode"] if receipt["returncode"] is not None else 124)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",choices=("data","selftest","train"));g.add_argument("--execute")
    p.add_argument("--arm",choices=tuple(PARENTS),default="dynamic");a=p.parse_args()
    if a.prepare:prepare(a.prepare,a.arm)
    else:execute(a.execute)


if __name__=="__main__":main()
