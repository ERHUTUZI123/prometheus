"""Trusted local freeze / ROOT-only run of NEW stage4 VALID memory audit.

No Torch import/unpickling; no model execution during preparation. Original
checkpoint/core/wrapper/hook/data bytes are only read, checked and copied.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PARENT="parameter-memory-live-stage-n8-seed0-dynamic-stage4-v1"
NAME=PARENT+"-valid-memory-path-v2"
REMOTE="/workspace/prometheus-v2-m0-architecture"
SSH=["ssh","-o","BatchMode=yes","-p","24343","-i","/Users/xiaoyangliu/.ssh/runpod_prometheus","root@195.26.232.180","python3","-"]
RUNNER_SHA="85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0"
CHECKPOINT_SHA="c82acdb59be35002ef19c069d79cf43d143902be6d0c2eb2aa54eebd1c11387a"
HOOK_SHA="17bb5fdf6ce77d7b0ffdb9bfc6904988ef2c813b77ebf21b8cba690b80f963f4"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(1<<20),b""):h.update(part)
    return h.hexdigest()


def checked(path,expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>100*1024*1024 or sha(path)!=expected:
        raise ValueError("regular bounded OWN hash-checked file required: "+str(path))


def inspect_parent():
    learned=ROOT/"runs"/PARENT/"learn";result=json.loads((learned/"result.json").read_text())
    rows=[json.loads(line) for line in (learned/"records.jsonl").read_text().splitlines()]
    configs=[r for r in rows if r.get("kind")=="configuration"]
    if len(configs)!=1:raise ValueError("actual parent configuration missing/ambiguous")
    config=configs[0]
    if (result.get("status")!="DECLARED_FOURTH_STAGE_DEVELOPMENT_COMPLETED" or result.get("arm")!="dynamic"
            or result.get("new_training_updates")!=600 or result["resources"]["parameters"]!=28381
            or result["source_sha256"]!=config["source_sha256"]
            or [r["step"] for r in rows if r.get("kind")=="actual_update"]!=list(range(1,601))):
        raise ValueError("actual completed fixed dynamic fourth-phase parent required")
    cp=result["checkpoint"];path=learned/cp["file"]
    if cp["file"]!="dynamic-stage3-step600.pt" or cp["bytes"]!=613915 or cp["sha256"]!=CHECKPOINT_SHA:raise ValueError("fixed actual NN receipt differs")
    checked(path,CHECKPOINT_SHA)
    sources=result["source_sha256"];frozen=ROOT/"versions"/("code-"+PARENT)
    for leaf,expected in sources.items():checked(frozen/leaf,expected)
    valid=[r for r in config["input_receipts"] if r.get("stage")==3 and r.get("split")=="valid"]
    if len(valid)!=1 or valid[0]["rows"]!=512 or valid[0]["file"]!="stage3-valid.npz":raise ValueError("actual parent VALID lineage missing")
    data=ROOT/"runs"/"parameter-memory-live-stage-data-n8-seed0-v1"/"generated"/"data"/"stage3-valid.npz"
    checked(data,valid[0]["file_sha256"])
    return result,sources,frozen,valid[0],data


REMOTE_CODE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];parent=c['parent'];m=c['manifest']
code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists() or out.exists() or inputs.exists():raise FileExistsError('new diagnostic namespace only; no overwrite/retry')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
 return h.hexdigest()
runner=r/'remote_runner_gpu_authorized.py'
if sha(runner)!=c['runner_sha256']:raise ValueError('unchanged external GPU runner differs')
for leaf,expected in m['parent_source_sha256'].items():
 if sha(r/('code-'+parent)/leaf)!=expected:raise ValueError('actual remote owned parent source differs')
source=r/'outputs'/parent/'learn'/'dynamic-stage3-step600.pt'
valid=pathlib.Path('/opt/prometheus-'+parent+'-inputs')/'data'/'stage3-valid.npz'
if source.is_symlink() or not source.is_file() or source.stat().st_size!=613915 or sha(source)!=m['checkpoint_sha256']:raise ValueError('fixed actual NN differs')
if valid.is_symlink() or not valid.is_file() or sha(valid)!=m['VALID_sha256']:raise ValueError('fixed actual VALID differs')
code.mkdir();inputs.mkdir();(inputs/'data').mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=m['all_source_sha256'][leaf]:raise ValueError('frozen source transport differs')
 target=code/leaf;target.write_bytes(raw);target.chmod(0o444)
code.chmod(0o555)
shutil.copyfile(source,inputs/'source.pt');shutil.copyfile(valid,inputs/'data'/'stage3-valid.npz')
if sha(inputs/'source.pt')!=m['checkpoint_sha256'] or sha(inputs/'data'/'stage3-valid.npz')!=m['VALID_sha256']:raise ValueError('read-only input copy differs')
(inputs/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
for target in inputs.rglob('*'):target.chmod(0o555 if target.is_dir() else 0o444)
inputs.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'memory_live_stage_valid_audit.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--checkpoint',str(inputs/'source.pt'),'--frozen-source',str(code),'--manifest',str(inputs/'manifest.json'),'--data',str(inputs/'data'),'--out',str(out/'audit'),'--device','cuda:0','--max-seconds','850']
print(json.dumps(dict(event='frozen_stage4_VALID_memory_audit_launch',command=cmd,protocol_sha256=c['protocol_sha256'],resource_overlap_note=c['resource_overlap_note'])),flush=True)
began=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=930)
 receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-began)
except subprocess.TimeoutExpired as error:
 receipt=dict(returncode=None,error='coordinator timeout; inspect actual bounded child, no rerun inferred',stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
receipt.update(source_checkpoint_unchanged=sha(source)==m['checkpoint_sha256'],copied_checkpoint_unchanged=sha(inputs/'source.pt')==m['checkpoint_sha256'],protocol_sha256=c['protocol_sha256'])
if out.exists():
 with (out/'coordinator-exit.json').open('x') as dest:dest.write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''


def prepare():
    version=ROOT/"versions"/("code-"+NAME);protocol_path=ROOT/(NAME+"-protocol.json")
    if any(path.exists() for path in (version,protocol_path,ROOT/"runs"/NAME,ROOT/(NAME+"-coordinator-receipt.json"))):raise FileExistsError("new diagnostic namespace only")
    result,sources,parent,valid,data=inspect_parent();checked(ROOT/"audit_context_memory.py",HOOK_SHA)
    origins={leaf:parent/leaf for leaf in sources}
    origins.update({"owned_memory_read_hooks.py":ROOT/"audit_context_memory.py",
                    "memory_live_stage_valid_audit.py":ROOT/"memory_live_stage_valid_audit.py",
                    Path(__file__).name:Path(__file__)})
    hashes={}
    for leaf,path in origins.items():ast.parse(path.read_bytes(),filename=leaf);hashes[leaf]=sha(path)
    manifest=dict(parent=PARENT,checkpoint_sha256=CHECKPOINT_SHA,VALID_sha256=valid["file_sha256"],
                  VALID_array_sha256=valid["array_sha256"],parent_source_sha256=sources,all_source_sha256=hashes)
    protocol=dict(schema="stage4-frozen-VALID-direct-memory-diagnostic-v1",before_execution=True,name=NAME,parent=PARENT,
        manifest=manifest,source_origins={k:str(v) for k,v in origins.items()},checkpoint=result["checkpoint"],
        source_parent_resources=result["resources"],source_parent_costs=result["costs"],
        interventions=["full","zero_read","joint_slot_roll1","value_slot_roll1"],interface="dynamic-soft",VALID_rows=512,
        fixed_forward_examples=2048,TRAIN_or_TEST_files_opened=0,optimizer_updates=0,new_teacher_labels=0,
        source_positions="observed last2 token9 positions, terminal READ10 and MIXREAD15; analysis only, never NN annotations",
        coefficient="R_s,u,j=product_{s<t<u}(1-g_t*a_tj): memory-after-WRITE to memory-before-query",
        limits="DIRECT fixed-feature path coefficients; no totalJacobian/feedback/infinite-retention/acquisition causality/content-address-generalization claim",
        stored_evidence="all512 raw read/write arrays, g/a/attention, predictions, direct coefficients; first8 complete memory/state histories per intervention",
        costs="actualP/model/Adam/checkpoint/data/trace/storage/time and all4forward+extra-hook-read MAC proxies; unmeasured FLOPs unknown",
        budget=dict(outside_seconds=900,inside_seconds=850,address_space_GiB=512,gpu_fraction=.3,file_MiB=100),
        runner_sha256=RUNNER_SHA,formal_Primary=False,current_override="TODO3 contextual/state/memory priority; Monodromy optional; Online USER-PARK; sameV2/no resource expansion")
    version.mkdir()
    for leaf,path in origins.items():(version/leaf).write_bytes(path.read_bytes())
    with protocol_path.open("x") as dest:json.dump(protocol,dest,indent=2,sort_keys=True);dest.write("\n")
    with (version/"manifest.json").open("x") as dest:json.dump(manifest,dest,indent=2,sort_keys=True);dest.write("\n")
    print(json.dumps(dict(status="FROZEN_NOT_EXECUTED",name=NAME,protocol=str(protocol_path),source_sha256=hashes)))


def run(note):
    protocol_path=ROOT/(NAME+"-protocol.json");protocol=json.loads(protocol_path.read_text());manifest=protocol["manifest"]
    if protocol["name"]!=NAME or protocol["parent"]!=PARENT:raise ValueError("frozen identity differs")
    if (ROOT/(NAME+"-coordinator-receipt.json")).exists() or (ROOT/"runs"/NAME).exists():raise FileExistsError("no rerun/overwrite")
    result,sources,parent,valid,data=inspect_parent()
    if sources!=manifest["parent_source_sha256"] or valid["file_sha256"]!=manifest["VALID_sha256"]:raise ValueError("actual parent/data changed after freeze")
    version=ROOT/"versions"/("code-"+NAME);encoded={}
    for leaf,expected in manifest["all_source_sha256"].items():checked(version/leaf,expected);encoded[leaf]=base64.b64encode((version/leaf).read_bytes()).decode()
    if sha(Path(__file__))!=manifest["all_source_sha256"][Path(__file__).name]:raise ValueError("trusted coordinator changed after freeze")
    config=dict(name=NAME,parent=PARENT,remote=REMOTE,manifest=manifest,files=encoded,runner_sha256=RUNNER_SHA,
                protocol_sha256=sha(protocol_path),resource_overlap_note=note)
    remote=REMOTE_CODE.replace("CONFIG",repr(config),1);ast.parse(remote);began=time.monotonic()
    try:
        x=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=960)
        receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-began)
    except subprocess.TimeoutExpired as error:
        receipt=dict(returncode=None,error="SSH timeout; inspect actual process, no automatic rerun",stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
    with (ROOT/(NAME+"-coordinator-receipt.json")).open("x") as dest:json.dump(receipt,dest,indent=2);dest.write("\n")
    print(json.dumps(receipt));raise SystemExit(receipt["returncode"] if receipt["returncode"] is not None else 124)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("prepare","run"))
    p.add_argument("--resource-overlap-note",default="Root schedules actual GPU occupancy; timings include intrusive hooks and traces")
    args=p.parse_args();prepare() if args.action=="prepare" else run(args.resource_overlap_note)


if __name__=="__main__":main()
