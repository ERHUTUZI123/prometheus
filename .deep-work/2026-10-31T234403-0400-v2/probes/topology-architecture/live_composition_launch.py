"""Trusted SAMEV2 freeze/ROOT-only execution for initial-six closure diagnosis.

Preparation is stdlib/hash/read-only on all old evidence. Data generation is a
separate bounded CPU job, with oracle/truth confined there. Each frozen NN gets
a distinct read-only GPU evaluation namespace with only labels/raw observations,
exact OWN source/checkpoint and evaluator. No training or new model admission.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parent
REMOTE="/workspace/prometheus-v2-m0-architecture"
DATA_NAME="parameter-live-composition-data-v1"
ORIGINAL_DATA="parameter-memory-live-stage-data-n8-seed0-v1"
CPU_SHA="c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335"
GPU_SHA="85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0"
SSH=["ssh","-o","BatchMode=yes","-p","24343","-i","/Users/xiaoyangliu/.ssh/runpod_prometheus","root@195.26.232.180","python3","-"]
KEYS=("dynamic","static","gru","transformer","delta","credit")


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path,expected=None):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>100<<20:
        raise ValueError("bounded regular OWN file required: "+str(path))
    actual=dict(bytes=path.stat().st_size,sha256=sha(path))
    if expected is not None and actual!=expected:raise ValueError("OWN actual bytes/SHA differ: "+str(path))
    return actual


def write_json(path,value):
    with path.open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def policy_module():
    path=ROOT/"evaluate_live_composition.py"
    spec=importlib.util.spec_from_file_location("_owned_closure_allowlist_stdlib",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if set(module.OWNED)!=set(KEYS) or module.ALLOWLIST_PROTOCOL!="live-composition-initial-six-v1":
        raise ValueError("initial SIX v1 allowlist is immutable; future NN requires a new protocol version")
    return module


def parent_facts(key,module):
    spec=module.OWNED[key];run=spec["parent"];learn=ROOT/"runs"/run/"learn"
    result=json.loads((learn/"result.json").read_text())
    if (result.get("status")!="DECLARED_FOURTH_STAGE_DEVELOPMENT_COMPLETED" or result.get("new_training_updates")!=600
            or result.get("seed")!=0 or result.get("formal_Primary") is not False
            or result.get("arm")!=spec["arm"] or result["resources"]["parameters"]!=spec["P"]
            or result["source_sha256"]!=module.expected_sources(spec)):
        raise ValueError("exact OWN completed phase4 model/source required: "+key)
    cp=result["checkpoint"];leaf=spec["arm"]+"-stage3-step600.pt"
    if (cp["file"]!=leaf or cp["sha256"]!=spec["sha256"] or cp["bytes"]!=spec["bytes"]):
        raise ValueError("actual checkpoint outside immutable six-model allowlist")
    checked(learn/leaf,dict(bytes=spec["bytes"],sha256=spec["sha256"]))
    version=ROOT/"versions"/("code-"+run);original={}
    for name,digest in module.expected_sources(spec).items():
        checked(version/name);raw=(version/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError("OWN frozen source changed")
        ast.parse(raw,filename=name);original[name]=raw
    if {p.name for p in version.glob("*.py")}!=set(original):raise ValueError("unexpected original executable source")
    updates=[];configurations=[];restore_events=[]
    with (learn/"records.jsonl").open() as stream:
        for line in stream:
            if any(('"kind": "'+kind+'"') in line for kind in ("actual_update","configuration","actual_parent_restore_and_storage_growth")):
                row=json.loads(line)
                if row["kind"]=="actual_update":updates.append(row)
                elif row["kind"]=="configuration":configurations.append(row)
                else:restore_events.append(row)
    if ([r["step"] for r in updates]!=list(range(1,601)) or len(configurations)!=1 or len(restore_events)!=1
            or configurations[0]["source_sha256"]!=result["source_sha256"]):
        raise ValueError("actual committed600 update/source history differs")
    protocol=json.loads((ROOT/(run+"-protocol.json")).read_text())
    lineage=protocol["parent_lineage"]
    if protocol["source_sha256"]!=result["source_sha256"]:raise ValueError("actual protocol/model source differs")
    historical=lineage["historical_training_reports"]
    phase0=lineage["historical_resume_lineage"]["historical_checkpoint_costs"]
    paid=[dict(stage=0,scope="saved OWN phase0 checkpoint costs",costs=phase0)]
    paid += [dict(stage=r["stage"],scope="OWN parent phase"+str(r["stage"]),costs=r["costs"]) for r in historical]
    paid.append(dict(stage=3,scope="current completed phase4",costs=result["costs"]))
    if sorted(r["stage"] for r in paid)!=[0,1,2,3] or any(r["costs"]["SGD_updates"]!=600 for r in paid):
        raise ValueError("distinct four-phase training cost lineage missing")
    costs=dict(phases=paid,distinct_paid_updates=2400,
        total_training_seconds=sum(r["costs"]["training_wall_seconds"] for r in paid),
        total_training_forward_MAC_proxy=sum(r["costs"]["training_forward_MAC_proxy"] for r in paid),
        phase2_parent_run=result["parent_run"],phase2_parent_checkpoint_sha256=lineage["checkpoint_sha256"],
        actual_phase4_result_sha256=sha(learn/"result.json"),actual_phase4_records_sha256=sha(learn/"records.jsonl"),
        scope="per-model lineage; initial phases are shared between dynamic/delta/credit, never sum these as independent costs or replicates",
        exact_total_training_FLOPs=None,all_old_performance_excluded=True)
    return dict(spec=spec,original=original,checkpoint_leaf=leaf,cost_lineage=costs,
                parent_current_resources=result["resources"],parent_current_paid_costs=result["costs"])


REMOTE_SCRIPT=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];mode=c['mode'];code=r/('code-'+name);out=r/'outputs'/name
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists() or out.exists() or inputs.exists():raise FileExistsError('newnamespace only; no overwrite/automaticretry')
code.mkdir();inputs.mkdir()
for leaf,encoded in c['sources'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('frozen execution source transport differs')
 target=code/leaf;target.write_bytes(raw);target.chmod(0o444)
code.chmod(0o555)
if mode=='data':
 for leaf,encoded in c['input_bytes'].items():
  raw=base64.b64decode(encoded);expected=c['input_receipts'][leaf]
  if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('originalCPUinput transport differs')
  target=inputs/leaf;target.write_bytes(raw);target.chmod(0o444)
 runner=r/'remote_runner_cpu_v2.py'
 if hashlib.sha256(runner.read_bytes()).hexdigest()!=c['CPU_SHA']:raise ValueError('unchanged CPUrunner differs')
 cmd=[sys.executable,str(runner),'--script',str(code/'build_live_composition_data.py'),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--train',str(inputs/'original-train.npz'),'--truth',str(inputs/'original-truth.npz'),'--original-metadata',str(inputs/'original-metadata.json'),'--original-generator',str(code/'original_generator.py'),'--manifest',str(inputs/'manifest.json'),'--out',str(out/'generated'),'--max-seconds','60']
else:
 if any(leaf in c['sources'] for leaf in ('build_live_composition_data.py','original_generator.py','memory_live_stage_data.py')):raise ValueError('oracle may not enter NN code')
 parent=c['model_spec']['parent'];source=r/'outputs'/parent/'learn'/c['checkpoint_leaf']
 if not source.is_file() or source.is_symlink() or source.stat().st_size!=c['model_spec']['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest()!=c['model_spec']['sha256']:raise ValueError('actual OWN frozen NN differs')
 parentcode=r/('code-'+parent)
 for leaf,digest in c['original_source_sha256'].items():
  if hashlib.sha256((parentcode/leaf).read_bytes()).hexdigest()!=digest:raise ValueError('original NN execution source differs')
 shutil.copyfile(source,inputs/'source.pt')
 data_source=r/'outputs'/c['data_run']/'generated'/'data';(inputs/'data').mkdir()
 for leaf,expected in c['data_files'].items():
  src=data_source/leaf
  if not src.is_file() or src.is_symlink() or src.stat().st_size!=expected['bytes'] or hashlib.sha256(src.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('once-generated shared data differs')
  shutil.copyfile(src,inputs/'data'/leaf)
 raw=base64.b64decode(c['manifest_bytes'])
 if hashlib.sha256(raw).hexdigest()!=c['manifest_sha256']:raise ValueError('evaluator manifest differs')
 (inputs/'manifest.json').write_bytes(raw)
 runner=r/'remote_runner_gpu_authorized.py'
 if hashlib.sha256(runner.read_bytes()).hexdigest()!=c['GPU_SHA']:raise ValueError('unchanged GPUrunner differs')
 cmd=[sys.executable,str(runner),'--script',str(code/'evaluate_live_composition.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--checkpoint',str(inputs/'source.pt'),'--frozen-source',str(code),'--data',str(inputs/'data'),'--manifest',str(inputs/'manifest.json'),'--out',str(out/'evaluate'),'--model',c['model_key'],'--device','cuda:0','--max-seconds','850']
for target in inputs.rglob('*'):target.chmod(0o555 if target.is_dir() else 0o444)
inputs.chmod(0o555)
print(json.dumps({'event':'frozen_live_composition_launch','command':cmd,'protocol':c['protocol']}),flush=True)
began=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=150 if mode=='data' else 930)
 receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-began)
except subprocess.TimeoutExpired as error:
 receipt=dict(returncode=None,error='coordinatorwaittimeout; inspect boundedrunner/noautomaticretry',stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
if mode!='data':receipt['original_checkpoint_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==c['model_spec']['sha256']
if out.exists():
 with (out/'coordinator-exit.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''


def commit(name,sources,protocol,config,input_snapshots=None):
    version=ROOT/"versions"/("code-"+name)
    for path in (version,ROOT/"runs"/name,ROOT/(name+"-protocol.json"),ROOT/(name+"-coordinator-receipt.json")):
        if path.exists():raise FileExistsError("new composition namespace required: "+str(path))
    hashes={}
    for leaf,raw in sources.items():ast.parse(raw,filename=leaf);hashes[leaf]=hashlib.sha256(raw).hexdigest()
    protocol=dict(protocol,source_sha256=hashes)
    config=dict(config,name=name,remote=REMOTE,sources={k:base64.b64encode(v).decode() for k,v in sources.items()},
                source_sha256=hashes,CPU_SHA=CPU_SHA,GPU_SHA=GPU_SHA,protocol=protocol)
    config=json.loads(json.dumps(config,sort_keys=True));remote=REMOTE_SCRIPT.replace("CONFIG",repr(config),1);ast.parse(remote)
    version.mkdir()
    for leaf,raw in sources.items():
        with (version/leaf).open("xb") as dest:dest.write(raw)
    if input_snapshots:
        snapshot=version/"input-bytes";snapshot.mkdir()
        for leaf,raw in input_snapshots.items():
            with (snapshot/leaf).open("xb") as dest:dest.write(raw)
    write_json(version/"launch-config.json",config)
    with (version/"remote-launch.py.source").open("x") as dest:dest.write(remote)
    write_json(ROOT/(name+"-protocol.json"),protocol)
    return dict(status="FROZEN_FOR_ROOT_ONLY_NO_EXECUTION",name=name,source_sha256=hashes,
                exact_remote_script_sha256=sha(version/"remote-launch.py.source"),protocol=str(ROOT/(name+"-protocol.json")))


def prepare_data():
    module=policy_module()
    facts={key:parent_facts(key,module) for key in KEYS}
    original=ROOT/"runs"/ORIGINAL_DATA/"generated"
    meta=original/"data/metadata.json";metadata=json.loads(meta.read_text())
    generator=ROOT/"versions"/("code-"+ORIGINAL_DATA)/"memory_live_stage_data.py"
    if sha(generator)!=metadata["source_sha256"]:raise ValueError("exact original generator changed")
    inputs={"original-train.npz":(original/"data"/metadata["files"]["train"]["file"]).read_bytes(),
            "original-truth.npz":(original/"counterfactual-truth.npz").read_bytes(),"original-metadata.json":meta.read_bytes()}
    manifest=dict(schema="live-composition-original-inputs-v1",original_metadata_sha256=sha(meta),
        initial_model_allowlist_protocol=module.ALLOWLIST_PROTOCOL,initial_six_model_allowlist=module.OWNED)
    inputs["manifest.json"]=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    for leaf,expected in (("original-train.npz",metadata["files"]["train"]["sha256"]),
                          ("original-truth.npz",metadata["counterfactual_truth_sha256"])):
        if hashlib.sha256(inputs[leaf]).hexdigest()!=expected:raise ValueError("exact original TRAIN/truth bytes changed")
    sources={"build_live_composition_data.py":(ROOT/"build_live_composition_data.py").read_bytes(),
             "original_generator.py":generator.read_bytes()}
    protocol=dict(before_execution=True,name=DATA_NAME,scope="NEW postfreeze DEVELOPMENT closure diagnostic; no model query, no oldtask/result edit",
        selection="FIRST complete contiguous256 originalTRAIN grid with max length<=46, length-only before labels/oracle/NN",
        examples=2048,base_conditions=["MIX15","last15 replaced by READ10"],suffixes=[[],[1,3],[2,3],[14]],
        initial_model_allowlist_protocol=module.ALLOWLIST_PROTOCOL,initial_six_model_allowlist=module.OWNED,
        owner_provenance_checked={k:dict(checkpoint_sha256=facts[k]["spec"]["sha256"],source_sha256=module.expected_sources(facts[k]["spec"])) for k in KEYS},
        oracle_isolation="only this CPU generator contains original oracle/truth; evaluators receive data/metadata only",
        outside_seconds=120,inside_seconds=60,address_space_GiB=64,SGD_updates=0,formal_Primary=False)
    receipts={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in inputs.items()}
    return commit(DATA_NAME,sources,protocol,dict(mode="data",input_bytes={k:base64.b64encode(v).decode() for k,v in inputs.items()},
                                                  input_receipts=receipts),inputs)


def prepare_eval(key):
    module=policy_module();facts=parent_facts(key,module);spec=facts["spec"]
    data_root=ROOT/"runs"/DATA_NAME/"generated"/"data"
    metadata=json.loads((data_root/"metadata.json").read_text())
    if (metadata.get("schema")!="live-composition-closure-data-v1" or metadata["rows"]!=2048
            or metadata["initial_model_allowlist_protocol"]!=module.ALLOWLIST_PROTOCOL
            or metadata["initial_six_model_allowlist"]!=module.OWNED):
        raise ValueError("once-generated initial-six closure data must be complete/synchronized first")
    receipts={leaf:checked(data_root/leaf) for leaf in ("closure.npz","metadata.json")}
    if receipts["closure.npz"]["sha256"]!=metadata["file"]["sha256"]:raise ValueError("shared data/metadata SHA differs")
    generator_version=ROOT/"versions"/("code-"+DATA_NAME)
    if sha(generator_version/"build_live_composition_data.py")!=metadata["source_sha256"]["generator"]:
        raise ValueError("actual data generator source differs from frozen receipt")
    data_receipt=ROOT/(DATA_NAME+"-coordinator-receipt.json")
    coordinator=json.loads(data_receipt.read_text())
    if coordinator.get("returncode")!=0 or coordinator.get("exact_remote_script_sha256")!=sha(generator_version/"remote-launch.py.source"):
        raise ValueError("actual once-generated CPU job receipt differs")
    remote_generation_receipt=ROOT/"runs"/DATA_NAME/"coordinator-exit.json"
    remote_generation=json.loads(remote_generation_receipt.read_text())
    if remote_generation.get("returncode")!=0:raise ValueError("actual bounded CPU generation did not complete")
    measured_generation=dict(algorithm_generation_wall_seconds=metadata["generation_wall_seconds"],
        complete_CPU_runner_wall_seconds=remote_generation["wall_seconds"],
        cpu_peak_RSS_bytes=metadata["cpu_peak_RSS_bytes"],complete_SSH_coordinator_wall_seconds=coordinator["wall_seconds"],
        coordinator_receipt_sha256=sha(data_receipt),CPU_coordinator_receipt_sha256=sha(remote_generation_receipt),
        scope="shared generation paid ONCE; algorithm/import/runner/SSH scopes nested, do not add or repeat per NN")
    name="parameter-live-composition-"+key+"-v1"
    evaluator=(ROOT/"evaluate_live_composition.py").read_bytes();sources=dict(facts["original"])
    sources["evaluate_live_composition.py"]=evaluator
    manifest=dict(schema="live-composition-evaluator-inputs-v1",initial_model_allowlist_protocol=module.ALLOWLIST_PROTOCOL,
        initial_six_model_allowlist=module.OWNED,model_key=key,model_spec=spec,data_run=DATA_NAME,
        data_files=receipts,evaluator_source_sha256=hashlib.sha256(evaluator).hexdigest(),
        model_training_cost_lineage=facts["cost_lineage"],current_parent_resources=facts["parent_current_resources"])
    manifest["data_generation_measured_resources"]=measured_generation
    raw=(json.dumps(manifest,indent=2,sort_keys=True)+"\n").encode()
    protocol=dict(before_execution=True,name=name,scope="NEW frozen SOFT semantic closure DEVELOPMENT; not oldTEST replacement, fitting or general-computation proof",
        initial_model_allowlist_protocol=module.ALLOWLIST_PROTOCOL,initial_six_model_allowlist=module.OWNED,
        model_key=key,actual_model_spec=spec,original_source_sha256=module.expected_sources(spec),
        data_run=DATA_NAME,data_files=receipts,examples=2048,SGD_updates=0,modes=["soft"],
        metrics=["base accuracy","suffix accuracy","jointbase&suffixcorrect/all256","prediction F consistency","suffix accuracy given ownbasecorrect with denominator/sourceindices"],
        metric_cautions="ownbasecorrect subsets differ; emptyconsistency tautological, CLEARconsistency=CLEARaccuracy; MIX/READ always separate",
        label_information="restore/hash/policy first; labels then read by evaluator; only initial/rawtokens/lengths passed to NN",
        oracle_source_excluded=True,model_training_cost_lineage=facts["cost_lineage"],
        outside_seconds=900,inside_seconds=850,gpu_fraction=.3,address_space_GiB=512,formal_Primary=False)
    return commit(name,sources,protocol,dict(mode="eval",model_key=key,model_spec=spec,
        original_source_sha256=module.expected_sources(spec),checkpoint_leaf=facts["checkpoint_leaf"],
        data_run=DATA_NAME,data_files=receipts,manifest_bytes=base64.b64encode(raw).decode(),
        manifest_sha256=hashlib.sha256(raw).hexdigest()),{"manifest.json":raw})


def execute(name):
    if name!=DATA_NAME and name not in {"parameter-live-composition-"+k+"-v1" for k in KEYS}:
        raise ValueError("only initial-six/data v1 frozen namespaces admitted")
    version=ROOT/"versions"/("code-"+name);config=json.loads((version/"launch-config.json").read_text())
    if config["name"]!=name:raise ValueError("frozen name differs")
    for leaf,digest in config["source_sha256"].items():
        if sha(version/leaf)!=digest:raise ValueError("frozen source changed")
    remote=REMOTE_SCRIPT.replace("CONFIG",repr(config),1);ast.parse(remote)
    if remote!=(version/"remote-launch.py.source").read_text():raise ValueError("coordinator changed after freeze")
    receipt_path=ROOT/(name+"-coordinator-receipt.json")
    if receipt_path.exists() or (ROOT/"runs"/name).exists():raise FileExistsError("actual run/evidence exists; no automatic overwrite/retry")
    began=time.monotonic()
    try:
        x=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=180 if config["mode"]=="data" else 960)
        receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-began,
                     exact_remote_script_sha256=sha(version/"remote-launch.py.source"))
    except subprocess.TimeoutExpired as error:
        receipt=dict(returncode=None,error="SSHwaittimeout; inspect actual boundedrunner/noautomaticretry",
                     stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
    write_json(receipt_path,receipt);print(json.dumps(receipt));raise SystemExit(receipt["returncode"] if receipt["returncode"] is not None else 124)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare-data",action="store_true");g.add_argument("--prepare-eval",choices=KEYS);g.add_argument("--execute")
    a=p.parse_args()
    if a.prepare_data:print(json.dumps(prepare_data()))
    elif a.prepare_eval:print(json.dumps(prepare_eval(a.prepare_eval)))
    else:execute(a.execute)


if __name__=="__main__":main()
