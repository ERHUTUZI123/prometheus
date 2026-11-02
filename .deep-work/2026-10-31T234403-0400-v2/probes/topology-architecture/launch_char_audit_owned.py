"""Trusted launcher for OWN frozen char inference; not experimental isolation.

Root executes this coordinator and schedules GPU audits serially. The scientific
child remains under the unchanged authorized GPU runner. No training, source
substitution, parameter selection, dependency installation or permission growth.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
REMOTE = "/workspace/prometheus-v2-m0-architecture"
SSH = ["ssh", "-o", "BatchMode=yes", "-p", "24343", "-i",
       "/Users/xiaoyangliu/.ssh/runpod_prometheus", "root@195.26.232.180", "python3", "-"]
GPU_RUNNER_SHA = "85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0"
CONTINUOUS_SHA = "b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True, help="owned synchronized training run name, or its runs/ directory")
    p.add_argument("--revision", default="v1")
    p.add_argument("--resource-overlap-note", default="Root schedules this audit serially; otherGPUoccupancy must be independently recorded, not assumed")
    args = p.parse_args()
    parent = Path(args.parent).name
    if not re.fullmatch(r"parameter-char-[A-Za-z0-9_-]{1,160}", parent) or not re.fullmatch(r"v[1-9][0-9]*", args.revision):
        p.error("safe owned character parent and vN revision required")
    supplied = Path(args.parent)
    actual = ROOT / "runs" / parent
    if len(supplied.parts) > 1 and supplied.resolve() != actual.resolve():
        p.error("parent path must be the existing owned runs/ directory")
    if not (actual / "sync-hashes.json").is_file(): raise ValueError("synchronized parent hash manifest required")
    summary = json.loads((actual / "learn" / "summary.json").read_text())
    raw = actual / "learn" / "records.jsonl"
    with raw.open() as stream:
        configuration = next(json.loads(line) for line in stream if json.loads(line).get("kind") == "configuration")
    if summary.get("completed_updates") != 300 or summary.get("formal_Primary") is not False:
        raise ValueError("owned actual300-update development parent required")
    arm = summary["arm"]
    if arm not in ("block", "gru", "transformer"): raise ValueError("unsupported actual parent architecture")
    if arm == "block" and configuration["args"].get("block_state") != "continuous":
        raise ValueError("categorical one-hot nativePPL is outside this audit")
    expected_checkpoint = summary["checkpoints"][-1]
    leaf = expected_checkpoint["file"]
    if Path(leaf).name != leaf or leaf != f"{arm}-frozen-final.pt": raise ValueError("actual frozen final checkpoint required")
    checkpoint = actual / "learn" / leaf
    if (checkpoint.stat().st_size != expected_checkpoint["bytes"] or sha(checkpoint) != expected_checkpoint["sha256"]):
        raise ValueError("parent checkpoint bytes differ from actual saved receipt")
    old_version = ROOT / "versions" / ("code-" + parent)
    old_manifest = json.loads((actual / "sync-hashes.json").read_text())
    old_code_hashes = old_manifest.get("code-" + parent)
    if not isinstance(old_code_hashes, dict): raise ValueError("actual parent original-code manifest missing")
    original = {}
    for source in sorted(old_version.glob("*.py")):
        if source.is_symlink(): raise ValueError("original code must be owned regular files")
        data = source.read_bytes(); expected = old_code_hashes.get(source.name)
        if expected is None or len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"original frozen code mismatch: {source.name}")
        ast.parse(data, filename=source.name); original[source.name] = data
    if not original: raise ValueError("no owned original Python source")
    for source, expected in configuration["source_sha256"].items():
        if source not in original or hashlib.sha256(original[source]).hexdigest() != expected:
            raise ValueError(f"actual training source absent or changed: {source}")
    if arm == "block" and hashlib.sha256(original.get("continuous_primitive_block.py", b"")).hexdigest() != CONTINUOUS_SHA:
        raise ValueError("exact originalb8a51 continuous source required")
    data_parent = ROOT / "runs" / "parameter-char-data-v1" / "data"
    metadata_path = data_parent / "metadata.json"
    if sha(metadata_path) != configuration["metadata_sha256"]: raise ValueError("exact original prepared metadata differs")
    data_manifest = json.loads((data_parent.parent / "sync-hashes.json").read_text())["outputs/parameter-char-data-v1"]
    expected_data = {}
    for source in sorted(data_parent.iterdir()):
        if source.is_symlink() or not source.is_file(): raise ValueError("owned dataset must contain regular files only")
        expected = data_manifest.get("data/" + source.name)
        if expected is None or source.stat().st_size != expected["bytes"] or sha(source) != expected["sha256"]:
            raise ValueError("original prepared dataset changed")
        expected_data[source.name] = expected
    source_input = configuration["args"]["data"]
    if source_input != f"/opt/prometheus-{parent}-inputs/data":
        raise ValueError("parent input directory outside exact owned scope")
    name = parent + "-inference-audit-" + args.revision
    version = ROOT / "versions" / ("code-" + name)
    protocol_path = ROOT / (name + "-protocol.json")
    receipt_path = ROOT / (name + "-coordinator-receipt.json")
    if version.exists() or protocol_path.exists() or receipt_path.exists(): raise FileExistsError("new revision required; do not overwrite audit evidence")
    version.mkdir()
    # Copy EVERY original Python file unchanged; the new audit/launcher are extra.
    files = dict(original)
    for leaf in ("audit_char_inference.py", Path(__file__).name):
        if leaf in files: raise ValueError("audit input must be an original training parent, not an audit parent")
        data = (ROOT / leaf).read_bytes(); ast.parse(data, filename=leaf); files[leaf] = data
    hashes = {leaf: hashlib.sha256(data).hexdigest() for leaf, data in files.items()}
    for leaf, data in files.items(): (version / leaf).write_bytes(data)
    protocol = dict(before_execution=True, name=name, parent=parent, parent_arm=arm,
        owner_scope="owned Prometheus300-update frozen char checkpoints only; no external pickle or model code",
        checkpoint_sha256=expected_checkpoint["sha256"], checkpoint_bytes=expected_checkpoint["bytes"],
        original_source_sha256={leaf: hashlib.sha256(data).hexdigest() for leaf,data in original.items()},
        deployed_source_sha256=hashes, exact_original_continuous_sha256=CONTINUOUS_SHA if arm == "block" else None,
        dataset_metadata_sha256=configuration["metadata_sha256"], input_files=expected_data,
        training_updates=0, selection="none; immutable fixed checkpoints and original512pairedtargets",
        probability_scope="continuous hardCALL retains continuous normalizedreadout; soft/hardCALLNLL/PPL both valid",
        inference_protocol=dict(contexts=[32,64,128], target_count=512, batch_sizes=[1,64], warmups=2, latency_draws=12),
        costs="all warmups/draws/evaluations counted; residentinput latency; inference no optimizer; source checkpoint optimizer storage reported separately",
        external_seconds=900, internal_seconds=850, gpu_fraction=.3,
        isolation="existing authorizedGPUrunner, no network/credential/permission expansion; existing selftaskcomm exception only",
        serial_GPU_scheduling_required=True, resource_overlap_note=args.resource_overlap_note,
        formal_Primary=False, scope="frozen inference development audit; no training or newbenchmark claim")
    with protocol_path.open("x") as stream: json.dump(protocol, stream, indent=2, sort_keys=True); stream.write("\n")
    config = dict(name=name, parent=parent, arm=arm, remote=REMOTE,
        files={leaf:base64.b64encode(data).decode() for leaf,data in files.items()}, hashes=hashes,
        source_input=source_input, expected_data=expected_data, checkpoint_leaf=expected_checkpoint["file"],
        checkpoint_sha256=expected_checkpoint["sha256"], checkpoint_bytes=expected_checkpoint["bytes"],
        runner_sha256=GPU_RUNNER_SHA, resource_overlap_note=args.resource_overlap_note)
    remote = '''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists() or out.exists() or inputs.exists():raise FileExistsError('new audit revision required; no existingartifact overwritten')
runner=r/'remote_runner_gpu_authorized.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()==c['runner_sha256']
parent_checkpoint=r/'outputs'/c['parent']/'learn'/c['checkpoint_leaf']
assert parent_checkpoint.is_file() and not parent_checkpoint.is_symlink()
assert parent_checkpoint.stat().st_size==c['checkpoint_bytes']
assert hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()==c['checkpoint_sha256']
source=pathlib.Path(c['source_input'])
assert source==pathlib.Path('/opt/prometheus-'+c['parent']+'-inputs/data') and source.is_dir() and not source.is_symlink()
actual_files={p.name for p in source.iterdir()}
assert actual_files==set(c['expected_data'])
for leaf,expected in c['expected_data'].items():
 p=source/leaf;assert p.is_file() and not p.is_symlink() and p.stat().st_size==expected['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
code.mkdir();inputs.mkdir();shutil.copytree(source,inputs/'data');shutil.copyfile(parent_checkpoint,inputs/'source.pt')
assert hashlib.sha256((inputs/'source.pt').read_bytes()).hexdigest()==c['checkpoint_sha256']
for leaf,encoded in c['files'].items():
 data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==c['hashes'][leaf]
 p=code/leaf;p.write_bytes(data);p.chmod(0o444)
code.chmod(0o555)
for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
inputs.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'audit_char_inference.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--checkpoint',str(inputs/'source.pt'),'--frozen-source',str(code),'--data',str(inputs/'data'),'--out',str(out/'audit'),'--device','cuda:0','--max-seconds','850','--latency-repeats','12','--resource-overlap-note',c['resource_overlap_note']]
print(json.dumps({'event':'frozen_owned_inference_launch','command':cmd,'parent':c['parent'],'checkpoint_sha256':c['checkpoint_sha256']}),flush=True)
start=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=930)
 receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-start}
except subprocess.TimeoutExpired as e:
 receipt={'returncode':None,'transport':'coordinator_wait_timeout; inspectactualrunner before assuming completion','stdout':str(e.stdout),'stderr':str(e.stderr),'wall_seconds':time.monotonic()-start}
receipt.update(parent=c['parent'],source_checkpoint_sha256=c['checkpoint_sha256'],source_checkpoint_unchanged=hashlib.sha256(parent_checkpoint.read_bytes()).hexdigest()==c['checkpoint_sha256'],copied_checkpoint_unchanged=hashlib.sha256((inputs/'source.pt').read_bytes()).hexdigest()==c['checkpoint_sha256'])
if out.exists():(out/'coordinator-exit.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''.replace("CONFIG", repr(config), 1)
    ast.parse(remote)
    started = time.monotonic()
    try:
        x = subprocess.run(SSH, input=remote, text=True, capture_output=True, timeout=960)
        receipt = dict(returncode=x.returncode, stdout=x.stdout, stderr=x.stderr, wall_seconds=time.monotonic()-started)
    except subprocess.TimeoutExpired as e:
        receipt = dict(returncode=None, error="SSH waittimeout; remote boundedrunner state must be inspected; no automatic rerun", stdout=str(e.stdout), stderr=str(e.stderr), wall_seconds=time.monotonic()-started)
    with receipt_path.open("x") as stream: json.dump(receipt, stream, indent=2); stream.write("\n")
    print(json.dumps(receipt)); raise SystemExit(receipt["returncode"] if receipt["returncode"] is not None else 124)


if __name__ == "__main__": main()
