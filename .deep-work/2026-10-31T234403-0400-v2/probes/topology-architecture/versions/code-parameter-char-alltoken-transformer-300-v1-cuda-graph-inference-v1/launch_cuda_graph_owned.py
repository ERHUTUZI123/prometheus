"""Trusted two-action freeze/launch for owned all-token CUDA Graph inference.

`prepare` is local, read-only on parents and makes NEW source/protocol artifacts.
`run` is ROOT ONLY: exact frozen bundle, existing SSH route and unchanged bounded
GPU runner. Preparation never opens SSH, executes a model or starts background
work. No existing evidence is overwritten; Root records actual GPU occupancy.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
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
RUNNER_SHA = "85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0"
SOURCES = {
    "continuous_primitive_block.py": "58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497",
    "train_char_alltoken.py": "567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02",
}
ARMS = ("dynamic", "static", "detached", "gru", "transformer")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def namespace(parent, revision):
    match = re.fullmatch(r"parameter-char-alltoken-(dynamic|static|detached|gru|transformer)-300-v1", parent)
    if match is None or not re.fullmatch(r"v[1-9][0-9]*", revision):
        raise ValueError("exact owned original all-token parent and new vN revision required")
    return parent + "-cuda-graph-inference-" + revision, match.group(1)


def inspect_parent(parent, arm):
    actual = ROOT / "runs" / parent
    summary = json.loads((actual / "learn/summary.json").read_text())
    manifest = json.loads((actual / "sync-hashes.json").read_text())
    configuration = None
    updates = []
    with (actual / "learn/records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "configuration":
                if configuration is not None:
                    raise ValueError("duplicate parent configuration")
                configuration = row
            elif row["kind"] == "actual_alltoken_update":
                updates.append(row["step"])
    if (summary.get("arm") != arm or summary.get("completed_updates") != 300
            or summary.get("formal_Primary") is not False or updates != list(range(1, 301))
            or configuration is None or configuration["source_sha256"] != SOURCES
            or summary["source_sha256"] != SOURCES):
        raise ValueError("actual completed300 exact-source parent required")
    checkpoint_receipt = summary["checkpoints"][-1]
    if checkpoint_receipt["file"] != arm + "-frozen-final.pt" or checkpoint_receipt["completed_updates"] != 300:
        raise ValueError("parent final frozen checkpoint differs")
    checkpoint = actual / "learn" / checkpoint_receipt["file"]
    receipt = manifest["outputs/" + parent]["learn/" + checkpoint_receipt["file"]]
    if (checkpoint.is_symlink() or checkpoint.stat().st_size != checkpoint_receipt["bytes"]
            or sha(checkpoint) != checkpoint_receipt["sha256"] or receipt["sha256"] != checkpoint_receipt["sha256"]
            or receipt["bytes"] != checkpoint_receipt["bytes"]):
        raise ValueError("actual checkpoint differs from saved/synchronized receipt")
    version = ROOT / "versions" / ("code-" + parent)
    original = {}
    for leaf, expected in SOURCES.items():
        path = version / leaf
        raw = path.read_bytes()
        tracked = manifest["code-" + parent][leaf]
        if path.is_symlink() or hashlib.sha256(raw).hexdigest() != expected or tracked != dict(bytes=len(raw), sha256=expected):
            raise ValueError("exact original frozen source mismatch: " + leaf)
        ast.parse(raw, filename=leaf)
        original[leaf] = raw
    if {x.name for x in version.glob("*.py")} != set(SOURCES):
        raise ValueError("unexpected original executable dependency; inspect before copying")
    data_run = ROOT / "runs/parameter-char-alltoken-data-v1"
    data = data_run / "data"
    metadata_file = data / "metadata.json"
    if sha(metadata_file) != configuration["metadata_sha256"] or configuration["metadata_sha256"] != summary["metadata_sha256"]:
        raise ValueError("original metadata differs from training lineage")
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("schema") != "char-alltoken-ar-development-v1" or metadata["files"]["train"]["file"] != "train.npz":
        raise ValueError("fixed original TRAIN all-token data required")
    data_manifest = json.loads((data_run / "sync-hashes.json").read_text())["outputs/parameter-char-alltoken-data-v1"]
    expected_data = {}
    for leaf in ("metadata.json", "train.npz"):
        path = data / leaf
        record = data_manifest["data/" + leaf]
        if path.is_symlink() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("actual original TRAIN input file changed")
        expected_data[leaf] = record
    if expected_data["train.npz"]["sha256"] != metadata["files"]["train"]["sha256"]:
        raise ValueError("TRAIN archive hash differs from original metadata")
    # Parse only literal allowlist constants from the NEW audit: no model import.
    tree = ast.parse((ROOT / "audit_cuda_graph_inference.py").read_bytes())
    owned_literal = next(ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OWNED" for t in node.targets))
    expected_p, expected_bytes, expected_hash = owned_literal[arm]
    if (summary["resources"]["parameters"] != expected_p or checkpoint_receipt["bytes"] != expected_bytes
            or checkpoint_receipt["sha256"] != expected_hash):
        raise ValueError("checkpoint not in the NEW audit's exact owned allowlist")
    source_input = configuration["args"]["data"]
    if source_input != f"/opt/prometheus-{parent}-inputs/data":
        raise ValueError("parent TRAIN input outside exact owned directory")
    return dict(summary=summary, configuration=configuration, original=original,
                checkpoint=checkpoint_receipt, data=expected_data, source_input=source_input)


def prepare(parent, revision):
    name, arm = namespace(parent, revision)
    version = ROOT / "versions" / ("code-" + name)
    protocol_path = ROOT / (name + "-protocol.json")
    if any(path.exists() for path in (version, protocol_path, ROOT / "runs" / name, ROOT / (name + "-coordinator-receipt.json"))):
        raise FileExistsError("new audit revision required; no evidence overwritten")
    old = inspect_parent(parent, arm)
    files = dict(old["original"])
    for leaf in ("audit_cuda_graph_inference.py", Path(__file__).name):
        raw = (ROOT / leaf).read_bytes()
        ast.parse(raw, filename=leaf)
        files[leaf] = raw
    hashes = {leaf: hashlib.sha256(raw).hexdigest() for leaf, raw in files.items()}
    protocol = dict(schema="owned-alltoken-cuda-graph-cost-diagnostic-v1", before_execution=True,
        frozen_at_UTC=datetime.now(timezone.utc).isoformat(), name=name, parent=parent, arm=arm,
        source_sha256=hashes, original_source_sha256=SOURCES,
        checkpoint=old["checkpoint"], source_input=old["source_input"], expected_input_files=old["data"],
        source_parent_resources=old["summary"]["resources"], source_parent_costs=old["summary"]["costs"],
        question="Does removing repeated Python/launch scheduling explain the measured resident inference cost gap while exact arithmetic and outputs are retained?",
        inputs="first32 originalTRAIN windows, B1/32, tile64tokens to L64/128/256; only metadata+TRAIN file copied; targets never decompressed/read",
        backends=["original_eager", "tensor_eager", "CUDA_graph_replay"],
        block_transform="original initial_carry checked outsidecapture; exact tensor initialization including encoder EVERY call; original _advance_embedded and alltokenreadout; all-active identity indexing removed",
        control_transform="original TF/GRU embedding/layers/attention/masks/position/norm/readout unchanged; remove outer CPU validation only",
        modes=["soft", "native"] if arm in ("dynamic", "static", "detached") else ["soft"],
        contexts=[64, 128, 256], batches=[1, 32], warmups_per_eager=3, graph_side_stream_warmups=3, resident_latency_draws=12,
        output_contract=dict(all_elements_atol=2e-5, all_elements_rtol=2e-6, identical_argmax_required=True,
                             lognormalizer_maximum=1e-5, original_vs_tensor_vs_graph=True, graph_checked_after_draws=True),
        graph=dict(keep_graph=True, explicit_instantiation=True, error_mode="global", private_pool_per_case=True,
                   setup_capture_instantiation_sync_cleanup_all_charged=True, no_compile_fallback=True),
        resources="allP/weights/buffers/savedoptimizer/checkpoint/history; liveTorchallocator/globalfree/RSS; perphaseforwardcounts/MACproxy; rawwall+CUDAstreamdraws; graphpool/setup/cleanup; exactCUDAexecutablebytes unavailable",
        amortization="primary ceil((graph sidewarmup+construction+capture+instantiation+firstvalidation+postdrawverification+poolcleanup)/(eagerP50-graphP50)); undefined if saving<=0; persistent-setup-only bound separately labeled; shared input/loading/audit costs separately charged",
        inference_scope="resident fullwindow alltoken outputs, excludes H2D/serving/cached sequential generation; streamtime is not kernel-only sum; no arithmetic/FLOP improvement claim",
        training_updates=0, label_reads=0, accuracy_selection=False, formal_Primary=False,
        external_seconds=900, internal_seconds=850, gpu_fraction=.3, runner_sha256=RUNNER_SHA,
        isolation="unchanged authorized GPU runner; no permissions/install/framework/network expansion in scientific child",
        serial_root_scheduling_required=True, preparation_does_not_launch=True)
    version.mkdir()
    for leaf, raw in files.items():
        with (version / leaf).open("xb") as dest:
            dest.write(raw)
    write_json(protocol_path, protocol)
    return dict(status="PREPARED_NOT_EXECUTED", name=name, protocol=str(protocol_path),
                frozen_source=str(version), source_sha256=hashes)


def run(parent, revision, overlap_note):
    name, arm = namespace(parent, revision)
    protocol_path = ROOT / (name + "-protocol.json")
    protocol = json.loads(protocol_path.read_text())
    if protocol["name"] != name or protocol["parent"] != parent or protocol["arm"] != arm:
        raise ValueError("prepared protocol namespace mismatch")
    if any(path.exists() for path in (ROOT / "runs" / name, ROOT / (name + "-coordinator-receipt.json"))):
        raise FileExistsError("audit run/evidence already exists; inspect actual state, no automatic rerun")
    old = inspect_parent(parent, arm)
    if old["checkpoint"] != protocol["checkpoint"] or old["data"] != protocol["expected_input_files"]:
        raise ValueError("actual parent/input lineage changed since freeze")
    version = ROOT / "versions" / ("code-" + name)
    files = {}
    for leaf, digest in protocol["source_sha256"].items():
        path = version / leaf
        if Path(leaf).name != leaf or path.is_symlink() or sha(path) != digest:
            raise ValueError("prepared source bytes changed: " + leaf)
        files[leaf] = base64.b64encode(path.read_bytes()).decode()
    if sha(Path(__file__)) != protocol["source_sha256"][Path(__file__).name]:
        raise ValueError("trusted coordinator changed after source freeze")
    config = dict(name=name, parent=parent, arm=arm, remote=REMOTE, files=files,
                  hashes=protocol["source_sha256"], source_input=protocol["source_input"],
                  expected_data=protocol["expected_input_files"], checkpoint=protocol["checkpoint"],
                  runner_sha256=RUNNER_SHA, overlap_note=overlap_note, protocol_sha256=sha(protocol_path))
    remote = '''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists() or out.exists() or inputs.exists():raise FileExistsError('newnamespace required; preserve actual priorattempt')
runner=r/'remote_runner_gpu_authorized.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()==c['runner_sha256']
source=r/'outputs'/c['parent']/'learn'/c['checkpoint']['file']
assert source.is_file() and not source.is_symlink() and source.stat().st_size==c['checkpoint']['bytes']
assert hashlib.sha256(source.read_bytes()).hexdigest()==c['checkpoint']['sha256']
data=pathlib.Path(c['source_input']);assert data==pathlib.Path('/opt/prometheus-'+c['parent']+'-inputs/data')
assert data.is_dir() and not data.is_symlink()
for leaf,expected in c['expected_data'].items():
 p=data/leaf;assert p.is_file() and not p.is_symlink() and p.stat().st_size==expected['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256']
code.mkdir();inputs.mkdir();(inputs/'data').mkdir()
shutil.copyfile(source,inputs/'source.pt')
for leaf in c['expected_data']:shutil.copyfile(data/leaf,inputs/'data'/leaf)
assert hashlib.sha256((inputs/'source.pt').read_bytes()).hexdigest()==c['checkpoint']['sha256']
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==c['hashes'][leaf]
 p=code/leaf;p.write_bytes(raw);p.chmod(0o444)
code.chmod(0o555)
for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
inputs.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'audit_cuda_graph_inference.py'),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--checkpoint',str(inputs/'source.pt'),'--frozen-source',str(code),'--data',str(inputs/'data'),'--out',str(out/'audit'),'--arm',c['arm'],'--device','cuda:0','--max-seconds','850','--resource-overlap-note',c['overlap_note']]
print(json.dumps({'event':'frozen_cuda_graph_inference_launch','command':cmd,'parent':c['parent'],'protocol_sha256':c['protocol_sha256'],'source_checkpoint_sha256':c['checkpoint']['sha256']}),flush=True)
started=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=930)
 receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired as error:
 receipt=dict(returncode=None,error='coordinatorwaittimeout; inspect actual boundedrunner; no automaticretry',stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-started)
receipt.update(parent=c['parent'],protocol_sha256=c['protocol_sha256'],source_checkpoint_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==c['checkpoint']['sha256'],copied_checkpoint_unchanged=hashlib.sha256((inputs/'source.pt').read_bytes()).hexdigest()==c['checkpoint']['sha256'])
if out.exists():
 with (out/'coordinator-exit.json').open('x') as dest:json.dump(receipt,dest,indent=2);dest.write('\\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''.replace("CONFIG", repr(config), 1)
    ast.parse(remote)
    started = time.monotonic()
    try:
        result = subprocess.run(SSH, input=remote, text=True, capture_output=True, timeout=960)
        receipt = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                       wall_seconds=time.monotonic() - started, protocol_sha256=sha(protocol_path))
    except subprocess.TimeoutExpired as error:
        receipt = dict(returncode=None, error="SSHwaittimeout; inspect actual boundedrunner before any retry",
                       stdout=str(error.stdout), stderr=str(error.stderr), wall_seconds=time.monotonic() - started,
                       protocol_sha256=sha(protocol_path))
    write_json(ROOT / (name + "-coordinator-receipt.json"), receipt)
    print(json.dumps(receipt))
    raise SystemExit(receipt["returncode"] if receipt["returncode"] is not None else 124)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--parent", required=True, choices=tuple("parameter-char-alltoken-" + arm + "-300-v1" for arm in ARMS))
    parser.add_argument("--revision", default="v1")
    parser.add_argument("--resource-overlap-note", default="Root must record actual GPU occupancy and schedule serially; exclusivity is not inferred from this note")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.parent, args.revision)))
    else:
        run(args.parent, args.revision, args.resource_overlap_note)


if __name__ == "__main__":
    main()
