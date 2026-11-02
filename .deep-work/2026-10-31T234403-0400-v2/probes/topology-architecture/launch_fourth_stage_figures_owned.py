"""Trusted local freeze / ROOT-only bounded CPU launch for NEW stage4 figures.

--prepare performs byte checks and stdlib arithmetic preflight only: no SSH,
Matplotlib, model load, inference, training, installation or old artifact edit.
--execute uses the frozen exact code/input bundle and unchanged CPU runner.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
NAME = "parameter-fourth-stage-development-figures-v1"
REMOTE = "/workspace/prometheus-v2-m0-architecture"
CPU_RUNNER_SHA = "c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335"
SSH = ["ssh", "-o", "BatchMode=yes", "-p", "24343", "-i",
       "/Users/xiaoyangliu/.ssh/runpod_prometheus", "root@195.26.232.180", "python3", "-"]
ARMS = ("dynamic", "static", "gru", "transformer")
GRAPH_ARMS = ("dynamic", "static", "detached", "gru", "transformer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def checked(path, expected=None):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 16 << 20:
        raise ValueError("owned bounded regular JSON/source file required: " + str(path))
    actual = dict(bytes=path.stat().st_size, sha256=sha(path))
    if expected is not None and actual != expected:
        raise ValueError("actual synchronized source bytes differ: " + str(path))
    return actual


def freeze_inputs(include_graphs):
    contents = {}
    files = {}
    stage4 = {}
    graphs = {}
    def add(source, relative, expected=None):
        if Path(relative).suffix not in (".json", ".jsonl") or ".." in Path(relative).parts:
            raise ValueError("only bounded source JSON/JSONL may enter plot inputs")
        receipt = checked(source, expected)
        contents[relative] = source.read_bytes()
        files[relative] = dict(**receipt, source_path=str(source))
        return relative
    ledger = ROOT / "memory-live-stage-parameter-complete-cost-evidence-v1.json"
    ledger_relative = add(ledger, "ledger/" + ledger.name)
    upstream = ROOT / "summarize_memory_live_stage.py"
    checked(upstream)
    for arm in ARMS:
        run = f"parameter-memory-live-stage-n8-seed0-{arm}-stage4-v1"
        folder = ROOT / "runs" / run
        sync_file = folder / "sync-hashes.json"
        sync = json.loads(sync_file.read_text())["outputs/" + run]
        entry = dict(run=run)
        for role, leaf in (("raw", "learn/records.jsonl"), ("result", "learn/result.json"),
                           ("remote_receipt", "coordinator-exit.json")):
            entry[role] = add(folder / leaf, "stage4/" + arm + "/" + Path(leaf).name, sync[leaf])
        entry["sync_manifest"] = add(sync_file, "stage4/" + arm + "/sync-hashes.json")
        local = ROOT / (run + "-coordinator-receipt.json")
        receipt = json.loads(local.read_text())
        frozen_launch = ROOT / "versions" / ("code-" + run) / "remote-launch.py.source"
        if receipt.get("returncode") != 0 or receipt.get("exact_remote_script_sha256") != sha(frozen_launch):
            raise ValueError("actual local complete-SSH receipt/source script mismatch")
        entry["local_receipt"] = add(local, "stage4/" + arm + "/local-coordinator-receipt.json")
        stage4[arm] = entry
    if include_graphs:
        for arm in GRAPH_ARMS:
            run = f"parameter-char-alltoken-{arm}-300-v1-cuda-graph-inference-v1"
            folder = ROOT / "runs" / run
            sync_file = folder / "sync-hashes.json"
            sync = json.loads(sync_file.read_text())["outputs/" + run]
            entry = dict(run=run)
            for role, leaf in (("raw", "audit/records.jsonl"), ("summary", "audit/summary.json")):
                entry[role] = add(folder / leaf, "graphs/" + arm + "/" + Path(leaf).name, sync[leaf])
            entry["sync_manifest"] = add(sync_file, "graphs/" + arm + "/sync-hashes.json")
            graphs[arm] = entry
    manifest = dict(schema="frozen-stage4-development-figure-inputs-v1", frozen_at_UTC=datetime.now(timezone.utc).isoformat(),
        files=files, ledger=ledger_relative, ledger_upstream_script_sha256=sha(upstream),
        stage4=stage4, graphs=graphs,
        source_policy="exact synchronized actual JSON/JSONL bytes + current canonical scoped ledger; no NN or TEST prediction archives",
        output_policy="TRAIN/VALID and raw600 losses only; graph resident inference is a separate frozen LM task",
        no_new_ledger=True, single_seed=True, formal_Primary=False)
    return contents, manifest


REMOTE_SCRIPT = r'''import base64,hashlib,json,pathlib,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists() or out.exists() or inputs.exists():raise FileExistsError('newnamespace only; no old artifacts overwritten or automaticretry')
runner=r/'remote_runner_cpu_v2.py'
if hashlib.sha256(runner.read_bytes()).hexdigest()!=c['runner_sha256']:raise ValueError('unchanged CPUrunner SHA differs')
if not pathlib.Path('/opt/prometheus-analysis-python').is_dir():raise RuntimeError('existing Matplotlib dependency directory absent; no installation/fallback')
code.mkdir();inputs.mkdir()
for leaf,encoded in c['sources'].items():
 raw=base64.b64decode(encoded)
 if hashlib.sha256(raw).hexdigest()!=c['source_sha256'][leaf]:raise ValueError('frozen plot/source transport SHA differs')
 target=code/leaf;target.write_bytes(raw);target.chmod(0o444)
code.chmod(0o555)
for relative,encoded in c['inputs'].items():
 rel=pathlib.Path(relative)
 if rel.is_absolute() or '..' in rel.parts or rel.suffix not in ('.json','.jsonl'):raise ValueError('JSON inputs only; no NN/predictionarchive')
 raw=base64.b64decode(encoded);expected=c['input_receipts'][relative]
 if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('frozen input transport SHA differs')
 target=inputs/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);target.chmod(0o444)
for target in sorted(inputs.rglob('*'),reverse=True):
 if target.is_dir():target.chmod(0o555)
inputs.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'plot_fourth_stage_development.py'),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--inputs',str(inputs),'--out',str(out/'figures'),'--max-seconds','100']
print(json.dumps({'event':'frozen_stage4_figures_CPU_launch','command':cmd,'input_manifest_sha256':c['input_manifest_sha256'],'input_files':len(c['input_receipts'])}),flush=True)
began=time.monotonic()
try:
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=150)
 receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-began,input_manifest_sha256=c['input_manifest_sha256'])
except subprocess.TimeoutExpired as error:
 receipt=dict(returncode=None,error='CPUcoordinatorwaittimeout; inspect boundedrunner/noautomaticretry',stdout=str(error.stdout),stderr=str(error.stderr),wall_seconds=time.monotonic()-began)
if out.exists():
 with (out/'coordinator-exit.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''


def prepare(include_graphs=True):
    version = ROOT / "versions" / ("code-" + NAME)
    protocol_path = ROOT / (NAME + "-protocol.json")
    for path in (version, protocol_path, ROOT / "runs" / NAME, ROOT / (NAME + "-coordinator-receipt.json")):
        if path.exists():
            raise FileExistsError("new fixed figure namespace already exists; no overwrite: " + str(path))
    contents, manifest = freeze_inputs(include_graphs)
    script = ROOT / "plot_fourth_stage_development.py"
    spec = importlib.util.spec_from_file_location("_owned_stage4_plot_stdlib_preflight", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    # Arithmetic/source preflight happens before committing the new namespace.
    with tempfile.TemporaryDirectory(prefix="prometheus-stage4-figure-preflight-") as temporary:
        folder = Path(temporary)
        for relative, raw in contents.items():
            file = folder / relative; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(raw)
        (folder / "manifest.json").write_bytes(manifest_bytes)
        checked_data = module.load_bundle(folder)
    sources = {}
    for leaf in (script.name, Path(__file__).name):
        raw = (ROOT / leaf).read_bytes(); ast.parse(raw, filename=leaf); sources[leaf] = raw
    source_hashes = {leaf:hashlib.sha256(raw).hexdigest() for leaf,raw in sources.items()}
    contents["manifest.json"] = manifest_bytes
    input_receipts = {leaf:dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for leaf,raw in contents.items()}
    protocol = dict(before_execution=True, name=NAME, source_sha256=source_hashes,
        input_manifest_sha256=input_receipts["manifest.json"]["sha256"], input_files=input_receipts,
        scope="actual fourth-phase fourarm single-seed DEVELOPMENT curves and scoped costs; not Primary",
        current_phase_only=True, no_TEST_performance_plot=True,
        plots=["stage4_acquisition", "stage4_old_valid_retention", "stage4_raw_train_losses", "stage4_paid_resources"] +
            (["frozen_lm_graph_resident_latency", "frozen_lm_graph_paid_lifecycle"] if include_graphs else []),
        formats=["pdf", "png"], fixed_raw_loss_EMA_display_only=.95,
        graph_costs_are_separate_frozen_LM_task=True, no_training_speed_comparison_claim=True,
        source_counts=dict(stage4_actual_updates_per_arm={a:600 for a in ARMS}, TRAIN_VALID_points_per_arm={a:7 for a in ARMS}),
        missing_values="not replaced with zero; full FLOPs/energy/capability-normalized deltaP remain unknown",
        label_policy="deltaP means added token embedding storage, not delta capability; native aliases not independent replicates",
        runner_sha256=CPU_RUNNER_SHA, dependency_directory="/opt/prometheus-analysis-python (already installed; no install)",
        outside_seconds=120, inside_seconds=100, address_space_GiB=64, file_limit_MiB=100,
        training_updates=0, model_inference=0, network_in_scientific_child=False, formal_Primary=False,
        visual_QA="Root inspects actual generated PNG/PDF before publication copying; no automatic visual-pass claim")
    config = dict(name=NAME, remote=REMOTE, sources={k:base64.b64encode(v).decode() for k,v in sources.items()},
        source_sha256=source_hashes, inputs={k:base64.b64encode(v).decode() for k,v in contents.items()},
        input_receipts=input_receipts, input_manifest_sha256=input_receipts["manifest.json"]["sha256"], runner_sha256=CPU_RUNNER_SHA)
    config = json.loads(json.dumps(config, sort_keys=True))
    remote = REMOTE_SCRIPT.replace("CONFIG", repr(config), 1); ast.parse(remote)
    version.mkdir()
    for leaf, raw in sources.items():
        with (version / leaf).open("xb") as stream:stream.write(raw)
    input_folder = version / "inputs"; input_folder.mkdir()
    for relative, raw in contents.items():
        file = input_folder / relative; file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("xb") as stream:stream.write(raw)
    write_json(version / "launch-config.json", config)
    with (version / "remote-launch.py.source").open("x") as stream:stream.write(remote)
    write_json(protocol_path, protocol)
    return dict(status="FROZEN_FOR_ROOT_CPU_EXECUTION_NO_SSH", name=NAME,
        source_sha256=source_hashes, input_files=len(input_receipts),
        input_manifest_sha256=protocol["input_manifest_sha256"], remote_script_sha256=sha(version / "remote-launch.py.source"),
        validated_new_updates=2400, validated_curve_points=28, graphs=bool(checked_data["graphs"]))


def execute():
    version = ROOT / "versions" / ("code-" + NAME)
    config = json.loads((version / "launch-config.json").read_text())
    if config["name"] != NAME:
        raise ValueError("frozen plot namespace differs")
    for leaf, digest in config["source_sha256"].items():
        if sha(version / leaf) != digest:
            raise ValueError("frozen plot source changed")
    if sha(Path(__file__)) != config["source_sha256"][Path(__file__).name]:
        raise ValueError("trusted coordinator changed after freeze")
    for relative, expected in config["input_receipts"].items():
        checked(version / "inputs" / relative, expected)
    remote = REMOTE_SCRIPT.replace("CONFIG", repr(config), 1); ast.parse(remote)
    if remote != (version / "remote-launch.py.source").read_text():
        raise ValueError("frozen transport command changed")
    receipt_path = ROOT / (NAME + "-coordinator-receipt.json")
    if receipt_path.exists() or (ROOT / "runs" / NAME).exists():
        raise FileExistsError("actual prior figure run exists; inspect instead of automatic retry")
    began = time.monotonic()
    try:
        result = subprocess.run(SSH, input=remote, text=True, capture_output=True, timeout=180)
        receipt = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                       wall_seconds=time.monotonic()-began, exact_remote_script_sha256=sha(version / "remote-launch.py.source"))
    except subprocess.TimeoutExpired as error:
        receipt = dict(returncode=None, error="SSHwaittimeout; inspect actual bounded CPU run/noautomaticretry",
                       stdout=str(error.stdout), stderr=str(error.stderr), wall_seconds=time.monotonic()-began)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt))
    raise SystemExit(receipt["returncode"] if receipt["returncode"] is not None else 124)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--without-graphs", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(not args.without_graphs)))
    else:
        if args.without_graphs:parser.error("execution uses the already frozen exact plot set")
        execute()


if __name__ == "__main__":
    main()
