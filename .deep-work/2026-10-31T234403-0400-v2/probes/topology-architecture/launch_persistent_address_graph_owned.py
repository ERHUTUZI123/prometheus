"""ROOT-only frozen persistent input-gated address Graph2400 launcher; exact actualCPUproof pinned; no automatic rerun."""
from __future__ import annotations
import ast, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAME = 'parameter-persistent-input-address-n8-seed0-fourstage-graph-v1'
VERSION = ROOT / 'versions' / ('code-' + NAME)
PROTOCOL = ROOT / (NAME + '-protocol.json')
REMOTE_SOURCE = VERSION / 'remote-launch.source'
RECEIPT = ROOT / (NAME + '-coordinator-receipt.json')
SOURCE_SHA256 = {'context_replay128.py': 'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140', 'continuous_primitive_block.py': 'bdf7e4fdf4d5d6ddc79c464e72ed2be165bd4f2adca6d6c6d8c6c6f4e5bacb10', 'memory_live_stage_train.py': 'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6', 'owned_fcba_core.py': 'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158', 'owned_parent_driver.py': 'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc', 'persistent_address_masked_graph_fb.py': 'f3e56ead0e54c39c718e9c551dd8026e78be28bc800235aa2a9b74fb5db70598', 'train_persistent_address_fourstage_graph.py': 'c0ed5b64848cfbe830af5c0d38a494c4bd770a257a6b5f426c41c1dbaeb9d3bc', 'persistent_address_graph_engine.py': 'ce78b6c41835ccae916be110431c28e2f25d1f296b197ceebd1f5ec5dcf6ddcb'}
PROTOCOL_SHA256 = '3adc221fc01e3e405f44b08f6b6c4c6cbae35a4cb383f83e99d12a46abe32d93'
REMOTE_SHA256 = '379c3433a5feaa8c734e36056593ef807587e54599040849804e3228d89a03d1'
SSH = ['ssh', '-o', 'BatchMode=yes', '-p', '24343', '-i', '/Users/xiaoyangliu/.ssh/runpod_prometheus', 'root@195.26.232.180', 'python3', '-']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if len(sys.argv) != 1:
        raise ValueError('this frozen pairedarm launcher takes no arguments')
    if RECEIPT.exists() or RECEIPT.is_symlink():
        raise FileExistsError('preserve existing receipt; no automatic rerun')
    if sha(PROTOCOL) != PROTOCOL_SHA256 or sha(REMOTE_SOURCE) != REMOTE_SHA256:
        raise ValueError('frozen protocol/coordinator SHA mismatch')
    for leaf, expected in SOURCE_SHA256.items():
        path = VERSION / leaf
        if sha(path) != expected:
            raise ValueError('frozen source SHA mismatch: ' + leaf)
        ast.parse(path.read_bytes())
    remote_source = REMOTE_SOURCE.read_text()
    ast.parse(remote_source)
    started = time.monotonic()
    try:
        x = subprocess.run(SSH, input=remote_source, text=True, capture_output=True, timeout=960)
        value = dict(returncode=x.returncode, stdout=x.stdout, stderr=x.stderr, wall_seconds=time.monotonic() - started)
    except subprocess.TimeoutExpired as error:

        def text(v):
            return v.decode(errors='replace') if isinstance(v, bytes) else v or ''
        value = dict(returncode=None, stdout=text(error.stdout), stderr=text(error.stderr), wall_seconds=time.monotonic() - started, error='SSH wait timeout; actual remote state unknown, inspect existing bounded runner, no automatic retry')
    value.update(name=NAME, protocol_sha256=PROTOCOL_SHA256, remote_launch_sha256=REMOTE_SHA256, source_sha256=SOURCE_SHA256, remote_source_embedded_in_receipt=False)
    with RECEIPT.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    summary = dict(name=NAME, returncode=value['returncode'], receipt=str(RECEIPT), wall_seconds=value['wall_seconds'])
    try:
        summary['remote_summary'] = json.loads(value['stdout'].strip().splitlines()[-1])
    except (ValueError, IndexError):
        summary['remote_summary'] = None
    print(json.dumps(summary, sort_keys=True), flush=True)
    return value['returncode'] if value['returncode'] is not None else 124
if __name__ == '__main__':
    raise SystemExit(main())
