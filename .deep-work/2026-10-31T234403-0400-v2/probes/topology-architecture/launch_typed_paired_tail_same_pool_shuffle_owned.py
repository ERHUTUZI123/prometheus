"""ROOT-only frozen OWNtypedstage2+600 sampler-conditioning tail; no old1800 rerun or automatic retry."""
from __future__ import annotations
import ast, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAME = 'parameter-typed-paired-tail-same-pool-shuffle-n8-seed0-600-v1'
VERSION = ROOT / 'versions' / ('code-' + NAME)
PROTOCOL = ROOT / (NAME + '-protocol.json')
REMOTE_SOURCE = VERSION / 'remote-launch.source'
RECEIPT = ROOT / (NAME + '-coordinator-receipt.json')
SOURCE_SHA256 = {'context_replay128.py': 'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140', 'continuous_primitive_block.py': 'efabc8cbd2754a4a1d60f52b907cb5842f11fcf716c52646f8f935db5170d3f2', 'memory_live_stage_train.py': 'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6', 'owned_fcba_core.py': 'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158', 'owned_parent_driver.py': 'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc', 'typed_memory_actions_masked_graph_fb.py': '3216c90f3fddd324ae81072f01c9ab858ec09ba8d32c2594dd4869818864c321', 'typed_memory_actions_graph_engine.py': '99acb12d2661f9bf8a346353c62da2f701ea8599aacf88c8bfb868d9225bced6', 'owned_full_driver.py': '16379dda778c0a97f194b238685ee80955d73bd44cfedf71dd36a7ed2b62adb3', 'train_typed_paired_tail_stage3_graph.py': '350ff4f26884a88d6f7f2d2d5be459fc8799bf1f3f7ceeed7961d9b9812a671c'}
PROTOCOL_SHA256 = '4a7d5ef79c77998df1ab1d7317c8d5e0ce325365376ee0e65528de4bb22f2e71'
REMOTE_SHA256 = '174c68e5c087fe9a0c0d378f12c977eacdc835097592c134db8e6e2891588626'
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
