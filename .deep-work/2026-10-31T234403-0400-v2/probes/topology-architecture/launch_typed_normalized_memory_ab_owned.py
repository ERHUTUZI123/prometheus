"""ROOT-only one fixed normalized-memory fresh A/B arm; no automatic retry or parallel NN."""
from __future__ import annotations
import ast, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAME = 'parameter-typed-normalized-memory-ab-n8-seed0-fourstage-graph-v1'
VERSION = ROOT / 'versions' / ('code-' + NAME)
PROTOCOL = ROOT / (NAME + '-protocol.json')
REMOTE_SOURCE = VERSION / 'remote-launch.source'
RECEIPT = ROOT / (NAME + '-coordinator-receipt.json')
SOURCE_SHA256 = {'owned_fcba_core.py': 'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158', 'owned_parent_driver.py': 'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc', 'context_replay128.py': 'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140', 'memory_live_stage_train.py': 'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6', 'continuous_primitive_block.py': '93bdfea778c2f079ec04a6696ab8df77dc8bcc52b7f3fabdc57c05cbcad5250f', 'typed_normalized_memory_masked_graph_fb.py': 'c322fb82f1095ae63463c9c584cf9cba05345514bdf71cb0a9a25b32b0f29294', 'typed_normalized_memory_graph_engine.py': 'fe723ea3ae774d0c848769dc256d54924a333f1611516a527a38936c69843782', 'train_typed_normalized_memory_fourstage_graph.py': 'f391af3b0df93f978e9639c09d730fc94fe93b3570184d005164785e26605a07'}
PROTOCOL_SHA256 = 'efaeab63e8632c36bfffa704bec908a4de81bab086f22a16ef8113378832000d'
REMOTE_SHA256 = '6160319eb31cb11dfa12e3c1e533702d3fd04f0937fa8c833d181d2441ec8971'
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
