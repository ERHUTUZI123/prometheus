"""ROOT-only frozen NEW stage2/stage3 shared-address VALID event audit. No old NN reruns, SGD or automatic retry."""
from __future__ import annotations
import ast, hashlib, json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
NAME = 'parameter-input-history-shared-address-two-nn-valid-diagnosis-v1'
VERSION = ROOT / 'versions' / ('code-' + NAME)
PROTOCOL = ROOT / (NAME + '-protocol.json')
REMOTE_SOURCE = VERSION / 'remote-launch.source'
RECEIPT = ROOT / (NAME + '-coordinator-receipt.json')
SOURCE_SHA256 = {'continuous_primitive_block.py': 'e52afc63b096607b3fe9b1c8e1e457347d9b7426bfc194e4cadc71618f4171f2', 'owned_fcba_core.py': 'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158', 'audit_shared_address_valid_two_nn.py': '96a01939c9580e596fa4768a6531fbc4a66173e6ba3d521aff1fcf26e497d7c0'}
PROTOCOL_SHA256 = '3c4631fe1ded5da0456039f8255874b81c1d4d46bf7dfb11671440b8fef429fd'
REMOTE_SHA256 = '3b9e847481e1cafc8f9577208a0b49705dfff7d2d9b3ae7925c61b2edf336d47'
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
