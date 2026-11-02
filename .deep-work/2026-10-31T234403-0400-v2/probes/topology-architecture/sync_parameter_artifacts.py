"""Trusted read-only coordinator: preserve completed remote runs, verify hashes.

Does not execute downloaded code, load Torch objects, or overwrite differing
local evidence. SSH uses the existing authorized Prometheus key without reading
its contents. Temporary copies are retained if verification fails.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
REMOTE = '/workspace/prometheus-v2-m0-architecture'
KEY = '/Users/xiaoyangliu/.ssh/runpod_prometheus'
HOST = 'root@195.26.232.180'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    for name in args.names:
        if not re.fullmatch(r'parameter-[a-zA-Z0-9_-]+', name):
            raise ValueError('owned parameter run names only')
        code = """import hashlib,json,pathlib
r=pathlib.Path(REMOTE);name=NAME;result={}
for relative in ['outputs/'+name,'code-'+name]:
 p=r/relative
 if not p.is_dir():raise FileNotFoundError(relative)
 result[relative]={str(f.relative_to(p)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob('*')) if f.is_file()}
print(json.dumps(result))
""".replace('REMOTE', repr(REMOTE), 1).replace('NAME', repr(name), 1)
        raw = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', '24343', '-i', KEY,
                              HOST, 'python3', '-'], input=code, text=True,
                             capture_output=True, check=True, timeout=120)
        manifest = json.loads(raw.stdout)
        temp = Path(tempfile.mkdtemp(prefix='sync-'+name+'-', dir=ROOT))
        for relative, files in manifest.items():
            source = HOST+':'+REMOTE+'/'+relative
            subprocess.run(['scp', '-q', '-r', '-P', '24343', '-i', KEY, source, str(temp)],
                           check=True, timeout=180)
            copied = temp/Path(relative).name
            actual = {str(f.relative_to(copied)) for f in copied.rglob('*') if f.is_file()}
            if actual != set(files):
                raise RuntimeError('remote/copy file-set mismatch: '+relative)
            for rel, info in files.items():
                data = (copied/rel).read_bytes()
                if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
                    raise RuntimeError('copied hash mismatch: '+rel)
            target = ROOT/('runs' if relative.startswith('outputs/') else 'versions')/Path(relative).name
            target.mkdir(parents=True, exist_ok=True)
            for rel in files:
                destination = target/rel
                if destination.exists():
                    if destination.read_bytes() != (copied/rel).read_bytes():
                        raise RuntimeError('refusing to replace differing local evidence: '+str(destination))
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(copied/rel, destination)
        receipt = ROOT/'runs'/name/'sync-hashes.json'
        if receipt.exists() and json.loads(receipt.read_text()) != manifest:
            raise RuntimeError('existing sync receipt differs')
        receipt.write_text(json.dumps(manifest, indent=2)+'\n')
        # scp preserves the immutable source directories'0555 permissions;
        # change only this verified disposable copy before removing it.
        for directory in [temp]+[p for p in temp.rglob('*') if p.is_dir()]:
            directory.chmod(0o700)
        shutil.rmtree(temp)
        print(json.dumps({'run':name, 'verified_files':sum(len(f) for f in manifest.values()),
                          'bytes':sum(v['bytes'] for f in manifest.values() for v in f.values())}), flush=True)


if __name__ == '__main__':
    main()
