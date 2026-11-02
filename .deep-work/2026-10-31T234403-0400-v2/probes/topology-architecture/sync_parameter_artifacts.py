"""Trusted coordinator: preserve completed remote runs, verify hashes.

Does not execute downloaded code, load Torch objects, or overwrite differing
local evidence. Archive mode creates only new remote exports, leaving original
outputs/source read-only. SSH uses the existing authorized Prometheus key without
reading its contents. Temporary copies are retained if verification fails.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
REMOTE = '/workspace/prometheus-v2-m0-architecture'
KEY = '/Users/xiaoyangliu/.ssh/runpod_prometheus'
HOST = 'root@195.26.232.180'
MAX_FILE_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_TAR_METADATA_BYTES = 16 * 1024 * 1024

# This runs only on the already-owned remote host. It reads completed run/source
# trees and creates NEW unique exports; it never executes their contents.
ARCHIVE_CODE = r'''import hashlib,json,os,pathlib,stat,tarfile,time,uuid
r=pathlib.Path(REMOTE);name=NAME
deadline=time.monotonic()+110
max_file=100*1024*1024;max_total=1024*1024*1024
def tick():
 if time.monotonic()>deadline:raise TimeoutError('archive packaging deadline')
def digest(path,limit):
 tick();h=hashlib.sha256();n=0
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f:
  if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('not regular: '+str(path))
  while True:
   tick();chunk=f.read(1024*1024)
   if not chunk:break
   n+=len(chunk)
   if n>limit:raise ValueError('file exceeds byte bound')
   h.update(chunk)
 return n,h.hexdigest()
manifest={};total=0
for relative in ['outputs/'+name,'code-'+name]:
 root=r/relative
 if root.is_symlink() or not root.is_dir():raise ValueError('owned root not a real directory: '+relative)
 files={}
 for parent,dirs,leaves in os.walk(root,followlinks=False):
  tick();dirs.sort();leaves.sort()
  for entry in dirs:
   p=pathlib.Path(parent)/entry
   if not stat.S_ISDIR(p.lstat().st_mode):raise ValueError('non-directory tree entry: '+str(p))
  for entry in leaves:
   p=pathlib.Path(parent)/entry
   if not stat.S_ISREG(p.lstat().st_mode):raise ValueError('non-regular tree entry: '+str(p))
   n,sha=digest(p,max_file)
   if n>max_file:raise ValueError('file exceeds 100 MiB: '+str(p))
   total+=n
   if total>max_total:raise ValueError('owned trees exceed 1 GiB')
   files[p.relative_to(root).as_posix()]={'bytes':n,'sha256':sha}
 manifest[relative]=files
manifest_bytes=json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()
exports=r/'sync-exports'
exports.mkdir(exist_ok=True)
if exports.is_symlink() or not exports.is_dir():raise ValueError('export root is not a real directory')
stem=name+'-'+uuid.uuid4().hex
archive=exports/(stem+'.tar.gz');receipt=exports/(stem+'.json')
class CheckedReader:
 def __init__(self,f):self.f=f;self.n=0;self.h=hashlib.sha256()
 def read(self,n=-1):
  tick();data=self.f.read(n);self.n+=len(data);self.h.update(data);return data
with archive.open('xb') as raw:
 with tarfile.open(fileobj=raw,mode='w:gz',format=tarfile.PAX_FORMAT,compresslevel=1) as tar:
  for relative,files in manifest.items():
   for rel,info in sorted(files.items()):
    tick();path=r/relative/rel
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as f:
     if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('changed file type: '+str(path))
     reader=CheckedReader(f);member=tarfile.TarInfo(relative+'/'+rel)
     member.size=info['bytes'];member.mode=0o444;member.mtime=0;member.uid=member.gid=0
     tar.addfile(member,reader)
     if reader.n!=info['bytes'] or reader.h.hexdigest()!=info['sha256'] or f.read(1):
      raise ValueError('source changed while packaging: '+str(path))
  tick()
size,sha=digest(archive,max_total)
if size>max_total:raise ValueError('archive exceeds 1 GiB')
descriptor={'manifest':manifest,'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
 'manifest_bytes':len(manifest_bytes),'archive':{'path':str(archive),'bytes':size,'sha256':sha},
 'remote_receipt':str(receipt),'format':'prometheus-owned-sync-tar-v1'}
with receipt.open('x') as f:json.dump(descriptor,f,sort_keys=True,indent=2);f.write('\n')
archive.chmod(0o444);receipt.chmod(0o444)
print(json.dumps(descriptor))
'''


def file_digest(path):
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise ValueError('invalid manifest/member path')
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in ('.', '..') for p in path.parts):
        raise ValueError('unsafe manifest/member path: '+value)
    return path


def validate_manifest(manifest, name):
    roots = {'outputs/'+name, 'code-'+name}
    if not isinstance(manifest, dict) or set(manifest) != roots:
        raise ValueError('archive manifest must contain exactly the two owned roots')
    expected = {}
    total = 0
    for relative, files in manifest.items():
        if not isinstance(files, dict):
            raise ValueError('invalid manifest file table')
        for rel, info in files.items():
            safe_relative(rel)
            if not isinstance(info, dict) or set(info) != {'bytes', 'sha256'}:
                raise ValueError('invalid manifest file metadata')
            size, sha = info['bytes'], info['sha256']
            if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
                raise ValueError('manifest file exceeds 100 MiB or has invalid size')
            if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
                raise ValueError('invalid manifest SHA256')
            total += size
            if total > MAX_TOTAL_BYTES:
                raise ValueError('manifest exceeds 1 GiB')
            expected[relative+'/'+rel] = (relative, rel, info)
    # A file cannot also be an ancestor directory of another manifest file.
    for member in expected:
        if any(parent.as_posix() in expected for parent in PurePosixPath(member).parents):
            raise ValueError('manifest file/directory collision')
    return expected


def extract_verified_archive(archive, temp, expected, manifest):
    staging = temp/'extracted'
    staging.mkdir()
    for relative in manifest:
        (staging/Path(relative).name).mkdir()
    seen = set()
    total = 0
    class BoundedReader:
        def __init__(self, source):
            self.source = source
            self.remaining = sum(info['bytes'] for _, _, info in expected.values()) + MAX_TAR_METADATA_BYTES

        def read(self, size):
            data = self.source.read(min(size, self.remaining + 1))
            self.remaining -= len(data)
            if self.remaining < 0:
                raise ValueError('archive decompressed metadata/padding exceeds bound')
            return data

    # Manual streaming extraction: no extract/extractall, symlink following, or
    # trust in archive permissions. ignore_zeros also visits appended members.
    with gzip.open(archive, 'rb') as compressed, tarfile.open(
            fileobj=BoundedReader(compressed), mode='r|', ignore_zeros=True) as tar:
        for member in tar:
            safe_relative(member.name)
            if (not member.isreg() or member.issparse() or member.linkname
                    or member.name not in expected or member.name in seen):
                raise ValueError('unexpected/nonregular/duplicate archive member: '+member.name)
            relative, rel, info = expected[member.name]
            if member.size != info['bytes'] or not 0 <= member.size <= MAX_FILE_BYTES:
                raise ValueError('archive member size mismatch: '+member.name)
            total += member.size
            if total > MAX_TOTAL_BYTES:
                raise ValueError('archive extracted bytes exceed 1 GiB')
            destination = staging/Path(relative).name/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError('missing archive data: '+member.name)
            h = hashlib.sha256()
            copied = 0
            with source, destination.open('xb') as out:
                while chunk := source.read(1024 * 1024):
                    copied += len(chunk)
                    if copied > info['bytes']:
                        raise ValueError('archive member grew: '+member.name)
                    h.update(chunk)
                    out.write(chunk)
            if copied != info['bytes'] or h.hexdigest() != info['sha256']:
                raise ValueError('archive member hash mismatch: '+member.name)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('archive/manifest member-set mismatch')
    return staging


def archive_transfer(name, temp):
    code = ARCHIVE_CODE.replace('REMOTE', repr(REMOTE), 1).replace('NAME', repr(name), 1)
    raw = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-p', '24343', '-i', KEY,
                          HOST, 'python3', '-'], input=code, text=True,
                         capture_output=True, check=True, timeout=120)
    descriptor = json.loads(raw.stdout)
    # Keep the remote reply and all subsequent temporary artifacts on failure.
    (temp/'archive-descriptor.json').write_text(json.dumps(descriptor, indent=2)+'\n')
    if not isinstance(descriptor, dict) or descriptor.get('format') != 'prometheus-owned-sync-tar-v1':
        raise ValueError('invalid archive descriptor')
    manifest = descriptor['manifest']
    expected = validate_manifest(manifest, name)
    canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
    if (descriptor['manifest_sha256'] != hashlib.sha256(canonical).hexdigest()
            or descriptor['manifest_bytes'] != len(canonical)):
        raise ValueError('archive manifest hash/size mismatch')
    (temp/'manifest.json').write_bytes(canonical)
    info = descriptor['archive']
    path = PurePosixPath(info['path'])
    pattern = re.escape(name)+r'-[0-9a-f]{32}\.tar\.gz'
    if path.parent != PurePosixPath(REMOTE)/'sync-exports' or not re.fullmatch(pattern, path.name):
        raise ValueError('archive is outside the unique owned export path')
    if (type(info['bytes']) is not int or not 0 < info['bytes'] <= MAX_TOTAL_BYTES
            or not isinstance(info['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', info['sha256'])):
        raise ValueError('invalid archive size/hash')
    if descriptor['remote_receipt'] != str(path.with_suffix('').with_suffix('.json')):
        raise ValueError('unexpected remote archive receipt path')
    archive = temp/'bundle.tar.gz'
    subprocess.run(['scp', '-q', '-P', '24343', '-i', KEY, HOST+':'+str(path), str(archive)],
                   check=True, timeout=180)
    if file_digest(archive) != (info['bytes'], info['sha256']):
        raise ValueError('whole archive hash/size mismatch')
    staging = extract_verified_archive(archive, temp, expected, manifest)
    return manifest, staging, descriptor


def preflight_archive_publication(manifest, name):
    def check_path(destination):
        current = ROOT
        for part in destination.relative_to(ROOT).parts:
            current = current/part
            if current.is_symlink():
                raise RuntimeError('refusing symlink in local publication path: '+str(current))
            if current.exists() and current != destination and not current.is_dir():
                raise RuntimeError('local publication ancestor is not a directory: '+str(current))

    for relative, files in manifest.items():
        target = ROOT/('runs' if relative.startswith('outputs/') else 'versions')/Path(relative).name
        check_path(target)
        if target.exists() and not target.is_dir():
            raise RuntimeError('local publication root is not a directory: '+str(target))
        for rel, info in files.items():
            destination = target/rel
            check_path(destination)
            if destination.exists() and (not destination.is_file()
                    or file_digest(destination) != (info['bytes'], info['sha256'])):
                raise RuntimeError('refusing to replace differing local evidence: '+str(destination))
    receipt = ROOT/'runs'/name/'sync-hashes.json'
    check_path(receipt)
    if receipt.is_symlink() or (receipt.exists() and json.loads(receipt.read_text()) != manifest):
        raise RuntimeError('existing sync receipt differs or is a symlink')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='store_true',
                        help='transfer one new manifest-verified archive instead of recursive SCP')
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    for name in args.names:
        if not re.fullmatch(r'parameter-[a-zA-Z0-9_-]+', name):
            raise ValueError('owned parameter run names only')
        descriptor = None
        if args.archive:
            temp = Path(tempfile.mkdtemp(prefix='prometheus-sync-'+name+'-'))
            print(json.dumps({'run':name, 'transfer_temp':str(temp), 'mode':'archive'}), flush=True)
            try:
                manifest, staging, descriptor = archive_transfer(name, temp)
                preflight_archive_publication(manifest, name)
            except BaseException:
                print(json.dumps({'run':name, 'retained_temp':str(temp), 'status':'archive-transfer-failed'}), flush=True)
                raise
        else:
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
            # Keep disposable transfers outside the repository: editor snapshots
            # must not capture a partial transfer as scientific evidence.
            temp = Path(tempfile.mkdtemp(prefix='prometheus-sync-'+name+'-'))
            staging = temp
        for relative, files in manifest.items():
            if not args.archive:
                source = HOST+':'+REMOTE+'/'+relative
                subprocess.run(['scp', '-q', '-r', '-P', '24343', '-i', KEY, source, str(temp)],
                               check=True, timeout=180)
            copied = staging/Path(relative).name
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
        if not args.archive or not receipt.exists():
            receipt.write_text(json.dumps(manifest, indent=2)+'\n')
        if descriptor is not None:
            archive_receipt = receipt.parent/('sync-archive-'+descriptor['archive']['sha256']+'.json')
            if archive_receipt.exists():
                if json.loads(archive_receipt.read_text()) != descriptor:
                    raise RuntimeError('existing archive receipt differs')
            else:
                with archive_receipt.open('x') as out:
                    out.write(json.dumps(descriptor, indent=2)+'\n')
                archive_receipt.chmod(0o444)
        # scp preserves the immutable source directories'0555 permissions;
        # change only this verified disposable copy before removing it.
        for directory in [temp]+[p for p in temp.rglob('*') if p.is_dir()]:
            directory.chmod(0o700)
        shutil.rmtree(temp)
        print(json.dumps({'run':name, 'verified_files':sum(len(f) for f in manifest.values()),
                          'bytes':sum(v['bytes'] for f in manifest.values() for v in f.values())}), flush=True)


if __name__ == '__main__':
    main()
