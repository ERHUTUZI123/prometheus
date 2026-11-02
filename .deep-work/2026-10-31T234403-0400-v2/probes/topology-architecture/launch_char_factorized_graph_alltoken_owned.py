"""Execute only the frozen20721P factorized alltokenGraph300 cost implementation ROOT coordinator.

This new launcher never edits a prior launcher/result or prepares a different
revision. Source/protocol were frozen before execution. It prints a short
receipt summary; no remote source/base64 payload is included in receipts.
"""
from __future__ import annotations
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-char-alltoken-factorized-graph-300-v1'
VERSION=ROOT/'versions'/('code-'+NAME)
PROTOCOL=ROOT/(NAME+'-protocol.json')
REMOTE_SOURCE=VERSION/'remote-launch.source'
RECEIPT=ROOT/(NAME+'-coordinator-receipt.json')
SOURCE_SHA256={'factorized_alltoken_graph.py': '29ae330b720fcaab9825d1f3bdb264c44c23c94fcc5bda44c84ba92fe8e1d141', 'train_char_factorized_graph_alltoken.py': '11b600dfc53e3e820c3ced396bb05768ecc5b9c56eda7e343e0f5091ef7f3842', 'continuous_primitive_block.py': 'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158', 'train_char_factorized_alltoken.py': '3e6447a2a102c65d216b82b297798429f80b4c9189d0a3f2cc45c4149f584960'}
PROTOCOL_SHA256='c7120d45514990fd4d7fa47482e4d11c2d8e512f3a18b1128a05378b1f6f4152'
REMOTE_SHA256='6aa28d7efebcfa8fc746eeb24d915354db1445a259bafba73d8947ec54029de8'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=1:raise ValueError('this frozen Graph300 launcher takes no arguments')
 if RECEIPT.exists()or RECEIPT.is_symlink():raise FileExistsError('preserve existing receipt; no automatic rerun')
 if sha(PROTOCOL)!=PROTOCOL_SHA256 or sha(REMOTE_SOURCE)!=REMOTE_SHA256:raise ValueError('frozen protocol/coordinator SHA mismatch')
 for leaf,expected in SOURCE_SHA256.items():
  path=VERSION/leaf
  if sha(path)!=expected:raise ValueError('frozen source SHA mismatch: '+leaf)
  ast.parse(path.read_bytes())
 remote_source=REMOTE_SOURCE.read_text();ast.parse(remote_source);started=time.monotonic()
 try:
  x=subprocess.run(SSH,input=remote_source,text=True,capture_output=True,timeout=960)
  value=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
 except subprocess.TimeoutExpired as error:
  def text(v):return v.decode(errors='replace')if isinstance(v,bytes)else(v or'')
  value=dict(returncode=None,stdout=text(error.stdout),stderr=text(error.stderr),wall_seconds=time.monotonic()-started,
   error='SSH wait timeout; actual remote state unknown, inspect existing bounded runner, no automatic retry')
 value.update(name=NAME,protocol_sha256=PROTOCOL_SHA256,remote_launch_sha256=REMOTE_SHA256,
  source_sha256=SOURCE_SHA256,remote_source_embedded_in_receipt=False)
 with RECEIPT.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
 summary=dict(name=NAME,returncode=value['returncode'],receipt=str(RECEIPT),wall_seconds=value['wall_seconds'])
 try:summary['remote_summary']=json.loads(value['stdout'].strip().splitlines()[-1])
 except (ValueError,IndexError):summary['remote_summary']=None
 print(json.dumps(summary,sort_keys=True),flush=True)
 return value['returncode']if value['returncode']is not None else 124
if __name__=='__main__':raise SystemExit(main())
