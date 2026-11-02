"""Execute only the frozen official Mamba2 all-token AR baseline ROOT coordinator.

This new launcher never edits a prior launcher/result or prepares a different
revision. Source/protocol were frozen before execution. It prints a short
receipt summary; no remote source/base64 payload is included in receipts.
"""
from __future__ import annotations
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-char-alltoken-mamba2-300-v1'
VERSION=ROOT/'versions'/('code-'+NAME)
PROTOCOL=ROOT/(NAME+'-protocol.json')
REMOTE_SOURCE=VERSION/'remote-launch.source'
RECEIPT=ROOT/(NAME+'-coordinator-receipt.json')
SOURCE_SHA256={'compiler_policy_checks.py': '11b7d86c151ec8fd5f1a49b8b32996dcf8f89279b4b0e687c48ff16931bcb231', 'continuous_primitive_block.py': '58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497', 'official_mamba2_alltoken_prologue.py': '67b0567b81cf7e98c2b4a6c5873d062273e6713c2f92a71de15a5480ebf3605f', 'remote_runner_gpu_official_mamba_alltoken.py': 'b6eb2af5ac9797d57c97645841945506e20e4b0f5e0d0877d9bb05db9fb5d864', 'train_char_alltoken.py': '567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02', 'train_char_official_mamba2_alltoken.py': 'ebaa704c59a6491827c03a63052a84bf9e7cbda0af9932d2e52764f52ba66371', 'validate_mamba2_alltoken.py': '0399687b26d2f1af02c4d2e5bc74bd0daa52c511f17838de51da171bd4ad7e83'}
PROTOCOL_SHA256='3da8bcdc57cca17bc8fe6e431d382bea42276499a8652609776ce35b6ea779a6'
REMOTE_SHA256='f120a53c7eac4308b9842a10b1ef3b364025159ac0f0110c62ebbaec6fc4f32a'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=1:raise ValueError('this frozen all-token baseline launcher takes no arguments')
 if RECEIPT.exists()or RECEIPT.is_symlink():raise FileExistsError('preserve existing baseline receipt; no automatic rerun')
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
