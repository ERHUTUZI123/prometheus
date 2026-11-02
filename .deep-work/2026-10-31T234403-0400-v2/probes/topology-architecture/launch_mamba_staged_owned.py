"""Execute only the frozen official Mamba2 staged-smoke v2 ROOT coordinator.

This new launcher never edits a prior launcher/result or prepares a different
revision. Source/protocol were frozen before execution. It prints a short
receipt summary; no remote source/base64 payload is included in receipts.
"""
from __future__ import annotations
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-official-mamba2-staged-smoke-v2'
VERSION=ROOT/'versions'/('code-'+NAME)
PROTOCOL=ROOT/(NAME+'-protocol.json')
REMOTE_SOURCE=VERSION/'remote-launch.source'
RECEIPT=ROOT/(NAME+'-coordinator-receipt.json')
SOURCE_SHA256={'remote_runner_gpu_compiler_authorized.py': 'eb50f4a87cc9ec8774ae4c666f4b7f38c1bc8f7f7b952e384bc0909a99ea775a', 'official_mamba2_compiler_prologue.py': '584e4b1e6b8cb42e698178cb64fd491666baa68c06e714ff026d1796a83025c3', 'compiler_policy_checks.py': '11b7d86c151ec8fd5f1a49b8b32996dcf8f89279b4b0e687c48ff16931bcb231', 'official_mamba2_smoke.py': '590a68889c0e8660c2ad2e3b5dbd5d736ba791d3e7ba5b5bdf3f3e36b58e8824', 'validate_gpu_staged_smoke.py': '8f8c6f9f12c498676845e0a4d91a02ebfc7e55e1c0582201f877b05006fcd0b9'}
PROTOCOL_SHA256='b290219d555c453283f55089f41ec7388ce63815d0f1cc5298d0e453768d12ef'
REMOTE_SHA256='18d2f28eef8503f03f295400770efdf99c05bcec867876723cd44600ec170258'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=1:raise ValueError('this frozen v2 launcher takes no arguments')
 if RECEIPT.exists()or RECEIPT.is_symlink():raise FileExistsError('preserve existing v2 receipt; no automatic rerun')
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
