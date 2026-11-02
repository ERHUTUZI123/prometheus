"""Execute only the frozen continuous-Block fusion GPU audit ROOT coordinator.

This new launcher never edits a prior launcher/result or prepares a different
revision. Source/protocol were frozen before execution. It prints a short
receipt summary; no remote source/base64 payload is included in receipts.
"""
from __future__ import annotations
import ast,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NAME='parameter-continuous-fusion-gpu-audit-v1'
VERSION=ROOT/'versions'/('code-'+NAME)
PROTOCOL=ROOT/(NAME+'-protocol.json')
REMOTE_SOURCE=VERSION/'remote-launch.source'
RECEIPT=ROOT/(NAME+'-coordinator-receipt.json')
SOURCE_SHA256={'fused_continuous_block_fp32.py': '5cf6929bb0da78d8a7dc65cbdfe7d9dcaf0ba9d182845564f3f36c06e6f88d29', 'continuous_block_fusion_gpu_prologue.py': 'e8076717774d15d1451b33650586834918d5bcab8ce486c73c8ed0e795be57a2', 'remote_runner_gpu_continuous_fusion_audit.py': '996eb63e7e7c555eddb12f82bdcf38da9a727f9ad13969d60f6f963ec77fdb75', 'audit_continuous_fusion_gpu.py': '3c6264c273b719c0a8b4c1dcb41e7820d2558bcb2afb8df2a72d4c1e6c0f069e', 'validate_continuous_fusion_gpu.py': 'f4a8d009dc238b0f7d1950528120bf50ae4b07ab92a9dd7a21b4856844f0b93d', 'compiler_policy_checks.py': '11b7d86c151ec8fd5f1a49b8b32996dcf8f89279b4b0e687c48ff16931bcb231', 'audit_cuda_graph_inference.py': '3e82598a157a5c41f47d7c3e0b7ddf417899526a4027146d1f2d1dac7dae29b1', 'continuous_primitive_block.py': '58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497', 'train_char_alltoken.py': '567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02'}
PROTOCOL_SHA256='96198f599fe86aa407d7112b9a26bbcef921ed4f7f318ce0c1b9010f7c2d81a0'
REMOTE_SHA256='fba9ae07febc7b62f95b9ad71f08b1a8b3d3af14c89b75d3a06ab977ae1696c9'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=1:raise ValueError('this frozen fusion GPU launcher takes no arguments')
 if RECEIPT.exists()or RECEIPT.is_symlink():raise FileExistsError('preserve existing fusion GPU receipt; no automatic rerun')
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
