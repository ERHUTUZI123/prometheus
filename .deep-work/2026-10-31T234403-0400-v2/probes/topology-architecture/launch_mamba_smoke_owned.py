"""Trusted launcher: unmodified official2.3.1 smoke in existing GPU isolation."""
import argparse,ast,base64,hashlib,json,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']


def main():
 p=argparse.ArgumentParser();p.add_argument('--revision',default='v1');a=p.parse_args()
 if not re.fullmatch(r'v[1-9][0-9]*',a.revision):raise ValueError('new vN revision required')
 name='parameter-official-mamba2-smoke-'+a.revision;version=ROOT/'versions'/('code-'+name)
 protocol_path=ROOT/(name+'-protocol.json');receipt_path=ROOT/(name+'-coordinator-receipt.json')
 if any(q.exists() for q in (version,ROOT/'runs'/name,protocol_path,receipt_path)):raise FileExistsError('new revision required; preserve old evidence')
 files=['official_mamba2_smoke.py',Path(__file__).name];encoded={};hashes={};version.mkdir()
 for leaf in files:
  raw=(ROOT/leaf).read_bytes();ast.parse(raw);(version/leaf).write_bytes(raw);encoded[leaf]=base64.b64encode(raw).decode();hashes[leaf]=hashlib.sha256(raw).hexdigest()
 protocol=dict(before_execution=True,name=name,source_sha256=hashes,scope='officialunmodified Mamba2 fullimport/onegradient/oneAdam/reload SMOKE, NOT trainedbaseline',
  package_root='/opt/prometheus-mamba231-r1',official_version='2.3.1',official_wheel_sha256='f9e375b47cee1ef27463c9249073f9065f0f8f0f98b32fabfbb8df4bf75e2b66',
  config=dict(d_model=32,d_state=16,expand=2,headdim=16,chunk_size=32,use_mem_eff_path=False),dtype='float32',input_shape=[2,16,32],
  outside_seconds=120,inside_seconds=60,gpu_fraction=.3,address_space_gib=512,file_limit_mib=100,
  execution='sameexisting GPUrunner and authorizedselfthreadexception; no prewarm/patch/fallback/seccompchange; compilerexecdenial recordedasfailure',
  costs='fullcoldimport/firstforward/backward/update/reload timings, actualparameter/optimizer/cachebytes, firstforward plus2reloadforwards counted',formal_Primary=False)
 with protocol_path.open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
 config=dict(name=name,files=encoded,hashes=hashes,remote=REMOTE)
 script='''import base64,hashlib,json,pathlib,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name
if code.exists() or out.exists():raise FileExistsError('newsmokerevisionrequired; nooverwrite')
runner=r/'remote_runner_gpu_authorized.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
if not pathlib.Path('/opt/prometheus-mamba231-r1/mamba_ssm/__init__.py').is_file():raise FileNotFoundError('officialpackageinstallationnotready; noinstallbylauncher')
code.mkdir()
for leaf,encoded in c['files'].items():
 data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==c['hashes'][leaf];p=code/leaf;p.write_bytes(data);p.chmod(0o444)
code.chmod(0o555)
cmd=[sys.executable,str(runner),'--script',str(code/'official_mamba2_smoke.py'),'--out',str(out),'--timeout','120','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--out',str(out/'smoke'),'--device','cuda:0','--max-seconds','60']
print(json.dumps({'event':'frozenofficialsmokelaunch','command':cmd}),flush=True);started=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=150);receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired as e:receipt=dict(returncode=None,error='coordinatorwaittimeout; inspectboundedrunner actualstate; noretryinferred',stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-started)
if out.exists():
 with (out/'coordinator-exit.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''.replace('CONFIG',repr(config),1)
 ast.parse(script);started=time.monotonic()
 try:
  x=subprocess.run(SSH,input=script,text=True,capture_output=True,timeout=180);receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
 except subprocess.TimeoutExpired as e:receipt=dict(returncode=None,error='SSHwaittimeout; remotestateunknown,norerun',stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-started)
 with receipt_path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)


if __name__=='__main__':main()
