"""Trusted launch/freeze for NEW all-token causal character development."""
from __future__ import annotations
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
CORE_SHA='58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497'


def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true')
 p.add_argument('--arm',choices=['dynamic','static','detached','gru','transformer'],default='transformer')
 p.add_argument('--revision',choices=['v1','v2'],default='v1');a=p.parse_args()
 if a.prepare and a.revision!='v1':raise ValueError('samefixeddata-v1 required; no automaticresampling')
 name='parameter-char-alltoken-data-v1' if a.prepare else f'parameter-char-alltoken-{a.arm}-300-'+a.revision
 driver='prepare_char_alltoken_data.py' if a.prepare else 'train_char_alltoken.py'
 files=[driver] if a.prepare else [driver,'continuous_primitive_block.py']
 version=ROOT/'versions'/('code-'+name);protocol_path=ROOT/(name+'-protocol.json');receipt_path=ROOT/(name+'-coordinator-receipt.json')
 if any(path.exists() for path in (version,ROOT/'runs'/name,protocol_path,receipt_path)):raise FileExistsError('newrevisionrequired; nooverwriteofrawresults')
 encoded={};hashes={}
 for leaf in files:
  raw=(ROOT/leaf).read_bytes();ast.parse(raw);encoded[leaf]=base64.b64encode(raw).decode();hashes[leaf]=hashlib.sha256(raw).hexdigest()
 if not a.prepare and hashes['continuous_primitive_block.py']!=CORE_SHA:raise ValueError('freeze reviewed all-token58feb69core; do notsilentlysubstitute latestsource')
 version.mkdir()
 for leaf,value in encoded.items():(version/leaf).write_bytes(base64.b64decode(value))
 protocol=dict(before_execution=True,name=name,source_sha256=hashes,scope='NEW properall-token causalLM DEVELOPMENT; notformalPrimary/oldprefixexperiment',
  original_source='samepreparedPG100body contiguous90/5/5 beforeTRAINvocabularyfit; no newdownload',seed=0,
  feedback='allnextcharacters64positions/window, no taskID/routes/intermediateoperations; futureinputsmasked/causal',
  TRAIN_windows=8192,VALID_windows=512,TEST_windows=512,TRAIN_context=64,TEST_contexts=[64,128,256],
  primary_length_comparison='same512endpoints/same32768last64targetlabels; fullwindowNLLseparate',
  updates=0 if a.prepare else 300,batch_size=32,training_label_observations=0 if a.prepare else 614400,
  optimizer='Adam.003,clipnorm1',Block='sameH64K4rank8token16M8S1; allVx64initialencoderPcounted; softtraining/nativehardCALLinference',
  GRU='44wide2layers,zeroinitialhidden,learnedembedding/readout',Transformer='36wide3layers4heads,FF72,causalboolmask,noCLS,sinusoidalPE256,finalLayerNorm',
  costs='allP/optimizer/buffers/data/checkpoints; alltraining/causality/VALID/soft-nativeTEST/reloadwork; forwardMACproxiesnotfullFLOPs; actualconcurrencywallnotisolatedthroughput',
  test_policy='ALL300updates+finaldiskfreeze beforeTESTopened; no score-dependentconfigurationchanges',
  outside_seconds=120 if a.prepare else 900,inside_seconds=90 if a.prepare else 850,gpu_fraction=None if a.prepare else .3,
  isolation='unchangedexistingCPU/GPUrunner; no network/install/credentials/permissiongrowth',formal_Primary=False,old_prefix_results_not_pooled=True)
 with protocol_path.open('x') as stream:json.dump(protocol,stream,indent=2);stream.write('\n')
 config=dict(name=name,driver=driver,files=encoded,hashes=hashes,prepare=a.prepare,arm=a.arm,remote=REMOTE)
 remote='''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);out=r/'outputs'/name;inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs')
if code.exists() or out.exists() or (not c['prepare'] and inputs.exists()):raise FileExistsError('newnamespace required; nooverwrite')
code.mkdir()
for leaf,encoded in c['files'].items():
 raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==c['hashes'][leaf];p=code/leaf;p.write_bytes(raw);p.chmod(0o444)
code.chmod(0o555)
if c['prepare']:
 runner=r/'remote_runner_cpu_v2.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
 cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--prepared','/opt/prometheus-dev-data/prepared/shakespeare','--out',str(out/'data'),'--seed','0','--max-seconds','90']
else:
 source=r/'outputs'/'parameter-char-alltoken-data-v1'/'data'
 if not source.is_dir() or source.is_symlink():raise ValueError('ownedpreparedalltokendatarequired')
 inputs.mkdir();shutil.copytree(source,inputs/'data')
 for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
 inputs.chmod(0o555)
 runner=r/'remote_runner_gpu_authorized.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
 cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--data',str(inputs/'data'),'--out',str(out/'learn'),'--arm',c['arm'],'--seed','0','--device','cuda:0','--max-seconds','850']
print(json.dumps({'event':'frozenalltokenARlaunch','command':cmd}),flush=True);started=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=150 if c['prepare'] else 930);receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
except subprocess.TimeoutExpired as e:receipt=dict(returncode=None,error='coordinatorwaittimeout; inspectactualboundedrunner/noautoretry',stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-started)
if out.exists():
 with (out/'coordinator-exit.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''.replace('CONFIG',repr(config),1)
 ast.parse(remote);started=time.monotonic()
 try:
  x=subprocess.run(SSH,input=remote,text=True,capture_output=True,timeout=180 if a.prepare else 960);receipt=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-started)
 except subprocess.TimeoutExpired as e:receipt=dict(returncode=None,error='SSHwaittimeout; checkpointremoteactualstate before anyretry',stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-started)
 with receipt_path.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
 print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)


if __name__=='__main__':main()
