"""ROOT-only bounded CPU render of frozen six-tail evidence. No installations."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;NAME='parameter-six-paired-tail-figure-v1';LEAF='plot_six_paired_tail_evidence.py';REMOTE='/workspace/prometheus-v2-m0-architecture';CPU='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(path,value):
 with path.open('x')as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
TEMPLATE=r'''import base64,hashlib,json,pathlib,shutil,subprocess,sys,time
c=CONFIG
r=pathlib.Path(c['remote']);code=r/('code-'+c['name']);out=r/'outputs'/c['name'];inputs=pathlib.Path('/opt/prometheus-'+c['name']+'-inputs');status=out.parent/(out.name+'.runner-status.json');receipt=out.parent/(out.name+'.coordinator-exit.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())if p.is_file()else None
if any(p.exists()or p.is_symlink()for p in(code,out,inputs,status,receipt)):raise FileExistsError('append-only newfigure namespace')
runner=r/'remote_runner_cpu_v2.py'
if sha(runner)!=c['runner_sha256']:raise ValueError('unchangedCPUrunner')
raw=base64.b64decode(c['source_b64'],validate=True)
if hashlib.sha256(raw).hexdigest()!=c['source_sha256']:raise ValueError('sourceSHA')
code.mkdir();p=code/c['leaf']
with p.open('xb')as f:f.write(raw)
p.chmod(0o444);code.chmod(0o555);inputs.mkdir()
for leaf,spec in c['data'].items():
 target=inputs/leaf
 if 'inline_b64'in spec:
  raw=base64.b64decode(spec['inline_b64'],validate=True)
  if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('inlineevidenceSHA: '+leaf)
  with target.open('xb')as f:f.write(raw)
 else:
  source=r/spec['remote_relative']
  if source.is_symlink()or not source.is_file()or source.stat().st_size>100*1024*1024 or sha(source)!=spec['sha256']:raise ValueError('actualfrozendataSHA: '+leaf)
  shutil.copyfile(source,target)
 target.chmod(0o444)
manifest=dict(schema='six-paired-tail-figure-inputs-v1',source_sha256=c['source_sha256'],data_sha256={leaf:spec['sha256']for leaf,spec in c['data'].items()})
p=inputs/'manifest.json'
with p.open('x')as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write('\n')
p.chmod(0o444);inputs.chmod(0o555);cmd=[sys.executable,str(runner),'--script',str(code/c['leaf']),'--out',str(out),'--timeout','150','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--inputs',str(inputs),'--out',str(out/'figures')];begin=time.monotonic()
try:x=subprocess.run(cmd,text=True,capture_output=True,timeout=180);rc=x.returncode;stdout=x.stdout;stderr=x.stderr;error=None
except subprocess.TimeoutExpired as e:rc=None;stdout=str(e.stdout or'');stderr=str(e.stderr or'');error='boundedCPUwaittimeout, inspectexistingstate; noautomaticretry'
for suffix,value in(('coordinator-stdout.log',stdout),('coordinator-stderr.log',stderr)):
 with(out.parent/(out.name+'.'+suffix)).open('x')as f:f.write(value)
rs=load(status);result=load(out/'figures/figure-result.json');passed=bool(rc==0 and rs and rs.get('status')=='completed'and result and result.get('status')=='RENDERED_SIX_PAIRED_TAIL_FIGURE'and result.get('source_sha256')==c['source_sha256']and result.get('new_NN_queries')==0 and result.get('SGD_updates')==0)
record=dict(name=c['name'],returncode=rc,protocol_completed=passed,error=error,whole_runner_wall_seconds=time.monotonic()-begin,source_sha256=c['source_sha256'],command=cmd,result_path=str(out/'figures/figure-result.json'),root_status_path=str(status),new_NN_queries=0,SGD_updates=0,dependency_scope='existing/opt/prometheus-analysis-python only; noinstall',CPU_only=True)
with receipt.open('x')as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(record,sort_keys=True));raise SystemExit(0 if passed else(rc if rc not in(None,0)else 1))
'''
def prepare():
 v=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
 if v.exists()or protocol.exists()or(ROOT/(NAME+'-coordinator-receipt.json')).exists():raise FileExistsError('newfreezeonly')
 proofpath=ROOT/'six-paired-tails-independent-raw-checks-v1.json';proof=json.loads(proofpath.read_text());assert proof['status']=='PASS_ALL_SIX_PAIRED_SAMEPOOL_SAVED_RAW';data={}
 def add(leaf,path,remote):data[leaf]=dict(local_source=str(path.resolve()),remote_relative=remote,sha256=sha(path),bytes=path.stat().st_size)
 for model in('gru32','transformer32','typed'):
  for sampler in('paired','same_pool_shuffle'):
   name=proof['models'][model][sampler]['run']
   for suffix in('result.json','records.jsonl'):
    f=ROOT/'runs'/name/'learn'/suffix;add(model+'-'+sampler+'-'+suffix,f,'outputs/'+str(f.relative_to(ROOT/'runs')))
 for leaf,name in [('summary.csv','six-paired-tails-summary-v1.csv'),('curves.csv','six-paired-tails-curves-v1.csv'),('six-evidence.json','six-paired-tails-independent-raw-checks-v1.json'),('history-inventory.json','paired-tail-stage2-parent-inventory-v1.json')]:
  f=ROOT/name;add(leaf,f,'local-frozen-evidence/'+leaf);data[leaf]['inline_b64']=base64.b64encode(f.read_bytes()).decode()
 source=(ROOT/LEAF).read_bytes();ast.parse(source);config=dict(name=NAME,remote=REMOTE,leaf=LEAF,runner_sha256=CPU,source_sha256=hashlib.sha256(source).hexdigest(),source_b64=base64.b64encode(source).decode(),data=data);remote=TEMPLATE.replace('CONFIG',repr(config),1);ast.parse(remote);v.mkdir();(v/LEAF).write_bytes(source);(v/LEAF).chmod(0o444);(v/'remote-launch.source').write_text(remote);(v/'remote-launch.source').chmod(0o444);put(v/'launch-config.json',config);(v/'launch-config.json').chmod(0o444)
 metadata={leaf:{k:spec[k]for k in('local_source','remote_relative','sha256','bytes')}for leaf,spec in data.items()};put(ROOT/(NAME+'-input-manifest.json'),metadata)
 doc=dict(name=NAME,before_execution=True,source_sha256=config['source_sha256'],data_sha256={k:x['sha256']for k,x in data.items()},inputs_bytes=sum(x['bytes']for x in data.values()),panels=['newVALIDlearning','meanoldVALID','P/train/science/cold/soft-eager-inference'],models=3,samplers=2,scope='onefrozendevelopmentseed0 diagnostic, noPrimary/seed-generalization/architectureadvantage; groupingANDtemporalorderingcontrast; oldcompetencevariesacrossfamilies',outputs=['PNG','SVG','PDF','receipt'],transient_cache='perrunownedMPLCONFIGDIR, formed/charged/removed',existing_dependencies='/opt/prometheus-analysis-python',new_NN_queries=0,SGD_updates=0,no_installation=True,runner_sha256=CPU,outside_seconds=150,AS_GiB=64,file_limit_MiB=100,remote_launch_sha256=sha(v/'remote-launch.source'),local_evidence_embedded=4,ROOT_upload_required=False)
 put(protocol,doc);protocol.chmod(0o444);v.chmod(0o555);print(json.dumps(dict(status='FROZEN_UNEXECUTED',name=NAME,source_sha256=config['source_sha256'],protocol_sha256=sha(protocol),remote_sha256=doc['remote_launch_sha256'],inputs_bytes=doc['inputs_bytes'],ROOT_upload_files=0)))
def execute():
 v=ROOT/'versions'/('code-'+NAME);config=json.loads((v/'launch-config.json').read_text());p=json.loads((ROOT/(NAME+'-protocol.json')).read_text());source=(v/'remote-launch.source').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='c'for t in n.targets));literal=ast.literal_eval(node.value)
 if literal!=config or source!=TEMPLATE.replace('CONFIG',repr(literal),1)or sha(v/'remote-launch.source')!=p['remote_launch_sha256']or sha(v/LEAF)!=config['source_sha256']:raise ValueError('frozen source/config changed')
 receipt=ROOT/(NAME+'-coordinator-receipt.json')
 if receipt.exists():raise FileExistsError('oneboundedrenderattemptonly')
 begin=time.monotonic()
 try:x=subprocess.run(SSH,input=source,text=True,capture_output=True,timeout=210);r=dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,wall_seconds=time.monotonic()-begin)
 except subprocess.TimeoutExpired as e:r=dict(returncode=None,stdout=str(e.stdout),stderr=str(e.stderr),wall_seconds=time.monotonic()-begin,error='SSHwaittimeout; inspectexistingboundedprocess')
 r['remote_launch_sha256']=sha(v/'remote-launch.source');put(receipt,r);print(json.dumps(dict(name=NAME,returncode=r['returncode'],wall_seconds=r['wall_seconds'])));raise SystemExit(r['returncode']if r['returncode']is not None else 124)
if __name__=='__main__':
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args();prepare()if a.prepare else execute()
