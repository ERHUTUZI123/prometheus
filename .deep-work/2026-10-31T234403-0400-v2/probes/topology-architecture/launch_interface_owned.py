"""Trusted coordinator for finite TODO2/3 CPU diagnostics, not a sandbox."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/workspace/prometheus-v2-m0-architecture'
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
RUNNER_SHA='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['conventional','reset','mean','pair'],required=True);p.add_argument('--revision',choices=['v1','v2','v3'],default='v1');p.add_argument('--gradient-samples',type=int,choices=[1,4],default=1);a=p.parse_args()
 name=({'conventional':'parameter-hybrid-conventional-controls-','reset':'parameter-factorized-reset-','mean':'parameter-factorized-reset-mean-','pair':'parameter-factorized-reset-pair-'}[a.kind])+a.revision
 driver={'conventional':'run_hybrid_conventional_controls.py','reset':'run_factorized_reset_diagnostic.py','mean':'mean_policy_reset_repair.py','pair':'mean_policy_reset_pair_repair.py'}[a.kind]
 files=['loop_boundary_learning.py','hybrid_loop_block.py',driver]+(['run_hybrid_loop_diagnostic.py'] if a.kind=='conventional' else ['factorized_reset_block.py'])
 input_rel=('parameter-factorized-reset-mean-v1/learn/topological-frozen.pt' if a.kind=='pair' else 'parameter-factorized-reset-v2/learn/topological-frozen.pt' if a.kind=='mean' else 'parameter-hybrid-loop-learning-v2/learn/'+('result.json' if a.kind=='conventional' else 'full_coupled/final.pt'))
 source=ROOT/'runs'/input_rel;input_hash=hashlib.sha256(source.read_bytes()).hexdigest()
 version=ROOT/'versions'/('code-'+name);version.mkdir(exist_ok=False)
 encoded={};hashes={}
 for f in files:
  data=(ROOT/f).read_bytes();ast.parse(data,filename=f);(version/f).write_bytes(data);encoded[f]=base64.b64encode(data).decode();hashes[f]=hashlib.sha256(data).hexdigest()
 scope='Same-information finite ordinary controls vs actual30P hybridV2' if a.kind=='conventional' else 'Frame-frozen noninvertible local interface learned from terminal samples; not shared-adaptation evidence'
 protocol={'schema':'topology-interface-development-v1','before_execution':True,'MAIN':'TOPOLOGY_USER_LOCKED','campaign_complete':False,'name':name,'kind':a.kind,'source_sha256':hashes,'input_sha256':input_hash,'input':input_rel,'internal_seconds':90,'outside_seconds':120,'device':'CPU','max_transactions':100,'scope':scope,'Primary_frozen':False,'training_feedback':'same terminal training rows; no desired routes or joint lifts supplied','selection':'TRAIN only; heldout after each arm freeze; priorreference only read after control freezes','gradient_samples':a.gradient_samples,'resources':'existingUID65534/Landlock4+/seccomp/offline/64GiBAS/100MiBfilelimits','costs':'all queries, discovery, NNupdates, rejected proposals, maintenance, cache, actual model and memory state; source run costs remain historical','acceptance':'experiment outcome does not by itself accept TODO2/3; both must pass before Primary freeze'}
 (ROOT/(name+'-protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
 control_rel=None;control_hash=None
 if a.kind in ('mean','pair') or (a.kind=='reset' and a.revision!='v1'):
  control_rel='parameter-factorized-reset-v1/learn/result.json';control_hash=hashlib.sha256((ROOT/'runs'/control_rel).read_bytes()).hexdigest()
  protocol.update(targeted_repair='Four independent gradients averaged pertransaction to diagnose sparse one-coordinate credit variance; same100transaction cap, extra sampling cost charged',reused_control_result=control_rel,reused_control_sha256=control_hash,no_completed_control_rerun=True)
  if a.kind=='mean':protocol.update(targeted_repair='Resume actual failedV2 NN; alternate24 paid mean-class-neighbor auditions with caller gradient of same meanCE; no stochastic-surrogate objective or programdonation; <=50pairs100transactions',gradient_samples=0,scope='TODO3 actual failed NN continuation, no parameter growth or core changes; ordinary mean policy optimization correspondence explicit')
  if a.kind=='pair':protocol.update(targeted_repair='Resume actual mean-v1 partial NN; scan24 singles then276 pairs only on single obstruction; strictmeanCE improvement; at most3pairs; stoponpair obstruction; no targetprogram supplied',gradient_samples=0,scope='TODO3 actual failed/partial NN continuation; existing86P unchanged, no strong-baseline advantage claim')
  (ROOT/(name+'-protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
 cfg=dict(name=name,remote=REMOTE,files=encoded,hashes=hashes,driver=driver,kind=a.kind,input_rel=input_rel,input_hash=input_hash,runner_sha=RUNNER_SHA,gradient_samples=a.gradient_samples,control_rel=control_rel,control_hash=control_hash)
 code='''import base64,hashlib,json,pathlib,subprocess,sys,time
c=CONFIG;r=pathlib.Path(c['remote']);name=c['name'];code=r/('code-'+name);code.mkdir()
for f,b in c['files'].items():
 d=base64.b64decode(b);assert hashlib.sha256(d).hexdigest()==c['hashes'][f];p=code/f;p.write_bytes(d);p.chmod(0o444)
code.chmod(0o555)
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs');inputs.mkdir();d=(r/'outputs'/c['input_rel']).read_bytes();assert hashlib.sha256(d).hexdigest()==c['input_hash'];ip=inputs/('reference.json' if c['kind']=='conventional' else 'core.pt');ip.write_bytes(d);ip.chmod(0o444);inputs.chmod(0o555)
if c['control_rel']:
 d=(r/'outputs'/c['control_rel']).read_bytes();assert hashlib.sha256(d).hexdigest()==c['control_hash'];cp=inputs/'previous-controls.json';inputs.chmod(0o755);cp.write_bytes(d);cp.chmod(0o444);inputs.chmod(0o555)
runner=r/'remote_runner_cpu_v2.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()==c['runner_sha'];out=r/'outputs'/name
cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out/'learn'),'--seconds','90']
if c['kind']=='conventional':cmd+=['--max-updates','100','--reference-result',str(ip)]
elif c['kind']=='reset':cmd+=['--max-updates','100','--checkpoint',str(ip),'--seed','0','--gradient-samples',str(c['gradient_samples'])]
elif c['kind']=='mean':cmd+=['--max-pairs','50','--max-transactions','100','--checkpoint',str(ip)]
else:cmd+=['--max-edit-depth','2','--max-pairs','3','--max-transactions','100','--checkpoint',str(ip)]
if c['control_rel']:cmd+=['--controls-result',str(cp)]
print(json.dumps({'event':'frozen_interface_launch','command':cmd}),flush=True);started=time.monotonic()
try:
 x=subprocess.run(cmd,capture_output=True,text=True,timeout=150);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-started};print(x.stdout,flush=True);print(x.stderr,file=sys.stderr,flush=True)
except subprocess.TimeoutExpired as e:
 receipt={'returncode':None,'transport':'coordinator_wait_timeout','wall_seconds':time.monotonic()-started,'stdout':str(e.stdout),'stderr':str(e.stderr)}
if out.exists():(out/'coordinator-exit.json').write_text(json.dumps(receipt,indent=2)+'\\n')
raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
'''.replace('CONFIG',repr(cfg),1)
 ast.parse(code);began=time.monotonic()
 try:
  x=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=180);receipt={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'wall_seconds':time.monotonic()-began}
 except subprocess.TimeoutExpired as e:
  receipt={'returncode':None,'error':'SSH coordinator timeout; remote exit is unknown','stdout':str(e.stdout),'stderr':str(e.stderr),'wall_seconds':time.monotonic()-began}
 (ROOT/(name+'-coordinator-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(receipt['returncode'] if receipt['returncode'] is not None else 124)
if __name__=='__main__':main()
