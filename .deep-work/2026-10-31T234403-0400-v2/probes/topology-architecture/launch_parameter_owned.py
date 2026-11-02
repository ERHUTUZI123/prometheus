"""Trusted bounded launcher for the frozen parameter-growth study (not sandbox)."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
SSH=['ssh','-o','BatchMode=yes','-p','24343','-i','/Users/xiaoyangliu/.ssh/runpod_prometheus','root@195.26.232.180','python3','-']
REMOTE='/workspace/prometheus-v2-m0-architecture'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scale',type=int,choices=(8,16,32),required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--data-only',action='store_true');p.add_argument('--arm',default='topology')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--consistency-refresh',action='store_true')
    p.add_argument('--curriculum',action='store_true')
    p.add_argument('--length-diagnostic',action='store_true')
    p.add_argument('--factor-diagnostic',action='store_true')
    p.add_argument('--external-baseline',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--hope-layers',type=int,default=1)
    p.add_argument('--revision',choices=('v1','v2','v3'),default='v1')
    p.add_argument('--stop-stage',type=int,default=5);p.add_argument('--resume')
    a=p.parse_args()
    if a.smoke and not a.external_baseline:p.error('smoke belongs to external baseline harness')
    if a.external_baseline and a.arm not in ('den','selfmod_cms'):p.error('explicit den/selfmod_cms external reconstruction required')
    suffix=(('resumed-' if a.resume else '')+f'through{a.stop_stage}' if a.stop_stage<5 else 'resumed' if a.resume else 'full')
    name=f'parameter-data-n{a.scale}-seed{a.seed}' if a.data_only else f'parameter-n{a.scale}-seed{a.seed}-{a.arm}-'+suffix
    if a.refresh:name=f'parameter-n{a.scale}-seed{a.seed}-{a.arm}-refresh-diagnostic-v1'
    if a.consistency_refresh:name+='-consistent'
    if a.curriculum:name+='-curriculum'
    if a.length_diagnostic:name=f'parameter-n{a.scale}-seed{a.seed}-transformer-length-diagnostic-v1'
    if a.factor_diagnostic:name=f'parameter-n{a.scale}-seed{a.seed}-factored-backend-diagnostic-v1'
    if a.external_baseline:
        tag=a.arm+str(a.width)+(f'-l{a.hope_layers}' if a.arm=='selfmod_cms' else '')
        name=f'parameter-n{a.scale}-seed{a.seed}-{tag}-'+('smoke-' if a.smoke else suffix+'-')+a.revision
    driver='parameter_efficiency_tasks.py' if a.data_only else 'diagnose_parameter_refresh.py' if a.refresh else 'parameter_efficiency_study.py'
    if a.length_diagnostic:driver='transformer_length_diagnostic.py'
    if a.factor_diagnostic:driver='check_factored_parameter_backend.py'
    if a.external_baseline:driver='parameter_external_baseline_study.py'
    files=([driver] if a.data_only else ['parameter_efficiency_study.py','parameter_native_models.py',
        'parameter_transformer_baselines.py','terminal_profile_formation.py','train.py','tasks.py',
        'torch_baselines.py','program_baseline_v2.py','topology_model.py','semantic_promotion_model.py',
        'workspace_refinement.py','continuous_native_model.py','continuous_monodromy.py','train_monodromy_extension.py',
        'integrate_native_repair.py','path_moment_cache.py','homotopy_lift_cache.py',
        'diagnose_projection_numeric.py','observed_operation_formation.py','history_operation_formation.py',
        'parameter_consistency_refresh.py','diagnose_parameter_refresh.py','transformer_length_diagnostic.py',
        'factored_monodromy_cache.py','check_factored_parameter_backend.py'])
    if a.external_baseline:files+=['parameter_external_baseline_study.py','parameter_den_baseline.py','parameter_hope_baseline.py']
    version=ROOT/'versions'/('code-'+name);version.mkdir(exist_ok=False)
    encoded={};hashes={}
    for f in files:
        data=(ROOT/f).read_bytes();ast.parse(data,filename=f);(version/f).write_bytes(data)
        encoded[f]=base64.b64encode(data).decode();hashes[f]=hashlib.sha256(data).hexdigest()
    protocol=dict(schema='parameter-efficiency-architecture-v1',before_execution=True,name=name,args=vars(a),
        MAIN='TOPOLOGY_USER_LOCKED',M0='NOT_PASSED',M3='ultimate',primary_metric='deltaP/delta_new_successfully_retained_capability',
        primary_baseline='Transformer 64/96,3layers,4heads,with/without replay',
        native_controls=['topology compact shared','ordinary constraint SAME compact/shared/formation','frozen backbone+private heads/ops','fully private dynamic modules','ordinary dense constraint'],
        same_information=f'raw final-label NPZ only;{16384 if a.scale==32 else 8192}TRAIN/512eachheldout,6tasks,6publicsymbols;no correct routes or hidden labels',
        gradient_budget_per_task=600,batch64=True,admission='VALID>=.95 only',
        evaluation_success='both deployed+neural IID>=.95,OOD>=.90,length16>=.90; stable previously admitted tasks counted',
        no_topology_parameter_advantage_assumed=True,ordinary_may_share_compress_reuse=True,
        all_initial_frozen_preallocated_parameters_counted=True,
        complete_costs='parameters,optimizer,buffers,reference/audit/word/replay/cache storage,discovery,training,maintenance,inference,reload; raw evidence preserved',
        source_sha256=hashes,external_seconds=120 if a.data_only else 900,internal_seconds=110 if a.data_only else 850)
    if a.refresh:protocol.update(gradient_budget_per_task=0,external_seconds=180,internal_seconds=150,scope='teacher-free actual NN consistency repair, P/K/Dfixed, no historical retention rewrite')
    if a.consistency_refresh:protocol['new_policy']='fixed600-step stage boundary teacher-free own-class/route consistency refresh; P/K/Dunchanged; oldv1failurespreserved; currentgraph only provisional target not prioradmission; same rule for ordinary/freeze/modules'
    if a.curriculum:protocol['prospective_baseline_repair']='same600updates/raw labels;first150 length<=1,next150<=3,last300 original fullpool; noTEST selection; originaluniform failurespreserved'
    if a.length_diagnostic:protocol.update(gradient_budget_per_task=0,external_seconds=120,internal_seconds=110,scope='actual saved TF64/96 stage0 checkpoints TRAIN/VALID stratified by observed input length; no training/noTEST')
    if a.factor_diagnostic:protocol.update(gradient_budget_per_task=0,external_seconds=120,internal_seconds=110,scope='N32 execution prerequisite: bounded multi-loop lift/quadrature/real NN reload; no task training or acquisition claim')
    if a.external_baseline:protocol.update(scope='DEN-RNN sequence adaptation or linear self-modifying CMS reconstruction; not full/official DEN or HOPE replication',smoke_not_benchmark=a.smoke,overlapping_GPU_workloads_possible=True)
    with (ROOT/(name+'-protocol.json')).open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
    cfg=dict(name=name,driver=driver,files=encoded,hashes=hashes,remote=REMOTE,args=vars(a))
    code='''import base64,hashlib,json,pathlib,shutil,subprocess,sys
c=CONFIG;a=c['args'];r=pathlib.Path(c['remote']);name=c['name']
code=r/('code-'+name);code.mkdir()
for n,b in c['files'].items():
 p=code/n;d=base64.b64decode(b);assert hashlib.sha256(d).hexdigest()==c['hashes'][n];p.write_bytes(d);p.chmod(0o444)
code.chmod(0o555);out=r/'outputs'/name
if a['data_only'] or a['factor_diagnostic']:
 runner=r/'remote_runner_cpu_v2.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='c9f00f8e85f68d99e76aa4203efe77d08ac9a6ea25536631cf592b9b8c711335'
 cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','120','--address-space-gib','64','--file-limit-mib','100','--cpu-only','--','--out',str(out/('learn' if a['factor_diagnostic'] else 'data')),'--max-seconds','110']
 if a['data_only']:cmd+=['--scale',str(a['scale']),'--seed',str(a['seed'])]
else:
 inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs');inputs.mkdir()
 shutil.copytree(r/'outputs'/('parameter-data-n'+str(a['scale'])+'-seed'+str(a['seed']))/'data'/'datasets',inputs/'datasets')
 if a['resume']:shutil.copyfile(pathlib.Path(a['resume']),inputs/'resume.pt')
 if a['length_diagnostic']:
    for width in [64,96]:shutil.copyfile(r/'outputs'/('parameter-n'+str(a['scale'])+'-seed'+str(a['seed'])+'-transformer'+str(width)+'-full')/'learn'/('transformer'+str(width)+'-stage0-step600.pt'),inputs/('tf'+str(width)+'.pt'))
 for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
 inputs.chmod(0o555)
 runner=r/'remote_runner_gpu_authorized.py';assert hashlib.sha256(runner.read_bytes()).hexdigest()=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
 cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--out',str(out/'learn'),'--datasets',str(inputs/'datasets'),'--scale',str(a['scale']),'--seed',str(a['seed']),'--arm',a['arm'],'--steps','600','--stop-after-stage',str(a['stop_stage']),'--max-seconds','850','--device','cuda:0']
 if a['refresh']:
    cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','180','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--out',str(out/'learn'),'--datasets',str(inputs/'datasets'),'--checkpoint',str(inputs/'resume.pt'),'--max-seconds','150']
 elif a['resume']:cmd+=['--resume',str(inputs/'resume.pt')]
 if a['consistency_refresh']:cmd+=['--consistency-refresh']
 if a['curriculum']:cmd+=['--curriculum']
 if a['length_diagnostic']:
    cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','120','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--out',str(out/'learn'),'--datasets',str(inputs/'datasets'),'--checkpoint',str(inputs/'tf64.pt'),str(inputs/'tf96.pt'),'--max-seconds','110','--device','cuda:0']
 if a['external_baseline']:
    cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--out',str(out/'learn'),'--datasets',str(inputs/'datasets'),'--scale',str(a['scale']),'--seed',str(a['seed']),'--arm',a['arm'],'--width',str(a['width']),'--hope-layers',str(a['hope_layers']),'--stages',str(a['stop_stage']+1),'--max-seconds','850','--device','cuda:0','--resource-overlap-note','Concurrent authorized parameter-scaling runs may overlap; wall times not isolated throughput ranks']
    if a['smoke']:cmd+=['--smoke']
    if a['resume']:cmd+=['--resume',str(inputs/'resume.pt')]
    if a['curriculum']:cmd+=['--curriculum']
print(json.dumps({'event':'frozen_parameter_study_launch','command':cmd}),flush=True)
x=subprocess.run(cmd,capture_output=True,text=True,timeout=930);print(x.stdout);print(x.stderr,file=sys.stderr)
if x.returncode:print((out/'stderr.log').read_text(errors='replace')[-5000:],file=sys.stderr)
raise SystemExit(x.returncode)
'''.replace('CONFIG',repr(cfg),1)
    ast.parse(code,filename='owned_remote_launch')
    result=subprocess.run(SSH,input=code,text=True,capture_output=True,timeout=960)
    with (ROOT/(name+'-coordinator-receipt.json')).open('x') as f:json.dump(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),f,indent=2)
    print(result.stdout,flush=True);print(result.stderr,flush=True);raise SystemExit(result.returncode)


if __name__=='__main__':main()
