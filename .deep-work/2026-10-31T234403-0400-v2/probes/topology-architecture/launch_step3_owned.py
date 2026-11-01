"""Trusted coordinator: freeze reviewed owned code, copy owned inputs, bounded run.

This is NOT experiment code or a sandbox; only the existing verified remote
runner executes experiment code. No background jobs, new credentials or installs.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REMOTE = '/workspace/prometheus-v2-m0-architecture'
SSH = ['ssh', '-o', 'BatchMode=yes', '-p', '24343', '-i',
       '/Users/xiaoyangliu/.ssh/runpod_prometheus', 'root@195.26.232.180', 'python3', '-']
COMMON = ('train.py tasks.py torch_baselines.py program_baseline_v2.py topology_model.py '
          'semantic_promotion_model.py workspace_refinement.py continuous_native_model.py '
          'continuous_monodromy.py train_monodromy_extension.py integrate_native_repair.py '
          'path_moment_cache.py observed_operation_formation.py history_operation_formation.py '
          'diagnose_projection_numeric.py').split()
SPECS = {
    'baseline': ('n16-matched-baselines-v1', 'matched_baseline_continuation.py'),
    'acquisition': ('n16-acquisition-contribution-v1', 'acquisition_contribution_study.py'),
    'monodromy': ('n16-monodromy-contribution-v1', 'monodromy_contribution_study.py'),
    'growth': ('n16-homotopy-growth-cost-v1', 'growth_protection_cost_study.py'),
    'growth_resume': ('n16-homotopy-growth-cost-v1a', 'growth_protection_cost_study.py'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', choices=SPECS)
    a = p.parse_args()
    name, driver = SPECS[a.study]
    files = COMMON + [driver] + (['homotopy_lift_cache.py'] if a.study in ('growth','growth_resume') else [])
    encoded = {}
    hashes = {}
    version = ROOT / 'versions' / ('code-' + name)
    version.mkdir(exist_ok=False)
    for filename in files:
        data = (ROOT / filename).read_bytes()
        ast.parse(data, filename=filename)
        (version / filename).write_bytes(data)
        encoded[filename] = base64.b64encode(data).decode()
        hashes[filename] = hashlib.sha256(data).hexdigest()
    protocol = dict(schema='step3-contribution-validation-v1', study=a.study,
        frozen_before_execution=True, MAIN='TOPOLOGY human_only', M0='NOT_PASSED', M3='ultimate',
        primary_architecture_baseline='Transformer', source_sha256=hashes,
        source_readonly=True, new_output=REMOTE+'/outputs/'+name,
        limits=dict(external_seconds=900, internal_seconds=850, address_space_GiB=512,
                    file_MiB=100, gpu_allocator_fraction=.3),
        allowed_proc_exception='self task WRITE_FILE/TRUNCATE only; existing authorized runner',
        tests=(dict(arms=['transformer','replay','adapter'], source='actual stage1-step800 checkpoints',
                    task1_extra=200,task2=1000,batch64=True,lr=.003,
                    information='same original final-label datasets; no teacher operations',
                    comparison='gradient-budget match, formation/search costs additional; seed0 only')
               if a.study=='baseline' else
               dict(arms=['continue_only','native_recompile','one_step','history_full','history_no_explicit_reuse'],
                    source='same actual V8 stage2-step800',steps=200,seed=9010,batch64=True,
                    objective='terminalCE+.01geometry',old_only_protection=[0,1],current_admission='END only',
                    compare='TRAIN-only formation interventions; parameter/search resources measured')
               if a.study=='acquisition' else
               dict(arms=['legacy','certified','cached_direct'], actual_growth_checkpoints=3,
                    source='actual N16 learned stage0/1/2 sources; not fresh task acquisitions',
                    steps_per_checkpoint_per_arm=200, seeds=[9020,9021,9022], batch64=True,
                    objective='terminalCE+.01geometry', cache_entries=128,
                    current_and_old_admitted_binding_contracts_preserved=True,
                    native_monitor_every20=True, curves=[0,100,200],
                    independent_final_fresh_trajectory_audits=9,
                    targets='same200 parameter-update hashes, parameter growth, actual geometry maintenance, protection/cache/inference/total cost',
                    interpretation='ideal-semantic homotopy reuse; digital work repair not physical execution or energy advantage')
               if a.study in ('growth','growth_resume') else
               dict(arms=['monodromy_guard_on','direct_guard_on','monodromy_guard_off','direct_guard_off'],
                    source='same actual V8 stage2-step800',steps=200,seed=9011,batch64=True,
                    objective='terminalCE+.01geometry, all arms retain4omega',
                    scope='execution backend and binding guard factorial; not removal of all topology',
                    targets='raw proposal hashes/old violations/accepted updates/native-soft accuracy/full costs')))
    if a.study == 'growth_resume':
        protocol['engineering_recovery'] = dict(
            failed_run='n16-homotopy-growth-cost-v1', error='FileExistsError: duplicate append-only partial-summary name',
            completed_arms_not_rerun=[[0,'legacy'],[0,'certified']], remaining_arms=7,
            warm_cache_not_fabricated='pay new actual origin lifts and record recovery cost',
            scientific_state_transition_and_learning_rule_unchanged=True)
    protocol_path = ROOT / (name + '-protocol.json')
    with protocol_path.open('x') as f:
        json.dump(protocol, f, indent=2)
        f.write('\n')
    config = dict(name=name, driver=driver, study=a.study, files=encoded, hashes=hashes, remote=REMOTE)
    remote_code = '''import base64, hashlib, json, pathlib, shutil, subprocess, sys
c=CONFIG
r=pathlib.Path(c['remote']); name=c['name']
runner=r/'remote_runner_gpu_authorized.py'
assert hashlib.sha256(runner.read_bytes()).hexdigest()=='85e8ae4ffb4ba9cd187fcc53f2cbb3dcbd00eb5be10f7bba079768096f4a13d0'
code=r/('code-'+name); code.mkdir(exist_ok=False)
for n,b in c['files'].items():
    p=code/n; d=base64.b64decode(b); assert hashlib.sha256(d).hexdigest()==c['hashes'][n];p.write_bytes(d);p.chmod(0o444)
code.chmod(0o555)
inputs=pathlib.Path('/opt/prometheus-'+name+'-inputs');inputs.mkdir(exist_ok=False)
source=r/'outputs'/('continuous-primary-v7-n16-seed0-baselines' if c['study'] in ['baseline','growth','growth_resume'] else 'n16-repaired-stage2-v8')/'learn'
shutil.copytree(source/'datasets',inputs/'datasets')
if c['study'] in ['growth','growth_resume']:
    growthsources=[r/'outputs'/'continuous-primary-v7-n16-seed0'/'learn'/'checkpoints'/'topo-stage0-step800-consolidated.pt',
        r/'outputs'/'n16-observed-formation-v1'/'learn'/'topo-stage1-step1000-observed-formation-trained.pt',
        r/'outputs'/'n16-history-formation-v1'/'learn'/'topo-stage2-step1000-observed-formation-trained.pt']
    for t,p in enumerate(growthsources):shutil.copyfile(p,inputs/('checkpoint'+str(t)+'.pt'))
    if c['study']=='growth_resume':shutil.copyfile(r/'outputs'/'n16-homotopy-growth-cost-v1'/'learn'/'failure.json',inputs/'actual-v1-failure.json')
elif c['study']=='baseline':
    for n in ['transformer','replay','adapter']:
        shutil.copyfile(source/'checkpoints'/(n+'-stage1-step800-consolidated.pt'), inputs/(n+'.pt'))
    shutil.copyfile(source/'batch-indices-stage2.npz',inputs/'stage2-indices.npz')
else:
    shutil.copyfile(source/'checkpoints'/'topo-stage2-step800-consolidated.pt',inputs/'checkpoint.pt')
    shutil.copyfile(source/'records.jsonl',inputs/'records.jsonl')
for p in inputs.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
inputs.chmod(0o555)
out=r/'outputs'/name
cmd=[sys.executable,str(runner),'--script',str(code/c['driver']),'--out',str(out),'--timeout','900','--address-space-gib','512','--file-limit-mib','100','--gpu-fraction','.3','--cuda-init-before-seccomp','--','--out',str(out/'learn'),'--datasets',str(inputs/'datasets'),'--device','cuda:0']
if c['study'] in ['growth','growth_resume']:
    for t in range(3):cmd+=['--checkpoint'+str(t),str(inputs/('checkpoint'+str(t)+'.pt'))]
    cmd+=['--max-seconds','850']
    if c['study']=='growth_resume':cmd+=['--resume-study',str(inputs/'actual-v1-failure.json')]
elif c['study']=='baseline':
    for n in ['transformer','replay','adapter']:cmd+=['--'+n+'-checkpoint',str(inputs/(n+'.pt'))]
    cmd+=['--stage2-indices',str(inputs/'stage2-indices.npz'),'--seconds','850']
else:
    cmd+=['--checkpoint',str(inputs/'checkpoint.pt'),'--max-seconds','850']
    if c['study']=='acquisition':cmd+=['--records',str(inputs/'records.jsonl'),'--seed','9010']
print(json.dumps({'event':'frozen_launch','study':c['study'],'source_hashes':c['hashes'],'command':cmd}),flush=True)
result=subprocess.run(cmd,timeout=920,capture_output=True,text=True)
print(result.stdout,flush=True);print(result.stderr,file=sys.stderr,flush=True)
if result.returncode:
    for fn in ['stderr.log','stdout.log']:
        p=out/fn
        if p.exists():print(fn+':'+p.read_text(errors='replace')[-4000:],file=sys.stderr,flush=True)
raise SystemExit(result.returncode)
'''.replace('CONFIG', repr(config), 1)
    result = subprocess.run(SSH, input=remote_code, text=True, capture_output=True, timeout=950)
    receipt = dict(study=a.study, ssh_exit=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   protocol=str(protocol_path), version=str(version))
    record = ROOT / (name + '-coordinator-receipt.json')
    with record.open('x') as f: json.dump(receipt, f, indent=2)
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__': main()
