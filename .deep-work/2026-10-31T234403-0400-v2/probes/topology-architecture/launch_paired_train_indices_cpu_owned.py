"""Freeze or ROOT-execute one bounded TRAIN-only pair-index formation CPU job."""
import argparse,ast,base64,hashlib,json,subprocess,time
from pathlib import Path
import launch_value_bridge_cpu_owned as transport

ROOT=Path(__file__).resolve().parent
NAME='parameter-paired-train-indices-n8-stage3-seed0-v1'
SOURCE='generate_paired_train_indices.py'
SCHEMA='paired-minibatch-TRAIN-credit-indices-v1'
STATUS='CPU_PAIRED_TRAIN_INDICES_GENERATED_PASS'
INPUT_RELATIVE='parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-train.npz'
INPUT_SHA256='b950377e180c06fc1ae2137c47637f1327a1a9b0d3dfebd54260cec45b037609'
TRANSPORT_SHA256='1bff46e15baaa095a901e3aa5e0d771f3f183f083438daa88c79a131f0692cfc'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def put(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')

def template():
    source=transport.TEMPLATE.replace('check_value_bridge_cpu.py',SOURCE).replace("out/'check'","out/'generate'").replace("out/'check/result.json'","out/'generate/result.json'").replace('CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS',STATUS).replace("actual_P=(result or{}).get('P')","eligible_rows=(result or{}).get('eligible_rows')")
    marker="value=dict(name=c['name'],returncode=rc"
    guard="schedule=out/'generate/paired-stage3-indices.npz'\npassed=bool(passed and result.get('eligible_rows')==4096 and result.get('eligible_groups')==2048 and result.get('target_array_reads')==0 and result.get('same_pool_row_exposures_exact')is True and result.get('reserved_tail16_matches_original')is True and schedule.is_file()and not schedule.is_symlink()and sha(schedule)==result['schedule']['sha256'])\n"
    if source.count(marker)!=1:raise ValueError('exact trusted CPU coordinator marker changed')
    return source.replace(marker,guard+marker,1)

def prepare():
    version=ROOT/'versions'/('code-'+NAME);protocol=ROOT/(NAME+'-protocol.json')
    if version.exists()or protocol.exists():raise FileExistsError('new frozen namespace only; preserve attempts')
    if sha(Path(transport.__file__))!=TRANSPORT_SHA256:raise ValueError('unchanged trusted CPU transport differs')
    raw=(ROOT/SOURCE).read_bytes();ast.parse(raw);source_sha={SOURCE:hashlib.sha256(raw).hexdigest()}
    original=ROOT/'runs'/INPUT_RELATIVE
    if sha(original)!=INPUT_SHA256:raise ValueError('exact original TRAIN input changed')
    files={'stage3-train.npz':dict(remote_relative=INPUT_RELATIVE,sha256=INPUT_SHA256,bytes=original.stat().st_size)}
    manifest=dict(schema=SCHEMA,source_sha256=source_sha,input_sha256=INPUT_SHA256,input_files=files,target_array_reads_required=0,only_public_raw_members=['initial.npy','tokens.npy','lengths.npy'])
    config=dict(name=NAME,remote=transport.REMOTE,runner_SHA=transport.CPU,source_sha256=source_sha,files={SOURCE:base64.b64encode(raw).decode()},input_files=files,manifest=manifest)
    config=json.loads(json.dumps(config,sort_keys=True));remote_source=template().replace('CONFIG',repr(config),1);ast.parse(remote_source);version.mkdir()
    with(version/SOURCE).open('xb')as stream:stream.write(raw)
    (version/SOURCE).chmod(0o444)
    with(version/'remote-launch.source').open('x')as stream:stream.write(remote_source)
    (version/'remote-launch.source').chmod(0o444);put(version/'launch-config.json',config);(version/'launch-config.json').chmod(0o444)
    doc=dict(name=NAME,schema=SCHEMA,status='FROZEN_UNEXECUTED',before_execution=True,formal_Primary=False,source_sha256=source_sha,launcher_sha256=sha(Path(__file__)),trusted_transport_sha256=TRANSPORT_SHA256,input_files=files,only_input_members_opened=['initial.npy','tokens.npy','lengths.npy'],archive_hash_scope='whole raw archive compressed bytes checked, targets.npy never opened/decoded/indexed',group_key='(initial, tuple(active tokens[:-4]))',suffix_width=4,expected_groups=2048,expected_rows=4096,group_order='first occurrence while scanning original rows0..4095',member_order='ascending original TRAIN row index',group_validation='every group exactly2 rows, equal active length, different raw4token suffix; all4096 rows exactly once; no label filtering or excluded groups',pair_seed=[0,8,3,9027,4],same_pool_shuffle_seed=[0,8,3,9027,4,0],original_seed=[0,8,3,9027],RNG_order='default_rng SeedSequence(pair_seed): draw entire600x24 group matrix, then entire600x24 binary order matrix',sampling='24 uniform groups per step with replacement; both members appear adjacently in independently random order',output_leaf='paired-stage3-indices.npz',output_arrays={'groups':[2048,2],'pairdraws':[600,24],'pairorders':[600,24],'indices':[600,64],'indices_same_pool_shuffled':[600,64]},output_dtype='int64',same_pool_control='globalpermutation of paired first48 pool with independent fixedshuffleSeedSequence; assert4096row exposure bincount EXACTsame; tail16 sameoriginal',same_pool_causal_scope='sameglobalrowmultiset/labelinformation/updatebudget; grouping+temporalordering jointly vary, notpurecovarianceonlyproof',prospective_study='3models typed/GRU32/Transformer32 x2samplers paired/samepoolshuffled, each600fromsameOWNstage2;6tails, noHPsearch',first48='24 selected pairs, two raw current TRAIN rows each',last16='exact original9027 indices[:,48:] reserved/unused provenance only; NOT Replay128 draws and NOT additional current TRAIN rows',Replay128='each NN tail restores OWN stage2 replay/RNG; actual original600x16 buffer_indices remains separate and unchanged',target_array_reads=0,equal_label_pairs_not_excluded=True,eligible_groups_scope='all2048 eligible; random draw need not cover every group, no coverage-based redraw',NN_or_checkpoint_or_teacher_or_counterfactual_truth_loaded=False,VALID_or_TEST_read=False,NN_updates=0,SGD_updates=0,research_updates=0,outside_seconds=150,address_space_GiB=64,file_limit_MiB=100,runner_sha256=transport.CPU,no_new_permissions=True,CPU_only=True,no_accuracy_gate=True,costs='all raw-input/hash/NumPy import/grouping/RNG/verification/serialization/reload/CPU/RSS/index storage plus outside runner/coordinator; later NN tails separately charged, no repeated historical optimizer updates',scope='one prospective paired-minibatch plus same-pool shuffle causal diagnostic for same typed/GRU32/Transformer32 tails; no task/test/label changes or route oracle',remote_launch_sha256=sha(version/'remote-launch.source'))
    put(protocol,doc);protocol.chmod(0o444);version.chmod(0o555)
    print(json.dumps(dict(name=NAME,status='FROZEN_UNEXECUTED',source_sha256=source_sha,protocol_sha256=sha(protocol),remote_launch_sha256=doc['remote_launch_sha256'],ROOT_execute_command='python3 '+str(Path(__file__))+' --execute'),sort_keys=True))

def execute():
    version=ROOT/'versions'/('code-'+NAME);protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());config=json.loads((version/'launch-config.json').read_text());source=(version/'remote-launch.source').read_text()
    if sha(Path(__file__))!=protocol['launcher_sha256']or sha(Path(transport.__file__))!=TRANSPORT_SHA256 or sha(version/'remote-launch.source')!=protocol['remote_launch_sha256']or source!=template().replace('CONFIG',repr(config),1):raise ValueError('frozen launcher/transport/coordinator differs')
    for leaf,digest in config['source_sha256'].items():
        if sha(version/leaf)!=digest:raise ValueError('frozen generator source differs')
        ast.parse((version/leaf).read_bytes())
    receipt=ROOT/(NAME+'-coordinator-receipt.json')
    if receipt.exists()or receipt.is_symlink():raise FileExistsError('preserve actual receipt; no automatic retry')
    started=time.monotonic()
    try:
        result=subprocess.run(transport.SSH,input=source,text=True,capture_output=True,timeout=210);value=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,wall_seconds=time.monotonic()-started)
    except subprocess.TimeoutExpired as error:
        def as_text(v):return v.decode(errors='replace')if isinstance(v,bytes)else v or''
        value=dict(returncode=None,stdout=as_text(error.stdout),stderr=as_text(error.stderr),wall_seconds=time.monotonic()-started,error='SSHwaittimeout; inspect existing bounded job, no automatic retry')
    value.update(name=NAME,source_sha256=config['source_sha256'],protocol_sha256=sha(ROOT/(NAME+'-protocol.json')),remote_launch_sha256=sha(version/'remote-launch.source'),NN_updates=0,research_updates=0);put(receipt,value)
    print(json.dumps(dict(name=NAME,returncode=value['returncode'],wall_seconds=value['wall_seconds'],receipt=str(receipt)),sort_keys=True));raise SystemExit(value['returncode']if value['returncode']is not None else 124)

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:execute()
