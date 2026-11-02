"""One TRAIN-only paired-minibatch credit diagnostic schedule, no NN or labels.

Only initial.npy, tokens.npy and lengths.npy are opened from the original
archive. Whole-file SHA verification also reads compressed archive bytes;
targets.npy is never opened, decoded, indexed or used to select groups.
"""
import time
MODULE_WALL_STARTED=time.perf_counter()
MODULE_CPU_STARTED=time.process_time()
import argparse,hashlib,json,os,resource,traceback,zipfile
from pathlib import Path
import numpy as np

SCHEMA='paired-minibatch-TRAIN-credit-indices-v1'
STATUS='CPU_PAIRED_TRAIN_INDICES_GENERATED_PASS'
SOURCE_LEAF='generate_paired_train_indices.py'
INPUT_LEAF='stage3-train.npz'
INPUT_SHA256='b950377e180c06fc1ae2137c47637f1327a1a9b0d3dfebd54260cec45b037609'
OUTPUT_LEAF='paired-stage3-indices.npz'
NROWS,NGROUPS,STEPS,PAIRS,BATCH,SUFFIX_WIDTH=4096,2048,600,24,64,4
ORIGINAL_SEED=[0,8,3,9027]
PAIR_SEED=[0,8,3,9027,4]
SHUFFLE_SEED=[0,8,3,9027,4,0]
RAW_MEMBERS=('initial.npy','tokens.npy','lengths.npy')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def put(path,value):
    with path.open('x')as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def receipt(array):
    return dict(shape=list(array.shape),dtype=array.dtype.str,payload_bytes=array.nbytes,sha256_C_payload=hashlib.sha256(array.tobytes(order='C')).hexdigest())

def generate(inputs,out):
    manifest=json.loads((inputs/'manifest.json').read_text())
    if manifest.get('schema')!=SCHEMA or manifest.get('input_sha256')!=INPUT_SHA256 or set(manifest.get('source_sha256',{}))!={SOURCE_LEAF}:raise ValueError('exact prospective single-source/TRAIN manifest required')
    if sha(Path(__file__))!=manifest['source_sha256'][SOURCE_LEAF]:raise ValueError('frozen generator source differs')
    source=inputs/INPUT_LEAF;tick=time.perf_counter();cpu=time.process_time()
    if source.is_symlink()or not source.is_file()or sha(source)!=INPUT_SHA256:raise ValueError('exact existing TRAIN4096 archive required')
    opened=[];raw={}
    with zipfile.ZipFile(source)as archive:
        for leaf in RAW_MEMBERS:
            with archive.open(leaf)as stream:
                raw[leaf[:-4]]=np.lib.format.read_array(stream,allow_pickle=False)
            opened.append(leaf)
    input_read=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,archive_bytes=source.stat().st_size,opened_members=opened,targets_array_reads=0,whole_archive_hash_includes_compressed_bytes_of_all_members=True)
    initial,tokens,lengths=(raw[key]for key in('initial','tokens','lengths'))
    input_read['raw_array_payload_bytes']=sum(array.nbytes for array in raw.values())
    if initial.shape!=(NROWS,)or lengths.shape!=(NROWS,)or tokens.shape!=(NROWS,128)or any(a.dtype.kind!='i'or a.dtype.itemsize!=8 for a in raw.values()):raise ValueError('exact original int64 raw array shapes required')
    if bool(np.any(lengths<SUFFIX_WIDTH))or bool(np.any(lengths>tokens.shape[1])):raise ValueError('active length outside fixed suffix grouping domain')
    tick=time.perf_counter();cpu=time.process_time();by_prefix={}
    # Stable first occurrence in ascending original TRAIN row order defines
    # group IDs; members of each group retain ascending original row order.
    for row in range(NROWS):
        length=int(lengths[row]);prefix=tuple(int(x)for x in tokens[row,:length-SUFFIX_WIDTH])
        key=(int(initial[row]),prefix)
        by_prefix.setdefault(key,[]).append(row)
    if len(by_prefix)!=NGROUPS or any(len(rows)!=2 for rows in by_prefix.values()):raise ValueError('all4096 TRAIN rows must partition into exactly2048 pairs; no filtering or fallback')
    groups=np.asarray(list(by_prefix.values()),dtype=np.int64)
    if not np.array_equal(np.sort(groups.reshape(-1)),np.arange(NROWS,dtype=np.int64)):raise ValueError('groups must cover every original TRAIN row exactly once')
    for left,right in groups:
        ll,rl=int(lengths[left]),int(lengths[right])
        if ll!=rl or np.array_equal(tokens[left,ll-SUFFIX_WIDTH:ll],tokens[right,rl-SUFFIX_WIDTH:rl]):raise ValueError('each pair needs equal active lengths and different raw suffixes')
    grouping=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,eligible_rows=NROWS,eligible_groups=NGROUPS,group_order='first occurrence while scanning original rows0..4095',member_order='ascending original TRAIN row index',group_key='(raw initial, tuple(active raw word[:-4]))',suffix_width=SUFFIX_WIDTH,label_filter=False,equal_label_pairs_not_excluded=True,equal_label_pair_count=None)
    tick=time.perf_counter();cpu=time.process_time()
    original=np.random.default_rng(np.random.SeedSequence(ORIGINAL_SEED)).integers(0,NROWS,size=(STEPS,BATCH),dtype=np.int64)
    pair_rng=np.random.default_rng(np.random.SeedSequence(PAIR_SEED))
    pairdraws=pair_rng.integers(0,NGROUPS,size=(STEPS,PAIRS),dtype=np.int64)
    pairorders=pair_rng.integers(0,2,size=(STEPS,PAIRS),dtype=np.int64)
    members=groups[pairdraws]
    ordered=np.take_along_axis(members,np.stack((pairorders,1-pairorders),axis=-1),axis=2)
    indices=original.copy();indices[:,:PAIRS*2]=ordered.reshape(STEPS,PAIRS*2)
    # Independent fixed global permutation of the SAME already selected pool.
    # No additional current examples or label reads enter the causal control.
    pool=indices[:,:48].reshape(-1)
    shuffled_pool=np.random.default_rng(np.random.SeedSequence(SHUFFLE_SEED)).permutation(pool)
    indices_same_pool_shuffled=original.copy();indices_same_pool_shuffled[:,:48]=shuffled_pool.reshape(STEPS,48)
    draw_formation=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,bit_generator=type(pair_rng.bit_generator).__name__,original_seed=ORIGINAL_SEED,pair_seed=PAIR_SEED,same_pool_shuffle_seed=SHUFFLE_SEED,draw_order='entire600x24 uniform groupdraw matrix, then entire600x24 binary pairorder matrix; independent global pool permutation after pairing',group_sampling='uniform with replacement',pairorder='0 retains original member order;1 reverses it',pair_draw_positions=STEPS*PAIRS,paired_current_row_positions=STEPS*PAIRS*2,reserved_unused_original_tail_positions=STEPS*(BATCH-PAIRS*2),same_pool_shuffle_temporary_payload_bytes=shuffled_pool.nbytes)
    tick=time.perf_counter();cpu=time.process_time()
    if not np.array_equal(indices[:,:48].reshape(STEPS,PAIRS,2),ordered)or not np.array_equal(indices[:,48:],original[:,48:]):raise ValueError('paired current48 or untouched unusedtail16 differs')
    hits=np.bincount(pairdraws.reshape(-1),minlength=NGROUPS);row_hits=np.bincount(indices[:,:48].reshape(-1),minlength=NROWS)
    if not np.array_equal(row_hits[groups[:,0]],hits)or not np.array_equal(row_hits[groups[:,1]],hits):raise ValueError('each selected pair must contribute both original rows once')
    shuffled_hits=np.bincount(indices_same_pool_shuffled[:,:48].reshape(-1),minlength=NROWS)
    if not np.array_equal(row_hits,shuffled_hits)or not np.array_equal(indices_same_pool_shuffled[:,48:],original[:,48:]):raise ValueError('same-pool control changed global row exposures or unusedtail16')
    verification=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,reserved_tail16_matches_original=True,same_pool_row_exposures_exact=True,same_pool_exposure_vector_sha256=hashlib.sha256(row_hits.astype(np.int64,copy=False).tobytes()).hexdigest(),selected_unique_groups=int(np.count_nonzero(hits)),selected_unique_current_rows=int(np.count_nonzero(row_hits)),group_hit_min=int(hits.min()),group_hit_max=int(hits.max()),both_pair_members_used_equally=True,all_groups_eligible_even_if_not_sampled=True)
    arrays=dict(groups=groups,pairdraws=pairdraws,pairorders=pairorders,indices=indices,indices_same_pool_shuffled=indices_same_pool_shuffled);tick=time.perf_counter();cpu=time.process_time()
    schedule=out/OUTPUT_LEAF
    with schedule.open('xb')as stream:np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    with np.load(schedule,allow_pickle=False)as saved:
        if set(saved.files)!=set(arrays)or any(not np.array_equal(saved[k],v)or saved[k].dtype!=np.dtype(np.int64)for k,v in arrays.items()):raise ValueError('actual schedule serialization/reload differs')
    serialization=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,compressed_bytes=schedule.stat().st_size,array_payload_bytes=sum(a.nbytes for a in arrays.values()),array_count=len(arrays),schedule_sha256=sha(schedule))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(schema=SCHEMA,status=STATUS,source_sha256=manifest['source_sha256'],input_sha256=INPUT_SHA256,input_leaf=INPUT_LEAF,eligible_rows=NROWS,eligible_groups=NGROUPS,pair_seed=PAIR_SEED,original_seed=ORIGINAL_SEED,same_pool_shuffle_seed=SHUFFLE_SEED,same_pool_row_exposures_exact=True,steps=STEPS,batch_index_width=BATCH,current_rows_per_step=48,replay_rows_per_step=16,replay_draws_generated=False,reserved_tail16_matches_original=True,unused_tail16_scope='format/provenance only; never Replay128 draws or extra current TRAIN examples',replay_scope='training must independently restore each OWN stage2 Replay128/RNG and reuse original600x16 buffer_indices; this generator never reads checkpoints/replay',target_array_reads=0,targets_decoded_or_indexed=False,teacher_or_counterfactual_truth_or_route_labels_read=False,VALID_or_TEST_read=False,label_based_filter=False,equal_label_pairs_not_excluded=True,NN_parameter_count=0,NN_updates=0,SGD_updates=0,research_updates=0,input_read=input_read,grouping=grouping,draw_formation=draw_formation,verification=verification,serialization=serialization,schedule=dict(file=OUTPUT_LEAF,sha256=serialization['schedule_sha256'],bytes=schedule.stat().st_size,arrays={k:receipt(v)for k,v in arrays.items()}),numpy_version=np.__version__,CPU_peak_RSS_bytes=int(peak)*1024,CPU_peak_RSS_scope='Linux ru_maxrss KiB; entire generator process including NumPy/input/grouping/formation/serialization',driver_wall_seconds=time.perf_counter()-MODULE_WALL_STARTED,driver_CPU_seconds=time.process_time()-MODULE_CPU_STARTED,full_training_FLOPs=None,provider_effective_compute=None,cost_scope='CPU generation only, all input/hash/group/NumPy RNG/verification/serialization/reload/storage charged; runner/coordinator cold costs separate; no NN or training executed',scientific_scope='one controlled paired-minibatch credit diagnostic, paired vs same-global-pool shuffled; changes grouping/batch covariance AND temporal order, not an isolated covariance-only proof; unchanged4096 TRAIN row universe/labels/tasks/tests, no accuracy gate or semantic parsing')
    put(out/'result.json',result)
    print(json.dumps(dict(status=STATUS,eligible_rows=NROWS,eligible_groups=NGROUPS,indices_shape=list(indices.shape),schedule_sha256=serialization['schedule_sha256'],target_array_reads=0,research_updates=0),sort_keys=True),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    try:generate(args.inputs,args.out)
    except BaseException as error:
        put(args.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),NN_updates=0,research_updates=0));raise

if __name__=='__main__':main()
