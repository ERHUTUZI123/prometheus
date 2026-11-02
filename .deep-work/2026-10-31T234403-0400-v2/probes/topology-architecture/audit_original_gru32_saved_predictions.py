"""ONE fixed original GRU32 NN: VALID and three TEST raw outputs, zero SGD.

Exact unmodified cc1 TerminalGRU and batch/packed backend. Each split receives
one soft forward per row; native aliases this same ordinary neural function.
All four logits/logp files are written before any target array is opened.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import owned_parent_driver as old
SPLITS=('valid','test_iid','test_long','test_unseen')
PARENT_SCHEMA='terminal-gru32-fourstage-training-v1'
STATUS='ORIGINAL_GRU32_FROZEN_4SPLIT_RAW_EXTRACTION_COMPLETE'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
    with p.open('x')as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def array_bytes(x):
    if isinstance(x,np.ndarray):return x.nbytes
    if isinstance(x,dict):return sum(array_bytes(v)for v in x.values())
    if isinstance(x,(tuple,list)):return sum(array_bytes(v)for v in x)
    return 0
def resources(model,device):
    return dict(parameters=sum(p.numel()for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),
        trainable_parameters=0,frozen_parameters=sum(p.numel()for p in model.parameters()),optimizer_tensor_bytes=0,
        registered_buffer_bytes=sum(v.numel()*v.element_size()for v in model.buffers()),
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        cpu_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cpu=time.process_time();records=[]
    def check():
        if time.perf_counter()-start>120:raise TimeoutError('120s scientific bound inside original900s runner')
    try:
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3);device=torch.device('cuda:0')
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if manifest['schema']!='original-gru32-frozen-4split-raw-v1':raise ValueError('fixed owned extraction only')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('exact frozen source')
        spec=manifest['checkpoint'];cp=a.inputs/spec['leaf']
        if sha(cp)!=spec['sha256']or cp.stat().st_size!=spec['bytes']:raise ValueError('original finalNN changed')
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        if payload['schema']!=PARENT_SCHEMA or payload['stage']!=3 or payload['step']!=600 or not payload['stage_complete']or not payload['maintenance_complete']or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('OWN original final completedNN only')
        if payload['model_spec']!=dict(n_states=8,n_tokens=16,max_length=128,width=32,layers=2):raise ValueError('original GRU32 dimensions')
        optimizer_storage=old.tensor_bytes(payload['optimizer']);rng_storage=old.tensor_bytes(payload['rng'])+array_bytes(payload['rng']);replay_storage=old.tensor_bytes(payload['replay_state'])+array_bytes(payload['replay_state'])
        model=old.TerminalGRU(**payload['model_spec']).to(device);model.load_state_dict(payload['model_state'],strict=True)
        before={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        if sum(p.numel()for p in model.parameters())!=14024 or not old.tensor_state_equal(model.state_dict(),payload['model_state']):raise ValueError('actualNN/P equality')
        del payload
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        for split in SPLITS:
            check();tick=time.perf_counter();ct=time.process_time();ds=manifest['data'][split];path=a.inputs/ds['leaf']
            if sha(path)!=ds['sha256']:raise ValueError('fixed public split')
            with np.load(path,allow_pickle=False)as z:public={k:z[k]for k in('initial','tokens','lengths')}
            if public['tokens'].shape!=(512,128):raise ValueError('same512 rawrows')
            # Exact old batch helper needs a target member, which the forward
            # never consumes. These are synthetic zeros, not opened labels.
            data=dict(public,targets=np.zeros(512,dtype=np.int64));pieces=[];macs=0;calls=0
            with torch.no_grad():
                for first in range(0,512,64):
                    check();ids=np.arange(first,first+64);batch,length,active=old.batch(data,ids,device)
                    logits=model(batch['initial'],batch['tokens'],batch['lengths'])
                    if not bool(torch.isfinite(logits).all()):raise FloatingPointError('nonfinite originalNN')
                    pieces.append(logits.cpu().numpy());macs+=model.proxy(64,length,active)['proxy_forward_macs'];calls+=1
            logits=np.concatenate(pieces);logp=torch.log_softmax(torch.from_numpy(logits),dim=-1).numpy();pred=logits.argmax(-1)
            raw=dict(logits=logits,soft_logp=logp,predictions=pred,row_index=np.arange(512,dtype=np.int64),initial=public['initial'],lengths=public['lengths'])
            leaf=split+'-original-gru32-raw-before-targets.npz';dest=a.out/leaf
            with dest.open('xb')as f:np.savez_compressed(f,**raw)
            records.append(dict(split=split,file=leaf,sha256=sha(dest),bytes=dest.stat().st_size,raw_array_payload_bytes=sum(v.nbytes for v in raw.values()),
                forward_examples=512,forward_calls=calls,forward_MAC_proxy=macs,targets_loaded=False,
                CPU_seconds=time.process_time()-ct,wall_seconds=time.perf_counter()-tick))
        old.synchronize(device)
        if not old.tensor_state_equal(model.state_dict(),before)or sha(cp)!=spec['sha256']:raise ValueError('NN bytes changed')
        put(a.out/'ALL_FOUR_RAW_SAVED_BEFORE_TARGETS.json',dict(all_four_saved=True,targets_loaded=False,records=records))
        scores={}
        for record in records:
            split=record['split']
            with np.load(a.inputs/manifest['data'][split]['leaf'],allow_pickle=False)as z:targets=z['targets']
            with np.load(a.out/record['file'],allow_pickle=False)as z:logp=z['soft_logp'];pred=z['predictions']
            correct=int((pred==targets).sum());acc=correct/512;nll=float(-logp[np.arange(512),targets].astype(np.float64).mean())
            expected=manifest['original_scores'][split]
            scores[split]=dict(correct=correct,rows=512,accuracy=acc,NLL=nll,original_accuracy=expected['accuracy'],original_NLL=expected['terminal_NLL'],
                accuracy_difference=acc-expected['accuracy'],NLL_difference=nll-expected['terminal_NLL'],
                reproduction_scope='descriptive frozenaggregate comparison; no quality gate/retry/tuning',native_alias_of_soft=True)
        result=dict(status=STATUS,source_sha256=manifest['source_sha256'],parent_source_sha256=manifest['parent_source_sha256'],
            checkpoint_sha256=spec['sha256'],checkpoint_bytes_loaded=spec['bytes'],schema=PARENT_SCHEMA,P=14024,SGD_updates=0,optimizer_steps=0,
            original_effective_training_updates=2400,training_updates_repeated=0,NN_unchanged=True,all_four_raw_saved_before_targets=True,
            target_OR_privileged_truth_feedback_to_NN=False,privileged_prefix_truth_loaded=False,records=records,scores=scores,
            forward_examples=2048,forward_calls=32,native_additional_forward_examples=0,live_optimizer_bytes=0,
            stored_optimizer_bytes_in_loaded_checkpoint=optimizer_storage,stored_RNG_tensor_bytes=rng_storage,stored_replay_tensor_bytes=replay_storage,
            resources=resources(model,device),wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,
            full_FLOPs=None,provider_effective_compute=None,scope='ONE originalNN4split extraction; no learner/taskdata/parameter/optimizer change; startup/batching/H2D/forward/D2H/storage paid; no hardrouting improvement claim')
        put(a.out/'result.json',result);print(json.dumps(dict(status=STATUS,P=14024,SGD_updates=0,forward_examples=2048,scores=scores)))
    except BaseException as error:put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),records=records));raise
if __name__=='__main__':main()
