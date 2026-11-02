"""Bounded CPU admission of both OWN baseline parent adapters, discarded fixtures."""
import argparse,copy,json,time,resource,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import paired_baseline_resume as core
import owned_parent_driver as old

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);start=time.perf_counter();cpu=time.process_time();device=torch.device('cpu');manifest=core.load_manifest(a.inputs);reports={}
 for arm in core.ARMS:
  hashes=[];proofs=[];synthetic=None
  for sampling in('paired','same_pool_shuffle'):
   model,opt,args,replay,used,train,payload,proof=core.restore(a.inputs,manifest,arm,device);growth,initial=core.grow(model,opt,args,device);assert sum(p.numel()for p in model.parameters())==core.ARMS[arm][2];indices,draws=core.schedules(a.inputs,manifest,replay,sampling);assert replay.draws==19200;hashes.append(initial);proofs.append(proof)
   if sampling=='paired':
    model.train();fixture=dict(initial=torch.tensor([0,1,2,3]),tokens=torch.tensor([[1,3,0,0],[2,4,3,0],[7,1,3,9],[8,2,4,10]]),lengths=torch.tensor([2,3,4,4]));target=torch.tensor([0,1,2,3]);opt.zero_grad(set_to_none=True);out=model(**fixture);loss=F.cross_entropy(out,target);loss.backward();assert loss.ndim==0 and bool(torch.isfinite(loss))and all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();synthetic=dict(loss=float(loss.detach()),all_parameter_gradients_present_finite=True,synthetic_optimizer_steps=1)
    saved=core.saved_payload(model,opt,replay,arm,sampling,manifest,used,0,{},False,synthetic=True);f=a.out/(arm+'-synthetic-step1.pt')
    with f.open('xb')as stream:torch.save(saved,stream)
    loaded=torch.load(f,map_location='cpu',weights_only=False);other,otheropt,otherreplay=core.reload(loaded,device);model.eval();other.eval()
    with torch.no_grad():assert torch.equal(model(**fixture),other(**fixture))
    core.restore_rng(loaded['rng'],device);assert old.nested_state_equal(core.state_now(device),loaded['rng']);synthetic.update(exact_NN_Adam_RNG_replay_forward_reload=True,checkpoint=dict(file=f.name,bytes=f.stat().st_size,sha256=core.sha(f)))
   del model,opt,replay,payload
  assert hashes[0]==hashes[1];reports[arm]=dict(P15=core.ARMS[arm][1],P16=core.ARMS[arm][2],restore_proofs=proofs,CPU_post_growth_hashes=hashes[0],same_CPU_start_both_samplers=True,growth=growth,synthetic=synthetic,CPU_GPU_newrow_equality_not_claimed=True)
 result=dict(status='CPU_PAIRED_BASELINE_OWN_PARENT_RESUME_PASS',source_sha256=manifest['source_sha256'],checker_sha256=core.sha(Path(__file__)),parents=reports,both_OWN_stage2_NN_Adam_available_RNG_replay_restored=True,old_NN_updates_repeated=0,original_growth_preserved_old_forward_and_Adam=True,same_start_both_samplers=True,exact_original_Replay_schedule=True,discarded_synthetic_NN_never_GPU_parent=True,CPU_only_no_CUDA_calls=True,SGD_updates=0,research_updates=0,synthetic_optimizer_steps=2,wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,CPU_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),full_training_FLOPs=None,provider_effective_compute=None,scope='twoOWN parentCPUadapter/growth/syntheticF_B_Adam/reload checks only; CUDA RNG originalpayload retained butunavailable streamsnotrestored, noqualitythreshold')
 with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(status=result['status'],research_updates=0)))
if __name__=='__main__':main()
