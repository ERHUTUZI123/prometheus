"""Bounded CPU interface/causality/reload check; zero SGD, TRAIN inputs only."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
from train_char_factorized_alltoken import AllTokenLM

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tensors_equal(left,right):
 return left.keys()==right.keys()and all(torch.equal(left[k].cpu(),right[k].cpu())for k in left)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=90);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(0)
 def bound():
  if time.perf_counter()-tick>a.max_seconds:raise TimeoutError('CPU90s interface diagnostic bound')
 manifest=json.loads((a.inputs/'manifest.json').read_text())
 for leaf,digest in manifest['source_sha256'].items():
  if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('exact frozen interface-check source required')
 for leaf,spec in manifest['data_files'].items():
  if sha(a.inputs/leaf)!=spec['sha256']:raise ValueError('exact original TRAIN/metadata bytes required')
 meta=json.loads((a.inputs/'metadata.json').read_text());v=meta['n_tokens']
 if v!=100 or meta['schema']!='char-alltoken-ar-development-v1':raise ValueError('same original vocabulary100 required')
 with np.load(a.inputs/'train.npz',allow_pickle=False)as z:
  # Do not decompress TRAIN target array; no terminal/all-token label read.
  initial=torch.as_tensor(z['initial'][:2],dtype=torch.long);tokens=torch.as_tensor(z['tokens'][:2,:8].copy(),dtype=torch.long)
 lengths=torch.full((2,),8,dtype=torch.long)
 model=AllTokenLM('dynamic',v);cfg=model.body.configuration();P=sum(x.numel()for x in model.parameters())
 if P!=20721 or any(cfg[k]!=expected for k,expected in dict(internal_states=32,state_dim=32,memory_dim=32,controller_dim=16,carry_gate_groups=8,n_primitives=4,primitive_rank=8,memory_slots=8,key_dim=16).items()):raise ValueError('explicit H32/c16/K4/r8/M8/key16/G8 architecture differs')
 before={k:x.detach().clone()for k,x in model.state_dict().items()};model.eval();checks={}
 with torch.no_grad():
  for mode in('soft','native'):
   bound();out=model(initial,tokens,lengths,mode)
   if out.shape!=(2,8,100)or not bool(torch.isfinite(out).all())or float(torch.logsumexp(out,-1).abs().max())>1e-5:raise ValueError('normalized full all-token output interface failed')
   changed=tokens.clone();changed[:,4:]=(changed[:,4:]+1)%v
   alternate=model(initial,changed,lengths,mode);causal=float((out[:,:4]-alternate[:,:4]).abs().max())
   if causal>2e-6:raise ValueError('future input leakage')
   carry=model.body.initial_carry(initial);parts=[]
   for step in range(8):
    logp,carry,_=model.body.step_embedded(carry,model.body.token_embedding(tokens[:,step]),execution=mode,output='logits');parts.append(logp)
   streamed=torch.stack(parts,1);difference=float((out-streamed).abs().max())
   if not torch.allclose(out,streamed,atol=2e-6,rtol=2e-5):raise ValueError('same core step/forward_all interface mismatch')
   terminal=model.body(initial,tokens,lengths,execution=mode,output='logits')
   if not torch.allclose(terminal,out[:,-1],atol=2e-6,rtol=2e-5):raise ValueError('terminal/post-token output alignment mismatch')
   checks[mode]=dict(future_suffix_difference=causal,streamed_maximum_absolute_difference=difference,terminal_posttoken_aligned=True)
  saved=a.out/'actual-core-checkpoint.pt';torch.save(model.body.checkpoint(),saved);state=torch.get_rng_state().clone()
  restored=core.ContinuousPrimitiveBlock.from_checkpoint(torch.load(saved,map_location='cpu',weights_only=True));torch.set_rng_state(state)
  if not tensors_equal(model.body.state_dict(),restored.state_dict())or restored.checkpoint_version!=core.ContinuousPrimitiveBlock.checkpoint_version:raise ValueError('exact version/policy/NN core reload differs')
  reloaded=restored.forward_all(initial,tokens,lengths,output='logits')
  if not torch.equal(reloaded,model(initial,tokens,lengths)):raise ValueError('actual sameinput reload output differs')
  # Terminal proxy would undercharge63/64 readouts during training. Independent
  # explicit products below count all supplied outputs, even padding.
  soft=model.sequence_resource_report(2,8,active_steps=16);native=model.sequence_resource_report(2,8,active_steps=16,execution='native')
  soft_per_transition=3*16*(16+32+16)+32*16+8*16+8*32+(96*32+32*12)+4*8*(96+32)+4*32+32+32*(2*16+1)+8*16+2*8*(16+32)
  native_per_transition=soft_per_transition-3*8*(96+32)-4*32
  if soft['proxy_forward_macs']!=16*(soft_per_transition+32*100)or native['proxy_forward_macs']!=16*(native_per_transition+32*100):raise ValueError('independent alltoken readout MAC accounting differs')
  padded=model.sequence_resource_report(2,8,active_steps=12)
  if padded['proxy_forward_macs']!=12*soft_per_transition+16*32*100:raise ValueError('padding outputs not charged')
  logical=model.body.resource_report(1,8)
  if logical['persistent_example_state_scalars']!=432 or logical['persistent_example_state_payload_bytes']!=1728:raise ValueError('logical carry accounting differs')
 bound();model.train();loss=-model(initial,tokens,lengths)[...,0].mean();loss.backward()
 gradients=[x.grad for x in model.parameters()if x.grad is not None]
 if not gradients or any(not bool(torch.isfinite(x).all())for x in gradients):raise ValueError('nonfinite actual alltoken gradient')
 if not tensors_equal(before,model.state_dict()):raise ValueError('CPU checker changed NN tensors')
 batches=np.random.default_rng(np.random.SeedSequence([0,v,300,0x41524254])).integers(0,8192,size=(300,32),dtype=np.int64)
 batch_file=a.out/'common-batch-indices.npz'
 with batch_file.open('xb')as stream:np.savez_compressed(stream,indices=batches)
 if sha(batch_file)!=manifest['common_batch_indices_sha256']:raise ValueError('original300 common batches differ')
 for leaf,digest in manifest['source_sha256'].items():
  if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('source changed after interface check')
 result=dict(status='CPU_FACTORIZED_ALLTOKEN_INTERFACE_PASS',SGD_updates=0,model_values_unchanged=True,TRAIN_input_rows=2,TRAIN_label_values_read=0,synthetic_gradient_objective='negative mean output logprob class0; backward only, no optimizer/update',P=P,
  configuration=model.configuration(),source_sha256=manifest['source_sha256'],data_files=manifest['data_files'],checks=checks,
  actual_core_checkpoint_sha256=sha(saved),actual_core_checkpoint_bytes=saved.stat().st_size,core_reload_exact=True,
  common_batch_indices_sha256=sha(batch_file),soft_MAC=soft,native_MAC=native,padding_MAC=padded,logical_carry=logical,
  actual_forward_MAC_proxy=6*soft['proxy_forward_macs']+3*native['proxy_forward_macs']+model.body.resource_report(2,8)['proxy_forward_macs']+model.body.resource_report(2,8,execution='native')['proxy_forward_macs'],
  actual_forward_count_scope='soft6 alloutput-equivalent calls +1terminal; native3 alloutput-equivalent +1terminal; each2rows8tokens; synthetic backward paid in wall, no complete backward FLOPs proxy',
  parameter_bytes=sum(x.numel()*x.element_size()for x in model.parameters()),registered_buffer_bytes=sum(x.numel()*x.element_size()for x in model.buffers()),
  gradient_payload_bytes=sum(x.numel()*x.element_size()for x in gradients),CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024),
  driver_wall_seconds=time.perf_counter()-tick,formal_Primary=False,scope='interface/causality/reload/accounting only; not training/accuracy evidence; cold imports/runner/SSH separately charged')
 with(a.out/'result.json').open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],P=P,wall=result['driver_wall_seconds'])))
if __name__=='__main__':main()
