"""Bounded CPU layout/formula contract only; no GPU launch or training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import traceback

CHECKPOINT_SHA='1e0fc8ec0e063aa1c86f4c7b0d09d91d65c32338273323a4c6d10c2752993f80'
CHECKPOINT_BYTES=549107
REFERENCE_SHA='3e82598a157a5c41f47d7c3e0b7ddf417899526a4027146d1f2d1dac7dae29b1'
KERNEL_SHA='5cf6929bb0da78d8a7dc65cbdfe7d9dcaf0ba9d182845564f3f36c06e6f88d29'
ATOL,RTOL=2e-4,2e-5

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError('owned module unavailable')
 value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--max-seconds',type=float,default=90);a=p.parse_args()
 if not 1<=a.max_seconds<=90:raise ValueError('fixed CPU scientific <=90s')
 a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cpu_started=time.process_time();deadline=started+a.max_seconds
 result=dict(status='running_CPU_formula_contract',GPU_or_kernel_executed=False,optimizer_updates=0,labels_read=0,cases=[])
 try:
  import torch
  import numpy as np
  torch.set_num_threads(1)
  folder=Path(__file__).resolve().parent
  if a.checkpoint.is_symlink()or a.checkpoint.stat().st_size!=CHECKPOINT_BYTES or sha(a.checkpoint)!=CHECKPOINT_SHA:raise ValueError('exact original frozen dynamic checkpoint required')
  if sha(folder/'audit_cuda_graph_inference.py')!=REFERENCE_SHA or sha(folder/'fused_continuous_block_fp32.py')!=KERNEL_SHA:raise ValueError('fixed formula/reference source changed')
  reference=module(folder/'audit_cuda_graph_inference.py','_owned_fusion_cpu_reference')
  reference.torch=torch;reference.np=np;reference.F=torch.nn.functional
  payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
  args=argparse.Namespace(arm='dynamic',frozen_source=folder)
  model,layouts=reference.restore(args,payload,torch.device('cpu'))
  before=reference.state_hash(model.state_dict());fusion=module(folder/'fused_continuous_block_fp32.py','fused_continuous_block_fp32')
  params=dict(model.body.named_parameters());views=[];parameter_checks=[]
  for name,shape in fusion.PARAMETER_LAYOUT:
   if tuple(params[name].shape)!=shape or params[name].dtype!=torch.float32:raise ValueError('original parameter shape/dtype differs')
   views.append(params[name].detach().contiguous().reshape(-1))
  tick=time.perf_counter();packed=torch.cat(views);pack_seconds=time.perf_counter()-tick
  if packed.numel()!=41593 or sum(v.numel()for v in model.parameters())!=41593:raise RuntimeError('all original P required')
  for name,shape in fusion.PARAMETER_LAYOUT:
   left=packed[fusion.OFFSETS[name]:fusion.OFFSETS[name]+params[name].numel()].view(shape)
   if not torch.equal(left,params[name]):raise RuntimeError('packed parameter values changed: '+name)
   parameter_checks.append(dict(name=name,shape=shape,exact_values_equal=True,original_stride=list(params[name].stride())))
  def compare(left,right):
   absolute=(left-right).abs();error=float(absolute.max())
   report=dict(max_abs=error,allclose=bool(torch.allclose(left,right,atol=ATOL,rtol=RTOL)))
   if error>ATOL or not report['allclose']or not bool(torch.isfinite(right).all()):raise RuntimeError('fixed CPU formula closeness failed')
   return report
  generator=torch.Generator(device='cpu').manual_seed(0)
  with torch.no_grad():
   for batch in(1,2):
    for length in(1,2,8):
     if time.perf_counter()>deadline:raise TimeoutError('CPU formula bound')
     tokens=torch.randint(0,100,(batch,length),generator=generator,dtype=torch.long)
     initial=torch.zeros(batch,dtype=torch.long);lengths=torch.full((batch,),length,dtype=torch.long)
     original,carry=model.body.forward_all(initial,tokens,lengths,execution='soft',output='logits',return_carry=True)
     pure=reference.tensor_forward(model,'dynamic',initial,tokens,'soft')
     output,packed_carry=fusion.formula_reference_from_packed(tokens,packed)
     row=dict(batch=batch,length=length,public_vs_pure=compare(original,pure),pure_vs_packed_formula=compare(pure,output),
              carry={key:compare(carry[key],packed_carry[key])for key in('h','keys','values')},
              argmax_mismatches=int((pure.argmax(-1)!=output.argmax(-1)).sum()))
     if row['argmax_mismatches']:raise RuntimeError('fixed CPU argmax contract failed')
     result['cases'].append(row)
  if before!=reference.state_hash(model.state_dict()):raise RuntimeError('CPU contract changed frozen parameters/buffers')
  result.update(status='completed_CPU_layout_and_formula_contract',completed=True,original_P=41593,
   original_parameter_bytes=41593*4,extra_packed_bytes=packed.numel()*4,CPU_pack_seconds=pack_seconds,
   parameter_checks=parameter_checks,all_25_parameters_exactly_preserved=True,model_state_sha256=before,
   checkpoint_sha256=CHECKPOINT_SHA,source_reference_sha256=REFERENCE_SHA,kernel_source_sha256=KERNEL_SHA,
   torch_version=str(torch.__version__),triton_version=fusion.triton.__version__,restored_layouts=layouts,
   limitation='CPU packed formula equivalence only; OWN Triton compilation/numerics/latency unverified')
 except BaseException as error:
  result.update(status='failed_CPU_contract',completed=False,error=f'{type(error).__name__}: {error}')
  (a.out/'exception.txt').write_text(traceback.format_exc());raise
 finally:
  result.update(wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,cpu_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  with(a.out/('summary.json'if result.get('completed')else'failure.json')).open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps(dict(status=result['status'],completed=result['completed'],cases=len(result['cases']),out=str(a.out))),flush=True)

if __name__=='__main__':main()
