"""CPU-only technical admission of fixed GRU34/TF24; no SGD or label read."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from train_char_parameter_matched_alltoken import AllTokenLM

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def equal(a,b):return a.keys()==b.keys()and all(torch.equal(a[k].cpu(),b[k].cpu())for k in a)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=90);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 tick=time.perf_counter();torch.set_num_threads(1);manifest=json.loads((a.inputs/'manifest.json').read_text())
 def bound():
  if time.perf_counter()-tick>a.max_seconds:raise TimeoutError('fixed controls CPU90s technical admission')
 for leaf,digest in manifest['source_sha256'].items():
  if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('exact frozen conventional-control source required')
 for leaf,spec in manifest['data_files'].items():
  if sha(a.inputs/leaf)!=spec['sha256']:raise ValueError('same original TRAIN/metadata bytes required')
 meta=json.loads((a.inputs/'metadata.json').read_text())
 if meta['schema']!='char-alltoken-ar-development-v1'or meta['n_tokens']!=100:raise ValueError('actual original vocabulary100 required')
 with np.load(a.inputs/'train.npz',allow_pickle=False)as z:
  initial=torch.as_tensor(z['initial'][:2],dtype=torch.long);tokens=torch.as_tensor(z['tokens'][:2,:8].copy(),dtype=torch.long)
 # Only raw TRAIN inputs decompressed; no target labels/NN-selected examples.
 lengths=torch.full((2,),8,dtype=torch.long);reports={}
 for arm in('gru','transformer'):
  bound();started=time.perf_counter();torch.manual_seed(0);model=AllTokenLM(arm,100);model.eval()
  expected=manifest['analytic_counts']['rows']['gru34'if arm=='gru'else'transformer24']
  count=sum(v.numel()for v in model.parameters());buffers=sum(v.numel()*v.element_size()for v in model.buffers())
  if count!=expected['P']or buffers!=expected['registered_buffer_bytes']:raise ValueError('actual P/buffers differ from analytic pre-model count')
  before={k:v.detach().clone()for k,v in model.state_dict().items()}
  with torch.no_grad():
   output=model(initial,tokens,lengths)
   if output.shape!=(2,8,100)or not bool(torch.isfinite(output).all())or float(torch.logsumexp(output,-1).abs().max())>1e-5:raise ValueError('finite normalized alltoken interface failed')
   changed=tokens.clone();changed[:,4:]=(changed[:,4:]+1)%100
   other=model(initial,changed,lengths);future=float((output[:,:4]-other[:,:4]).abs().max())
   if future>2e-6:raise ValueError('actual future-token leakage')
   prefix=model(initial,tokens[:,:4].contiguous(),torch.full((2,),4,dtype=torch.long));prefix_difference=float((output[:,:4]-prefix).abs().max())
   if not torch.allclose(output[:,:4],prefix,atol=2e-6,rtol=2e-5):raise ValueError('prefix causal alignment differs')
   cp=a.out/(arm+'-actual-interface-checkpoint.pt')
   torch.save(dict(schema='parameter-matched-alltoken-interface-v1',arm=arm,configuration=model.configuration(),state_dict=model.state_dict(),source_sha256=manifest['source_sha256']),cp)
   payload=torch.load(cp,map_location='cpu',weights_only=True)
   if payload['schema']!='parameter-matched-alltoken-interface-v1'or payload['arm']!=arm or payload['source_sha256']!=manifest['source_sha256']:raise ValueError('actual checkpoint header/source differs')
   saved_rng=torch.get_rng_state().clone();restored=AllTokenLM(arm,100);restored.load_state_dict(payload['state_dict'],strict=True);torch.set_rng_state(saved_rng);restored.eval()
   if restored.configuration()!=payload['configuration']or not equal(model.state_dict(),restored.state_dict()):raise ValueError('same OWN configuration/NN/buffer reload differs')
   reloaded=restored(initial,tokens,lengths)
   if not torch.equal(output,reloaded):raise ValueError('actual sameinput checkpoint output differs')
  model.train();loss=-model(initial,tokens,lengths)[...,0].mean();loss.backward();gradients=[v.grad for v in model.parameters()if v.grad is not None]
  if not gradients or any(not bool(torch.isfinite(v).all())for v in gradients):raise ValueError('nonfinite alltoken scalar backward')
  if not equal(before,model.state_dict()):raise ValueError('CPU check changed model parameter/buffer values')
  if arm=='gru':
   independent=2*8*(2*(3*34*34+3*34*34)+34*100)
  else:
   independent=2*(3*(4*8*24**2+2*8**2*24+2*8*24*48)+8*24*100)
  if model.proxy(2,8)!=independent:raise ValueError('independent source-product forward MAC accounting differs')
  reports[arm]=dict(status='PASS',P=count,configuration=model.configuration(),registered_buffer_bytes=buffers,parameter_bytes=sum(v.numel()*v.element_size()for v in model.parameters()),
   logical_carry_scalars=68 if arm=='gru'else None,logical_carry_bytes_float32=272 if arm=='gru'else None,
   carry_scope='GRU two34wide hidden states; Transformer full-window API has no reusable KV cache, not zero attention workspace',
   future_suffix_difference=future,prefix_maximum_difference=prefix_difference,checkpoint_exact_NN_buffers_outputs=True,checkpoint_sha256=sha(cp),checkpoint_bytes=cp.stat().st_size,
   gradient_payload_bytes=sum(v.numel()*v.element_size()for v in gradients),actual_forward_MAC_proxy=4*model.proxy(2,8)+model.proxy(2,4),
   forward_count_scope='output/futurechanged/reload/scalar-gradient four2x8 forwards plus one2x4prefix; actual scalar backward paid in wall, full backward FLOPs unknown',
   SGD_updates=0,NN_values_unchanged=True,driver_wall_seconds=time.perf_counter()-started)
 batches=np.random.default_rng(np.random.SeedSequence([0,100,300,0x41524254])).integers(0,8192,size=(300,32),dtype=np.int64);path=a.out/'common-batch-indices.npz'
 with path.open('xb')as stream:np.savez_compressed(stream,indices=batches)
 if sha(path)!=manifest['common_batch_indices_sha256']:raise ValueError('original fixed300 common batches differ')
 bound()
 for leaf,digest in manifest['source_sha256'].items():
  if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('source differs after technical admission')
 result=dict(status='CPU_PARAMETER_MATCHED_ALLTOKEN_CONTROLS_PASS',SGD_updates=0,models=reports,source_sha256=manifest['source_sha256'],analytic_counts=manifest['analytic_counts'],
  TRAIN_label_values_read=0,TRAIN_input_rows=2,input_selection='first2 raw originalTRAIN rows/first8tokens; no label/outcome choice',
  synthetic_gradient_objective='negative mean logprob of fixedclass0; backward only, no optimizer or NN update',common_batch_indices_sha256=sha(path),
  driver_wall_seconds=time.perf_counter()-tick,CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024),
  live_optimizer_bytes=0,formal_Primary=False,scope='technical interface/P/resource/causality/reload admission only; no accuracy gate; cold import/runner/SSH separately charged')
 with(a.out/'result.json').open('x')as stream:json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],P={k:v['P']for k,v in reports.items()},driver_wall_seconds=result['driver_wall_seconds'])))
if __name__=='__main__':main()
