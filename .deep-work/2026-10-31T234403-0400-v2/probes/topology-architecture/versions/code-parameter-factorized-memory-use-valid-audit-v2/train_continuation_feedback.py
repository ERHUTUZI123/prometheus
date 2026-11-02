"""Two NEW frozen-VALID interventions; previous baseline vectors reused honestly."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--family');p.add_argument('--max-seconds',type=float,default=100);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();torch.set_num_threads(1);device=torch.device('cuda:0')
 manifest=json.loads((a.inputs/'manifest.json').read_text());cp=a.inputs/'resume.pt'
 if sha(cp)!=manifest['checkpoint_sha256']or sha(a.inputs/'valid.npz')!=manifest['valid_sha256']or sha(a.inputs/'previous.npz')!=manifest['previous_sha256']:raise ValueError('exact immutable NN/data/previous vectors required')
 payload=torch.load(cp,map_location='cpu',weights_only=False);m=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']).to(device);m.eval()
 before={n:v.detach().cpu().clone()for n,v in m.state_dict().items()}
 for v in m.parameters():v.requires_grad_(False)
 with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:data={k:z[k]for k in z.files}
 with np.load(a.inputs/'previous.npz',allow_pickle=False)as z:base=z['original-predictions'];base_logp=z['original-logp']
 result={};raw={'targets':data['targets']}
 with torch.no_grad():
  for mode in('zero_values_every_token','zero_keys_every_token'):
   t0=time.perf_counter();out=[];mac=0
   for begin in range(0,512,64):
    if time.perf_counter()-tick>100:raise TimeoutError('100s')
    sl=slice(begin,begin+64);lengths=torch.as_tensor(data['lengths'][sl],device=device);tokens=torch.as_tensor(data['tokens'][sl,:int(data['lengths'][sl].max())],device=device);initial=torch.as_tensor(data['initial'][sl],device=device);carry=m.initial_carry(initial)
    for t in range(tokens.shape[1]):
     key='values'if mode=='zero_values_every_token'else'keys';carry[key]=torch.zeros_like(carry[key])
     ids=(t<lengths).nonzero(as_tuple=False).flatten()
     if ids.numel():
      updated,_=m._transition({k:v.index_select(0,ids)for k,v in carry.items()},m.token_embedding(tokens[:,t].index_select(0,ids)),'soft');carry={k:v.index_copy(0,ids,updated[k])for k,v in carry.items()}
    out.append(F.log_softmax(m.readout(carry['h']),-1).cpu().numpy());mac+=m.resource_report(batch_size=64,sequence_length=tokens.shape[1],active_steps=int(lengths.sum()))['proxy_forward_macs']
   torch.cuda.synchronize();logp=np.concatenate(out);pred=logp.argmax(-1);raw[mode+'-logp']=logp;raw[mode+'-predictions']=pred
   result[mode]=dict(accuracy=float((pred==data['targets']).mean()),NLL=float(-logp[np.arange(512),data['targets']].mean()),paired_previous_prediction_changes=int((pred!=base).sum()),forward_MAC_proxy=mac,wall_seconds=time.perf_counter()-t0)
 with(a.out/'raw.npz').open('xb')as f:np.savez_compressed(f,**raw)
 if any(not torch.equal(t,m.state_dict()[n].detach().cpu())for n,t in before.items()):raise ValueError('NN changed')
 report=dict(status='ACTUAL_EXTERNAL_MEMORY_INTERVENTIONS_COMPLETE',P=m.parameter_count(),SGD_updates=0,NN_unchanged=True,baseline_source_sha256=manifest['previous_sha256'],baseline_reexecuted=False,baseline_accuracy=float((base==data['targets']).mean()),results=result,checkpoint_sha256=sha(cp),VALID_sha256=sha(a.inputs/'valid.npz'),raw_sha256=sha(a.out/'raw.npz'),driver_wall_seconds=time.perf_counter()-tick,scope='all write/query arithmetic still computed and charged; zeroKeys removes BOTH read and write content addressing, not a learned/pruned architecture; no protection/advantage claim')
 (a.out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
