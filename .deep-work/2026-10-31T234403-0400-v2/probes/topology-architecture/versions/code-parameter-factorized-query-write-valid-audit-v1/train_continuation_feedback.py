"""Frozen VALID causal diagnostic; hand-selected query boundary is NOT learned.

Reuses exact NN _transition, never supplies true values/routes to computation.
All learned writes are still computed/paid, even when outputs are discarded.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def execute(m,initial,tokens,lengths,mode):
 carry=m.initial_carry(initial);snapK=torch.zeros_like(carry['keys']);snapV=torch.zeros_like(carry['values']);has=torch.zeros_like(lengths,dtype=torch.bool)
 for t in range(tokens.shape[1]):
  capture=t==lengths-3
  snapK=torch.where(capture[:,None,None],carry['keys'],snapK);snapV=torch.where(capture[:,None,None],carry['values'],snapV);has=has|capture
  ids=(t<lengths).nonzero(as_tuple=False).flatten()
  if ids.numel():
   selected={k:v.index_select(0,ids)for k,v in carry.items()};x=m.token_embedding(tokens[:,t].index_select(0,ids))
   updated,_=m._transition(selected,x,'soft')
   carry={k:v.index_copy(0,ids,updated[k])for k,v in carry.items()}
  if mode!='manual_original':
   K=snapK.roll(1,1)if mode=='joint_roll'else snapK
   V=snapV.roll(1,1)if mode in('joint_roll','value_only_roll')else snapV
   carry['keys']=torch.where(has[:,None,None],K,carry['keys']);carry['values']=torch.where(has[:,None,None],V,carry['values'])
 return F.log_softmax(m.readout(carry['h']),-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--family');p.add_argument('--max-seconds',type=float,default=100);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();torch.set_num_threads(1);device=torch.device('cuda:0')
 manifest=json.loads((a.inputs/'manifest.json').read_text());cp=a.inputs/'resume.pt'
 if sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('OWN exact completed NN required')
 payload=torch.load(cp,map_location='cpu',weights_only=False)
 if payload['schema']!='factorized-value-development-training-v1'or payload['stage']!=3 or payload['step']!=600 or not payload['stage_complete']:raise ValueError('exact fourstage NN')
 m=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']).to(device);m.eval();before=state_sha(m)
 for q in m.parameters():q.requires_grad_(False)
 data_path=a.inputs/'valid.npz'
 if sha(data_path)!=manifest['valid_sha256']:raise ValueError('exact original VALID required')
 with np.load(data_path,allow_pickle=False)as z:data={k:z[k]for k in z.files}
 lengths=data['lengths'];rows=np.arange(512)
 if len(lengths)!=512 or not np.all(data['tokens'][rows,lengths-3]==10)or not np.all(data['tokens'][rows,lengths-1]==15):raise ValueError('fixed benchmark READ/KEY/MIX query suffix')
 outputs={};results={};modes=('original','manual_original','freeze_query_writes','joint_roll','value_only_roll')
 with torch.no_grad():
  for mode in modes:
   t=time.perf_counter();values=[];mac=0;active=0
   for i in range(0,512,64):
    if time.perf_counter()-tick>100:raise TimeoutError('bounded100s diagnostic')
    sl=slice(i,i+64);l=int(lengths[sl].max());initial=torch.as_tensor(data['initial'][sl],device=device);tokens=torch.as_tensor(data['tokens'][sl,:l],device=device);lens=torch.as_tensor(lengths[sl],device=device)
    out=m(initial,tokens,lens,output='logits')if mode=='original'else execute(m,initial,tokens,lens,mode)
    torch.cuda.synchronize();values.append(out.cpu().numpy());n=int(lengths[sl].sum());active+=n;mac+=m.resource_report(batch_size=64,sequence_length=l,active_steps=n)['proxy_forward_macs']
   logp=np.concatenate(values);pred=logp.argmax(-1);outputs[mode+'-logp']=logp;outputs[mode+'-predictions']=pred
   results[mode]=dict(accuracy=float((pred==data['targets']).mean()),terminal_NLL=float(-logp[np.arange(512),data['targets']].mean()),evaluation_wall_seconds=time.perf_counter()-t,forward_MAC_proxy=mac,active_token_steps=active)
 manual=float(np.max(np.abs(outputs['original-logp']-outputs['manual_original-logp'])))
 if manual>1e-6 or np.any(outputs['original-predictions']!=outputs['manual_original-predictions']):raise ValueError('manual transition replay differs from real NN')
 for mode in modes:results[mode]['paired_original_prediction_changes']=int((outputs[mode+'-predictions']!=outputs['original-predictions']).sum())
 outputs['targets']=data['targets']
 with(a.out/'raw.npz').open('xb')as f:np.savez_compressed(f,**outputs)
 if state_sha(m)!=before or sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('NN changed')
 report=dict(status='ACTUAL_FROZEN_QUERY_WRITE_AUDIT_COMPLETE',SGD_updates=0,P=m.parameter_count(),logical_carry=m.resource_report()['persistent_example_state_scalars'],results=results,manual_maximum_logp_difference=manual,NN_unchanged=True,raw_sha256=sha(a.out/'raw.npz'),checkpoint_sha256=sha(cp),VALID_sha256=sha(data_path),driver_wall_seconds=time.perf_counter()-tick,source_sha256=manifest['source_sha256'],scope='diagnostic freezes only K/V after benchmark READ boundary; h/controller/calls still execute learned rules; NOT learned protection/newarchitecture or actual skip-cost saving')
 (a.out/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
