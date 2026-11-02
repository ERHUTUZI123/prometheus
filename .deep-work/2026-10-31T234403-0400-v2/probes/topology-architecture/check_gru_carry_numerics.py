"""Actual frozen NN numerical diagnosis, never a training/configuration search."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import owned_parent_driver as old
import frozen_profile_driver as trial
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--family');p.add_argument('--max-seconds',type=float,default=60);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();torch.set_num_threads(1);device=torch.device('cuda:0')
 m=json.loads((a.inputs/'manifest.json').read_text());cp=a.inputs/'resume.pt'
 if sha(cp)!=m['checkpoint_sha256']:raise ValueError('OWN prephase3 NN required')
 payload=torch.load(cp,map_location='cpu',weights_only=False)
 if payload['source_sha256']!=m['parent_source_sha256']:raise ValueError('exact parent source mismatch')
 model,opt,args=old.restore_model(payload,device);model.eval();before={n:v.detach().cpu().clone()for n,v in model.state_dict().items()}
 data={}
 for key,split in [('train','train'),('valid','valid')]:
  spec=m['files']['0:'+split];path=a.inputs/spec['leaf']
  if sha(path)!=spec['sha256']:raise ValueError('own original raw arrays changed')
  with np.load(path,allow_pickle=False)as z:data[key]={k:z[k]for k in z.files}
 words=[]
 for tokens,length in zip(data['train']['tokens'],data['train']['lengths']):
  word=tuple(int(x)for x in tokens[:length])
  if 1<=length<=2 and word not in words:words.append(word)
  if len(words)==4:break
 b,_,_=old.batch(data['valid'],np.arange(16),device);rows=[];raw={};flag=torch.backends.cudnn.allow_tf32
 with torch.no_grad():
  for allow_tf32 in(True,False):
   torch.backends.cudnn.allow_tf32=allow_tf32;base,carry=trial.prefix(model,b,'gru');original=F.log_softmax(old.forward(model,b,args,'soft'),-1)
   for wi,word in enumerate(words):
    if time.perf_counter()-tick>60:raise TimeoutError('60s scientific diagnosis')
    dense=trial.suffix(model,carry,word,'gru',16)
    wt=torch.tensor(word,device=device)[None].expand(16,-1)
    packed=pack_padded_sequence(model.tokens(wt),torch.full((16,),len(word),dtype=torch.long),batch_first=True,enforce_sorted=False)
    _,hidden=model.rnn(packed,carry.contiguous());pack=F.log_softmax(model.readout(model.norm(hidden[-1])),-1)
    lens=b['lengths'].cpu().tolist();full=torch.zeros((16,max(lens)+len(word)),device=device,dtype=torch.long)
    for i,length in enumerate(lens):full[i,:length]=b['tokens'][i,:length];full[i,length:length+len(word)]=torch.tensor(word,device=device)
    joined=dict(initial=b['initial'],tokens=full,lengths=b['lengths']+len(word));ref=F.log_softmax(old.forward(model,joined,args,'soft'),-1)
    prefix_key=('TF32on'if allow_tf32 else'TF32off')+'-word'+str(wi)
    for name,value in [('original_prefix',original),('extracted_prefix',base),('dense_tail',dense),('packed_tail',pack),('full_reference',ref)]:raw[prefix_key+'-'+name]=value.cpu().numpy()
    def compare(x,y):return dict(maximum_absolute_difference=float((x-y).abs().max()),prediction_disagreements=int((x.argmax(-1)!=y.argmax(-1)).sum()),allclose_2e4=bool(torch.allclose(x,y,atol=2e-4,rtol=2e-5)))
    rows.append(dict(allow_tf32=allow_tf32,word=list(word),prefix=compare(base,original),dense=compare(dense,ref),packed=compare(pack,ref)))
 torch.backends.cudnn.allow_tf32=flag
 with(a.out/'raw.npz').open('xb')as f:np.savez_compressed(f,**raw)
 if not old.tensor_state_equal(before,model.state_dict()):raise ValueError('frozen NN mutated')
 result=dict(status='ACTUAL_GRU_CARRY_NUMERICAL_DIAGNOSIS_COMPLETE',SGD_updates=0,model_unchanged=True,checkpoint_sha256=sha(cp),P=sum(p.numel()for p in model.parameters()),rows=rows,raw_sha256=sha(a.out/'raw.npz'),wall_seconds=time.perf_counter()-tick,precision='diagnose packed/dense split and cudnn TF32, never accuracy-selected configurations',torch=str(torch.__version__),cudnn=torch.backends.cudnn.version())
 (a.out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
