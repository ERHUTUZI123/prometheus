"""Small real integration check: GRU32/Adam growth, replay padding and reload.
Synthetic data only; one diagnostic CPU Adam step, not baseline training.
"""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage
import train_gru32_fourstage as driver

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0);device=torch.device('cpu')
 model=old.TerminalGRU(8,15,48,width=32,layers=2);opt=torch.optim.Adam(model.parameters(),lr=.003);args=argparse.Namespace(arm='gru',baseline_width=32)
 if driver.parameters(model)!=13992:raise AssertionError('actual original GRU32 P differs')
 tokens=np.zeros((64,128),dtype=np.int64);tokens[:,:48]=np.random.default_rng(0).integers(1,15,(64,48));lengths=np.full(64,48,dtype=np.int64)
 data=dict(initial=np.arange(64,dtype=np.int64)%8,tokens=tokens,lengths=lengths,targets=(np.arange(64,dtype=np.int64)+1)%8)
 b,_,_=old.batch(data,np.arange(64),device);loss=F.cross_entropy(old.forward(model,b,args,'soft'),b['targets']);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad(set_to_none=True)
 model.eval()
 with torch.no_grad():before=old.forward(model,b,args,'soft').clone()
 growth=grow_storage(model,opt,args)
 with torch.no_grad():after=old.forward(model,b,args,'soft')
 if driver.parameters(model)!=14024 or not torch.equal(before,after):raise AssertionError('actual growth P/old output changed')
 original=tokens.copy();view,proof=driver.maintenance_view(data,48)
 if not np.array_equal(tokens,original)or not np.shares_memory(tokens,view['tokens'])or not np.array_equal(view['lengths'],lengths):raise AssertionError('training data/lengths changed')
 invalid=dict(data,tokens=tokens.copy());invalid['tokens'][0,100]=1
 try:driver.maintenance_view(invalid,48)
 except ValueError:pass
 else:raise AssertionError('nonzero removed padding accepted')
 replay=Replay128(128,48,0,8);used=np.arange(64,dtype=np.int64)[None]
 # An independently named diagnostic checkpoint is written before consume.
 def save(leaf,complete):
  payload=dict(schema='gru32-CPU-diagnostic-v1',synthetic_Adam_steps=1,baseline_updates=0,maintenance_complete=complete,model_spec=model.spec,model_state=model.state_dict(),optimizer=opt.state_dict(),RNG=driver.rng(device),replay=replay.state_dict())
  path=a.out/leaf
  with path.open('xb')as f:torch.save(payload,f)
  return path
 pre=save('CPU-NN-before-maintenance.pt',False)
 replay.consume(view,used,0);replay.validate_sources({0:view},{0:used});post=save('CPU-NN-after-maintenance.pt',True)
 for path in(pre,post):
  saved=driver.rng(device);loaded=torch.load(path,map_location='cpu',weights_only=False);other=old.TerminalGRU(**loaded['model_spec']);other.load_state_dict(loaded['model_state'],strict=True);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);copy_replay=Replay128.from_state_dict(loaded['replay'])
  if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict()):raise AssertionError('actual NN/Adam reload differs')
  driver.restore_rng(loaded['RNG'],device)
  if not old.nested_state_equal(driver.rng(device),loaded['RNG']):raise AssertionError('actual CPU RNG reload differs')
  other.eval()
  with torch.no_grad():
   if not torch.equal(after,old.forward(other,b,args,'soft')):raise AssertionError('actual old-input forward reload differs')
  if loaded['maintenance_complete']:copy_replay.validate_sources({0:view},{0:used})
  elif copy_replay.size!=0:raise AssertionError('pre-maintenance checkpoint contains uncommitted replay')
  driver.restore_rng(saved,device)
 result=dict(source_sha256={leaf:driver.sha(Path(__file__).with_name(leaf))for leaf in('train_gru32_fourstage.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')},status='CPU_GRU32_FOURSTAGE_SHAPE_RELOAD_PASS',parameters_before=13992,parameters_after=14024,synthetic_Adam_steps=1,baseline_updates=0,actual_NN_Adam_CPU_RNG_replay_reload=True,CUDA_RNG_checked=False,CPU_only_scope='GPU RNG checked by actual training reloads',growth=growth,padding_view=proof,nonzero_padding_rejected=True,full_raw_training_shape=list(tokens.shape),lengths_preserved=True,wall_seconds=time.perf_counter()-started)
 with(a.out/'result.json').open('x')as f:json.dump(old.plain(result),f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(status=result['status'],parameters_after=14024)))
if __name__=='__main__':main()
