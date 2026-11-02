"""Owned baseline checkpoint adapters; exact original ordinary NN classes.

No fresh baseline training, task parsing, teacher, network or compilation.
Factories are neutralized by restoring OWN RNG before original token growth.
"""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage
ARMS={'gru32':('gru',13992,14024),'transformer32':('transformer',26728,26760)}
SCHEMA='paired-baseline-stage3-training-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def NNhash(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():
  a=v.detach().cpu().contiguous();h.update(json.dumps([n,list(a.shape),str(a.dtype)]).encode());h.update(a.numpy().tobytes())
 return h.hexdigest()
def sources(manifest):
 for leaf,digest in manifest['source_sha256'].items():
  if Path(leaf).name!=leaf or sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source differs: '+leaf)
 return manifest['source_sha256']
def load_manifest(inputs):
 m=json.loads((inputs/'manifest.json').read_text())
 if m['schema']!='paired-baseline-stage3-inputs-v1':raise ValueError('fixed owned baseline manifest')
 sources(m);return m
def path(inputs,manifest,key):
 spec=manifest['files'][key];f=inputs/spec['leaf']
 if not f.is_file()or f.is_symlink()or sha(f)!=spec['sha256']:raise ValueError('exact owned input differs: '+key)
 return f
def data(inputs,manifest,stage,split):
 f=path(inputs,manifest,f'data:{stage}:{split}')
 with np.load(f,allow_pickle=False)as z:
  if set(z.files)!={'initial','tokens','lengths','targets'}:raise ValueError('raw terminal inputs only')
  result={k:z[k]for k in z.files}
 n=len(result['initial']);tokens=result['tokens'];lens=result['lengths'];active=np.arange(tokens.shape[1])[None]<lens[:,None]
 if any(v.dtype!=np.int64 for v in result.values())or tokens.ndim!=2 or any(result[k].shape!=(n,)for k in('initial','lengths','targets'))or np.any((lens<0)|(lens>tokens.shape[1]))or np.any((result['initial']<0)|(result['initial']>=8))or np.any((result['targets']<0)|(result['targets']>=8))or np.any(active&((tokens<0)|(tokens>=(16 if stage==3 else 15)))):raise ValueError('exact public shape/ranges')
 return result
def used_old(inputs,manifest,arm):
 used={}
 for stage in range(3):
  with np.load(path(inputs,manifest,f'oldindices:{arm}:{stage}'),allow_pickle=False)as z:indices=z['indices'].copy()
  if indices.shape!=(600,64)or indices.dtype!=np.int64:raise ValueError('actual old 600x64 schedule')
  used[stage]=indices if stage==0 else indices[:,:48]
 return used
def state_now(device):
 return dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),cuda=torch.cuda.get_rng_state_all()if device.type=='cuda'else[])
def restore_rng(state,device):
 torch.set_rng_state(state['cpu']);np.random.set_state(state['numpy']);random.setstate(state['python'])
 if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])
 actual=state_now(device);fields=('cpu','numpy','python','cuda')if device.type=='cuda'else('cpu','numpy','python')
 if any(not old.nested_state_equal(actual[k],state[k])for k in fields):raise RuntimeError('OWN available RNG restore differs')
def restore(inputs,manifest,arm,device):
 started=time.perf_counter();p=path(inputs,manifest,'parent:'+arm);payload=torch.load(p,map_location='cpu',weights_only=False);entry=manifest['parents'][arm];legacy,p15,p16=ARMS[arm]
 if payload.get('schema')!=entry['schema']or payload.get('stage')!=2 or payload.get('step')!=600 or payload.get('stage_complete')is not True or payload['source_sha256']!=entry['source_sha256']:raise ValueError('OWN complete stage2 checkpoint/source only')
 if arm=='gru32':
  if not payload.get('maintenance_complete')or payload['fixed_policy']!=dict(seed=0,steps_per_stage=600,lr=.003,clip=1.,width=32,layers=2):raise ValueError('GRU original applied policy')
  model=old.TerminalGRU(**payload['model_spec']).to(device);model.load_state_dict(payload['model_state'],strict=True);optimizer=torch.optim.Adam(model.parameters(),lr=.003);optimizer.load_state_dict(payload['optimizer']);args=argparse.Namespace(arm='gru',states=8,seed=0,lr=.003,baseline_width=32,replay_capacity=128,replay_per_batch=16,stages=4,steps=600)
 else:
  saved=payload['args']
  if any(saved[k]!=v for k,v in dict(arm='transformer',seed=0,states=8,lr=.003,steps=600,baseline_width=32,replay_capacity=128,replay_per_batch=16).items()):raise ValueError('Transformer original applied policy')
  model,optimizer,args=old.restore_model(payload,device)
 if model.spec!=entry['model_spec']or sum(p.numel()for p in model.parameters())!=p15 or any(not p.requires_grad for p in model.parameters()):raise ValueError('exact original ordinary class/spec/P/trainability')
 if not old.tensor_state_equal(model.state_dict(),payload['model_state'])or not old.nested_state_equal(optimizer.state_dict(),payload['optimizer']):raise RuntimeError('actual NN/Adam restore differs')
 replay=Replay128.from_state_dict(payload['replay_state'])
 if(replay.size,replay.capacity,replay.width,replay.seen,replay.draws)!=(128,128,48,96000,19200)or replay.digest()!=entry['replay_digest']or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('actual OWN completed Replay128')
 used=used_old(inputs,manifest,arm);oldtrain={s:data(inputs,manifest,s,'train')for s in range(3)};audit=replay.validate_sources(oldtrain,used)
 if arm=='gru32'and any(not np.array_equal(used[s],payload['committed_used_indices'][s])for s in used):raise ValueError('GRU committed indices differ')
 restore_rng(payload['rng'],device)
 proof=dict(parent_checkpoint_sha256=sha(p),NN_Adam_replay_exact=True,available_RNG_exact=True,CUDA_RNG_checked=device.type=='cuda',CUDA_RNG_payload_preserved=True,source_audit=audit,P15=p15,old_effective_updates=1800,old_updates_repeated=0,restore_wall_seconds=time.perf_counter()-started,pre_growth_initial_theta_sha256=NNhash(model))
 return model,optimizer,args,replay,used,oldtrain,payload,proof

def grow(model,optimizer,args,device):
 probe=dict(initial=torch.tensor([0,1],device=device),tokens=torch.tensor([[1,3,0],[2,4,3]],device=device),lengths=torch.tensor([2,3],device=device));model.eval()
 with torch.no_grad():before=model(**probe).clone()
 report=grow_storage(model,optimizer,args)
 with torch.no_grad():after=model(**probe)
 if not torch.equal(before,after):raise RuntimeError('original token/position growth changed old forward')
 return report,dict(post_growth_initial_theta_sha256=NNhash(model),new_token_row_sha256=tensor_hash(model.tokens.weight[15]),scope='device-specific OWN restored RNG; no CPU/GPU row equality claim')

def schedules(inputs,manifest,replay,sampling):
 with np.load(path(inputs,manifest,'paired_schedule'),allow_pickle=False)as z:
  key='indices'if sampling=='paired'else'indices_same_pool_shuffled';indices=z[key].copy();paired=z['indices'];control=z['indices_same_pool_shuffled']
  if indices.shape!=(600,64)or indices.dtype!=np.int64 or not np.array_equal(paired[:,48:],control[:,48:])or not np.array_equal(np.bincount(paired[:,:48].reshape(-1),minlength=4096),np.bincount(control[:,:48].reshape(-1),minlength=4096)):raise ValueError('paired/samepool exposure control')
 with np.load(path(inputs,manifest,'replay_draws'),allow_pickle=False)as z:expected=z['buffer_indices'].copy()
 before=replay.draws;generated=replay.sample_indices(3,600,16)
 if generated.shape!=(600,16)or not np.array_equal(generated,expected)or replay.draws!=before:raise ValueError('ONE original Replay draw schedule; no admissionmix')
 return indices,generated

def saved_payload(model,optimizer,replay,arm,sampling,manifest,used,step,costs,complete,synthetic=False):
 return dict(schema=SCHEMA,arm=arm,sampling=sampling,stage=3,step=step,stage_complete=complete,maintenance_complete=complete,synthetic_check_only=synthetic,model_spec=copy.deepcopy(model.spec),model_state={n:v.detach().cpu().clone()for n,v in model.state_dict().items()},optimizer=optimizer.state_dict(),optimizer_parameter_names=[n for n,p in model.named_parameters()],rng=state_now(next(model.parameters()).device),replay_state=replay.state_dict(),used_current_indices={s:a.copy()for s,a in used.items()},source_sha256=manifest['source_sha256'],parent_checkpoint_sha256=manifest['parents'][arm]['checkpoint_sha256'],costs=copy.deepcopy(costs),formal_Primary=False)
def reload(payload,device):
 if payload['schema']!=SCHEMA:raise ValueError('new baseline continuation schema only')
 cls=old.TerminalGRU if payload['arm']=='gru32'else old.TerminalTransformer;model=cls(**payload['model_spec']).to(device);model.load_state_dict(payload['model_state'],strict=True);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
 if[n for n,p in model.named_parameters()]!=payload['optimizer_parameter_names']:raise ValueError('exact baseline Adam parameter order')
 optimizer.load_state_dict(payload['optimizer']);replay=Replay128.from_state_dict(payload['replay_state'])
 if not old.tensor_state_equal(model.state_dict(),payload['model_state'])or not old.nested_state_equal(optimizer.state_dict(),payload['optimizer'])or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise RuntimeError('actual final NN/Adam/replay reload')
 return model,optimizer,replay
