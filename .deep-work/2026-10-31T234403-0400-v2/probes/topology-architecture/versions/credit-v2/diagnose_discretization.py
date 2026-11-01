"""Matched checkpoint component ablation; inference/gradient diagnosis, no retrain.
Only trusted generated checkpoint and terminal evaluation arrays are read.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from topology_model import TopologyExecutor, assignment


def hybrid(model,tokens,starts,tasks,hard=(0,0,0,0),credit='prob'):
    he,ho,hr,hd=hard
    enc=F.softmax(model.encoder/model.temperature,-1)
    dec=F.softmax(model.decoder/model.temperature,-1)
    ops=model.soft_ops()
    route=F.softmax(model.calls(tokens,tasks)/model.temperature,-1)
    if he:enc=F.one_hot(model.encoder.argmax(-1),model.d).float()
    if hd:dec=F.one_hot(model.decoder.argmax(-1),model.n).float()
    if ho:
        perms=[np.arange(model.d)]+[assignment(x) for x in model.operations]
        ops=torch.stack([F.one_hot(torch.tensor(p,device=ops.device),model.d).float() for p in perms])
    if hr:route=F.one_hot(route.argmax(-1),model.n_ops+1).float()
    state=enc[starts]
    if credit=='log':state=state.clamp_min(1e-30).log()
    for pos in range(int(tokens.ne(0).sum(-1).max())):
        for slot in range(model.slots):
            matrix=torch.einsum('bk,kij->bij',route[:,pos,slot],ops)
            following=(torch.logsumexp(state[:,:,None]+matrix.clamp_min(1e-30).log(),1)
                       if credit=='log' else torch.bmm(state[:,None],matrix).squeeze(1))
            state=torch.where(tokens[:,pos,None]!=0,following,state)
    if credit=='log':return torch.logsumexp(state[:,:,None]+dec.clamp_min(1e-30).log()[None],1)
    return (state@dec).clamp_min(1e-12).log()


@torch.no_grad()
def joint_compile(model,tokens,starts,tasks,external=False):
    pred=torch.empty_like(starts)
    ops=model.soft_ops()
    enc=F.softmax(model.encoder/model.temperature,-1)
    dec=F.softmax(model.decoder/model.temperature,-1)
    for task in tasks.unique().tolist():
        pick=tasks==task
        latent=starts[pick].cpu().numpy() if external else model.encoder.argmax(-1)[starts[pick]].cpu().numpy()
        words=tokens[pick].cpu().numpy()
        maps={}
        for token in range(1,model.vocab+1):
            x=torch.tensor([[token]],device=tokens.device)
            tid=torch.tensor([task],device=tokens.device)
            routes=F.softmax(model.calls(x,tid)[0,0]/model.temperature,-1)
            m=torch.eye(model.d,device=tokens.device)
            for slot in range(model.slots):m=m@torch.einsum('k,kij->ij',routes[slot],ops)
            if external:m=enc@m@dec
            maps[token]=assignment(m.clamp_min(1e-30).log())
        for pos in range(words.shape[1]):
            for token,p in maps.items():
                active=words[:,pos]==token
                latent[active]=p[latent[active]]
        result=latent if external else model.decoder.argmax(-1).cpu().numpy()[latent]
        pred[pick]=torch.tensor(result,device=tokens.device)
    return pred


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    n=int(checkpoint['args']['scale']);factor=int(checkpoint['args']['latent_factor'])
    model=TopologyExecutor(n,latent_factor=factor,protection=False,seed=checkpoint['args']['seed'])
    model.load_state_dict(checkpoint['model']);model.set_temperature(checkpoint['model_extra']['temperature']);model.eval()
    with np.load(a.data,allow_pickle=False) as z:
        batch={k:torch.from_numpy(z[k]).long() for k in ['tokens','starts','task_ids','targets']}
    rows=[]
    for hard in itertools.product((0,1),repeat=4):
        with torch.no_grad():y=hybrid(model,batch['tokens'],batch['starts'],batch['task_ids'],hard)
        rows.append(dict(hard_encoder=hard[0],hard_operations=hard[1],hard_calls=hard[2],hard_decoder=hard[3],
                         accuracy=float((y.argmax(-1)==batch['targets']).float().mean()),loss=float(F.nll_loss(y,batch['targets']))))
    compiled=[]
    for external in (False,True):
        y=joint_compile(model,batch['tokens'],batch['starts'],batch['task_ids'],external)
        compiled.append(dict(kind='joint_external_map' if external else 'joint_latent_operator',accuracy=float((y==batch['targets']).float().mean()),
                             teacher_operations_used=False))
    gradients=[]
    vectors=[]
    for credit in ['prob','log']:
        model.zero_grad(set_to_none=True)
        y=hybrid(model,batch['tokens'][:64],batch['starts'][:64],batch['task_ids'][:64],credit=credit)
        loss=F.nll_loss(y,batch['targets'][:64]);loss.backward()
        v=torch.cat([x.grad.reshape(-1) if x.grad is not None else torch.zeros_like(x).reshape(-1) for x in model.parameters()])
        vectors.append(v)
        gradients.append(dict(credit=credit,loss=float(loss),gradient_norm=float(v.norm())))
    result=dict(checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
                n_states=n,temperature=model.temperature,component_ablation=rows,joint_compiler=compiled,credit=gradients,
                gradient_relative_difference=float((vectors[0]-vectors[1]).norm()/vectors[1].norm().clamp_min(1e-12)),
                gradient_cosine=float(F.cosine_similarity(vectors[0],vectors[1],dim=0)),
                inference_only_no_parameter_updates=True,M0='NOT_PASSED')
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
