"""Small actual CPU rule/gradient/reload test; no research data or training."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
import continuous_primitive_block as core

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(20261102)
    m=core.ContinuousPrimitiveBlock(8,15);m.eval()
    initial=torch.tensor([0,3,7]);tokens=torch.randint(0,15,(3,8));lengths=torch.tensor([8,5,0])
    out,trace=m(initial,tokens,lengths,output='logits',return_traces=True)
    assert out.shape==(3,8) and torch.isfinite(out).all()
    assert m.parameter_count()==13381 and m.resource_report()['persistent_example_state_scalars']==432
    log_all,carry=m.forward_all(initial,tokens,lengths,return_carry=True)
    assert torch.equal(out,log_all[:,-1]) and torch.equal(carry['h'],trace['final_state'])
    changed=tokens.clone();changed[:,4:]=(changed[:,4:]+1)%15
    alt=m.forward_all(initial,changed,lengths)
    assert torch.equal(log_all[:,:4],alt[:,:4])
    assert torch.equal(trace['states'][2,0],trace['states'][2,-1])
    assert torch.equal(carry['h'][2],m.encode_state(initial)[2])
    one=m.initial_carry(initial[:2]);x=m.token_embedding(tokens[:2,0]);n,t=m._transition(one,x,'soft')
    amount=(t['write_gate']*t['write_addresses'])[:,:,None]
    assert torch.equal(n['values'],one['values']+amount*(n['h'][:,None]-one['values']))
    assert torch.equal(one['values'],torch.zeros_like(one['values']))
    loss=torch.nn.functional.nll_loss(out,torch.tensor([2,4,6]))
    anchor=(trace['final_state']-m.encode_state(torch.tensor([2,4,6])).detach()).square().mean()
    (loss+anchor).backward()
    groups={n:float(v.grad.abs().sum())if v.grad is not None else None for n,v in m.named_parameters()}
    for key in ('controller.weight_ih','primitive_down','primitive_up','caller.0.weight','write_key.weight','read_query.weight','encoder.weight'):
        assert groups[key]is not None and groups[key]>0,key
    opt=torch.optim.Adam(m.parameters(),lr=.003);opt.step();opt.zero_grad(set_to_none=True)
    payload=m.checkpoint();path=a.out/'synthetic-block.pt';torch.save(payload,path)
    restored=core.ContinuousPrimitiveBlock.from_checkpoint(torch.load(path,weights_only=True));restored.eval()
    for mode in ('soft','native'):
        with torch.no_grad():assert torch.equal(m(initial,tokens,lengths,mode,output='logits'),restored(initial,tokens,lengths,mode,output='logits'))
    try:core.ContinuousPrimitiveBlock.from_checkpoint(dict(payload,version='continuous-primitive-block-v1'))
    except ValueError:pass
    else:raise AssertionError('legacy checkpoint was reinterpreted')
    result=dict(status='CPU_FACTOR_VALUE_RULE_CHECK_PASSED',parameters=m.parameter_count(),carry_scalars=432,
        synthetic_SGD_updates=1,research_data_loaded=False,source_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        future_causality_exact=True,ragged_and_zero_length_exact=True,identity_value_write_exact=True,
        checkpoint_reload_exact=True,legacy_rejected=True,parameter_grad_abs_sum=groups,
        value_interface='sqrt(H)*normalize(E(y)); terminal y only, no internal teacher trace',
        elapsed_seconds=time.perf_counter()-start,resource_report=m.resource_report(batch_size=3,sequence_length=8,active_steps=13))
    (a.out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
