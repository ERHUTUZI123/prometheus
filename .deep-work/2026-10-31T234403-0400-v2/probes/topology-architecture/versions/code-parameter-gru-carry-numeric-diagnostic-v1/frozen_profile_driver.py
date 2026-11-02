"""Development repair: learn future executable behavior, not a latent point.

OWN pre-stage4 model supplies finite continuation distributions conditioned on
the ALREADY observed terminal label. Both Block and ordinary GRU receive this
same learning algorithm; no true intermediate states or operation maps supplied.
"""
import argparse,hashlib,json,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import continuous_primitive_block as core
import owned_parent_driver as old
from memory_live_stage_train import grow_storage
from context_replay128 import Replay128
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def prefix(model,b,family):
    if family=='block':return model.forward_value(b['initial'],b['tokens'],b['lengths'])
    batch=len(b['initial']);w=model.spec['width'];layers=model.spec['layers']
    h0=model.initial(b['initial']).reshape(batch,layers,w).transpose(0,1).contiguous()
    ids=(b['lengths']>0).nonzero(as_tuple=False).flatten();hidden=h0
    if ids.numel():
        words=b['tokens'].index_select(0,ids);lengths=b['lengths'].index_select(0,ids)
        active=torch.arange(words.shape[1],device=words.device)[None]<lengths[:,None]
        words=torch.where(active,words,torch.zeros_like(words))
        packed=pack_padded_sequence(model.tokens(words),lengths.detach().cpu(),batch_first=True,enforce_sorted=False)
        _,h=model.rnn(packed,h0.index_select(1,ids).contiguous());hidden=h0.index_copy(1,ids,h)
    return F.log_softmax(model.readout(model.norm(hidden[-1])),-1),hidden

def suffix(model,carry,word,family,count):
    if family=='block':
        state={k:v[:count]for k,v in carry.items()}
        for symbol in word:
            x=model.token_embedding(torch.full((count,),symbol,device=state['h'].device,dtype=torch.long))
            logp,state,_=model.step_embedded(state,x,output='logits')
        return logp
    state=carry[:,:count].contiguous();tokens=torch.tensor(word,device=state.device,dtype=torch.long)[None].expand(count,-1)
    _,state=model.rnn(model.tokens(tokens),state)
    return F.log_softmax(model.readout(model.norm(state[-1])),-1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in('inputs','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--family',choices=('block','gru'),required=True);p.add_argument('--max-seconds',type=float,default=850);a=p.parse_args()
    rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');completed=0;costs={}
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required, no alternative path')
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        manifest=json.loads((a.inputs/'manifest.json').read_text());cp=a.inputs/'resume.pt'
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('execution source changed')
        if sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('exact OWN checkpoint required before loading')
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        if payload['stage']!=2 or payload['step']!=600 or payload['stage_complete']is not True or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('same OWN completed phase2 source/NN required')
        if a.family=='block':
            if payload.get('schema')!='factorized-value-development-training-v1'or payload['args']['anchor']!=0:raise ValueError('OWN no-anchor factorized parent required')
            model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.003);opt.load_state_dict(payload['optimizer'])
            args=argparse.Namespace(**payload['args'])
        else:
            if payload.get('schema')!='contextual-primitives-training-v1'or payload['args']['arm']!='gru':raise ValueError('OWN ordinary GRU parent required')
            model,opt,args=old.restore_model(payload,device)
        if not old.tensor_state_equal(model.state_dict(),payload['model_state'])or not old.nested_state_equal(opt.state_dict(),payload['optimizer']):raise ValueError('real NN/Adam restoration differs')
        replay=Replay128.from_state_dict(payload['replay_state']);old.restore_rng(payload['rng'])
        if not old.nested_state_equal(old.rng_state(),payload['rng']):raise ValueError('RNG restore differs')
        restore=dict(parent=manifest['parent_run'],checkpoint_sha256=sha(cp),actual_NN_Adam_RNG_restored=True,repeated_old_stages=0)
        rec.emit('actual_restore',**restore);del payload
        def load(stage,split):
            spec=manifest['files'][f'{stage}:{split}'];path=a.inputs/spec['leaf']
            if sha(path)!=spec['sha256']:raise ValueError('original raw data changed')
            with np.load(path,allow_pickle=False)as z:
                if set(z.files)!={'initial','tokens','lengths','targets'}:raise ValueError('raw terminal fields only')
                data={k:z[k]for k in z.files}
            rec.emit('input_opened',stage=stage,split=split,sha256=spec['sha256'],rows=len(data['initial']),bytes=sum(v.nbytes for v in data.values()))
            return data
        datasets={s:{k:load(s,k)for k in('train','valid')}for s in range(4)}
        # Raw-only selection; no targets, teacher outcomes or TEST consulted.
        words=[]
        for tokens,length in zip(datasets[0]['train']['tokens'],datasets[0]['train']['lengths']):
            word=tuple(int(x)for x in tokens[:length])
            if 1<=length<=2 and word not in words:words.append(word)
            if len(words)==4:break
        if len(words)!=4:raise ValueError('four original short TRAIN words required')
        # Before any update, verify explicit carry continuation against normal
        # concatenated execution on the OWN NN, no terminal labels involved.
        model.eval();b,_,_=old.batch(datasets[0]['valid'],np.arange(16),device)
        with torch.no_grad():
            base,carry=prefix(model,b,a.family);differences=[]
            for word in words:
                incremental=suffix(model,carry,word,a.family,16)
                lens=b['lengths'].detach().cpu().tolist();extent=max(lens)+len(word)
                full=torch.zeros((16,extent),device=device,dtype=torch.long)
                for i,length in enumerate(lens):full[i,:length]=b['tokens'][i,:length];full[i,length:length+len(word)]=torch.tensor(word,device=device)
                joined=dict(initial=b['initial'],tokens=full,lengths=b['lengths']+len(word))
                reference=old.forward(model,joined,args,'soft');reference=reference if a.family=='block'else F.log_softmax(reference,-1)
                if not torch.allclose(reference,incremental,atol=2e-4,rtol=2e-5):raise RuntimeError('carry continuation differs from full original execution')
                differences.append(float((reference-incremental).abs().max()))
        rec.emit('actual_carry_concatenation_check',rows=16,words=4,maximum_absolute_differences=differences,no_labels_used=True)
        model.eval();cache=[];tick=time.perf_counter();saved_rng=old.rng_state();teacher_macs=0
        with torch.no_grad():
            for word in words:
                b=dict(initial=torch.arange(8,device=device),tokens=torch.tensor(word,device=device)[None].expand(8,-1),lengths=torch.full((8,),len(word),device=device,dtype=torch.long))
                logp,_=prefix(model,b,a.family);cache.append(logp.exp().detach())
                teacher_macs+=(model.resource_report(batch_size=8,sequence_length=len(word),active_steps=8*len(word))if a.family=='block'else model.proxy(8,len(word),8*len(word)))['proxy_forward_macs']
                reference=old.forward(model,b,args,'soft');reference=reference if a.family=='block'else F.log_softmax(reference,-1)
                if not torch.allclose(reference,logp,atol=1e-6,rtol=1e-5):raise RuntimeError('original prefix execution contract differs')
        old.synchronize(device);target_cache=torch.stack(cache).detach();cache_seconds=time.perf_counter()-tick
        if not old.nested_state_equal(saved_rng,old.rng_state()):raise ValueError('teacher cache changed training RNG')
        with(rec.out/'old-continuation-cache.npz').open('xb')as f:np.savez_compressed(f,probabilities=target_cache.cpu().numpy())
        teacher=dict(words=[list(w)for w in words],selection='first4distinct raw phase0 TRAIN words of length1or2; no outcome selection',examples=32,external_new_labels=0,cache_bytes=target_cache.numel()*4,forward_MAC_proxy=teacher_macs,comparison_forward_queries=32,wall_seconds=cache_seconds,old_model_retained_for_learning=False,teacher_output_file_sha256=sha(rec.out/'old-continuation-cache.npz'))
        rec.emit('old_behavior_cache_frozen',**teacher)
        growth=grow_storage(model,opt,args)
        train=datasets[3]['train'];indices=np.random.default_rng(np.random.SeedSequence([0,8,3,9027])).integers(0,4096,size=(600,64),dtype=np.int64);draws=replay.sample_indices(3,600,16)
        with(rec.out/'stage3-common-indices.npz').open('xb')as f:np.savez_compressed(f,indices=indices)
        with(rec.out/'stage3-replay-draws.npz').open('xb')as f:np.savez_compressed(f,buffer_indices=draws)
        costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_seconds=0.,base_forward_MAC_proxy=0,continuation_forward_MAC_proxy=0,batch_assembly_seconds=0.,current_external_label_reads=0,replay_external_label_reads=0,derived_continuation_target_rows=0,evaluation_wall_seconds=0.,evaluation_forward_MAC_proxy=0,full_training_FLOPs=None)
        curves=[];rec.emit('configuration',family=a.family,source_sha256=manifest['source_sha256'],P=sum(p.numel()for p in model.parameters()),growth=growth,teacher=teacher,feedback='same terminal y plus OWN old-behavior cache indexed by y; no correct call/operation or intermediate truth',objective='NLL + KL(old_cache[word,y] || student(prefix+word)); coefficient1, current48rows only',fixed_updates=600,word_schedule='step modulo4, same two methods')
        for step in range(601):
            rec.check()
            if step%100==0:
                metrics={}
                for s in range(4):
                    metrics[s],timing=old.evaluate(model,datasets[s]['valid'],args,device,rec,modes=('soft',));costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy']
                row=dict(step=step,valid=metrics);curves.append(row);rec.emit('actual_VALID_curve',**row)
            if step==600:break
            tick=time.perf_counter();mixed=replay.mix(train,indices[step,:48],draws[step]);b,l,active=old.batch(mixed,np.arange(64),device);old.synchronize(device);costs['batch_assembly_seconds']+=time.perf_counter()-tick
            model.train();opt.zero_grad(set_to_none=True);tick,cpu=time.perf_counter(),time.process_time();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
            logp,carry=prefix(model,b,a.family);nll=F.nll_loss(logp,b['targets']);wi=step%4;word=words[wi]
            next_logp=suffix(model,carry,word,a.family,48);teacher_prob=target_cache[wi,b['targets'][:48]]
            auxiliary=F.kl_div(next_logp,teacher_prob,reduction='batchmean');loss=nll+auxiliary
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite actual training loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();end.record();old.synchronize(device);elapsed=time.perf_counter()-tick
            base=(model.resource_report(batch_size=64,sequence_length=l,active_steps=active)if a.family=='block'else model.proxy(64,l,active))['proxy_forward_macs']
            tail=(model.resource_report(batch_size=48,sequence_length=len(word),active_steps=48*len(word))if a.family=='block'else model.proxy(48,len(word),48*len(word)))['proxy_forward_macs']
            if a.family=='block':tail+=48*(len(word)-1)*model.internal_states*model.n_states
            for key,value in [('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_seconds',start.elapsed_time(end)/1000.),('base_forward_MAC_proxy',base),('continuation_forward_MAC_proxy',tail),('current_external_label_reads',48),('replay_external_label_reads',16),('derived_continuation_target_rows',48)]:costs[key]+=value
            completed=step+1;rec.emit('actual_update',step=completed,terminal_NLL=float(nll.detach()),continuation_KL=float(auxiliary.detach()),loss=float(loss.detach()),word_index=wi,batch_accuracy=float((logp.argmax(-1)==b['targets']).float().mean()),training_wall_seconds=elapsed,base_forward_MAC_proxy=base,continuation_forward_MAC_proxy=tail,preclip_gradient=float(gn))
        # Save trained NN BEFORE maintenance, so a storage failure cannot erase it.
        def save(leaf,maintenance_complete):
            path=rec.out/leaf;tick=time.perf_counter();value=dict(schema='continuation-feedback-stage4-training-v1',stage=3,step=600,stage_complete=True,maintenance_complete=maintenance_complete,family=a.family,formal_Primary=False,args=vars(args),model_state={n:v.detach().cpu().clone()for n,v in model.state_dict().items()},optimizer=opt.state_dict(),rng=old.rng_state(),replay_state=replay.state_dict(),block_checkpoint=model.checkpoint()if a.family=='block'else None,baseline_spec=model.spec if a.family=='gru'else None,source_sha256=manifest['source_sha256'],teacher=teacher,costs=costs)
            torch.save(value,path);return dict(file=leaf,sha256=sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-tick)
        safety=save('trained-before-maintenance.pt',False)
        if np.any(train['lengths']>48)or np.any(train['tokens'][:,48:]):raise ValueError('no active data may be cropped for replay')
        maintenance=replay.consume(dict(train,tokens=train['tokens'][:,:48]),indices[:,:48],3,emit=rec.emit)
        checkpoint=save('continuation-stage3-step600.pt',True)
        saved_rng=old.rng_state();tick=time.perf_counter();payload=torch.load(rec.out/checkpoint['file'],map_location='cpu',weights_only=False)
        other=(core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'])if a.family=='block'else old.TerminalGRU(**payload['baseline_spec'])).to(device)
        other.load_state_dict(payload['model_state']);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(payload['optimizer'])
        if not old.tensor_state_equal(other.state_dict(),model.state_dict())or not old.nested_state_equal(other_opt.state_dict(),opt.state_dict()):raise RuntimeError('actual NN/Adam checkpoint reload differs')
        b,_,_=old.batch(datasets[3]['valid'],np.arange(16),device);model.eval();other.eval()
        with torch.no_grad():
            if not torch.equal(old.forward(model,b,args,'soft'),old.forward(other,b,args,'soft')):raise RuntimeError('actual reloaded execution differs')
        reload=dict(actual_NN_Adam_forward_equal=True,RNG_factory_neutral=True,wall_seconds=time.perf_counter()-tick)
        del other,other_opt,payload;old.restore_rng(saved_rng)
        rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(checkpoint=checkpoint,actual_updates=600,test_read=False,no_later_selection=True))
        tests={};test_costs=[]
        for s in range(4):
            tests[s]={}
            for split in('test_iid','test_long','test_unseen'):
                data=load(s,split);tests[s][split],timing=old.evaluate(model,data,args,device,rec);test_costs.append(timing)
        tick=time.perf_counter();inference=old.inference_report(model,datasets[3]['valid'],args,device,rec);inference_seconds=time.perf_counter()-tick
        rec.json('result.json',dict(status='ACTUAL_CONTINUATION_FEEDBACK600_COMPLETE',family=a.family,formal_Primary=False,restore=restore,teacher=teacher,growth=growth,curves=curves,tests=tests,costs=costs,resources=old.resources(model,opt,device,replay),checkpoint=checkpoint,premaintenance_safety_checkpoint=safety,reload=reload,maintenance=maintenance,test_costs=test_costs,inference=inference,inference_total_seconds=inference_seconds,driver_wall_seconds=time.perf_counter()-rec.started,source_sha256=manifest['source_sha256'],parent_paid_lineage=manifest['parent_paid_lineage'],scope='development same terminal information; teacher/cache and extra suffix forward/backward charged; cache words are training now, not unseen composition tests'))
        rec.emit('completed',actual_updates=600);print(json.dumps(dict(status='ACTUAL_CONTINUATION_FEEDBACK600_COMPLETE',family=a.family)))
    except BaseException as e:
        rec.json('failure.json',dict(completed_updates=completed,costs=costs,error=repr(e),traceback=traceback.format_exc()));raise
    finally:rec.stream.close()
if __name__=='__main__':main()
