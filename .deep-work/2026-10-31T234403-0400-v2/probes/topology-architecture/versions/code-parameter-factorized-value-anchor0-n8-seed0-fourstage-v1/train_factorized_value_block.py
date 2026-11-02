"""Fixed four-stage development, terminal feedback only, no oracle or search.

Two prospective arms differ ONLY terminal value-interface loss weight0/1.
Same old600*64 then three600*(48new+16Replay128) schedules as existing arms.
New architecture starts from scratch; historical results never pooled.
"""
import argparse,copy,hashlib,json,random,time,traceback
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

RAW=('initial','tokens','lengths','targets')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--anchor',type=int,choices=(0,1),required=True);p.add_argument('--max-seconds',type=float,default=850)
    a=p.parse_args();rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');model=opt=None;reports=[];stage=-1;completed=0
    try:
        if not torch.cuda.is_available():raise RuntimeError('authorized CUDA required; no fallback')
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        torch.manual_seed(0);np.random.seed(0);random.seed(0)
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source differs: '+leaf)
        args=argparse.Namespace(arm='dynamic',states=8,seed=0,lr=.003,block_state='continuous',block_width=32,
            primitive_rank=8,memory_slots=8,slots=1,block_training='soft_only',replay_capacity=128,replay_per_batch=16,
            stages=4,steps=600,anchor=a.anchor,organization=core.ContinuousPrimitiveBlock.organization_policy)
        def load(stage,split):
            spec=manifest['files'][f'{stage}:{split}'];path=a.inputs/spec['leaf']
            if sha(path)!=spec['sha256']:raise ValueError('frozen input differs')
            with np.load(path,allow_pickle=False)as z:
                if set(z.files)!=set(RAW):raise ValueError('raw terminal-feedback fields only')
                data={k:z[k]for k in RAW}
            if any(x.dtype!=np.int64 for x in data.values()) or data['tokens'].ndim!=2:raise ValueError('integer raw arrays required')
            n=len(data['initial']);mask=np.arange(data['tokens'].shape[1])[None]<data['lengths'][:,None]
            if any(data[k].shape!=(n,)for k in ('initial','lengths','targets')) or np.any((data['initial']<0)|(data['initial']>=8)) or np.any((data['targets']<0)|(data['targets']>=8)) or np.any(mask&((data['tokens']<0)|(data['tokens']>=(16 if stage==3 else 15)))):raise ValueError('public input ranges')
            rec.emit('actual_input_opened',stage=stage,split=split,sha256=spec['sha256'],rows=n,array_bytes=sum(x.nbytes for x in data.values()))
            return data
        datasets={s:{k:load(s,k)for k in ('train','valid')}for s in range(4)}
        model=core.ContinuousPrimitiveBlock(8,15).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003)
        replay=Replay128(128,48,0,8);used={};train_sources={s:datasets[s]['train']for s in range(4)}
        rec.emit('configuration',args=vars(args),manifest_sha256=sha(a.inputs/'manifest.json'),source_sha256=manifest['source_sha256'],
            model_resources=model.resource_report(),initial_RNG=old.rng_state()['cpu'].tolist(),
            test_read=False,scope='fresh development; same labels/batches, different factorized organization; not Primary',
            objective='NLL(y|hT)+anchor*mean((hT-stopgrad(sqrtH*normalize(E(y))))^2); same terminal label, no extra intermediate truth')
        def evaluate(stage,split,modes=('soft',)):
            return old.evaluate(model,datasets[stage][split],args,device,rec,modes=modes)
        for stage in range(4):
            rec.check();completed=0;growth=None
            if stage==3:
                before=model(torch.tensor([0],device=device),torch.tensor([[1,3]],device=device),torch.tensor([2],device=device),output='logits').detach()
                growth=grow_storage(model,opt,args)
                after=model(torch.tensor([0],device=device),torch.tensor([[1,3]],device=device),torch.tensor([2],device=device),output='logits').detach()
                if not torch.equal(before,after):raise RuntimeError('vocabulary growth changed old execution')
            data=datasets[stage]['train'];indices=np.random.default_rng(np.random.SeedSequence([0,8,stage,9027])).integers(0,len(data['initial']),size=(600,64),dtype=np.int64)
            draws=replay.sample_indices(stage,600,16)if stage else None
            with(rec.out/f'stage{stage}-common-indices.npz').open('xb')as f:np.savez_compressed(f,indices=indices)
            if stage:
                with(rec.out/f'stage{stage}-replay-draws.npz').open('xb')as f:np.savez_compressed(f,buffer_indices=draws)
            rec.emit('stage_begin',stage=stage,parameters=model.parameter_count(),growth=growth,indices_sha256=sha(rec.out/f'stage{stage}-common-indices.npz'),replay=replay.statistics())
            costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,forward_MAC_proxy=0,batch_assembly_wall_seconds=0.,current_training_label_reads=0,old_replay_training_label_reads=0,evaluation_wall_seconds=0.,evaluation_forward_MAC_proxy=0,evaluation_forward_examples=0,exact_training_FLOPs=None,search_FLOPs=0,search_scope='no search or parameter selection; fixed architecture/loss/budget',anchor_target_encoder_lookups=0,anchor_coordinate_loss_elements=0)
            curves=[]
            for step in range(601):
                rec.check()
                if step%100==0:
                    metrics={}
                    for s in range(stage+1):
                        metrics[s],timing=evaluate(s,'valid')
                        costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy'];costs['evaluation_forward_examples']+=timing['actual_forward_examples']
                    row=dict(stage=stage,step=step,valid=metrics);curves.append(row);rec.emit('actual_VALID_curve',**row)
                if step==600:break
                tick=time.perf_counter()
                if stage:
                    mixed=replay.mix(data,indices[step,:48],draws[step]);b,l,active=old.batch(mixed,np.arange(64),device)
                else:b,l,active=old.batch(data,indices[step],device)
                old.synchronize(device);costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick
                model.train();opt.zero_grad(set_to_none=True);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                tick,cpu=time.perf_counter(),time.process_time();start.record()
                out,carry=model.forward_value(b['initial'],b['tokens'],b['lengths'])
                nll=F.nll_loss(out,b['targets']);target=model.encode_state(b['targets']).detach()
                binding=(carry['h']-target).square().mean();loss=nll+a.anchor*binding
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite actual loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();end.record();old.synchronize(device)
                elapsed=time.perf_counter()-tick;proxy=model.resource_report(batch_size=64,sequence_length=l,active_steps=active)['proxy_forward_macs']
                completed=step+1
                for key,value in [('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_elapsed_seconds',start.elapsed_time(end)/1000.),('forward_MAC_proxy',proxy),('current_training_label_reads',48 if stage else 64),('old_replay_training_label_reads',16 if stage else 0),('anchor_target_encoder_lookups',64),('anchor_coordinate_loss_elements',64*32)]:costs[key]+=value
                rec.emit('actual_update',stage=stage,step=completed,terminal_NLL=float(nll.detach()),value_interface_loss=float(binding.detach()),total_loss=float(loss.detach()),batch_accuracy=float((out.argmax(-1)==b['targets']).float().mean()),preclip_gradient=float(norm),forward_MAC_proxy=proxy,training_wall_seconds=elapsed,active_token_steps=active)
            used[stage]=indices if stage==0 else indices[:,:48]
            maintenance=replay.consume(data,used[stage],stage,emit=rec.emit)
            source_audit=replay.validate_sources(train_sources,used)
            rec.emit('replay_boundary',stage=stage,maintenance=maintenance,source_audit=source_audit)
            path=rec.out/f'factorized-anchor{a.anchor}-stage{stage}-step600.pt';tick=time.perf_counter()
            payload=dict(schema='factorized-value-development-training-v1',stage=stage,step=600,stage_complete=True,formal_Primary=False,
                args=vars(args),block_checkpoint=model.checkpoint(),model_state={n:v.detach().cpu().clone()for n,v in model.state_dict().items()},optimizer=opt.state_dict(),rng=old.rng_state(),replay_state=replay.state_dict(),source_sha256=manifest['source_sha256'],costs=costs)
            torch.save(payload,path);checkpoint=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-tick)
            saved_rng=old.rng_state();tick=time.perf_counter()
            loaded=torch.load(path,map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint']).to(device)
            other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer'])
            if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict()):raise RuntimeError('actual NN/Adam reload mismatch')
            check_batch,_,_=old.batch(datasets[stage]['valid'],np.arange(16),device);model.eval();other.eval()
            with torch.no_grad():
                for mode in ('soft','native'):
                    if not torch.equal(old.forward(model,check_batch,args,mode),old.forward(other,check_batch,args,mode)):raise RuntimeError('actual reloaded execution mismatch')
            del other,other_opt,loaded;old.restore_rng(saved_rng)
            reload=dict(exact_NN_Adam_forward=True,RNG_factory_neutral=True,wall_seconds=time.perf_counter()-tick)
            resources=old.resources(model,opt,device,replay)
            report=dict(stage=stage,costs=costs,checkpoint=checkpoint,reload=reload,resources=resources,curves=curves,growth=growth,replay_maintenance=maintenance)
            reports.append(report);rec.json(f'completed-stage{stage}-before-TEST.json',dict(report=report,test_data_read=False));rec.emit('stage_complete',**report)
        rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(reports=reports,no_more_updates=True,TEST_read=False))
        tests={};test_costs=[]
        for stage in range(4):
            tests[stage]={}
            for split in ('test_iid','test_long','test_unseen'):
                data=load(stage,split);metrics,timing=old.evaluate(model,data,args,device,rec,modes=('soft','native'));tests[stage][split]=metrics;test_costs.append(timing)
        result=dict(status='ACTUAL_FOUR_STAGE_FACTORIZED_DEVELOPMENT_COMPLETE',formal_Primary=False,anchor=a.anchor,
            source_sha256=manifest['source_sha256'],reports=reports,tests=tests,actual_training_updates=2400,
            resources=old.resources(model,opt,device,replay),test_costs=test_costs,teacher_or_intermediate_state_supplied=False,
            cost_boundary='training includes actual backward+Adam time; MAC proxy forward only; anchor compute measured in time but not in MAC; report maintenance/reload/eval/storage and peak separately')
        rec.json('result.json',result);rec.emit('completed',actual_updates=2400,Primary_frozen=False);print(json.dumps(dict(status=result['status'],anchor=a.anchor,parameters=model.parameter_count())))
    except BaseException as e:
        rec.json('failure.json',dict(stage=stage,completed_updates_in_current_stage=completed,completed_stages=reports,error=repr(e),traceback=traceback.format_exc()));raise
    finally:rec.stream.close()
if __name__=='__main__':main()
