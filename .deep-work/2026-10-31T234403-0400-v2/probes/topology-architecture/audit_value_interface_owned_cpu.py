"""One CPU job: two fixed OWN encoder/readout interfaces, zero SGD/task data.

Only public values0..7 are encoded, then the SAVED independent readout applied.
All64 logp entries, learned prototypes/Gram/norms and hypothetical R/R² are
saved. An identity count is an interface property, never task/carry accuracy.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import owned_fcba_core as baseline_core
import value_bridge_core as bridge_core

KEYS=('baseline','bridge')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v)for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v)for v in value)
    return 0
def numpy_bytes(value):
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,dict):return sum(numpy_bytes(v)for v in value.values())
    if isinstance(value,(tuple,list)):return sum(numpy_bytes(v)for v in value)
    return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();cpu=time.process_time();results={}
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if manifest.get('schema')!='two-owned-value-interface-CPU-v1'or set(manifest['models'])!=set(KEYS):raise ValueError('exact two-owned-NN job required')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('exact execution source changed')
        for key in KEYS:
            tick=time.perf_counter();phase_cpu=time.process_time();spec=manifest['models'][key];path=a.inputs/spec['leaf']
            if path.is_symlink()or path.stat().st_size!=spec['bytes']or sha(path)!=spec['sha256']:raise ValueError('exact completed OWN checkpoint required')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            if payload.get('schema')!=spec['schema']or payload.get('stage')!=3 or payload.get('step')!=600 or payload.get('stage_complete')is not True or payload.get('maintenance_complete')is not True or payload['source_sha256']!=spec['parent_source_sha256']:raise ValueError('actual completed/source/schema mismatch')
            factory_rng=torch.get_rng_state();module=baseline_core if key=='baseline'else bridge_core
            model=module.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']);torch.set_rng_state(factory_rng)
            if model.parameter_count()!=spec['P']or model.n_states!=8 or model.internal_states!=32 or not all(torch.equal(model.state_dict()[n],v)for n,v in payload['model_state'].items()):raise ValueError('actual saved NN weights/P/config differ')
            saved_Adam=tensor_bytes(payload['optimizer']);saved_replay=numpy_bytes(payload['replay_state']);before={n:v.detach().clone()for n,v in model.state_dict().items()};model.eval()
            for p in model.parameters():p.requires_grad_(False)
            load_wall=time.perf_counter()-tick;calculation=time.perf_counter()
            with torch.no_grad():
                index=torch.arange(8,dtype=torch.long);E=model.encode_state(index);logits=model.readout(E);logp=F.log_softmax(logits,-1);posterior=F.softmax(logits,-1)
                R_E=F.normalize(posterior@E,dim=-1,eps=1e-6)*torch.linalg.vector_norm(E,dim=-1,keepdim=True)
                R_logits=model.readout(R_E);R_posterior=F.softmax(R_logits,-1)
                R2_E=F.normalize(R_posterior@E,dim=-1,eps=1e-6)*torch.linalg.vector_norm(R_E,dim=-1,keepdim=True)
                Gram=E@E.T;norms=torch.linalg.vector_norm(E,dim=-1);raw_norms=torch.linalg.vector_norm(model.encoder.weight,dim=-1);pred=logp.argmax(-1)
                fixed_residual=torch.linalg.vector_norm(R_E-E,dim=-1);idempotence_residual=torch.linalg.vector_norm(R2_E-R_E,dim=-1)
                normalizer=torch.logsumexp(logp.double(),dim=-1)
                pair_dist=torch.cdist(E,E);pair=pair_dist[torch.triu(torch.ones(8,8,dtype=torch.bool),diagonal=1)]
            arrays=dict(public_value_indices=index.numpy(),encoded_values=E.numpy(),readout_weight=model.readout.weight.detach().numpy(),readout_bias=model.readout.bias.detach().numpy(),log_probabilities=logp.numpy(),decoded_argmax=pred.numpy(),log_probability_row_logsumexp=normalizer.numpy(),prototype_Gram=Gram.numpy(),prototype_norms=norms.numpy(),raw_encoder_row_norms=raw_norms.numpy(),hypothetical_R_E=R_E.numpy(),hypothetical_R2_E=R2_E.numpy(),hypothetical_R_E_fixedpoint_residual=fixed_residual.numpy(),hypothetical_R2_minus_R_residual=idempotence_residual.numpy())
            if any(not np.isfinite(value).all()for value in arrays.values())or float(normalizer.abs().max())>1e-6:raise ValueError('finite normalized numerical output required; identity score is NOT gated')
            geometry_wall=time.perf_counter()-calculation;raw=a.out/(key+'-interface-raw.npz');write=time.perf_counter()
            with raw.open('xb')as f:np.savez_compressed(f,**arrays)
            if not all(torch.equal(model.state_dict()[n],v)for n,v in before.items())or sha(path)!=spec['sha256']:raise ValueError('0SGD audit changed NN/checkpoint')
            alpha=float(model.value_bridge_alpha)if hasattr(model,'value_bridge_alpha')else None
            results[key]=dict(parent=spec['parent'],parent_checkpoint_sha256=spec['sha256'],P=model.parameter_count(),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),checkpoint_bytes=path.stat().st_size,saved_Adam_tensor_bytes=saved_Adam,saved_replay_array_bytes=saved_replay,live_optimizer_bytes=0,SGD_updates=0,task_examples_or_labels_read=0,public_values=list(range(8)),all64_log_probabilities=logp.tolist(),decoded_argmax=pred.tolist(),same_index_argmax_count=int((pred==index).sum()),row_log_normalization=normalizer.tolist(),prototype_Gram=Gram.tolist(),prototype_norms=norms.tolist(),raw_encoder_row_norms=raw_norms.tolist(),prototype_min_pair_L2=float(pair.min()),prototype_equal_norm_max_error=float((norms-32**.5).abs().max()),hypothetical_R_E_fixedpoint_residual=fixed_residual.tolist(),hypothetical_R2_minus_R_residual=idempotence_residual.tolist(),R2_minus_R_max=float(idempotence_residual.max()),value_bridge_alpha=alpha,value_bridge_strength=float(np.tanh(alpha))if alpha is not None else None,NN_unchanged=True,CPU_factory_RNG_restored=True,source_parent_sha256=spec['parent_source_sha256'],raw_file=raw.name,raw_sha256=sha(raw),raw_file_bytes=raw.stat().st_size,raw_array_payload_bytes=sum(v.nbytes for v in arrays.values()),load_and_verify_wall_seconds=load_wall,geometry_wall_seconds=geometry_wall,raw_write_and_rehash_wall_seconds=time.perf_counter()-write,whole_model_wall_seconds=time.perf_counter()-tick,whole_model_CPU_seconds=time.process_time()-phase_cpu,linear_dot_MAC_proxy=10240,MAC_scope='two B8 readoutlinears +two B8 probabilitybasis products +8x8 prototypeGram; softmax/logsoftmax/norm/distance/index/serialization excluded, notfullFLOPs',full_FLOPs=None,scope='hypotheticalnormpreservingR on learnedinitialprototypes, notinstepR(hpre), not fullcarry/transition closure or task accuracy')
            del model,payload,before
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('source changed after audit')
        result=dict(status='COMPLETED_TWO_OWNED_VALUE_INTERFACE_CPU_AUDIT',source_sha256=manifest['source_sha256'],SGD_updates=0,models=results,task_data_or_labels_read=False,new_external_labels=0,parameter_or_alpha_tuning=False,only_two_frozen_NNs=True,CPU_only=True,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),input_file_bytes=sum(spec['bytes']for spec in manifest['models'].values()),source_code_bytes=sum(Path(__file__).with_name(leaf).stat().st_size for leaf in manifest['source_sha256']),full_FLOPs=None,provider_effective_compute=None,scope='8publicinputvalues→actualsavedreadout andhypotheticalR/R² geometryonly; identitycount is nottaskNNscore, no labels/routes/intermediatestates supplied')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],identity_counts={k:v['same_index_argmax_count']for k,v in results.items()})))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),partial_results=results),f,indent=2)
        raise
if __name__=='__main__':main()
