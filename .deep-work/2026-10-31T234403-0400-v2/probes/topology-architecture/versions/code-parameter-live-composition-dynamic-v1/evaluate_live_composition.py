"""ROOT-RUN frozen SOFT closure evaluation, initial SIX OWN phase4 NNs only.

No SGD, fitting, selection, teacher/oracle imports, installation or subprocess.
Restore/check exact NN source+checkpoint/policy BEFORE loading any new labels
or data metadata. Only initial/tokens/lengths reach the model. Suffix transforms
below are evaluator-only consistency rules; they are never model inputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

ALLOWLIST_PROTOCOL = "live-composition-initial-six-v1"
DRIVER_SHA = "cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc"
TRAINER_SHA = "dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6"
REPLAY_SHA = "cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140"
OWNED = {
    "dynamic": dict(parent="parameter-memory-live-stage-n8-seed0-dynamic-stage4-v1", arm="dynamic", P=28381, bytes=613915,
        sha256="c82acdb59be35002ef19c069d79cf43d143902be6d0c2eb2aa54eebd1c11387a",
        core="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f",
        version="continuous-primitive-block-v1", write_policy=None, call_policy=None),
    "static": dict(parent="parameter-memory-live-stage-n8-seed0-static-stage4-v1", arm="static", P=28381, bytes=613718,
        sha256="d33df99870d10c3ea65c3d3f8fdd00c51378ac3b67f732b78fd0ec52d34ebe33",
        core="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f",
        version="continuous-primitive-block-v1", write_policy=None, call_policy=None),
    "gru": dict(parent="parameter-memory-live-stage-n8-seed0-gru-stage4-v1", arm="gru", P=25616, bytes=439720,
        sha256="b40308bcedc3dca768001b2806d7a49ed9ee02a09dd0cd2d81bb1379611412eb",
        core=None, version=None, write_policy=None, call_policy=None),
    "transformer": dict(parent="parameter-memory-live-stage-n8-seed0-transformer-stage4-v1", arm="transformer", P=26760, bytes=512431,
        sha256="2e45733719de88a25f641e8da6f8df62847980b1364fda9d7b2d3f6627886183",
        core=None, version=None, write_policy=None, call_policy=None),
    "delta": dict(parent="parameter-memory-live-delta-stage-n8-seed0-dynamic-stage4-v1", arm="dynamic", P=28381, bytes=615451,
        sha256="12dcb476f170448346be6d571396f66ef56b72c20e2dab92e36f26047d2db8d0",
        core="33a2d9f27f2a12f844727490a66edafe984fdf98cdb053e2c910a5cbcb0c46bd",
        version="continuous-primitive-block-slot-delta-v1", write_policy="slot-delta-residual-write-v1", call_policy=None),
    "credit": dict(parent="parameter-memory-live-credit-stage-n8-seed0-dynamic-stage4-v1", arm="dynamic", P=28381, bytes=615003,
        sha256="02d19c43073e2c0f87589ea2234fdb8a1552c5ed3938f7d5b53057f7b146bf5f",
        core="f41cb0dced3d14e11f77f50587573c1c49b6f6aacc5d009ad01d6f47d6fbb499",
        version="continuous-primitive-block-call-credit-v1", write_policy=None, call_policy="dynamic-CALL-h-read-stop-gradient-v1"),
}
CONDITIONS = ("MIX15", "READ10_control")
SUFFIX_NAMES = ("empty", "SET0_STEP", "SET1_STEP", "CLEAR")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_sources(spec):
    result = {"owned_parent_driver.py": DRIVER_SHA, "memory_live_stage_train.py": TRAINER_SHA, "context_replay128.py": REPLAY_SHA}
    if spec["core"]:result["continuous_primitive_block.py"] = spec["core"]
    return result


def suffix_result(q, suffix):
    """Evaluator-only F, never called inside/model inputs."""
    if suffix == 0:return q
    if suffix == 1:return (q + 1) % 8
    if suffix == 2:return (3 * q + 2) % 8
    if suffix == 3:return 0
    raise ValueError("fixed suffix ID required")


def metrics(predictions, targets, source_rows):
    """Pure scalar arithmetic, fixed condition/base/suffix flattening."""
    if len(predictions) != 2048 or len(targets) != 2048 or len(source_rows) != 2048:
        raise ValueError("fixed2048 row metric contract")
    output = {}
    for condition, name in enumerate(CONDITIONS):
        start = condition * 256 * 4
        base_pred = [int(predictions[start+i*4]) for i in range(256)]
        correct = [[int(predictions[start+i*4+s]) == int(targets[start+i*4+s]) for s in range(4)] for i in range(256)]
        base_correct = [c[0] for c in correct]; denominator = sum(base_correct)
        row = dict(base_examples=256, base_correct_count=denominator, base_accuracy=denominator/256,
            base_correct_source_TRAIN_indices=[int(source_rows[start+i*4]) for i in range(256) if base_correct[i]], suffixes={})
        for suffix, suffix_name in enumerate(SUFFIX_NAMES):
            suffix_correct = sum(c[suffix] for c in correct)
            joint = sum(base_correct[i] and correct[i][suffix] for i in range(256))
            consistent = [int(predictions[start+i*4+suffix]) == suffix_result(base_pred[i], suffix) for i in range(256)]
            row["suffixes"][suffix_name] = dict(examples=256, suffix_correct_count=suffix_correct,
                suffix_accuracy=suffix_correct/256, joint_base_and_suffix_correct_count=joint,
                joint_base_and_suffix_accuracy=joint/256, consistency_count=sum(consistent), consistency=sum(consistent)/256,
                conditional_suffix_accuracy_given_base_correct=joint/denominator if denominator else None,
                conditional_denominator_base_correct=denominator, conditional_numerator=joint,
                consistent_with_wrong_base_count=sum(consistent[i] and not base_correct[i] for i in range(256)),
                note=("identity consistency is tautological" if suffix == 0 else
                      "CLEAR consistency equals CLEAR accuracy; not MIX reuse evidence" if suffix == 3 else
                      "model-relative consistency can hold with a wrong base"))
        output[name] = row
    return output


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value, dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0


def state_hash(state):
    h = hashlib.sha256()
    for name,tensor in sorted(state.items()):
        x=tensor.detach().cpu().contiguous()
        h.update(json.dumps([name,str(x.dtype),list(x.shape)],separators=(",",":")).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def load_module(path,name):
    previous=sys.modules.get(name)
    if previous is not None and Path(getattr(previous,"__file__","")).resolve()!=path.resolve():
        raise RuntimeError("owned module import collision: "+name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def restore(spec,payload,source,device):
    if (payload.get("schema")!="contextual-primitives-training-v1" or payload.get("stage")!=3
            or payload.get("step")!=600 or payload.get("stage_complete") is not True
            or payload.get("formal_Primary") is not False or payload["args"]["arm"]!=spec["arm"]
            or payload["source_sha256"]!=expected_sources(spec)):
        raise ValueError("exact OWN complete phase4 checkpoint/source required")
    for leaf,digest in expected_sources(spec).items():
        path=source/leaf
        if path.is_symlink() or sha(path)!=digest:raise ValueError("exact owned execution source mismatch: "+leaf)
    if {p.name for p in source.glob("*.py")} != set(expected_sources(spec))|{Path(__file__).name}:
        raise ValueError("unlisted executable source in isolated evaluator namespace")
    if any((source/leaf).exists() for leaf in ("build_live_composition_data.py","original_generator.py","memory_live_stage_data.py")):
        raise ValueError("teacher/oracle source must not enter the NN evaluator namespace")
    if spec["core"]:
        core=load_module(source/"continuous_primitive_block.py","continuous_primitive_block")
        block=payload["block_checkpoint"]
        if (block.get("version")!=spec["version"] or block.get("write_update_policy")!=spec["write_policy"]
                or block.get("call_credit_policy")!=spec["call_policy"]):
            raise ValueError("checkpoint execution policy/version differs")
        model=core.ContinuousPrimitiveBlock.from_checkpoint(block).to(device)
        if getattr(model,"checkpoint_version",None)!=spec["version"]:raise ValueError("loaded core version differs")
    else:
        driver=load_module(source/"owned_parent_driver.py","owned_parent_driver")
        cls=driver.TerminalGRU if spec["arm"]=="gru" else driver.TerminalTransformer
        model=cls(**payload["baseline_spec"]).to(device)
    model.load_state_dict(payload["model_state"],strict=True)
    layouts=[]
    for name,tensor in model.state_dict(keep_vars=True).items():
        saved=payload["model_state"][name];factory_stride=tuple(tensor.stride())
        if tensor.shape!=saved.shape or tensor.dtype!=saved.dtype:raise ValueError("saved tensor shape/dtype differs")
        if tuple(tensor.stride())!=tuple(saved.stride()):
            tensor.data=torch.empty_strided(saved.shape,saved.stride(),dtype=saved.dtype,device=device).copy_(saved)
        if not torch.equal(tensor.detach().cpu(),saved):raise RuntimeError("actual saved NN tensor differs")
        layouts.append(dict(name=name,shape=list(saved.shape),saved_stride=list(saved.stride()),factory_stride=list(factory_stride)))
    if sum(p.numel() for p in model.parameters())!=spec["P"]:raise ValueError("actual all-parameter count differs")
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,layouts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","frozen-source","data","manifest","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--model",choices=tuple(OWNED),required=True)
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--max-seconds",type=float,default=850)
    a=p.parse_args()
    if a.out.exists() or not 0<a.max_seconds<=850:raise ValueError("new output and bounded850s required")
    global torch,np,F
    import numpy as np
    import torch
    from torch.nn import functional as F
    began=time.perf_counter();deadline=began+a.max_seconds;a.out.mkdir(parents=True)
    stream=(a.out/"records.jsonl").open("x",buffering=1)
    model=None;before_hash=None;checkpoint_sha=None;completed_rows=0;costs={}
    def check():
        if time.perf_counter()>deadline:raise TimeoutError("bounded closure inference deadline")
    def emit(kind,**value):
        stream.write(json.dumps(dict(kind=kind,session_wall_seconds=time.perf_counter()-began,**value),sort_keys=True,allow_nan=False)+"\n")
    device=torch.device(a.device)
    def sync():torch.cuda.synchronize(device)
    def memory():
        return dict(cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)))
    try:
        if not torch.cuda.is_available():raise RuntimeError("authorized CUDA required; no fallback")
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        spec=OWNED[a.model];manifest=json.loads(a.manifest.read_text())
        if (manifest.get("schema")!="live-composition-evaluator-inputs-v1"
                or manifest.get("initial_model_allowlist_protocol")!=ALLOWLIST_PROTOCOL
                or manifest["model_key"]!=a.model or manifest["model_spec"]!=spec):
            raise ValueError("immutable initial-six allowlist manifest required")
        if sha(Path(__file__))!=manifest["evaluator_source_sha256"]:
            raise ValueError("frozen evaluator source differs from protocol")
        if a.checkpoint.is_symlink() or a.checkpoint.stat().st_size!=spec["bytes"] or sha(a.checkpoint)!=spec["sha256"]:
            raise ValueError("checkpoint is outside exact OWN allowlist")
        checkpoint_sha=sha(a.checkpoint);tick=time.perf_counter()
        # Exact own hash and source guards precede the authorized OWN unpickle.
        payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
        model,layouts=restore(spec,payload,a.frozen_source,device)
        before_hash=state_hash(model.state_dict())
        if before_hash!=state_hash(payload["model_state"]):raise RuntimeError("restored NN state hash differs")
        saved_optimizer=tensor_bytes(payload["optimizer"])
        resources=dict(parameters=spec["P"],parameter_payload_bytes=tensor_bytes(dict(model.named_parameters())),
            registered_buffer_bytes=tensor_bytes(dict(model.named_buffers())),checkpoint_bytes=spec["bytes"],
            saved_training_optimizer_tensor_bytes=saved_optimizer,live_inference_optimizer_tensor_bytes=0,
            frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))
        if spec["core"]:
            resources["carry"]=dict(scalars_per_example=model.resource_report()["persistent_example_state_scalars"],
                payload_bytes_per_example=model.resource_report()["persistent_example_state_payload_bytes"],
                scope="logical h/K/V execution carry; reset per example, not peak/autograd memory")
        elif spec["arm"]=="gru":
            count=model.spec["layers"]*model.spec["width"]
            resources["carry"]=dict(scalars_per_example=count,payload_bytes_per_example=count*next(model.parameters()).element_size(),
                scope="recurrent hidden payload; wrapper resets/packing and other workspace counted separately")
        else:
            resources["carry"]=dict(scalars_per_example=None,payload_bytes_per_example=None,
                scope="this Transformer wrapper recomputes full prefix; no persistent cached-carry interface; not zero workspace")
        restore_seconds=time.perf_counter()-tick
        emit("actual_frozen_NN_restored_before_new_labels",model=a.model,checkpoint_sha256=checkpoint_sha,
             model_state_sha256=before_hash,source_sha256=expected_sources(spec),layouts=layouts,
             policy=dict(checkpoint_version=spec["version"],write_policy=spec["write_policy"],call_policy=spec["call_policy"]),
             resources=resources,restore_wall_seconds=restore_seconds,new_labels_loaded=False)
        del payload;gc.collect()
        # Only now may the evaluator open new metadata/terminal labels.
        tick=time.perf_counter()
        for leaf,expected in manifest["data_files"].items():
            file=a.data/leaf
            if file.is_symlink() or file.stat().st_size!=expected["bytes"] or sha(file)!=expected["sha256"]:
                raise ValueError("frozen common composition input bytes changed")
        metadata=json.loads((a.data/"metadata.json").read_text())
        if (metadata.get("schema")!="live-composition-closure-data-v1" or metadata["rows"]!=2048
                or metadata["initial_model_allowlist_protocol"]!=ALLOWLIST_PROTOCOL
                or metadata["initial_six_model_allowlist"]!=manifest["initial_six_model_allowlist"]):
            raise ValueError("common once-generated closure data differs")
        with np.load(a.data/"closure.npz",allow_pickle=False) as z:
            expected={"initial","tokens","lengths","targets","source_train_index","base_local_index","condition","suffix"}
            if set(z.files)!=expected:raise ValueError("only fixed raw/evaluator fields accepted")
            data={k:z[k] for k in z.files}
        if (any(v.dtype!=np.int64 for v in data.values()) or data["tokens"].shape!=(2048,48)
                or any(v.shape!=(2048,) for k,v in data.items() if k!="tokens")
                or np.any((data["lengths"]<1)|(data["lengths"]>48))
                or np.any((data["targets"]<0)|(data["targets"]>=8))
                or np.any((data["initial"]<0)|(data["initial"]>=8))
                or not np.array_equal(data["condition"],np.repeat(np.arange(2),1024))
                or not np.array_equal(data["base_local_index"],np.tile(np.repeat(np.arange(256),4),2))
                or not np.array_equal(data["suffix"],np.tile(np.arange(4),512))):
            raise ValueError("fixed2048 flattening/ranges violated")
        active=np.arange(48)[None]<data["lengths"][:,None]
        if np.any(active&((data["tokens"]<0)|(data["tokens"]>=16))):raise ValueError("active raw token range differs")
        data_load_seconds=time.perf_counter()-tick
        costs=dict(restore_wall_seconds=restore_seconds,data_load_wall_seconds=data_load_seconds,
            actual_forward_rows=0,actual_active_token_steps=0,forward_MAC_proxy=0,forward_wall_seconds=0.,
            forward_CPU_seconds=0.,forward_CUDA_stream_seconds=0.,batch_assembly_and_H2D_wall_seconds=0.,
            prediction_label_comparisons=2048,NLL_target_reads=2048,unique_new_label_rows_loaded=2048,
            metric_rules="evaluator only; no labels/condition/suffix/source indices enter model")
        emit("configuration",model=a.model,resources=resources,common_data_sha256=metadata["file"]["sha256"],
             metadata_sha256=sha(a.data/"metadata.json"),data_array_bytes=sum(v.nbytes for v in data.values()),
             teacher_generation_costs_paid_once=metadata["teacher_costs"],teacher_generation_namespace=manifest["data_run"],
             teacher_generation_measured_resources=manifest["data_generation_measured_resources"],
             model_training_cost_lineage=manifest["model_training_cost_lineage"],
             supervision_to_model="initial/raw tokens/length only",mode="soft only",SGD_updates=0,
             evaluator_source_sha256=sha(Path(__file__)),formal_Primary=False,
             label_timing="new labels opened only after frozen NN restoration/hash/policy verification")
        torch.cuda.reset_peak_memory_stats(device);outputs=[]
        with torch.no_grad():
            for start in range(0,2048,64):
                check();stop=start+64;sl=slice(start,stop);length=max(1,int(data["lengths"][sl].max()))
                tick=time.perf_counter()
                batch={k:torch.as_tensor(data[k][sl] if k!="tokens" else data[k][sl,:length],device=device)
                       for k in ("initial","tokens","lengths")}
                sync();costs["batch_assembly_and_H2D_wall_seconds"]+=time.perf_counter()-tick
                tick,cpu=time.perf_counter(),time.process_time()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
                if spec["core"]:logp=model(batch["initial"],batch["tokens"],batch["lengths"],execution="soft",output="logits")
                else:logp=F.log_softmax(model(batch["initial"],batch["tokens"],batch["lengths"]),-1)
                end.record();sync();elapsed=time.perf_counter()-tick;cpu_elapsed=time.process_time()-cpu;gpu_elapsed=begin.elapsed_time(end)/1000.
                if logp.shape!=(64,8) or not bool(torch.isfinite(logp).all()):raise FloatingPointError("invalid frozen model output")
                active_steps=int(data["lengths"][sl].sum())
                report=(model.resource_report(batch_size=64,sequence_length=length,active_steps=active_steps,execution="soft")
                        if spec["core"] else model.proxy(64,length,active_steps))
                proxy=report["proxy_forward_macs"];outputs.append(logp.cpu().numpy());completed_rows=stop
                for key,value in (("actual_forward_rows",64),("actual_active_token_steps",active_steps),("forward_MAC_proxy",proxy),
                    ("forward_wall_seconds",elapsed),("forward_CPU_seconds",cpu_elapsed),("forward_CUDA_stream_seconds",gpu_elapsed)):
                    costs[key]+=value
                emit("actual_frozen_forward",first_row=start,stop_row=stop,rows=64,active_token_steps=active_steps,
                     padded_length=length,forward_MAC_proxy=proxy,wall_seconds=elapsed,CPU_seconds=cpu_elapsed,
                     CUDA_stream_seconds=gpu_elapsed,labels_or_rules_supplied_to_NN=False)
        tick=time.perf_counter();logp=np.concatenate(outputs);prediction=logp.argmax(-1)
        normalizer=float(np.max(np.abs(np.logaddexp.reduce(logp,axis=-1))))
        if normalizer>1e-5:raise RuntimeError("frozen logprob normalization failed")
        scored=metrics(prediction.tolist(),data["targets"].tolist(),data["source_train_index"].tolist())
        nll=-logp[np.arange(2048),data["targets"]]
        file=a.out/"frozen-closure-outputs.npz"
        with file.open("xb") as dest:np.savez_compressed(dest,log_probabilities=logp,predictions=prediction,targets=data["targets"],
            source_train_index=data["source_train_index"],base_local_index=data["base_local_index"],condition=data["condition"],
            suffix=data["suffix"],terminal_NLL=nll)
        costs["metric_and_output_wall_seconds"]=time.perf_counter()-tick
        after_hash=state_hash(model.state_dict())
        if after_hash!=before_hash or sha(a.checkpoint)!=checkpoint_sha:raise RuntimeError("frozen NN/checkpoint changed")
        for leaf,digest in expected_sources(spec).items():
            if sha(a.frozen_source/leaf)!=digest:raise RuntimeError("owned NN source changed")
        if sha(Path(__file__))!=manifest["evaluator_source_sha256"]:raise RuntimeError("evaluator source changed")
        result=dict(status="FROZEN_LIVE_COMPOSITION_EVALUATION_COMPLETED",model=a.model,parent=spec["parent"],
            resources=resources,source_sha256=expected_sources(spec),checkpoint_sha256=checkpoint_sha,
            execution_policy=dict(version=spec["version"],write=spec["write_policy"],call=spec["call_policy"]),
            conditions=scored,costs=costs,observed_memory=memory(),data_file_sha256=metadata["file"]["sha256"],
            metadata_sha256=sha(a.data/"metadata.json"),teacher_generation_costs_paid_once=metadata["teacher_costs"],
            teacher_generation_measured_resources=manifest["data_generation_measured_resources"],
            common_input_storage=dict(compressed_data_bytes=metadata["file"]["bytes"],
                numeric_array_bytes=sum(v.nbytes for v in data.values()),
                NN_source_code_bytes=sum((a.frozen_source/leaf).stat().st_size for leaf in expected_sources(spec)),
                evaluator_source_bytes=Path(__file__).stat().st_size),
            training_cost_lineage=manifest["model_training_cost_lineage"],
            raw_outputs=dict(file=file.name,sha256=sha(file),bytes=file.stat().st_size,rows=2048),
            state_hash_before=before_hash,state_hash_after=after_hash,NN_and_checkpoint_bytes_unchanged=True,
            max_log_normalizer_error=normalizer,SGD_updates=0,formal_Primary=False,driver_wall_seconds=time.perf_counter()-began,
            scope="postfreeze DEVELOPMENT closure on fixed originalTRAIN prefix grid; not oldTEST replacement or general-computation proof",
            conditional_scope="base-correct subsets differ by model; denominators/indices reported, not unmatched causal superiority",
            degeneracies="empty consistency tautological; CLEAR consistency equals CLEAR accuracy; keep MIX and READ control separate",
            input_information="no teacher/oracle source; suffix F only evaluator; NN sees initial/tokens/lengths",
            cost_scope="forward MAC proxies exclude backward/optimizer; recorded nested wall/stream/RSS/allocator scopes are not isolated throughput or full hardware cost",
            initial_model_allowlist_protocol=ALLOWLIST_PROTOCOL)
        with (a.out/"result.json").open("x") as dest:json.dump(result,dest,indent=2,sort_keys=True,allow_nan=False);dest.write("\n")
        emit("completed",rows=2048,SGD_updates=0,NN_and_checkpoint_bytes_unchanged=True)
        print(json.dumps(dict(status=result["status"],model=a.model,rows=2048,out=str(a.out),wall_seconds=result["driver_wall_seconds"])))
    except BaseException as error:
        emit("failed",error_type=type(error).__name__,error=str(error),completed_rows=completed_rows)
        (a.out/"exception.txt").write_text(traceback.format_exc())
        with (a.out/"failure.json").open("x") as dest:json.dump(dict(error_type=type(error).__name__,error=str(error),
            completed_rows=completed_rows,costs=costs,wall_seconds=time.perf_counter()-began),dest,indent=2)
        raise
    finally:stream.close()


if __name__=="__main__":main()
