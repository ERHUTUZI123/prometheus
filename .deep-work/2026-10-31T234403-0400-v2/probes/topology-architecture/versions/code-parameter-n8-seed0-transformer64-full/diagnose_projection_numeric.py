"""One-proposal alpha=1 numerical counterfactual on an ACTUAL N16 checkpoint.

Only this campaign's trusted native-v7 topo stage1-step700 checkpoint is valid.
Restore its optimizer and RNG, consume saved next minibatch index700, compute
ONE original CE+MI+.01geometry gradient and ONE Adam proposal, then compare two
applications of that SAME proposal. No old stage or second optimizer step is run.
Validation is diagnostic only, never a training/selection signal. No new teacher
operations, states, paths or correct routes are imported. Run in the existing
external bounded GPU sandbox; this module is not itself a sandbox.

MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate; Transformer primary.
"""
import argparse
import copy
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import topology_model
from continuous_native_model import ContinuousNativeModel
from integrate_native_repair import MAX_FILE_BYTES, Recorder, plain, sha256, synchronize


STAGE = 1
STEP = 700
ALPHABET = (1, 2, 3, 4, 5)
BATCH = 64



def _same_tensor_bits(left, right):
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    # Inspect raw bytes, including signed zero, without executing checkpoint code.
    a = left.detach().cpu().contiguous().numpy().tobytes()
    b = right.detach().cpu().contiguous().numpy().tobytes()
    return a == b


def _copy_owned_layout(source, target):
    """Restore finite, nonoverlapping owned2D layouts; reject exotic strides."""
    shape, stride = tuple(source.shape), tuple(source.stride())
    if shape != tuple(target.shape) or source.dtype != target.dtype:
        raise ValueError("owned checkpoint shape/dtype differs during layout restore")
    if source.stride() == target.stride():
        return target, False
    if source.ndim != 2 or any(n <= 0 for n in shape) or any(s <= 0 for s in stride):
        raise ValueError("only finite positive nonoverlapping2D stride repairs are allowed")
    offsets = {i*stride[0]+j*stride[1] for i in range(shape[0]) for j in range(shape[1])}
    allocation_elements = 1+(shape[0]-1)*stride[0]+(shape[1]-1)*stride[1]
    if len(offsets) != source.numel() or allocation_elements > 100000:
        raise ValueError("overlapping or excessive owned source stride allocation")
    restored = torch.empty_strided(shape,stride,device=target.device,dtype=target.dtype)
    restored.copy_(source.to(device=target.device,dtype=target.dtype))
    return restored, True


@torch.no_grad()
def restore_parameter_layouts(model, state):
    records=[]
    for name,p in model.named_parameters():
        source=state[name]
        before_stride=tuple(p.stride())
        identity=id(p)
        value,changed=_copy_owned_layout(source,p.detach())
        if changed:
            p.data=value
        equal=_same_tensor_bits(source,p)
        if id(p)!=identity or tuple(p.stride())!=tuple(source.stride()) or not equal:
            raise RuntimeError("source parameter value/stride/object restoration failed")
        records.append(dict(name=name,shape=list(source.shape),source_stride=list(source.stride()),
            source_storage_offset=int(source.storage_offset()),constructor_stride=list(before_stride),
            restored_stride=list(p.stride()),restored_storage_offset=int(p.storage_offset()),
            source_device=str(source.device),restored_device=str(p.device),layout_replaced=changed,
            exact_tensor_bits=True,same_Parameter_object=True))
    invalidate = getattr(model, '_invalidate_derived_caches', None)
    if callable(invalidate):
        invalidate()
    return records


@torch.no_grad()
def optimizer_layout_report(optimizer, saved):
    records=[]
    if len(optimizer.param_groups)!=len(saved['param_groups']):
        raise ValueError("optimizer group count differs from owned source")
    for group_index,(actual_group,source_group) in enumerate(zip(optimizer.param_groups,saved['param_groups'])):
        if len(actual_group['params'])!=len(source_group['params']):
            raise ValueError("optimizer parameter order/count differs")
        for parameter_index,(parameter,source_id) in enumerate(zip(actual_group['params'],source_group['params'])):
            source_state=saved['state'].get(source_id,{})
            actual_state=optimizer.state[parameter]
            if set(source_state)!=set(actual_state):
                raise ValueError("loaded optimizer state fields differ")
            for key,source in source_state.items():
                if not torch.is_tensor(source):
                    continue
                target=actual_state[key]
                before_stride=tuple(target.stride())
                value,changed=_copy_owned_layout(source,target)
                if changed:
                    actual_state[key]=value
                    target=value
                equal=_same_tensor_bits(source,target)
                if tuple(target.stride())!=tuple(source.stride()) or not equal:
                    raise RuntimeError("owned optimizer state value/stride differs")
                records.append(dict(group=group_index,parameter_index=parameter_index,state_key=key,
                    shape=list(source.shape),source_stride=list(source.stride()),
                    source_storage_offset=int(source.storage_offset()),loaded_stride=list(before_stride),
                    restored_stride=list(target.stride()),source_device=str(source.device),
                    restored_device=str(target.device),dtype=str(target.dtype),
                    layout_replaced=changed,exact_tensor_bits=True))
    return records


def load_source_model(checkpoint, device):
    """Construct the same source type; load dynamic bank, geometry and references."""
    state, extra = checkpoint["model"], checkpoint.get("model_extra", {})
    n, d = state["encoder"].shape
    count, a, b = state["operations"].shape
    vocab_plus_pad, emb_dim = state["token.weight"].shape
    tasks = state["task.weight"].shape[0]
    slots = state["call.weight"].shape[0] // (count + 1)
    if ((n, d) != (16, 16) or (a, b) != (d, d) or not 1 <= count <= 24
            or vocab_plus_pad != 7 or tasks != 3 or slots != 2):
        raise ValueError("checkpoint is not the declared actual N16 stage1 architecture")
    geometric = extra.get("continuous_monodromy")
    if not geometric or "path_omega" not in state:
        raise ValueError("actual continuous-primary checkpoint required")
    model = ContinuousNativeModel(n, vocab=vocab_plus_pad-1, max_len=18,
        n_tasks=tasks, latent_factor=d//n, protection=True,
        seed=int(checkpoint["args"]["seed"]), n_ops=count, slots=slots,
        emb_dim=emb_dim, max_bank=int(extra.get("max_bank", 24)),
        initial_omega=geometric.get("initial_omega", (0., 0., 0., 0.)),
        continuous_lift_seconds=min(60., float(geometric.get("continuous_lift_seconds", 110.))),
        **extra.get("promotion_settings", {})).to(device)
    model.load_state_dict(state)
    model.load_extra_state_dict(extra)
    model._owned_parameter_layout_report = restore_parameter_layouts(model,state)
    model.stage = STAGE
    if model.compilation != "component" or model.credit != "log" or not model.protection:
        raise ValueError("source execution/credit/protection configuration differs")
    if {int(r["task"]) for r in model.reference} != {0}:
        raise ValueError("actual step700 must have only admitted old task0")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
        if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
            raise ValueError("diagnostic expects finite actual float32 parameters")
    return model


def load_data(path, recorder):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: np.asarray(archive[k], dtype=np.int64)
                for k in ("tokens", "starts", "targets", "task_ids")}
    n = len(data["starts"])
    if (data["tokens"].shape != (n, 18) or data["targets"].shape != (n,)
            or data["task_ids"].shape != (n,) or not 1 <= n <= 16384
            or np.any(data["task_ids"] != STAGE)
            or np.any(data["starts"] < 0) or np.any(data["starts"] >= 16)
            or np.any(data["targets"] < 0) or np.any(data["targets"] >= 16)
            or set(int(v) for v in np.unique(data["tokens"]))-set((0,)+ALPHABET)):
        raise ValueError("observed data shape/task/alphabet bounds violated")
    recorder.emit("input_dataset", file=str(path), sha256=sha256(path),
                  examples=n, array_bytes=sum(v.nbytes for v in data.values()))
    return data


def finite_float32_statistics(direct, reconstructed):
    """Monotone IEEE754 ranks, treating signed zeros as adjacent bit patterns."""
    a = np.ascontiguousarray(direct.detach().cpu().numpy(), dtype=np.float32)
    b = np.ascontiguousarray(reconstructed.detach().cpu().numpy(), dtype=np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite or mismatched proposal tensors")
    aa, bb = a.view(np.uint32), b.view(np.uint32)
    def rank(bits):
        unsigned = bits.astype(np.int64)
        return np.where((bits & np.uint32(0x80000000)) != 0,
                        np.int64(0xffffffff)-unsigned, unsigned+np.int64(0x80000000))
    ulp = np.abs(rank(aa)-rank(bb))
    absolute = np.abs(a.astype(np.float64)-b.astype(np.float64))
    changed = ulp != 0
    return dict(elements=a.size, bit_different=int(changed.sum()),
        numerically_different=int(np.count_nonzero(a != b)),
        max_absolute_difference=float(absolute.max(initial=0.)),
        max_ulp_distance=int(ulp.max(initial=0)),
        mean_ulp_distance=float(ulp.mean()) if ulp.size else 0.,
        changed_mean_ulp_distance=float(ulp[changed].mean()) if changed.any() else 0.,
        direct_tensor_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        reconstructed_tensor_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        ulp_convention="monotone float32 bit-rank; -0,+0 adjacent")


def assignment_diagnostic(scores, solve, recorder):
    """Exact best/runner-up assignment by forbidding each winning edge in turn."""
    a = np.asarray(scores, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("finite square matching scores required")
    row, col = solve(-a)
    best = float(a[row, col].sum())
    permutation = np.empty(len(row), dtype=np.int64)
    permutation[row] = col
    runner_score, runner_perm, forbidden = None, None, None
    for i, j in zip(row, col):
        recorder.check()
        costs = -a.copy()
        costs[i, j] = np.inf
        try:
            rr, cc = solve(costs)
        except ValueError:
            # D=1 has no distinct assignment; D16 here normally always feasible.
            continue
        score = float(a[rr, cc].sum())
        if runner_score is None or score > runner_score:
            runner_score = score
            other = np.empty(len(rr), dtype=np.int64)
            other[rr] = cc
            runner_perm, forbidden = other, [int(i), int(j)]
    return dict(permutation=permutation, best_score=best,
        runner_up_score=runner_score,
        best_minus_runner_up=None if runner_score is None else best-runner_score,
        runner_up_permutation=runner_perm, runner_up_winning_edge_forbidden=forbidden,
        scope="global best distinct permutation, union of every excluded winning edge")


@torch.no_grad()
def manual_old_certificate(model, references, recorder):
    checks = []
    for old in references:
        recorder.check()
        new = model.graph(old["task"], old["tokens"])
        okay, pairs = model.related(old, new)
        checks.append(dict(task=int(old["task"]), related=bool(okay), relation_pairs=int(pairs)))
    return dict(passed=all(c["related"] for c in checks), checks=checks,
                parameter_projection_performed=False)


@torch.no_grad()
def matching_scores(model):
    bank = [w.detach().cpu().double().numpy().copy() for w in model.operations]
    # Match proposal_eval EXACTLY: it extracts all vocabulary symbols,
    # including the unused task1 symbol6. GEMM shape is part of this numeric test.
    proposal_alphabet = tuple(range(1, model.vocab+1))
    tokens = torch.as_tensor([proposal_alphabet], dtype=torch.long, device=model.encoder.device)
    tasks = torch.as_tensor([STAGE], dtype=torch.long, device=model.encoder.device)
    probabilities = torch.softmax(model.calls(tokens, tasks)[0]/model.temperature, -1)
    soft = model.soft_ops()
    joint = {}
    for index, token in enumerate(proposal_alphabet):
        if token not in ALPHABET:
            continue
        matrix = torch.eye(model.d, device=soft.device, dtype=soft.dtype)
        for slot in range(model.slots):
            matrix = matrix @ torch.einsum("k,kij->ij", probabilities[index, slot], soft)
        joint[int(token)] = matrix.clamp_min(1e-30).log().cpu().double().numpy().copy()
    return bank, joint


@torch.no_grad()
def validation_evaluation(model, data, device, recorder, costs):
    model.eval()
    log_probabilities, native, joint = [], [], []
    synchronize(device)
    began = time.perf_counter()
    for start in range(0, len(data["starts"]), BATCH):
        recorder.check()
        b = {k: torch.as_tensor(v[start:start+BATCH], dtype=torch.long, device=device)
             for k, v in data.items()}
        probabilities = model(b["tokens"], b["starts"], b["task_ids"])
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("nonfinite diagnostic soft log probability")
        log_probabilities.append(probabilities.cpu().double().numpy())
        native.append(model.hard_eval(b["tokens"], b["starts"], b["task_ids"]).argmax(-1).cpu().numpy())
        joint.append(model.proposal_eval(b["tokens"], b["starts"], b["task_ids"]).argmax(-1).cpu().numpy())
    synchronize(device)
    costs["validation_seconds"] += time.perf_counter()-began
    result = np.concatenate(log_probabilities)
    native, joint = np.concatenate(native), np.concatenate(joint)
    targets = data["targets"]
    nll = -result[np.arange(len(targets)), targets]
    costs["soft_validation_episodes"] += len(targets)
    costs["native_validation_episodes"] += len(targets)
    costs["joint_validation_episodes"] += len(targets)
    model.train()
    return dict(examples=len(targets), soft_CE=float(nll.mean()),
        soft_accuracy=float(np.mean(result.argmax(-1)==targets)),
        native_accuracy=float(np.mean(native==targets)),
        joint_accuracy=float(np.mean(joint==targets)),
        soft_log_probabilities=result, target_nll=nll,
        native_predictions=native, joint_predictions=joint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=60.)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 60 or args.out.exists():
        parser.error("new output directory and internal wall<=60s required")
    for pathname in (args.checkpoint, args.train, args.valid, args.batches):
        if not pathname.is_file() or pathname.stat().st_size > MAX_FILE_BYTES:
            parser.error("trusted input missing or over100MiB cap")
    recorder = Recorder(args.out, args.max_seconds)
    model = optimizer = None
    original_solver = topology_model.linear_sum_assignment
    phase = "startup"
    costs = dict(gradient_forwards=0, gradient_backwards=0, Adam_proposals=0,
        training_sample_uses=0, gradient_and_Adam_seconds=0., application_seconds={},
        validation_seconds=0., soft_validation_episodes=0, native_validation_episodes=0,
        joint_validation_episodes=0, hungarian_assignments=0,
        hungarian_seconds=0., assignments_by_phase={}, old_certificate_checks=0,
        old_certificate_relation_pairs=0)
    def measured_solve(matrix):
        nonlocal phase
        recorder.check()
        began = time.perf_counter()
        try:
            return original_solver(matrix)
        finally:
            elapsed = time.perf_counter()-began
            costs["hungarian_assignments"] += 1
            costs["hungarian_seconds"] += elapsed
            item = costs["assignments_by_phase"].setdefault(phase, dict(calls=0, seconds=0.))
            item["calls"] += 1
            item["seconds"] += elapsed
    topology_model.linear_sum_assignment = measured_solve
    try:
        device = torch.device(args.device)
        torch.set_num_threads(1)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("authorized GPU unavailable; no silent CPU substitute")
            torch.cuda.set_per_process_memory_fraction(.3, device)
        provenance = dict(inputs={k:dict(file=str(getattr(args,k)),sha256=sha256(getattr(args,k)))
                                  for k in ("checkpoint","train","valid","batches")},
            sources={name:sha256(Path(__file__).with_name(name)) for name in
                (Path(__file__).name,"continuous_native_model.py","continuous_monodromy.py",
                 "topology_model.py","semantic_promotion_model.py","train_monodromy_extension.py",
                 "integrate_native_repair.py")})
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("args", {})
        if (checkpoint.get("schema") != "terminal-architecture-checkpoint-v1"
                or checkpoint.get("model_name") != "topo" or checkpoint.get("stage") != STAGE
                or checkpoint.get("step") != STEP or config.get("scale") != 16
                or not config.get("continuous_monodromy") or not config.get("native_promotion")
                or config.get("router_mi") != .05 or config.get("geometry_weight") != .01
                or config.get("steps") != 800 or config.get("batch") != BATCH):
            raise ValueError("only actual primaryN16 topo stage1-step700 config is allowed")
        model = load_source_model(checkpoint, device)
        model.set_geometry_deadline(time.monotonic()+max(.001,recorder.deadline-time.perf_counter()))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_layouts=optimizer_layout_report(optimizer,checkpoint["optimizer"])
        recorder.emit("owned_checkpoint_layout_restored",
            parameters=model._owned_parameter_layout_report,optimizer_tensors=optimizer_layouts,
            all_parameter_values_and_strides_exact=True,all_optimizer_values_and_strides_exact=True,
            same_Parameter_objects_preserved=True,source_GPU_pointer_identity_claimed=False)
        if any(float(g["lr"]) != .01 for g in optimizer.param_groups):
            raise ValueError("original optimizer LR differs from actual primary .01")
        references = copy.deepcopy(model.reference)
        training, validation = load_data(args.train, recorder), load_data(args.valid, recorder)
        if len(training["starts"]) != 4096 or len(validation["starts"]) != 512:
            raise ValueError("original4096train/512validation split required")
        with np.load(args.batches, allow_pickle=False) as archive:
            schedule = np.asarray(archive["current"],dtype=np.int64)
        if schedule.shape != (800,BATCH) or np.any(schedule < 0) or np.any(schedule >= len(training["starts"])):
            raise ValueError("original800x64 current schedule required")
        index = schedule[STEP].copy()
        b = {k:torch.as_tensor(v[index],dtype=torch.long,device=device) for k,v in training.items()}
        # Restore AFTER all constructor/loader setup. There is no random dropout
        # or stochastic training sampling beyond this saved schedule.
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        expected_temperature = 1.-.5*max(0.,(STEP/800.-.7)/.3)
        if abs(model.temperature-expected_temperature)>1e-12:
            raise ValueError("saved source temperature differs from original index700 schedule")
        phase = "source_old_certificate"
        source_certificate = manual_old_certificate(model,references,recorder)
        if not source_certificate["passed"]:
            raise ValueError("actual source checkpoint already violates old contract")
        recorder.emit("configuration",MAIN="TOPOLOGY_USER_LOCKED",M0="NOT_PASSED",M3="ultimate",
            primary_baseline="Transformer",provenance=provenance,source_stage=STAGE,
            source_step=STEP,next_training_schedule_index=STEP,consumed_indices=index,
            source_temperature=model.temperature,actual_parameters=sum(p.numel() for p in model.parameters()),
            source_old_certificate=source_certificate,checkpoint_optimizer_state_restored=True,
            checkpoint_RNG_restored=True,validation_is_diagnostic_only=True,
            no_full_trajectory_causal_claim=True,no_second_optimizer_step=True)
        before = {name:p.detach().clone() for name,p in model.named_parameters()}
        model.train()
        phase = "single_original_gradient_and_Adam_proposal"
        synchronize(device)
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(b["tokens"],b["starts"],b["task_ids"])
        terminal = F.cross_entropy(logits,b["targets"])
        diversity = model.router_regularizer(b["tokens"],b["task_ids"])
        geometry = model.geometry_regularizer()
        objective = terminal+.05*diversity+.01*geometry
        if not torch.isfinite(objective):
            raise FloatingPointError("nonfinite actual one-step objective")
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite actual one-step gradient")
        optimizer.step()
        synchronize(device)
        costs.update(gradient_forwards=1,gradient_backwards=1,Adam_proposals=1,
                     training_sample_uses=BATCH,gradient_and_Adam_seconds=time.perf_counter()-began)
        proposal = {name:p.detach().clone() for name,p in model.named_parameters()}
        reconstructed = {name:before[name]+(proposal[name]-before[name]) for name in before}
        if not all(torch.isfinite(x).all() for x in list(proposal.values())+list(reconstructed.values())):
            raise FloatingPointError("nonfinite proposal/application")
        statistics = {name:finite_float32_statistics(proposal[name],reconstructed[name]) for name in proposal}
        recorder.emit("one_actual_Adam_proposal",terminal_CE=terminal,router_MI=diversity,
            normalized_kinetic_action=geometry,objective=objective,preclip_gradient_norm=norm,
            parameter_application_difference=statistics,
            elementwise_reconstruction_is_ONLY_intervention=True)
        arms = {}
        for label,values in (("direct_copy",proposal),("alpha1_reconstruction",reconstructed)):
            recorder.check()
            phase = label+"_application"
            synchronize(device)
            began=time.perf_counter()
            with torch.no_grad():
                for name,p in model.named_parameters():p.copy_(values[name])
            model._invalidate_derived_caches()
            synchronize(device)
            costs["application_seconds"][label]=time.perf_counter()-began
            phase = label+"_manual_old_certificate"
            certificate=manual_old_certificate(model,references,recorder)
            costs["old_certificate_checks"]+=len(certificate["checks"])
            costs["old_certificate_relation_pairs"]+=sum(x["relation_pairs"] for x in certificate["checks"])
            phase = label+"_bank_and_joint_assignment_gaps"
            bank,joint_scores=matching_scores(model)
            bank_reports=[assignment_diagnostic(scores,measured_solve,recorder) for scores in bank]
            joint_reports={token:assignment_diagnostic(scores,measured_solve,recorder)
                           for token,scores in joint_scores.items()}
            phase = label+"_actual_validation_execution"
            result=validation_evaluation(model,validation,device,recorder,costs)
            arms[label]=dict(certificate=certificate,bank_assignments=bank_reports,
                joint_assignments=joint_reports,validation=result,
                joint_log_score_matrices=joint_scores,
                continuous_geometry=model.geometry_resource_report())
            recorder.emit("application_result",application=label,**arms[label])
        direct,recon=arms["direct_copy"],arms["alpha1_reconstruction"]
        d,r=direct["validation"],recon["validation"]
        logdiff=np.abs(d["soft_log_probabilities"]-r["soft_log_probabilities"])
        comparison=dict(parameter_bit_differences=sum(x["bit_different"] for x in statistics.values()),
            parameters_with_any_bit_difference=[k for k,v in statistics.items() if v["bit_different"]],
            bank_experts_with_different_assignment=[i+1 for i,(a,b) in enumerate(zip(direct["bank_assignments"],recon["bank_assignments"]))
                if not np.array_equal(a["permutation"],b["permutation"])],
            tokens_with_different_joint_assignment=[a for a in ALPHABET if not np.array_equal(
                direct["joint_assignments"][a]["permutation"],recon["joint_assignments"][a]["permutation"])],
            max_absolute_soft_log_probability_difference=float(logdiff.max(initial=0.)),
            mean_absolute_soft_log_probability_difference=float(logdiff.mean()),
            soft_CE_reconstruction_minus_direct=r["soft_CE"]-d["soft_CE"],
            native_prediction_differences=int(np.count_nonzero(d["native_predictions"]!=r["native_predictions"])),
            joint_prediction_differences=int(np.count_nonzero(d["joint_predictions"]!=r["joint_predictions"])),
            max_absolute_joint_log_score_difference={a:float(np.max(np.abs(
                direct["joint_log_score_matrices"][a]-recon["joint_log_score_matrices"][a]))) for a in ALPHABET},
            both_old_certificates_pass=direct["certificate"]["passed"] and recon["certificate"]["passed"])
        recorder.emit("counterfactual_comparison",**comparison)
        recorder.json("summary.json",dict(status="ONE_PROPOSAL_NUMERIC_COUNTERFACTUAL_COMPLETED",
            MAIN="TOPOLOGY_USER_LOCKED",M0="NOT_PASSED",M3="ultimate",primary_baseline="Transformer",
            provenance=provenance,original_stage=STAGE,original_step=STEP,
            actual_gradient_steps=1,actual_Adam_proposals=1,next_batch_index=STEP,
            source_parameter_layouts=model._owned_parameter_layout_report,
            source_optimizer_layouts=optimizer_layouts,
            statistics=statistics,arms=arms,comparison=comparison,costs=costs,
            wall_seconds=time.perf_counter()-recorder.started,
            checkpoint_files_written=0,heldout_test_data_used=False,
            scope="local SAME-proposal numerical intervention only; not proof of full ON/OFF trajectory cause"))
        print(__import__("json").dumps(plain(dict(status="ONE_PROPOSAL_NUMERIC_COUNTERFACTUAL_COMPLETED",
            out=args.out,comparison=comparison,costs=costs)),sort_keys=True))
    except BaseException as error:
        recorder.emit("failure",error_type=type(error).__name__,error=str(error),costs=costs)
        recorder.json("failure.json",dict(status="FAILED_WITH_PARTIAL_RECORDS",error_type=type(error).__name__,
            error=str(error),costs=costs,wall_seconds=time.perf_counter()-recorder.started))
        raise
    finally:
        topology_model.linear_sum_assignment=original_solver
        recorder.stream.close()


if __name__=="__main__":
    main()
