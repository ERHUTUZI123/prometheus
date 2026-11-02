"""Same-information ordinary controls for the actual30P hybrid loop model.

NEW bounded CPU diagnostic. No experiment, SSH or network call at import.
Root supplies the existing external isolation/120s watchdog; internal<=90s.
MAIN=TOPOLOGY human-locked. This is prerequisite TODO2/3 attribution, not a
Primary freeze, general Block result, novelty claim or benchmark replacement.

Arms:
 ordinary_event: generic event+neural derivative, SAME common certificates.
 ordinary_chamber_heads: paid finite chamber selection, then E/R/call SGD.
 ordinary_program_search: paid chamber x3-call search installed in actual NN,
                          then identical neural SGD only if still necessary.

All use current HybridLoopBlock E/R/H/loop/call parameters, initial uniform new
call, supplied old mechanism and same four terminal TRAIN cases. No desired
endpoint map, new route or intermediate teacher state is given. Program-search
routes are hypotheses queried against terminal data, not supplied answers.
Every temporary candidate transaction is charged in the100-transaction cap.
The common analytic certificate is available to ordinary controls; making them
reexecute every old loop unnecessarily would weaken the strongest comparator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

import torch

from loop_boundary_learning import (BudgetStop,CostLedger,Limits,NumericalFailure,
    ProgramExample,_diagnostic_data,slice_events)
from hybrid_loop_block import (HybridLearningSession,HybridTerminalObjective,
    SOFT_NATIVE_MIN_MASS,coupled_gradient,native_output)
from run_hybrid_loop_diagnostic import (own_native_confidence_maintenance,
    save_actual_checkpoint,snapshot_parameters,parameter_displacements)


ARMS=("ordinary_event","ordinary_chamber_heads","ordinary_program_search")


def hfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _remaining_transaction(session):
    if session.ledger.counts.get("updates",0)>=session.ledger.limits.max_updates:
        raise BudgetStop("all actual transactions, including candidate trials, exhausted")


def _copy_parameters(session):
    return {name:p.detach().clone() for name,p in session.model.named_parameters()}


@torch.no_grad()
def _restore_parameters(session,values,license_hash):
    for name,p in session.model.named_parameters():p.copy_(values[name])
    session.bank.semantic_license_hash=license_hash
    session.ledger.add("ordinary_candidate_rollbacks")
    session.ledger.add("ordinary_rollback_scalar_copies",sum(v.numel() for v in values.values()))


@torch.no_grad()
def _fit_candidate(session,action,new_route):
    """Fit paid path coordinates; optionally fit an ENUMERATED call hypothesis.

    Old geometry and old hard call choices are OWN current values. A candidate
    new route ranges over ALL choices0..2, never a teacher-selected target.
    SharedH/E/R stay untouched during installation. All actual weights remain
    in the source model, with exactly the same parameter shapes/count.
    """
    model,ledger=session.model,session.ledger
    operations=[*sorted(session.protected.admitted),1]
    hidden=model.shared(model.operation_features[operations])
    desired=model.loop_head(hidden).detach().clone()
    desired[-1]=desired.new_tensor(action)
    ledger.add("ordinary_geometry_least_squares_solves")
    loop_solution=torch.linalg.lstsq(hidden,desired).solution.T
    loop_residual=float((hidden@loop_solution.T-desired).abs().max())
    targets={"loop_head.weight":loop_solution}
    report=dict(selected_action=list(action),geometry_fit_residual=loop_residual,
        desired_old_actions=desired[:-1].reshape(-1).cpu().tolist(),
        shared_H_unchanged=True,encoder_readout_unchanged=True,
        new_route_hypothesis=new_route,new_route_not_provided_by_teacher=True)
    if not bool(torch.isfinite(loop_solution).all()) or loop_residual>1e-8:
        raise NumericalFailure("same NN cannot fit the paid action while preserving own old action")
    if new_route is not None:
        token_hidden=model.shared(model.token_features)
        route_reports=[]
        for slot,head in enumerate(model.calls):
            own_routes=head(token_hidden).argmax(-1)
            hypothesis=own_routes.clone();hypothesis[1]=int(new_route)
            desired_logits=torch.full((len(token_hidden),3),-6.,dtype=token_hidden.dtype)
            desired_logits.scatter_(1,hypothesis[:,None],6.)
            ledger.add("ordinary_call_least_squares_solves")
            solution=torch.linalg.lstsq(token_hidden,desired_logits).solution.T
            fitted=token_hidden@solution.T
            residual=float((fitted-desired_logits).abs().max())
            if (not bool(torch.isfinite(solution).all()) or residual>1e-8
                    or not torch.equal(fitted.argmax(-1),hypothesis)):
                raise NumericalFailure("actual same call head cannot fit enumerated program hypothesis")
            targets[f"calls.{slot}.weight"]=solution
            route_reports.append(dict(slot=slot,old_own_routes=own_routes.cpu().tolist(),
                hypothesis=hypothesis.cpu().tolist(),maximum_fit_residual=residual))
        report["call_fits"]=route_reports
    if any(float(v.abs().max())>16. for v in targets.values()):
        raise NumericalFailure("candidate fit exceeds the common finite neural box")
    gradients=[torch.zeros_like(p) if name not in targets else
               (p.detach()-targets[name])/.05 for name,p in model.named_parameters()]
    ledger.add("ordinary_candidate_fitted_scalar_targets",sum(v.numel() for v in targets.values()))
    return gradients,report


def finite_program_selection(session,training,*,search_calls):
    """Cold finite discovery using source events and actual terminal feedback.

    Trial parameters are genuinely installed through the unchanged retention
    transaction, queried, then atomically restored. Costs/observations remain.
    No test example enters selection. Candidate maps are never read from a
    desired-operation table. Temporary admitted transactions count toward100.
    """
    ledger,model=session.ledger,session.model
    began=time.perf_counter()
    if session.family.dimension!=1 or model.slots!=1:
        raise ValueError("this finite diagnostic supports only the actual1D,one-slot source")
    mean=model.action(1)
    lo=session.family.action_box[0]+session.kernel.sigma
    hi=session.family.action_box[1]-session.kernel.sigma
    events=slice_events(session.family,mean,0,lo,hi,session.executor.critical_controls,ledger)
    boundaries=[lo,*events,hi]
    actions=[[(left+right)/2] for left,right in zip(boundaries,boundaries[1:])]
    if not 1<=len(actions)<=8:raise NumericalFailure("finite chamber enumeration bound exceeded")
    baseline=_copy_parameters(session);license_hash=session.bank.semantic_license_hash
    baseline_hash=model.parameter_hash();P=model.parameter_count()
    ledger.add("ordinary_search_snapshot_scalar_values",sum(p.numel() for p in baseline.values()))
    objective=HybridTerminalObjective(model,training,ledger)
    candidates=[]
    routes=range(3) if search_calls else (None,)
    best=None
    for action in actions:
        for route in routes:
            _remaining_transaction(session)
            fit=None;commit=None
            try:
                gradients,fit=_fit_candidate(session,action,route)
                ledger.add("ordinary_discrete_candidate_trials")
                commit=session.transaction(gradients,operation=1,learning_rate=.05)
                # A backtracked alternative is a real candidate but is labelled
                # by its ACTUAL resulting action/route, not the desired fit.
                with torch.no_grad():
                    maps=session.bank.maps()
                    loss=float(objective.tensor(maps))
                    signature=model.native_signature(maps,ledger)
                    correct=sum(native_output(signature,e,ledger)==e.target for e in training)
                if not math.isfinite(loss):raise NumericalFailure("nonfinite actual candidate terminal loss")
                candidate=dict(action_proposal=list(action),route_proposal=route,
                    fitted_geometry_and_calls=fit,transaction=commit,terminal_CE=loss,
                    TRAIN_native_accuracy=correct/len(training),actual_signature=signature,
                    actual_parameter_hash=model.parameter_hash())
                candidates.append(candidate)
                if best is None or (loss,-correct)<(best["candidate"]["terminal_CE"],-best["correct"]):
                    best=dict(candidate=candidate,parameters=_copy_parameters(session),correct=correct)
                ledger.emit("ordinary_paid_program_candidate",**candidate)
            finally:
                _restore_parameters(session,baseline,license_hash)
                if model.parameter_hash()!=baseline_hash or model.parameter_count()!=P:
                    raise NumericalFailure("temporary candidate failed exact same-NN rollback")
    if best is None:raise NumericalFailure("no executable candidate was actually evaluated")
    _remaining_transaction(session)
    selected=best["parameters"]
    gradients=[(p.detach()-selected[name])/.05 for name,p in model.named_parameters()]
    install=session.transaction(gradients,operation=1,learning_rate=.05)
    ledger.add("ordinary_selected_program_installs")
    ledger.elapsed("ordinary_finite_program_selection",began)
    result=dict(events=events,enumerated_chamber_representatives=actions,
        searched_call_hypotheses=search_calls,candidates=candidates,
        selected=best["candidate"],final_install=install,
        parameter_count_before=P,parameter_count_after=model.parameter_count(),
        candidate_snapshot_transient_bytes=sum(v.numel()*v.element_size() for v in baseline.values()),
        selected_parameter_transient_bytes=sum(v.numel()*v.element_size() for v in selected.values()),
        feedback="four supplied TRAIN terminal cases only; common actual full-fiber observations",
        selection="minimum actual current-neural TRAIN NLL, tie greater native correctness, then stable enumeration order",
        installation="same real NN and same old geometry/native/soft contracts; selected route is paid search output")
    ledger.emit("ordinary_finite_program_selected",**result)
    return result


def head_learning_gradient(session,training):
    """Actual terminal gradient of E/R/calls with selected geometry module fixed."""
    maps=session.bank.maps()
    objective=HybridTerminalObjective(session.model,training,session.ledger)
    loss=objective.tensor(maps)
    parameters=list(session.model.parameters())
    gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
    result=[];norms={}
    for (name,p),g in zip(session.model.named_parameters(),gradients):
        keep=name in ("encoder_logits","readout_logits") or name.startswith("calls.")
        raw=torch.zeros_like(p) if g is None else g.detach()
        used=raw if keep else torch.zeros_like(raw)
        result.append(used)
        norms[name]=dict(raw_norm=float(torch.linalg.vector_norm(raw)),
                         applied_norm=float(torch.linalg.vector_norm(used)),updated=keep)
    session.ledger.add("ordinary_terminal_neural_vjps")
    session.ledger.add("ordinary_geometry_freeze_mask_scalars",sum(p.numel() for p in parameters))
    if any(not bool(torch.isfinite(g).all()) for g in result):raise NumericalFailure("nonfinite actual ordinary neural gradient")
    return result,dict(terminal_CE=float(loss.detach()),parameter_groups=norms,
        update_scope="E/R/call weights; selected sharedH and loop head retained as ordinary frozen module",
        no_extra_trainable_parameters=True)


def run_arm(arm,directory,limits,deadline,seed):
    directory.mkdir(exist_ok=False)
    stream=(directory/"records.jsonl").open("x",encoding="utf8")
    def write(record):
        stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n");stream.flush()
    ledger=CostLedger(limits,deadline,writer=write)
    torch.manual_seed(seed)
    session=HybridLearningSession(ledger,slots=1,sigma=2.,new_route_seeded=False,
                                  protection="same_certificate",transaction_policy="block_fallback")
    rng=random.Random(seed)
    training,unused=_diagnostic_data();del unused
    initial=snapshot_parameters(session);started=time.perf_counter()
    result=dict(arm=arm,status="started",parameters_total=session.model.parameter_count(),
        buffer_scalars=sum(b.numel() for b in session.model.buffers()),seed=seed,
        old_ability_initialization="same supplied loop and old call; paid continuation",
        new_call_initialization="same uniform new call",encoder_readout="same trainable near-identity initial frame",
        common_certificates="same available analytic loop guard, exact native binding closure, L16 own-path mass",
        update_count_scope="ALL temporary search transactions, selected installs, gradient commits, zero commits and maintenance",
        training=[asdict(e) for e in training],checkpoints=[],curves=[],maintenance=dict(attempted=False),
        heldout_selection=False,discrete_selection=None)
    old_examples=();admitted=False;maintenance_attempted=False
    try:
        contract=session.initialize_old_contract();reference=contract["reference"]
        old_examples=tuple(ProgramExample(f"old-derived-{name}-{x}",x,word,
            native_output(reference,ProgramExample("placeholder",x,word,0),ledger))
            for x in range(3) for name,word in (("empty",()),("old",(0,))))
        result["initial_old_evaluation"]=session.evaluate(old_examples)
        result["initial_training_evaluation"]=session.evaluate(training)
        result["initial_parameters"]=initial
        result["checkpoints"].append(save_actual_checkpoint(session,directory/"initial.pt","initial"))
        latest_train=result["initial_training_evaluation"]
        latest_old=result["initial_old_evaluation"]
        if arm!="ordinary_event":
            result["discrete_selection"]=finite_program_selection(session,training,search_calls=arm=="ordinary_program_search")
            latest_train=session.evaluate(training);latest_old=session.evaluate(old_examples)
        while ledger.counts.get("updates",0)<limits.max_updates:
            if time.monotonic()>=deadline-2.:raise BudgetStop("reserve actual final checkpoint/reload time")
            train,old=latest_train,latest_old
            if old["native_accuracy"]!=1. or old["soft_accuracy"]!=1.:
                raise NumericalFailure("ordinary proposal violated actual old capability")
            if train["native_accuracy"]==1. and train["soft_accuracy"]==1.:
                bound=session.model.soft_native_path_bound((0,1),16,ledger)
                if bound<=SOFT_NATIVE_MIN_MASS and not maintenance_attempted:
                    maintenance_attempted=True;_remaining_transaction(session)
                    result["maintenance"]=own_native_confidence_maintenance(session)
                admitted=session.admit_if_successful(training,operation=1,new_tokens=(1,))
                if admitted:break
            _remaining_transaction(session);before=snapshot_parameters(session)
            if arm=="ordinary_event":
                objective=HybridTerminalObjective(session.model,training,ledger)
                computed=coupled_gradient(session.model,session.bank,objective,1,session.kernel,rng,
                                           path_estimator="ordinary_event",within_samples=1)
                gradients,record=computed["parameter_gradients"],computed["record"]
            else:gradients,record=head_learning_gradient(session,training)
            commit=session.transaction(gradients,operation=1,learning_rate=.05)
            after=snapshot_parameters(session)
            latest_train=session.evaluate(training);latest_old=session.evaluate(old_examples)
            row=dict(actual_transactions=ledger.counts.get("updates",0),gradient=record,transaction=commit,
                parameters=after,parameter_displacements=parameter_displacements(before,after),
                train=latest_train,old=latest_old,
                costs=ledger.snapshot(),derived_storage=session.bank.storage())
            result["curves"].append(row);ledger.emit("ordinary_hybrid_learning_curve",**row)
        # Admission is checked once more when the final permitted gradient used
        # the envelope's last transaction; no extra gradient/maintenance is added.
        if not admitted:admitted=session.admit_if_successful(training,operation=1,new_tokens=(1,))
        result["status"]="finite_training_complete"
        result["admitted_new_operation_and_binding"]=admitted
        result["final_training_evaluation"]=latest_train
        result["final_old_evaluation"]=latest_old
    except BudgetStop as error:result.update(status="censored",reason=str(error))
    except NumericalFailure as error:result.update(status="numerical_unresolved",reason=str(error))
    except Exception as error:
        result.update(status="implementation_failure",reason=f"{type(error).__name__}: {error}")
        (directory/"exception.txt").write_text(traceback.format_exc())
    try:
        frozen=session.model.parameter_hash();result["frozen_before_heldout_hash"]=frozen
        for p in session.model.parameters():p.requires_grad_(False)
        result["checkpoints"].append(save_actual_checkpoint(session,directory/"final.pt","frozen-final-or-partial"))
        result["final_parameters"]=snapshot_parameters(session)
        result["parameter_displacements_from_initial"]=parameter_displacements(initial,result["final_parameters"])
        result["admitted_operations"]=sorted(session.protected.admitted)
        result["protected_token_alphabet"]=list(session.old_tokens)
        result["actual_parameter_transactions"]=ledger.counts.get("updates",0)
        if time.monotonic()<deadline:
            _,heldout=_diagnostic_data()
            result["heldout_after_frozen_weights"]=session.evaluate(heldout)
            if old_examples:result["final_old_evaluation"]=session.evaluate(old_examples)
            payload=torch.load(directory/"final.pt",map_location="cpu",weights_only=False)
            restored=HybridLearningSession.restore(payload,ledger)
            for p in restored.model.parameters():p.requires_grad_(False)
            result["reload_heldout"]=restored.evaluate(heldout)
            if old_examples:result["reload_old"]=restored.evaluate(old_examples)
            cold={op:restored.executor.execute([restored.family.vertices(restored.model.action(op))])["endpoint_map"]
                  for op in range(restored.model.bank_size)}
            ledger.add("reload_cold_primitive_reexecutions",len(cold))
            result["reload_check"]=dict(actual_nn_hash_equal=restored.model.parameter_hash()==frozen,
                cold_reexecuted_maps=cold,persisted_maps=restored.bank.maps(),
                cold_maps_equal=cold==restored.bank.maps(),final_original_hash_unchanged=session.model.parameter_hash()==frozen)
        else:result["heldout_reload_status"]="not executed: finite deadline exhausted"
    except (BudgetStop,NumericalFailure) as error:result["heldout_reload_status"]=f"incomplete: {type(error).__name__}: {error}"
    except Exception as error:
        result["delivery_status"]=f"failure: {type(error).__name__}: {error}"
        (directory/"delivery-exception.txt").write_text(traceback.format_exc())
    finally:
        result.update(wall_seconds=time.perf_counter()-started,costs=ledger.snapshot(),
            forward_counts=dict(session.model.forward_counts),final_storage=session.bank.storage(),
            checkpoint_total_bytes=sum(x["bytes"] for x in result["checkpoints"]),
            baseline_validity="protocol/implementation/reproduction, not achieved accuracy")
        write(dict(kind="ordinary_hybrid_arm_end",**result));stream.close()
        (directory/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=90.)
    p.add_argument("--max-updates",type=int,default=100)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--reference-result",type=Path)
    args=p.parse_args()
    if args.out.exists() or not 6<args.seconds<=90 or not 1<=args.max_updates<=100 or len(set(args.arms))!=len(args.arms):
        p.error("new output, unique arms, internal<=90s and<=100 total transactions/arm required")
    torch.set_num_threads(1);args.out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parent
    protocol=dict(version="hybrid-conventional-controls-v1",MAIN="TOPOLOGY_USER_LOCKED",seed=0,
        seconds_internal=args.seconds,outside_timeout_required=120,max_transactions_per_arm=args.max_updates,
        arms=args.arms,learning_rate=.05,shared_information="same cubic source, events, terminal TRAIN, actual full-fiber observations, certificates and caches",
        shared_NN="same30P HybridLoopBlock; all frozen/non-updated parameters counted",
        search="finite safe chamber candidates from supplied source events; optional enumerated call hypotheses scored on TRAIN only",
        teacher_endpoint_or_route_given=False,test_selection=False,
        costs="cold discovery, all temporary installs/rollbacks, final install, queries, neural updates, maintenance and reload",
        wall_allocation="equal shares of remaining time; tiny uncontrolled host times are not an architecture efficiency claim",
        source_sha256={name:hfile(source/name) for name in
            (Path(__file__).name,"loop_boundary_learning.py","hybrid_loop_block.py","run_hybrid_loop_diagnostic.py")})
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    deadline=time.monotonic()+args.seconds;results=[]
    limits=Limits(max_feedback=12000,max_programs=4000,max_continuation_steps=160000,max_updates=args.max_updates)
    for i,arm in enumerate(args.arms):
        remaining=len(args.arms)-i;now=time.monotonic();available=deadline-now-3.
        if available<=3.*remaining:results.append(dict(arm=arm,status="not_started_internal_wall_limit"));continue
        arm_deadline=min(deadline-3.,now+available/remaining)
        results.append(run_arm(arm,args.out/arm,limits,arm_deadline,seed=0))
    reference=None
    if args.reference_result:
        # Read only AFTER all actual control learning/selection has frozen.
        if args.reference_result.stat().st_size>100*1024*1024:
            raise ValueError("reference receipt exceeds100MiB; actual control files remain preserved")
        ref=json.loads(args.reference_result.read_text())
        reference=dict(path=str(args.reference_result),sha256=hfile(args.reference_result),
            protocol=ref.get("protocol"),results=[{k:r.get(k) for k in
                ("arm","status","parameters_total","actual_parameter_updates","admitted_new_operation_and_binding",
                 "heldout_after_frozen_weights","final_old_evaluation","costs","wall_seconds")}
                for r in ref.get("results",[])],used_for_learning_or_selection=False)
    summary=dict(protocol=protocol,results=results,previous_frozen_reference=reference,
        evidence_boundary="actual new control records only; original scalar and hybrid failures preserved; Primary not frozen",
        no_uncontrolled_wall_speedup_or_general_architecture_claim=True)
    (args.out/"result.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(out=str(args.out),arms=[(r["arm"],r["status"]) for r in results])))


if __name__=="__main__":main()
