"""Bounded terminal-only training harness; importing starts no experiment.

Example smoke (the coordinator runs it):
  python train.py --out RUN_DIR --scale 4 --seed 0 --pilot --models topo,topo_off
Formal invocation: --scale 8 --steps 800 --models topo,transformer,replay,adapter,topo_off,program
No automatic tuning, retry, growth, grid expansion, remote access or M0 verdict.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from tasks import make_data, collision_report, data_sha256, ARRAY_KEYS
from torch_baselines import TinyTransformer, ProgramLearner, parameter_counts


CAPS = (3, 5, 7)
NEW_TOKENS = ((1, 2, 3), (4, 5), (6,))
ALPHABETS = ((1, 2, 3), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6))
ARMS = ("topo", "transformer", "replay", "adapter", "topo_off", "program", "constraint")
MAX_FILE_BYTES = 100 * 1024 * 1024


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Recorder:
    def __init__(self, out, seconds):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.file = (out / "records.jsonl").open("x", encoding="utf-8")
        self.started = time.perf_counter()
        self.deadline = self.started + seconds
        self.bytes = 0

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("finite harness wall budget exhausted; no automatic retry")

    def emit(self, kind, **fields):
        row = json.dumps(plain(dict(kind=kind, wall=time.perf_counter()-self.started,
                                   **fields)), sort_keys=True, allow_nan=False) + "\n"
        size = len(row.encode("utf-8"))
        if self.bytes + size > MAX_FILE_BYTES:
            raise RuntimeError("100 MiB JSONL file cap")
        self.file.write(row)
        self.file.flush()
        self.bytes += size

    def summary(self, payload):
        text = json.dumps(plain(payload), sort_keys=True, indent=2, allow_nan=False)
        if len(text.encode()) > MAX_FILE_BYTES:
            raise RuntimeError("100 MiB summary cap")
        temporary = self.out / "summary.json.tmp"
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(self.out / "summary.json")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def data_bytes(data):
    return sum(data[k].nbytes for k in ARRAY_KEYS)


def tensors(data):
    return {k: torch.from_numpy(data[k]).long() for k in ARRAY_KEYS}


def fetch(data, indices, device):
    ix = torch.as_tensor(indices, dtype=torch.long)
    return {k: data[k].index_select(0, ix).to(device) for k in ARRAY_KEYS}


def build_data(args, recorder):
    folder = recorder.out / "datasets"
    folder.mkdir()
    all_data, all_tensors = [], []
    for task in range(3):
        seed = args.seed * 10_000_000 + args.scale * 10_000 + task * 1000
        specs = {"train": (11, 4096, 1, CAPS[task]),
                 "valid": (23, 512, 1, CAPS[task]),
                 "iid": (37, 512, 1, CAPS[task]),
                 "ood": (53, 512, 8, 12),
                 "length16": (71, 512, 16, 16)}
        split_data = {}
        for split, (offset, count, lo, hi) in specs.items():
            d = make_data(args.scale, seed+offset, task, count, 18,
                          length_lo=lo, length_hi=hi)
            split_data[split] = d
            path = folder / f"task{task}-{split}.npz"
            np.savez_compressed(path, **{k: d[k] for k in ARRAY_KEYS},
                                metadata=np.asarray(json.dumps(plain(d["metadata"]), sort_keys=True)))
            if path.stat().st_size > MAX_FILE_BYTES:
                raise RuntimeError("dataset file exceeds cap")
        for split, d in split_data.items():
            recorder.emit("dataset", task=task, split=split, metadata=d["metadata"],
                          sha256=data_sha256(d), array_bytes=data_bytes(d),
                          collisions_with_task_train=collision_report(split_data["train"], d),
                          file=str((folder / f"task{task}-{split}.npz").relative_to(recorder.out)))
        all_data.append(split_data)
        all_tensors.append({s: tensors(d) for s, d in split_data.items()})
    return all_data, all_tensors


def make_model(name, args, restored_latent_factor=None):
    # Identical Transformer backbone/head initialization across the three arms.
    torch.manual_seed(args.seed + 2020)
    if name in ("transformer", "replay", "adapter"):
        base = TinyTransformer(args.scale, vocab=7, max_len=18, n_tasks=3,
                               d_model=48, n_heads=4, n_layers=2, adapter=False)
        if name != "adapter":
            return base
        torch.manual_seed(args.seed + 2020)
        model = TinyTransformer(args.scale, vocab=7, max_len=18, n_tasks=3,
                                d_model=48, n_heads=4, n_layers=2, adapter=True)
        state, common = model.state_dict(), base.state_dict()
        for key in state:
            if key in common and state[key].shape == common[key].shape:
                state[key] = common[key].clone()
        model.load_state_dict(state)
        for head in model.heads:
            head.load_state_dict(base.head.state_dict())
        return model
    if name == "program":
        if args.program_v2:
            from program_baseline_v2 import ProgramLearner as DirectProgramLearner
            return DirectProgramLearner(args.scale, vocab=7, n_tasks=3,
                                        latent_factor=2, modular=not args.program_plastic,
                                        hard_assignment=args.program_hard_assignment)
        return ProgramLearner(args.scale, vocab=7, n_tasks=3,
                              latent_factor=2, modular=True)
    if name in ("topo", "topo_off", "constraint"):
        from topology_model import TopologyExecutor
        if args.native_promotion:
            from semantic_promotion_model import SemanticPromotionExecutor
            TopologyExecutor = SemanticPromotionExecutor
        if getattr(args, 'continuous_monodromy', False):
            from continuous_native_model import ContinuousNativeModel
            TopologyExecutor = ContinuousNativeModel
        geometry_options = ({'geometry_cache_capacity':getattr(args, 'geometry_moment_cache', 0),
                             'homotopy_lift_cache_capacity':getattr(args, 'homotopy_lift_cache', 0)}
                            if getattr(args, 'continuous_monodromy', False) else {})
        model=TopologyExecutor(args.scale, vocab=6, max_len=18, n_tasks=3,
                                latent_factor=restored_latent_factor or args.latent_factor,
                                protection=name != "topo_off", seed=args.seed,
                                constraint=name == "constraint", **geometry_options)
        model.compilation=args.compilation
        model.credit=args.credit
        if args.protection_timing=='off':model.protection=False
        return model
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def evaluate(model, datasets, device, batch_size, recorder, proposal=False):
    was_training = model.training
    model.eval()
    result = {}
    hard_method = getattr(model, "proposal_eval" if proposal else "hard_eval", None)
    for task, splits in enumerate(datasets):
        result[task] = {}
        for split in ("valid", "iid", "ood", "length16"):
            recorder.check()
            d = splits[split]
            count, soft_right, hard_right, loss = 0, 0, 0, 0.
            for begin in range(0, len(d["starts"]), batch_size):
                b = fetch(d, np.arange(begin, min(begin+batch_size, len(d["starts"]))), device)
                logits = model(b["tokens"], b["starts"], b["task_ids"])
                hard = hard_method(b["tokens"], b["starts"], b["task_ids"]) if callable(hard_method) else logits
                if not torch.isfinite(logits).all() or not torch.isfinite(hard).all():
                    raise FloatingPointError("nonfinite evaluation logits")
                count += len(b["starts"])
                loss += F.cross_entropy(logits, b["targets"], reduction="sum").item()
                soft_right += int((logits.argmax(-1) == b["targets"]).sum().item())
                hard_right += int((hard.argmax(-1) == b["targets"]).sum().item())
            result[task][split] = dict(n=count, loss=loss/count,
                                       soft_acc=soft_right/count, hard_acc=hard_right/count,
                                       hard_mode="discrete_executor" if callable(hard_method) else "soft_logits_argmax")
    model.train(was_training)
    return result


def parameter_delta(model, before):
    sq, maximum, changed = 0., 0., 0
    groups, shape_changes = {}, {}
    for name, p in model.named_parameters():
        if name not in before or tuple(p.shape) != tuple(before[name].shape):
            shape_changes[name] = dict(before=None if name not in before else list(before[name].shape), after=list(p.shape))
            continue
        diff = p.detach().cpu().double() - before[name].double()
        value = float(diff.square().sum().item())
        groups[name] = math.sqrt(value)
        sq += value
        maximum = max(maximum, float(diff.abs().max().item()))
        changed += int(torch.count_nonzero(diff).item())
    return dict(L2=math.sqrt(sq), Linf=maximum, changed_scalars=changed,
                per_parameter_L2=groups,
                shape_changes=shape_changes, L2_scope='same-shape tensors only; grown parameters not silently aligned',
                limitation="parameter movement alone is not shared semantic acquisition")


@torch.no_grad()
def learned_snapshot(model, stage):
    graphs = []
    if not callable(getattr(model, "graph", None)):
        return dict(available=False)
    for task in range(stage+1):
        g = model.graph(task, ALPHABETS[task])
        graphs.append(dict(task=task, reachable=len(g["reachable"]),
                           routes=g["routes"], braid_lengths=g["braid_lengths"],
                           encoder=g["enc"], decoder=g["dec"],
                           learned_transitions=g["trans"]))
    return dict(available=True, graphs=graphs, teacher_states_used=False)


def save_checkpoint(model, optimizer, args, name, stage, step, recorder, suffix=""):
    started = time.perf_counter()
    folder = recorder.out / "checkpoints"
    folder.mkdir(exist_ok=True)
    filename = f"{name}-stage{stage}-step{step}{suffix}.pt"
    temporary, path = folder / (filename+".tmp"), folder / filename
    extra = model.extra_state_dict() if callable(getattr(model, "extra_state_dict", None)) else {}
    masks = {k: getattr(model, k) for k in ("_row_masks", "_row_snapshots", "_initialized_tasks")
             if hasattr(model, k)}
    payload = dict(schema="terminal-architecture-checkpoint-v1", args=vars(args), model_name=name,
                   stage=stage, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   model_extra=extra, masks_and_initialized_tasks=masks,
                   torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   no_automatic_resume_or_retry=True, M0="NOT_PASSED")
    torch.save(payload, temporary)
    size = temporary.stat().st_size
    if size > MAX_FILE_BYTES:
        temporary.unlink()
        raise RuntimeError("checkpoint exceeds100 MiB cap")
    temporary.replace(path)
    recorder.emit("checkpoint", model=name, stage=stage, step=step,
                  file=str(path.relative_to(recorder.out)), file_bytes=size,
                  seconds=time.perf_counter()-started,
                  protection_reference_saved=bool(extra),
                  caution="load only trusted locally generated checkpoints")


def reset_guard_counts():
    return dict(calls=0, accepted_full=0, accepted_backtracked=0, rejected=0,
                checks=0, relation_pairs=0, measured_seconds=0.,
                reported_seconds=0., snapshot_seconds=0.)


def run_model(name, args, datasets, raw_data, batches, old_indices, device, recorder):
    recorder.check()
    started = time.perf_counter()
    checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False) if args.resume_from else None
    restored_factor = (checkpoint['model']['encoder'].shape[1] // args.scale
                       if checkpoint and name in ('topo','topo_off','constraint') else None)
    model = make_model(name, args, restored_factor).to(device)
    if callable(getattr(model, 'set_geometry_deadline', None)):
        model.set_geometry_deadline(time.monotonic() + max(.001, recorder.deadline-time.perf_counter()))
    if args.resume_from:
        # Only coordinator-selected checkpoints generated by this campaign.
        repaired = checkpoint.get('observed_operation_formation', {})
        source_step_valid = checkpoint['step'] == args.steps
        if getattr(args, 'resume_repaired_stage_checkpoint', False):
            extra_updates = repaired.get('additional_updates')
            source_step_valid = (isinstance(extra_updates, int) and extra_updates > 0
                                 and repaired.get('extra_budget_not_retroactive') is True
                                 and repaired.get('original_training_args_preserved') is True
                                 and checkpoint['args'].get('steps') == args.steps
                                 and checkpoint['step'] == args.steps + extra_updates)
        if checkpoint['model_name'] != name or checkpoint['stage'] != args.start_stage-1 or not source_step_valid:
            raise ValueError('resume must select the completed preceding stage of the same arm/budget')
        for field in ('scale', 'seed', 'latent_factor', 'native_promotion', 'continuous_monodromy'):
            if checkpoint['args'].get(field, False) != getattr(args, field):
                raise ValueError('resume configuration mismatch: '+field)
        model.load_state_dict(checkpoint['model'])
        extra = checkpoint.get('model_extra', {})
        if callable(getattr(model, 'load_extra_state_dict', None)):
            model.load_extra_state_dict(extra)
        elif extra:
            for field in ('reference', 'temperature', 'stage', 'compilation', 'credit', 'protection'):
                if field in extra:
                    setattr(model, field, extra[field])
        for field, value in checkpoint.get('masks_and_initialized_tasks', {}).items():
            setattr(model, field, value)
        torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        recorder.emit('restored_completed_previous_stage', model=name, start_stage=args.start_stage,
                      checkpoint=str(args.resume_from), sha256=hashlib.sha256(args.resume_from.read_bytes()).hexdigest(),
                      actual_source_step=checkpoint['step'],
                      repaired_source_additional_updates=repaired.get('additional_updates', 0),
                      previous_stage_not_retrained=True, fresh_optimizer_at_new_stage=True)
    recorder.emit("model_begin", model=name, parameters=parameter_counts(model),
                  exact_constraint_correspondence=(name == "constraint"),
                  constraint_note="same ordinary graph-equivalence algorithm as topology arm" if name == "constraint" else None,
                  topology_latent_factor=args.latent_factor,
                  actual_latent_states=getattr(model, "d", getattr(model, "dim", None)),
                  terminal_supervision_only=True)
    stages, retention = [], []
    optimizer = None
    promoted_stages = set()
    for stage in range(args.start_stage, args.stages):
        recorder.check()
        if args.workspace_refinement and callable(getattr(model, 'materialize', None)):
            from workspace_refinement import find_state_witness, grow_workspace
            witness = find_state_witness(raw_data[stage]['train'], NEW_TOKENS[stage], model.d)
            recorder.emit('observed_history_state_witness', model=name, stage=stage, report=witness,
                          feedback='actual current-task terminal training records only')
            if witness['certificate_valid'] and witness['dimension_exceeded']:
                requested = 1 << (int(witness['lower_bound'])-1).bit_length()
                growth = grow_workspace(model, requested)
                recorder.emit('protected_workspace_refinement', model=name, stage=stage, report=growth,
                              actual_observed_lower_bound=witness['lower_bound'])
                if not growth['success']:
                    raise RuntimeError('witness-triggered workspace transaction failed: '+growth.get('reason','unknown'))
                model.workspace_history = getattr(model,'workspace_history',[]) + [dict(witness=witness, growth=growth)]
        model.set_stage(stage, new_token_ids=NEW_TOKENS[stage])
        if stage==0 and args.protection_timing=='early' and callable(getattr(model,'consolidate',None)):
            early=model.consolidate(stage,ALPHABETS[stage])
            recorder.emit('early_unvalidated_protection',model=name,stage=stage,reference=early,
                          intentionally_tests_protection_timing_not_an_acquired_capability=True)
        model.train()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        before_stage = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        guards = reset_guard_counts()
        stage_started = time.perf_counter()
        train_seconds = eval_seconds = snapshot_seconds = 0.
        train_current = train_old = 0
        losses, gradients, auxiliary_losses, geometry_losses = [], [], [], []
        temperature_checks = temperature_rejected = 0
        temperature_seconds = 0.
        replay_memory = None
        if name == "replay" and stage:
            replay_memory = {k: torch.cat([datasets[t]["train"][k] for t in range(stage)], 0)
                             for k in ARRAY_KEYS}
        recorder.emit("stage_begin", model=name, stage=stage, new_token_ids=NEW_TOKENS[stage],
                      parameters=parameter_counts(model), fresh_adam=True,
                      replay_memory_rows=0 if replay_memory is None else len(replay_memory["starts"]),
                      replay_memory_tensor_bytes=tensor_bytes(replay_memory),
                      all_models_same_task_descriptors=True,
                      current_teacher_dataset_bytes=data_bytes(raw_data[stage]["train"]),
                      replay_policy="32current+32uniformold" if replay_memory is not None else "64current")
        try:
            for step in range(args.steps+1):
                recorder.check()
                temperature = 1. - .5 * max(0., (step/max(1,args.steps)-.7)/.3)
                setter = getattr(model, "set_temperature", None)
                if callable(setter):
                    temperature_report = setter(temperature)
                    if isinstance(temperature_report, dict):
                        temperature_checks += temperature_report.get('checks', 0)
                        temperature_rejected += int(temperature_report.get('rejected', False))
                        temperature_seconds += temperature_report.get('seconds', 0.)
                if step == 0 or step % args.curve_every == 0 or step == args.steps:
                    sync(device)
                    t0 = time.perf_counter()
                    metrics = evaluate(model, datasets, device, args.batch, recorder)
                    proposal_metrics = None
                    if callable(getattr(model, 'materialize', None)):
                        proposal_metrics = evaluate(model, datasets, device, args.batch, recorder, proposal=True)
                        if stage not in promoted_stages and proposal_metrics[stage]['valid']['hard_acc'] >= .95:
                            report = model.materialize(stage, ALPHABETS[stage], ALPHABETS)
                            recorder.emit('operation_materialization', model=name, stage=stage, step=step,
                                          proposal_valid=proposal_metrics[stage]['valid'], report=report,
                                          candidate_from_trained_model_not_teacher=True)
                            if report.get('success', False):
                                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
                                admitted = model.consolidate(stage, ALPHABETS[stage])
                                promoted_stages.add(stage)
                                recorder.emit('native_operation_admitted', model=name, stage=stage, step=step,
                                              reference=admitted, optimizer_rebuilt=True,
                                              remaining_parameters_trainable=True, router_MI_stopped_for_current_stage=True)
                                metrics = evaluate(model, datasets, device, args.batch, recorder)
                    sync(device)
                    eval_seconds += time.perf_counter()-t0
                    t0 = time.perf_counter()
                    delta = parameter_delta(model, before_stage)
                    graph = learned_snapshot(model, stage)
                    snapshot_seconds += time.perf_counter()-t0
                    recorder.emit("curve", model=name, stage=stage, step=step,
                                  temperature=temperature, terminal_metrics=metrics,
                                  actual_temperature=getattr(model, 'temperature', None),
                                  proposal_terminal_metrics=proposal_metrics,
                                  native_promotion=args.native_promotion,
                                  actual_operation_bank_size=getattr(model, 'n_ops', None),
                                  router_MI_weight=args.router_mi,
                                  router_negative_MI_mean=float(np.mean(auxiliary_losses)) if auxiliary_losses else None,
                                  geometry_loss_mean=float(np.mean(geometry_losses)) if geometry_losses else None,
                                  continuous_geometry=model.geometry_resource_report() if callable(getattr(model, 'geometry_resource_report', None)) else None,
                                  temperature_guard_checks=temperature_checks,
                                  temperature_guard_rejected=temperature_rejected,
                                  temperature_guard_seconds=temperature_seconds,
                                  current=metrics[stage], old={t: metrics[t] for t in range(stage)},
                                  unlearned={t: metrics[t] for t in range(stage+1, 3)},
                                  train_CE_mean=float(np.mean(losses)) if losses else None,
                                  gradient_norm_mean=float(np.mean(gradients)) if gradients else None,
                                  parameter_delta_from_stage_start=delta,
                                  learned_structure=graph, guard=guards,
                                  measured_train_seconds=train_seconds, measured_eval_seconds=eval_seconds,
                                  measured_snapshot_seconds=snapshot_seconds,
                                  optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                                  train_current_samples=train_current, train_old_samples=train_old,
                                  gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
                    losses.clear(); gradients.clear(); auxiliary_losses.clear(); geometry_losses.clear()
                    if step and (step % 100 == 0 or step == args.steps or args.pilot):
                        save_checkpoint(model, optimizer, args, name, stage, step, recorder)
                if step == args.steps:
                    break
                # Step k consumes exactly the same fixed current indices in every arm.
                index = batches[stage][step]
                if replay_memory is None:
                    b = fetch(datasets[stage]["train"], index, device)
                    train_current += args.batch
                else:
                    current = fetch(datasets[stage]["train"], index[:args.batch//2], device)
                    old = fetch(replay_memory, old_indices[stage][step], device)
                    b = {k: torch.cat([current[k], old[k]], 0) for k in ARRAY_KEYS}
                    train_current += args.batch//2
                    train_old += args.batch//2
                t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                logits = model(b["tokens"], b["starts"], b["task_ids"])
                ce = F.cross_entropy(logits, b["targets"])
                effective_router_mi = 0. if stage in promoted_stages else args.router_mi
                auxiliary = (model.router_regularizer(b['tokens'], b['task_ids'])
                             if effective_router_mi and callable(getattr(model, 'router_regularizer', None))
                             else ce.new_zeros(()))
                geometry = (model.geometry_regularizer()
                            if callable(getattr(model, 'geometry_regularizer', None)) else ce.new_zeros(()))
                if callable(getattr(model, 'geometry_moment_trace', None)):
                    geometry_event = model.geometry_moment_trace()
                    if geometry_event is not None:
                        recorder.emit('geometry_moment_state', model=name, stage=stage, step=step,
                                      moment_state=geometry_event)
                loss = ce + effective_router_mi*auxiliary + args.geometry_weight*geometry
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite terminal cross entropy")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError("nonfinite gradient norm")
                need_guard = name in ("topo", "constraint")
                t_snapshot = time.perf_counter()
                before = ({n:p.detach().clone() for n,p in model.named_parameters()}
                          if need_guard and getattr(model, "reference", []) else {})
                guards["snapshot_seconds"] += time.perf_counter()-t_snapshot
                optimizer.step()
                enforce = getattr(model, "enforce_frozen_rows_", None)
                if callable(enforce):
                    enforce()
                sync(device)
                train_seconds += time.perf_counter()-t0
                if need_guard:
                    t_guard = time.perf_counter()
                    report = model.project_update(before)
                    sync(device)
                    guards["measured_seconds"] += time.perf_counter()-t_guard
                    guards["reported_seconds"] += float(report.get("seconds", 0.))
                    guards["calls"] += 1
                    alpha = float(report.get("alpha", 1.))
                    guards["accepted_full" if alpha == 1 else "accepted_backtracked" if alpha > 0 else "rejected"] += 1
                    guards["checks"] += int(report.get("checks", 0))
                    guards["relation_pairs"] += int(report.get("relation_pairs", 0))
                losses.append(float(ce.item())); gradients.append(float(norm.item()))
                auxiliary_losses.append(float(auxiliary.item()))
                geometry_losses.append(float(geometry.item()))
            t0 = time.perf_counter()
            learned_enough = metrics[stage]['valid']['hard_acc'] >= .8
            consolidation = (model.consolidate(stage, ALPHABETS[stage])
                             if callable(getattr(model, "consolidate",None)) and learned_enough and stage not in promoted_stages
                             else {'available': True, 'reason':'already admitted native learned operation at materialization'} if stage in promoted_stages
                             else {"available":False,"reason":"validation hard accuracy below admission .8",
                                   "measured_valid_hard":metrics[stage]['valid']['hard_acc']})
            consolidation_seconds = time.perf_counter()-t0
            row = dict(stage=stage, terminal_metrics=metrics,
                       acquisition_iid=metrics[stage]["iid"],
                       old_iid={t: metrics[t]["iid"] for t in range(stage)},
                       guard=guards, parameters=parameter_counts(model),
                       optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                       representation_state_bytes=model.representation_state_bytes() if callable(getattr(model, "representation_state_bytes", None)) else 0,
                       consolidation=consolidation, consolidation_seconds=consolidation_seconds,
                       temperature_guard_checks=temperature_checks,
                       temperature_guard_rejected=temperature_rejected,
                       temperature_guard_seconds=temperature_seconds,
                       wall_seconds=time.perf_counter()-stage_started,
                       train_seconds=train_seconds, eval_seconds=eval_seconds,
                       train_current_samples=train_current, train_old_samples=train_old,
                       replay_memory_tensor_bytes=tensor_bytes(replay_memory),
                       continuous_geometry=model.geometry_resource_report() if callable(getattr(model, 'geometry_resource_report', None)) else None,
                       gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       optimizer_state_not_rolled_back_on_guard_rejection=True)
            stages.append(row)
            retention.append({t: metrics[t]["iid"] for t in range(3)})
            recorder.emit("stage_end", model=name, **row)
            save_checkpoint(model, optimizer, args, name, stage, args.steps, recorder, "-consolidated")
        except BaseException as error:
            try:
                save_checkpoint(model, optimizer, args, name, stage, step, recorder, "-interrupted")
            except Exception as save_error:
                recorder.emit("interrupted_checkpoint_failed", model=name, stage=stage, reason=str(save_error))
            recorder.emit("model_interrupted", model=name, stage=stage, step=step,
                          error=type(error).__name__, reason=str(error), completed_stages=stages)
            raise
    result = dict(model=name, status="THREE_STAGES_COMPLETED" if args.stages==3 and args.start_stage==0 else "FOCUSED_STAGE_TRAINING_COMPLETED", stages=stages,
                  actual_stages_executed=list(range(args.start_stage,args.stages)), resumed_previous_stage=str(args.resume_from) if args.resume_from else None,
                  task_by_training_stage_iid=retention,
                  model_wall_seconds=time.perf_counter()-started,
                  M0="NOT_PASSED", scientific_success_not_inferred_from_completion=True)
    recorder.emit("model_end", **result)
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--models", default="topo,transformer,replay,adapter,topo_off,program")
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument('--router-mi', type=float, default=0.)
    parser.add_argument('--native-promotion', action='store_true')
    parser.add_argument('--workspace-refinement', action='store_true')
    parser.add_argument('--continuous-monodromy', action='store_true')
    parser.add_argument('--geometry-weight', type=float, default=.01)
    parser.add_argument('--geometry-moment-cache', type=int, choices=(0,32,128), default=0)
    parser.add_argument('--homotopy-lift-cache', type=int, choices=(0,32,128), default=0)
    parser.add_argument('--program-v2', action='store_true')
    parser.add_argument('--program-plastic', action='store_true')
    parser.add_argument('--program-hard-assignment', choices=('rowargmax','bijective'), default='rowargmax')
    parser.add_argument('--start-stage', type=int, choices=(0,1,2), default=0)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--resume-repaired-stage-checkpoint', action='store_true')
    parser.add_argument("--latent-factor", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument('--stages',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--compilation',choices=('component','joint'),default='component')
    parser.add_argument('--credit',choices=('prob','log'),default='log')
    parser.add_argument('--protection-timing',choices=('early','admitted','off'),default='admitted')
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-seconds", type=float, default=1650.)
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (120 if args.pilot else 800)
    args.curve_every = 40 if args.pilot else 100
    args.models = [x.strip() for x in args.models.split(",") if x.strip()]
    if not args.models or len(set(args.models)) != len(args.models) or any(x not in ARMS for x in args.models):
        parser.error("--models must contain distinct supported comma-separated arm names")
    if args.seed < 0 or not 1 <= args.steps <= 800 or args.batch != 64:
        parser.error("require seed>=0,1<=steps<=800 and the frozen batch64")
    if args.pilot and args.steps > 120:
        parser.error("pilot is bounded to120 updates/stage")
    if not 0 < args.lr <= .1 or not 0 < args.max_seconds < 1800:
        parser.error("require0<lr<=.1 and0<max-seconds<1800")
    if not 0 <= args.router_mi <= 1.:
        parser.error('router MI weight must lie in[0,1]')
    if args.continuous_monodromy and not args.native_promotion:
        parser.error('continuous monodromy extends the current native-promotion architecture')
    if args.geometry_moment_cache and not args.continuous_monodromy:
        parser.error('geometry moment cache requires the continuous model')
    if args.homotopy_lift_cache and not args.continuous_monodromy:
        parser.error('homotopy lift cache requires the continuous model')
    if not 0 <= args.geometry_weight <= 1.:
        parser.error('geometry weight must lie in[0,1]')
    if args.start_stage >= args.stages or bool(args.resume_from) != bool(args.start_stage) or (args.resume_from and len(args.models)!=1):
        parser.error('resume requires one arm, preceding completed checkpoint and start-stage in1..stages-1')
    if args.resume_repaired_stage_checkpoint and not args.resume_from:
        parser.error('repaired-stage resume requires an explicit checkpoint with charged repair provenance')
    return args


def main():
    args = arguments()
    recorder = Recorder(args.out.resolve(), args.max_seconds)
    results = []
    try:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                              else "cpu" if args.device == "auto" else args.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_per_process_memory_fraction(.30, device)
        torch.set_num_threads(1)
        recorder.emit("configuration", args=vars(args), device=str(device),
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      cuda_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                      MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_PASSED", M3="ultimate",
                      supervision="terminal_CE_only", all_task_ids_equal_access=True,
                      teacher_matrices_or_states_passed=False, gpu_allocation_fraction=.30 if device.type == "cuda" else None,
                      automatic_tuning_retry_growth=False)
        raw_data, datasets = build_data(args, recorder)
        batches, old_indices = {}, {}
        for stage in range(3):
            rng = np.random.default_rng(args.seed*1009+stage*100_003+7_000_001)
            lengths = np.count_nonzero(raw_data[stage]['train']['tokens'],axis=1)
            schedule=[]
            for step in range(args.steps):
                fraction=step/max(1,args.steps)
                cap=(2 if stage==2 else 1) if fraction<.2 else min(3,CAPS[stage]) if fraction<.5 else CAPS[stage]
                pool=np.flatnonzero(lengths<=cap)
                schedule.append(rng.choice(pool,size=args.batch,replace=True))
            batches[stage] = np.asarray(schedule,dtype=np.int64)
            replay_rng = np.random.default_rng(args.seed*1013+stage*100_019+9_000_001)
            old_indices[stage] = replay_rng.integers(0, max(1, 4096*stage),
                                                  size=(args.steps, args.batch//2), dtype=np.int64)
            np.savez_compressed(recorder.out / f"batch-indices-stage{stage}.npz",
                                current=batches[stage], uniform_old=old_indices[stage])
            recorder.emit("fixed_batch_schedule", stage=stage,
                          current_sha256=hashlib.sha256(batches[stage].tobytes()).hexdigest(),
                          old_sha256=hashlib.sha256(old_indices[stage].tobytes()).hexdigest(),
                          replay_uses_first32_current_indices=True)
        recorder.emit("teacher_data_storage", actual_array_bytes=sum(data_bytes(d) for task in raw_data for d in task.values()),
                      shared_across_arms=True, not_optimizer_or_model_input_labels_beyond_terminal=True)
        for name in args.models:
            results.append(run_model(name, args, datasets, raw_data, batches, old_indices, device, recorder))
            recorder.summary(dict(status="PARTIAL_MODELS_COMPLETE", completed=results,
                                  requested_models=args.models, M0="NOT_PASSED"))
        recorder.emit("run_end", status="REQUESTED_TRAINING_COMPLETED", completed_models=[r["model"] for r in results], M0="NOT_PASSED")
        recorder.summary(dict(status="REQUESTED_TRAINING_COMPLETED", completed=results,
                              requested_models=args.models, M0="NOT_PASSED",
                              limitation="actual numerical learning records require analysis; smoke or completion is not milestone passage"))
    except BaseException as error:
        recorder.emit("run_failed", error=type(error).__name__, reason=str(error),
                      traceback=traceback.format_exc(), completed_models=[r["model"] for r in results],
                      no_automatic_retry=True)
        recorder.summary(dict(status="FAILED_WITH_PARTIAL_RECORDS", completed=results,
                              error=type(error).__name__, reason=str(error), M0="NOT_PASSED"))
        raise
    finally:
        recorder.file.close()


if __name__ == "__main__":
    main()
