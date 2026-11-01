"""Inspect an OWN TRUSTED acquisition checkpoint; never train or query a teacher.

python inspect_acquisition.py --checkpoint RUN/checkpoints/topo-stage0-step800.pt \
    --data-dir RUN/datasets --out NEW_DIAGNOSTIC_DIR --tasks 0 --device cuda

Loads the supplied local checkpoint with weights_only=False (trusted files only).
Uses saved terminal examples, never tasks.py/teacher maps or intermediate labels.
Writes prediction arrays, token/length strata, effective operator diagnostics and
terminal-loss gradient norms. No optimizer step, guard, consolidation or retry.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F


MAX_BYTES = 100 * 1024 * 1024


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, set)):
        return [plain(v) for v in x]
    if torch.is_tensor(x):
        return plain(x.detach().cpu().tolist())
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_sha(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def infer_temperature(checkpoint, payload):
    extra = payload.get("model_extra", {})
    if "temperature" in extra:
        return float(extra["temperature"]), "checkpoint.model_extra"
    records = checkpoint.parent.parent / "records.jsonl"
    found = None
    if records.exists():
        for line in records.open():
            row = json.loads(line)
            if (row.get("kind") == "curve" and row.get("model") == payload["model_name"]
                    and row.get("stage") == payload["stage"] and row.get("step", 10**9) <= payload["step"]):
                found = row.get("temperature")
    return (float(found), "saved_curve") if found is not None else (1., "UNCONFIRMED_default_override_with_--temperature")


def restore(payload, checkpoint, device, override_temperature):
    cfg, state = payload["args"], payload["model"]
    name = payload["model_name"]
    extra = payload.get("model_extra", {})
    if "operations" in state:
        from topology_model import TopologyExecutor
        n, d = state["encoder"].shape
        n_ops = state["operations"].shape[0]
        slots = state["call.bias"].numel() // (n_ops+1)
        model = TopologyExecutor(n, vocab=state["token.weight"].shape[0]-1,
                                n_tasks=state["task.weight"].shape[0], latent_factor=d//n,
                                n_ops=n_ops, slots=slots, emb_dim=state["token.weight"].shape[1],
                                protection=bool(extra.get("protection", name != "topo_off")),
                                constraint=name == "constraint", seed=int(cfg.get("seed", 0)))
        model.compilation = extra.get("compilation", cfg.get("compilation", "component"))
        model.credit = extra.get("credit", cfg.get("credit", "prob"))
        model.reference = extra.get("reference", [])
    elif "transitions" in state:
        from torch_baselines import ProgramLearner
        n, d = state["encoder"].shape
        model = ProgramLearner(n, vocab=state["transitions"].shape[0],
                               n_tasks=state["task_bias"].shape[0], latent_factor=d//n, modular=True)
    else:
        from torch_baselines import TinyTransformer
        n, dm = state["starts.weight"].shape
        adapter = "heads.0.weight" in state
        layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("layers."))
        model = TinyTransformer(n, vocab=state["tokens.weight"].shape[0],
                                max_len=state["positions"].shape[0]-2, d_model=dm,
                                n_heads=4, n_layers=layers,
                                n_tasks=state["tasks.weight"].shape[0], adapter=adapter)
        initialized = payload.get("masks_and_initialized_tasks", {}).get("_initialized_tasks")
        if initialized is not None:
            model._initialized_tasks = set(initialized)
    model.load_state_dict(state, strict=True)
    model.to(device)
    stage = int(payload["stage"])
    if hasattr(model, "_initialized_tasks"):
        model._initialized_tasks.update(range(stage+1))
    # Restore actual training masks without overwriting trained adapter heads.
    model.set_stage(stage, new_token_ids=((1, 2, 3), (4, 5), (6,))[stage])
    temperature, source = infer_temperature(checkpoint, payload)
    if override_temperature is not None:
        temperature, source = override_temperature, "explicit_diagnostic_override"
    if callable(getattr(model, "set_temperature", None)):
        model.set_temperature(temperature)
    model.eval()
    return model, dict(model=name, n_states=n, latent_states=getattr(model, "d", getattr(model, "dim", None)),
                       stage=stage, step=payload["step"], temperature=temperature,
                       temperature_source=source, compilation=getattr(model, "compilation", None),
                       credit=getattr(model, "credit", None),
                       training_masks_restored=True, source_code_is_current_not_historical=True)


def metrics(rows, soft, hard, nll, target_probability, mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return dict(n=0)
    target = rows["targets"][ids]
    return dict(n=len(ids), soft_acc=float(np.mean(soft[ids] == target)),
                hard_acc=float(np.mean(hard[ids] == target)),
                mean_terminal_NLL=float(nll[ids].mean()),
                mean_observed_target_probability=float(target_probability[ids].mean()),
                soft_hard_disagreement=float(np.mean(soft[ids] != hard[ids])))


@torch.no_grad()
def predict(model, data, device, batch_size, check):
    probs, hard = [], []
    for begin in range(0, len(data["starts"]), batch_size):
        check()
        sl = slice(begin, begin+batch_size)
        tokens = torch.as_tensor(data["tokens"][sl], device=device)
        starts = torch.as_tensor(data["starts"][sl], device=device)
        tasks = torch.as_tensor(data["task_ids"][sl], device=device)
        logits = model(tokens, starts, tasks)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite soft diagnostic logits")
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        call = getattr(model, "hard_eval", None)
        h = call(tokens, starts, tasks) if callable(call) else logits
        hard.append(h.argmax(-1).cpu().numpy())
    p = np.concatenate(probs)
    h = np.concatenate(hard)
    sp = p.argmax(-1)
    tp = p[np.arange(len(p)), data["targets"]]
    nll = -np.log(np.maximum(tp, 1e-30))
    lengths = np.count_nonzero(data["tokens"], axis=1)
    report = dict(overall=metrics(data, sp, h, nll, tp, np.ones(len(h), dtype=bool)),
                  per_length={int(l): metrics(data, sp, h, nll, tp, lengths == l)
                              for l in np.unique(lengths)},
                  per_token_presence={}, one_step_by_token={})
    for token in np.unique(data["tokens"]):
        if token == 0:
            continue
        present = (data["tokens"] == token).any(-1)
        report["per_token_presence"][int(token)] = metrics(data, sp, h, nll, tp, present)
        report["one_step_by_token"][int(token)] = metrics(data, sp, h, nll, tp, present & (lengths == 1))
    arrays = dict(tokens=data["tokens"], starts=data["starts"], task_ids=data["task_ids"],
                  observed_terminal_targets=data["targets"], lengths=lengths,
                  soft_probabilities=p, soft_predictions=sp, hard_predictions=h,
                  observed_target_probability=tp, terminal_nll=nll)
    return report, arrays


def entropy(p):
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


def matrix_stats(matrix):
    m = matrix.detach()
    singular = torch.linalg.svdvals(m)
    return dict(mean_row_entropy=float(entropy(m).mean()),
                maximum_row_entropy=float(entropy(m).max()),
                mean_row_max_probability=float(m.max(-1).values.mean()),
                minimum_row_max_probability=float(m.max(-1).values.min()),
                mean_identity_diagonal=float(m.diagonal().mean()),
                max_row_sum_error=float((m.sum(-1)-1).abs().max()),
                max_column_sum_error=float((m.sum(-2)-1).abs().max()),
                second_singular_value=float(singular[1]) if len(singular)>1 else None,
                max_pairwise_row_TV=float((m[:, None]-m[None]).abs().sum(-1).max()/2))


def permutation_matrix(permutation, device, dtype):
    p = torch.as_tensor(permutation, device=device)
    return F.one_hot(p, len(p)).to(dtype)


def covered_accuracy(probabilities, counts):
    predictions = probabilities.argmax(-1).detach().cpu().numpy()
    total = int(counts.sum())
    return dict(n_observed_one_step_samples=total,
                accuracy_on_observed_samples=float(sum(counts[i, predictions[i]] for i in range(len(counts)))/total) if total else None,
                observed_target_mean_probability=float((counts*probabilities.detach().cpu().numpy()).sum()/total) if total else None,
                predictions=predictions, probabilities=probabilities)


@torch.no_grad()
def token_diagnostics(model, training, n, device):
    """Compare model matrices only to length-one labels already in the dataset."""
    from topology_model import assignment
    is_bank = hasattr(model, "soft_ops")
    if not is_bank and not hasattr(model, "transitions"):
        return dict(available=False, reason="no explicit learned transition bank")
    tau = model.temperature
    dim = model.d if is_bank else model.dim
    decoder = torch.softmax(model.decoder/tau, -1)
    hard_decoder = F.one_hot(model.decoder.argmax(-1), n).to(decoder.dtype)
    bank = model.soft_ops() if is_bank else torch.softmax(model.transitions/tau, -1)
    hard_bank = (torch.stack([permutation_matrix(assignment(x), device, x.dtype) for x in model.operations])
                 if is_bank else F.one_hot(model.transitions.argmax(-1), dim).to(bank.dtype))
    if is_bank:
        hard_bank = torch.cat([torch.eye(dim, device=device)[None], hard_bank], 0)
    result = dict(available=True, teacher_transition_map_used=False,
                  one_step_scope="saved initial-state terminal samples only; no hidden-mode transition oracle",
                  bank_stats={i: matrix_stats(m) for i, m in enumerate(bank)},
                  bank_pairwise_mean_L1=(bank[:, None]-bank[None]).abs().mean((-1, -2)), tasks={})
    for task, data in training.items():
        lengths = np.count_nonzero(data["tokens"], -1)
        token_ids = sorted(int(x) for x in np.unique(data["tokens"]) if x)
        encoding_logits = model.encoder if is_bank else model.encoder + model.task_bias[task]
        enc = torch.softmax(encoding_logits/tau, -1)
        hard_enc = F.one_hot(encoding_logits.argmax(-1), dim).to(enc.dtype)
        token_rows = {}
        for token in token_ids:
            counts = np.zeros((n, n), np.int64)
            ids = np.flatnonzero((lengths == 1) & (data["tokens"] == token).any(-1))
            np.add.at(counts, (data["starts"][ids], data["targets"][ids]), 1)
            eye = torch.eye(dim, device=device)
            if is_bank:
                tok = torch.tensor([[token]], device=device)
                task_id = torch.tensor([task], device=device)
                call_logits = model.calls(tok, task_id)[0, 0]
                route = torch.softmax(call_logits/tau, -1)
                effective, hard_call, soft_call_hard_bank, component = [eye.clone() for _ in range(4)]
                for slot in range(model.slots):
                    k = int(route[slot].argmax())
                    effective = effective @ torch.einsum('k,kij->ij', route[slot], bank)
                    hard_call = hard_call @ bank[k]
                    soft_call_hard_bank = soft_call_hard_bank @ torch.einsum('k,kij->ij', route[slot], hard_bank)
                    component = component @ hard_bank[k]
                routing = dict(call_logits=call_logits, mixtures=route, entropy_by_slot=entropy(route),
                               argmax_bank_by_slot=route.argmax(-1),
                               skip_probability_by_slot=route[:, 0])
            else:
                effective = bank[token]
                hard_call = effective
                soft_call_hard_bank = hard_bank[token]
                component = hard_bank[token]
                routing = dict(available=False)
            joint = permutation_matrix(assignment(effective.clamp_min(1e-30).log()), device, effective.dtype)
            variants = {"full_soft": enc @ effective @ decoder,
                        "hard_calls_soft_banks": enc @ hard_call @ decoder,
                        "soft_calls_hard_banks": enc @ soft_call_hard_bank @ decoder,
                        "component_hard_enc_transition_dec": hard_enc @ component @ hard_decoder,
                        "joint_hard_enc_transition_dec": hard_enc @ joint @ hard_decoder,
                        "soft_transition_hard_enc_dec": hard_enc @ effective @ hard_decoder,
                        "hard_joint_transition_soft_enc_dec": enc @ joint @ decoder}
            token_rows[token] = dict(observed_one_step_counts=counts,
                                     covered_starts=np.flatnonzero(counts.sum(-1)),
                                     conflicting_observed_starts=np.flatnonzero((counts>0).sum(-1)>1),
                                     effective_latent_transition=effective,
                                     effective_stats=matrix_stats(effective), routing=routing,
                                     diagnostic_variants={name: covered_accuracy(p, counts) for name,p in variants.items()},
                                     variant_scope="execution diagnostics only; no updates, extra labels, or learning claims")
        result["tasks"][task] = dict(tokens=token_rows, encoder_mean_entropy=float(entropy(enc).mean()),
                                      decoder_mean_entropy=float(entropy(decoder).mean()))
    return result


def gradient_diagnostics(model, data, task, device, batch, max_batches, seed, check):
    """Gradients of observed terminal losses only. Does not write param.grad."""
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    lengths = np.count_nonzero(data["tokens"], -1)
    subsets = {"all_sequences": np.arange(len(lengths)), "one_step": np.flatnonzero(lengths == 1),
               "multi_step": np.flatnonzero(lengths > 1)}
    for token in sorted(int(x) for x in np.unique(data["tokens"]) if x):
        subsets[f"one_step_token{token}"] = np.flatnonzero((lengths == 1) & (data["tokens"] == token).any(-1))
    output = {}
    rng = np.random.default_rng(seed+task*1009)
    for subset, candidates in subsets.items():
        check()
        if not len(candidates) or not params:
            output[subset] = dict(n=0, skipped=True)
            continue
        chosen = rng.permutation(candidates)[:batch*max_batches]
        sums = [torch.zeros_like(p, device="cpu") for _,p in params]
        total, loss_sum = 0, 0.
        for begin in range(0, len(chosen), batch):
            check()
            ids = chosen[begin:begin+batch]
            tok = torch.as_tensor(data["tokens"][ids], device=device)
            starts = torch.as_tensor(data["starts"][ids], device=device)
            tids = torch.as_tensor(data["task_ids"][ids], device=device)
            target = torch.as_tensor(data["targets"][ids], device=device)
            loss = F.cross_entropy(model(tok, starts, tids), target)
            grad = torch.autograd.grad(loss, [p for _,p in params], allow_unused=True)
            total += len(ids); loss_sum += float(loss.detach()) * len(ids)
            for value, g in zip(sums, grad):
                if g is not None:
                    value.add_(g.detach().cpu(), alpha=len(ids))
        norms = {}
        for (name, _), value in zip(params, sums):
            g = value / total
            norms[name] = dict(L2=float(g.norm()), Linf=float(g.abs().max()),
                               zero_fraction=float((g == 0).float().mean()), finite=bool(torch.isfinite(g).all()))
            if name in ("operations", "transitions"):
                norms[name]["per_bank_L2"] = g.flatten(1).norm(dim=1)
            if name in ("token.weight", "tokens.weight", "encoder", "decoder", "call.weight"):
                norms[name]["per_row_L2"] = g.flatten(1).norm(dim=1)
            if name == "call.bias":
                norms[name]["entries"] = g
        output[subset] = dict(n=total, terminal_CE=loss_sum/total, parameter_gradients=norms,
                              gradient_is_for_current_training_mask=True,
                              optimizer_step_or_new_supervision=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", default="0,1,2")
    parser.add_argument("--splits", default="train,valid,iid,ood,length16")
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=4096)
    parser.add_argument("--gradient-batches", type=int, default=4)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=300.)
    args = parser.parse_args()
    if not 1 <= args.batch <= 128 or not 1 <= args.max_rows <= 4096 or not 0 <= args.gradient_batches <= 8:
        parser.error("bounded batch<=128, rows<=4096, gradient-batches<=8 required")
    tasks = [int(t) for t in args.tasks.split(",")]
    if any(t not in (0,1,2) for t in tasks) or not 0 < args.max_seconds <= 600:
        parser.error("tasks must be0..2 and max-seconds in(0,600]")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def check():
        if time.perf_counter()-started >= args.max_seconds:
            raise TimeoutError("bounded diagnostic time exhausted")
    torch.set_num_threads(1)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.30, device)
    # Explicitly restricted to the user's own trusted generated checkpoint.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, restored = restore(payload, args.checkpoint, device, args.temperature)
    before = parameter_sha(model)
    data_dir = args.data_dir or args.checkpoint.parent.parent / "datasets"
    report = dict(schema="inspect-acquisition-v1", checkpoint=str(args.checkpoint),
                  checkpoint_sha256=sha(args.checkpoint), restored=restored,
                  model_source_sha256=sha(Path(__file__).with_name("topology_model.py")),
                  MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_PASSED", M3="ultimate",
                  no_teacher_import=True, no_training=True, no_new_terminal_queries=True,
                  datasets={}, gradients={}, warnings=[])
    training = {}
    def save():
        text = json.dumps(plain(report), sort_keys=True, indent=2, allow_nan=False)
        if len(text.encode()) > MAX_BYTES:
            raise RuntimeError("100MiB diagnostic report cap")
        (args.out / "diagnostics.json").write_text(text+"\n")
    save()
    for task in tasks:
        for split in args.splits.split(","):
            check()
            path = data_dir / f"task{task}-{split}.npz"
            if not path.exists():
                report["warnings"].append(f"missing saved dataset:{path}")
                continue
            with np.load(path, allow_pickle=False) as source:
                data = {k: np.asarray(source[k], dtype=np.int64)[:args.max_rows]
                        for k in ("tokens","starts","targets","task_ids")}
            if not len(data["starts"]):
                continue
            summary, predictions = predict(model, data, device, args.batch, check)
            prediction_file = args.out / f"task{task}-{split}-predictions.npz"
            np.savez_compressed(prediction_file, **predictions)
            if prediction_file.stat().st_size > MAX_BYTES:
                raise RuntimeError("prediction file cap exceeded")
            report["datasets"][f"task{task}-{split}"] = dict(file=str(path), sha256=sha(path),
                    prediction_file=prediction_file.name, metrics=summary,
                    per_token_presence_is_descriptive_not_causal=True)
            if split == "train":
                training[task] = data
            save()
    if training:
        report["token_transition_diagnostics"] = token_diagnostics(model, training, restored["n_states"], device)
        save()
        if args.gradient_batches:
            for task, data in training.items():
                report["gradients"][task] = gradient_diagnostics(model, data, task, device, args.batch,
                                                                  args.gradient_batches, args.seed, check)
                save()
    report["parameters_unchanged"] = parameter_sha(model) == before
    report["wall_seconds"] = time.perf_counter()-started
    report["status"] = "DIAGNOSTIC_COMPLETE"
    report["limitation"] = "current source reconstructs checkpoint; diagnostic execution variants and gradients are not training or milestone evidence"
    save()
    if not report["parameters_unchanged"]:
        raise RuntimeError("diagnostic unexpectedly changed parameter values")
    print(json.dumps(plain(dict(status=report["status"], report=str(args.out/"diagnostics.json"),
                                wall_seconds=report["wall_seconds"], parameters_unchanged=True))))


if __name__ == "__main__":
    main()
