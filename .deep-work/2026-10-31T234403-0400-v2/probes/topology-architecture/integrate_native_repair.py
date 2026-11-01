"""Install an admitted native repair in REAL neural weights and continue SGD.

Only trusted checkpoints produced by this campaign may be loaded (PyTorch's
weights_only=False is necessary for its saved RNG/NumPy/reference state). The
admitted repair is checked against the original model's regenerated graphs.
No teacher operation, hidden-state label, correct route or intermediate target
is supplied. No experiment runs at import; no network, SSH or subprocess calls.

MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
Run inside the existing external CUDA sandbox; this script is not a sandbox.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from semantic_promotion_model import SemanticPromotionModel


MAX_FILE_BYTES = 100 * 1024 * 1024
SPLITS = ("train", "valid", "iid", "ood", "length16")


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
        raise ValueError("nonfinite evidence value")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class Recorder:
    def __init__(self, out, seconds):
        out.mkdir(parents=True, exist_ok=False)
        self.out, self.started = out, time.perf_counter()
        self.deadline = self.started + seconds
        self.stream = (out / "records.jsonl").open("x", encoding="utf8")
        self.bytes = 0

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("finite integration/training wall budget reached")

    def emit(self, kind, **fields):
        line = json.dumps(plain(dict(kind=kind, wall=time.perf_counter()-self.started,
                                     **fields)), sort_keys=True, allow_nan=False) + "\n"
        self.bytes += len(line.encode())
        if self.bytes > MAX_FILE_BYTES:
            raise RuntimeError("100MiB event-file cap")
        self.stream.write(line)
        self.stream.flush()

    def json(self, name, payload):
        text = json.dumps(plain(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if len(text.encode()) > MAX_FILE_BYTES:
            raise RuntimeError("100MiB JSON-file cap")
        with (self.out / name).open("x", encoding="utf8") as stream:
            stream.write(text)


def declared_graph(raw):
    enc, dec = np.asarray(raw["encoder"], dtype=np.int64), np.asarray(raw["decoder"], dtype=np.int64)
    transitions = {int(a): np.asarray(p, dtype=np.int64)
                   for a, p in raw["learned_transitions"].items()}
    tokens = tuple(sorted(transitions))
    return dict(task=int(raw["task"]), enc=enc, dec=dec, trans=transitions,
                tokens=tokens, routes=np.asarray(raw["routes"], dtype=np.int64))


def compare_graph(expected, actual):
    if (expected["task"] != actual["task"] or tuple(expected["tokens"]) != tuple(actual["tokens"])
            or not np.array_equal(expected["enc"], actual["enc"])
            or not np.array_equal(expected["dec"], actual["dec"])
            or not np.array_equal(expected["routes"], actual["routes"])):
        return False
    return all(np.array_equal(expected["trans"][a], actual["trans"][a]) for a in expected["tokens"])


def graph_summary(graph):
    return dict(task=graph["task"], encoder=graph["enc"], decoder=graph["dec"],
                routes=graph["routes"], learned_transitions=graph["trans"],
                reachable=graph.get("reachable"), braid_lengths=graph.get("braid_lengths"),
                teacher_states_used=False)


def source_model(checkpoint, device):
    state, extra = checkpoint["model"], checkpoint.get("model_extra", {})
    n, d = state["encoder"].shape
    count, first, second = state["operations"].shape
    if (not 1 <= n <= 32 or d % n or not 1 <= d//n <= 2
            or not 1 <= count <= 24 or (first, second) != (d, d)):
        raise ValueError("checkpoint architecture exceeds declared dimensions")
    vocab, emb_dim = state["token.weight"].shape
    n_tasks = state["task.weight"].shape[0]
    slots = state["call.weight"].shape[0] // (count + 1)
    if vocab != 7 or n_tasks != 3 or slots != 2:
        raise ValueError("checkpoint differs from current native architecture")
    settings = extra.get("promotion_settings", {})
    model = SemanticPromotionModel(n, vocab=vocab-1, max_len=18, n_tasks=n_tasks,
        latent_factor=d//n, protection=True, seed=int(checkpoint["args"]["seed"]),
        n_ops=count, slots=slots, emb_dim=emb_dim,
        max_bank=int(extra.get("max_bank", 24)), **settings).to(device)
    model.load_state_dict(state)
    model.load_extra_state_dict(extra)
    model.protection = True
    model.compilation = "component"
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def load_datasets(folder, alphabets, recorder):
    result, inputs = {}, []
    for task in sorted(alphabets):
        result[task] = {}
        for split in SPLITS:
            pathname = folder / f"task{task}-{split}.npz"
            with np.load(pathname, allow_pickle=False) as archive:
                arrays = {k: np.asarray(archive[k], dtype=np.int64)
                          for k in ("tokens", "starts", "targets", "task_ids")}
            count = len(arrays["starts"])
            if (arrays["tokens"].ndim != 2 or arrays["tokens"].shape != (count, 18)
                    or count > 16384 or arrays["targets"].shape != (count,)
                    or arrays["task_ids"].shape != (count,)
                    or np.any(arrays["task_ids"] != task)
                    or set(int(a) for a in np.unique(arrays["tokens"]) if a) - set(alphabets[task])):
                raise ValueError("input dataset shape, task, token or budget mismatch")
            if np.any(arrays["tokens"] < 0):
                raise ValueError("negative observed token")
            result[task][split] = arrays
            inputs.append(dict(task=task, split=split, file=str(pathname),
                               sha256=sha256(pathname), rows=count,
                               array_bytes=sum(a.nbytes for a in arrays.values())))
    recorder.emit("input_datasets", datasets=inputs,
                  selection_policy="repair already frozen; only task2 TRAIN supplies subsequent SGD")
    return result, inputs


@torch.no_grad()
def evaluate(model, datasets, alphabets, device, batch, recorder, costs):
    synchronize(device)
    began, result = time.perf_counter(), {}
    model.eval()
    for task in sorted(datasets):
        graph = model.graph(task, alphabets[task])
        result[task] = {}
        for split, arrays in datasets[task].items():
            recorder.check()
            native = graph["enc"][arrays["starts"]].copy()
            words = arrays["tokens"]
            for column in range(words.shape[1]):
                for a in np.unique(words[:, column]):
                    if a:
                        selected = words[:, column] == a
                        native[selected] = graph["trans"][int(a)][native[selected]]
            hard_correct = int(np.count_nonzero(graph["dec"][native] == arrays["targets"]))
            count, loss, soft_correct = len(native), 0., 0
            for start in range(0, count, batch):
                recorder.check()
                b = {k: torch.as_tensor(v[start:start+batch], device=device, dtype=torch.long)
                     for k, v in arrays.items()}
                logits = model(b["tokens"], b["starts"], b["task_ids"])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite actual neural evaluation")
                loss += float(F.cross_entropy(logits, b["targets"], reduction="sum").item())
                soft_correct += int((logits.argmax(-1) == b["targets"]).sum().item())
            result[task][split] = dict(n=count, CE=loss/count, soft_acc=soft_correct/count,
                                      native_acc=hard_correct/count)
            costs["evaluation_soft_episodes"] += count
            costs["evaluation_native_transition_uses"] += int(np.count_nonzero(words))
    synchronize(device)
    costs["evaluation_seconds"] += time.perf_counter()-began
    model.train()
    return result


def save_checkpoint(model, optimizer, original, recorder, completed, provenance, suffix):
    synchronize(next(model.parameters()).device)
    began = time.perf_counter()
    path = recorder.out / f"topo-stage2-step{int(original['step'])+completed}-{suffix}.pt"
    payload = dict(schema="terminal-architecture-checkpoint-v1", args=copy.deepcopy(original["args"]),
        model_name=original["model_name"], stage=2, step=int(original["step"])+completed,
        model=model.state_dict(), optimizer=optimizer.state_dict(), model_extra=model.extra_state_dict(),
        masks_and_initialized_tasks={}, torch_rng=torch.get_rng_state(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        no_automatic_resume_or_retry=True, M0="NOT_PASSED",
        native_repair_integration=dict(provenance=provenance, completed_additional_updates=completed,
            original_training_args_preserved=True, original_stage_step=original["step"],
            additional_updates_are_NOT_retroactive_original_training=True))
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    if temporary.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError("saved checkpoint exceeds100MiB")
    temporary.replace(path)
    recorder.emit("checkpoint", file=path.name, bytes=path.stat().st_size,
                  sha256=sha256(path), additional_updates=completed,
                  seconds=time.perf_counter()-began, actual_neural_parameters=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True,
                        help="directory containing frozen-native-graphs.json and summary.json")
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--max-seconds", type=float, default=850.)
    args = parser.parse_args()
    if (args.updates != 200 or args.batch != 64 or args.lr != .01
            or not 0 < args.max_seconds <= 1500 or not 0 <= args.seed < 2**31):
        parser.error("frozen diagnostic requires200 updates,batch64,LR.01,bounded seed/time")
    for pathname in (args.checkpoint, args.repair / "summary.json",
                     args.repair / "frozen-native-graphs.json"):
        if not pathname.is_file() or pathname.stat().st_size > MAX_FILE_BYTES:
            parser.error("trusted checkpoint/repair input missing or exceeds100MiB")
    if args.out.exists():
        parser.error("new output directory required; no source record is overwritten")
    repair = json.loads((args.repair / "frozen-native-graphs.json").read_text())
    repair_summary = json.loads((args.repair / "summary.json").read_text())
    if (repair.get("status") != "ADMITTED_NATIVE_GRAPH_REPAIR"
            or not repair_summary.get("admitted") or not repair.get("admitted_bank_updates")):
        raise ValueError("only an ACTUALLY admitted terminal-validated repair may be installed")
    before_expected = {int(g["task"]): declared_graph(g) for g in repair["source_graphs"]}
    after_expected = {int(g["task"]): declared_graph(g) for g in repair["graphs"]}
    if set(before_expected) != {0, 1, 2} or set(after_expected) != {0, 1, 2}:
        raise ValueError("three-stage actual repair graphs required")
    alphabets = {t: before_expected[t]["tokens"] for t in before_expected}
    recorder = Recorder(args.out, args.max_seconds)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested existing GPU is unavailable; no silent CPU substitute")
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.3, device)
        torch.cuda.reset_peak_memory_stats(device)
    source_files = [Path(__file__), Path(__file__).with_name("topology_model.py"),
                    Path(__file__).with_name("semantic_promotion_model.py")]
    provenance = dict(checkpoint_path=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        repair_graph_sha256=sha256(args.repair / "frozen-native-graphs.json"),
        repair_summary_sha256=sha256(args.repair / "summary.json"),
        actual_code_sha256={p.name: sha256(p) for p in source_files})
    recorder.emit("configuration", args=vars(args), provenance=provenance,
        terminal_training_only=True, original_inputs_readonly_by_contract=True,
        old_task_scope=[0, 1], current_task_reference_revisable_only_by_admitted_repair=True,
        primary_architecture_baseline="Transformer", MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_PASSED")
    model = optimizer = None
    completed = 0
    costs = dict(train_seconds=0., guard_seconds=0., evaluation_seconds=0.,
                 evaluation_soft_episodes=0, evaluation_native_transition_uses=0,
                 training_current_episodes=0, training_old_episodes=0,
                 guard_calls=0, guard_full=0, guard_backtracked=0, guard_rejected=0,
                 guard_relation_checks=0, guard_relation_pairs=0)
    try:
        recorder.check()
        original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (original.get("schema") != "terminal-architecture-checkpoint-v1"
                or original.get("model_name") != "topo" or int(original.get("stage", -1)) != 2):
            raise ValueError("trusted current stage2 topo checkpoint required")
        model = source_model(original, device)
        if "torch_rng" in original:
            torch.set_rng_state(original["torch_rng"])
        if "numpy_rng" in original:
            np.random.set_state(original["numpy_rng"])
        if "python_rng" in original:
            random.setstate(original["python_rng"])
        if device.type == "cuda" and original.get("cuda_rng"):
            torch.cuda.set_rng_state_all(original["cuda_rng"])
        reference_before = copy.deepcopy(model.reference)
        old_references = [r for r in reference_before if int(r["task"]) < 2]
        replaced_references = [r for r in reference_before if int(r["task"]) == 2]
        if any(int(r["task"]) > 2 for r in reference_before) or {int(r["task"]) for r in old_references} != {0, 1}:
            raise ValueError("checkpoint old protection references missing or out of scope")
        actual_before = {t: model.graph(t, alphabet) for t, alphabet in alphabets.items()}
        if not all(compare_graph(before_expected[t], actual_before[t]) for t in before_expected):
            raise ValueError("repair's source graphs do not match REAL checkpoint native execution")
        before_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        recorder.emit("source_verified", actual_graphs=[graph_summary(g) for g in actual_before.values()],
            latent_states=model.d, bank_size=model.n_ops, parameters=sum(p.numel() for p in model.parameters()),
            current_temperature=model.temperature, old_reference_count=len(old_references),
            provisional_current_references_to_replace=len(replaced_references))
        datasets, input_sources = load_datasets(args.datasets, alphabets, recorder)
        before_metrics = evaluate(model, datasets, alphabets, device, args.batch, recorder, costs)
        recorder.emit("before_neural_installation", metrics=before_metrics)
        synchronize(device)
        installation_start = time.perf_counter()
        with torch.no_grad():
            for update in repair["admitted_bank_updates"]:
                expert = int(update["expert"])
                permutation = np.asarray(update["native_permutation"], dtype=np.int64)
                if (not 1 <= expert <= model.n_ops or permutation.shape != (model.d,)
                        or not np.array_equal(np.sort(permutation), np.arange(model.d))):
                    raise ValueError("admitted bank update outside current architecture")
                weights = model.operations[expert-1]
                weights.fill_(-8.)
                weights[torch.arange(model.d, device=device),
                        torch.as_tensor(permutation, device=device)] = 8.
            for name, parameter in model.named_parameters():
                if name != "operations" and not torch.equal(parameter, before_params[name]):
                    raise RuntimeError("installation altered encoder/decoder/call representation")
        actual_after = {t: model.graph(t, alphabet) for t, alphabet in alphabets.items()}
        if not all(compare_graph(after_expected[t], actual_after[t]) for t in after_expected):
            raise RuntimeError("REAL regenerated neural graphs differ from the admitted repair")
        old_certificates = []
        for reference in old_references:
            okay, count = model.related(reference, actual_after[int(reference["task"])])
            old_certificates.append(dict(task=reference["task"], related=bool(okay), relation_pairs=count))
            if not okay:
                raise RuntimeError("neural installation changed protected old task behavior")
        # Current-task learned semantics were provisional and contradicted by
        # terminal records. Revise ONLY that reference after the admitted repair.
        model.reference = old_references
        admission = model.consolidate(2, alphabets[2])
        model.stage = 2
        model.protection = True
        synchronize(device)
        costs["installation_seconds"] = time.perf_counter()-installation_start
        recorder.emit("actual_neural_repair_installed", actual_graphs=[graph_summary(g) for g in actual_after.values()],
            changed_experts=[int(u["expert"]) for u in repair["admitted_bank_updates"]],
            changed_operation_scalars=int(torch.count_nonzero(model.operations-before_params["operations"]).item()),
            other_parameter_tensors_unchanged=True, old_certificates=old_certificates,
            replaced_provisional_reference_count=len(replaced_references), new_current_reference=admission,
            reference_revision_source="previously admitted training-terminal repair; no teacher states")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        installed_checkpoint = save_checkpoint(model, optimizer, original, recorder, 0, provenance, "repair-installed")
        generator = np.random.default_rng(args.seed)
        indices = generator.integers(0, len(datasets[2]["train"]["starts"]),
                                    size=(args.updates, args.batch), dtype=np.int64)
        np.savez_compressed(recorder.out / "repair-training-batch-indices.npz", indices=indices,
                            seed=np.asarray(args.seed), original_stage=np.asarray(2))
        recorder.emit("additional_training_begin", updates=args.updates, batch=args.batch, lr=args.lr,
            fresh_adam=True, no_router_auxiliary=True, fixed_temperature=model.temperature,
            task=2, data="original task2-train.npz only", all_parameters_trainable=True,
            sample_indices_sha256=sha256(recorder.out / "repair-training-batch-indices.npz"),
            additional_budget_not_original800=True,
            optimizer_state_not_rolled_back_on_guard_rejection=True)
        curves = []
        for step in range(args.updates+1):
            recorder.check()
            if step in (0, 50, 100, 200):
                metrics = evaluate(model, datasets, alphabets, device, args.batch, recorder, costs)
                curve = dict(step=step, metrics=metrics,
                    actual_graphs=[graph_summary(model.graph(t, a)) for t, a in alphabets.items()],
                    optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                    parameters=sum(p.numel() for p in model.parameters()),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    representation_state_bytes=model.representation_state_bytes(),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    guard_counts=costs.copy())
                curves.append(curve)
                recorder.emit("curve", **curve)
                if step:
                    save_checkpoint(model, optimizer, original, recorder, step, provenance, "repair-trained")
            if step == args.updates:
                break
            ix = indices[step]
            batch = {k: torch.as_tensor(v[ix], device=device, dtype=torch.long)
                     for k, v in datasets[2]["train"].items()}
            synchronize(device)
            train_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"], batch["starts"], batch["task_ids"])
            loss = F.cross_entropy(logits, batch["targets"])
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite terminal training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError("nonfinite training gradient")
            proposal_before = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer.step()
            synchronize(device)
            costs["train_seconds"] += time.perf_counter()-train_start
            guard_start = time.perf_counter()
            guard = model.project_update(proposal_before)
            synchronize(device)
            costs["guard_seconds"] += time.perf_counter()-guard_start
            alpha = float(guard.get("alpha", 1.))
            costs["guard_calls"] += 1
            costs["guard_full" if alpha == 1 else "guard_backtracked" if alpha > 0 else "guard_rejected"] += 1
            costs["guard_relation_checks"] += int(guard.get("checks", 0))
            costs["guard_relation_pairs"] += int(guard.get("relation_pairs", 0))
            costs["training_current_episodes"] += args.batch
            completed = step+1
            recorder.emit("actual_parameter_update", step=completed, terminal_CE=float(loss.item()),
                          preclip_gradient_norm=float(gradient.item()), guard=guard)
        final_path = recorder.out / f"topo-stage2-step{int(original['step'])+completed}-repair-trained.pt"
        recorder.json("summary.json", dict(schema="neural-native-repair-integration-v1",
            status="REAL_NEURAL_REPAIR_AND200_ADDITIONAL_UPDATES_COMPLETED", provenance=provenance,
            input_datasets=input_sources, original_step=int(original["step"]),
            actual_additional_updates=completed, before_metrics=before_metrics, curves=curves,
            installed_checkpoint=installed_checkpoint.name, final_checkpoint=final_path.name,
            costs=costs, wall_seconds=time.perf_counter()-recorder.started,
            params=sum(p.numel() for p in model.parameters()), M0="NOT_PASSED",
            evidence_boundary="actual neural weights, native execution, terminal SGD and old/current semantic guard; one development instance"))
        print(json.dumps(dict(status="REAL_NEURAL_REPAIR_AND200_ADDITIONAL_UPDATES_COMPLETED",
                              out=str(args.out), updates=completed, costs=costs), sort_keys=True))
    except BaseException as error:
        if model is not None and optimizer is not None:
            try:
                save_checkpoint(model, optimizer, original, recorder, completed, provenance, "interrupted")
            except Exception as save_error:
                recorder.emit("interrupted_checkpoint_failed", error=str(save_error))
        recorder.emit("interrupted", exception=type(error).__name__, reason=str(error),
                      actually_completed_additional_updates=completed)
        recorder.json("failure.json", dict(status="FAILED_WITH_PARTIAL_RECORDS",
            exception=type(error).__name__, reason=str(error), completed_updates=completed, costs=costs))
        raise
    finally:
        recorder.stream.close()


if __name__ == "__main__":
    main()
