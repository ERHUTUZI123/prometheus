"""Observed-history capacity witnesses and protected latent workspace growth.

No teacher operations, hidden modes or intermediate states are read. Nothing
runs at import. Search caps are64 history nodes,100000 visits and2seconds.
Growth is a separate transaction; the caller decides from a strict witness.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate.
"""
import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from semantic_promotion_model import SemanticPromotionModel


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def find_state_witness(data, new_token_ids, current_dim, max_nodes=64):
    """A valid observed-history clique lower bound, NOT a hidden-state label.

    Histories are (observed start, empty prefix) and (observed start, one newly
    observed token). Two histories need different deterministic states if one
    ACTUALLY OBSERVED common remaining suffix produces different terminal
    outputs under the same public task ID. No suffix oracle is manufactured.
    A returned clique need not be maximum, including when search is censored.
    """
    started = time.perf_counter()
    deadline = started + 2.
    max_nodes = int(max_nodes)
    current_dim = int(current_dim)
    if not 1 <= max_nodes <= 64 or current_dim < 1:
        raise ValueError("require1<=max_nodes<=64 and positive current_dim")
    new_tokens = sorted(set(int(token) for token in new_token_ids))
    if any(token <= 0 for token in new_tokens):
        raise ValueError("new tokens must exclude PAD0")
    tokens, starts = _numpy(data["tokens"]), _numpy(data["starts"])
    target_key = "targets" if "targets" in data else "terminal_targets"
    targets = _numpy(data[target_key])
    if tokens.ndim != 2 or starts.shape != (len(tokens),) or targets.shape != starts.shape:
        raise ValueError("expected tokens[B,L], starts[B], terminal targets[B]")
    task_ids = _numpy(data["task_ids"]) if "task_ids" in data else None
    if task_ids is not None:
        unique_tasks = np.unique(task_ids)
        if task_ids.shape != starts.shape or len(unique_tasks) != 1:
            raise ValueError("capacity witness requires one FIXED task ID")
        fixed_task = int(unique_tasks[0])
    else:
        fixed_task = data.get("metadata", {}).get("stage")
        if fixed_task is None:
            raise ValueError("recorded fixed task ID required; no cross-task witness")
    digest = hashlib.sha256()
    for name, array in (("tokens", tokens), ("starts", starts), ("targets", targets)):
        canonical = np.ascontiguousarray(array, dtype="<i8")
        digest.update(name.encode())
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    digest.update(str(fixed_task).encode())

    words, eligible, first_rows = [], set(), {}
    censored, stop_reason = False, None
    for row in range(len(tokens)):
        if time.perf_counter() >= deadline:
            censored, stop_reason = True, "2s wall cap during record indexing"
            break
        # PAD is not an input symbol. Original arrays/targets are not modified.
        word = tuple(int(token) for token in tokens[row] if int(token) != 0)
        start = int(starts[row])
        words.append(word)
        history = start, ()
        eligible.add(history)
        first_rows.setdefault(history, row)
        if word and word[0] in new_tokens:
            history = start, (word[0],)
            eligible.add(history)
            first_rows.setdefault(history, row)
    record_audit_complete = len(words) == len(tokens)
    histories = sorted(eligible, key=lambda history: (history[0], len(history[1]), history[1]))[:max_nodes]
    history_index = {history: index for index, history in enumerate(histories)}
    observed = [dict() for _ in histories]
    conflicts = []
    indexed_rows = 0
    for row, word in enumerate(words):
        if time.perf_counter() >= deadline:
            censored, stop_reason = True, "2s wall cap during suffix collection"
            break
        start, label = int(starts[row]), int(targets[row])
        matches = [(start, ())]
        if word and word[0] in new_tokens:
            matches.append((start, (word[0],)))
        for history in matches:
            if history not in history_index:
                continue
            suffix = word[len(history[1]):]
            table = observed[history_index[history]]
            previous = table.get(suffix)
            if previous is not None and previous[0] != label:
                conflicts.append(dict(history=history_index[history], suffix=list(suffix),
                    labels=[previous[0], label], source_rows=[previous[1], row]))
            else:
                table.setdefault(suffix, (label, row))
        indexed_rows += 1
    record_audit_complete = record_audit_complete and indexed_rows == len(tokens)
    node_records = [dict(id=i, start=history[0], prefix=list(history[1]),
                         first_matching_source_row=first_rows[history],
                         observed_suffix_count=len(observed[i]))
                    for i, history in enumerate(histories)]
    report = dict(success=not conflicts, certificate_valid=not conflicts,
        fixed_task_id=fixed_task, current_dim=current_dim,
        candidate_nodes=node_records, eligible_history_count=len(eligible),
        candidate_selection_truncated=len(eligible) > max_nodes,
        dataset_sha256=digest.hexdigest(), observed_record_count=len(tokens),
        indexed_records=indexed_rows, record_audit_complete=record_audit_complete,
        conflicting_observed_labels=conflicts[:32],
        assumptions=["deterministic recurrent state with fixed task ID",
                     "same continuation and same internal state imply same terminal output",
                     "terminal labels are observed training feedback, not intermediate states"],
        state_labels_supplied=False, teacher_operations_used=False,
        maximum_clique_claimed=False, lower_bound=0,
        dimension_exceeded=False, clique_nodes=[], pair_witnesses=[])
    if conflicts:
        report.update(status="INCONSISTENT_TERMINAL_RECORDS_NO_CAPACITY_TRIGGER",
                      search_censored=censored, stop_reason=stop_reason,
                      costs=dict(seconds=time.perf_counter()-started, visits=0, pairs_examined=0))
        return report
    adjacency = [0 for _ in histories]
    witnesses = {}
    examined = 0
    for left in range(len(histories)):
        if time.perf_counter() >= deadline:
            censored, stop_reason = True, "2s wall cap during pair construction"
            break
        for right in range(left + 1, len(histories)):
            if time.perf_counter() >= deadline:
                censored, stop_reason = True, "2s wall cap during pair construction"
                break
            examined += 1
            # Prefer short witnesses; only actual common suffixes are eligible.
            shared = set(observed[left]).intersection(observed[right])
            for suffix in sorted(shared, key=lambda value: (len(value), value)):
                a, b = observed[left][suffix], observed[right][suffix]
                if a[0] != b[0]:
                    adjacency[left] |= 1 << right
                    adjacency[right] |= 1 << left
                    witnesses[left, right] = dict(left=left, right=right,
                        suffix=list(suffix), terminal_labels=[a[0], b[0]],
                        source_rows=[a[1], b[1]])
                    break
    best, visits = [], 0
    search_stopped = False

    def expand(clique, candidates):
        nonlocal best, visits, censored, stop_reason, search_stopped
        if search_stopped:
            return
        if visits >= 100000 or time.perf_counter() >= deadline:
            censored, search_stopped = True, True
            stop_reason = "100000 recursive visit cap" if visits >= 100000 else "2s wall cap during clique search"
            return
        visits += 1
        if len(clique) > len(best):
            best = list(clique)
        while candidates and not search_stopped:
            if len(clique) + candidates.bit_count() <= len(best):
                return
            available = [v for v in range(len(histories)) if candidates & (1 << v)]
            vertex = max(available, key=lambda v: ((adjacency[v] & candidates).bit_count(), -v))
            candidates &= ~(1 << vertex)
            expand(clique + [vertex], candidates & adjacency[vertex])
    if histories:
        # A single observed history is already a valid (weak) lower bound.
        best = [0]
    if not censored:
        expand([], (1 << len(histories)) - 1)
    # Even a partial graph contains only positively witnessed edges. Search
    # censored during construction can still return a cheap greedy valid clique.
    if censored and best:
        candidate = list(best)
        for vertex in range(len(histories)):
            if time.perf_counter() >= deadline:
                break
            if vertex not in candidate and all(adjacency[vertex] & (1 << old) for old in candidate):
                candidate.append(vertex)
        if len(candidate) > len(best):
            best = candidate
    best = sorted(best)
    proof = [witnesses[a, b] for i, a in enumerate(best) for b in best[i+1:]]
    report.update(status="OBSERVED_HISTORY_CLIQUE_LOWER_BOUND", lower_bound=len(best),
        dimension_exceeded=len(best) > current_dim,
        clique_nodes=[node_records[index] for index in best], pair_witnesses=proof,
        search_censored=censored, stop_reason=stop_reason,
        search_exhausted=not censored,
        costs=dict(seconds=time.perf_counter()-started, visits=visits,
                   pairs_examined=examined, witnessed_edges=len(witnesses),
                   max_nodes=max_nodes, visit_cap=100000, wall_cap_seconds=2.))
    return report


@torch.no_grad()
def grow_workspace(model, new_dim):
    """Enlarge a promoted native executor transactionally, preserving OLD words.

    This function never infers a hidden mode or supplies its transitions. New
    capacity/discovery expert are neutral random parameters. Old references
    keep their old-dimensional graphs; related() checks across the dimensions.
    Caller must rebuild Adam and record returned checkpoint dimension metadata.
    """
    started = time.perf_counter()
    new_dim = int(new_dim)
    if not isinstance(model, SemanticPromotionModel):
        raise TypeError("workspace growth supports only the promoted native model")
    old_dim, old_count = int(model.d), int(model.n_ops)
    report = dict(success=False, accepted=False, old_dim=old_dim, requested_dim=new_dim,
                  old_bank_size=old_count, state_growth=False,
                  teacher_operations_used=False, hidden_mode_labels_used=False,
                  optimizer_rebuild_required=False,
                  justification="caller must provide a strict observed-history lower bound; no stage-based trigger")
    if not old_dim < new_dim <= 64:
        report.update(status="FAILED_CLOSED", reason="require old_dim < new_dim <=64",
                      costs=dict(seconds=time.perf_counter()-started))
        return report
    if old_count + 1 > model.max_bank:
        report.update(status="FAILED_CLOSED", reason="neutral discovery expert would exceed bank cap",
                      costs=dict(seconds=time.perf_counter()-started))
        return report
    original = dict(encoder=model.encoder, decoder=model.decoder, operations=model.operations,
                    call=model.call, d=model.d, n_ops=model.n_ops, eye=model._eye)
    missing = object()
    original_optional = {name: getattr(model, name, missing) for name in ("workspace_dim", "latent_factor")}
    original_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    original_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    device, dtype = model.encoder.device, model.encoder.dtype
    rng_cpu = torch.random.get_rng_state()
    rng_device = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    costs = dict(native_graphs=0, graph_hungarian_assignments=0,
                 relation_checks=0, relation_pairs=0, parameter_bytes_before=original_bytes)
    try:
        domains = {}
        for reference in model.reference:
            domains.setdefault(int(reference["task"]), set()).update(reference["tokens"])
        for promotion in model.materialization_history:
            if promotion.get("success"):
                for item in promotion.get("learned_call_targets", []):
                    domains.setdefault(int(item["task"]), set()).add(int(item["token"]))
        if not domains:
            raise RuntimeError("no accepted promoted native domains exist to certify")
        old_graphs = {}
        for task, alphabet in sorted(domains.items()):
            old_graphs[task] = model.graph(task, sorted(alphabet))
            costs["native_graphs"] += 1
            costs["graph_hungarian_assignments"] += old_count
        for reference in model.reference:
            okay, pairs = model.related(reference, old_graphs[int(reference["task"])])
            costs["relation_checks"] += 1
            costs["relation_pairs"] += pairs
            if not okay:
                raise RuntimeError("pre-growth native graph already violates its old reference")
        extra = new_dim - old_dim
        encoder = model.encoder.new_empty((model.n, new_dim))
        encoder[:, :old_dim] = model.encoder
        low = torch.minimum(model.encoder.amin(-1, keepdim=True), model.encoder.new_full((model.n, 1), -20.))
        encoder[:, old_dim:] = low - 20.
        decoder = model.decoder.new_empty((new_dim, model.n))
        decoder[:old_dim] = model.decoder
        copy_rows = torch.arange(extra, device=device) % old_dim
        decoder[old_dim:] = model.decoder[copy_rows]
        operations = model.operations.new_full((old_count + 1, new_dim, new_dim), -20.)
        operations[:old_count, :old_dim, :old_dim] = model.operations
        operations[:old_count, old_dim:, old_dim:] = .03 * torch.randn(
            (old_count, extra, extra), device=device, dtype=dtype)
        # Exactly one extra full-dimensional discovery expert; no correct map.
        operations[old_count] = .03 * torch.randn((new_dim, new_dim), device=device, dtype=dtype)
        old_width, new_width = old_count + 1, old_count + 2
        weight = model.call.weight.new_zeros((model.slots, new_width, model.call.in_features))
        bias = model.call.bias.new_zeros((model.slots, new_width))
        weight[:, :old_width] = model.call.weight.reshape(model.slots, old_width, -1)
        bias[:, :old_width] = model.call.bias.reshape(model.slots, old_width)
        call = copy.deepcopy(model.call)
        call.out_features = model.slots * new_width
        call.weight = nn.Parameter(weight.reshape(-1, model.call.in_features))
        call.bias = nn.Parameter(bias.reshape(-1))
        model.encoder, model.decoder = nn.Parameter(encoder), nn.Parameter(decoder)
        model.operations, model.call = nn.Parameter(operations), call
        model.d, model.n_ops, model._eye = new_dim, old_count + 1, None
        certificates = []
        for task, before in old_graphs.items():
            after = model.graph(task, before["tokens"])
            costs["native_graphs"] += 1
            costs["graph_hungarian_assignments"] += model.n_ops
            okay, pairs = model.related(before, after)
            costs["relation_checks"] += 1
            costs["relation_pairs"] += pairs
            routes_preserved = np.array_equal(before["routes"], after["routes"])
            certificates.append(dict(task=task, tokens=list(before["tokens"]),
                before_dim=old_dim, after_dim=new_dim, related=bool(okay),
                native_routes_preserved=bool(routes_preserved), relation_pairs=pairs,
                all_old_words=True))
            if not okay or not routes_preserved:
                raise RuntimeError("workspace extension changed an old native graph or hard route")
        for reference in model.reference:
            after = model.graph(reference["task"], reference["tokens"])
            costs["native_graphs"] += 1
            costs["graph_hungarian_assignments"] += model.n_ops
            okay, pairs = model.related(reference, after)
            costs["relation_checks"] += 1
            costs["relation_pairs"] += pairs
            if not okay:
                raise RuntimeError("workspace extension violated a protected old reference")
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.workspace_dim = new_dim
        model.latent_factor = new_dim / model.n
        costs.update(parameter_bytes_after=sum(p.numel() * p.element_size() for p in model.parameters()),
                     appended_discovery_experts=1,
                     new_free_workspace_rows=extra,
                     random_initialized_scalars=old_count * extra * extra + new_dim * new_dim,
                     rng_snapshot_bytes=rng_cpu.numel() + (0 if rng_device is None else rng_device.numel()))
        costs["parameter_byte_increase"] = costs["parameter_bytes_after"] - original_bytes
        report.update(success=True, accepted=True, state_growth=True,
            status="WORKSPACE_GROWN_NATIVE_OLD_RELATIONS_VERIFIED", new_dim=new_dim,
            new_bank_size=model.n_ops, discovery_expert=model.n_ops,
            certificates=certificates, optimizer_rebuild_required=True,
            all_parameters_trainable=True, old_reference_graphs_left_at_original_dim=True,
            old_logit_top_left_unchanged=True, offblock_logit=-20., new_expert_call_logit=0.,
            decoder_new_rows="cycle copies of OLD LEARNED decoder rows; no hidden labels",
            checkpoint_metadata=dict(d=new_dim, n=model.n, n_ops=model.n_ops,
                                     latent_factor=new_dim/model.n),
            cached_flags_invalidated=["_eye"], rng_advanced_for_new_neutral_parameters=True)
    except BaseException as error:
        model.encoder, model.decoder = original["encoder"], original["decoder"]
        model.operations, model.call = original["operations"], original["call"]
        model.d, model.n_ops, model._eye = original["d"], original["n_ops"], original["eye"]
        for name, value in original_optional.items():
            if value is missing:
                if hasattr(model, name):
                    delattr(model, name)
            else:
                setattr(model, name, value)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags[name])
        torch.random.set_rng_state(rng_cpu)
        if rng_device is not None:
            torch.cuda.set_rng_state(rng_device, device)
        if not isinstance(error, Exception):
            raise
        report.update(status="FAILED_CLOSED_ROLLED_BACK", reason=str(error),
                      error_type=type(error).__name__, original_parameter_objects_restored=True,
                      rng_restored=True)
    costs["seconds"] = time.perf_counter() - started
    report["costs"] = costs
    return report
