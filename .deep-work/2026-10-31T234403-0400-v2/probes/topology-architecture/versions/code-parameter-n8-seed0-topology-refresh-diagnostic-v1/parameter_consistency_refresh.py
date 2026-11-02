"""Teacher-free, parameter-count-preserving neural/native consistency refresh.

Targets are ONLY this compact model's own CURRENT hard encoder, decoder, expert
permutations and both learned call slots. No labels, evaluation data, teacher,
new expert, whole-operation materialization or extra learned parameter is used.
The same maintenance rule is available to ordinary constrained/frozen/modular
compact models; it is not a topology-exclusive benefit.

API: refresh_native_consistency(model,max_length=16)->report. A successful call
replaces real Parameters, so a subsequent trainer MUST rebuild its optimizer.
On failure the original Parameter objects/flags are restored atomically. RNG is
audited, never rewound to conceal consumption. No import-time execution.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 unpassed; M3 ultimate.
"""
import hashlib
import math
import random
import time

import numpy as np
import torch
from torch import nn


def _sync(model):
    if model.encoder.device.type == "cuda":
        torch.cuda.synchronize(model.encoder.device)


def _rng_fingerprint():
    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_initialized():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    state = np.random.get_state()
    digest.update(str((state[0], state[2:])).encode())
    digest.update(state[1].tobytes())
    digest.update(repr(random.getstate()).encode())
    return digest.hexdigest()


def _same_graph(first, second):
    return (first["task"] == second["task"]
        and tuple(first["tokens"]) == tuple(second["tokens"])
        and np.array_equal(first["enc"], second["enc"])
        and np.array_equal(first["dec"], second["dec"])
        and np.array_equal(first["routes"], second["routes"])
        and len(first["permutations"]) == len(second["permutations"])
        and all(np.array_equal(p, q) for p, q in zip(first["permutations"], second["permutations"]))
        and all(np.array_equal(first["trans"][a], second["trans"][a]) for a in first["tokens"]))


@torch.no_grad()
def _path_mass(model, pairs, routes, encoder, decoder, permutations, max_length):
    """Real-probability path lemma, numerically evaluated on actual soft_ops."""
    device, tau = model.encoder.device, float(model.temperature)
    tokens = torch.as_tensor([[a] for _, a in pairs], device=device, dtype=torch.long)
    tasks = torch.as_tensor([t for t, _ in pairs], device=device, dtype=torch.long)
    call_probabilities = torch.softmax(model.calls(tokens, tasks)[:, 0]/tau, -1)
    target_calls = torch.as_tensor(routes, device=device, dtype=torch.long)
    route_min = float(call_probabilities.gather(-1, target_calls[..., None]).amin().item())
    enc = torch.softmax(model.encoder/tau, -1)
    dec = torch.softmax(model.decoder/tau, -1)
    encoder_min = float(enc[torch.arange(model.n, device=device),
                            torch.as_tensor(encoder, device=device)].amin().item())
    decoder_min = float(dec[torch.arange(model.d, device=device),
                            torch.as_tensor(decoder, device=device)].amin().item())
    operations = model.soft_ops()
    rows = torch.arange(model.d, device=device)
    operation_min = min(float(operations[k, rows, torch.as_tensor(p, device=device)].amin().item())
                        for k, p in enumerate(permutations))
    minima = (encoder_min, decoder_min, route_min, operation_min)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in minima):
        raise FloatingPointError("actual soft probabilities outside finite unit interval")
    exponent = model.slots*max_length
    mass = (encoder_min*decoder_min*(route_min*operation_min)**exponent)
    return dict(min_p_encoder=encoder_min, min_p_decoder=decoder_min,
        min_p_route_correct_slot=route_min, min_p_operation_correct_edge=operation_min,
        soft_ops_source="actual model.soft_ops(), all experts/all latent rows, including identity",
        slots=int(model.slots), max_length=max_length, operation_count_upper=exponent,
        own_hard_path_mass_lower_bound=mass,
        own_native_output_has_probability_above_half=mass > .5,
        lemma="p(own native output)>=minE*minR*(minRoute*minOp)^(slots*Lmax)",
        scope="analytical nonnegative path-mass lemma evaluated in floating arithmetic; not interval-certified",
        temperature=tau, later_parameter_or_temperature_changes_require_rechecking=True)


@torch.no_grad()
def refresh_native_consistency(model, max_length=16):
    started = time.perf_counter()
    original = dict(model.named_parameters())
    original_shapes = {name: tuple(p.shape) for name, p in original.items()}
    original_flags = {name: bool(p.requires_grad) for name, p in original.items()}
    original_count = sum(p.numel() for p in original.values())
    original_KD = (int(model.n_ops), int(model.d))
    original_call = model.call
    rng_before = _rng_fingerprint()
    costs = dict(graph_calls=0, graph_hungarian_assignments=0,
        relation_checks=0, relation_pairs=0, svd_calls=0, least_squares_calls=0,
        label_reads=0, external_queries=0, gradient_updates=0)
    report = dict(success=False, accepted=False, status="NOT_COMMITTED",
        target_source="OWN current hard encoder/decoder/expert classes/both learned slot routes",
        teacher_or_evaluation_labels_used=False, topology_exclusive=False,
        wholemap_materialization_used=False, new_experts=0,
        parameter_count_before=original_count, costs=costs,
        optimizer_rebuild_required=False)
    committed = False

    def graph(task, tokens):
        result = model.graph(task, tokens)
        costs["graph_calls"] += 1
        costs["graph_hungarian_assignments"] += model.n_ops
        return result

    def invalidate():
        method = getattr(model, "_invalidate_derived_caches", None)
        if callable(method):
            method()

    def replace(name, value):
        parent, leaf = (model.call, name.split(".", 1)[1]) if name.startswith("call.") else (model, name)
        setattr(parent, leaf, value)

    try:
        if (not isinstance(max_length, int) or not 1 <= max_length <= 16
                or "operation_coordinates" not in original
                or not 1 <= model.n_ops <= int(model.max_bank) <= 48
                or not 1 <= model.d <= 64):
            raise ValueError("compact native model within existing K48,D64,L16 bounds required")
        tau = float(model.temperature)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("finite positive execution temperature required")
        domains = {}
        for reference in model.reference:
            domains.setdefault(int(reference["task"]), set()).update(int(a) for a in reference["tokens"])
        if not domains:
            raise ValueError("no learned protected task domains to refresh")
        domains = {task: tuple(sorted(tokens)) for task, tokens in sorted(domains.items())}
        before = {task: graph(task, tokens) for task, tokens in domains.items()}
        first = next(iter(before.values()))
        encoder, decoder = first["enc"].copy(), first["dec"].copy()
        permutations = [p.copy() for p in first["permutations"]]
        if len(permutations) != model.n_ops+1:
            raise ValueError("own bank count disagrees with native graphs")
        for source in before.values():
            if (not np.array_equal(source["enc"], encoder)
                    or not np.array_equal(source["dec"], decoder)
                    or any(not np.array_equal(p, q) for p, q in zip(source["permutations"], permutations))):
                raise ValueError("learned task graphs do not share the stated encoder/decoder/bank")
        for reference in model.reference:
            okay, pairs_checked = model.related(reference, before[int(reference["task"])])
            costs["relation_checks"] += 1
            costs["relation_pairs"] += pairs_checked
            if not okay:
                raise ValueError("source already violates an old learned reference")
        pairs, target_routes = [], []
        for task, source in before.items():
            for token, route in zip(source["tokens"], source["routes"]):
                pairs.append((task, int(token)))
                target_routes.append([int(k) for k in route])
        target_routes = np.asarray(target_routes, dtype=np.int64)
        report.update(learned_domains={t: list(a) for t, a in domains.items()},
            own_target_encoder=encoder.tolist(), own_target_decoder=decoder.tolist(),
            own_target_expert_permutations=[p.tolist() for p in permutations[1:]],
            own_target_calls=[dict(task=t, token=a, slots=target_routes[i].tolist())
                              for i, (t, a) in enumerate(pairs)])
        report["before_path_mass"] = _path_mass(model, pairs, target_routes, encoder, decoder,
                                                  permutations, max_length)
        delta = .25/(2+2*model.slots*max_length)
        competitor_cap = 48
        tau_cap = max(1., tau)
        base_h = .5*tau_cap*math.log(competitor_cap*(1-delta)/delta)
        h = base_h+.5*tau_cap  # one-temperature margin slack for device rounding
        required_margin = tau*math.log(competitor_cap*(1-delta)/delta)
        device, dtype = model.call.weight.device, model.call.weight.dtype
        tokens = torch.as_tensor([[a] for _, a in pairs], device=device, dtype=torch.long)
        tasks = torch.as_tensor([t for t, _ in pairs], device=device, dtype=torch.long)
        features = torch.tanh(model.token(tokens[:, 0])+model.task(tasks)).cpu().double().numpy()
        features = np.concatenate([features, np.ones((len(pairs), 1))], axis=1)
        outputs = model.slots*(model.n_ops+1)
        desired = np.full((len(pairs), outputs), -h)
        for i in range(len(pairs)):
            for slot in range(model.slots):
                desired[i, slot*(model.n_ops+1)+target_routes[i, slot]] = h
        singular = np.linalg.svd(features, compute_uv=False)
        costs["svd_calls"] += 1
        tolerance = max(features.shape)*np.finfo(np.float64).eps*singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf
        report["fit"] = dict(rows=len(pairs), features=features.shape[1], rank=rank,
            condition=condition if math.isfinite(condition) else None, condition_limit=1e8,
            delta=delta, max_bank_competitors=competitor_cap, base_h=base_h,
            h=h, rounding_slack=.5*tau_cap, required_actual_raw_margin=required_margin,
            temperature_cap_used=tau_cap)
        if rank < len(pairs) or not math.isfinite(condition) or condition > 1e8:
            raise ValueError("current own-call targets lack a well-conditioned full-row-rank feature fit")
        solution, _, solved_rank, _ = np.linalg.lstsq(features, desired, rcond=None)
        costs["least_squares_calls"] += 1
        residual = float(np.max(np.abs(features@solution-desired)))
        report["fit"].update(solved_rank=int(solved_rank), CPU_double_max_residual=residual)
        costs.update(CPU_double_feature_bytes=features.nbytes, CPU_double_target_bytes=desired.nbytes,
                     CPU_double_solution_bytes=solution.nbytes)
        if solved_rank < len(pairs) or not math.isfinite(residual) or residual > 1e-5:
            raise ValueError("own-call CPU-double least-squares residual exceeds1e-5")
        coordinates = model.operation_coordinates.new_tensor(np.stack(permutations[1:]))
        new_encoder = torch.full_like(model.encoder, -8.)
        new_encoder[torch.arange(model.n, device=new_encoder.device),
                    torch.as_tensor(encoder, device=new_encoder.device)] = 8.
        new_decoder = torch.full_like(model.decoder, -8.)
        new_decoder[torch.arange(model.d, device=new_decoder.device),
                    torch.as_tensor(decoder, device=new_decoder.device)] = 8.
        values = {"operation_coordinates": coordinates, "encoder": new_encoder, "decoder": new_decoder,
            "call.weight": torch.as_tensor(solution[:-1].T, device=device, dtype=dtype).clone(),
            "call.bias": torch.as_tensor(solution[-1], device=device, dtype=dtype).clone()}
        proposed = {name: nn.Parameter(value, requires_grad=original_flags[name])
                    for name, value in values.items()}
        for name, parameter in proposed.items():
            if tuple(parameter.shape) != original_shapes[name]:
                raise RuntimeError("consistency refresh attempted to change a parameter shape")
        for name, parameter in proposed.items():
            replace(name, parameter)
        committed = True
        invalidate()
        logits = model.calls(tokens, tasks)[:, 0]
        expected = torch.as_tensor(target_routes, device=device)
        winning = logits.gather(-1, expected[..., None]).squeeze(-1)
        competing = logits.clone().scatter_(-1, expected[..., None], -torch.inf).amax(-1)
        actual_margin = float((winning-competing).amin().item())
        report["fit"].update(actual_device_minimum_raw_margin=actual_margin,
            actual_device_max_target_residual=float((logits.reshape(len(pairs), -1)
                -logits.new_tensor(desired)).abs().amax().item()))
        if (not torch.isfinite(logits).all() or not torch.equal(logits.argmax(-1), expected)
                or not math.isfinite(actual_margin) or actual_margin < required_margin):
            raise RuntimeError("actual-device own-route/margin check failed")
        after = {task: graph(task, alphabet) for task, alphabet in domains.items()}
        certificates = []
        for task in domains:
            exact = _same_graph(before[task], after[task])
            certificates.append(dict(task=task, all_routes_and_full_latent_transitions_exact=bool(exact),
                                     arbitrary_word_native_behavior_unchanged=bool(exact)))
            if not exact:
                raise RuntimeError("consistency refresh changed an own native graph or route")
        for reference in model.reference:
            okay, count = model.related(reference, after[int(reference["task"])])
            costs["relation_checks"] += 1
            costs["relation_pairs"] += count
            if not okay:
                raise RuntimeError("refreshed model violates an old learned reference")
        current = dict(model.named_parameters())
        if (set(current) != set(original) or any(tuple(current[n].shape) != original_shapes[n] for n in original)
                or sum(p.numel() for p in current.values()) != original_count
                or (model.n_ops, model.d) != original_KD or model.call is not original_call
                or any(bool(current[n].requires_grad) != original_flags[n] for n in original)):
            raise RuntimeError("parameter shape/count/K/D/flags/module invariance check failed")
        report["after_path_mass"] = _path_mass(model, pairs, target_routes, encoder, decoder,
                                                 permutations, max_length)
        if not report["after_path_mass"]["own_native_output_has_probability_above_half"]:
            raise RuntimeError("measured own-hard path-mass sufficient bound does not exceed1/2")
        if _rng_fingerprint() != rng_before:
            raise RuntimeError("refresh unexpectedly consumed RNG; state is reported, not rewound")
        report.update(success=True, accepted=True, status="OWN_NATIVE_SOFT_CONSISTENCY_REFRESHED",
            parameter_count_after=original_count, delta_parameters=0,
            K_before_after=[original_KD[0], model.n_ops], D_before_after=[original_KD[1], model.d],
            replaced_parameter_names=sorted(proposed), all_parameter_shapes_and_flags_unchanged=True,
            native_certificates=certificates, optimizer_rebuild_required=True,
            proof_interpretation=f"under the actual stochastic computation, own native output mass>.5 gives its unique argmax for all learned-domain words through L{max_length}; numerical evaluation is not interval arithmetic")
    except BaseException as error:
        # Restore REAL old Parameter objects, including coordinate values and
        # flags; do not regenerate a dense score proxy or a new optimizer state.
        for name in ("operation_coordinates", "encoder", "decoder", "call.weight", "call.bias"):
            if name in original:
                replace(name, original[name])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_flags[name])
        invalidate()
        restored = dict(model.named_parameters())
        exact_objects = all(restored.get(name) is parameter for name, parameter in original.items())
        report.update(status="FAILED_CLOSED_ORIGINAL_PARAMETERS_RESTORED", reason=str(error),
            error_type=type(error).__name__, original_parameter_objects_restored=exact_objects,
            mutation_had_started=committed, parameter_count_after=sum(p.numel() for p in restored.values()),
            optimizer_rebuild_required=False)
        if not isinstance(error, Exception):
            raise
    finally:
        _sync(model)
        report["rng_state_unchanged"] = _rng_fingerprint() == rng_before
        report["costs"]["synchronized_seconds"] = time.perf_counter()-started
        report["costs"]["persistent_parameter_byte_change"] = (
            sum(p.numel()*p.element_size() for p in model.parameters())
            -sum(p.numel()*p.element_size() for p in original.values()))
    return report
