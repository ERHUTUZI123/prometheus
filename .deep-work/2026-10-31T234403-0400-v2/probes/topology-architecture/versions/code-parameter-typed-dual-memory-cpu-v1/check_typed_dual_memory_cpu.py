"""One bounded synthetic CPU admission for the fixed dual-memory rule.

Restore exactly OWN AB theta, reset new C/V episode state, run no optimizer.
Tests equations/ridge algebra/gradients/reload, never task capability.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
import resource
import sys
import textwrap
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_ab_core as previous
import typed_dual_memory_masked_fb as fb


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    return left.keys() == right.keys() and all(torch.equal(value, right[key]) for key, value in left.items())


def finite(value):
    return bool(torch.isfinite(value).all())


def close(left, right, message, atol=2e-5, rtol=2e-5):
    if not finite(left) or not finite(right) or not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise AssertionError(message)


def timed(function):
    wall, cpu = time.perf_counter(), time.process_time()
    result = function()
    return result, dict(wall_seconds=time.perf_counter()-wall, CPU_seconds=time.process_time()-cpu)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(bytes(value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def save(path, value):
    with path.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def rejects(function, label):
    try:
        function()
    except (ValueError, TypeError, RuntimeError, KeyError):
        return
    raise AssertionError('invalid checkpoint/configuration accepted: '+label)


def source_check():
    def assignments(method):
        result = {}
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(method)))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        result[target.id] = ast.dump(node.value, include_attributes=False)
        return result
    old, new = assignments(previous.ContinuousPrimitiveBlock._transition), assignments(core.ContinuousPrimitiveBlock._transition)
    names = ('c', 'x', 'attention', 'features', 'all_logits', 'action_logits', 'probabilities', 'qcomp', 'chosen', 'chosen_compute', 'hard_weights', 'gates', 'proposal', 'compute_total', 'h', 'amount', 'v', 'address')
    for name in names:
        if old.get(name) != new.get(name):
            raise AssertionError('unchanged AB transition assignment differs: '+name)
    return dict(unchanged_AB_assignments=list(names), candidate_changes='read rule and new C recurrence only')


def transition_check(model):
    h = torch.linspace(-.4, .8, 128).reshape(4, 32)
    carry = model.initial_carry(h)
    raw = torch.linspace(-.2, .5, 64).reshape(4, 16)
    if set(carry) != {'h', 'controller', 'values', 'address', 'logp', 'key_composition'} or bool(carry['key_composition'].count_nonzero()) or bool(carry['values'].count_nonzero()):
        raise AssertionError('new C/V episode must start at zero, no old coverage')
    carry['key_composition'] = torch.linspace(.002, .03, 256).reshape(4, 8, 8)
    carry['values'] = torch.linspace(-.01, .02, 1024).reshape(4, 8, 32)
    carry['address'] = torch.linspace(-1., 1., 32).reshape(4, 8)
    with torch.no_grad():
        native = None
        for mode in ('soft', 'native', 'straight_through'):
            updated, trace = model._transition(carry, raw, mode)
            key = F.softmax(carry['address'], -1)
            weights = trace['action_weights']
            amount = weights[:, 5:6]*key
            expected_C = carry['key_composition']+amount[:, :, None]*(key[:, None]-carry['key_composition'])
            expected_V = carry['values']+amount[:, :, None]*(h[:, None]-carry['values'])
            close(updated['key_composition'], expected_C, 'actual C write differs')
            close(updated['values'], expected_V, 'OLDh value write changed')
            old_z = carry['key_composition'].sum(-1)
            expected_z = old_z+amount*(1-old_z)
            close(updated['key_composition'].sum(-1), expected_z, 'C1 coverage recurrence differs')
            close(trace['write_value'], h, 'STORE no longer writes OLDh')
            close(trace['read_key_composition'], carry['key_composition'], 'read uses updated C')
            close(trace['address_proposal'], F.linear(raw, model.address_proposal.weight, model.address_proposal.bias), 'raw-x proposal changed')
            if mode == 'native':
                native = updated
            if mode == 'straight_through':
                for name in native:
                    close(updated[name], native[name], 'ST forward differs from native: '+name)
    bad = dict(carry)
    bad['coverage'] = bad.pop('key_composition').sum(-1)
    rejects(lambda: model._transition(bad, raw, 'soft'), 'legacy coverage carry')
    return dict(C_V_zero_episode_state=True, C_recurrence_and_C1_coverage=True,
                coverage_scope='real algebra and fixed numeric tolerance, not FP32 bitwise equality',
                unchanged_OLDh_V_write=True, OLD_C_read=True, raw_x_address_proposal=True,
                soft_native_ST_new_C_rule_checked=True, legacy_coverage_carry_rejected=True)


def history(keys, hidden, strengths):
    C = keys.new_zeros(1, 8, 8)
    V = hidden.new_zeros(1, 8, 32)
    coverage = keys.new_zeros(1, 8)
    for key, value, strength in zip(keys, hidden, strengths):
        w = strength*key[None]
        C = C+w[:, :, None]*(key[None, None]-C)
        V = V+w[:, :, None]*(value[None, None]-V)
        coverage = coverage+w*(1-coverage)
    return C, V, coverage


def ridge_fixture(model, out):
    dtype = torch.float64
    key0 = torch.tensor([.36, .24, .16, .10, .06, .04, .025, .015], dtype=dtype)
    keys = torch.stack((key0, key0.flip(0)))
    B = torch.sin(torch.arange(1, 257, dtype=dtype).reshape(8, 32)*.17)*.3
    hidden = keys@B
    C, V, old_z = history(keys, hidden, (.8, .7))
    close(V, C@B, 'linear association V=CB invariant failed', 1e-12, 1e-12)
    close(C.sum(-1), old_z, 'synthetic C1 coverage invariant failed', 1e-12, 1e-12)
    V, C = V.expand(2, -1, -1).clone(), C.expand(2, -1, -1).clone()
    read, dual, z, A, Vbar = model.dual_read(V, C, keys)
    # Ordinary ridge least-squares via SVD is an independent algorithmic
    # reference to the executed Gram solve, not a novelty comparator.
    U, singular, Vh = torch.linalg.svd(A, full_matrices=False)
    ridge = model.dual_ridge
    projected = torch.bmm(Vh, keys[:, :, None]).squeeze(-1)
    standard_dual = torch.bmm(U, (singular/(singular.square()+ridge)*projected)[:, :, None]).squeeze(-1)
    standard_read = torch.bmm(standard_dual[:, None], Vbar).squeeze(1)
    close(dual, standard_dual, 'ordinary SVD ridge reference differs', 2e-7, 2e-7)
    close(read, standard_read, 'ordinary ridge reference read differs', 2e-7, 2e-7)
    T = torch.bmm(Vh.transpose(1, 2), (singular.square()/(singular.square()+ridge))[:, :, None]*Vh)
    predicted = torch.bmm(keys[:, None], torch.bmm(T, B[None].expand(2, -1, -1))).squeeze(1)
    close(read, predicted, 'r=kT_lambda B consistency/bias formula differs', 2e-7, 2e-7)
    gram = torch.bmm(A, A.transpose(1, 2))+ridge*torch.eye(8, dtype=dtype)[None]
    rhs = torch.bmm(A, keys[:, :, None]).squeeze(-1)
    residual = torch.bmm(gram, dual[:, :, None]).squeeze(-1)-rhs
    eigenvalues = torch.linalg.eigvalsh(gram)
    condition = eigenvalues[:, -1]/eigenvalues[:, 0]
    norm_bound = keys.norm(dim=-1)/(2*ridge**.5)
    if not bool((dual.norm(dim=-1) <= norm_bound*(1+2e-6)).all()) or not bool((condition <= 80001*(1+2e-6)).all()):
        raise AssertionError('ridge norm/condition bound failed')
    if int((singular[0] > 1e-10).sum()) != 2 or not bool((keys[0]*keys[1] > 0).all()):
        raise AssertionError('fixture must contain overlapping independent keys and rank2 mixtures')
    if not finite(residual) or float(residual.norm(dim=-1).max()) > 2e-7:
        raise AssertionError('ridge solve equation residual too large')
    # Same key, inconsistent values: this read has no key distinction with
    # which to choose either past write. Uniform key gives a closed-form EMA.
    same = torch.full((2, 8), 1/8, dtype=dtype)
    incompatible = torch.eye(32, dtype=dtype)[:2]
    same_C, same_V, same_z = history(same, incompatible, (.6, .6))
    same_read, same_b, _, _, _ = model.dual_read(same_V.expand(2, -1, -1), same_C.expand(2, -1, -1), same)
    amount = .6/8
    average = (amount*(1-amount)*incompatible[0]+amount*incompatible[1])/(amount*(1-amount)+amount)
    expected_same = average/(1+ridge)
    close(same_read, expected_same[None].expand(2, -1), 'identical-key EMA/ridge counterexample differs', 2e-7, 2e-7)
    if not torch.equal(same_read[0], same_read[1]):
        raise AssertionError('identical query must not distinguish incompatible writes')
    empty = model.dual_read(torch.zeros_like(V), torch.zeros_like(C), keys)
    if bool(empty[0].count_nonzero()) or bool(empty[1].count_nonzero()):
        raise AssertionError('empty C/V must read exactly zero')
    receipt = save(out/'dual-algebra-fixtures.pt', dict(keys=keys, B=B, C=C, V=V, read=read, b=dual,
                    singular_values=singular, standard_ridge_b=standard_dual, predicted=predicted,
                    solve_residual=residual, identical_key_read=same_read, identical_key_expected=expected_same))
    return dict(overlapping_independent_two_key_fixture=True, linear_association_invariant=True,
                standard_ridge_least_squares_SVD_reference=True,
                ridge_bias_and_rowspan_limit_formula_checked=True,
                target_read_error_norm=(read-hidden).norm(dim=-1).tolist(),
                predicted_ridge_bias_norm=(predicted-hidden).norm(dim=-1).tolist(),
                key_residual_norm=(torch.bmm(A.transpose(1, 2), dual[:, :, None]).squeeze(-1)-keys).norm(dim=-1).tolist(),
                solve_residual_norm=residual.norm(dim=-1).tolist(), condition=condition.tolist(),
                b_norm=dual.norm(dim=-1).tolist(), b_norm_upper_bound=norm_bound.tolist(),
                min_b=float(dual.min()), identical_key_counterexample=True,
                identical_key_scope='same query cannot choose two incompatible values in fixed memory; not an NN impossibility claim',
                empty_read_exact_zero=True, raw=receipt)


def solve_gradient_check(model, out):
    diagnostics, tensors = {}, {}
    for name, amount in (('empty', 0.), ('tiny', 1e-10), ('rank_one', .01)):
        C = torch.full((2, 8, 8), amount, dtype=torch.float64, requires_grad=True)
        V = torch.linspace(-1e-3, 1e-3, 512, dtype=torch.float64).reshape(2, 8, 32)
        if name == 'empty':
            V.zero_()
        V.requires_grad_()
        q = torch.linspace(.1, .3, 16, dtype=torch.float64).reshape(2, 8)
        q = (q/q.sum(-1, keepdim=True)).requires_grad_()
        read, dual, z, _, _ = model.dual_read(V, C, q)
        loss = read.square().mean()+.01*dual.square().mean()+.001*z.square().mean()
        grads = torch.autograd.grad(loss, (V, C, q), allow_unused=True)
        if any(grad is None or not finite(grad) for grad in grads):
            raise AssertionError('read-solve input gradient missing/nonfinite: '+name)
        diagnostics[name] = dict(finite_read=finite(read), finite_dual=finite(dual),
                                 gradients={key: dict(finite=True, norm=float(grad.norm())) for key, grad in zip(('V', 'C', 'query'), grads)})
        tensors[name] = dict(V=V.detach(), C=C.detach(), q=q.detach(), read=read.detach(), b=dual.detach(), gradients=[grad.detach() for grad in grads])
    return dict(cases=diagnostics, all_read_solve_input_gradients_finite=True,
                scope='finite derivative diagnostics at empty/tiny/rank-one states; no full Jacobian or learning guarantee',
                raw=save(out/'dual-read-gradient-fixtures.pt', tensors))


def nn_gradient_check(model, out):
    row = torch.arange(64)
    data = dict(initial=row % 8, tokens=(row[:, None]*3+torch.arange(16)[None]*5) % 16,
                lengths=row % 17, targets=(row+3) % 8)
    initial_hash = state_hash(model)
    model.zero_grad(set_to_none=True)
    def eager():
        loss, output = fb.objective(model, fb.original_forward_value, **data)
        loss.backward()
        return output.detach().clone(), loss.detach().clone(), {name: None if parameter.grad is None else parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    (left, left_loss, left_grad), eager_cost = timed(eager)
    fb.allocate_gradient_buffers(model)
    pointers = {name: parameter.grad.data_ptr() for name, parameter in model.named_parameters()}
    (right, right_loss), masked_cost = timed(lambda: fb.masked_step(model, **data))
    right_grad = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
    raw = save(out/'dual-NN-B64-L16-all20-gradients.pt', dict(eager_output=left, masked_output=right,
               eager_loss=left_loss, masked_loss=right_loss, eager_gradients=left_grad, masked_gradients=right_grad,
               synthetic_batch=data))
    close(left, right, 'full NN eager/masked emission differs', fb.OUTPUT_ATOL, fb.OUTPUT_RTOL)
    close(left_loss, right_loss, 'full NN eager/masked NLL differs', fb.OUTPUT_ATOL, fb.OUTPUT_RTOL)
    if len(left_grad) != 20 or left_grad.keys() != right_grad.keys():
        raise AssertionError('exact20 parameter-gradient names required')
    diagnostics = {}
    for name, left_value in left_grad.items():
        right_value = right_grad[name]
        if left_value is None or not finite(left_value) or not finite(right_value):
            raise AssertionError('full-sequence all20 gradient missing/nonfinite: '+name)
        close(left_value, right_value, 'all20 numeric gradient mismatch: '+name, fb.GRAD_ATOL, fb.GRAD_RTOL)
        diagnostics[name] = dict(finite=True, allclose=True, max_abs=float((left_value-right_value).abs().max()))
    (_, _), repeat_cost = timed(lambda: fb.masked_step(model, **data))
    for name, parameter in model.named_parameters():
        close(parameter.grad, left_grad[name], 'masked repeat accumulated gradients: '+name, fb.GRAD_ATOL, fb.GRAD_RTOL)
    if pointers != {name: parameter.grad.data_ptr() for name, parameter in model.named_parameters()}:
        raise AssertionError('gradient buffer storage changed')
    with torch.no_grad():
        eager_out, eager_carry = model.forward_value(data['initial'], data['tokens'], data['lengths'])
        masked_out, masked_carry = fb.masked_forward_value(model, data['initial'], data['tokens'], data['lengths'])
        for name in eager_carry:
            close(eager_carry[name], masked_carry[name], 'masked fullcarry differs: '+name)
        first = model.initial_carry(data['initial'])
        zero_out, zero_carry = fb.masked_forward_value(model, data['initial'], data['tokens'], torch.zeros_like(data['lengths']))
        if not equal(first, zero_carry) or not torch.equal(zero_out, first['logp']):
            raise AssertionError('inactive fullcarry including C changed')
    if state_hash(model) != initial_hash:
        raise AssertionError('zero-optimizer F/B changed theta')
    return dict(all20_gradients_present_finite=True, all20_eager_masked_gradients_allclose=True,
                gradients=diagnostics, cached_output_max_abs=float((left-right).abs().max()),
                NLL_abs=float((left_loss-right_loss).abs()), all_carry_fields_masked=True,
                inactive_C_and_fullcarry_exact=True, gradient_buffers_stable_and_zeroed=True,
                theta_unchanged=True, eager_FB=eager_cost, masked_FB=masked_cost, repeated_masked_FB=repeat_cost,
                raw=raw)


def reload_check(model, payload, manifest, out):
    probe = dict(initial=torch.tensor([0, 3]), tokens=torch.tensor([[1, 3, 7], [2, 3, 8]]), lengths=torch.tensor([3, 2]))
    with torch.no_grad():
        output, carry = model.forward_value(**probe)
    saved = dict(schema='typed-dual-memory-CPU-no-optimizer-v1', block_checkpoint=model.checkpoint(),
                 episode_carry={key: value.detach().clone() for key, value in carry.items()},
                 optimizer_instantiated=False, SGD_updates=0)
    path = out/'dual-CPU-no-optimizer-checkpoint.pt'
    receipt = save(path, saved)
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    other = core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'])
    if not equal(model.state_dict(), other.state_dict()) or not equal(carry, loaded['episode_carry']):
        raise AssertionError('theta/C-carry serialized reload differs')
    with torch.no_grad():
        reloaded, second = other.forward_value(**probe)
    if not torch.equal(output, reloaded) or not equal(carry, second):
        raise AssertionError('strict reloaded NN execution differs')
    rejects(lambda: core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']), 'old AB checkpoint')
    rejects(lambda: previous.ContinuousPrimitiveBlock.from_checkpoint(saved['block_checkpoint']), 'new checkpoint as old AB')
    rejects(lambda: core.ContinuousPrimitiveBlock.from_checkpoint(saved['block_checkpoint'], expected_action_policy='wrong'), 'expected policy')
    for key, value in (('dual_ridge', 1e-5), ('dual_epsilon', 1e-5), ('input_only_address', False), ('coverage_normalization', True)):
        changed = copy.deepcopy(saved['block_checkpoint'])
        changed['configuration'][key] = value
        rejects(lambda changed=changed: core.ContinuousPrimitiveBlock.from_checkpoint(changed), key)
    for key in ('version', 'organization_policy'):
        changed = copy.deepcopy(saved['block_checkpoint'])
        changed[key] = 'wrong'
        rejects(lambda changed=changed: core.ContinuousPrimitiveBlock.from_checkpoint(changed), key)
    changed = copy.deepcopy(saved['block_checkpoint'])
    changed['state_dict']['key_composition'] = torch.zeros(8, 8)
    rejects(lambda: core.ContinuousPrimitiveBlock.from_checkpoint(changed), 'C is episode state, not theta')
    for key, value in (('schema', 'wrong'), ('architecture_arm', 'A'), ('stage', 2), ('step', 599), ('maintenance_complete', False)):
        bad = dict(payload)
        bad[key] = value
        rejects(lambda bad=bad: core.ContinuousPrimitiveBlock.from_owned_ab_weights(bad, expected_source_sha256=manifest['parent_source_sha256']), 'AB ownership '+key)
    rejects(lambda: core.ContinuousPrimitiveBlock.from_owned_ab_weights(payload, expected_source_sha256={}), 'wrong expected AB source')
    return dict(exact_theta_C_carry_and_execution_reload=True, strict_new_schema_and_old_AB_rejection=True,
                strict_ridge_epsilon_input_mode_and_episode_state_rejection=True, strict_owned_AB_source_stage_schema=True,
                optimizer_instantiated=False, raw=receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 120:
        raise ValueError('120science/150outer CPU envelope')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    wall, cpu = time.perf_counter(), time.process_time()
    partial = {}
    try:
        manifest = json.loads((args.inputs/'manifest.json').read_text())
        if manifest['schema'] != 'typed-dual-memory-CPU-v1':
            raise ValueError('fixed dual-memory manifest required')
        for leaf, digest in manifest['source_sha256'].items():
            if Path(leaf).name != leaf or sha(Path(__file__).with_name(leaf)) != digest:
                raise ValueError('frozen source differs: '+leaf)
        checkpoint_path = args.inputs/'ab.pt'
        if sha(checkpoint_path) != manifest['checkpoint_sha256']:
            raise ValueError('exact OWN AB checkpoint bytes required')
        payload, load_cost = timed(lambda: torch.load(checkpoint_path, map_location='cpu', weights_only=False))
        old_optimizer_payload_bytes = tensor_bytes(payload['optimizer'])
        before_rng = torch.get_rng_state()
        model, restore_cost = timed(lambda: core.ContinuousPrimitiveBlock.from_owned_ab_weights(payload, expected_source_sha256=manifest['parent_source_sha256']))
        after_rng = torch.get_rng_state()
        torch.set_rng_state(before_rng)
        old_model = previous.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'], expected_input_only_address=True, expected_coverage_normalization=True)
        if not equal(model.state_dict(), payload['model_state']) or not equal(model.state_dict(), old_model.state_dict()) or not torch.equal(after_rng, torch.get_rng_state()):
            raise AssertionError('exact OWN AB20 theta/factory RNG required')
        del old_model
        torch.set_rng_state(after_rng)
        if model.parameter_count() != 10351 or len(list(model.named_parameters())) != 20:
            raise AssertionError('P10351/all20 theta only')
        buffers = dict(model.named_buffers())
        if set(buffers) != {'dual_identity'} or tensor_bytes(buffers) != 256 or 'dual_identity' in model.state_dict():
            raise AssertionError('one shared nonpersistent256B identity only')
        initial_hash = state_hash(model)
        partial['source_rules'] = source_check()
        for name, function in (('transition', lambda: transition_check(model)), ('algebra', lambda: ridge_fixture(model, args.out)),
                               ('read_gradients', lambda: solve_gradient_check(model, args.out)), ('NN_gradients', lambda: nn_gradient_check(model, args.out)),
                               ('reload', lambda: reload_check(model, payload, manifest, args.out))):
            if time.perf_counter()-wall > args.max_seconds:
                raise TimeoutError('bounded CPU dual admission')
            partial[name], partial[name+'_cost'] = timed(function)
        if state_hash(model) != initial_hash or not equal(model.state_dict(), payload['model_state']):
            raise AssertionError('CPU admission changed owned theta')
        first = model.initial_carry(torch.tensor([0]))
        report = model.resource_report(64, 16, active_steps=64*16)
        if sum(value.numel() for value in first.values()) != 384 or report['persistent_example_state_payload_bytes'] != 64*1536:
            raise AssertionError('C64 replaces z8, carry384/1536B required')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(status='CPU_TYPED_DUAL_MEMORY_ADMISSION_PASS', P=10351, parameter_tensor_count=20,
                      exact_owned_AB_theta_unchanged=True, old_z_not_interpreted_as_C=True,
                      new_C_zero_at_episode_start=True, shared_nonpersistent_identity_bytes=256,
                      carry_scalars=384, carry_bytes_per_example=1536, extra_state_scalars_vs_AB=56,
                      all20_gradients_present_finite=True, all20_eager_masked_gradients_allclose=True,
                      strict_checkpoint_and_theta_only_restore=True, optimizer_instantiated=False,
                      optimizer_steps=0, SGD_updates=0, research_updates=0, task_data_or_teacher_used=False,
                      actual_checkpoint_sha256=manifest['checkpoint_sha256'], source_sha256=manifest['source_sha256'],
                      actual_theta_sha256=initial_hash, checks=partial,
                      costs=dict(checkpoint_load=load_cost, theta_restore=restore_cost,
                                 historical_optimizer_payload_bytes=old_optimizer_payload_bytes, live_optimizer_bytes=0,
                                 model_parameter_bytes=tensor_bytes(model.state_dict()), shared_buffer_bytes=256,
                                 carry_bytes_per_example=1536, original_checkpoint_bytes=checkpoint_path.stat().st_size,
                                 resource_report=report, CPU_peak_RSS_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                                 wall_seconds=time.perf_counter()-wall, CPU_seconds=time.process_time()-cpu,
                                 full_forward_FLOPs=None, full_backward_FLOPs=None, full_training_FLOPs=None,
                                 solve_workspace_bytes=None, factorization_or_library_cache_bytes=None,
                                 formation_search=dict(performed=False, measured_cost=None)),
                      scope='one fixed candidate CPU equation/ridge/finite-gradient/reload admission only; no optimizer, capability, novelty, learned binding, CUDA or efficiency claim')
        with (args.out/'result.json').open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], P=10351, carry_scalars=384, optimizer_steps=0)))
    except BaseException as error:
        with (args.out/'failure.json').open('x') as stream:
            json.dump(dict(error=repr(error), traceback=traceback.format_exc(), partial_checks=partial), stream, indent=2)
        raise


if __name__ == '__main__':
    main()
