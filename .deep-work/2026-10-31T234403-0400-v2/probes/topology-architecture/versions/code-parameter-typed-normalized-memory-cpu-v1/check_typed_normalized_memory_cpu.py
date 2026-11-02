"""Bounded synthetic CPU admission for the fixed A/B memory interface.

No research task data or teacher actions are read. A/B receive no optimizer
update; AB receives one discarded synthetic Adam update to check real moment
migration. CPU admission is not a capability or CUDA-Graph result.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
import random
import resource
import sys
import textwrap
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import continuous_primitive_block_typed_normalized_memory as core
import owned_typed_memory_core as old_typed
import owned_parent_driver as old
import typed_normalized_memory_masked_graph_fb as fb
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage


MODES = {'A': (True, False), 'B': (False, True), 'AB': (True, True)}
REQUIRED_SOURCES = {
    'continuous_primitive_block_typed_normalized_memory.py',
    'typed_normalized_memory_masked_graph_fb.py', 'owned_typed_memory_core.py',
    'owned_fcba_core.py', 'owned_parent_driver.py', 'context_replay128.py',
    'memory_live_stage_train.py',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng():
    return dict(cpu=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())


def restore(state):
    torch.set_rng_state(state['cpu'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def timed(function):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    return value, dict(wall_seconds=time.perf_counter()-wall, CPU_seconds=time.process_time()-cpu)


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def close(left, right, label, atol=fb.OUTPUT_ATOL, rtol=fb.OUTPUT_RTOL):
    if left.keys() != right.keys():
        raise AssertionError(label+' carry field mismatch')
    for key in left:
        if not bool(torch.isfinite(left[key]).all()) or not torch.allclose(left[key], right[key], atol=atol, rtol=rtol):
            raise AssertionError(label+' mismatch: '+key)


def assignment_map(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = ast.dump(node.value, include_attributes=False)
    return result


def source_rules():
    previous = assignment_map(old_typed.ContinuousPrimitiveBlock._transition)
    current = assignment_map(core.ContinuousPrimitiveBlock._transition)
    names = ('c', 'x', 'attention', 'features', 'all_logits', 'action_logits',
             'probabilities', 'qcomp', 'chosen', 'chosen_compute', 'hard_weights',
             'gates', 'compute_total', 'h', 'amount', 'v', 'address')
    for name in names:
        if name not in previous or previous[name] != current.get(name):
            raise AssertionError('unchanged routing/OLDh write source differs: '+name)
    helper = ast.parse(inspect.getsource(fb.masked_forward_value))
    forbidden = {'nonzero', 'item', 'cpu', 'numpy', 'detach'}
    if any(isinstance(node, ast.Attribute) and node.attr in forbidden for node in ast.walk(helper)):
        raise AssertionError('masked helper contains host/data-dependent transition operation')
    return dict(unchanged_routing_and_value_write_AST_assignments=list(names),
                masked_helper_without_host_or_detach_operations=True)


def reference_transition(model, carry, raw):
    """Independent soft equations; never calls source _transition."""
    h, c, values, address = (carry[key] for key in ('h', 'controller', 'values', 'address'))
    c = model.controller(raw, c)
    x = raw+model.slot_embedding.weight[0]
    attention = F.softmax(address, -1)
    slots = values / torch.maximum(carry['coverage'], values.new_tensor(1e-4))[:, :, None] if model.coverage_normalization else values
    read = (attention[:, :, None]*slots).sum(1)
    features = torch.cat((h, c, x, read), -1)
    logits = model.caller(features)
    alpha, q = F.softmax(logits[:, :7], -1), F.softmax(logits[:, :4], -1)
    gate = torch.sigmoid(logits[:, 7:15]).repeat_interleave(4, -1)
    low = F.gelu(torch.einsum('bf,krf->bkr', features, model.primitive_down)+model.primitive_bias)
    candidates = torch.einsum('bkr,khr->bkh', low, model.primitive_up)
    delta = torch.einsum('bk,bkh->bh', q, candidates)
    chi = h+gate*(model.residual_norm(h+delta)-h)
    new_h = h+alpha[:, :4].sum(-1, keepdim=True)*(chi-h)+alpha[:, 4:5]*(read-h)
    amount = alpha[:, 5:6]*attention
    proposal = F.linear(raw if model.input_only_address else features, model.address_proposal.weight, model.address_proposal.bias)
    updated = dict(h=new_h, controller=c, values=values+amount[:, :, None]*(h[:, None]-values),
                   address=address+alpha[:, 6:7]*(proposal-address), logp=F.log_softmax(model.readout(new_h), -1))
    if model.coverage_normalization:
        updated['coverage'] = carry['coverage']+amount*(1-carry['coverage'])
    return updated, dict(read=read, attention=attention, features=features, alpha=alpha, q=q,
                         gate=gate, proposal=proposal, logits=logits, amount=amount)


def equation_checks(model):
    before, before_rng = copy.deepcopy(model.state_dict()), rng()
    initial = torch.arange(4)
    carry = model.initial_carry(initial)
    expected_fields = {'h', 'controller', 'values', 'address', 'logp'} | ({'coverage'} if model.coverage_normalization else set())
    if carry.keys() != expected_fields or len(list(model.named_buffers())):
        raise AssertionError('coverage must be optional per-example carry, no persistent buffer')
    if any(bool(carry[key].count_nonzero()) for key in ('values', 'address')+ (('coverage',) if model.coverage_normalization else ())):
        raise AssertionError('memory/address/coverage initial state must be zero')
    carry['h'] = torch.linspace(-.8, .9, 128).reshape(4, 32)
    carry['controller'] = torch.linspace(-.2, .4, 64).reshape(4, 16)
    carry['values'] = torch.linspace(-.01, .02, 1024).reshape(4, 8, 32)
    carry['address'] = torch.linspace(-2., 2., 32).reshape(4, 8)
    if model.coverage_normalization:
        carry['coverage'] = torch.linspace(.2, .9, 32).reshape(4, 8)
    raw = torch.linspace(-1., 1., 64).reshape(4, 16)
    with torch.no_grad():
        actual, trace = model._transition(carry, raw, 'soft')
        expected, rule = reference_transition(model, carry, raw)
        close(actual, expected, 'independent soft equations')
        for key, reference in (('read', rule['read']), ('action_probabilities', rule['alpha']),
                               ('call_probabilities', rule['q']), ('state_gate', rule['gate']),
                               ('address_proposal', rule['proposal'])):
            if not torch.allclose(trace[key], reference, atol=2e-6, rtol=2e-6):
                raise AssertionError('independent trace mismatch: '+key)
        if not torch.equal(trace['write_value'], carry['h']) or not torch.equal(trace['address_query'], carry['address']):
            raise AssertionError('write/read no longer use OLD h/address')
        if model.coverage_normalization and (not torch.equal(trace['read_coverage'], carry['coverage']) or not torch.equal(trace['coverage'], actual['coverage'])):
            raise AssertionError('coverage trace does not distinguish old/new state')
        if model.input_only_address:
            changed = {key: value.flip(0)+.73 for key, value in carry.items()}
            _, other = model._transition(changed, raw, 'soft')
            if not torch.equal(trace['address_proposal'], other['address_proposal']):
                raise AssertionError('A address proposal depends on incoming carry')
            temporary = copy.deepcopy(model)
            temporary.slot_embedding.weight.add_(3.)
            _, other = temporary._transition(carry, raw, 'soft')
            if not torch.equal(trace['address_proposal'], other['address_proposal']):
                raise AssertionError('A address proposal includes slot offset')
            del temporary
    if not old.tensor_state_equal(before, model.state_dict()) or not old.nested_state_equal(before_rng, rng()):
        raise AssertionError('equation check mutated source NN or RNG')
    return dict(independent_soft_equations=True, exact_OLDh_value_write=True,
                OLDaddress_read_and_nexttoken_address_update=True, full_context_router_unchanged=True,
                A_raw_x_only_no_slot_offset=bool(model.input_only_address),
                B_per_cell_normalization_before_attention=bool(model.coverage_normalization),
                parameter_or_buffer_added_by_B=False, source_NN_and_RNG_unchanged=True)


def coverage_checks(model):
    if not model.coverage_normalization:
        return dict(enabled=False, coverage_key_absent=True)
    before, before_rng = copy.deepcopy(model.state_dict()), rng()
    forced = copy.deepcopy(model)
    raw = torch.linspace(-.2, .3, 16).reshape(1, 16)
    h1 = torch.linspace(-.7, .8, 32).reshape(1, 32)
    h2 = h1.flip(-1)*.3+.2
    def set_store_mass(mass):
        probabilities = forced.caller[2].bias.new_full((7,), (1-mass)/6)
        probabilities[5] = mass
        with torch.no_grad():
            forced.caller[2].weight[:7].zero_()
            forced.caller[2].bias[:7].copy_(probabilities.log())
            forced.address_proposal.weight.zero_()
            forced.address_proposal.bias.zero_()
    with torch.no_grad():
        set_store_mass(.08)
        state = forced.initial_carry(h1)
        zero, ztrace = forced._transition(state, raw, 'soft')
        if not bool(torch.isfinite(ztrace['read']).all()) or bool(ztrace['read'].count_nonzero()):
            raise AssertionError('zero V/zero z must read finite zero')
        w1 = ztrace['write_gate']*ztrace['read_attention']
        if not torch.equal(zero['values'], w1[:, :, None]*h1[:, None]) or not torch.equal(zero['coverage'], w1):
            raise AssertionError('single weak write exact V/z equations failed')
        _, first_read = forced._transition(zero, raw, 'soft')
        if not torch.allclose(first_read['read'], h1, atol=1e-6, rtol=1e-6):
            raise AssertionError('single weak write above epsilon did not recover h')
        zero['h'] = h2
        second, second_trace = forced._transition(zero, raw, 'soft')
        w2 = second_trace['write_gate']*second_trace['read_attention']
        expected_v = (w1*(1-w2))[:, :, None]*h1[:, None]+w2[:, :, None]*h2[:, None]
        expected_z = w1*(1-w2)+w2
        close({'values': second['values'], 'coverage': second['coverage']},
              {'values': expected_v, 'coverage': expected_z}, 'two recency weighted writes', 1e-7, 2e-6)
        _, read_two = forced._transition(second, raw, 'soft')
        expected_read = (read_two['read_attention'][:, :, None]*expected_v/expected_z[:, :, None]).sum(1)
        if not torch.allclose(read_two['read'], expected_read, atol=1e-6, rtol=1e-6):
            raise AssertionError('repeated writes are not recency-weighted normalized average')
        set_store_mass(4e-4)
        tiny, tiny_trace = forced._transition(forced.initial_carry(h1), raw, 'soft')
        _, tiny_read = forced._transition(tiny, raw, 'soft')
        tiny_expected = (tiny_read['read_attention'][:, :, None]*tiny['values']/1e-4).sum(1)
        if not torch.allclose(tiny_read['read'], tiny_expected, atol=1e-6, rtol=1e-6) or not bool((tiny['coverage'] < 1e-4).all()):
            raise AssertionError('subepsilon read must retain declared clamp attenuation')
    # Focused new-interface derivative check at zero/tiny/both sides of clamp.
    differentiable = model.initial_carry(torch.tensor([0, 1]))
    differentiable['coverage'] = torch.tensor([0., 1e-8, 5e-5, 1e-4, 2e-4, .01, .5, 1.]).repeat(2, 1)
    differentiable['values'] = torch.linspace(-1e-3, 1e-3, 512).reshape(2, 8, 32)
    differentiable = {key: value.detach().clone().requires_grad_(key != 'logp') for key, value in differentiable.items()}
    raw2 = torch.linspace(-.4, .4, 32).reshape(2, 16).requires_grad_()
    updated, trace = model._transition(differentiable, raw2, 'soft')
    cotangent = torch.linspace(.01, .8, 64).reshape(2, 32)
    independent_read = (F.softmax(differentiable['address'], -1)[:, :, None]*differentiable['values']/differentiable['coverage'].clamp_min(1e-4)[:, :, None]).sum(1)
    variables = tuple(differentiable[key] for key in ('values', 'coverage', 'address'))
    actual_grad = torch.autograd.grad((trace['read']*cotangent).sum(), variables, retain_graph=True)
    expected_grad = torch.autograd.grad((independent_read*cotangent).sum(), variables, retain_graph=True)
    maximum = 0.
    for actual, expected in zip(actual_grad, expected_grad):
        if not bool(torch.isfinite(actual).all()) or not torch.allclose(actual, expected, atol=2e-5, rtol=2e-5):
            raise AssertionError('finite coverage read derivative differs from equation')
        maximum = max(maximum, float((actual-expected).abs().max()))
    objective = sum(value.square().mean() for value in updated.values())
    full_variables = tuple(model.parameters())+tuple(value for key, value in differentiable.items() if key != 'logp')+(raw2,)
    gradients = torch.autograd.grad(objective, full_variables, allow_unused=True)
    if any(value is None or not bool(torch.isfinite(value).all()) for value in gradients):
        raise AssertionError('new full transition has missing/nonfinite derivative at tiny z')
    if not old.tensor_state_equal(before, model.state_dict()) or not old.nested_state_equal(before_rng, rng()):
        raise AssertionError('coverage diagnostic changed source NN/RNG')
    return dict(enabled=True, zero_V_zero_z_finite_zero=True, single_weak_write_exact_V_z=True,
                single_weak_write_above_floor_recovers_value=True, repeated_write_recency_weighted_average=True,
                weighting='w_i*product_{j>i}(1-w_j), normalized by 1-product_i(1-w_i)',
                subepsilon_coverage_attenuation_checked=True, epsilon=1e-4,
                zero_tiny_threshold_and_positive_z_derivatives_finite=True,
                read_derivative_independent_equation_max_abs=maximum,
                full_transition_all_parameter_carry_raw_gradients_finite=True,
                source_NN_and_RNG_unchanged=True)


def synthetic_batch(length, n_tokens):
    row = torch.arange(64)
    lengths = row % (length+1)
    lengths[-1] = length
    return dict(initial=row % 8, tokens=(row[:, None]*3+torch.arange(length)[None]*5) % n_tokens,
                lengths=lengths, targets=(row+3) % 8)


def numeric_case(model, data, outdir, leaf, repeat=False):
    initial_state, initial_rng = copy.deepcopy(model.state_dict()), rng()
    def eager_step():
        model.zero_grad(set_to_none=True)
        loss, output = fb.objective(model, fb.original_forward_value, **data)
        loss.backward()
        return output.detach().clone(), loss.detach().clone(), {name: p.grad.detach().clone() if p.grad is not None else None for name, p in model.named_parameters()}
    (eager_output, eager_loss, eager_grads), eager_cost = timed(eager_step)
    fb.allocate_gradient_buffers(model)
    pointers = {name: p.grad.data_ptr() for name, p in model.named_parameters()}
    (masked_output, masked_loss), masked_cost = timed(lambda: fb.masked_step(model, **data))
    masked_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    if len(eager_grads) != 20 or eager_grads.keys() != masked_grads.keys():
        raise AssertionError('exact all20 gradient names required')
    close({'output': eager_output, 'loss': eager_loss}, {'output': masked_output, 'loss': masked_loss}, 'eager/masked output/NLL')
    gradient_metrics = {}
    for name, eager in eager_grads.items():
        masked = masked_grads[name]
        if eager is None or not bool(torch.isfinite(eager).all()) or not bool(torch.isfinite(masked).all()) or not torch.allclose(eager, masked, atol=fb.GRAD_ATOL, rtol=fb.GRAD_RTOL):
            raise AssertionError('all20 finite numeric gradients failed: '+name)
        gradient_metrics[name] = dict(max_abs=float((eager-masked).abs().max()), eager_norm=float(eager.norm()), allclose=True)
    repeated_cost = None
    if repeat:
        (again_output, again_loss), repeated_cost = timed(lambda: fb.masked_step(model, **data))
        close({'output': eager_output, 'loss': eager_loss}, {'output': again_output, 'loss': again_loss}, 'repeated zeroed gradient output')
        for name, parameter in model.named_parameters():
            if not torch.allclose(parameter.grad, eager_grads[name], atol=fb.GRAD_ATOL, rtol=fb.GRAD_RTOL):
                raise AssertionError('masked repeat accumulated gradients: '+name)
    if pointers != {name: p.grad.data_ptr() for name, p in model.named_parameters()}:
        raise AssertionError('masked gradient storage pointer changed')
    with torch.no_grad():
        def inference():
            eager = model.forward_value(data['initial'], data['tokens'], data['lengths'])
            masked = fb.masked_forward_value(model, data['initial'], data['tokens'], data['lengths'])
            return eager, masked
        ((eager_out, eager_carry), (masked_out, masked_carry)), inference_cost = timed(inference)
        close(eager_carry, masked_carry, 'full ragged carry including coverage/cache')
        zero_lengths = torch.zeros_like(data['lengths'])
        zero_out, zero_carry = fb.masked_forward_value(model, data['initial'], data['tokens'], zero_lengths)
        first = model.initial_carry(data['initial'])
        if not torch.equal(zero_out, first['logp']) or any(not torch.equal(zero_carry[key], first[key]) for key in first):
            raise AssertionError('masked helper changed inactive full carry')
        traced, record = model(data['initial'], data['tokens'], data['lengths'], output='logits', return_traces=True)
        close({'output': eager_out}, {'output': traced}, 'traced emission')
        close(eager_carry, record['final_carry'], 'traced true final carry')
        if model.coverage_normalization:
            if record['coverage_states'].shape != (64, data['tokens'].shape[1]+1, 8):
                raise AssertionError('coverage history must include true initial/final states')
            for step in range(data['tokens'].shape[1]):
                inactive = step >= data['lengths']
                if not torch.equal(record['coverage_states'][inactive, step], record['coverage_states'][inactive, step+1]):
                    raise AssertionError('inactive traced coverage changed')
    if not old.tensor_state_equal(initial_state, model.state_dict()) or not old.nested_state_equal(initial_rng, rng()):
        raise AssertionError('numeric check changed NN/RNG')
    arrays = dict(eager_output=eager_output.numpy(), masked_output=masked_output.numpy(), eager_loss=eager_loss.numpy(), masked_loss=masked_loss.numpy(), gradient_names_json=np.frombuffer(json.dumps(list(eager_grads)).encode(), dtype=np.uint8))
    for index, name in enumerate(eager_grads):
        arrays[f'eager_grad_{index:02d}'], arrays[f'masked_grad_{index:02d}'] = eager_grads[name].numpy(), masked_grads[name].numpy()
    for key in eager_carry:
        arrays['eager_carry_'+key], arrays['masked_carry_'+key] = eager_carry[key].numpy(), masked_carry[key].numpy()
    for key, value in data.items():
        arrays['synthetic_'+key] = value.numpy()
    path = outdir/leaf
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(all20_gradients_present_finite=True, all20_gradients_allclose=True,
                gradients=gradient_metrics, output_max_abs=float((eager_output-masked_output).abs().max()),
                NLL_abs=float((eager_loss-masked_loss).abs()), all_carry_fields_masked=True,
                all_inactive_full_carry_bitwise_unchanged=True, coverage_history_checked=bool(model.coverage_normalization),
                masked_gradient_storage_stable=True, gradient_zero_accumulation_repeat_checked=repeat,
                eager_forward_backward_cost=eager_cost, masked_forward_backward_cost=masked_cost,
                masked_repeat_cost=repeated_cost, one_eager_plus_one_masked_inference_cost=inference_cost,
                raw_file=path.name, raw_bytes=path.stat().st_size, raw_sha256=sha(path))


def rejected(function, label):
    try:
        function()
    except (ValueError, RuntimeError, TypeError, KeyError):
        return
    raise AssertionError('invalid checkpoint/configuration accepted: '+label)


def checkpoint_check(model, optimizer, replay, carry, path):
    saved_rng = rng()
    payload = dict(schema='typed-normalized-memory-CPU-synthetic-v1', research_updates=0,
                   synthetic_optimizer_steps=int(model.input_only_address and model.coverage_normalization),
                   block_checkpoint=model.checkpoint(), optimizer=optimizer.state_dict(),
                   replay=replay.state_dict(), carry={key: value.detach().clone() for key, value in carry.items()}, rng=saved_rng)
    with path.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    other = core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'], expected_input_only_address=model.input_only_address, expected_coverage_normalization=model.coverage_normalization)
    other_optimizer = torch.optim.Adam(other.parameters(), lr=.003)
    other_optimizer.load_state_dict(loaded['optimizer'])
    other_replay = Replay128.from_state_dict(loaded['replay'])
    if not old.tensor_state_equal(model.state_dict(), other.state_dict()) or not old.nested_state_equal(optimizer.state_dict(), other_optimizer.state_dict()) or not old.nested_state_equal(replay.state_dict(), other_replay.state_dict()) or not old.nested_state_equal(carry, loaded['carry']):
        raise AssertionError('actual NN/Adam/replay/fullcarry reload mismatch')
    probe = dict(initial=torch.tensor([0, 3]), tokens=torch.tensor([[1, 3], [2, 3]]), lengths=torch.tensor([2, 2]))
    with torch.no_grad():
        for execution in ('soft', 'native', 'straight_through'):
            first, last = model.forward_value(**probe, execution=execution)
            second, other_last = other.forward_value(**probe, execution=execution)
            if not torch.equal(first, second) or any(not torch.equal(last[key], other_last[key]) for key in last):
                raise AssertionError('reloaded execution/fullcarry differs: '+execution)
    cp = loaded['block_checkpoint']
    rejected(lambda: old_typed.ContinuousPrimitiveBlock.from_checkpoint(cp), 'old typed schema')
    rejected(lambda: core.ContinuousPrimitiveBlock.from_checkpoint(cp, expected_action_policy='wrong'), 'wrong expected policy')
    for key, argument in (('input_only_address', 'expected_input_only_address'), ('coverage_normalization', 'expected_coverage_normalization')):
        rejected(lambda argument=argument, key=key: core.ContinuousPrimitiveBlock.from_checkpoint(cp, **{argument: not getattr(model, key)}), 'wrong expected '+key)
        for invalid in (None, 0, 1, 'true'):
            changed = copy.deepcopy(cp)
            changed['configuration'][key] = invalid
            rejected(lambda changed=changed: core.ContinuousPrimitiveBlock.from_checkpoint(changed), 'nonbool '+key)
        changed = copy.deepcopy(cp)
        del changed['configuration'][key]
        rejected(lambda changed=changed: core.ContinuousPrimitiveBlock.from_checkpoint(changed), 'missing '+key)
    mutations = [('version', 'wrong'), ('organization_policy', 'wrong')]
    for key, value in mutations:
        changed = copy.deepcopy(cp)
        changed[key] = value
        rejected(lambda changed=changed: core.ContinuousPrimitiveBlock.from_checkpoint(changed), key)
    for key, value in (('coverage_epsilon', 1e-5), ('action_policy', 'wrong')):
        changed = copy.deepcopy(cp)
        changed['configuration'][key] = value
        rejected(lambda changed=changed: core.ContinuousPrimitiveBlock.from_checkpoint(changed), key)
    changed = copy.deepcopy(cp)
    changed['state_dict']['unexpected_coverage_buffer'] = torch.zeros(8)
    rejected(lambda: core.ContinuousPrimitiveBlock.from_checkpoint(changed), 'strict unexpected buffer')
    changed = copy.deepcopy(cp)
    changed['configuration'].update(input_only_address=False, coverage_normalization=False)
    rejected(lambda: core.ContinuousPrimitiveBlock.from_checkpoint(changed), 'none arm is frozen historical control')
    restore(loaded['rng'])
    if not old.nested_state_equal(rng(), saved_rng):
        raise AssertionError('CPU/Python/NumPy RNG reload failed')
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                actual_NN_Adam_replay_fullcarry_CPU_RNG_reload=True,
                strict_mode_flags_and_expected_flags_rejected=True,
                legacy_schema_wrong_policy_epsilon_extra_buffer_rejected=True,
                soft_native_ST_reloaded_execution_exact=True)


def mode_check(name, flags, outdir):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    initial_rng = rng()
    model, init_cost = timed(lambda: core.ContinuousPrimitiveBlock(8, 15, input_only_address=flags[0], coverage_normalization=flags[1]))
    after_init, initial_hash = rng(), state_sha(model)
    restore(initial_rng)
    historical = old_typed.ContinuousPrimitiveBlock(8, 15)
    expected = {key: value.detach().clone() for key, value in historical.state_dict().items()}
    if flags[0]:
        expected['address_proposal.weight'] = expected['address_proposal.weight'][:, 48:64].clone()
    if not old.tensor_state_equal(model.state_dict(), expected) or not old.nested_state_equal(after_init, rng()):
        raise AssertionError('exact historical common tensors/subset and RNG required')
    # On zero-memory carry, router/conditional-experts/G8 must still equal efab.
    with torch.no_grad():
        raw = torch.linspace(-.8, .8, 64).reshape(4, 16)
        initial = torch.arange(4)
        _, previous = historical._transition(historical.initial_carry(initial), raw, 'soft')
        _, current = model._transition(model.initial_carry(initial), raw, 'soft')
        for key in ('action_logits', 'action_probabilities', 'call_probabilities', 'state_gate', 'candidate_h'):
            if not torch.equal(previous[key], current[key]):
                raise AssertionError('unchanged routing/candidate differs at common incoming carry: '+key)
    del historical
    restore(after_init)
    p15 = 10335 if flags[0] else 10975
    carry_scalars = 328 if flags[1] else 320
    if model.parameter_count() != p15 or len(list(model.named_parameters())) != 20 or not all(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError('exact P15/all20 live tensors required')
    equation, equation_cost = timed(lambda: equation_checks(model))
    coverage, coverage_cost = timed(lambda: coverage_checks(model))
    numeric16 = numeric_case(model, synthetic_batch(16, 15), outdir, name+'-B64-L16-all20.npz', repeat=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    synthetic_update = None
    if name == 'AB':
        def one_discarded_adam():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = fb.objective(model, fb.original_forward_value, **synthetic_batch(16, 15))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if len(optimizer.state) != 20 or not all(float(optimizer.state[parameter]['step']) == 1 for parameter in model.parameters()):
                raise AssertionError('one actual all20 Adam step required')
            return dict(NLL=float(loss.detach()), preclip_gradient_norm=float(norm))
        synthetic_update, update_cost = timed(one_discarded_adam)
        synthetic_update['cost'] = update_cost
    probe = dict(initial=torch.tensor([0, 3]), tokens=torch.tensor([[1, 3], [2, 3]]), lengths=torch.tensor([2, 2]))
    with torch.no_grad():
        before_growth, carry_before_growth = model.forward_value(**probe)
    saved_rng = rng()
    expected_row = torch.randn((1, 16))
    expected_growth_rng = rng()
    restore(saved_rng)
    growth, growth_cost = timed(lambda: grow_storage(model, optimizer, argparse.Namespace(arm='dynamic', block_state='continuous')))
    if not torch.equal(model.token_embedding.weight[-1:], expected_row) or not old.nested_state_equal(rng(), expected_growth_rng):
        raise AssertionError('growth must consume exactly one random 1x16 embedding row')
    with torch.no_grad():
        after_growth, carry_after_growth = model.forward_value(**probe)
    if model.parameter_count() != p15+16 or not torch.equal(before_growth, after_growth) or any(not torch.equal(carry_before_growth[key], carry_after_growth[key]) for key in carry_before_growth):
        raise AssertionError('15 to16 growth changed old-token fullcarry/parameters')
    if model.configuration()['input_only_address'] is not flags[0] or model.configuration()['coverage_normalization'] is not flags[1]:
        raise AssertionError('growth changed prospective mode flags')
    numeric48 = numeric_case(model, synthetic_batch(48, 16), outdir, name+'-B64-L48-all20.npz')
    with torch.no_grad():
        _, final_carry = model.forward_value(**probe)
    measured_carry_scalars = sum(value.numel() for value in final_carry.values())//2
    report = model.resource_report(64, 48, active_steps=int(synthetic_batch(48, 16)['lengths'].sum()), return_traces=True)
    if measured_carry_scalars != carry_scalars or report['persistent_example_state_scalars'] != 64*carry_scalars:
        raise AssertionError('logical carry payload count wrong')
    if report['parameter_payload_bytes'] != sum(parameter.numel()*parameter.element_size() for parameter in model.parameters()):
        raise AssertionError('parameter payload accounting wrong')
    replay = Replay128(128, 48, 0, 8)
    raw_data = {key: value.numpy() for key, value in synthetic_batch(48, 16).items()}
    used = np.arange(64, dtype=np.int64)[None]
    replay.consume(raw_data, used, 0)
    replay.validate_sources({0: raw_data}, {0: used})
    receipt, checkpoint_cost = timed(lambda: checkpoint_check(model, optimizer, replay, final_carry, outdir/(name+'-synthetic-checkpoint.pt')))
    return dict(actual_P15=p15, actual_P16=model.parameter_count(), parameter_tensor_count=20,
                input_only_address=flags[0], coverage_normalization=flags[1],
                initial_NN_sha256=initial_hash, exact_historical_common_initialization_and_RNG=True,
                only_address_weight_raw_x_subset_pruned=bool(flags[0]),
                unchanged_router_conditional_experts_G8_common_input_exact=True,
                logical_carry_scalars=carry_scalars, model_buffers=[], equations=equation,
                coverage=coverage, numeric16=numeric16, numeric48=numeric48,
                all20_gradients_checked=True, growth=growth, checkpoint=receipt,
                research_updates=0, synthetic_optimizer_updates=int(name == 'AB'), synthetic_update=synthetic_update,
                costs=dict(initialization=init_cost, equations=equation_cost, coverage_diagnostics=coverage_cost,
                           growth=growth_cost, checkpoint_save_load_validation=checkpoint_cost,
                           parameters=model.parameter_count(), parameter_bytes=report['parameter_payload_bytes'],
                           optimizer_tensor_bytes=old.tensor_bytes(optimizer.state_dict()),
                           carry_bytes_per_example=carry_scalars*4, replay_bytes=sum(value.nbytes for value in replay.arrays.values()),
                           buffer_bytes=0, resource_report=report, full_forward_FLOPs=None, full_training_FLOPs=None,
                           formation_search=dict(performed=False, measured_FLOPs=None),
                           compilation=dict(performed=False, seconds=None, measured_FLOPs=None),
                           protection=dict(performed=False, seconds=None, measured_FLOPs=None),
                           cache_or_allocator_peak_bytes=None,
                           measurement_scope='actual synthetic CPU operations only; component wall/CPU and tensor/file payloads; no CUDA/kernel peak or complete FLOPs measurements'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    wall, cpu = time.perf_counter(), time.process_time()
    modes = {}
    try:
        manifest = json.loads((args.inputs/'manifest.json').read_text())
        if manifest.get('schema') != 'typed-normalized-memory-CPU-v1' or not REQUIRED_SOURCES <= manifest['source_sha256'].keys():
            raise AssertionError('exact CPU synthetic source manifest required')
        folder = Path(__file__).resolve().parent
        for leaf, digest in manifest['source_sha256'].items():
            if Path(leaf).name != leaf or sha(folder/leaf) != digest:
                raise AssertionError('frozen execution source mismatch: '+leaf)
        if sha(folder/'owned_typed_memory_core.py') != 'efabc8cbd2754a4a1d60f52b907cb5842f11fcf716c52646f8f935db5170d3f2':
            raise AssertionError('historical efab initializer must remain exact')
        source = source_rules()
        for name, flags in MODES.items():
            modes[name], costs = timed(lambda name=name, flags=flags: mode_check(name, flags, args.out))
            modes[name]['whole_mode_cost'] = costs
        if len(modes) != 3 or sum(value['synthetic_optimizer_updates'] for value in modes.values()) != 1:
            raise AssertionError('three fixed modes and one discarded AB update only')
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = dict(status='CPU_TYPED_NORMALIZED_MEMORY_GRAPH_ADMISSION_PASS',
                      source_sha256=manifest['source_sha256'], source_rules=source, modes=modes,
                      all20_gradients_checked=True, all_carry_fields_masked=True,
                      exact_historical_common_initialization_and_RNG=True,
                      strict_mode_flags_and_checkpoint_rejection_checked=True,
                      research_updates=0, SGD_updates=0, synthetic_optimizer_updates=1,
                      task_data_or_teacher_labels_used=False, CPU_only_not_CUDA_result=True,
                      fresh_GPU_NN_must_not_load_synthetic_checkpoints=True,
                      wall_seconds=time.perf_counter()-wall, CPU_seconds=time.process_time()-cpu,
                      peak_RSS_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
                      evidence_artifact_bytes=sum(path.stat().st_size for path in args.out.iterdir() if path.is_file()),
                      full_forward_FLOPs=None, full_training_FLOPs=None, provider_effective_compute=None,
                      scope='fixed A/B/AB source/equation/gradient/initialization/growth/reload CPU admission; no task quality, novelty, retention, broad advantage or CUDA-Graph claim')
        with (args.out/'result.json').open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print(json.dumps(dict(status=result['status'], modes={name: {'P15': value['actual_P15'], 'P16': value['actual_P16']} for name, value in modes.items()})))
    except BaseException as error:
        with (args.out/'failure.json').open('x') as stream:
            json.dump(dict(error=repr(error), traceback=traceback.format_exc(), partial_modes=modes), stream, indent=2)
        raise


if __name__ == '__main__':
    main()
