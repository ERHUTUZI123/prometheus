"""ROOT-RUN finite CPU checks for NEW slot-delta write, no research data.

Runs actual original/new _write functions on synthetic finite tensors, checks
uniform addressed mean/contrasts, exact one-hot equivalence, gradients, NN
storage/checkpoint compatibility and the +M(Q+H) forward MAC proxy. No optimizer,
training, data labels, network, subprocess or import-time computation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ORIGINAL_SHA = "b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
DELTA_SHA = "33a2d9f27f2a12f844727490a66edafe984fdf98cdb053e2c910a5cbcb0c46bd"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name, expected):
    if path.is_symlink() or sha(path) != expected:
        raise ValueError("exact owned source required: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ContinuousPrimitiveBlock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-source", type=Path, required=True)
    parser.add_argument("--delta-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.out.exists() or not 0 < args.max_seconds <= 60:
        raise ValueError("new output and <=60-second CPU check required")
    import torch
    began = time.perf_counter()
    deadline = began + args.max_seconds
    args.out.mkdir(parents=True)
    results = []

    def check():
        if time.perf_counter() > deadline:
            raise TimeoutError("finite CPU delta write check bound")

    def equal_state(a, b):
        return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

    def close(a, b):
        if not torch.allclose(a, b, atol=2e-6, rtol=2e-6):
            raise AssertionError("synthetic tensor identity failed")
        return float((a - b).abs().max())

    try:
        torch.set_num_threads(1)
        torch.manual_seed(2718)
        Original = load(args.original_source, "_owned_b8_slot_write_check", ORIGINAL_SHA)
        Delta = load(args.delta_source, "_owned_delta_slot_write_check", DELTA_SHA)
        dims = dict(n_states=8, n_tokens=15, internal_states=4, n_primitives=2,
                    token_dim=3, state_dim=4, memory_dim=4, caller_hidden=3,
                    n_slots=1, memory_slots=3, key_dim=2, primitive_rank=2)
        old = Original(**dims)
        new = Delta(**dims)
        with torch.no_grad():
            old.write_query.weight.zero_(); old.write_query.bias.zero_()
            old.write_gate.weight.zero_(); old.write_gate.bias.zero_()
            old.write_key.weight.zero_(); old.write_key.bias.copy_(torch.tensor([.3, -.2]))
            old.write_value.weight.zero_(); old.write_value.bias.copy_(torch.tensor([.1, .2, .3, .4]))
        new.load_state_dict(old.state_dict(), strict=True)
        assert old.configuration() == new.configuration()
        assert equal_state(old.state_dict(), new.state_dict())
        assert old.parameter_count() == new.parameter_count()
        h, x, read = torch.randn(2, 4), torch.randn(2, 3), torch.randn(2, 4)
        keys = torch.tensor([[[.2, -.3], [.7, .1], [-.4, .8]]]).expand(2, -1, -1).clone()
        values = torch.randn(2, 3, 4)
        with torch.no_grad():
            ok, ov, oa, og = old._write(h, x, read, keys, values)
            nk, nv, na, ng = new._write(h, x, read, keys, values)
        close(oa, torch.full_like(oa, 1 / 3)); close(og, torch.full_like(og, .5))
        assert torch.equal(oa, na) and torch.equal(og, ng)
        mean_error = max(close(ok.mean(1), nk.mean(1)), close(ov.mean(1), nv.mean(1)))
        contrast_error = max(close(nk[:, 1:] - nk[:, :1], keys[:, 1:] - keys[:, :1]),
                             close(nv[:, 1:] - nv[:, :1], values[:, 1:] - values[:, :1]))
        old_contraction_error = max(close(ok[:, 1:] - ok[:, :1], (1 - .5 / 3) * (keys[:, 1:] - keys[:, :1])),
                                   close(ov[:, 1:] - ov[:, :1], (1 - .5 / 3) * (values[:, 1:] - values[:, :1])))
        results.append(dict(check="actual_uniform_write", addressed_mean_same_max_error=mean_error,
                            new_slot_contrasts_preserved_max_error=contrast_error,
                            old_expected_contraction_max_error=old_contraction_error))
        check()
        with torch.no_grad():
            old.write_query.bias.copy_(torch.tensor([1., 0.]))
            new.load_state_dict(old.state_dict(), strict=True)
        hot_keys = torch.tensor([[[1e4, 0.], [-1e4, 0.], [-1e4, 0.]]]).expand(2, -1, -1).clone()
        with torch.no_grad():
            ok, ov, oa, og = old._write(h, x, read, hot_keys, values)
            nk, nv, na, ng = new._write(h, x, read, hot_keys, values)
        assert torch.equal(oa, torch.tensor([[1., 0., 0.]]).expand(2, -1))
        assert torch.equal(ok, nk) and torch.equal(ov, nv)
        results.append(dict(check="actual_representable_onehot_write", outputs_exactly_equal=True,
                            source="finite extreme synthetic logits produce exact FP softmax1/0, no patched softmax"))
        check()
        torch.manual_seed(3141)
        gradient_model = Delta(**dims)
        state_before = {k: v.clone() for k, v in gradient_model.state_dict().items()}
        inputs = [torch.randn(2, 4, requires_grad=True), torch.randn(2, 3, requires_grad=True),
                  torch.randn(2, 4, requires_grad=True), torch.randn(2, 3, 2, requires_grad=True),
                  torch.randn(2, 3, 4, requires_grad=True)]
        output = gradient_model._write(*inputs)
        variables = inputs + [p for n, p in gradient_model.named_parameters() if n.startswith("write_")]
        gradients = torch.autograd.grad(output[0].square().sum() + output[1].square().sum(), variables)
        assert all(bool(torch.isfinite(g).all()) for g in gradients)
        assert equal_state(state_before, gradient_model.state_dict())
        results.append(dict(check="actual_full_write_gradient", finite_gradient_tensors=len(gradients),
                            optimizer_updates=0, NN_values_unchanged=True))
        check()
        legacy_payload = old.checkpoint()
        restored_new = Delta.from_checkpoint(legacy_payload)
        delta_payload = new.checkpoint()
        roundtrip_delta = Delta.from_checkpoint(delta_payload)
        assert equal_state(restored_new.state_dict(), old.state_dict())
        assert equal_state(roundtrip_delta.state_dict(), new.state_dict())
        assert delta_payload["version"] == "continuous-primitive-block-slot-delta-v1"
        assert delta_payload["write_update_policy"] == "slot-delta-residual-write-v1"
        try:
            Original.from_checkpoint(delta_payload)
        except ValueError:
            pass
        else:
            raise AssertionError("old loader silently accepted a new execution rule")
        bad_policy = dict(delta_payload, write_update_policy="wrong-rule")
        try:
            Delta.from_checkpoint(bad_policy)
        except ValueError:
            pass
        else:
            raise AssertionError("delta loader accepted a different policy")
        before = old.resource_report(batch_size=2, sequence_length=5, active_steps=7)
        after = new.resource_report(batch_size=2, sequence_length=5, active_steps=7)
        assert after["proxy_forward_macs"] - before["proxy_forward_macs"] == 7 * 3 * (2 + 4)
        assert after["persistent_example_state_scalars"] == before["persistent_example_state_scalars"]
        assert after["write_summary_workspace_scalars"] == 2 * (2 + 4)
        results.append(dict(check="actual_checkpoint_and_resource_contract", NN_shapes_values_and_P_compatible=True,
                            constructor_configuration_unchanged=True, optional_policy_metadata_outside_NN_constructor=True,
                            source_selected_legacy_NN_migration_explicit=True,
                            new_version_policy_and_NN_roundtrip_exact=True, old_loader_rejects_new_rule=True,
                            proxy_extra_per_active_token=18, no_extra_carry=True,
                            summary_workspace_scalars_per_example=6,
                            function_boundary="storage-compatible does not mean original-rule function-preserving"))
        result = dict(status="ACTUAL_CPU_DELTA_WRITE_CONTRACT_PASSED", results=results,
                      source_sha256=dict(original=ORIGINAL_SHA, delta=DELTA_SHA, checker=sha(Path(__file__))),
                      wall_seconds=time.perf_counter() - began, optimizer_updates=0, research_labels_read=0,
                      torch_version=str(torch.__version__), scope="finite synthetic code/math checks, not memory-learning success")
        (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result))
    except BaseException as error:
        (args.out / "failure.json").write_text(json.dumps(dict(error_type=type(error).__name__, error=str(error),
            completed=results, wall_seconds=time.perf_counter() - began), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
