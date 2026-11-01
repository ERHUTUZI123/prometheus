"""Finite construction checks and a counterexample; supplied laws, no learner."""
import json
from pathlib import Path
from algebra_core import (all_words, normal_form, reference_stack,
                          stack_rules, reference_counters, counter_rules)

report = {"epistemic_status": "EXPLORATORY_EXECUTION",
          "neural_learning_executed": False, "laws_supplied": True,
          "rewrite_budget_per_word": 1000, "failures": []}


def checkpoint(label):
    path = Path("results.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert json.loads(path.read_text()) == report
    print(json.dumps({"phase": label, "report": report}, sort_keys=True), flush=True)


stack_count = 0
stack_max_steps = 0
rules = stack_rules()
for actions in all_words(("Ca", "Cb", "Da", "Db"), 8):
    actual, steps = normal_form(tuple(reversed(actions)) + ("E",), rules)
    expected = reference_stack(actions)
    stack_count += 1
    stack_max_steps = max(stack_max_steps, steps)
    if actual != expected:
        report["failures"].append({"phase": "stack", "actions": list(actions),
                                   "actual": list(actual), "expected": list(expected)})
        checkpoint("stack_failure")
        raise AssertionError("stack mismatch")
report["stack"] = {"words_checked": stack_count, "max_length": 8,
                   "max_rewrite_steps": stack_max_steps,
                   "matched_independent_reference": True}
checkpoint("stack_complete")

counter_count = 0
counter_max_steps = 0
rules = counter_rules()
for actions in all_words(("x", "X", "y", "Y"), 8):
    actual, steps = normal_form(tuple(reversed(actions)), rules)
    expected = reference_counters(actions)
    counter_count += 1
    counter_max_steps = max(counter_max_steps, steps)
    if actual != expected:
        report["failures"].append({"phase": "counters", "actions": list(actions),
                                   "actual": list(actual), "expected": list(expected)})
        checkpoint("counter_failure")
        raise AssertionError("counter mismatch")
report["counters"] = {"words_checked": counter_count, "max_length": 8,
                      "max_rewrite_steps": counter_max_steps,
                      "matched_independent_reference": True}
checkpoint("counter_complete")

# Shift U_N and transpose D_N satisfy D_N U_N e_j=e_j except at j=N-1.
# This is a direct exact implementation, not a matrix-roundoff experiment.
losses = []
for dimension in (2, 4, 8, 16, 32, 64, 128):
    residuals = []
    for j in range(dimension):
        pushed = j + 1 if j + 1 < dimension else None
        popped = pushed - 1 if pushed is not None else None
        residuals.append(0 if popped == j else 1)
    assert sum(residuals) == 1
    losses.append({"dimension": dimension, "sum_squared_residual": sum(residuals),
                   "mean_squared_residual": sum(residuals) / dimension,
                   "failure_basis_index": dimension - 1})
report["truncated_shift_relation_loss"] = losses

dimension = 17
training_depth = 8
assert training_depth < dimension - 1
sample_failures = []
for j in range(training_depth + 1):
    pushed = j + 1 if j + 1 < dimension else None
    popped = pushed - 1 if pushed is not None else None
    if popped != j:
        sample_failures.append(j)
assert not sample_failures
state = 0
for _ in range(dimension):
    state = state + 1 if state is not None and state + 1 < dimension else None
assert state is None
report["held_out_capacity_witness"] = {
    "dimension": dimension, "observed_basis_indices": list(range(training_depth + 1)),
    "observed_relation_errors": 0, "held_out_action": "U repeated 17 times from e_0",
    "finite_result": "zero_invalid", "growing_result": "e_17",
    "conclusion": "zero finite-probe relation loss does not identify unbounded allocation"}
checkpoint("complete")
