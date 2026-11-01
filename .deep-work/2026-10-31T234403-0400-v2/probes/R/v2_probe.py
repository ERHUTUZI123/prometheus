"""R v2 causal mechanism check, not neural acquisition or a speed benchmark."""
import itertools
import json
from pathlib import Path
from fractions import Fraction
from v2_core import (Span, ZERO, IDENTITY, matrix, unit, add, scale, mul,
                     algebra, edit_closure, compile_tables, pair_product)

report = {"status": "EXPLORATORY", "neural_acquisition_executed": False,
          "claim": "composition can require a new transported credit channel",
          "failures": []}


def checkpoint(phase):
    path = Path("v2-results.json")
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    assert json.loads(path.read_text()) == report
    print(json.dumps({"phase": phase, "report": report}, sort_keys=True), flush=True)


# Task loss: 0.5 * ((X B)[0,2] - 1)^2 at B=0.
# Derivative wrt each B_ij is computed by the actual product with unit_ij.
X = unit(0, 1)
B_initial = ZERO
gradient = tuple(-mul(X, unit(i, j))[2]
                 for i in range(3) for j in range(3))
chosen_index = max(range(9), key=lambda i: abs(gradient[i]))
chosen_edit = scale(unit(chosen_index // 3, chosen_index % 3), -gradient[chosen_index])
assert chosen_edit == unit(1, 2)
old_carrier = Span()
old_carrier.append(chosen_edit)
transported = mul(X, chosen_edit)
assert old_carrier.coordinates(transported) is None
naive_projected_derivative = ZERO  # E13 is orthogonal to initial span{E23}.
exact_output_derivative = transported[2]
assert exact_output_derivative == 1 and naive_projected_derivative[2] == 0

D = matrix(((1, 0, 0), (0, 2, 0), (0, 0, 3)))
current_a, current_a_products = algebra((D, X))
carrier, closure_products = edit_closure((D, X), (chosen_edit,))
assert len(current_a.basis) == 4
assert len(carrier.basis) == 2
assert carrier.coordinates(unit(0, 2)) is not None
report["birth"] = {"dense_gradient_entries_inspected": 9,
                   "chosen_edit_coordinate_zero_based": [1, 2],
                   "execution_algebra_dimension_before_update": len(current_a.basis),
                   "initial_credit_dimension": 1,
                   "closed_credit_dimension": len(carrier.basis),
                   "derived_channel_zero_based": [0, 2],
                   "exact_output_directional_derivative": str(exact_output_derivative),
                   "fixed_carrier_output_derivative": "0",
                   "closure_matrix_products": closure_products,
                   "execution_closure_matrix_products": current_a_products}
checkpoint("birth_and_missing_credit")

# A single legal task-gradient step B <- B - gradient reaches B=E23.
B = chosen_edit
assert mul(X, B)[2] == 1
new_a, new_a_products = algebra((D, X, B))
stable_m, new_m_products = edit_closure((D, X, B), (chosen_edit,))
assert len(new_a.basis) == 6 and len(stable_m.basis) == 2
assert new_a.coordinates(unit(0, 2)) is not None
tables = compile_tables(new_a, stable_m)
report["finite_update"] = {"task_loss_before": "1/2", "task_loss_after": "0",
                           "learning_rate": "1", "execution_algebra_dimension": 6,
                           "credit_dimension": 2,
                           "execution_closure_matrix_products": new_a_products,
                           "credit_closure_matrix_products": new_m_products,
                           "table_entries_compiled": 6 * 6 + 2 * 6 * 2}
checkpoint("finite_update_and_tables")

names = ("D", "X", "B")
operators = {"D": D, "X": X, "B": B}
edits = {"D": ZERO, "X": ZERO, "B": chosen_edit}
paired_letters = {name: (new_a.coordinates(operators[name]),
                         stable_m.coordinates(edits[name])) for name in names}
identity_pair = (new_a.coordinates(IDENTITY), stable_m.coordinates(ZERO))
checked = 0
nonzero_derivatives = 0
for length in range(7):
    for word in itertools.product(names, repeat=length):
        pair = identity_pair
        # Independent dense full forward AD reference; retains every matrix entry.
        dense_value, dense_derivative = IDENTITY, ZERO
        for name in word:
            pair = pair_product(paired_letters[name], pair, tables, 6, 2)
            dense_derivative = add(mul(edits[name], dense_value),
                                   mul(operators[name], dense_derivative))
            dense_value = mul(operators[name], dense_value)
        actual_value = new_a.reconstruct(pair[0])
        actual_derivative = stable_m.reconstruct(pair[1])
        if actual_value != dense_value or actual_derivative != dense_derivative:
            report["failures"].append({"word": list(word), "phase": "credit_reference"})
            checkpoint("failure")
            raise AssertionError("compiled carrier differs from exact dense forward AD")
        checked += 1
        nonzero_derivatives += int(any(dense_derivative))
report["composition_check"] = {"all_words_max_length": 6,
                               "words_checked": checked,
                               "words_with_nonzero_derivative": nonzero_derivatives,
                               "matched_dense_forward_ad_exactly": True,
                               "resource_advantage_claimed": False}
checkpoint("exact_reference_complete")

# A new reverse interaction destroys the compactness: the generated algebra is full.
reverse = unit(2, 0)
full_a, full_a_products = algebra((X, B, reverse))
full_m, full_m_products = edit_closure((X, B, reverse), (chosen_edit,))
assert len(full_a.basis) == 9 and len(full_m.basis) == 9
report["negative_full_closure"] = {"additional_interaction_zero_based": [2, 0],
                                    "execution_dimension": 9, "credit_dimension": 9,
                                    "execution_closure_matrix_products": full_a_products,
                                    "credit_closure_matrix_products": full_m_products,
                                    "conclusion": "exact compact edit closure can disappear"}
checkpoint("complete")
