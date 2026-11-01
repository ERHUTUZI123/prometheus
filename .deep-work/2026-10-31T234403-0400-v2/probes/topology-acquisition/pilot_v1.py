"""One bounded topology-acquisition pilot. Importing this file performs no work.

The task target occurs only inside TaskEnvironment. Learner decisions use scalar
rewards, measured permutations, owned polynomial coefficients, and charged work.
No successful critical loop is supplied. The external runner enforces isolation.
"""
from collections import deque
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np


OUT = Path(__file__).resolve().parent
BASE_U = 1.0 + 0.0j
MAX_SECONDS = 110.0
MAX_STEPS = 200000
MAX_DEPTH = 12
N_CIRCLE = 256
N_HANDLE = 128
N_MIGRATION = 64
TOL = 1e-11
LABEL_TOL = 1e-6
IDENTITY = (0, 1, 2, 3)
HELDOUT = (
    ("A", "C"),
    ("C", "A"),
    ("A", "C", "A-", "C-"),
    ("C", "A", "A", "C", "A-"),
)


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class PilotFailure(RuntimeError):
    pass


class Run:
    def __init__(self):
        self.start = time.monotonic()
        self.raw = (OUT / "raw.jsonl").open("x")
        self.cost = {
            "step_calls": 0, "accepted_vector_steps": 0,
            "accepted_branch_steps": 0, "subdivisions": 0,
            "corrector_iterations": 0, "F_scalar_evals": 0,
            "Fz_scalar_evals": 0, "Qu_scalar_evals": 0,
            "Qzz_scalar_evals": 0, "J_scalar_evals": 0,
            "critical_root_solves": 0, "critical_scalar_evals": 0,
            "verification_root_solves": 0, "verification_root_matches": 0,
            "path_executions": 0, "inverse_reset_executions": 0,
            "word_trials": 0, "task_queries": 0,
            "task_input_examples": 0, "failed_paths": 0,
            "physical_initial_branch_preparations": 4,
            "coefficient_migration_branch_trajectories": 0,
            "anchor_migration_branch_trajectories": 0,
            "prepared_input_episodes": 0,
            "path_model_segment_checks": 0,
            "certificate_calls": 0, "certificate_root_disks": 0,
            "certificate_path_disk_whole_path_comparisons": 0,
            "certificate_path_segment_disk_comparisons": 0,
            "certificate_seconds": 0.0,
            "virtual_model_newton_batches": 0,
            "virtual_model_branch_point_newton_updates": 0,
            "virtual_model_branch_points": 0,
            "bfs_graph_states_discovered": 0, "bfs_graph_states_expanded": 0,
            "bfs_graph_transitions_examined": 0,
            "table_compositions": 0, "table_entry_lookups": 0,
            "digital_rival_words": 0, "digital_rival_table_entry_lookups": 0,
        }
        self.metrics = {
            "minimum_branch_separation": None,
            "maximum_scale_invariant_correction": 0.0,
            "maximum_root_verifier_error": 0.0,
            "maximum_control_sensitivity": 0.0,
            "maximum_endpoint_error": 0.0,
        }
        self.last = None
        self.emit("serialization_smoke", config={
            "owned_relation": "(z-3)*(z**3-u)-epsilon",
            "relation_observation_labels": "none: coefficients are owned",
            "task_access": "scalar reward for four actually executed input cases",
            "max_seconds_internal": MAX_SECONDS, "max_step_calls": MAX_STEPS,
            "max_depth": MAX_DEPTH, "circle_segments": N_CIRCLE,
            "handle_segments": N_HANDLE, "migration_segments": N_MIGRATION,
            "initial_epsilon": 0.0, "growth_epsilon": 0.1,
            "heldout_words": HELDOUT,
        }, complex_serialization=np.array([1 + 2j]), cost=self.cost)

    def check(self):
        if time.monotonic() - self.start > MAX_SECONDS:
            raise PilotFailure("internal wall-time limit")
        if self.cost["step_calls"] > MAX_STEPS:
            raise PilotFailure("global vector step-call budget")

    def emit(self, kind, **fields):
        row = {"kind": kind, **json_safe(fields)}
        encoded = json.dumps(row, allow_nan=False, sort_keys=True)
        if json.loads(encoded)["kind"] != kind:
            raise PilotFailure("serialization round-trip failed")
        self.raw.write(encoded + "\n")
        self.raw.flush()

    def finish(self, status, **fields):
        summary = json_safe({
            "status": status, "seconds": time.monotonic() - self.start,
            "cost": self.cost, "metrics": self.metrics, **fields,
        })
        encoded = json.dumps(summary, allow_nan=False, indent=2, sort_keys=True)
        with (OUT / "summary.json").open("x") as output:
            output.write(encoded + "\n")
        self.emit("finished", **summary)
        self.raw.close()


class Tracker:
    def __init__(self, run):
        self.run = run

    def F(self, z, u, epsilon):
        self.run.cost["F_scalar_evals"] += int(np.size(z))
        return (z - 3) * (z ** 3 - u) - epsilon

    def Fz(self, z, u):
        self.run.cost["Fz_scalar_evals"] += int(np.size(z))
        return 4 * z ** 3 - 9 * z ** 2 - u

    def Qu(self, z):
        self.run.cost["Qu_scalar_evals"] += int(np.size(z))
        return 3 - z

    def Qzz(self, z):
        self.run.cost["Qzz_scalar_evals"] += int(np.size(z))
        return 12 * z ** 2 - 18 * z

    def observe(self, z, u, epsilon):
        if len(z) != 4 or not np.all(np.isfinite(z)):
            raise PilotFailure("nonfinite or wrong-size carried state")
        separation = min(abs(z[i] - z[j]) for i in range(4) for j in range(i))
        if not np.isfinite(separation) or separation <= 1e-8:
            raise PilotFailure("tracked branch separation exhausted")
        old = self.run.metrics["minimum_branch_separation"]
        self.run.metrics["minimum_branch_separation"] = (
            float(separation) if old is None else min(old, float(separation)))
        den = self.Fz(z, u)
        correction = float(np.max(np.abs(self.F(z, u, epsilon) / den)))
        sensitivity = float(np.max(np.abs(self.Qu(z) / den)))
        if not np.isfinite(correction + sensitivity):
            raise PilotFailure("nonfinite residual correction/sensitivity")
        self.run.metrics["maximum_scale_invariant_correction"] = max(
            self.run.metrics["maximum_scale_invariant_correction"], correction)
        self.run.metrics["maximum_control_sensitivity"] = max(
            self.run.metrics["maximum_control_sensitivity"], sensitivity)
        # This solve is a verifier ONLY. Its roots never replace a carried z,
        # choose a predictor/corrector branch, seed a path, or choose a word.
        oracle = np.roots([1.0, -3.0, 0.0, -u, 3 * u - epsilon])
        self.run.cost["verification_root_solves"] += 1
        self.run.cost["verification_root_matches"] += 16
        error = max(min(abs(a - b) for b in oracle) for a in z)
        self.run.metrics["maximum_root_verifier_error"] = max(
            self.run.metrics["maximum_root_verifier_error"], float(error))
        if error >= 1e-7:
            raise PilotFailure("independent root verifier rejected tracking")
        self.run.last = {"u": u, "epsilon": epsilon, "state": z.copy()}

    def step(self, z, u, epsilon, v, new_epsilon, depth=0):
        self.run.cost["step_calls"] += 1
        self.run.check()
        den = self.Fz(z, u)
        prediction = ((z - 3) * (v - u) + new_epsilon - epsilon) / den
        finite = np.all(np.isfinite(prediction))
        if not finite or np.max(np.abs(prediction)) > 0.08:
            return self.subdivide(z, u, epsilon, v, new_epsilon, depth,
                                  "predictor cap/nonfinite")
        trial = z + prediction
        good = False
        for _ in range(12):
            self.run.cost["corrector_iterations"] += 1
            change = self.F(trial, v, new_epsilon) / self.Fz(trial, v)
            trial = trial - change
            if np.all(np.isfinite(change)) and np.max(np.abs(change)) < TOL:
                good = True
                break
        if not good or not np.all(np.isfinite(trial)):
            return self.subdivide(z, u, epsilon, v, new_epsilon, depth,
                                  "corrector failure")
        self.observe(trial, v, new_epsilon)
        self.run.cost["accepted_vector_steps"] += 1
        self.run.cost["accepted_branch_steps"] += 4
        return trial

    def subdivide(self, z, u, epsilon, v, new_epsilon, depth, reason):
        if depth >= MAX_DEPTH:
            raise PilotFailure(reason + ": subdivision limit")
        self.run.cost["subdivisions"] += 1
        middle_u = (u + v) / 2
        middle_epsilon = (epsilon + new_epsilon) / 2
        middle_z = self.step(z, u, epsilon, middle_u, middle_epsilon, depth + 1)
        return self.step(middle_z, middle_u, middle_epsilon, v,
                         new_epsilon, depth + 1)

    def migrate(self, states, anchors, old_epsilon, new_epsilon, tag):
        # Semantic anchors are independently transported with the same flow;
        # their values are never replaced by a fresh polynomial root solve.
        self.run.emit("batch_start", batch=tag, old_epsilon=old_epsilon,
                      new_epsilon=new_epsilon, states=states, anchors=anchors)
        result = []
        for category, initial in (("state", states), ("anchor", anchors)):
            z = initial.copy()
            previous = old_epsilon
            for value in np.linspace(old_epsilon, new_epsilon, N_MIGRATION + 1)[1:]:
                z = self.step(z, BASE_U, previous, BASE_U, float(value))
                previous = float(value)
            field = ("coefficient_migration_branch_trajectories" if category == "state"
                     else "anchor_migration_branch_trajectories")
            self.run.cost[field] += 4
            result.append(z)
        self.run.emit("migration", tag=tag, old_epsilon=old_epsilon,
                      new_epsilon=new_epsilon, states=result[0], anchors=result[1])
        return result

    def path(self, state, path, epsilon, tag, save_trace=False, inverse=False):
        if abs(path[0] - BASE_U) > 1e-12 or abs(path[-1] - BASE_U) > 1e-12:
            raise PilotFailure("non-home closed path")
        self.run.cost["path_executions"] += 1
        if inverse:
            self.run.cost["inverse_reset_executions"] += 1
        z = state.copy()
        trace_z = [z.copy()] if save_trace else None
        previous = complex(path[0])
        try:
            for value in path[1:]:
                z = self.step(z, previous, epsilon, complex(value), epsilon)
                previous = complex(value)
                if save_trace:
                    # Declared polygon nodes are the fixed objective samples.
                    # Adaptive substeps are charged but do not change J's grid.
                    trace_z.append(z.copy())
        except Exception as error:
            self.run.cost["failed_paths"] += 1
            self.run.emit("path_failure", tag=tag, epsilon=epsilon,
                          error=repr(error), last_valid=self.run.last)
            raise
        trace = None if not save_trace else {
            "u": np.asarray(path, dtype=complex), "z": np.asarray(trace_z)}
        if trace is not None:
            self.run.emit("path_trace", tag=tag, epsilon=epsilon, **trace)
        return z, trace


def permutation(state, anchors, run):
    labels = tuple(int(np.argmin(np.abs(anchors - value))) for value in state)
    error = max(abs(state[i] - anchors[label]) for i, label in enumerate(labels))
    run.metrics["maximum_endpoint_error"] = max(
        run.metrics["maximum_endpoint_error"], float(error))
    if sorted(labels) != list(IDENTITY) or error >= LABEL_TOL:
        raise PilotFailure("endpoint is not a permutation of carried anchors")
    return labels


def compose(first, second, run):
    """Execute first, then second: second[first[i]]."""
    run.cost["table_compositions"] += 1
    run.cost["table_entry_lookups"] += 4
    return tuple(second[first[i]] for i in IDENTITY)


def inverse_permutation(p):
    return tuple(p.index(i) for i in IDENTITY)


def assert_home(state, anchors, run):
    if permutation(state, anchors, run) != IDENTITY:
        raise PilotFailure("inverse path failed to restore carried input states")


def old_path():
    result = np.exp(1j * np.linspace(0.0, 2 * np.pi, N_CIRCLE + 1))
    result[0] = result[-1] = BASE_U
    return result


def cluster_values(values):
    clusters = []
    for value in sorted(values, key=lambda x: (float(x.real), float(x.imag))):
        for cluster in clusters:
            if any(abs(value - member) <= 1e-5 * (1 + max(abs(value), abs(member)))
                   for member in cluster):
                cluster.append(complex(value))
                break
        else:
            clusters.append([complex(value)])
    return np.asarray([sum(cluster) / len(cluster) for cluster in clusters])


def point_segment_distance(point, start, end):
    difference = end - start
    if abs(difference) == 0:
        return float(abs(point - start))
    projection = np.clip(np.real((point - start) * np.conj(difference)) /
                         abs(difference) ** 2, 0.0, 1.0)
    return float(abs(point - start - projection * difference))


def compile_all(run, epsilon):
    run.check()
    critical_z = np.roots([3.0, -18.0, 27.0, 0.0, epsilon])
    run.cost["critical_root_solves"] += 1
    critical_u_raw = 4 * critical_z ** 3 - 9 * critical_z ** 2
    run.cost["critical_scalar_evals"] += 4
    centers = cluster_values(critical_u_raw)
    paths = {}
    run.emit("compiler_geometry", epsilon=epsilon, critical_z=critical_z,
             critical_u_raw=critical_u_raw, unique_critical_u=centers,
             duplicate_tolerance="1e-5*(1+max(abs(a),abs(b)))")
    for index, center in enumerate(centers):
        neighbors = [abs(center - other) for other in centers if other != center]
        radius = min(0.2, float(min(neighbors)) / 5) if neighbors else 0.2
        direction = BASE_U - center
        if abs(direction) <= radius or not np.isfinite(radius) or radius <= 0:
            raise PilotFailure("critical loop cannot have a regular home handle")
        entry = center + radius * direction / abs(direction)
        handle = np.linspace(BASE_U, entry, N_HANDLE + 1, dtype=complex)
        phase = np.angle(entry - center)
        loop = center + radius * np.exp(
            1j * (phase + np.linspace(0.0, 2 * np.pi, N_CIRCLE + 1)))
        loop[0] = loop[-1] = entry
        path = np.r_[handle, loop[1:], handle[-2::-1]]
        path[0] = path[-1] = BASE_U
        # This is a paid model-geometry check, not an exact certificate.
        clearance = float("inf")
        for a, b in zip(path[:-1], path[1:]):
            for critical in centers:
                run.cost["path_model_segment_checks"] += 1
                clearance = min(clearance, point_segment_distance(critical, a, b))
        if not np.isfinite(clearance) or clearance <= 1e-10:
            raise PilotFailure("compiled polygon touches a modeled critical value")
        tag = "L" + str(index)
        paths[tag] = path
        run.emit("compiled_path", epsilon=epsilon, name=tag, center=center,
                 radius=radius, model_clearance=clearance,
                 polyline_length=float(np.sum(np.abs(np.diff(path)))), path=path)
    return critical_z, paths


def execute_word(tracker, state, word, operations, epsilon, tag, inverse=False):
    z = state
    for index, letter in enumerate(word):
        path = operations[letter]["path"]
        z, _ = tracker.path(z, path, epsilon, tag + "/" + str(index) + letter,
                            inverse=inverse)
    return z


def inverse_word(word):
    return tuple(letter[:-1] if letter.endswith("-") else letter + "-"
                 for letter in reversed(word))


def measured_operation(tracker, state, anchors, path, epsilon, name, trace=False):
    run = tracker.run
    before = run.cost["accepted_branch_steps"]
    end, samples = tracker.path(state, path, epsilon, name, save_trace=trace)
    measured = permutation(end, anchors, run)
    forward_cost = run.cost["accepted_branch_steps"] - before
    back, _ = tracker.path(end, path[::-1], epsilon, name + "/inverse-reset",
                           inverse=True)
    assert_home(back, anchors, run)
    result = {"path": path, "permutation": measured,
              "forward_branch_steps": forward_cost,
              "polyline_length": float(np.sum(np.abs(np.diff(path))))}
    run.emit("measured_operation", epsilon=epsilon, name=name,
             permutation=measured, forward_branch_steps=forward_cost,
             trace_saved=trace)
    return back, result, samples


def add_inverses(operations):
    result = dict(operations)
    for name, operation in operations.items():
        result[name + "-"] = {
            **operation, "path": operation["path"][::-1],
            "permutation": inverse_permutation(operation["permutation"])}
    return result


def bfs_words(operations, run):
    queue = deque([(IDENTITY, ())])
    visited = {IDENTITY}
    run.cost["bfs_graph_states_discovered"] += 1
    while queue:
        current, word = queue.popleft()
        run.cost["bfs_graph_states_expanded"] += 1
        yield current, word
        for letter in sorted(operations):
            run.cost["bfs_graph_transitions_examined"] += 1
            effect = compose(current, operations[letter]["permutation"], run)
            if effect not in visited:
                visited.add(effect)
                run.cost["bfs_graph_states_discovered"] += 1
                queue.append((effect, word + (letter,)))


class TaskEnvironment:
    """The target is ENV-ONLY: no learner/helper reads its labels."""
    def reward(self, actual_labels):
        target = (1, 2, 3, 0)
        return sum(actual_labels[i] == target[i] for i in range(4)) / 4.0


def acquire_task(tracker, state, anchors, operations, epsilon, environment, phase):
    run = tracker.run
    all_words = list(bfs_words(operations, run))
    run.emit("batch_start", batch=phase + "/task-search",
             measured_group_size=len(all_words), operations={
                 key: value["permutation"] for key, value in operations.items()})
    success_word = None
    observed_rewards = []
    for predicted, word in all_words:
        run.check()
        run.cost["word_trials"] += 1
        run.cost["prepared_input_episodes"] += 4
        end = execute_word(tracker, state, word, operations, epsilon,
                           phase + "/task-word")
        actual = permutation(end, anchors, run)
        if actual != predicted:
            raise PilotFailure("measured composition disagrees with actual word")
        # Only this scalar crosses the environment/learner interface.
        reward = environment.reward(actual)
        run.cost["task_queries"] += 1
        run.cost["task_input_examples"] += 4
        observed_rewards.append(reward)
        run.emit("task_trial", phase=phase, epsilon=epsilon, word=word,
                 predicted_permutation=predicted, observed_permutation=actual,
                 scalar_reward=reward, task_examples=4)
        state = execute_word(tracker, end, inverse_word(word), operations,
                             epsilon, phase + "/task-inverse-reset", inverse=True)
        assert_home(state, anchors, run)
        if reward == 1.0:
            success_word = word
            break
    return state, {"success": success_word is not None,
                   "word": success_word, "group_size": len(all_words),
                   "rewards": observed_rewards}


def objective(tracker, trace, epsilon):
    z, u = trace["z"], trace["u"][:, None]
    den = tracker.Fz(z, u)
    qu = tracker.Qu(z)
    ratio = qu / den
    derivative = -1 / den ** 2 - qu * tracker.Qzz(z) / den ** 3
    tracker.run.cost["J_scalar_evals"] += int(z.size)
    values = np.abs(ratio) ** 2
    gradients = 2 * np.real(np.conj(ratio) * derivative)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradients)):
        raise PilotFailure("nonfinite conditioning objective/gradient")
    return values.ravel(), gradients.ravel()


def virtual_trace_shift(tracker, trace, epsilon, displacement):
    """Owned-model FD check at fixed observed controls; no physical reset.

    Each saved branch point is continued a tiny parameter distance on a virtual
    copy. This does not alter the physically carried state or its anchors.
    """
    run = tracker.run
    original = trace["z"]
    u = trace["u"][:, None]
    z = original + displacement / tracker.Fz(original, u)
    run.cost["virtual_model_branch_points"] += int(z.size)
    good = False
    for _ in range(12):
        run.check()
        run.cost["virtual_model_newton_batches"] += 1
        run.cost["virtual_model_branch_point_newton_updates"] += int(z.size)
        change = tracker.F(z, u, epsilon + displacement) / tracker.Fz(z, u)
        z = z - change
        if np.all(np.isfinite(change)) and np.max(np.abs(change)) < TOL:
            good = True
            break
    if not good or not np.all(np.isfinite(z)):
        raise PilotFailure("virtual finite-difference Newton continuation failed")
    nearest_separation = np.full(len(original), np.inf)
    for i in range(4):
        for j in range(i):
            nearest_separation = np.minimum(nearest_separation,
                                            np.abs(original[:, i] - original[:, j]))
    if np.any(np.max(np.abs(z - original), axis=1) >= nearest_separation / 4):
        raise PilotFailure("virtual finite-difference branch identification ambiguous")
    return {"u": trace["u"], "z": z}


def finite_difference_gradient(tracker, traces, epsilon):
    step = 1e-6
    mean_values = []
    start_cost = tracker.run.cost.copy()
    for displacement in (step, -step):
        values = []
        for trace in traces:
            shifted = virtual_trace_shift(tracker, trace, epsilon, displacement)
            values.append(objective(tracker, shifted, epsilon + displacement)[0])
        mean_values.append(float(np.mean(np.concatenate(values))))
    gradient = (mean_values[0] - mean_values[1]) / (2 * step)
    charged_difference = {
        key: tracker.run.cost[key] - start_cost[key] for key in start_cost
        if tracker.run.cost[key] != start_cost[key]}
    return gradient, {"h": step, "J_plus": mean_values[0],
                      "J_minus": mean_values[1], "gradient": gradient,
                      "virtual_model_only": True, "additional_cost": charged_difference}


def check_word(tracker, state, anchors, word, operations, epsilon, tag):
    prediction = IDENTITY
    for letter in word:
        prediction = compose(prediction, operations[letter]["permutation"], tracker.run)
    end = execute_word(tracker, state, word, operations, epsilon, tag)
    actual = permutation(end, anchors, tracker.run)
    if actual != prediction:
        raise PilotFailure("held-out retained word changed its measured semantics")
    restored = execute_word(tracker, end, inverse_word(word), operations, epsilon,
                            tag + "/inverse-reset", inverse=True)
    assert_home(restored, anchors, tracker.run)
    tracker.run.emit("word_check", tag=tag, epsilon=epsilon, word=word,
                     predicted_permutation=prediction, observed_permutation=actual)
    return restored


def compiled_rivals(run, learned_word, operations):
    outputs = []
    for kind, word in (("learned_task_word", tuple(learned_word)),
                       *(("heldout", word) for word in HELDOUT)):
        run.cost["digital_rival_words"] += 1
        result = IDENTITY
        for letter in word:
            result = compose(result, operations[letter]["permutation"], run)
            run.cost["digital_rival_table_entry_lookups"] += 4
        outputs.append({"kind": kind, "word": word, "outputs": result,
                        "logical_table_entry_lookups": 4 * len(word)})
    result = {
        "same_actuator_conventional_constructor": {
            "correspondence": "identical owned-polynomial critical-loop compiler, measured permutation library, and BFS task search implemented by this pilot",
            "acquisition_information_and_paid_work": "shared exactly with the pilot",
            "independently_reexecuted": False,
            "performance_advantage_claim": False,
        },
        "digital_compiled_fsm": {
            "acquisition_information_and_paid_work": "same already paid measured A/C library; no new labels",
            "outputs": outputs, "stored_directed_permutation_entries": 16,
            "logical_table_entry_lookups": sum(item["logical_table_entry_lookups"]
                                               for item in outputs),
            "scope": "digital outputs only; actual continuous actuation still requires charged paths",
        },
    }
    run.emit("explicit_compiled_rivals", **result)
    return result


def pilot(run):
    tracker = Tracker(run)
    omega = np.exp(2j * np.pi / 3)
    states = np.asarray([1 + 0j, omega, omega ** 2, 3 + 0j])
    anchors = states.copy()
    epsilon = 0.0
    tracker.observe(states, BASE_U, epsilon)
    environment = TaskEnvironment()
    A = old_path()
    states, op_a0, _ = measured_operation(
        tracker, states, anchors, A, epsilon, "epsilon0/A")
    if op_a0["permutation"] != (1, 2, 0, 3):
        raise PilotFailure("supplied old A did not realize its old three-cycle")
    critical0, paths0 = compile_all(run, epsilon)
    zero_library = {"A": op_a0}
    for name, path in paths0.items():
        states, operation, _ = measured_operation(
            tracker, states, anchors, path, epsilon, "epsilon0/" + name)
        zero_library[name] = operation
    states, baseline = acquire_task(tracker, states, anchors,
                                   add_inverses(zero_library), epsilon,
                                   environment, "epsilon0")
    run.emit("no_coupling_result", **baseline, critical_z_reused=critical0)
    if baseline["success"]:
        raise PilotFailure("disconnected control unexpectedly solved the task")

    states, anchors = tracker.migrate(states, anchors, epsilon, 0.1, "growth")
    epsilon = 0.1
    assert_home(states, anchors, run)
    states, op_a, trace_a = measured_operation(
        tracker, states, anchors, A, epsilon, "epsilon.1/A", trace=True)
    if op_a["permutation"] != op_a0["permutation"]:
        raise PilotFailure("old A changed during structural growth")
    critical_z, candidates = compile_all(run, epsilon)
    measured = {}
    for name, path in candidates.items():
        states, operation, _ = measured_operation(
            tracker, states, anchors, path, epsilon, "epsilon.1/" + name)
        measured[name] = operation
    choices = []
    for name, operation in measured.items():
        p = operation["permutation"]
        moved = [i for i in IDENTITY if p[i] != i]
        if len(moved) == 2 and 3 in moved and all(p[p[i]] == i for i in IDENTITY):
            choices.append((operation["forward_branch_steps"],
                            operation["polyline_length"], name))
    if not choices:
        raise PilotFailure("generic compiler found no measured old/new transposition")
    _, _, chosen = min(choices)
    C = measured[chosen]["path"]
    # Re-execute selected C to save its fixed declared-node trace for J.
    states, op_c, trace_c = measured_operation(
        tracker, states, anchors, C, epsilon, "epsilon.1/C", trace=True)
    operations = add_inverses({"A": op_a, "C": op_c})
    run.emit("published_library", epsilon=epsilon, chosen_candidate=chosen,
             selection_rule="lowest measured forward branch-steps, then length, then name",
             operations={key: value["permutation"] for key, value in operations.items()},
             diagnostic_only_candidates={key: value["permutation"]
                                         for key, value in measured.items()})
    states, task = acquire_task(tracker, states, anchors, operations, epsilon,
                               environment, "epsilon.1")
    if not task["success"] or task["group_size"] != 24:
        raise PilotFailure("published learned library did not acquire the task/S4")
    run.emit("structural_acceptance", accepted=True,
             criterion="zero-coupling task failure and coupled actual task success",
             no_coupling=baseline, coupled=task,
             statement="supplied mutation grammar; acquired paths/library/word")
    for index, word in enumerate(HELDOUT):
        states = check_word(tracker, states, anchors, word, operations, epsilon,
                            "pre-update/heldout" + str(index))
    rivals = compiled_rivals(run, task["word"], operations)

    values_a, grads_a = objective(tracker, trace_a, epsilon)
    values_c, grads_c = objective(tracker, trace_c, epsilon)
    J_before = float(np.mean(np.r_[values_a, values_c]))
    gradient = float(np.mean(np.r_[grads_a, grads_c]))
    finite_difference, fd_result = finite_difference_gradient(
        tracker, (trace_a, trace_c), epsilon)
    fd_result["analytic_gradient"] = gradient
    fd_result["absolute_error"] = abs(finite_difference - gradient)
    fd_result["relative_error_with_unit_floor"] = (
        abs(finite_difference - gradient) / max(1.0, abs(gradient)))
    fd_result["tolerance"] = 1e-4 * max(1.0, abs(gradient), abs(finite_difference))
    fd_result["passed"] = fd_result["absolute_error"] <= fd_result["tolerance"]
    run.emit("gradient_finite_difference_check", **fd_result)
    if not fd_result["passed"]:
        raise PilotFailure("fixed finite-difference gradient tolerance failed")
    delta = -0.0005 * gradient / (1 + abs(gradient))
    proposed = float(epsilon + delta)
    bounds = (Fraction(199, 2000), Fraction(201, 2000))
    if not (bounds[0] <= Fraction(float(epsilon)) <= bounds[1] and
            bounds[0] <= Fraction(proposed) <= bounds[1]):
        raise PilotFailure("float epsilon/update outside exact certificate interval")
    run.emit("proposed_learning_step", epsilon=epsilon, J_before=J_before,
             gradient=gradient, delta=delta, proposed_epsilon=proposed,
             objective="mean |Qu/Qz|^2 on fixed actual A/C polygon-node traces",
             note="conditioning derivative; not a discrete task-reward gradient")
    # Root-owned module. This call precedes the native parameter update.
    import certificate
    cert_start = time.monotonic()
    run.cost["certificate_calls"] += 1
    certificate_result = certificate.certify(critical_z, {"A": A, "C": C})
    run.cost["certificate_seconds"] += time.monotonic() - cert_start
    run.cost["certificate_root_disks"] += 4
    run.cost["certificate_path_disk_whole_path_comparisons"] += 8
    run.cost["certificate_path_segment_disk_comparisons"] += 4 * ((len(A) - 1) + (len(C) - 1))
    run.emit("retention_certificate", result=certificate_result,
             exact_start_fraction=str(Fraction(float(epsilon))),
             exact_proposed_fraction=str(Fraction(proposed)))
    run.check()
    if not certificate_result.get("passed", False):
        raise PilotFailure("exact certificate failed: native retention update not executed")
    states, anchors = tracker.migrate(states, anchors, epsilon, proposed,
                                      "one-conditioning-learning-step")
    epsilon = proposed
    assert_home(states, anchors, run)
    updated_traces = {}
    for name in ("A", "C"):
        states, observed, trace = measured_operation(
            tracker, states, anchors, operations[name]["path"], epsilon,
            "post-update/" + name, trace=True)
        if observed["permutation"] != operations[name]["permutation"]:
            raise PilotFailure("certified fixed-path operation changed after update")
        updated_traces[name] = trace
    states = check_word(tracker, states, anchors, tuple(task["word"]), operations,
                        epsilon, "post-update/learned-task-word")
    for index, word in enumerate(HELDOUT):
        states = check_word(tracker, states, anchors, word, operations, epsilon,
                            "post-update/heldout" + str(index))
    new_values = [objective(tracker, updated_traces[name], epsilon)[0]
                  for name in ("A", "C")]
    J_after = float(np.mean(np.concatenate(new_values)))
    run.emit("learning_result", epsilon=epsilon, J_before=J_before, J_after=J_after,
             observed_improvement=J_after < J_before, gradient=gradient,
             fixed_paths=True, new_task_reward_queries=0)
    storage = {
        "published_directed_paths": 4,
        "published_distinct_control_nodes": int(len(A) + len(C)),
        "published_permutation_entries_with_inverses": 16,
        "relation_complex_coefficients": 5,
        "carried_state_complex_scalars": 4, "decoder_anchor_complex_scalars": 4,
        "diagnostic_candidate_paths": len(measured),
        "diagnostic_candidate_control_nodes": sum(len(v["path"]) for v in measured.values()),
    }
    return {"no_coupling": baseline, "acquired_task": task,
            "chosen_candidate": chosen, "epsilon_after": epsilon,
            "J_before": J_before, "J_after": J_after,
            "gradient_finite_difference": fd_result,
            "compiled_rivals": rivals,
            "certificate": certificate_result, "storage_proxies": storage,
            "scope": "owned designed growth; learned path effects/library/task word; one conditioning update; no neural or algorithmic advantage claim"}


def main():
    run = Run()
    try:
        result = pilot(run)
    except Exception as error:
        run.emit("failure", error=repr(error), last_valid=run.last,
                 retention_claim="unverified on failure; inspect prior certificate and traces")
        run.finish("failed", error=repr(error), last_valid=run.last)
        raise
    run.finish("completed", **result)


if __name__ == "__main__":
    main()
