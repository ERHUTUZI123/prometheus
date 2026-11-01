"""Native expert execution with transactional promotion of learned joint maps.

No teacher operator, target program or correct token/expert correspondence is
input to materialize. It compiles the model's OWN graphs, which may be wrong.
No state growth, experiment, SSH, training or work at module import.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate.
"""
import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from topology_model import TopologyExecutor, assignment, braid_word


class _PromotionFailure(RuntimeError):
    pass


class SemanticPromotionModel(TopologyExecutor):
    def __init__(self, *args, max_bank=24, condition_limit=1e8,
                 fit_tolerance=1e-5, minimum_margin=1., head_scale=4.,
                 expert_scale=8., **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= self.n_ops <= max_bank <= 24:
            raise ValueError("expert bank must contain1..24 entries")
        self.max_bank = int(max_bank)
        self.condition_limit = float(condition_limit)
        self.fit_tolerance = float(fit_tolerance)
        self.minimum_margin = float(minimum_margin)
        self.head_scale, self.expert_scale = float(head_scale), float(expert_scale)
        if min(self.condition_limit, self.fit_tolerance, self.minimum_margin,
               self.head_scale, self.expert_scale) <= 0:
            raise ValueError("positive numerical admission settings required")
        self.compilation = "component"
        self.materialization_history = []

    @torch.no_grad()
    def _graph_with_mode(self, task_id, token_ids, mode):
        requested = self.compilation
        try:
            self.compilation = mode
            # Call the BASE implementation, never self.graph: the latter is
            # intentionally always native, including inside the default guard.
            graph = super().graph(task_id, tuple(token_ids))
        finally:
            self.compilation = requested
        graph["execution_semantics"] = ("native_component_experts" if mode == "component"
                                        else "joint_soft_candidate_only")
        return graph

    @torch.no_grad()
    def graph(self, task_id, token_ids):
        return self._graph_with_mode(task_id, token_ids, "component")

    @torch.no_grad()
    def joint_graph(self, task_id, token_ids):
        return self._graph_with_mode(task_id, token_ids, "joint")

    @torch.no_grad()
    def hard_eval(self, tokens, starts, task_ids):
        # Base hard_eval calls this subclass's always-native graph().
        return super().hard_eval(tokens, starts, task_ids)

    @torch.no_grad()
    def proposal_eval(self, tokens, starts, task_ids):
        predictions = torch.empty_like(starts)
        for task in task_ids.unique().tolist():
            selected = task_ids == task
            graph = self.joint_graph(task, range(1, self.vocab + 1))
            state = graph["enc"][starts[selected].cpu().numpy()].copy()
            words = tokens[selected].cpu().numpy()
            for position in range(words.shape[1]):
                for token, transition in graph["trans"].items():
                    active = words[:, position] == token
                    state[active] = transition[state[active]]
            predictions[selected] = torch.as_tensor(graph["dec"][state], device=starts.device)
        logits = self.encoder.new_full((len(starts), self.n), -30.)
        return logits.scatter_(1, predictions[:, None], 0.)

    @staticmethod
    def _permutation_key(permutation, dim):
        permutation = np.asarray(permutation, dtype=np.int64)
        if permutation.shape != (dim,) or not np.array_equal(np.sort(permutation), np.arange(dim)):
            raise _PromotionFailure("learned candidate is not a full latent permutation")
        return tuple(int(value) for value in permutation)

    @staticmethod
    def _class_record(index, permutation, born):
        p = np.asarray(permutation, dtype="<i8")
        return dict(expert=int(index), born=bool(born), permutation=p.tolist(),
                    class_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
                    canonical_braid_length=len(braid_word(p)))

    def _old_domains(self, stage, old_alphabets):
        domains = {}
        if old_alphabets is not None:
            items = old_alphabets.items() if isinstance(old_alphabets, dict) else enumerate(old_alphabets)
            for task, tokens in items:
                task = int(task)
                if task < stage:
                    domains.setdefault(task, set()).update(int(token) for token in tokens)
        # Never omit a protected old reference just because its alphabet was
        # absent from the caller's convenience list.
        for reference in self.reference:
            task = int(reference["task"])
            if task >= stage:
                raise _PromotionFailure("materialize must precede consolidation of its current task")
            domains.setdefault(task, set()).update(reference["tokens"])
        for task, tokens in domains.items():
            if not 0 <= task < self.n_tasks or any(not 1 <= t <= self.vocab for t in tokens):
                raise _PromotionFailure("invalid old learned domain")
        return {task: tuple(sorted(tokens)) for task, tokens in sorted(domains.items()) if tokens}

    @torch.no_grad()
    def materialize(self, stage, token_ids, old_alphabets=None):
        started = time.perf_counter()
        original_ops, original_call, original_count = self.operations, self.call, self.n_ops
        original_flags = {name: p.requires_grad for name, p in self.named_parameters()}
        costs = dict(native_graphs=0, joint_graphs=0, bank_assignments=0,
                     graph_bank_assignments=0, joint_transition_assignments=0,
                     learned_target_pairs=0, svd_calls=0, least_squares_calls=0,
                     certificate_checks=0, certificate_relation_pairs=0,
                     expert_classes_born=0, original_bank_size=original_count)
        report = dict(stage=int(stage), success=False, accepted=False, materialized=False,
                      target_source="old CURRENT native graphs + current learned joint candidate",
                      teacher_operations_used=False, teacher_correctness_not_checked=True,
                      state_growth=False, all_parameters_trainable_on_success=True,
                      optimizer_rebuild_required=False, costs=costs)
        try:
            stage = int(stage)
            token_ids = tuple(sorted(set(int(t) for t in token_ids)))
            if not 0 <= stage < self.n_tasks or not token_ids:
                raise _PromotionFailure("invalid stage or empty current alphabet")
            if any(not 1 <= token <= self.vocab for token in token_ids):
                raise _PromotionFailure("current token outside model vocabulary")
            old_domains = self._old_domains(stage, old_alphabets)
            old_graphs = {}
            for task, alphabet in old_domains.items():
                old_graphs[task] = self.graph(task, alphabet)
                costs["native_graphs"] += 1
                costs["graph_bank_assignments"] += self.n_ops
            # Current soft proposals for OLD tasks are deliberately never read.
            current_candidate = self.joint_graph(stage, token_ids)
            costs["joint_graphs"] += 1
            costs["graph_bank_assignments"] += self.n_ops
            costs["joint_transition_assignments"] += len(token_ids)
            for reference in self.reference:
                native = old_graphs[int(reference["task"])]
                okay, pairs = self.related(reference, native)
                costs["certificate_checks"] += 1
                costs["certificate_relation_pairs"] += pairs
                if not okay:
                    raise _PromotionFailure("pre-existing native graph already violates an old reference")

            permutations = [np.arange(self.d, dtype=np.int64)]
            permutations.extend(assignment(logits) for logits in self.operations)
            costs["bank_assignments"] += original_count
            classes = {}
            for index, permutation in enumerate(permutations):
                classes.setdefault(self._permutation_key(permutation, self.d), index)
            targets, graphs = {}, list(old_graphs.values()) + [current_candidate]
            births = []
            for graph in graphs:
                for token in graph["tokens"]:
                    key = self._permutation_key(graph["trans"][token], self.d)
                    if key not in classes:
                        if len(permutations) - 1 >= self.max_bank:
                            raise _PromotionFailure("24-expert bank cap; no state/bank-cap growth authorized")
                        classes[key] = len(permutations)
                        permutations.append(np.asarray(key, dtype=np.int64))
                        births.append(self._class_record(classes[key], key, True))
                    pair = int(graph["task"]), int(token)
                    if pair in targets and targets[pair] != classes[key]:
                        raise _PromotionFailure("inconsistent learned graph targets for one task/token")
                    targets[pair] = classes[key]
            new_count = len(permutations) - 1
            pairs = sorted(targets)
            costs["learned_target_pairs"] = len(pairs)
            costs["proposed_new_classes"] = len(births)
            device, dtype = original_call.weight.device, original_call.weight.dtype
            tokens = torch.tensor([token for _, token in pairs], device=device, dtype=torch.long)
            tasks = torch.tensor([task for task, _ in pairs], device=device, dtype=torch.long)
            hidden = torch.tanh(self.token(tokens) + self.task(tasks))
            features = np.concatenate([hidden.cpu().double().numpy(), np.ones((len(pairs), 1))], axis=1)
            costs["cpu_double_feature_bytes"] = features.nbytes
            singular = np.linalg.svd(features, compute_uv=False)
            costs["svd_calls"] += 1
            tolerance = max(features.shape) * np.finfo(np.float64).eps * singular[0]
            rank = int(np.count_nonzero(singular > tolerance))
            condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
            report["fit"] = dict(rows=len(pairs), features=features.shape[1], rank=rank,
                                 full_row_rank=rank == len(pairs),
                                 condition=condition if np.isfinite(condition) else None,
                                 condition_limit=self.condition_limit, target_margin=2 * self.head_scale)
            # No fictional new hidden states/features are inserted to make the
            # desired head fit. A singular/ill-conditioned representation fails.
            if rank < len(pairs) or not np.isfinite(condition) or condition > self.condition_limit:
                raise _PromotionFailure("observed hidden features lack a well-conditioned full-row-rank fit")
            desired = np.full((len(pairs), self.slots * (new_count + 1)), -self.head_scale)
            costs["cpu_double_target_bytes"] = desired.nbytes
            desired_calls = np.zeros((len(pairs), self.slots), dtype=np.int64)
            desired_calls[:, 0] = [targets[pair] for pair in pairs]
            for row in range(len(pairs)):
                for slot in range(self.slots):
                    desired[row, slot * (new_count + 1) + desired_calls[row, slot]] = self.head_scale
            solution, _, solved_rank, _ = np.linalg.lstsq(features, desired, rcond=None)
            costs["least_squares_calls"] += 1
            residual = float(np.max(np.abs(features @ solution - desired)))
            report["fit"].update(solved_rank=int(solved_rank),
                                 cpu_double_max_residual=residual if np.isfinite(residual) else None)
            if solved_rank < len(pairs) or not np.isfinite(residual) or residual > self.fit_tolerance:
                raise _PromotionFailure("CPU-double learned-graph head fit residual exceeded tolerance")

            # Existing experts retain their OWN learned permutation classes;
            # only their sharpness changes. Appended rows are newly born classes.
            used = sorted(set(targets.values()) - {0})
            operation_values = original_ops.detach().clone()
            if new_count > original_count:
                operation_values = torch.cat([operation_values, operation_values.new_zeros(
                    (new_count - original_count, self.d, self.d))], dim=0)
            for expert in used:
                operation_values[expert - 1].fill_(-self.expert_scale)
                permutation = torch.as_tensor(permutations[expert], device=operation_values.device)
                operation_values[expert - 1, torch.arange(self.d, device=permutation.device), permutation] = self.expert_scale
            # Construct deterministically: random Linear initialization would
            # needlessly consume training RNG even in a rolled-back attempt.
            new_call = copy.deepcopy(original_call)
            new_call.out_features = self.slots * (new_count + 1)
            new_call.weight = nn.Parameter(torch.as_tensor(solution[:-1].T, device=device, dtype=dtype).clone())
            new_call.bias = nn.Parameter(torch.as_tensor(solution[-1], device=device, dtype=dtype).clone())
            self.operations, self.call, self.n_ops = nn.Parameter(operation_values), new_call, new_count

            actual_logits = self.calls(tokens[:, None], tasks)[:, 0]
            expected = torch.as_tensor(desired_calls, device=device)
            actual = actual_logits.argmax(-1)
            winning = actual_logits.gather(-1, expected[..., None]).squeeze(-1)
            competing = actual_logits.clone().scatter_(-1, expected[..., None], -torch.inf).amax(-1)
            actual_margin = float((winning - competing).amin().item())
            report["fit"].update(actual_dtype=str(dtype),
                                 actual_minimum_margin=actual_margin if np.isfinite(actual_margin) else None,
                                 required_minimum_margin=self.minimum_margin)
            if (not torch.isfinite(actual_logits).all() or not torch.equal(actual, expected)
                    or not np.isfinite(actual_margin) or actual_margin < self.minimum_margin):
                raise _PromotionFailure("installed learned head failed actual-device routing/margin check")

            certificates = []
            for before in graphs:
                after = self.graph(before["task"], before["tokens"])
                costs["native_graphs"] += 1
                costs["graph_bank_assignments"] += self.n_ops
                exact = (np.array_equal(before["enc"], after["enc"])
                         and np.array_equal(before["dec"], after["dec"])
                         and all(np.array_equal(before["trans"][token], after["trans"][token])
                                 for token in before["tokens"]))
                related, count = self.related(before, after)
                costs["certificate_checks"] += 1
                costs["certificate_relation_pairs"] += count
                certificates.append(dict(task=before["task"], tokens=list(before["tokens"]),
                    source="current_joint_candidate" if before["task"] == stage else "old_CURRENT_native",
                    exact_full_latent_transition_equality=bool(exact), related=bool(related),
                    relation_pairs=count, all_words_over_declared_alphabet=True))
                if not exact or not related:
                    raise _PromotionFailure("native post-promotion graph differs from its learned source")
            for reference in self.reference:
                current = self.graph(reference["task"], reference["tokens"])
                costs["native_graphs"] += 1
                costs["graph_bank_assignments"] += self.n_ops
                okay, count = self.related(reference, current)
                costs["certificate_checks"] += 1
                costs["certificate_relation_pairs"] += count
                if not okay:
                    raise _PromotionFailure("post-promotion native graph violates protected old relation")
            for parameter in self.parameters():
                parameter.requires_grad_(True)
            costs.update(final_bank_size=self.n_ops, expert_classes_born=len(births),
                         rewritten_head_scalars=self.call.weight.numel() + self.call.bias.numel(),
                         expert_parameter_bytes=self.operations.numel() * self.operations.element_size(),
                         head_parameter_bytes=sum(p.numel() * p.element_size() for p in self.call.parameters()))
            report.update(success=True, accepted=True, materialized=True, status="NATIVE_SEMANTIC_PROMOTION_ACCEPTED",
                          optimizer_rebuild_required=True, old_parameter_objects_replaced=True,
                          born_classes=births, reused_experts=sorted(set(targets.values()) - {r["expert"] for r in births}),
                          sharpened_to_own_classes=[self._class_record(k, permutations[k], k > original_count) for k in used],
                          learned_call_targets=[dict(task=task, token=token, calls=desired_calls[i].tolist())
                                                for i, (task, token) in enumerate(pairs)],
                          certificates=certificates, native_execution="one whole expert followed by skips",
                          terminal_correctness="not inferred; evaluate candidate/native task performance separately")
        except BaseException as error:
            # Restore the actual original Parameter/module objects, so a failed
            # transaction cannot strand an existing optimizer on stale weights.
            self.operations, self.call, self.n_ops = original_ops, original_call, original_count
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(original_flags[name])
            if not isinstance(error, Exception):
                raise
            report.update(status="FAILED_CLOSED_ROLLED_BACK", reason=str(error),
                          error_type=type(error).__name__, original_parameter_objects_restored=True,
                          optimizer_rebuild_required=False)
        costs["seconds"] = time.perf_counter() - started
        costs["total_completed_hungarian_assignments"] = (costs["bank_assignments"]
            + costs["graph_bank_assignments"] + costs["joint_transition_assignments"])
        self.materialization_history.append(copy.deepcopy(report))
        return report

    def extra_state_dict(self):
        extra = super().extra_state_dict()
        extra.update(n_ops=self.n_ops, slots=self.slots, emb_dim=self.call.in_features,
                     native_execution="component", proposal_execution="joint",
                     max_bank=self.max_bank, materialization_history=copy.deepcopy(self.materialization_history),
                     promotion_settings=dict(condition_limit=self.condition_limit,
                         fit_tolerance=self.fit_tolerance, minimum_margin=self.minimum_margin,
                         head_scale=self.head_scale, expert_scale=self.expert_scale),
                     optimizer_rebuild_after_promotion=True)
        return extra

    def load_extra_state_dict(self, extra):
        self.reference = copy.deepcopy(extra.get("reference", []))
        self.temperature = float(extra.get("temperature", self.temperature))
        self.stage = int(extra.get("stage", self.stage))
        self.protection = bool(extra.get("protection", self.protection))
        self.credit = extra.get("credit", self.credit)
        self.compilation = "component"
        self.materialization_history = copy.deepcopy(extra.get("materialization_history", []))
        if "n_ops" in extra and int(extra["n_ops"]) != self.n_ops:
            raise ValueError("load model state_dict (dynamic bank shape) before extra state")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Standard checkpoints retain dynamic bank size even if a caller built
        # the model with the initial number of experts. No new state dimension.
        if "operations" in state_dict:
            count, a, b = state_dict["operations"].shape
            if a != self.d or b != self.d or not 1 <= count <= self.max_bank:
                raise ValueError("checkpoint bank/state dimensions exceed declared bounds")
            expected = self.slots * (count + 1)
            if tuple(state_dict["call.weight"].shape) != (expected, self.call.in_features):
                raise ValueError("checkpoint head shape disagrees with dynamic bank")
            if count != self.n_ops:
                self.operations = nn.Parameter(self.operations.new_empty((count, self.d, self.d)))
                self.call = nn.Linear(self.call.in_features, expected, device=self.call.weight.device,
                                      dtype=self.call.weight.dtype)
                self.n_ops = count
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


SemanticPromotionExecutor = SemanticPromotionModel
