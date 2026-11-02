"""DEN-RNN: explicit sequence adaptation of Yoon et al. ICLR2018, not official DEN.

Implements sparse-head selection, shared-subnetwork retraining, loss-triggered
group-sparse expansion, incoming-weight drift splitting and timestamped old-task
inference. The feedforward author code is read-only evidence, never imported.
Input consists ONLY of public tokens/start/taskID; supervision is terminal CE.
No profile constructor, teacher operation, hidden-state label or replay is used.

Training API (fixed total updates; every phase counts):
    model.set_stage(t, total_steps=600)
    optimizer = Adam(model.parameters(), lr=...)
    loss = CE(model(tokens, starts, task_ids), targets) + model.regularization_loss()
    loss.backward(); model.mask_gradients_(); optimizer.step()
    event = model.after_optimizer_step_(float(CE.detach()), step+1)
    if event['optimizer_rebuild_required']: optimizer = Adam(model.parameters(), ...)
Clip gradients AFTER masking if the runner uses clipping. Rebuild on returned
phase/shape changes, matching the author's phase resets.
All inactive values are actually restored
after Adam; masks alone do not protect against stored momentum.
MAIN=TOPOLOGY human_locked; Transformer primary; M0 unpassed; M3 ultimate.
"""
import copy
import math
import time

import torch
from torch import nn
from torch.nn import functional as F


BODY_NAMES = ("input_weight", "start_weight", "recurrent_weight", "hidden_bias")


def _bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_bytes(v) for v in value)
    return 0


class DENSequenceBaseline(nn.Module):
    """One shared dynamic ReLU recurrent feature layer and task-specific heads.

    h0=ReLU(Wstart onehot(start)+b);
    h[j+1]=ReLU(R h[j]+Wtoken onehot(token[j])+b), PAD preserves h;
    logits=head[task](h_final). Old tasks crop BOTH R axes and all channels to
    their saved task timestamp. Input width6 symbols+PAD is shared at all steps,
    avoiding a fixed-position MLP's untrained long-word input coordinates.

    Recurrent connectivity/terminal CE are deliberate adaptations. Defaults for
    sparsity/expansion/drift follow the author's public DEN_run.py; phase budgets,
    multiclass CE and recurrent unit-group definitions are declared deviations.
    No official accuracy, exact DEN replication or protection theorem is claimed.
    """
    def __init__(self, n_states, vocab=6, n_tasks=6, hidden=64, max_hidden=256,
                 ex_k=10, l1=1e-5, l2=1e-4, group_lasso=1e-3,
                 drift_regularization=.5, loss_threshold=.01, split_threshold=.05,
                 max_splits=10, l1_on_bias=True, parameter_budget=250000):
        super().__init__()
        if (n_states not in (8, 16, 32) or vocab != 6 or not 1 <= n_tasks <= 64
                or not 4 <= hidden <= max_hidden <= 512 or not 1 <= ex_k <= 64
                or not 1 <= max_splits <= 64 or not isinstance(parameter_budget, int)
                or parameter_budget < 1):
            raise ValueError("bounded N8/16/32, shared6-symbol DEN-RNN configuration required")
        values = (l1, l2, group_lasso, drift_regularization, loss_threshold, split_threshold)
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in values):
            raise ValueError("finite nonnegative DEN hyperparameters required")
        self.n_states, self.vocab, self.n_tasks = int(n_states), int(vocab), int(n_tasks)
        self.hidden, self.initial_hidden, self.max_hidden = int(hidden), int(hidden), int(max_hidden)
        self.ex_k, self.max_splits = int(ex_k), int(max_splits)
        self.parameter_budget = int(parameter_budget)
        self.l1, self.l2, self.group_lasso = float(l1), float(l2), float(group_lasso)
        self.drift_regularization = float(drift_regularization)
        self.loss_threshold, self.split_threshold = float(loss_threshold), float(split_threshold)
        self.l1_on_bias = bool(l1_on_bias)
        self.input_weight = nn.Parameter(torch.empty(hidden, vocab+1))
        self.start_weight = nn.Parameter(torch.empty(hidden, n_states))
        self.recurrent_weight = nn.Parameter(torch.empty(hidden, hidden))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))
        nn.init.xavier_uniform_(self.input_weight)
        nn.init.xavier_uniform_(self.start_weight)
        nn.init.xavier_uniform_(self.recurrent_weight)
        self.heads = nn.ModuleList()
        self.register_buffer("unit_birth_task", torch.zeros(hidden, dtype=torch.long))
        self.timestamps, self.stage, self.phase = {}, -1, "unstarted"
        self._previous, self._masks, self._step_before = {}, {}, {}
        self._updates = self._total_steps = self._phase_count = 0
        self._phase_loss_sum = 0.
        self._expansion_base = None
        self.history = []
        self.counters = dict(actual_updates=0, backward_selection_edges_examined=0,
            neurons_added=0, neurons_pruned=0, neurons_split=0,
            optimizer_rebuilds_required=0, phase_changes=0,
            gradient_mask_elements=0, inactive_restore_elements=0,
            selection_seconds=0., structural_seconds=0., proximal_seconds=0.)
        if self._parameter_count_at(hidden, 1) > self.parameter_budget:
            raise ValueError("initial shared body and first task head exceed the declared P budget")

    def _parameter_count_at(self, width, heads=None):
        heads = len(self.heads) if heads is None else int(heads)
        return width*width+width*(self.vocab+2+self.n_states+heads*self.n_states)+heads*self.n_states

    def _capacity_width(self):
        width = self.hidden
        while width < self.max_hidden and self._parameter_count_at(width+1) <= self.parameter_budget:
            width += 1
        return width

    def _parameter_masks(self, mode, selected=None):
        masks = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in self.named_parameters()}
        if mode == "all":
            for name in BODY_NAMES:
                masks[name].fill_(True)
        elif mode == "selected":
            selected = selected.to(self.hidden_bias.device)
            for name in ("input_weight", "start_weight", "hidden_bias"):
                masks[name][selected] = True
            masks["recurrent_weight"] = selected[:, None] & selected[None, :]
        elif mode == "expanded":
            base = int(self._expansion_base)
            for name in ("input_weight", "start_weight", "hidden_bias"):
                masks[name][base:] = True
            masks["recurrent_weight"][base:] = True
            masks["recurrent_weight"][:, base:] = True
        elif mode != "head":
            raise ValueError("unknown DEN phase mask")
        masks[f"heads.{self.stage}.weight"].fill_(True)
        masks[f"heads.{self.stage}.bias"].fill_(True)
        if mode == "selected":
            masks[f"heads.{self.stage}.weight"] &= selected[None, :]
        self._masks = masks

    def set_stage(self, stage, total_steps=600, new_token_ids=None):
        if (isinstance(stage, bool) or stage != len(self.heads) or not 0 <= stage < self.n_tasks
                or not 20 <= int(total_steps) <= 6000
                or self.stage >= 0 and self.phase != "complete"):
            raise ValueError("start the next actual task after completing the previous fixed budget")
        if self._parameter_count_at(self.hidden, len(self.heads)+1) > self.parameter_budget:
            raise RuntimeError("DEN parameter budget exhausted by the next public task head; record censored acquisition")
        self.stage, self._total_steps, self._updates = int(stage), int(total_steps), 0
        self._phase_count, self._phase_loss_sum = 0, 0.
        self._previous = {name: getattr(self, name).detach().clone() for name in BODY_NAMES}
        self._expansion_base = None
        self.heads.append(nn.Linear(self.hidden, self.n_states,
                                    device=self.hidden_bias.device, dtype=self.hidden_bias.dtype))
        self.phase = "initial" if stage == 0 else "sparse_head"
        self._parameter_masks("all" if stage == 0 else "head")
        self.counters["optimizer_rebuilds_required"] += 1
        return self

    def forward(self, tokens, starts, task_ids):
        if (tokens.ndim != 2 or tokens.shape[0] > 128 or tokens.shape[1] > 18
                or starts.shape != (tokens.shape[0],) or task_ids.shape != starts.shape):
            raise ValueError("bounded B128,L18 terminal sequence interface required")
        if (torch.any((tokens < 0) | (tokens > self.vocab))
                or torch.any((starts < 0) | (starts >= self.n_states))):
            raise ValueError("public terminal inputs outside declared domain")
        logits = self.hidden_bias.new_empty((len(starts), self.n_states))
        for task in task_ids.unique().tolist():
            if not 0 <= task < len(self.heads):
                raise ValueError("unallocated task head")
            selected = task_ids == task
            width = self.hidden if task == self.stage and self.phase != "complete" else self.timestamps[task]
            state = F.relu(F.embedding(starts[selected], self.start_weight[:width].T)
                           + self.hidden_bias[:width])
            words = tokens[selected]
            for column in range(words.shape[1]):
                token = words[:, column]
                following = F.relu(F.linear(state, self.recurrent_weight[:width, :width],
                                           self.hidden_bias[:width])
                                   + F.embedding(token, self.input_weight[:width].T))
                state = torch.where(token[:, None] != 0, following, state)
            head = self.heads[task]
            logits[selected] = F.linear(state, head.weight[:, :width], head.bias)
        return logits

    def regularization_loss(self):
        result = self.hidden_bias.new_zeros(())
        for name, parameter in self.named_parameters():
            mask = self._masks.get(name)
            if mask is not None:
                result = result + .5*self.l2*parameter[mask].square().sum()
        if self.stage > 0:
            for name, previous in self._previous.items():
                p = getattr(self, name)
                slices = tuple(slice(0, n) for n in previous.shape)
                change = p[slices]-previous
                result = result + .5*self.drift_regularization*change[self._masks[name][slices]].square().sum()
        return result

    @torch.no_grad()
    def mask_gradients_(self):
        if self.phase in ("unstarted", "complete") or self._step_before:
            raise RuntimeError("DEN gradient mask outside one active, unfinished optimizer step")
        for name, p in self.named_parameters():
            mask = self._masks[name]
            if p.grad is not None:
                p.grad.mul_(mask)
            self.counters["gradient_mask_elements"] += mask.numel()
            if not bool(mask.all()):
                self._step_before[name] = p.detach().clone()

    @torch.no_grad()
    def _select(self):
        began = time.perf_counter()
        selected = self.heads[self.stage].weight.ne(0).any(0)
        # In a recurrent layer, relevant predecessors are the transitive
        # closure of nonzero recurrent edges, rather than a feedforward sweep.
        for _ in range(self.hidden):
            previous = selected.clone()
            selected |= self.recurrent_weight[selected].ne(0).any(0)
            self.counters["backward_selection_edges_examined"] += int(previous.sum())*self.hidden
            if torch.equal(previous, selected):
                break
        self._parameter_masks("selected", selected)
        self.counters["selection_seconds"] += time.perf_counter()-began
        return dict(selected_units=int(selected.sum()), total_units=self.hidden,
                    selected_parameter_scalars=sum(int(m.sum()) for m in self._masks.values()))

    @torch.no_grad()
    def _resize(self, keep=None, duplicate=(), restore_split=True):
        """Structural transaction; no training/data calls. Old stamps stay valid."""
        old = self.hidden
        keep = list(range(old)) if keep is None else [int(x) for x in keep]
        duplicates = [int(x) for x in duplicate]
        index = torch.as_tensor(keep+duplicates, device=self.hidden_bias.device, dtype=torch.long)
        if (len(set(keep)) != len(keep) or any(not 0 <= i < old for i in keep+duplicates)
                or not 1 <= len(index) <= self.max_hidden
                or self._parameter_count_at(len(index)) > self.parameter_budget):
            raise ValueError("finite valid DEN resize required")
        if any(keep[:stamp] != list(range(stamp)) for stamp in self.timestamps.values()):
            raise ValueError("structural edit may not remove/reorder timestamped old units")
        body = {name: getattr(self, name).detach() for name in BODY_NAMES}
        values = dict(input_weight=body["input_weight"][index].clone(),
            start_weight=body["start_weight"][index].clone(),
            recurrent_weight=body["recurrent_weight"][index][:, index].clone(),
            hidden_bias=body["hidden_bias"][index].clone())
        previous_width = self._previous["hidden_bias"].numel()
        if restore_split:
            for original in duplicates:
                if original >= previous_width:
                    raise ValueError("only pre-task old units have a semantic-drift snapshot")
                for name in ("input_weight", "start_weight", "hidden_bias"):
                    values[name][original] = self._previous[name][original]
                values["recurrent_weight"][original].zero_()
                values["recurrent_weight"][original, :previous_width] = self._previous["recurrent_weight"][original]
        for name, value in values.items():
            setattr(self, name, nn.Parameter(value))
        for task, head in enumerate(self.heads):
            weights = head.weight.detach()[:, index].clone()
            if task != self.stage and duplicates:
                weights[:, len(keep):].zero_()
            head.in_features = len(index)
            head.weight = nn.Parameter(weights)
        births = self.unit_birth_task[torch.as_tensor(keep, device=index.device)]
        self.unit_birth_task = torch.cat((births, births.new_full((len(duplicates),), self.stage)))
        self.hidden = len(index)

    @torch.no_grad()
    def _expand(self):
        old = self.hidden
        added = min(self.ex_k, self._capacity_width()-old)
        if not added:
            return dict(added=0, capacity_censored=True)
        new = old+added
        values = {}
        for name, width in (("input_weight", self.vocab+1), ("start_weight", self.n_states)):
            p = getattr(self, name)
            value = p.new_empty((new, width)); nn.init.xavier_uniform_(value)
            value[:old] = p; values[name] = value
        recurrent = self.recurrent_weight.new_empty((new, new)); nn.init.xavier_uniform_(recurrent)
        recurrent[:old, :old] = self.recurrent_weight
        values["recurrent_weight"] = recurrent
        values["hidden_bias"] = torch.cat((self.hidden_bias.detach(), self.hidden_bias.new_zeros(added)))
        for name, value in values.items():
            setattr(self, name, nn.Parameter(value))
        for task, head in enumerate(self.heads):
            extension = head.weight.new_zeros((self.n_states, added))
            if task == self.stage:
                nn.init.xavier_uniform_(extension)
            head.weight = nn.Parameter(torch.cat((head.weight.detach(), extension), dim=1))
            head.in_features = new
        self.unit_birth_task = torch.cat((self.unit_birth_task,
            self.unit_birth_task.new_full((added,), self.stage)))
        self.hidden, self._expansion_base = new, old
        self.counters["neurons_added"] += added
        return dict(added=added, before=old, after=new, capacity_censored=added < self.ex_k)

    @torch.no_grad()
    def _prune_and_split(self):
        report = dict(pruned=0, split=0)
        if self._expansion_base is not None:
            base = self._expansion_base
            norm2 = (self.input_weight.square().sum(1)+self.start_weight.square().sum(1)
                     +self.recurrent_weight.square().sum(1))
            retained = list(range(base))+[j for j in range(base, self.hidden) if float(norm2[j]) > 0.]
            report["pruned"] = self.hidden-len(retained)
            if report["pruned"]:
                self._resize(retained, restore_split=False)
                self.counters["neurons_pruned"] += report["pruned"]
        width = self._previous["hidden_bias"].numel()
        distance2 = sum((getattr(self, name)[:width]-self._previous[name]).square().sum(1)
                        for name in ("input_weight", "start_weight"))
        distance2 += (self.recurrent_weight[:width, :width]-self._previous["recurrent_weight"]).square().sum(1)
        drift = distance2.sqrt()
        eligible = torch.where(drift > self.split_threshold)[0]
        order = eligible[torch.argsort(drift[eligible], descending=True)]
        count = min(len(order), self.max_splits, self._capacity_width()-self.hidden)
        chosen = order[:count].tolist()
        report.update(drift=drift.cpu().tolist(), above_threshold=len(eligible),
            split_indices=chosen, split=count,
            threshold=self.split_threshold, deferred_due_to_caps=len(eligible)-count)
        if chosen:
            self._resize(duplicate=chosen)
            self.counters["neurons_split"] += count
        return report

    @torch.no_grad()
    def after_optimizer_step_(self, terminal_CE, step):
        if (not math.isfinite(float(terminal_CE)) or step != self._updates+1
                or step > self._total_steps):
            raise ValueError("finite loss and actual consecutive DEN optimizer-step count required")
        began = time.perf_counter()
        for name, p in self.named_parameters():
            mask = self._masks[name]
            if name in self._step_before:
                p[~mask] = self._step_before[name][~mask]
                self.counters["inactive_restore_elements"] += int((~mask).sum())
            if name.endswith("weight") or self.l1_on_bias:
                selected = p[mask]
                p[mask] = selected.sign()*(selected.abs()-self.l1).clamp_min(0.)
        self._step_before.clear()
        if self.phase == "expansion":
            base = self._expansion_base
            norms = (self.input_weight[base:].square().sum(1)
                     +self.start_weight[base:].square().sum(1)
                     +self.recurrent_weight[base:].square().sum(1)).sqrt()
            factors = (1-self.group_lasso/norms.clamp_min(1e-30)).clamp_min(0.)
            for name in ("input_weight", "start_weight", "recurrent_weight"):
                getattr(self, name)[base:] *= factors[:, None]
        self.counters["proximal_seconds"] += time.perf_counter()-began
        self._updates, self._phase_count = int(step), self._phase_count+1
        self._phase_loss_sum += float(terminal_CE)
        self.counters["actual_updates"] += 1
        event = dict(step=step, phase_before=self.phase, phase_after=self.phase,
            optimizer_rebuild_required=False, hidden=self.hidden)
        if self.stage > 0:
            head_end = max(1, self._total_steps//10)
            selective_end = max(head_end+1, 3*self._total_steps//10)
            split_at = max(selective_end+1, 7*self._total_steps//10)
            structural_start = time.perf_counter()
            if step == head_end:
                event["selection"] = self._select(); self.phase = "selective"
                event["optimizer_rebuild_required"] = True
            elif step == selective_end:
                measured = self._phase_loss_sum/max(1, self._phase_count)
                event["selective_mean_observed_terminal_CE"] = measured
                if measured > self.loss_threshold and self.hidden < self._capacity_width():
                    event["expansion"] = self._expand()
                    self.phase = "expansion"; self._parameter_masks("expanded")
                    event["optimizer_rebuild_required"] = bool(event["expansion"]["added"])
                else:
                    self.phase = "stability"; self._parameter_masks("all")
                event["optimizer_rebuild_required"] = True
            elif step == split_at:
                event["split"] = self._prune_and_split()
                event["optimizer_rebuild_required"] = bool(event["split"]["pruned"] or event["split"]["split"])
                self.phase = "post_split"; self._parameter_masks("all")
                event["optimizer_rebuild_required"] = True
            self.counters["structural_seconds"] += time.perf_counter()-structural_start
        if step == self._total_steps:
            self.timestamps[self.stage] = self.hidden; self.phase = "complete"
        event.update(phase_after=self.phase, hidden=self.hidden)
        if event["phase_before"] != self.phase:
            event["completed_phase_mean_observed_terminal_CE"] = self._phase_loss_sum/max(1, self._phase_count)
            self._phase_count, self._phase_loss_sum = 0, 0.
            self.counters["phase_changes"] += 1
        if event["optimizer_rebuild_required"]:
            self.counters["optimizer_rebuilds_required"] += 1
        if len(event) > 5 or event["phase_before"] != self.phase:
            self.history.append(copy.deepcopy(event))
        return event

    def extra_state_dict(self):
        return dict(kind="DEN-RNN-terminal-adaptation-v1", n_states=self.n_states, vocab=self.vocab,
            n_tasks=self.n_tasks, initial_hidden=self.initial_hidden, hidden=self.hidden,
            max_hidden=self.max_hidden, ex_k=self.ex_k, max_splits=self.max_splits,
            parameter_budget=self.parameter_budget,
            l1=self.l1, l2=self.l2, group_lasso=self.group_lasso,
            drift_regularization=self.drift_regularization, loss_threshold=self.loss_threshold,
            split_threshold=self.split_threshold, l1_on_bias=self.l1_on_bias,
            stage=self.stage, phase=self.phase, timestamps=dict(self.timestamps),
            updates=self._updates, total_steps=self._total_steps,
            phase_count=self._phase_count, phase_loss_sum=self._phase_loss_sum,
            expansion_base=self._expansion_base, previous={k:v.detach().cpu().clone() for k,v in self._previous.items()},
            masks={k:v.detach().cpu().clone() for k,v in self._masks.items()},
            counters=dict(self.counters), history=copy.deepcopy(self.history),
            source_reference="Yoon et al. ICLR2018 Eq2–6/Algorithms1–4; jaehong31/DEN",
            adaptation_not_official_reproduction=True, replay_used=False,
            sequence_architecture="shared expandable ReLU recurrent layer; public task-specific multiclass terminal heads")

    def load_extra_state_dict(self, extra):
        if (extra.get("kind") != "DEN-RNN-terminal-adaptation-v1"
                or int(extra["hidden"]) != self.hidden or int(extra["n_states"]) != self.n_states
                or len(self.heads) != int(extra["stage"])+1):
            raise ValueError("actual DEN-RNN configuration/state mismatch")
        for name in ("max_hidden", "ex_k", "max_splits", "parameter_budget", "l1", "l2", "group_lasso",
                     "drift_regularization", "loss_threshold", "split_threshold", "l1_on_bias"):
            if getattr(self, name) != extra[name]:
                raise ValueError("saved DEN configuration differs")
        self.stage, self.phase = int(extra["stage"]), str(extra["phase"])
        self.timestamps = {int(k):int(v) for k,v in extra["timestamps"].items()}
        self._updates, self._total_steps = int(extra["updates"]), int(extra["total_steps"])
        self._phase_count, self._phase_loss_sum = int(extra["phase_count"]), float(extra["phase_loss_sum"])
        self._expansion_base = extra["expansion_base"]
        device = self.hidden_bias.device
        self._previous = {k:v.to(device) for k,v in extra["previous"].items()}
        self._masks = {k:v.to(device) for k,v in extra["masks"].items()}
        self._step_before = {}; self.counters = dict(extra["counters"])
        self.history = copy.deepcopy(extra["history"])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        width = state_dict["hidden_bias"].numel()
        count = len([name for name in state_dict if name.startswith("heads.") and name.endswith(".weight")])
        if (not self.initial_hidden <= width <= self.max_hidden
                or self._parameter_count_at(width, count) > self.parameter_budget):
            raise ValueError("saved DEN width outside configured budget")
        device, dtype = self.hidden_bias.device, self.hidden_bias.dtype
        for name in BODY_NAMES:
            setattr(self, name, nn.Parameter(torch.empty_like(state_dict[name], device=device, dtype=dtype)))
        self.heads = nn.ModuleList([nn.Linear(width, self.n_states, device=device, dtype=dtype) for _ in range(count)])
        self.unit_birth_task = torch.empty_like(state_dict["unit_birth_task"], device=device)
        self.hidden = width
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def resource_counts(self, optimizer=None):
        params = list(self.parameters())
        active = sum(int(mask.sum()) for mask in self._masks.values())
        return dict(total=sum(p.numel() for p in params), trainable=sum(p.numel() for p in params if p.requires_grad),
            active_after_DEN_masks=active, parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in params),
            registered_buffer_bytes=sum(b.numel()*b.element_size() for b in self.buffers()),
            old_parameter_snapshot_bytes=_bytes(self._previous), active_mask_bytes=_bytes(self._masks),
            transient_inactive_step_snapshot_bytes=_bytes(self._step_before),
            optimizer_tensor_bytes=0 if optimizer is None else _bytes(optimizer.state_dict()),
            hidden_units=self.hidden, task_heads=len(self.heads), timestamps=dict(self.timestamps),
            actual_update_count=self.counters["actual_updates"], counters=dict(self.counters),
            task_timestamp_int64_bytes=8*len(self.timestamps), replay_bytes=0,
            history_is_extra_checkpoint_metadata=True,
            parameter_budget_includes_all_frozen_unused_and_old_head_entries=True,
            parameter_budget=self.parameter_budget,
            resource_boundary="tensor sizes and counted rules; Python metadata/allocator overhead and full serialized checkpoint require separate accounting")


def den_model_from_checkpoint(payload, device="cpu"):
    """Restore an OWN state plus explicit DEN metadata, not downloaded code.

    The coordinator owns trusted torch.load, original RNG restoration and all
    execution isolation. Construction consumes neutral initialization RNG; it
    must not be called a free continuation unless that RNG is then restored.
    """
    extra = payload["model_extra"]
    if extra.get("kind") != "DEN-RNN-terminal-adaptation-v1":
        raise ValueError("owned DEN-RNN state and metadata required")
    options = {name:extra[name] for name in ("n_states", "vocab", "n_tasks", "max_hidden", "ex_k",
        "l1", "l2", "group_lasso", "drift_regularization", "loss_threshold",
        "split_threshold", "max_splits", "l1_on_bias", "parameter_budget")}
    options["hidden"] = extra["initial_hidden"]
    model = DENSequenceBaseline(**options).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.load_extra_state_dict(extra)
    return model
