"""Non-official linear self-modifying memory + CMS terminal-task baseline.

This is NOT a replication of the complete experimental HOPE architecture.
It implements the explicit linear/L2 specialization of Eq.93, C=1, in
Behrouz et al., arXiv:2512.24695v1, followed by sequential multi-frequency
MLPs (Eq.70/71/97). See hope-reading.md for the precise reconstruction gap.

The inner memories generate their own update targets, including the memories
which generate keys, values, learning rates and retention gates. Their initial
states are learned by terminal cross-entropy through the actual recurrence.
CMS weights persist across tasks, with distinct feedback-update periods.

Use make_optimizer(model), not Adam(model.parameters()), to retain CMS.
No dataset, labels, experiment, network call or training runs at import.
MAIN=TOPOLOGY human-locked; this ordinary baseline serves strong comparisons.
"""
from __future__ import annotations

import copy
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


SOURCE = "https://arxiv.org/html/2512.24695v1"
MAX_TASKS = 64
MAX_BATCH = 256
MAX_WIDTH = 128
MEMORIES = ("key", "value", "eta", "alpha", "memory")


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


class CMSMLP(nn.Module):
    """Residual MLP; its parameters have a real distinct update frequency."""

    def __init__(self, width, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.in_proj = nn.Linear(width, expansion * width)
        self.out_proj = nn.Linear(expansion * width, width)

    def forward(self, value):
        return value + self.out_proj(F.silu(self.in_proj(self.norm(value))))


class LinearSelfModBlock(nn.Module):
    """Five *changing* linear memories, followed by a CMS chain.

    For a column-vector convention, with C=1 and a half-squared L2 loss,
    Eq.93 is exactly

      target_j = M_j v,
      M_j' = alpha M_j - eta [2 M_j k - target_j] k^T.

    The second M_j k is the L2 residual; the first is the separate DGD
    rank-one term. Neither term is silently dropped. All proposals use the
    SAME pre-update state. The self-generated target is held constant in the
    *inner* gradient formula, but outer BPTT differentiates its computation.
    q is a slow learned projection, following the explicit Eq.83/94 text.

    Gate squashing, residual wiring, depthwise convolution and initialization
    are disclosed terminal-task engineering choices, not claimed author code.
    """

    def __init__(self, width, periods, eta_max=.1, decay_max=.01):
        super().__init__()
        self.width = int(width)
        self.eta_max, self.decay_max = float(eta_max), float(decay_max)
        self.input_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.memory_output = nn.Linear(width, width, bias=False)
        self.conv_kernel = nn.Parameter(torch.zeros(4, width))
        with torch.no_grad():
            self.conv_kernel[-1].fill_(.5)
        self.initial_memories = nn.Parameter(torch.empty(5, width, width))
        for matrix in self.initial_memories:
            nn.init.orthogonal_(matrix)
        with torch.no_grad():
            self.initial_memories.mul_(.5)
        self.cms = nn.ModuleList([CMSMLP(width) for _ in periods])

    def forward(self, sequence, active):
        batch, length, width = sequence.shape
        # Per-episode fast matrices. No tensor survives into another example
        # or evaluation batch; learned initial_memories persist as real P.
        states = self.initial_memories[None].expand(batch, -1, -1, -1)
        local = sequence.new_zeros((batch, 4, width))
        outputs = []
        for time in range(length):
            use = active[:, time]
            observed = sequence[:, time]
            next_local = torch.cat((local[:, 1:], observed[:, None]), dim=1)
            local = torch.where(use[:, None, None], next_local, local)
            conv = (local * self.conv_kernel[None]).sum(dim=1)
            x = self.input_norm(observed + F.silu(conv))
            # Every quantity below reads the old state, including gate and
            # projection memories; there is no teacher/value/state oracle.
            projections = torch.einsum("bmij,bj->bmi", states, x)
            key = F.normalize(projections[:, 0], p=2, dim=-1, eps=1e-6)
            value = projections[:, 1]
            eta = self.eta_max * torch.sigmoid(projections[:, 2].mean(-1))
            alpha = 1. - self.decay_max * torch.sigmoid(projections[:, 3].mean(-1))
            query = F.normalize(self.query(x), p=2, dim=-1, eps=1e-6)
            read = torch.einsum("bij,bj->bi", states[:, 4], query)
            result = observed + self.memory_output(read)
            # Target generation and residual evaluation for ALL five memories.
            at_key = torch.einsum("bmij,bj->bmi", states, key)
            own_target = torch.einsum("bmij,bj->bmi", states, value)
            proposal = (alpha[:, None, None, None] * states
                        - eta[:, None, None, None]
                        * (2. * at_key - own_target)[..., None]
                        * key[:, None, None, :])
            states = torch.where(use[:, None, None, None], proposal, states)
            for level in self.cms:
                result = level(result)
            outputs.append(torch.where(use[:, None], result, observed))
        return torch.stack(outputs, dim=1)


class SelfModCMSReconstruction(nn.Module):
    """Trainable terminal sequence model; NOT complete or official HOPE.

    Inputs are only raw symbol IDs, public initial state and public task ID.
    The learner receives a terminal label through the caller's CE loss.
    No operation tables, intermediate-state labels or correct routes enter.

    CMS periods count TERMINAL-FEEDBACK MINIBATCHES, not input tokens. This is
    an explicit adaptation of Eq.71 to this benchmark's available feedback.
    They remain global across task boundaries if the optimizer is retained.
    Six task codes are preallocated by default, fully charged in P. A caller
    may request 1 and grow codes; rebuild the optimizer after growth and report
    the resulting reset, rather than silently claiming preserved CMS history.
    """

    def __init__(self, n_states, vocab=6, n_tasks=6, max_len=18,
                 d_model=64, n_layers=2, cms_periods=(1, 4, 16),
                 eta_max=.1, decay_max=.01):
        super().__init__()
        if n_states not in (8, 16, 32) or vocab not in range(1, 7):
            raise ValueError("N must be 8/16/32, with 1..6 non-PAD symbols")
        if (isinstance(d_model, bool) or not isinstance(d_model, Integral)
                or not 8 <= d_model <= MAX_WIDTH or n_layers not in (1, 2, 3)):
            raise ValueError("bounded width8..128 and1..3 blocks required")
        periods = tuple(int(x) for x in cms_periods)
        if (not periods or len(periods) > 4 or any(x < 1 or x > 256 for x in periods)
                or tuple(sorted(set(periods))) != periods):
            raise ValueError("CMS needs1..4 strictly increasing periods in1..256")
        if not 1 <= n_tasks <= MAX_TASKS or max_len != 18:
            raise ValueError("task codes1..64, fixed benchmark padded length18")
        if not 0 < eta_max <= .5 or not 0 <= decay_max <= .1:
            raise ValueError("bounded positive eta and retention gate required")
        self.n_states, self.vocab, self.n_tasks = int(n_states), int(vocab), int(n_tasks)
        self.max_len, self.width = int(max_len), int(d_model)
        self.layers_count, self.cms_periods = int(n_layers), periods
        self.tokens = nn.Embedding(vocab + 1, d_model, padding_idx=0)
        self.starts = nn.Embedding(n_states, d_model)
        self.tasks = nn.Embedding(n_tasks, d_model)
        self.read_token = nn.Parameter(torch.randn(d_model) * .02)
        self.layers = nn.ModuleList([
            LinearSelfModBlock(d_model, periods, eta_max=eta_max, decay_max=decay_max)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_states)
        self.register_buffer("_saved_task_stage", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_task_growth_events", torch.tensor(0, dtype=torch.long))
        self.stage = 0
        self.last_growth_report = dict(grew=False, optimizer_rebuild_required=False)
        self._calls = self._logical_positions = self._max_batch_seen = 0
        self._max_padded_seen = 0
        self._constructor = dict(n_states=self.n_states, vocab=self.vocab,
            max_len=self.max_len, d_model=self.width, n_layers=self.layers_count,
            cms_periods=list(periods), eta_max=float(eta_max), decay_max=float(decay_max))

    def forward(self, tokens, starts, task_ids):
        if tokens.ndim != 2 or starts.ndim != 1 or task_ids.ndim != 1:
            raise ValueError("tokens[B,L], starts[B], task_ids[B] required")
        batch, length = tokens.shape
        if not 1 <= batch <= MAX_BATCH or length > self.max_len:
            raise ValueError("batch1..256 and padded sequence length<=18 required")
        if starts.shape[0] != batch or task_ids.shape[0] != batch:
            raise ValueError("raw input batch sizes differ")
        # START and final READ are model tokens, not teacher states. PAD steps
        # do not advance either fast memory or local-convolution history.
        sequence = torch.cat((self.starts(starts)[:, None], self.tokens(tokens),
                              self.read_token[None, None].expand(batch, 1, -1)), dim=1)
        sequence = sequence + self.tasks(task_ids)[:, None]
        active = torch.cat((torch.ones((batch, 1), device=tokens.device, dtype=torch.bool),
                            tokens.ne(0),
                            torch.ones((batch, 1), device=tokens.device, dtype=torch.bool)), dim=1)
        for layer in self.layers:
            sequence = layer(sequence, active)
        self._calls += 1
        self._logical_positions += int(batch * (length + 2) * self.layers_count)
        self._max_batch_seen = max(self._max_batch_seen, int(batch))
        self._max_padded_seen = max(self._max_padded_seen, int(length))
        return self.head(self.norm(sequence[:, -1]))

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    requires_grad=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    active_after_row_masks=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _resize_task_codes(self, count, loading=False):
        count = int(count)
        if not 1 <= count <= MAX_TASKS:
            raise ValueError("task code allocation exceeds64")
        old, previous = self.tasks.weight, self.n_tasks
        if count == previous:
            return
        if loading:
            values = old.new_zeros((count, self.width))
        else:
            if count < previous:
                raise ValueError("cannot discard learned task codes")
            rows = old.new_empty((count - previous, self.width))
            nn.init.normal_(rows)
            values = torch.cat((old.detach(), rows), dim=0)
        self.tasks.weight = nn.Parameter(values, requires_grad=old.requires_grad)
        self.tasks.num_embeddings, self.n_tasks = count, count
        if not loading:
            self._task_growth_events.add_(1)
        self.last_growth_report = dict(grew=not loading, loading_shape_restoration_only=bool(loading),
            old_task_codes=previous, new_task_codes=count,
            delta_parameter_scalars=(count - previous) * self.width,
            optimizer_rebuild_required=True, old_computational_values_preserved=not loading)

    def set_stage(self, stage, new_token_ids=None):
        if isinstance(stage, bool) or not isinstance(stage, Integral) or not 0 <= stage < MAX_TASKS:
            raise ValueError("integer public task ID0..63 required")
        if stage >= self.n_tasks:
            self._resize_task_codes(int(stage) + 1)
        else:
            self.last_growth_report = dict(grew=False, optimizer_rebuild_required=False,
                                          delta_parameter_scalars=0)
        self.stage = int(stage)
        self._saved_task_stage.fill_(self.stage)
        # No freezing, reset, or new task-specific expert is hidden here.
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        weights = state_dict.get("tasks.weight")
        if weights is None or weights.ndim != 2 or weights.shape[1] != self.width:
            raise ValueError("matching full task-code tensor required")
        self._resize_task_codes(weights.shape[0], loading=True)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.stage = int(self._saved_task_stage.item())
        if not 0 <= self.stage < self.n_tasks:
            raise ValueError("restored public stage exceeds task-code allocation")
        return result

    def get_extra_state(self):
        return dict(schema="selfmod-cms-linear-reconstruction-v1", constructor=self._constructor,
                    official_implementation=False, complete_HOPE_replication=False,
                    source=SOURCE, source_equations=[70, 71, 83, 84, 93, 97],
                    query_policy="slow fixed-within-context projection, explicit Eq83 text",
                    inner_chunk_size=1, gradient_convention="half squared L2, Eq93",
                    cms_context_unit="terminal-feedback minibatch, explicit benchmark adaptation",
                    telemetry=dict(forward_calls=self._calls, logical_block_positions=self._logical_positions,
                                   max_batch=self._max_batch_seen, max_padded=self._max_padded_seen))

    def set_extra_state(self, state):
        if state.get("schema") != "selfmod-cms-linear-reconstruction-v1":
            raise ValueError("unsupported self-modifying reconstruction metadata")
        if state.get("constructor") != self._constructor:
            raise ValueError("reconstruction configuration mismatch on restore")
        stats = state.get("telemetry", {})
        self._calls = int(stats.get("forward_calls", 0))
        self._logical_positions = int(stats.get("logical_block_positions", 0))
        self._max_batch_seen = int(stats.get("max_batch", 0))
        self._max_padded_seen = int(stats.get("max_padded", 0))

    def extra_state_dict(self):
        # Compatibility with current evidence runner, not another live plan.
        return self.get_extra_state()

    def resource_counts(self, optimizer=None):
        scalar_bytes = self.read_token.element_size()
        fast_per_example = self.layers_count * 5 * self.width * self.width
        expected = (self.layers_count * ((7 + 4 * len(self.cms_periods)) * self.width**2
                    + (6 + 5 * len(self.cms_periods)) * self.width)
                    + (self.vocab + 2*self.n_states + self.n_tasks + 4)*self.width + self.n_states)
        if self.parameter_counts()["total"] != expected:
            raise RuntimeError("real reconstruction P differs from its declared architecture")
        return dict(**self.parameter_counts(),
            expected_parameter_count=expected,
            model_parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
            registered_buffer_tensor_bytes=sum(b.numel()*b.element_size() for b in self.buffers()),
            learned_fast_initial_state_scalars=fast_per_example,
            fast_state_current_logical_bytes_per_example=fast_per_example * scalar_bytes,
            fast_state_current_logical_bytes_at_max_batch=fast_per_example * scalar_bytes * self._max_batch_seen,
            created_fast_matrix_values_upper_per_forward=(self._max_padded_seen+3)*fast_per_example*self._max_batch_seen,
            fast_state_accounting_boundary="logical matrices only; autograd saved tensors/allocator peak require external measurement",
            cumulative_dense_dot_product_MAC_proxy=self._logical_positions*(18+4*len(self.cms_periods))*self.width**2,
            dot_product_proxy_boundary="forward dense products only, including padded work; excludes rank updates, scalar/gate/norm/activation work, backward and optimizer",
            persistent_inference_fast_weight_bytes=0,
            independent_episode_reset=True, input_labels_used_in_forward=0,
            cms_parameter_scalars=sum(p.numel() for layer in self.layers for level in layer.cms for p in level.parameters()),
            cms_periods=list(self.cms_periods), cms_period_unit="feedback minibatch",
            allocated_task_codes=self.n_tasks, stage=self.stage,
            replay_bytes=0, optimizer_tensor_bytes=None if optimizer is None else _tensor_bytes(optimizer.state_dict()),
            learning_rule_fidelity=self.get_extra_state(), last_growth_report=copy.deepcopy(self.last_growth_report))


class MultiFrequencyOptimizer:
    """Adam base + true delayed CMS updates using accumulated gradients.

    Level l receives G_l <- G_l + terminal_grad each feedback minibatch.
    On step t divisible by C_l, Adam_l consumes G_l/C_l, and G_l is zeroed.
    This instantiates Eq71's permitted optimizer f with an explicit 1/C_l
    normalization; Adam moments advance ONLY on that level's update events.
    The period clock and pending buffers survive task changes/checkpointing.
    Recreating this object at task boundaries discards that history and MUST
    be reported as a separate policy. No extra labels are requested.
    """

    def __init__(self, model, lr=3e-4, cms_lr=None, weight_decay=0., betas=(.9, .999)):
        if not isinstance(model, SelfModCMSReconstruction):
            raise TypeError("SelfModCMSReconstruction required")
        if not 0 < lr <= .1 or cms_lr is not None and not 0 < cms_lr <= .1:
            raise ValueError("positive bounded learning rates required")
        self.model = model
        self.periods = tuple(model.cms_periods)
        groups = [[] for _ in self.periods]
        for layer in model.layers:
            for index, module in enumerate(layer.cms):
                groups[index].extend(module.parameters())
        cms_ids = {id(p) for group in groups for p in group}
        base = [p for p in model.parameters() if id(p) not in cms_ids]
        self.base = torch.optim.AdamW(base, lr=lr, weight_decay=weight_decay, betas=betas)
        self.levels = [torch.optim.AdamW(group, lr=lr if cms_lr is None else cms_lr,
                       weight_decay=weight_decay, betas=betas) for group in groups]
        self._parameters = groups
        self.accumulators = [[torch.zeros_like(p, memory_format=torch.preserve_format) for p in group]
                             for group in groups]
        self.pending = [0] * len(groups)
        self.level_updates = [0] * len(groups)
        self.global_step = 0
        self._parameter_ids = tuple(id(p) for p in model.parameters())

    @property
    def param_groups(self):
        return self.base.param_groups + [group for optimizer in self.levels for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)
        for optimizer in self.levels:
            optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("explicit terminal loss/backward required; no hidden closure feedback")
        if tuple(id(p) for p in self.model.parameters()) != self._parameter_ids:
            raise RuntimeError("parameter objects changed: rebuild optimizer and account lost nested state")
        self.global_step += 1
        for level, (group, buffers) in enumerate(zip(self._parameters, self.accumulators)):
            for parameter, accumulator in zip(group, buffers):
                if parameter.grad is not None:
                    accumulator.add_(parameter.grad)
            self.pending[level] += 1
        self.base.step()
        for level, (optimizer, group, buffers, period) in enumerate(zip(
                self.levels, self._parameters, self.accumulators, self.periods)):
            if self.global_step % period:
                continue
            for parameter, accumulator in zip(group, buffers):
                parameter.grad = accumulator / self.pending[level]
            optimizer.step()
            for accumulator in buffers:
                accumulator.zero_()
            self.pending[level] = 0
            self.level_updates[level] += 1
        return None

    def state_dict(self):
        return dict(schema="multifrequency-terminal-adam-v1", periods=list(self.periods),
                    global_step=self.global_step, pending=list(self.pending),
                    level_updates=list(self.level_updates), base=self.base.state_dict(),
                    levels=[x.state_dict() for x in self.levels],
                    accumulated_gradients=self.accumulators)

    def load_state_dict(self, state):
        if state.get("schema") != "multifrequency-terminal-adam-v1" or tuple(state["periods"]) != self.periods:
            raise ValueError("nested optimizer configuration differs")
        if len(state["levels"]) != len(self.levels) or len(state["accumulated_gradients"]) != len(self.levels):
            raise ValueError("nested optimizer level count differs")
        for group, saved in zip(self.accumulators, state["accumulated_gradients"]):
            if len(group) != len(saved) or any(a.shape != b.shape for a, b in zip(group, saved)):
                raise ValueError("pending CMS gradient shapes differ")
        self.base.load_state_dict(state["base"])
        for optimizer, saved in zip(self.levels, state["levels"]):
            optimizer.load_state_dict(saved)
        for group, saved in zip(self.accumulators, state["accumulated_gradients"]):
            for accumulator, source in zip(group, saved):
                accumulator.copy_(source.to(accumulator))
        self.global_step = int(state["global_step"])
        self.pending = [int(x) for x in state["pending"]]
        self.level_updates = [int(x) for x in state["level_updates"]]
        if len(self.pending) != len(self.periods) or len(self.level_updates) != len(self.periods):
            raise ValueError("invalid CMS clock")
        if any(p != self.global_step % c for p, c in zip(self.pending, self.periods)):
            raise ValueError("pending gradient counts do not match CMS clock")

    def resource_report(self):
        return dict(global_feedback_steps=self.global_step, cms_periods=list(self.periods),
                    cms_update_events=list(self.level_updates), pending_gradients=list(self.pending),
                    pending_gradient_tensor_bytes=_tensor_bytes(self.accumulators),
                    total_optimizer_tensor_bytes=_tensor_bytes(self.state_dict()),
                    state_reset_policy="retain optimizer across all stages; checkpoints include pending buffers",
                    feedback="same caller-provided terminal minibatch labels; no extra labels")


def make_hope_reconstruction(n_states, width=64, n_tasks=6, vocab=6,
                             n_layers=2, cms_periods=(1, 4, 16)):
    return SelfModCMSReconstruction(n_states=n_states, d_model=width,
        n_tasks=n_tasks, vocab=vocab, n_layers=n_layers, cms_periods=cms_periods)


def make_optimizer(model, lr=3e-4, cms_lr=None, weight_decay=0., betas=(.9, .999)):
    return MultiFrequencyOptimizer(model, lr=lr, cms_lr=cms_lr,
                                   weight_decay=weight_decay, betas=betas)
