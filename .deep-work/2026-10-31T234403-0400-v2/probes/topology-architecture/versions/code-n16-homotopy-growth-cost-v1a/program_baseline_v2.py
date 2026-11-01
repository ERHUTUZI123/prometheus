"""Direct terminal-trained state-machine baseline v2; no experiment at import.

No operation bank/slot router and no teacher transitions, modes or programs.
All tokens have direct learned D-by-D stochastic transitions. Public task IDs
select a learned dense residual, permitting task-specific internal semantics.
Compare fully plastic and modular modes explicitly and charge their parameters.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ProgramLearner(nn.Module):
    def __init__(self, n_states, vocab=6, n_tasks=3, latent_factor=1,
                 modular=False, temperature=1., hard_assignment="rowargmax"):
        super().__init__()
        self.n_states, self.n = int(n_states), int(n_states)
        self.vocab, self.n_tasks = int(vocab), int(n_tasks)
        self.latent_factor, self.modular = int(latent_factor), bool(modular)
        if self.n < 2 or self.vocab < 2 or self.n_tasks < 1 or self.latent_factor < 1:
            raise ValueError("invalid finite-state model dimensions")
        if hard_assignment not in ("rowargmax", "bijective"):
            raise ValueError("hard_assignment must be rowargmax or bijective")
        self.dim = self.d = self.n * self.latent_factor
        self.stage, self.temperature, self.hard_assignment = 0, float(temperature), hard_assignment
        d = self.d
        # Neutral observable identity coding, not an operation answer. Every
        # entry remains learned. Extra latent states initially decode copies
        # of observable values; they have no supplied hidden-mode labels.
        encoder = .02 * torch.randn(self.n, d)
        encoder[torch.arange(self.n), torch.arange(self.n)] += 6.
        decoder = .02 * torch.randn(d, self.n)
        decoder[torch.arange(d), torch.arange(d) % self.n] += 6.
        self.encoder = nn.Parameter(encoder)
        self.decoder = nn.Parameter(decoder)
        # NO identity preference for any token: first short terminal examples
        # can move direct one-step maps immediately, without overcoming a
        # diagonal logit barrier or an extra shared-bank routing problem.
        self.transitions = nn.Parameter(.02 * torch.randn(self.vocab - 1, d, d))
        self.task_transitions = nn.Parameter(torch.zeros(self.n_tasks, self.vocab - 1, d, d))
        self.task_bias = nn.Parameter(torch.zeros(self.n_tasks, d))
        self._initialized_tasks = {0}
        self._hooks, self._masks, self._snapshots = [], {}, {}
        self.last_new_token_ids = None
        self.set_temperature(temperature)
        self.set_stage(0)

    def set_temperature(self, value):
        value = float(value)
        if not value > 0:
            raise ValueError("temperature must be positive")
        self.temperature = max(.05, value)
        return self

    def _clear_masks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks, self._masks, self._snapshots = [], {}, {}

    def _mask_task(self, name, task):
        parameter = dict(self.named_parameters())[name]
        mask = torch.zeros_like(parameter, dtype=torch.bool)
        mask[task] = True
        self._masks[name] = mask
        self._snapshots[name] = parameter.detach().clone()
        self._hooks.append(parameter.register_hook(
            lambda grad, name=name: grad * self._masks[name].to(grad.device)))

    def set_stage(self, stage, freeze_old=None, new_token_ids=None):
        stage = int(stage)
        if not 0 <= stage < self.n_tasks:
            raise ValueError("stage outside task range")
        self._clear_masks()
        self.stage = stage
        self.last_new_token_ids = None if new_token_ids is None else list(new_token_ids)
        # Transfer only an actually learned previous task, never a teacher map.
        # Guard against accidental reset if the harness calls set_stage twice.
        if stage not in self._initialized_tasks:
            with torch.no_grad():
                self.task_transitions[stage].copy_(self.task_transitions[stage - 1])
                self.task_bias[stage].copy_(self.task_bias[stage - 1])
            self._initialized_tasks.add(stage)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        frozen = self.modular if freeze_old is None else bool(freeze_old)
        if frozen and stage > 0:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.transitions.requires_grad_(False)
            # New task can change all of its token transitions, including old
            # tokens on previously unused latent states. Freezing these rows
            # would artificially obstruct acquisition of a new memory mode.
            self._mask_task("task_transitions", stage)
            self._mask_task("task_bias", stage)
        return self

    @torch.no_grad()
    def enforce_frozen_rows_(self):
        """Call after optimizer.step, especially with momentum/weight decay."""
        parameters = dict(self.named_parameters())
        for name, mask in self._masks.items():
            p = parameters[name]
            frozen = ~mask.to(p.device)
            p[frozen] = self._snapshots[name].to(p.device)[frozen]
        return self

    def _log_components(self):
        tau = self.temperature
        encoding = F.log_softmax((self.encoder[None] + self.task_bias[:, None]) / tau, dim=-1)
        transitions = F.log_softmax((self.transitions[None] + self.task_transitions) / tau, dim=-1)
        decoding = F.log_softmax(self.decoder / tau, dim=-1)
        return encoding, transitions, decoding

    def forward(self, tokens, starts, task_ids, hard=False):
        if hard:
            return self.hard_eval(tokens, starts, task_ids)
        if tokens.ndim != 2 or starts.ndim != 1 or task_ids.ndim != 1:
            raise ValueError("expected tokens[B,L], starts[B], task_ids[B]")
        encoding, transitions, decoding = self._log_components()
        state = encoding[task_ids, starts]
        # Stable log semiring. PAD is a skipped identity, not a trainable
        # transition or an -infinity branch whose derivatives can become NaN.
        for position in range(tokens.shape[1]):
            token = tokens[:, position]
            active = token.ne(0)
            index = (token - 1).clamp_min(0)
            matrix = transitions[task_ids, index]
            following = torch.logsumexp(state[:, :, None] + matrix, dim=1)
            state = torch.where(active[:, None], following, state)
        return torch.logsumexp(state[:, :, None] + decoding[None], dim=1)

    def _hard_maps(self):
        logits = self.transitions[None] + self.task_transitions
        if self.hard_assignment == "rowargmax":
            return logits.argmax(-1)
        # Optional generic one-to-one projection for a declared permutation
        # task prior. It uses learned logits only, never teacher operations.
        from scipy.optimize import linear_sum_assignment
        result = torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
        for task in range(self.n_tasks):
            for token in range(self.vocab - 1):
                rows, cols = linear_sum_assignment(-logits[task, token].detach().cpu().numpy())
                result[task, token, torch.as_tensor(rows, device=logits.device)] = torch.as_tensor(
                    cols, dtype=torch.long, device=logits.device)
        return result

    @torch.no_grad()
    def hard_eval(self, tokens, starts, task_ids):
        """Execute learned discrete states, not soft predictions relabeled hard."""
        encoding = (self.encoder[None] + self.task_bias[:, None]).argmax(-1)
        transitions = self._hard_maps()
        decoding = self.decoder.argmax(-1)
        state = encoding[task_ids, starts]
        for position in range(tokens.shape[1]):
            token = tokens[:, position]
            following = transitions[task_ids, (token - 1).clamp_min(0), state]
            state = torch.where(token.ne(0), following, state)
        prediction = decoding[state]
        # Finite terminal logits compatible with the common CE/accuracy API.
        logits = self.encoder.new_full((len(starts), self.n), -30.)
        logits.scatter_(1, prediction[:, None], 0.)
        return logits

    def parameter_counts(self):
        named = dict(self.named_parameters())
        return dict(total=sum(p.numel() for p in named.values()),
                    requires_grad=sum(p.numel() for p in named.values() if p.requires_grad),
                    active_after_row_masks=sum(int(self._masks[name].sum()) if name in self._masks else p.numel()
                        for name, p in named.items() if p.requires_grad),
                    latent_states=self.d, latent_factor=self.latent_factor,
                    shared_transition_scalars=self.transitions.numel(),
                    task_transition_scalars=self.task_transitions.numel())

    def representation_state_bytes(self):
        """Additional mask/snapshot tensors; learned weights counted separately."""
        tensors = list(self._masks.values()) + list(self._snapshots.values())
        return sum(t.numel() * t.element_size() for t in tensors)

    @torch.no_grad()
    def discrete_program(self, task=0, token_ids=None):
        task = int(task)
        token_ids = list(range(1, self.vocab)) if token_ids is None else list(token_ids)
        maps = self._hard_maps()
        encoding = (self.encoder + self.task_bias[task]).argmax(-1)
        return dict(task=task, encoder=encoding.cpu().tolist(),
                    decoder=self.decoder.argmax(-1).cpu().tolist(),
                    transitions={str(token): maps[task, token - 1].cpu().tolist() for token in token_ids},
                    latent_states=self.d, modular=self.modular,
                    hard_assignment=self.hard_assignment, teacher_operations_used=False,
                    initialization="neutral observable identity coding; flat random token maps")


def parameter_counts(model):
    return model.parameter_counts()
