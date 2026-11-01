"""Trainable GPU baselines. Importing constructs/runs no experiment.

No teacher operation table, internal state labels or correct call sequences.
All forward methods return terminal class logits; optimizer/data are external.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class _MaskedRows(nn.Module):
    """Partial-parameter masks; snapshots also protect against AdamW decay.

    Recreate Adam at each stage. Call enforce_frozen_rows_ after every optimizer
    step if partially frozen parameters receive momentum or weight decay.
    """
    def __init__(self):
        super().__init__()
        self._mask_hooks, self._row_masks, self._row_snapshots = [], {}, {}

    def _clear_masks(self):
        for hook in self._mask_hooks:
            hook.remove()
        self._mask_hooks, self._row_masks, self._row_snapshots = [], {}, {}

    def _mask_rows(self, name, rows):
        p = dict(self.named_parameters())[name]
        mask = torch.zeros_like(p, dtype=torch.bool)
        rows = [int(r) for r in rows if 0 <= int(r) < p.shape[0]]
        if rows:
            mask[rows] = True
        self._row_masks[name] = mask
        self._row_snapshots[name] = p.detach().clone()
        self._mask_hooks.append(p.register_hook(
            lambda grad, name=name: grad * self._row_masks[name].to(grad.device)))

    @torch.no_grad()
    def enforce_frozen_rows_(self):
        params = dict(self.named_parameters())
        for name, mask in self._row_masks.items():
            p = params[name]
            frozen = ~mask.to(p.device)
            p[frozen] = self._row_snapshots[name].to(p.device)[frozen]
        return self

    def parameter_counts(self):
        params = dict(self.named_parameters())
        return dict(total=sum(p.numel() for p in params.values()),
                    requires_grad=sum(p.numel() for p in params.values() if p.requires_grad),
                    active_after_row_masks=sum(
                        int(self._row_masks[n].sum()) if n in self._row_masks else p.numel()
                        for n, p in params.items() if p.requires_grad))


class TinyTransformer(_MaskedRows):
    def __init__(self, n_states, vocab=6, max_len=12, d_model=48,
                 n_heads=4, n_layers=2, n_tasks=3, adapter=False):
        super().__init__()
        self.n_states, self.vocab, self.max_len = n_states, vocab, max_len
        self.n_tasks, self.adapter, self.stage = n_tasks, bool(adapter), 0
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.tokens = nn.Embedding(vocab, d_model, padding_idx=0)
        self.starts = nn.Embedding(n_states, d_model)
        self.tasks = nn.Embedding(n_tasks, d_model)
        self.positions = nn.Parameter(torch.randn(max_len + 2, d_model) * .02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * .02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                dropout=0., activation="relu", batch_first=True, norm_first=True)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        if self.adapter:
            self.adapters = nn.ModuleList([nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, d_model))
                for _ in range(n_layers)]) for _ in range(n_tasks)])
            self.heads = nn.ModuleList([nn.Linear(d_model, n_states) for _ in range(n_tasks)])
            # Identity residual initialization; these are genuinely trainable.
            for task_adapters in self.adapters:
                for block in task_adapters:
                    nn.init.zeros_(block[-1].weight)
                    nn.init.zeros_(block[-1].bias)
        else:
            self.head = nn.Linear(d_model, n_states)
        self._initialized_tasks = {0}
        self.set_stage(0)

    def set_stage(self, stage, new_token_ids=None):
        stage = int(stage)
        if not 0 <= stage < self.n_tasks:
            raise ValueError("stage outside task range")
        self.stage = stage
        self._clear_masks()
        for p in self.parameters():
            p.requires_grad_(True)
        if not self.adapter or stage == 0:
            return self
        # Warm-start every new task from the trained stage0 adapter/head. Never
        # overwrite a task that has already been initialized/trained.
        if stage not in self._initialized_tasks:
            self.adapters[stage].load_state_dict(copy.deepcopy(self.adapters[0].state_dict()))
            self.heads[stage].load_state_dict(copy.deepcopy(self.heads[0].state_dict()))
            self._initialized_tasks.add(stage)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.adapters[stage].parameters():
            p.requires_grad_(True)
        for p in self.heads[stage].parameters():
            p.requires_grad_(True)
        self.tokens.weight.requires_grad_(True)
        self.tasks.weight.requires_grad_(True)
        rows = [stage + 3] if new_token_ids is None else list(new_token_ids)
        self._mask_rows("tokens.weight", rows)
        self._mask_rows("tasks.weight", [stage])
        return self

    def forward(self, tokens, starts, task_ids):
        b, length = tokens.shape
        if length > self.max_len:
            raise ValueError("program exceeds max_len")
        x = torch.cat([self.starts(starts)[:, None], self.tokens(tokens),
                       self.cls.expand(b, -1, -1)], dim=1)
        x = x + self.positions[:length + 2] + self.tasks(task_ids)[:, None]
        padding = torch.cat([torch.zeros((b, 1), device=tokens.device, dtype=torch.bool),
                             tokens.eq(0),
                             torch.zeros((b, 1), device=tokens.device, dtype=torch.bool)], dim=1)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=padding)
            if self.adapter:
                additions = torch.zeros_like(x)
                for task in range(self.n_tasks):
                    ids = torch.where(task_ids == task)[0]
                    if ids.numel():
                        additions[ids] = self.adapters[task][layer_id](x[ids])
                x = x + additions
        pooled = self.norm(x[:, -1])
        if not self.adapter:
            return self.head(pooled)
        logits = pooled.new_zeros((b, self.n_states))
        for task in range(self.n_tasks):
            ids = torch.where(task_ids == task)[0]
            if ids.numel():
                logits[ids] = self.heads[task](pooled[ids])
        return logits


class ProgramLearner(_MaskedRows):
    """Direct learned probabilistic finite-state program, no correct operations.

    D=N*latent_factor. The neutral identity initialization carries raw start
    information through long strings; it supplies no token->operation mapping.
    With latent_factor=2 the entire doubled workspace exists from the start.
    Default is fully plastic; modular=True is a separately countable control.
    """
    def __init__(self, n_states, vocab=6, n_tasks=3, latent_factor=1,
                 modular=False, temperature=1.):
        super().__init__()
        self.n_states, self.vocab, self.n_tasks = n_states, vocab, n_tasks
        self.latent_factor, self.modular = int(latent_factor), bool(modular)
        if self.latent_factor < 1:
            raise ValueError("positive latent_factor required")
        self.dim = n_states * self.latent_factor
        self.temperature, self.stage = float(temperature), 0
        d = self.dim
        encoder = .02 * torch.randn(n_states, d)
        encoder[torch.arange(n_states), torch.arange(n_states)] += 6.
        transitions = .02 * torch.randn(vocab, d, d)
        transitions += 6. * torch.eye(d)[None]
        decoder = .02 * torch.randn(d, n_states)
        decoder[torch.arange(d), torch.arange(d) % n_states] += 6.
        self.encoder = nn.Parameter(encoder)
        self.transitions = nn.Parameter(transitions)
        self.decoder = nn.Parameter(decoder)
        self.task_bias = nn.Parameter(torch.zeros(n_tasks, d))
        self.register_buffer("identity", torch.eye(d))
        self.set_stage(0)

    def set_temperature(self, value):
        if not float(value) > 0:
            raise ValueError("positive temperature required")
        self.temperature = max(.05, float(value))
        return self

    def set_stage(self, stage, freeze_old=None, new_token_ids=None):
        stage = int(stage)
        if not 0 <= stage < self.n_tasks:
            raise ValueError("stage outside task range")
        self.stage = stage
        self._clear_masks()
        for p in self.parameters():
            p.requires_grad_(True)
        freeze_old = self.modular if freeze_old is None else bool(freeze_old)
        if freeze_old and stage > 0:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            rows = [stage + 3] if new_token_ids is None else list(new_token_ids)
            self._mask_rows("transitions", rows)
            self._mask_rows("task_bias", [stage])
        return self

    def forward(self, tokens, starts, task_ids, hard=False):
        tau = self.temperature
        encoding = self.encoder[starts] + self.task_bias[task_ids]
        if hard:
            state = F.one_hot(encoding.argmax(-1), self.dim).to(self.encoder.dtype)
            matrices = F.one_hot(self.transitions.argmax(-1), self.dim).to(state.dtype)
            decoder = F.one_hot(self.decoder.argmax(-1), self.n_states).to(state.dtype)
        else:
            state = F.softmax(encoding / tau, dim=-1)
            matrices = F.softmax(self.transitions / tau, dim=-1)
            decoder = F.softmax(self.decoder / tau, dim=-1)
        # PAD is actual identity, independent of its unused trainable row.
        matrices = torch.cat([self.identity[None], matrices[1:]], dim=0)
        for position in range(tokens.shape[1]):
            state = torch.bmm(state[:, None], matrices[tokens[:, position]]).squeeze(1)
        probabilities = state @ decoder
        return probabilities.clamp_min(1e-12).log()

    @torch.no_grad()
    def hard_eval(self, tokens, starts, task_ids):
        """Return hard-program terminal logits, suitable for argmax/accuracy."""
        return self.forward(tokens, starts, task_ids, hard=True)


def parameter_counts(model):
    """Actual stored weights plus requires-grad/effective masked scalar counts."""
    if hasattr(model, "parameter_counts"):
        return model.parameter_counts()
    return dict(total=sum(p.numel() for p in model.parameters()),
                requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad))
