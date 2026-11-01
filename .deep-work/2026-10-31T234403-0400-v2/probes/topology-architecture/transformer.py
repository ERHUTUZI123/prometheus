"""Small trainable NumPy Transformer; no work is performed at import.

Manual analytical backpropagation, float64, no autodiff or external model.
The learner receives tokens/start/task ID and TERMINAL targets only. It has
no teacher operation table, correct per-token calls, or state labels.
MAIN=TOPOLOGY USER_LOCKED; M0 not passed; M3 ultimate. This is a baseline.
"""
import numpy as np


class Transformer:
    def __init__(self, n_states, vocab, max_len, seed, d_model=32,
                 n_heads=4, n_layers=2, ff_mult=2, adapter=False, n_tasks=3):
        self.n_states, self.vocab = int(n_states), int(vocab)
        self.max_len, self.d_model = int(max_len), int(d_model)
        self.n_heads, self.n_layers = int(n_heads), int(n_layers)
        self.ff_dim, self.n_tasks = int(ff_mult * d_model), int(n_tasks)
        self.adapter, self.stage = bool(adapter), 0
        if (self.n_states < 2 or self.vocab < 2 or self.max_len < 0
                or self.n_heads < 1 or self.n_layers < 1
                or self.d_model % self.n_heads or self.ff_dim < 1
                or self.n_tasks < 1):
            raise ValueError("invalid Transformer dimensions")
        self.head_dim = self.d_model // self.n_heads
        self.adapter_dim = max(4, self.d_model // 4)
        self.eps = 1e-5
        self.params = {}
        rng = np.random.default_rng(seed)

        def normal(name, shape, scale):
            self.params[name] = np.asarray(rng.normal(0., scale, shape), dtype=np.float64)

        def zeros(name, shape):
            self.params[name] = np.zeros(shape, dtype=np.float64)

        def ones(name, shape):
            self.params[name] = np.ones(shape, dtype=np.float64)

        d, f = self.d_model, self.ff_dim
        normal("token_embed", (self.vocab, d), .04)
        normal("start_embed", (self.n_states, d), .04)
        normal("task_embed", (self.n_tasks, d), .04)
        normal("position_embed", (self.max_len + 2, d), .02)
        normal("cls_embed", (d,), .04)
        for layer in range(self.n_layers):
            p = "layer." + str(layer) + "."
            for ln in ("ln1", "ln2"):
                ones(p + ln + ".g", (d,))
                zeros(p + ln + ".b", (d,))
            for projection in ("q", "k", "v", "o"):
                normal(p + projection + ".W", (d, d), 1 / np.sqrt(d))
                zeros(p + projection + ".b", (d,))
            normal(p + "ff1.W", (d, f), np.sqrt(2 / d))
            zeros(p + "ff1.b", (f,))
            normal(p + "ff2.W", (f, d), 1 / np.sqrt(f))
            zeros(p + "ff2.b", (d,))
            if self.adapter:
                for task in range(self.n_tasks):
                    a = "adapter." + str(task) + "." + str(layer) + "."
                    normal(a + "down.W", (d, self.adapter_dim), 1 / np.sqrt(d))
                    zeros(a + "down.b", (self.adapter_dim,))
                    normal(a + "up.W", (self.adapter_dim, d), .01)
                    zeros(a + "up.b", (d,))
        ones("final_ln.g", (d,))
        zeros("final_ln.b", (d,))
        if self.adapter:
            for task in range(self.n_tasks):
                p = "head." + str(task) + "."
                normal(p + "W", (d, self.n_states), 1 / np.sqrt(d))
                zeros(p + "b", (self.n_states,))
        else:
            normal("output.W", (d, self.n_states), 1 / np.sqrt(d))
            zeros("output.b", (self.n_states,))

    def set_stage(self, stage):
        stage = int(stage)
        if not 0 <= stage < self.n_tasks:
            raise ValueError("stage must be in [0,n_tasks)")
        self.stage = stage

    def trainable_names(self):
        if not self.adapter:
            return list(self.params)
        a, h = "adapter." + str(self.stage) + ".", "head." + str(self.stage) + "."
        if self.stage == 0:
            return [name for name in self.params
                    if (not name.startswith(("adapter.", "head."))
                        or name.startswith(a) or name.startswith(h))]
        return [name for name in self.params if name.startswith(a) or name.startswith(h)]

    def _inputs(self, tokens, starts, task_ids):
        tokens = np.asarray(tokens, dtype=np.int64)
        starts = np.asarray(starts, dtype=np.int64)
        tasks = np.asarray(task_ids, dtype=np.int64)
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("tokens must have shape [B,L], B>0")
        b, length = tokens.shape
        if length > self.max_len or starts.shape != (b,) or tasks.shape != (b,):
            raise ValueError("input shapes/max_len disagree")
        if (np.any(tokens < 0) or np.any(tokens >= self.vocab)
                or np.any(starts < 0) or np.any(starts >= self.n_states)
                or np.any(tasks < 0) or np.any(tasks >= self.n_tasks)):
            raise ValueError("token, start or task ID out of range")
        return tokens, starts, tasks

    def _ln(self, x, prefix):
        centered = x - np.mean(x, axis=-1, keepdims=True)
        inv = 1 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + self.eps)
        normalized = centered * inv
        y = normalized * self.params[prefix + ".g"] + self.params[prefix + ".b"]
        return y, (normalized, inv, prefix)

    def _ln_backward(self, dy, cache, grads):
        normalized, inv, prefix = cache
        axes = tuple(range(dy.ndim - 1))
        grads[prefix + ".g"] += np.sum(dy * normalized, axis=axes)
        grads[prefix + ".b"] += np.sum(dy, axis=axes)
        v = dy * self.params[prefix + ".g"]
        return inv * (v - np.mean(v, axis=-1, keepdims=True)
                      - normalized * np.mean(v * normalized, axis=-1, keepdims=True))

    def _linear(self, x, prefix):
        return x @ self.params[prefix + ".W"] + self.params[prefix + ".b"]

    def _linear_backward(self, dy, x, prefix, grads):
        flat_x, flat_dy = x.reshape(-1, x.shape[-1]), dy.reshape(-1, dy.shape[-1])
        grads[prefix + ".W"] += flat_x.T @ flat_dy
        grads[prefix + ".b"] += np.sum(flat_dy, axis=0)
        return dy @ self.params[prefix + ".W"].T

    def _heads(self, x):
        b, s, _ = x.shape
        return x.reshape(b, s, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _unheads(self, x):
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], self.d_model)

    def _forward(self, tokens, starts, task_ids):
        tokens, starts, tasks = self._inputs(tokens, starts, task_ids)
        p = self.params
        b, length = tokens.shape
        seq = length + 2
        mask = np.concatenate((np.ones((b, 1), dtype=bool), tokens != 0,
                               np.ones((b, 1), dtype=bool)), axis=1)
        x = np.empty((b, seq, self.d_model), dtype=np.float64)
        x[:, 0] = p["start_embed"][starts]
        x[:, 1:-1] = p["token_embed"][tokens]
        x[:, -1] = p["cls_embed"]
        x += p["position_embed"][:seq][None]
        x += p["task_embed"][tasks, None, :]
        layers = []
        for layer in range(self.n_layers):
            prefix = "layer." + str(layer) + "."
            u, ln1 = self._ln(x, prefix + "ln1")
            q = self._heads(self._linear(u, prefix + "q"))
            k = self._heads(self._linear(u, prefix + "k"))
            v = self._heads(self._linear(u, prefix + "v"))
            scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(self.head_dim)
            scores = np.where(mask[:, None, None, :], scores, -1e30)
            scores -= np.max(scores, axis=-1, keepdims=True)
            attention = np.exp(scores)
            attention /= np.sum(attention, axis=-1, keepdims=True)
            context = self._unheads(attention @ v)
            residual = x + self._linear(context, prefix + "o")
            z, ln2 = self._ln(residual, prefix + "ln2")
            pre_ff = self._linear(z, prefix + "ff1")
            hidden = np.maximum(pre_ff, 0.)
            block = residual + self._linear(hidden, prefix + "ff2")
            adapter_caches = []
            if self.adapter:
                x = block.copy()
                for task in np.unique(tasks):
                    indices = np.flatnonzero(tasks == task)
                    a = "adapter." + str(int(task)) + "." + str(layer) + "."
                    ax = block[indices]
                    pre = self._linear(ax, a + "down")
                    ah = np.maximum(pre, 0.)
                    x[indices] += self._linear(ah, a + "up")
                    adapter_caches.append((indices, a, ax, pre, ah))
            else:
                x = block
            layers.append((prefix, u, ln1, q, k, v, attention, context,
                           z, ln2, pre_ff, hidden, adapter_caches))
        normalized, final_ln = self._ln(x, "final_ln")
        cls = normalized[:, -1]
        if self.adapter:
            logits = np.empty((b, self.n_states), dtype=np.float64)
            for task in np.unique(tasks):
                ids = np.flatnonzero(tasks == task)
                h = "head." + str(int(task)) + "."
                logits[ids] = cls[ids] @ p[h + "W"] + p[h + "b"]
        else:
            logits = cls @ p["output.W"] + p["output.b"]
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("nonfinite Transformer logits")
        cache = (tokens, starts, tasks, mask, layers, final_ln, cls, x.shape)
        return logits, cache

    def loss_grad(self, tokens, starts, task_ids, targets):
        logits, cache = self._forward(tokens, starts, task_ids)
        tokens, starts, tasks, mask, layers, final_ln, cls, final_shape = cache
        targets = np.asarray(targets, dtype=np.int64)
        b, length = tokens.shape
        if targets.shape != (b,) or np.any(targets < 0) or np.any(targets >= self.n_states):
            raise ValueError("targets must have shape [B] and valid terminal classes")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        log_sum = np.log(np.sum(exp_logits, axis=1))
        loss = float(np.mean(log_sum - shifted[np.arange(b), targets]))
        dlogits = probs.copy()
        dlogits[np.arange(b), targets] -= 1
        dlogits /= b
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        dcls = np.zeros_like(cls)
        if self.adapter:
            for task in np.unique(tasks):
                ids = np.flatnonzero(tasks == task)
                h = "head." + str(int(task)) + "."
                grads[h + "W"] += cls[ids].T @ dlogits[ids]
                grads[h + "b"] += np.sum(dlogits[ids], axis=0)
                dcls[ids] = dlogits[ids] @ self.params[h + "W"].T
        else:
            grads["output.W"] += cls.T @ dlogits
            grads["output.b"] += np.sum(dlogits, axis=0)
            dcls = dlogits @ self.params["output.W"].T
        dx = np.zeros(final_shape, dtype=np.float64)
        dx[:, -1] = dcls
        dx = self._ln_backward(dx, final_ln, grads)
        for layer_cache in reversed(layers):
            (prefix, u, ln1, q, k, v, attention, context,
             z, ln2, pre_ff, hidden, adapter_caches) = layer_cache
            if self.adapter:
                dblock = dx.copy()
                for indices, a, ax, pre, ah in adapter_caches:
                    dah = self._linear_backward(dx[indices], ah, a + "up", grads)
                    dpre = dah * (pre > 0)
                    dblock[indices] += self._linear_backward(dpre, ax, a + "down", grads)
                dx = dblock
            # FF residual and pre-LayerNorm branch.
            dresidual = dx.copy()
            dhidden = self._linear_backward(dx, hidden, prefix + "ff2", grads)
            dpre_ff = dhidden * (pre_ff > 0)
            dz = self._linear_backward(dpre_ff, z, prefix + "ff1", grads)
            dresidual += self._ln_backward(dz, ln2, grads)
            # Attention residual and output projection.
            dx = dresidual.copy()
            dcontext = self._linear_backward(dresidual, context, prefix + "o", grads)
            dc_heads = self._heads(dcontext)
            dattention = dc_heads @ v.swapaxes(-1, -2)
            dv = attention.swapaxes(-1, -2) @ dc_heads
            dscores = attention * (dattention - np.sum(dattention * attention,
                                                       axis=-1, keepdims=True))
            dscores *= mask[:, None, None, :]
            dq = (dscores @ k) / np.sqrt(self.head_dim)
            dk = (dscores.swapaxes(-1, -2) @ q) / np.sqrt(self.head_dim)
            du = self._linear_backward(self._unheads(dq), u, prefix + "q", grads)
            du += self._linear_backward(self._unheads(dk), u, prefix + "k", grads)
            du += self._linear_backward(self._unheads(dv), u, prefix + "v", grads)
            dx += self._ln_backward(du, ln1, grads)
        np.add.at(grads["start_embed"], starts, dx[:, 0])
        np.add.at(grads["token_embed"], tokens.ravel(), dx[:, 1:-1].reshape(-1, self.d_model))
        np.add.at(grads["task_embed"], tasks, np.sum(dx, axis=1))
        grads["position_embed"][:length + 2] += np.sum(dx, axis=0)
        grads["cls_embed"] += np.sum(dx[:, -1], axis=0)
        trainable = set(self.trainable_names())
        for name in grads:
            if name not in trainable:
                grads[name].fill(0.)
        finite = all(np.all(np.isfinite(g)) for g in grads.values())
        if not np.isfinite(loss) or not finite:
            raise FloatingPointError("nonfinite Transformer loss/gradient")
        stats = dict(accuracy=float(np.mean(np.argmax(logits, axis=1) == targets)),
                     logits_finite=True, gradients_finite=finite,
                     logits_min=float(np.min(logits)), logits_max=float(np.max(logits)),
                     max_abs_gradient=max(float(np.max(np.abs(g))) for g in grads.values()),
                     parameter_count=sum(v.size for v in self.params.values()),
                     trainable_parameter_count=sum(self.params[n].size for n in trainable),
                     n_heads=self.n_heads, n_layers=self.n_layers,
                     stage=self.stage, adapter=self.adapter,
                     sequence_slots=length + 2, valid_input_tokens=int(np.sum(tokens != 0)))
        return loss, grads, stats

    def predict(self, tokens, starts, task_ids):
        logits, _ = self._forward(tokens, starts, task_ids)
        return np.argmax(logits, axis=1)

    def gradient_check_batch(self, epsilon=1e-5, checks=12, seed=12345):
        """Optional bounded finite-difference smoke; never called automatically.

        Checks only currently trainable coordinates and restores every value.
        Numerical ReLU kink crossings can cause finite-difference mismatch.
        """
        rng = np.random.default_rng(seed)
        length = min(3, self.max_len)
        tokens = rng.integers(1, self.vocab, (2, length), dtype=np.int64)
        if length > 1:
            tokens[0, -1] = 0
        starts = np.arange(2, dtype=np.int64) % self.n_states
        tasks = np.full(2, self.stage, dtype=np.int64)
        targets = (starts + 1) % self.n_states
        loss, gradients, _ = self.loss_grad(tokens, starts, tasks, targets)
        names = self.trainable_names()
        results = []
        for check in range(int(checks)):
            name = names[check % len(names)]
            value = self.params[name]
            index = tuple(int(rng.integers(size)) for size in value.shape)
            original = float(value[index])
            try:
                value[index] = original + epsilon
                plus = self.loss_grad(tokens, starts, tasks, targets)[0]
                value[index] = original - epsilon
                minus = self.loss_grad(tokens, starts, tasks, targets)[0]
            finally:
                value[index] = original
            numerical = (plus - minus) / (2 * epsilon)
            analytic = float(gradients[name][index])
            scale = max(1e-6, abs(numerical) + abs(analytic))
            results.append(dict(name=name, index=index, analytic=analytic,
                                numerical=float(numerical),
                                absolute_error=abs(numerical - analytic),
                                relative_error=abs(numerical - analytic) / scale))
        return dict(loss=loss, checks=results,
                    max_absolute_error=max((r["absolute_error"] for r in results), default=0.),
                    max_relative_error=max((r["relative_error"] for r in results), default=0.))
