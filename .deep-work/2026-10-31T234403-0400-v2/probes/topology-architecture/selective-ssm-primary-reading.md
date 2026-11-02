# Selective SSM comparator: original reading and implementation boundary

2026-11-01, same V2 development. No training, imports of downloaded code,
installation or SSH was performed for this artifact. Existing negative results
are preserved. This comparator is not an accuracy target or a reason to tune
baselines until their results become favorable.

## Original paper actually entered

Albert Gu and Tri Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
Spaces*, [arXiv:2312.00752v2](https://arxiv.org/html/2312.00752v2), §§2–3.5,
especially Algorithm 2, equations (2), (4), and §3.4. The checked selection rule
makes the step size and write/read maps input-dependent. The recurrence can then
selectively retain or overwrite state. The paper's efficient implementation
requires hardware-aware scan, fusion and recomputation. Those optimizations are
not reproduced here. The backbone permits ordinary learned arithmetic and
continuous recurrent state; its mechanism provides a relevant conventional
control for context-dependent operation calling. No published task result or
speedup is transferred to our terminal-feedback benchmark.

## Official implementation actually entered

Read author repository [mamba_simple.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py),
constructor and forward/step paths, and
[selective_scan_interface.py](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py),
the real-valued `selective_scan_ref` path, on the above date. These mutable `main`
URLs are reading locations, not a claim of a frozen official-source revision.
The implementation uses negative exponential rates, low-rank step-size
projection with positive softplus step, input-dependent write/read vectors,
causal depthwise convolution, a skip term and SiLU output gating. The reference
input injection is `delta * B * u`; it is not the complete exact-ZOH input
coefficient of paper equation (4). This distinction is implemented explicitly.

## Our newly written reference adaptation

`selective_ssm_baseline.py` exposes `TerminalSelectiveSSM` and reusable
`SelectiveSSMBackbone`. Defaults are width64, three layers, state-size16,
expansion2 and convolution-width4. A learned public initial-state embedding is
one causal prefix, followed by learned token embeddings; the last valid output
predicts the terminal label. It receives no task ID, intermediate state,
operator table or route supervision. Padding only uses supplied lengths.

For channel d and latent coordinate n, the implemented core is

$$
\Delta_t=\operatorname{softplus}(W_\Delta s_\Delta(x_t)+b_\Delta),\qquad
A_{dn}=-\exp(a_{dn}),
$$
$$
h_{tdn}=\exp(\Delta_{td} A_{dn})h_{t-1,dn}
       +\Delta_{td}B_{tn}x_{td},
\qquad y_{td}=\sum_n C_{tn}h_{tdn}+D_dx_{td}.
$$

The gated, projected y joins a residual path. RMS normalization precedes each
block. Carry starts at zero for each example, and inactive suffixes cannot
change it. All weights train only through terminal loss if Root schedules a
run. There is no replay, protected topology or autonomous parameter growth.

This is **a source-grounded Mamba-style selective-SSM reference adaptation,
not official Mamba, not Mamba-2 and not a speed/memory replication**. The plain
PyTorch sequential loop exposes its expensive autograd intermediates; costs
must include actual CUDA peaks and wall time. Forward MAC proxies are named
approximations excluding nonlinearities, backward and backend workspace.
The official implementation remains preferable for an optimized-Mamba claim
if Root verifies it is available in the authorized environment.

## Readiness and permitted next integration

Only AST parsing has executed. No forward/backward, checkpoint round-trip,
numerical equivalence or training success has been observed. Public API:

```python
model = TerminalSelectiveSSM(n_states=8, n_tokens=15, max_length=48)
logits = model(initial, tokens, lengths)
model.proxy(batch_size, supplied_width, sum_lengths)
payload = model.checkpoint()
restored = TerminalSelectiveSSM.from_checkpoint(payload)
```

Factories consume random draws; exact resumed training must restore RNG after
construction. Root may integrate this new arm in a separately frozen driver;
the already frozen contextual-init comparison is unchanged. The current
source is a concrete conventional comparator, not evidence of superiority.
