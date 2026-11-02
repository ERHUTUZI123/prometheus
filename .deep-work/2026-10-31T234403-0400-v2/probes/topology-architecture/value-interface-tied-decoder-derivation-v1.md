# Frozen constructive note: what the soft bridge can and cannot identify

ROOT RECONSTRUCTION, 2026-11-02. This is a derivation artifact, **not another current Method, selected implementation, training result or new experiment allocation**. The selected next experiment remains the function-preserving +1P bridge in `water/method.md`.

## A concrete interface ambiguity

The implemented encoder and decoder are independently learned. Terminal NLL constrains `readout(h_T)`, but does not imply `h_T = encode(y)` or even `argmax readout(encode(y)) = y` on zero-length execution. Subsequent control and memory remain essential: terminal `y=q` does not summarize mode, saved value, address or cells. Replacing the complete carry by `initial(y)` is therefore invalid.

The selected bridge uses these existing learned coordinates and a scalar trained by downstream task loss. For its shared scalar `a`, at zero initialization the exact chain rule is

\[
\left.\frac{\partial L}{\partial a}\right|_{a=0}
=\sum_t \langle \bar h_t, R_\theta(\tilde h_t)-\tilde h_t\rangle,
\]

where `bar h_t` includes all later value, controller and memory uses through BPTT. No intermediate semantic teacher appears. A positive result would still need a causal check: the scalar may select generic conditioning, or a negative coefficient, rather than alignment with semantic input coordinates.

Counterexample to treating the bridge as a retraction: let the decoder assign constant equal probability to two orthogonal encoder prototypes. The bridge target has the same bisector direction for both prototypes, erasing their distinction. Even a tied decoder has a nonzero mixture at finite temperature; exact idempotence does not follow.

## A stronger candidate if this finite test exposes that ambiguity

Let normalized learned prototypes have common norm `sqrt(H)`. A tied decoder could use

\[
\ell_y(h)=\beta\,E_y^\top h/\sqrt H,\qquad \beta>0,
\]

instead of an independent output matrix and independent class biases. For distinct prototypes, at `h=E_i`, Cauchy--Schwarz gives `E_i^T E_i > E_j^T E_i` for every `j != i`. Thus zero-operation initial-state classification has the correct **argmax**, without providing any transition or correct intermediate state. This proves neither one-hot probabilities at finite temperature nor preservation of learned operations during parameter updates. Prototype collisions and full-carry context remain separate obligations.

The corresponding learning rule would still be ordinary terminal NLL/Adam over the same controller, experts and memory. A fresh implementation would remove `Y*(H+1)` independent decoder parameters and add one temperature parameter; this is a resource change, **not the currently selected +1P model**. No previous checkpoint can be claimed function-preserved by simply tying these weights. No new training or numerical claim is made here.

The proposed structural bias is inspectable and has conventional prototype/tied-embedding precedents; no novelty is claimed. It may be useful for tasks whose input values and terminal outputs share an alphabet. Language, continuous inputs and real-data classification need their own measured interface and cost checks. It cannot be assumed to produce broad architecture advantages.

## Decision boundary

First complete the selected bridge's actual 600-step training and unchanged seven-suffix closure evaluation. Do not create additional strength/temperature trials. If the bridge has no useful closure gain, retain that result and use its learned coefficient/behavior to decide whether this exact coordinate-coupling issue warrants a fresh tied-interface construction. This note supplies a concrete next representation and counterexample; it does not automatically authorize or announce that experiment.


## Independent scope correction appended 2026-11-02

`fusion_raw_verify`, independent mathematical reconstruction; no NN execution.
The pre-append artifact SHA was `c564d894d8dbf46f9f3faa7f94ccadcb776bfab9e38a90a69e3601b96b6853e8`.

The constant-uniform-decoder example fails to fix the individual prototypes, so
it refutes a retraction **onto the intended prototype set**. Its norm-preserving
map `R(v)=||v|| u` is nevertheless idempotent onto its own ray. Erasing prototype
distinctions alone is therefore not a counterexample to idempotence.

A finite tied-decoder counterexample is explicit: H=2, beta=1,
E1=(sqrt(2),0), E2=(0,sqrt(2)), and the note's norm-preserving R. Direct stdlib
floating arithmetic gives R(E1)=(1.3741854174694406,0.33408747120228777) and
R(R(E1))=(1.3333891936141764,0.47124649426063275), with L2 residual
0.14309762222751443. This establishes failure of exact idempotence in this
specified example. The distinct, equal, nonzero-norm prototype assumption and
positive temperature still give the note's unique initial-state argmax result;
finite-precision collisions and normalization-epsilon edge cases must be checked.
Neither fact establishes context-independent continuation or full-carry closure.
The actual learned encoder/decoder alignment remains a measurement question.
