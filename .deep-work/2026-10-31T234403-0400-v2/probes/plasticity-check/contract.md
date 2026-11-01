# Coordinator mechanism check — not a novelty or acquisition experiment

CLAIM: For a trainable linear factor block W=A B under simultaneous SGD, the
state (W,S=A A^T,T=B^T B) is sufficient to reproduce its full future SGD for
arbitrary subsequent differentiable losses depending on W. Current W, and even
its initial complete gradient-response operator, need not be sufficient.

GIVEN: A in R^(m*h), B in R^(h*n), G=dL/dW, eta fixed per step. Proposed block
state H=[[S,W],[W^T,T]], and M=[[I,-eta G],[-eta G^T,I]]. Macro update is
H_next=M H M^T. A source of new units [Lout;0] and [0;Lin] adds block-diagonal
PSD D without changing current W. This exact moment construction was independently
derived in coordinator reasoning and by K, but their agreement is not proof.

VERIFIER: One inspected current-run script. Independent micro oracle updates A
and B directly; macro uses the block congruence. Three fixed seeds, h=512,m=3,n=2,
180 online examples, both nonlinear outer readout tanh and trainable preceding
feature tanh(Jx). External targets change each60 examples. Both sides update J
simultaneously from their own outputs. At steps60,120 add explicitly counted new
inactive paths and compare all future states/outputs. No expensive sweep.

Exact adversary: W0=0 with (S,T)=(2I,2I) versus (3I,I) has identical current
function and initial gradient-response operator for every G. Apply eta=1/4,
G1=diag(1,2), then G2=E12. Symbolic rational macro calculation and algebraic micro
factor updates independently compare the second result. This is a witness that
the full learning state contains more than current value/initial response, not
an impossibility proof against neural nets or a superiority claim over optimizers.

RESOURCE: inspected process via run_isolated.py --timeout60; one CPU thread,
no numerical network/secrets/outside writes. JSON smoke then incremental per-step
JSONL and atomic final JSON; save failures. Report dimensions and static state
counts separately from runtime. No physical energy or overall acquisition saving.

NON-GOALS: H is supplied in this check; automatic closure discovery, meta-learning
the growth policy, usefulness over direct W optimizers, general nonlinear-hidden
NN compression, later-task accuracy superiority and novelty are not verified.
Linear factorization redundancy is explicit, and its compression alone cannot
satisfy the paradigm-level objective.

STOP: one finite micro/macro and exact witness check, including any failure;
no retry/tuning or follow-on experiment without a new scientific reason.
