# Conventional neural alternatives to generated local execution

SUPPORT artifact, same after014 campaign. Effective mode unknown. Parent reports fork-none with a seeded context; no independent-context claim. No experiments, installations or model training. Read candidate-v0.md and root-bounded-semantics.md. raw-stream-v1.md was not yet present at first checkpoint; comparison below addresses the parent's specified revision: only a raw input sequence graph, no parse graph/global encoder, and the same learned pair emitter must form and execute binding/copy organization.

**Judgment:** Neural GPU is the strongest of these checked architecture-level comparators. It already learns a shared local execution transition from raw symbolic streams and output labels without a supplied symbolic evaluator or copying algorithm. The candidate's potentially meaningful difference is acquired exact communication/binding topology and local graph surgery. “Learn execution rather than use an interpreter” alone does not distinguish it from Neural GPU. NPS and NTM expose two other substantial supplied organizations, without granting them the candidate's new emitter API.

## 1. Neural Production Systems — SOURCE

[Didolkar et al., Neural Production Systems](https://proceedings.neurips.cc/paper/2021/file/d785bf9067f8af9e078b93cf26de2b54-Paper.pdf), printed pp.4–5, §3.1–3.2; [arXiv v3](https://arxiv.org/pdf/2103.01937), Algorithm1 p.15 and coordinate-arithmetic details pp.18–19, actually read.

State is M entity-slot vectors V_j. Each of N rules has embedding R_i and MLP_i. Sequential selection uses (r,p)=argmax_{i,j}(q_j k_i+Gumbel), then contextual slot c via another attention selection. Execution is

    V_p <- V_p + MLP_r([V_p,V_c]); other slots unchanged.

Straight-through Gumbel selection and rule weights learn through end-to-end backpropagation. Rule/slot counts and selection/update machinery are supplied. Step1's entity encoder is external to NPS; learned attention provides rule-to-entity binding. No port allocation/reconnection is specified. Copy-like behavior could be learned in an MLP; no named general function-copy operation is supplied.

Important feedback boundary: the coordinate-arithmetic experiment supplies output coordinates to the rule/slot selector, although the chosen rule applies only to input coordinates. Its operation recovery is therefore not evidence for terminal-output-only evaluator induction. Other tasks have different encoders/feedback. Unread: full experimental suite, code and optimization implementation. No general higher-order interpretation result claimed.

## 2. Neural GPU — SOURCE

[Kaiser and Sutskever, Neural GPUs Learn Algorithms, v3](https://arxiv.org/pdf/1511.08228), printed pp.3–4 §2, pp.6–7 §3.3 and pp.7–8 discussion, actually read. State s has shape [w,n,m]; raw symbol embeddings occupy one column, remaining cells start zero. Its learned transition is

    CGRU(s) = u*s + (1-u)*tanh(U conv (r*s)+B),
    u=sigmoid(U' conv s+B'), r=sigmoid(U'' conv s+B'').

Compose l CGRUs per round, for n rounds; a linear output projection yields per-position token logits. Training uses output cross-entropy, backpropagation/Adam, curriculum, noise, dropout, and parameter-sharing relaxation before consolidating shared parameters. No parse/binding graph, symbolic evaluator, call stack or copy routine is supplied. Grid topology, convolution/gating, embedding layout and output interface are designed; algorithms are learned neural dynamics.

Reported long-length generalization depends on substantial search: 729 configurations, with only a few generalizing to the longest cases. Data duplication is studied; higher-order executable-function copying is not established. Unread: released implementation and complete optimization details. Fixed-grid local computation is already a serious precedent; its arithmetic examples are not evidence it solves this new task family.

## 3. Neural Turing Machine — SOURCE

[Graves et al., Neural Turing Machines, v2](https://arxiv.org/pdf/1410.5401), printed pp.5–10 §§3–4.1, pp.21–22 §4.6, actually read. State includes controller, memory matrix M and attention weights. Supplied read/write equations are

    r = sum_i w(i)M(i),
    M_new(i)=M_old(i)*(1-w(i)e)+w(i)a.       Eq2–4

Addressing supplies cosine-content softmax, interpolation with previous weights, circular shift and sharpening (Eq5–9). A learned controller emits their parameters. Training uses supervised output cross-entropy, backpropagation and RMSProp; episodic state resets. Memory size, head count and shift range are designed resources.

Thus raw memory read/erase/add/address primitives are supplied; storing variables, retrieving them and sequencing copy behavior are acquired. The paper does not supply a high-level COPY algorithm. It also does not demonstrate autonomous discovery of a higher-order evaluator or copying executable binding structure. Dense attention touches memory broadly; count that work rather than price a read as constant-time. Unread: implementation, all experiments, and exact numerical conditioning.

## Constructive coverage and differences — RECONSTRUCTION

There are three different boundaries, rather than a division between “learning systems” and “supplied interpreters”:

| Method | Learned organization relevant here | Fixed substrate relevant here |
|---|---|---|
| NPS | contents/use of sparse binary rules and their bindings to slots | slot attention, rule bank, residual slot update |
| Neural GPU | shared local state-transition computation over a raw sequence | grid adjacency and gated convolution |
| NTM | controller program, memory use and addressing choices | memory addressing/read/erase/add algebra |
| proposed raw-stream emitter | initial binding/copy/execution graph and pair-conditioned local rules | port graph API, allocator, pairing legality, scheduler, neural emitter algebra |

The graph API is a supplied execution substrate just as convolution or memory update is. The distinctive testable hypothesis is whether learning over exact mutable links makes reusable computational organization easier or gives an honest total-work/space advantage. It is insufficient to show the emitter can implement beta, or that the other architectures have no named beta primitive. Conversely, adding the candidate's exact reconnection/emission API to NPS would define a new graph-emitter comparator; it would not show that original NPS already contains the mechanism.

## Concrete matched comparator proposal — AGENT PROPOSAL

Use an actual **Neural GPU with unchanged CGRU transition** as the first conventional comparator. Input the identical raw token serialization and target serialization, with the same token-wise embedding lookup available to the graph machine. No parser, scope labels, binding pointers, reduction traces or subroutine labels go to either learner. Both receive the same task examples and terminal output loss. If the candidate is restricted to scalar exact-match reward, give that same restriction to the comparator; do not quietly give it token cross-entropy. If token targets/gradients are allowed, make them available equally.

Pad to a frozen maximum output/workspace length in both machines and use the same output alphabet/end marker. Match persistent parameter bits and peak dynamic state bits within declared tolerances, including candidate pointer/port/queue storage and Neural GPU auxiliary gates. Use identical finite precision. The comparator's recurrent round cap T(n) may follow the candidate's external execution envelope; this is an explicitly declared timing variant of the original n-round model, not a new learned K or rewrite machine.

Charge arithmetic and training work, not only sequential steps: a dense grid round updates every cell while a pair rewrite updates one bounded neighborhood. Record total cell updates, neural arithmetic, rewires/allocations, encoding/decoding, failed training and hyperparameter search. Give both the same training-data/query budget, tuning-work ceiling and declared curriculum; original paper's 729-way search cannot be free background work. Do not force matched wall time to conceal parallel hardware differences.

Hold out increased input length, increased binding depth, repeated use of the same constructed function, and unseen compositions separately. Overall output accuracy measures task acquisition. Mechanism evidence additionally requires explicit interventions on acquired links/copy structures, with capability-matched sham controls. Neural GPU success would refute an unqualified necessity claim for mutable graph topology. Candidate success alone would not prove new organization or better acquisition.

This is a prospective comparison, not an experimental authorization or frozen benchmark. An empirical win would remain scoped to these resource/feedback conditions. No proof currently establishes a learning-cost or execution-cost separation.

## What matters before implementing

Read the exact raw-stream revision when saved. Freeze its permitted terminal observer, target supervision, finite precision, scheduling and output protocol before selecting comparator dimensions. If the graph learner receives an oracle normalizer, scope-resolving loader, or free semantic wire labels, the matched comparison changes and cannot support acquisition of that supplied work. If both only see raw strings, binding and executable-copy acquisition remain real unresolved learning burdens; conventional sources do not settle them by analogy.
