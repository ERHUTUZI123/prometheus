# Independent manual audit of raw-stream v1

SUPPORT artifact, same after014 campaign. Read candidates/after014-higher-order/raw-stream-v1.md in full, including §§5–7, after it was saved. Also entered root-bounded-semantics.md and the actual NRAM original method. No machine execution, port-validator code, training or experiment. Effective mode unknown; no new independence claim from this seeded context. Candidate left unchanged.

**Judgment:** I found no invalid port matching or scope/copy transition in the successful-path witness after applying the author's explicit polarity assignment. The handwritten protocol genuinely constructs and executes its parser/closure machinery; it does not call a hidden host parser. This is a manual reconstruction of a designed evaluator, not verification of arbitrary emissions or evidence that terminal-feedback learning acquires it. The most consequential remaining interface issues are the terminal/garbage stopping rule, non-idempotent reuse discrimination, and honest nominal-information matching to conventional neural learners.

## Actual polarity and port audit — RECONSTRUCTION

The coherent assignment is:

- Positive: Tok, End, AST constructors, Empty, Frame, Closure, Await frames, PFinish, KArg, KApply, Pair, Done.
- Negative: Start, P, ReadL/ReadV/ReadC, PReturn, PCheckEnd, Eval, Ret, Lookup, UnpackArg, Apply, D, Erase.

PReturn and Ret are return **controllers**, not the positive continuation frames they activate. Under this interpretation every displayed active pair has opposite principal polarity. The broad prose about positive continuations is potentially easy to misread, but it does not require a semantically different rule. Implementations should enumerate signs explicitly.

I checked every parser row, every evaluator row, and the r=0,1,2 copy/erase templates. Each old external boundary wire occurs once; each new internal wire occurs twice. New-agent counts stay within four; no node exceeds two auxiliaries. In particular:

- Eval/App uses four arity-two nodes. The old environment boundary e goes to D's principal; its two outputs feed the function Eval and the Pair saved for argument evaluation. Function/argument syntax and continuation are each retained exactly once.
- Ret/KArg transports the completed function value into UnpackArg, which later puts it in KApply. Ret/KApply transports the completed argument to Apply. These are legal bounded rewires, not host calls.
- Apply/Closure connects Eval to the closure's owned body and a fresh Frame whose parent is the **captured** environment. It never obtains the caller environment implicitly.
- Matching Lookup returns the frame value and erases its unused parent. Nonmatching Lookup erases that value and continues down the parent. Neither duplicates an unowned pointer.
- D/T creates two new constructors and one D per child. Parent outputs connect to clone principals; original child boundaries connect to the new D principals. Output consumers can wait on D auxiliary wires without creating an invalid active pair.

These checks address the actual table, not just a generic claim that interaction nets are well behaved. No executable validator was run, and no theorem about a learned, malformed rule set follows.

## Scope, ownership and cost audit — RECONSTRUCTION

The raw worked token chain parses to the printed AST. Its prefix arity choices create AwaitFun/AwaitArg/AwaitLam frames through the stated rows. There is no supplied occurrence-to-binder alias in that chain. The grammar and lexical labels are designed target semantics, but their parsing work is performed by the constructed rule set.

A useful simultaneous induction is: every evaluator owns one syntax tree, one passive environment tree and one continuation endpoint; every closure owns a passive body and captured environment tree; every Frame owns a value and parent environment. Starting with Empty, Eval/App branches only through D. Apply consumes its closure/argument and creates one owned Frame. No letrec or cyclic environment is introduced. Thus a later D receives only passive constructor trees, not an active evaluator or arbitrary executable net. The streamability of copying does not change this invariant: an unfinished copy presents a pending data endpoint, not permission to copy its control actor.

Lookup's first equal-name frame gives lexical shadowing. The copied F closure captures x=a before the caller creates inner x=b. Both later applications attach their argument beneath that captured environment, so both return a. The hand trace is consistent with the table. Global name permutation covariance alone would not imply any of these lexical facts.

The stated algebraic costs check for finite trees. A tree has sum of constructor arities N-1. Therefore cumulative new agents in copying are sum_v(2+arity(v))=3N-1, and copying takes N collisions. Final live cloned data are 2N nodes; 3N-1 is cumulative allocation, not final size. Erasure generates N-1 further erasers and takes N collisions. Parser events are n token consumptions, m expression returns, Start, and End check: n+m+2. This does not bound total evaluation cost; environment copying can grow exponentially.

### Stopping/observer obligation

The raw table may connect an output Atom before independent Erase/D work has completed. Freeze one of these semantics before implementation:

1. Drain the entire worklist to the declared normal form, then observe the output; blocked/fuel-disabled garbage has an explicit failure policy.
2. Declare a lazy early-output semantics and count retained garbage plus eventual cleanup/next-episode reset costs explicitly.

Stopping immediately at a readable output must not be presented as full-normal-form execution or inherit the uncensored schedule statement. root-bounded-semantics.md already excludes early observation that ignores active garbage from its exact property. This is currently an interface obligation, not a discovered incorrect row.

## What the example does and does not discriminate

The captured function is F=lambda y.x. It is constant, so the result a distinguishes captured lexical scope from a dynamic-scope implementation but **does not require two extensional uses** of F. Replacing F(F b) by one F call yields the same answer. The witness really has two calls in its explicit trace; terminal evidence alone cannot infer that trace.

An unexecuted useful contrast is Church negation N=lambda b.lambda p.lambda q.b q p with T=lambda p.lambda q.p:

    (lambda f. (f (f T)) a b) N  -> a,
    (N T) a b                    -> b.

This distinguishes double application from one application. Varying Boolean inputs, repetition parity, ignored arguments and shadowing is needed to prevent one syntactic output shortcut from fitting the family. It still does not prove physical copying: a sharing evaluator can reuse a function, and an algebraic simplifier can compute the answer differently. An organization claim needs interventions on the actual formed copies/links plus capability-matched controls. No new test is authorized here.

## NRAM actual original boundary — SOURCE

[Kurach, Andrychowicz and Sutskever, Neural Random-Access Machines, v3](https://arxiv.org/pdf/1511.06392), printed pp.2–7 §§3–4.3, read. A learned controller chooses fuzzy circuits over fixed modules and updates register distributions through Eq1–3. Controller inputs are register zero-probabilities; it does not directly read whole register distributions. Memory READ is M^T p; WRITE modifies addressed cells. Integer arithmetic/comparison, constants, READ and WRITE are supplied modules. Only controller parameters learn. Training uses raw I/O, negative log-likelihood/backpropagation/Adam, with no memory-access trace supervision. Discretization can give efficient integer memory access, but its success is conditional; training uses costly distributions/matrix operations. The paper studies array/list/tree tasks, not this raw higher-order evaluator. Unread: code, all appendices, full hyperparameter search and a proof that learned solutions implement lexical interpretation.

## Concrete same-information rival and its limits — AGENT PROPOSAL

An NRAM comparator should receive the raw tokens in an ordinary sequential memory layout plus empty scratch cells. It receives no AST, scope links, closure record, parsing stack or reduction trace. Keep its fixed READ/WRITE/arithmetic/comparison architecture; train its controller on the same input/target pairs and curriculum. It can learn to allocate records in scratch, store names and parent/value addresses, chase them, and copy records using its existing primitives. This is a precise available search space, **not** a claim that the original paper learned this evaluator or that training cost is equal. The comparator does not get a new pair-graph K.

The graph machine already receives exact allocation/connection primitives. NRAM's mutable addressed records can express nonlocal references, so a candidate win over fixed-grid Neural GPU alone would not isolate the value of mutable communication topology. NRAM is consequently the useful second comparator; its centralized controller and fixed arithmetic module algebra differ from the proposed learned pair-local transition semantics.

Two matching issues cannot be hidden:

- **Opaque atoms:** original NRAM treats values as integers and permits arithmetic/order; raw v1's atoms expose only equality/reference transport. Original Neural GPU uses a finite embedding vocabulary and does not guarantee transport of genuinely unseen opaque atoms. Either use a declared finite-vocabulary condition, give both equal integer information, or build explicitly labelled nominal variants with atom payloads restricted to equality/copy. A typed nominal NRAM variant is a prospective modification, not an existing published result. Random renaming helps diagnose shortcuts but does not by itself make the APIs identical.
- **Feedback:** if target atom identity is available as an I/O label, NRAM can use its ordinary differentiable terminal likelihood and Neural GPU its output likelihood. Different gradient routes from the same label are legitimate method differences. If the environment exposes only a returned exact-match reward while hiding target identity, their usual supervised losses receive extra information; use a clearly declared reward-trained variant instead. Freeze which interface raw v1 actually means.

Match all input/output handling, parameter/state bits, finite precision, peak storage, maximum inference work, task queries, training work, curricula and search/tuning budgets. Count NRAM's fuzzy matrix/distribution work during formation and verify any discretized runtime separately. Count every candidate pointer/queue/atom reference and all emitted copying/erasure. Equal step counts do not equalize work.

**Result of this audit:** successful-path wiring and passive-copy ownership survive manual attack; no hidden host parser/evaluator identified. Acquisition remains unexecuted and unexplained relative to these actual neural alternatives. The next meaningful decision is an acquisition rule and information-matched comparison, after the terminal/nominal interfaces are frozen, not a correctness or novelty certificate for the hand evaluator.
