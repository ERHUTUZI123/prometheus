# A consequential cost boundary of the raw-stream witness

RECONSTRUCTION / ATTACK by root, before any executable probe or training. This is a counterexample to the efficiency of **the explicit v1 witness**, not an impossibility result for pair-local generated laws or learning. No new measurement.

## A linear-size closed typed family

Write `let f = r in b` only as reader notation for the raw source application `(lambda f.b) r`; the initializer still receives its literal prefix-token chain. Use distinct names f_1,...,f_k,u_1,...,u_k and a ground atom a. Consider

    let f_1 = (lambda u_1. C a) in
    let f_2 = (lambda u_2. V f_1) in
    ...
    let f_k = (lambda u_k. V f_(k-1)) in
    C a.

No recursion, supplied module or cyclic value occurs. A simple type exists: f_1 has type alpha_1 -> Atom, f_i has type alpha_i -> type(f_(i-1)). Type annotations are absent from the input and are not used by the witness. The number of raw tokens is O(k), with atom references charged by their bits. The actual answer is always a. This is not proposed as an acquisition benchmark; a source shortcut can solve this family and would not demonstrate evaluator learning.

## Exact representation recurrence

Let E_i be the owned passive environment immediately after applying the i-th let binder, and N_i its constructor-node count, excluding the current syntax/continuation. E_0=Empty, N_0=1. At each source App, v1's Eval/App duplicates E_(i-1). The function `lambda f_i.body` captures one copy, and the argument closure captures the other. Apply makes

    E_i = Frame_f_i[
              Closure_u_i[one-node-body, E_(i-1)^(left)],
              E_(i-1)^(right)] .

The two superscripts are distinct physical trees. The body is Atom_a for i=1 and Var_f_(i-1) thereafter. Thus

    N_i = 2 N_(i-1) + 3 = 4*2^i - 3.

Completing each environment copy before consuming it is a permitted explanatory schedule; after the k-th Apply it leaves N_k live data nodes. The copying events alone across the k App rules are

    sum_(i=1)^k N_(i-1) = 4(2^k-1)-3k.

Even if a streaming schedule avoids that peak, Erase at the final Atom recursively visits the same generated owned environment; no unimplemented sharing/copy fusion may be assumed. No claim of identical peak space across schedules is made. Neural emitter work, parsing and continuation work add costs. This recurrence is a structural derivation from the actual rows, not an experimental runtime result.

For the frozen deterministic witness under **full cleanup**, there is a stronger schedule statement. This closed typed family has a finite finish-copy-first normalization. Each legal principal-pair rule preserves its boundary and different active pairs are disjoint. The standard strong-diamond/random-descent argument therefore gives the same complete event count under any maximal normalization, although its peak live space may differ. Hence streaming does not remove the exponential **total interaction work** of this witness. Early-output observation, modified copy/erase fusion, or censored runs are different semantics and do not inherit this conclusion.

## Concrete conventional alternative

A conventional lexical closure interpreter can store immutable environment frames in a heap, put a parent pointer and a captured-environment pointer into each frame/closure, and share the already-built immutable E_(i-1). Each let then adds O(1) heap objects and pointer operations; this family takes O(k) representation work and O(k) space. It uses the same raw program and no extra task truth. If it is supplied as a handwritten interpreter, it does not have equal acquisition cost to the proposed learner, so this is an execution-organization counterexample only. NRAM's supplied read/write/pointer arithmetic provides concrete machinery on which a learner could represent such heap sharing, but its original experiments do not establish acquisition of this evaluator.

## Constructive consequence

The witness has exposed the scope/copy discipline honestly, but it cannot be the evidence for an honest efficiency advantage. Stable learned laws and exact links do not alone fix owned-tree expansion. A serious next formation test must either allow the learner to acquire a genuinely sharing or selectively capturing protocol, or state the copying limitation and test a different consequential advantage. Merely increasing fuel, allowing longer training, or reporting per-collision constant cost does not repair it. Efficient sharing is itself known, and its hand implementation would remain a construction rather than an acquired new principle. No automatic abandonment of the architecture follows.
