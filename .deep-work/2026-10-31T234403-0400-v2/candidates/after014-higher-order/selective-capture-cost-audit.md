# Exact selective capture: cost repair and remaining owned-tree expansion

RECONSTRUCTION / mathematical support only. Read root-environment-growth.md and actual acquisition-v2.md §§1–4, including FV/Remove/Insert/Union/Project rows. No code, execution, training or literature expansion. Candidate files unchanged. The thunked two-predecessor construction was derived in this support pass and independently proposed by root; the calculations below check the concrete construction, not a claim of mathematical independence from model agreement.

**Judgment:** exact capture at **every** lambda changes the original family from exponential to linear retained environments. With v2's unchanged Eval/App copying and actual Project implementation, total work is still quadratic there. Exact free-variable capture cannot generally eliminate expansion of recursively owned closure values: two different necessary free bindings can own overlapping closure ancestry. The fixed-type thunk family below gives a Fibonacci size recurrence even though a shared immutable heap can build and execute it with linear representation/selected-call work.

## Size conventions

Count every passive constructor node in a closure/environment, including each closure's body syntax, nested captured closures, Frame and Empty. Atom payload/reference bits are additional. Current syntax, continuations, FV/set controllers, pending copies and garbage are excluded from the **retained environment snapshot**, and included when charging work/peak storage. An environment's size is not just its number of Frames.

A source `let f=r in b` means the actual AST `App(Lam_f(b),r)` and its raw prefix tokens. No let operation or type annotation is supplied. All equal-looking values below are physically distinct owned trees unless sharing is expressly introduced as the rival.

## 1. Root's original chain under actual v2 capture

The source is the one in root-environment-growth.md:

    let f1 = (lambda u1. C a) in
    let f2 = (lambda u2. V f1) in ...
    let fk = (lambda uk. V f_(k-1)) in C a.

It has k let applications, 4k+1 AST nodes, and 7k+2 raw tokens excluding End/Start. Types exist as stated by root; they grow with the chain, while type annotations are absent from input. With distinct names represented by references, structural token length is O(k), or O(k log k) bits under ordinary indexed-reference accounting.

The crucial closure is the **let function**, not only its argument. Each `lambda fi.remaining_body` is closed in its ambient environment: the remaining source needs fi, which that lambda binds, and later names are internally bound. V2 therefore captures Empty for it. Keeping the old ambient parent would analyze a different, weaker repair.

Let Ci be the owned value size of fi, and Ni the whole environment immediately after applying that let binder. Exact v2 Project produces:

    C1 = |Closure_u1[Atom_a,Empty]| = 3,
    Ci = 1 Closure + 1 Var + 1 Frame + C_(i-1) + 1 Empty
       = C_(i-1)+4 = 4i-1,                 i>=2;
    Ei = Frame_fi[Fi,Empty],
    Ni = Ci+2 = 4i+1;                     N0=1.

Thus root's proposed recurrence is correct. Selective capture of only the argument closures, while leaving let-function captures untrimmed, would instead retain historical bindings and yield quadratic environments; that is **not** the all-lambda v2 rule.

### Actual copying cost, including Project

Keeping the stated v2 rows, D still receives entire owned trees before unwanted data are erased. Count D/constructor collisions, not only top-level D starts:

| Source of copying | Exact sum for k>=1 |
|---|---:|
| k Eval/App environment copies | sum_i N_(i-1) = 2k²-k |
| Eval/Lam preservation of let-function bodies | sum_i [4(k-i)+1] = 2k²-k |
| Eval/Lam preservation of argument bodies | k |
| Project's singleton captures for arguments i=2..k | sum_(i=2..k) N_(i-1) = 2k²-k-1 |
| **D/constructor collisions in these rows** | **6k²-2k-1** |

The initial constant argument has no free name, so its Project has no environment duplication. Closed let functions also use Project/Nil, not Project/Cons. FV recursively consumes its scan copy and introduces no additional D beyond the body-preservation row. These exact counts assume the displayed tree protocol and count complete normalization; they are handwritten derivations, not executed measurements.

Every copied N-node tree causes N D collisions and 3N-1 cumulative fresh agents across its copying process. FV/list/lookup/erase/continuation work and neural kernel work remain additional. Here all intermediate free-name sets have bounded cardinality, each suffix body has O(k-i) nodes, and each projected environment has O(i) nodes. Consequently those additional traversals are O(k²), supporting v2's total O(k²) claim on this family. Exact numerical total-event constants were not derived.

The retained Ei is O(k), not O(k²). A finish-copy/analysis/cleanup-first schedule can keep the passive data/syntax and traversal stack O(k) on this family. A different schedule may accumulate pending garbage; the recurrence is not a uniform peak-space theorem. Complete-drain semantics prevents unpaid output cleanup.

This repair does not make execution linear. A stronger **branch-sensitive environment splitting** protocol could notice the let function needs nothing and only its argument needs f_(i-1), then move the owned relevant value instead of first duplicating everything. That is a further transition-rule change, not something supplied by closure capture alone. Immutable shared environments already avoid these repeated data copies without acquiring such a new local protocol.

## 2. Fixed-type family with two distinct relevant predecessor closures

Use ground type A, distinct atoms a,b:A, thunk type H=A->A, and a monomorphic Church selector type

    B = H -> H -> H,
    T = lambda l:H. lambda r:H. l.

Define all fi at the **same type B->A**:

    f0 = lambda q:B. C a;
    f1 = lambda q:B. C b;
    fi = lambda q:B.
           (q (lambda v:A. f_(i-1) q)
              (lambda w:A. f_(i-2) q)) (C a),    2<=i<=k.

Nest these as ordinary sequential lets, then finish with `f_k T`. All names are bound; no recursive definition, cyclic value, extra primitive or supplied conditional is used. The type order is fixed: A has order zero, H order one, B order two, and fi order three. The type descriptions do not grow with k.

Call-by-value evaluates both branch arguments only into **thunk closures**. T returns the first closure; only that returned thunk is applied to a. Thus the actual selected semantic call chain is fk,f_(k-1),...,f1, of length O(k). A false selector chooses the other predecessor. Both predecessor names occur freely in fi's body, whereas q,v,w are bound. This avoids the invalid claim that two occurrences of the **same** free name require two captured values: a set captures that name only once.

For k>=1 the nested source has 16k-3 AST nodes and 26k-3 raw tokens, using the exact source grammar and excluding End/Start. Each recursive RHS has one outer lambda and a 13-node body; the final call includes the three-node T term. Reference bits add the usual name-size factor.

Exact FV projection stores two Frames, an Empty, and owned values F_(i-1),F_(i-2). Therefore

    C0=C1=3,
    Ci=C_(i-1)+C_(i-2)+17,                 i>=2.

With Fibonacci numbers F_0=0,F_1=1:

    Ci+17=(C_(i-1)+17)+(C_(i-2)+17),
    Ci=20 F_(i+1)-17 = Theta(phi^i).

The constant 17 is one Closure, 13 body-syntax nodes, two Frames and one Empty. The growth is inside the recursively captured **values**, not a list of irrelevant ambient Frames. Even a perfectly trimmed caller environment cannot collapse two independently owned predecessor trees into a shared ancestor.

For an intermediate binder i<k, the remaining let-function body needs the previous name f_(i-1) after removing its own fi, so the retained environment has two bindings:

    Ei=Frame_fi[Fi,Frame_f_(i-1)[F_(i-1),Empty]],
    Ni=Ci+C_(i-1)+3                        (1<=i<k).

At the final binder the function `lambda fk.fk T` is closed; then Nk=Ck+2. Thus exponential representation is already present in the final closure, before its selected call chain is executed. A materialize-captures-first schedule exposes that many retained live nodes. Streaming may change peak space, but the unmodified D/erase protocol still pays for its generated copies during full cleanup. No copy/erase fusion or hash-consing is silently assumed.

**Limit:** FV relevance is syntactic dependency, not a proof of semantic minimality. Some fi specialize to the same extensional function on T/F, and a whole-program optimizer can exploit identities. This is a counterexample to exact-FV owned-tree capture alone. It is not a lower bound against all representations or a proposed performance benchmark. The heap comparison below does not require those semantic simplifications to win.

## 3. Stronger semantic distinction if needed

One can avoid relying on merely distinct free names by using two values at each level:

    A0=C a; B0=C b;
    Ai=lambda s. s A_(i-1) B_(i-1);
    Bi=lambda s. s B_(i-1) A_(i-1).

Let T0=A and Ti=(T_(i-1)->T_(i-1)->T_(i-1))->T_(i-1). Both Ai,Bi:Ti. Appropriate true/false selector arguments choose either predecessor at each level and can expose either distinct ground atom. Both captured values are behaviorally selectable; each owned value size obeys C0=1, Ci=2C_(i-1)+9=10*2^i-9. There are two constant-size definitions per level, and k selector applications suffice for an output atom. The tradeoff is growing type order; this is a supplementary counterexample, not a fixed-type strengthening. It requires no new experiment.

## 4. Shared immutable heap alternative

A conventional closure evaluator can retain each source body once, allocate each closure/frame as an immutable heap object, and store pointers to predecessor values/environments. Capturing two predecessors creates O(1) records and edges rather than recursively cloning their contents. For either family it builds O(k) heap objects. In the fixed-type thunk family, constant-sized body work and one selected recursive call per level give O(k) execution/temporary-record work after construction, with O(k) storage. Even **untrimmed** immutable environments suffice here: each fi captured the then-current chain whose nearest two relevant bindings are f_(i-1),f_(i-2); the call/thunk adds only a constant number of q/dummy frames. Relevant lookups therefore have bounded depth without an FV oracle or repeated suffix analysis. Garbage accounting can visit the O(k) actual objects/edges rather than unfolded tree copies.

This alternative uses no extra task truth. If hand supplied, it has a different acquisition cost; that qualification does not erase its execution advantage over the hand witness. NRAM's addressed records are an actual substrate on which such sharing can be represented, but its source does not prove acquisition of this evaluator. Pair-local laws could also learn explicit immutable sharing, reference discipline or a different representation; no impossibility result for them follows.

**Conclusion:** v2's root-chain repair is real and its quadratic cost claim survives this audit. Exact selective capture is insufficient as a general efficiency principle because relevant closure dependencies can overlap. A substantive repair must acquire sharing or avoid materializing repeated dependency descriptions, or else declare the owned-tree limitation. None of this shows that the learner discovers FV, Project or sharing from terminal labels: those remain hand target constructions and unexecuted acquisition hypotheses.
