# P after v0 freeze — finite reconstruction, two stronger attacks, and decision

The frozen v0 remains unchanged. No other candidate or old campaign artifact has been read. This note records new reasoning and executed evidence. **Judgment: the interface-acquisition opportunity remains meaningful, but v0 is not ready for a substantive neural experiment. Its exact formulation is restrictive, and its stated approximate loss has an explicit false-success route.**

## Executed finite evidence

The inspected standard-library probe was frozen with its exploratory contract after a separate JSON roundtrip smoke check. It ran once using the coordinator's external isolated runner with a 30-second wall cap; process exit code was 0. The runner used a pre-existing read-only numerical environment, but this script imported only standard-library modules. It denied experiment network access and non-assigned writes externally. This is process containment, not independent data or agent-context isolation.

The full six-state domain and all 720 proposed charts were enumerated:

| Case | Predicted before execution | Observed |
| --- | --- | --- |
| inc only, full transitions | 36 charts | 36 |
| inc and reset, shared chart | 12 charts | 12 |
| nonproduct teacher, full transitions | 0 charts | 0 |
| same nonproduct teacher, only two observed transitions | Some spurious charts | 120 |
| Valid product, every operation word through length 6 and every start | 0 mismatches | 0 / 762 comparisons |

The witness partial-data chart matched both observed transitions yet failed on three other starting states. All three were unobserved sources. This is an explicit counterexample to promoting sampled factorization agreement into a full-domain contract. Joint operation families reduced the relative-coordinate gauge but did not uniquely identify a chart; the remaining charts are equivalent active relabelings and a context swap in this construction.

Artifacts: probes/P/probe-contract.md, finite_factorization.py, probe-freeze.json, serialization-smoke.log, results.jsonl, finite-factorization.log. These results establish the stated finite enumeration and implementation consistency only. They do not establish neural acquisition, inferred factor size, computational advantage, novelty, or improved later learning. The permutation seed is fixed for transparency; it is not an empirical replication count.

## Attack 1: the exact frame demands a conserved context

**RECONSTRUCTION / proof.** Suppose S is a metric space, T:S->S is one of the admitted operations, and for every s, T^n(s) converges to a single s_star in S. Let C:S->Cspace be continuous, with Cspace a metric space, and assume exact frame preservation C(T(s))=C(s) for every s. Then for each n, C(T^n(s))=C(s). By continuity at s_star,

    C(s) = lim_n C(T^n(s)) = C(s_star).

Thus C is constant. Any continuous invertible chart E(s)=(A(s),C(s)) satisfying v0's equation cannot have a nontrivial context factor for this operation. No probabilistic or finite-data premise enters this proof. This is a checked elementary implication, not a claim of novel mathematics.

Example: S=[-1,1]^2, T(x,z)=(0,z/2). All states converge to (0,0), so no nonconstant continuous preserved C exists. Yet relative to future computations that read only x, T contains an ordinary reusable reset-x operation; z is just drifting scratch. The exact full-state formulation therefore rejects some computationally useful extractions merely because irrelevant neural activity changes. If future computations can read z, however, discarding its drift is not behavior-preserving. That difference is the hard issue rather than a technical nuisance.

This does not refute v0 on domains where meaningful memory is truly preserved. It refutes treating its exact frame as a general solution to extraction from arbitrary distributed neural trajectories. A designed machine with paused/copyable context may supply the invariant, but its contribution must then be separated from the learned boundary.

## Attack 2: an invertible chart can hide physical error by rescaling

**ATTACK / explicit counterexample.** The squared latent loss in v0 is not coordinate-invariant. Use the same T(x,z)=(0,z/2), with epsilon>0,

    E_epsilon(x,z) = (x,epsilon*z),
    D_epsilon(a,c) = (a,c/epsilon),
    f(a) = 0.

These maps are exact continuous inverses on their corresponding domains. For s=(x,z),

    E(T(s)) = (0,epsilon*z/2),
    (f(A(s)),C(s)) = (0,epsilon*z).

The latent squared discrepancy is epsilon^2*z^2/4 and tends to zero. But the promoted call is

    D(f(A(s)),C(s)) = (0,z),

whereas the actual teacher is (0,z/2); the physical squared error is z^2/4, independent of epsilon. Exact zero factorization cannot be faked in this way, but arbitrarily small approximate loss can. V0's phrase “compares complete poststates” was not enough: those poststates were compared under a learned, potentially ill-conditioned metric.

A direct physical-state discrepancy

    ||T(D(a,c)) - D(f(a),c)||^2

or an explicit uniform bi-Lipschitz/conditioning bound repairs this particular loophole. Such a repair does not solve the conserved-context obstruction, meaningful-mixture problem, or interface-discovery problem. No numerical test is needed for these algebraic values, and none was executed.

## A possible repair changes the scientific claim

The minimal conceptual repair is to preserve the distinctions that allowed future computation can use, rather than every physical coordinate. Let B be the allowed continuation programs and O the externally grounded observation/readout. Define

    s ~_B t iff O(b(s)) = O(b(t)) for every b in B.

With all finite continuations from an operation set closed under prefixing, this relation is a right congruence: s~t implies U(s)~U(t), because b o U is another allowed continuation. A quotient can therefore support closed computation even when physically irrelevant scratch drifts. This is an established semantic pattern, not a new principle invented here. Using bounded continuations gives only a horizon-limited equivalence; it is not indefinitely preserved automatically.

An acquisition variant would maintain a growing set of *executed distinguishing continuations*, train the candidate replacement against them, and actively search for a continuation that distinguishes teacher from replacement. A found distinction refines the interface or rejects the replacement. The actual update is counterexample-guided refinement, not an abstract appeal to future usefulness. Grounding O and the allowed operation set is designed; neural encoders, implementations and distinguishers would be learned.

But this variant creates a difficult circularity: the representation is compressed relative to the current continuation language, while newly acquired operations can expose distinctions that the old quotient discarded. A call library must either preserve a stronger future-closed abstraction, retain the ability to reconstruct lost distinctions, or accept that later acquisition can force retraining of old interfaces. Renaming this “self-organization” would not resolve it. Frozen modules plus a growing critic do not supply a theorem about future operations outside the critic's language.

Postfreeze primary reading found close precedents in predictive state representations and state-abstraction theory; exact technical records are in sources/P/source-record-postfreeze.md. Consequently, a repair based simply on future-test sufficiency or bisimulation does not create a novelty claim. The potentially consequential remaining question is whether an *expanding executable language and its state distinctions can be jointly acquired* without destroying earlier useful computation. That is a sharper missing mechanism, not a solved follow-up or authorization to run another experiment.

## Decision for the coordinator

Do not treat the positive six-state algebra check as support for expensive P implementation. Preserve the candidate as a concrete learned-interface construction with explicit scope failures. Its strongest simple rival remains an ordinary typed growing program library with neural search. Before choosing P as a main line, one must identify a case where the learned interface changes later acquisition beyond what that rival provides, while avoiding hidden context preservation, conditioning and semantic-oracle advantages. If that cannot be made credible, close this branch rather than polishing the toy.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Exact factorized calls compose over their closed domain | VERIFIED in elementary algebra and exhaustive stated six-state check | v0 equation (2); probes/P/results.jsonl | Existence/acquisition of useful closed domains in a neural system | No executed failure; nonproduct control correctly rejected | Is this restricted frame scientifically worth pursuing against typed library learning?
Arbitrarily low v0 latent loss establishes a useful approximate full-state module | FALSIFIED | Explicit epsilon-chart counterexample above | No remaining link in this general claim | Original latent metric criterion | Require physical error/conditioning if candidate retained
Exact continuous preserved-context factorization applies to arbitrary useful neural segments | FALSIFIED as a general statement | Shared-attractor proof and T(x,z) example | Whether a meaningful restricted machine class suffices | Generality of the exact frame | Keep restricted claim or abandon this representation
Future-test refinement resolves arbitrary new computational demands | OPEN, not claimed | Right-congruence derivation and source comparison | New operations may expose discarded state distinctions | No test executed | Need an actual mechanism for expanding interfaces before a new run
