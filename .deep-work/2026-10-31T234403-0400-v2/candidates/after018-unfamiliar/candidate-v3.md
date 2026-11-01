# V3: let acquired operations generate the code coordinates they can execute isometrically

**AGENT CONSTRUCTION / HAND DERIVATIONS, unexecuted.** Same V2 run; root sole coordinator. V2 and its hashes are preserved. No experiment, training, installation, new paid route or MAIN decision. Effective provider mode unknown. This is one substantive repair of operation-compatible representation formation, not a claim of a new coding or Koopman theorem.

## 1. Judgment and substantive change from v2

V2 grew a generic separated code under a human-selected guard radius. Here, an acquired operation changes **which Boolean functions of state become code coordinates**, and the admitted encoded operation must act by a coordinate permutation. That constraint has a specific reliability consequence: old coordinate faults are transported without amplification even when the semantic operation is nonlinear. Restoration still has a physical cost and can fail.

The worked case changes seven linear code coordinates into eleven coordinates containing four particular nonlinear functions. Merely adding the familiar nonlinear helper xy gives a smaller eight-coordinate linear lift, but a single helper fault fans out and causes incorrect restoration. The eleven-coordinate repair instead makes the same nonlinear semantic update a permutation of physical coordinates. A second, noncommuting operation uses the same code without further growth. This is an actual operation-dependent representation rule and worked consequence, rather than adjusting q or adding repetition copies.

The broad cost boundary is severe: exact permutation closure of a rich operation library can require exponentially many physical coordinates, even when each semantic operation has a short ordinary program. Root supplied the cycle/transposition discriminator before this freeze; its explicit proof is reconstructed in §7. Therefore this is a restricted constructive possibility, not a general efficient architecture or a reason to run the example.

## 2. Information, state and executable transition

Retain v2's paid, inspectable incumbent computation. Focus this version on a declared finite local state set S and **bijective** incumbent updates `F_a:S->S`. Their clean reference programs and internal states are available equally to competitors. They may be already learned neural mechanisms; their current semantics, including systematic mistakes, are what the construction preserves. A trusted reference route is not an unknown-environment oracle.

The selected block of state variables is supplied by the incumbent graph in this version. Encoding a whole large machine as one block is not assumed cheap. Overlapping blocks, irreversible updates and new groupings are limitations below.

The representation is a finite list of Boolean observable programs

`O=(phi_1,...,phi_n), phi_j:S->{0,1}`,

with code `Phi(s)=(phi_1(s),...,phi_n(s))`. Runtime state is a possibly faulty vector h in `{0,1}^n`, not the clean s. Store the actual feature expression DAGs, encoding cost, mapping tables for operations, correction circuit and reference/counterexample bank. No unbounded real precision is used.

An admitted operation a has a permutation pi_a such that

`phi_j(F_a(s))=phi_(pi_a(j))(s)` for every admitted s.           (1)

It executes `U_a(h)_j=h_(pi_a(j))`, followed by the declared physical faults and correction schedule. A copied output may be a noisy threshold identity gate, so coordinate routing is not declared physically error-free. Operation selection/multiplexing, routing memory and controls cost additional work. Fixed wiring for a selected operation is a designed resource given equally to rivals.

For any error vector e,

`U_a(Phi(s) XOR e)=Phi(F_a(s)) XOR P_a e`,

and `wt(P_a e)=wt(e)`. This is exact Hamming-error transport, not merely low average clean-state error. It is stronger than a generic linear lift whose columns can fan one fault into many coordinates.

## 3. Actual formation law: operation pullbacks create representation coordinates

Start with a finite injective observable list, supplied or acquired by the current learner, and record its source/cost. Do not credit its initial information as emerging in this repair. Then iterate under fixed limits on coordinates, feature-program size, reference executions and equivalence checking:

1. For each current coordinate phi_j and acquired update F_a, construct the literal candidate observable `psi=phi_j composed_with F_a`. Its values come from executing our own incumbent programs. No teacher supplies the useful nonlinear formula or a desired codeword.
2. Check whether psi equals an existing phi_k on the declared domain. In the finite fully enumerated version this is exact truth-column comparison, with every reference evaluation charged. In a sampled version, it is only an empirical identity. A discrepancy produces an actual state witness and revokes that identity; finite agreement is not promoted to universal correctness.
3. If no existing coordinate matches, append psi, or fit a new finite neural Boolean feature to its internally generated labels. Neural fitting is optional compression, not a source of new semantic information. Retain the reference expression until an exact replacement or a bounded approximation contract is established. Failed fitting and all reference computation count.
4. Repeat until closure is obtained or a real bound is reached. For a finite closed dictionary of distinct functions and bijective F_a, the pullback map is injective: `phi_i composed_with F_a=phi_j composed_with F_a` implies phi_i=phi_j. Hence its map into the same finite dictionary is a permutation, giving (1).
5. Rebuild a physical correction network for the actual new code and test/reason about its actual local fault contract. Code distance alone does not certify a noisy correction circuit. Compare total encoding, execution, repair and formation cost before retaining the new representation version.

The update is structural/model-based learning from an acquired computation. It is not ordinary gradient descent through a hard restorer. If neural gates are used to fit new features or correction circuits, v2's explicit pre-fault Bernoulli decision/score-function procedure remains available; deterministic frozen gates require their own actual assessment. No straight-through gradient is called exact.

Semantic writes are the admitted F_a transitions, which intentionally move between codewords. Representation changes occur between frozen trials through the paid trusted update/version port, with explicit re-encoding. Raw state writes outside that interface can be corrected away. A conventional digital/compiler baseline receives the same port.

### An explicit cost/growth criterion

Closure is a candidate constructor, not mandatory unlimited expansion. An external finite search can retain an unclosed incumbent while comparing a closure proposal. Its score includes reference/equality cost, stored feature programs, initialization encoder cost, per-operation physical work, correction work and actual failed trials on the shared bank. If closure exceeds the budget or only moves cost into encoding/decoding, reject it. The algorithm has a concrete termination path even when useful closure is enormous.

## 4. Why ordinary nonlinear lifting alone is insufficient

Let S be the eight triples `(x,y,z)` over F_2. Consider the bijective nonlinear update

`T(x,y,z)=(x,y,z XOR x*y)`.

Begin with the seven nonzero linear forms `ell_abc=a*x XOR b*y XOR c*z`, `(a,b,c)!=0`. This is an injective seven-coordinate code. For distinct states its Hamming distance is four: a nonzero state difference makes exactly four nonzero linear forms change.

One tempting repair adds only `p=x*y`, creating an eight-coordinate lift. Then T can update each z-containing linear form by XOR with p, while p is unchanged. This is a correct clean-state linear program, but it is not an isometry. There are four z-containing coordinates. One error in p becomes errors in p and all four of those coordinates.

Let s' differ from s only in z. Their clean eight-coordinate codes differ in precisely those four linear coordinates, while p is equal. Therefore the output of this faulty lifted execution is only one bit away from the **wrong** code Phi(s'). A nearest-code restorer will confidently restore the wrong semantic state. Correct clean dynamics and an error-correcting storage representation did not provide operation-compatible reliability.

This counterexample is derived here on paper. It is not a run or a novelty claim about error propagation in linear coding.

## 5. Worked operation-generated code and two noncommuting updates

For c=0, `ell_abc composed_with T=ell_abc`. For c=1,

`ell_ab1 composed_with T=ell_ab1 XOR x*y`.

Exactly four missing functions are therefore proposed by the formation law:

`q_ab=z XOR a*x XOR b*y XOR x*y`, `(a,b) in F_2^2`.

Append those four coordinates. Pulling each q_ab back through T returns ell_ab1. Thus the eleven-coordinate code consists of the seven original linear forms plus these four nonlinear forms, and T swaps four coordinate pairs while fixing the three x/y-only forms. It creates no coordinate fanout. The different operation `z<-z XOR x XOR y` was already accommodated by the original linear dictionary; it would not have generated these new features. The representation choice depends on the actual nonlinear operation, not on a changed fault radius.

Now add the linear reversible update

`C(x,y,z)=(x,y XOR x,z)`.

C permutes the seven nonzero linear forms. On the four new coordinates,

`q_ab composed_with C = q_(a XOR b XOR 1, b)`.

So the same eleven-coordinate dictionary is closed under C as well. T and C do not commute: applying C before T changes the added z term from xy to `x*(y XOR x)=xy XOR x`. The code therefore supports arbitrary words in a noncommuting library, not only repeated use of one involution. Their semantic state set remains finite; this is not evidence of unbounded computation.

### Minimum distance and actual correction circuit

For two states with the same (x,y) and different z, all four z-linear and all four q coordinates differ, giving distance eight. If their (x,y) differ, the original seven-coordinate part contributes four differences. Among the four q coordinates exactly two differ, since their difference is a nonconstant affine function of (a,b). Thus the new code has minimum distance six and corrects any two coordinate errors by nearest-code decoding.

Do not hide the decoder as a fault-free oracle. For a physical construction on this finite eight-state example, define for each clean code c_s a threshold detector

`D_s(h)=1 iff d_H(h,c_s)<=2`.

Hamming distance to a fixed binary word is an affine function of the eleven h bits, so each detector is one finite-precision threshold neuron with fan-in eleven. Every code coordinate is balanced over the eight states: four states have that coordinate one.

For each output coordinate j, give it its **own disjoint** four detectors for the four codewords with c_sj=1, followed by one OR threshold neuron. Within distance two of a valid codeword, exactly its detector can be true; this five-neuron subcircuit emits the correct clean j-th bit. Duplicate the detector computations across output coordinates rather than sharing their unreliable intermediate values. Eleven such subcircuits use **55 threshold neurons**. Shared input bits are covered by the input error bound; internal detector/OR nodes are disjoint by output.

One physical neuron fault anywhere inside this restorer can affect at most one output coordinate. Consequently the following actual contract holds:

- input code has at most one old coordinate error;
- an encoded T or C uses eleven noisy identity/copy outputs following its learned permutation, with at most one new output fault;
- the pre-restoration vector has at most two errors because the permutation does not amplify the old one;
- the 55-neuron restorer has at most one physical neuron fault, so its output has at most one error around the correct next codeword.

This renews radius one for any finite composition as long as every local fault budget holds. It costs eleven stored coordinates, eleven copied/routed outputs plus 55 restoration neurons per corrected update, all detector input wires and the operation-selection circuitry. It is not automatically cheaper than fixed coding. Under independent physical flips with probability p, a conservative pair-fault union bound is `(binom(11,2)+binom(55,2))*H*p^2=1540 H p^2` for H corrected invocations. It becomes vacuous at sufficient depth; no indefinite threshold or reliable final scalar readout is obtained.

There may be cheaper dedicated decoders. This deliberately explicit one establishes physical fault containment rather than asserting that a shared perfect decoder is available. Its structure is a hand witness, not a trained result.

## 6. Relation to operation-induced conflict sets and to actual sources

After the initial Toffoli/naive-lift fault derivation but before this artifact was written, root and the supervisor supplied the Y_v repair suggestion: collect actual possible pre-restoration outputs ending in v; intersecting Y sets expose unavoidable conflicts; sampled radii are witnesses, not certified upper bounds. This is a useful external contribution and is not silently attributed to this worker.

The present repair addresses one cause of Y growth directly: replace a fault-amplifying encoded operation by an isometry through operation-generated coordinate choice. Under a one-fault operation budget, the actual pre-restoration set is contained in a radius-two neighborhood of the next codeword when the incoming set has radius one. This bound is proved from the actual permutation transition, not inferred from an empirical maximum. It does not resolve input-confusion impossibility or arbitrary code conflict.

SOURCE boundaries from originals entered in this pass:

- Function-correcting codes already impose different distances on messages with different desired function values; for a bijective desired function, their correction requirement is ordinary full-message correction. Therefore choosing relevant output distinctions or irregular distances alone is not a new principle. This v3 retains all state in the reversible block and instead constrains **how computation transports coordinate errors**.
- Koopman invariant-observable methods already add nonlinear functions so transformed dynamics close in a selected observable space. The source's analytic/regression closure mechanisms directly undermine any claim that adding xy or pullbacks to linearize an operation is new. Our additional permutation/isometry restriction is specific, but its priority and value remain open.

The actual paper bodies and read locations are in `source-v3.md`. They do not prove v3 acquisition efficiency or noncoverage by code-automorphism/orbit compilers.

## 7. Root-supplied scaling discriminator: exact closure can be prohibitive

This argument was supplied by root before freeze, and is reconstructed here rather than claimed as this worker's invention. Let S have N=2^m states. The short reversible updates `c(s)=s+1 mod N` and tau swapping 0 and 1 generate the entire symmetric group S_N: conjugating tau by powers of c gives adjacent transpositions, which generate all permutations. The semantic descriptions/circuits of these two updates can be polynomial in m; the closure requirement is the source of the following cost.

A nonconstant Boolean coordinate phi is the indicator of a subset A of size k, 1<=k<=N-1. Pullbacks through all permutations produce indicators of **all** k-element subsets. An exactly coordinate-closed dictionary containing phi must therefore contain at least `binom(N,k)>=N` distinct coordinates. Any injective code contains a nonconstant coordinate, so a pure-coordinate-permutation representation of this whole library has at least N coordinates. If it retains an original balanced bit, the orbit has `binom(N,N/2)` coordinates, at least `2^N/(N+1)` by the largest-binomial-coefficient bound.

Thus a compact semantic algorithm library can require an exponential, or much worse balanced-orbit, representation under exact isometric coordinate closure. This is a concrete limitation of **this rule**, not of all learned coding or neural computation. Our eleven-coordinate example is a favorable small orbit. Adding more reversible gates can destroy that advantage.

Allowing signed permutations could merge complementary coordinates and alter small constants; this version uses pure permutations and claims the stated bound only for that version. Context-specific codes, on-demand coordinates, neural decoders or approximate closure can move work elsewhere, but are not supplied as repairs. Each would have to pay actual encoding, decoder, routing and approximation costs. No unexamined cheap contextual extension rescues the generality claim.

## 8. Important scope and the strongest same-signal alternatives

This is a representation/repair rule for an acquired **reversible local block** with a tractable observable orbit. It does not supply isometric encoding of arbitrary irreversible neural updates. A noninjective F can map different observable coordinates to the same pullback, causing fanout or collapse; the permutation proof fails. Reversible embeddings with ancillas/uncomputation cost additional state and operations. No such universal compiler is imported for free.

Multiple overlapping blocks or operations that couple previously separate registers may force a larger joint code and re-encoding. Exact identity checks can require enumeration of the whole admitted state domain; finite observed agreement does not establish closure on unseen states. The neural feature-fitting version adds approximation error and loses the exact invariant until bounded. Faults in routing/selectors/wires must be included in the real fault model. No protected path is hidden as a neuronal result.

The most serious same-information competitor is an ordinary **observable-orbit/code-automorphism compiler** given the same acquired F programs, seed observables and paid reference access. It can perform the exact same pullback construction and generate the same wiring and decoder. The pure formation rule currently has no advantage over that compiler. A second competitor quantizes and applies conventional fault-tolerant coding directly to the incumbent computation; a third jointly learns code and operations with the same reference tuples and all physical faults. Both may be substantially cheaper than orbit closure.

Consequently the actual result is not an established new learning principle. It is a substantive repair of the v2 causal gap: **operation behavior selects nonlinear representation coordinates, and the resulting execution law eliminates a specific source of fault amplification**. The worked improvement is real in its stated mathematical sense, while generality and acquisition economy remain unsupported. Merely naming the compiler neural does not change that judgment.

## 9. Broad prospect and next single decision

The important possibility is that a learner's existing computations can generate an encoding in which their future compositions are mechanically stable, without retraining a whole deep graph. The structural prospect is favorable when the acquired operation library has a small, computationally accessible orbit of task-relevant observables. The explicit S_N boundary says why that cannot be presumed for arbitrary computation.

The weakest step is now precise: finding a **useful intermediate representation between full coordinate-orbit expansion and arbitrary fault-amplifying updates**, at lower total formation/execution cost than same-signal compilation or noise-aware co-training. This v3 has not supplied that intermediate rule. It should not be repaired by merely increasing bits, changing q, hiding contextual decoding or running a tiny Toffoli demonstration.

Root's next decision is whether the specific isometric-representation tradeoff warrants another genuinely different construction, or whether to retain this exact boundary and redirect the main line. This worker requests no experiment and makes no MAIN conclusion.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`Actual operation pullbacks can select new nonlinear code coordinates and yield isometric encoded updates | EXPLICIT CONSTRUCTION / HAND DERIVED | §§3–5, eleven-coordinate T/C code | exact identities/domain; acquisition cost; compiler equivalence | eight-coordinate xy lift amplifies one fault into five | assess restricted orbit-cost prospect`

`The encoded block can renew its correction allowance with faults inside the decoder | HAND RECONSTRUCTION | 55 disjoint-output decoder neurons and radius-one induction §5 | specified fault placement and budgets; no trained acquisition; final scalar floor | shared unprotected decoder would fail containment | independent mathematical/circuit audit, no experiment inferred`

`This exact-isometry rule scales efficiently to broad short-program operation libraries | FALSE in general for stated rule | root-supplied S_N orbit lower bound §7 | restricted small-orbit libraries may remain useful | generic universality/cheap closure prospect rejected | seek actual intermediate rule or stop this branch`
