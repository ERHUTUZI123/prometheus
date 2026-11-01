# Constructive attack and repair: growing a relation while retaining its operations

AGENT PROPOSAL + JOINT DERIVATION. v0 remains frozen. Root independently proposed the sheet-growth law; this note derives its preservation conditions and local new operation. The binary-register construction below was independently proposed in this branch. No numerical execution has occurred. Mathematical statements here are informal proofs/reconstructions, not formally verified theorems or empirical results.

## Judgment

One important mathematical possibility survives the attack: a relation can acquire a new branch and a new operation coupling that branch to old computation, while every declared old operation remains exactly the same permutation under a documented identification of states. The old state coordinates and trajectories need not remain fixed. A finite execution corridor, rather than an entire weight tensor, is what must remain protected.

This is not yet an important demonstrated neural learning principle. Classical continuation supplies the execution mechanism; fixed-substrate braid programming already exists. Adding sheets alone is state-capacity expansion. The unresolved scientific residual is whether task feedback can use this geometry to acquire useful *interacting operations*, while preserving existing ones, more effectively than an ordinary learned state machine or growing recurrent model with the same information and complete cost. No current argument establishes that residual.

## Attack 1: protection blocks the proposed ordinary gate gradient

Let parameters s continuously deform both a relation and a closed control loop. Assume throughout s in a connected neighborhood that the relation remains a covering along the whole loop, all roots remain separated, and starting fibers are identified continuously. The terminal branch permutation is a continuous function into a finite discrete set. It is therefore locally constant. This follows equivalently from homotopy lifting.

Consequently an exact loss on branch identity has zero ordinary derivative throughout each safe region. Gradients can improve trajectory length, conditioning, residual error and coordinates; they cannot change the discrete operation while staying inside the protected region. A coordinate endpoint loss can misleadingly decrease by moving the readout coordinates rather than acquiring the requested permutation. Readout semantics must be held through branch identification.

This refutes v0's suggestion that differentiating through safe transported dynamics is by itself an adequate acquisition law for new gate identities. A successful system needs jumps between loop classes, changes of the relation's singular structure, or additional non-topological learning signals. None is free.

Repair: use a finite proposal/selection process for the topology-changing choice, with gradients only for relation fitting and within-class trajectory refinement. Growth can place new singular structure outside old execution corridors. The relation-growth derivation below explains exactly which retention can then be certified; it does not prove efficient proposal search.

## Attack 2: nearest prior mechanisms

SOURCE boundary: Duff et al.'s original monodromy-solver paper, sections 2.1–2.3 and 3, already uses retained roots, parameter paths, induced root correspondences, and an adaptively extended graph of homotopies. Its task is to discover a polynomial system's roots. Give this algorithm our relation and paths and it supplies the same transport mechanism. This is an actual mechanistic match, stronger than saying an ordinary computer can simulate us.

SOURCE boundary: Jipdi et al. 2026, section V.6, explicitly formulates learning as a bounded discrete search over a predefined alphabet of exceptional-point braid gates. Generic claims about learning by loop composition, robustness, or forgetting prevention therefore cannot constitute our residual. We checked the formulation, not the paper's broad guarantees or numerical results. Some of its physical gates also include phases; our classical branch permutations make no quantum claim.

The proposed difference is learning and structurally expanding the *relation that generates available operations*, with preservation attached to existing execution corridors. This difference is describable; its novelty and scientific advantage are not established.

## Central repair: add a sheet and a coupling

Let P(z,u) be a monic degree-n polynomial in z, with coefficients continuous in a complex boundary parameter u. Let Gamma be a finite union of compact old control loops based at u0. Assume every P root is simple on Gamma. Choose a fixed complex r separated from all those roots over Gamma. Start the enlarged relation

    R(z,u)=(z-r)P(z,u)=0.

It adds one resting state but initially no operation connects it to an old sheet. Now propose

    Q_epsilon(z,u)=(z-r)P(z,u)-epsilon phi(u)=0,

where phi is bounded on Gamma and sufficiently regular near the new coupling region. In the witness phi=1. This is a manually designed structural mutation operator; neural acquisition does not invent multiplication or the mutation grammar.

### Scoped preservation lemma

Choose rho>0 smaller than one third of the minimum separation among all n+1 roots of R over Gamma. Around each old root and r take a radius-rho disc. Compactness and simplicity give

    m = inf {|(z-r)P(z,u)| : u in Gamma,
             z on one of these root-disc boundaries} > 0.

Require

    |epsilon| sup_Gamma |phi(u)| < m.

For every s in [0,1], R and Q_(s epsilon) have the same number of roots inside each disc: one, counting multiplicity, by the elementary argument-principle/Rouche comparison. In particular each perturbed root is simple and belongs to a unique original root disc. These local root identifications vary continuously along each old loop. The original root discs themselves undergo exactly P's original permutation; the disc centered at fixed r returns to itself. Hence

    permutation_Q(gamma) = permutation_P(gamma) extended by fixing new,

under the root-disc identification at u0, for every declared old loop gamma. This holds through the whole coupling homotopy. It does not preserve numerical root positions, velocities, path lengths, condition numbers, or arbitrary old loops outside Gamma. It does not ensure that a learned approximate model satisfies the bound. A numerical implementation would need validated bounds or would report only an exploratory observation.

The same argument works with a finite collection of protected compact corridors surrounding those loops. It can permit large changes away from them, while changing nothing at the discrete operation level along them.

### Why the coupling creates a new operation

Suppose outside the protected old corridor there is u* with P(r,u*)=0 and a transverse crossing:

    A=P_z(r,u*) != 0, B=P_u(r,u*) != 0, phi(u*) != 0.

Write z=r+zeta and u=u*+v. To leading order,

    Q = A zeta^2 + B v zeta - epsilon phi(u*) + higher terms.

The local pair of roots has discriminant

    B^2 v^2 + 4 A epsilon phi(u*).

For sufficiently small nonzero epsilon, the transverse double crossing splits generically into two simple branch values

    v = +/- 2 sqrt(-A epsilon phi(u*))/B + O(epsilon).

A small boundary loop around either one exchanges the two coupled local roots. One belongs, under an appropriate handle from u0, to an old sheet and the other to the new sheet. Thus the addition creates an operation with support on old and new computational state, not just a disconnected memory cell.

The qualification is substantial: the new branch-value separation scales as sqrt(|epsilon|). Taking epsilon tiny to make old preservation easy can make resolving the new operation expensive in precision, path steps and conditioning. There is no cost-free compatibility of perfect retention and arbitrarily fine new computation.

## Fully worked growth process

1. Begin with P(z,u)=z^2-u and resting boundary u0=1. The two states are -1,+1. The old control A is u(t)=exp(2 pi i t); it swaps them.

2. Add r=3, phi=1, obtaining Q=(z-3)(z^2-u)-epsilon. Over |u|=1 the old roots have modulus one. Their mutual distance is two and their distance from r is at least two. Choose rho=1/4. On every old-root or new-root disc boundary,

       |(z-3)(z^2-u)| >= (1/4)(7/4)^2 = 49/64.

   Therefore every 0<|epsilon|<49/64 preserves A as a swap of the old two states and a fix of the new one, under the disc identification. This is an explicit uniform bound, not a statement that epsilon looks small.

3. At u0=1 the resting coordinates move. First-order expansions are

       z_old-negative = -1 + epsilon/8 + O(epsilon^2),
       z_old-positive = +1 - epsilon/4 + O(epsilon^2),
       z_new          =  3 + epsilon/8 + O(epsilon^2).

   Retention refers to these continued labels, not the old literal coordinates.

4. The critical conditions Q=Q_z=0 reduce exactly to

       u=3z^2-6z,   2z(z-3)^2=-epsilon.

   For sufficiently small positive epsilon one critical point continues the old one near z=0,u=0. The other two have

       z=3 +/- sqrt(-epsilon/6)+O(epsilon),
       u=9 +/- sqrt(-24 epsilon)+O(epsilon).

   Near u=9, the sheets that at epsilon=0 were z=+sqrt(u) and z=3 meet; the z=-sqrt(u) sheet remains separate. A loop C around one of the split branch values, with a noncrossing handle from u0, exchanges the continued positive and new state. Its exact handle must be part of a numerical contract.

5. Label the resting fiber (1=negative,2=positive,3=new). A=(12), C=(23). Starting in state 1, A then C gives state 3; C then A gives state 2. The previous operation A is retained, and a new interacting generator exists. All six permutations become executable. Nothing in this derivation shows that task feedback efficiently finds C or chooses this growth proposal.

## A scaling construction that does not encode every global state separately

The one-sheet growth proof alone remains a finite-state extension. Here is a separate representation consequence of the same relation-transport organization. It shows interacting registers are possible with linear register count; it does not establish their acquisition advantage.

Let resting z_j in {-1,+1} be m Boolean registers, represented by z_j^2=1. For a gate channel e with target t and control registers a,b, use one complex drive x_e and

    z_t^2 = 1 + a_e(z_a,z_b) x_e,
    a_e = 3 - (1+z_a)(1+z_b)/2.

At x_e=0 every register has the same two resting roots. With controls fixed, a_e=1 when both are +1 and a_e=3 otherwise. Consequently the target's critical drive value is x=-1 in the active case and x=-1/3 otherwise. Choose a closed drive loop based at zero that encircles -1 once and has winding zero about -1/3. The lifted target changes sign exactly in the active control case. This is a Toffoli transformation in +/-1 coding. The controls remain fixed.

With no controls set a_e=1, giving NOT. Multiple gate channels can share registers using

    z_j^2 = 1 + sum_(e:target(e)=j) a_e(z_controls(e)) x_e.

Activate one channel at a time and return all drives to zero between operations. If a target serves as a control elsewhere, the other channels are zero, so those inactive terms impose no extra motion. At rest the representation has 2^m branch states using m complex coordinates; enumerating them is not necessary during one execution. Local NOT/Toffoli transformations act by the same continuation transition.

The relation, channel identities, control incidence and gate coefficient formula in this witness are designed. Replacing coefficients by learned smooth functions does not establish autonomous discovery of useful topology or efficient program acquisition. A same-information sparse relation compiler executes the same gates with the same asymptotic representation size. Reversible universality therefore cannot by itself make this a contribution.

## Explicit acquisition law that remains to be evaluated

One complete but unvalidated adaptive system can be specified:

* Representation: a finite set of smooth neural relation factors, retained internal branch coordinates, protected operation corridors with branch-label identifications, and a finite set of proposed closed control paths.
* Ordinary update: fit observed local relation data by gradient steps. Accept steps only with a corridor-preservation bound, or label preservation unverified. Refine an existing safe path's cost by gradients; do not claim this changes its gate identity.
* Structural update: on unmet training feedback, propose a bounded number of relation couplings or added sheets/channels from a fixed mutation grammar. Candidate r, epsilon and phi are obtained from a bounded parameter proposal distribution or a learned proposal model. They are not supplied the correct transformation. Reject candidates violating protected-corridor bounds.
* Operation acquisition: sample/mutate a finite set of closed paths; evaluate their actually transported endpoints on the training states; retain a path only when it improves the declared task loss after charging all fitting, proposal, path-tracking and maintenance cost. Gradients can refine surviving path geometry but discrete exploration must cross homotopy classes.
* Inheritance: retain accepted relation factors, starting-state identifications and execution corridors. At execution the transformed branch is the data state; no operation-output cache substitutes for it.

This makes every proposed step executable in principle. It does not remove combinatorial search. In particular, access to all resting states, task labels or local relation observations must be matched to the rival; testing all 2^m states is an exponential cost if done. Sampling them does not certify gate semantics.

## Rival and current decision boundary

The strongest rival is a classical learned relation model with certified continuation, adaptive homotopy graph construction, and the same structural proposal grammar. It reconstructs the candidate exactly. A second relevant rival is a growing recurrent/finite-state model with explicit reversible gates and equally protected retained operations. Neither can be excluded by calling its organization hand-designed: our mutation grammar and solver are also designed.

What might remain beyond these rivals is an acquisition law that derives where to introduce new relation couplings from local failure/evidence, while preserving useful old operation semantics through corridor topology. The current proposal/selection law does not derive this. It may simply be a costly geometric encoding of state-machine growth and program search.

A small cubic transport experiment would check implementation of known mathematics. It would not resolve this central residual and should not be promoted as a discovery result. The next meaningful decision is whether to invent the missing acquisition mechanism or move attention to another main candidate. This branch does not close the campaign.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Old declared permutations can survive acquisition of a new coupled sheet | RECONSTRUCTION with explicit sufficient conditions | Uniform root-disc argument; explicit quadratic-to-cubic witness | Whether learned updates can satisfy useful bounds at competitive total cost | Gradient-only gate acquisition fails in safe regions; fixed braid programming covered | Decide whether local task failure can specify productive relation coupling beyond generic structural search
Interacting multi-register operations exist in this organization | RECONSTRUCTION, hand-designed witness | Conditional square-root Toffoli construction | Learning these relations/incidences and new useful operations | Cubic alone is a trivial finite-state machine | Use only as representational feasibility, not evidence of acquisition or advantage
