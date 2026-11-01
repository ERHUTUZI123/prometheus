# F postfreeze attack and one constructive repair

No other branch has been read. v0 is unchanged.

## Exact algebraic reconstruction behind the observation

Write the two learned strict-lower matrices as

  M0 = alpha E21 + beta E31 + gamma E32,
  M1 = delta E21 + epsilon E31 + zeta E32.

Execute A,B,A^-1,B^-1 (in that temporal order), so the payload matrix is exp(-M1) exp(-M0) exp(M1) exp(M0). All length-three matrix commutators vanish. Direct multiplication or the terminating Baker–Campbell–Hausdorff formula gives

  C = I + (alpha*zeta - delta*gamma) E31.

Consequences:

- The central parameters beta, epsilon cancel exactly. Primitive instructions can change while the composite C remains unchanged. This is a precisely specified semantic invariant, stronger than demanding every constituent retain its entire old map.
- Isotropic weight changes alter the determinant k=alpha*zeta-delta*gamma. For the intended values (alpha,zeta)=(1,1), (delta,gamma)=(0,0), its first-order drift is d alpha+d zeta. There is no generic immunity to later learning.
- A constrained adaptation family can preserve C while improving other behavior: make arbitrary beta/epsilon changes and vary the 2x2 matrix [[alpha,delta],[gamma,zeta]] within determinant 1. For example alpha=s,zeta=1/s,delta=gamma=0 for nonzero s. The cost is that constituent semantics change; preserving C is not preserving all previous tasks. Choosing the correct invariant to retain is itself an unresolved task-inference problem.
- The entire payload operator family here is the three-dimensional unipotent lower-triangular group. Higher commutators vanish. Formal expression depth does not produce an unbounded instruction algebra.

## Strongest causal attack

The probe trains two deliberately chosen elementary maps using complete basis-vector supervision, then performs external syntax search. Hats, disjoint supports, reversible dynamics, loop paths, strict-lower matrices, feedback tasks and search grammar are supplied. Only six scalar connection weights are learned. The machine does not discover these representational choices, autonomously produce the loop controller, or discover the primitive task objectives. An identical action-sequence search with H_A,H_B has exactly the same word scores and output for every candidate, not just the observed seeds. Naming the word as a new instruction adds no intrinsic speedup or new reachable function.

The actual open proposal is jointly learning a connection and loop repertoire so later tasks become cheaper to solve. No matched comparison for this claim has been run. The present probe is a mechanism sanity check on the central transport law, not an evolvability experiment or an architecture comparison. Holding physical dynamics constant while changing the visible action alphabet would measure search parameterization and memory, not construction of a new underlying reachability set.

## Smooth geometry is not topological protection

In our exact model a rectangle of height b over the first hat yields exp(b M0). Changing its height from 1 to 1+eta changes the output, although the controller path remains closed and continuously deformable to the original path. Closed paths are therefore an interface for controller-state restoration, not an intrinsic guarantee of functional robustness. Reparameterizing an identical geometric path leaves its ideal driftless transport unchanged; deforming the geometric path generally does not.

Nearby neural prior material uses learned orthogonal matrices. Orthogonality also does not fix accumulated operator error: if the intended action is I but a learned action is a planar rotation R(delta), then L applications produce R(L*delta). The norm is exactly preserved while the semantics drift by order L*delta for small L*delta. Thus isometry, fixed-input sensitivity, topological invariance and long-horizon correctness are separate claims. This counterexample addresses an inference one might be tempted to import; it does not refute any executed numerical result in that paper.

## One constructive repair: punctured controller space and branch-cut transport

Try to make loop geometry genuinely insensitive to small path deformations. Let q lie in a plane with two excluded points p_A,p_B. Draw disjoint cuts from those points to infinity. Retain two learned invertible matrices U_A,U_B. Payload h is constant between cut crossings. On a transverse crossing of cut j, execute h <- U_j^s h, with s=+1 or -1 determined by oriented crossing. Full discrete state is (q,h,theta,previous-side-of-each-cut,library). Straight-line controller moves are subdivided at cut intersections in temporal order; ties at a puncture are rejected. U_j=exp(M_j), using the same six learned parameters and feedback update as v0. Accepted loop words are retained by the same bounded enumeration. No continuous integration oracle is present: the crossing event is an explicit geometric predicate plus a matrix action.

A loop winding once around p_A realizes U_A; one around p_B realizes U_B. Small deformations that avoid the punctures change the crossing word only by adjacent U_j,U_j^-1 cancellations, so the output is unchanged. The same four-loop worked operation is available. This gives an honest limited path-perturbation invariance absent from smooth v0. Gate-parameter drift and erroneous crossing detection remain unprotected.

Why it does not rescue the present originality/learning claim: the repair becomes an explicitly routed gate catalog whose geometry encodes a free-group word. Gates, excluded points, cuts, invertibility and crossing semantics are designed. Learning the gates remains ordinary matrix training. Learning where to create new punctures and their transition rules would require an additional mechanism, currently unspecified. It would be misleading to call that topology self-organized. The repair is a concrete alternative interpretation, preserved here, but is not executed or promoted to a validated architecture. It does not justify more experiments in this stage.

## Decision

CONTINUE only as an open research hypothesis if one can identify a scientific advantage of jointly learned geometry and instruction loops that survives comparison with the exact cached-action baseline. STOP this stage: the finite mechanism check is complete, its broad novelty is defeated by close precedent, and the actual learning benefit remains unresolved. No extra toy tasks or larger training runs are authorized by this certificate.
