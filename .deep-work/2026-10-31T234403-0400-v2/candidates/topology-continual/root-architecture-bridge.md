# From protected operations to protected learned computation: the missing binding

ROOT CONSTRUCTION / ATTACK / REPAIR, 2026-11-01. Same V2; topology MAIN human-locked, M0 NOT PASSED, M3 final ambition. Online USER-PARK, standalone Persistent State USER-PAUSED. This follows the actual three-stage pilot and the human clarification that quantum control is a research carrier, not the endpoint. No new numerical experiment is claimed. The calculation below is not a general impossibility result or a proposed claim of novelty.

## 1. What actually has to transfer

The pilot stores learned target-angle records and uses a supplied robust geometric constructor to generate gates. It really acquires three executable native transformations from sampled supervision, and a shared compensation change affects an earlier acquired gate. It does not learn the constructor, a richer representation, a growing computation grammar, or an input-to-program interpreter. Ordinary process estimation plus direct control explains the result; UGD produces the same new programs. The known physical gate family is not already a general computational learning architecture.

For a potential transferred architecture, write its actual state as

    S=(theta, K, {p_j,B_j}, U, data),
    rho_next = Execute_theta[p_theta(x)](Encode_theta(x)),
    output = Read_theta(rho_next).

Here K is a code/execution representation, p_j are acquired reusable programs with task-evidence B_j, U is the paid uncertainty state, and theta includes the SHARED encoder/router/executor/readout. A neural implementation is allowed. New programs and shared changes must come from explicit feedback/update rules; program/state growth, finite precision, routing, execution and maintenance all cost resources. The fixed six-mode example instantiates only a small part of Execute and its control calibration.

The desired computational benefit cannot be inferred from a protected Execute alone. A universal fixed gate alphabet is also not a learned general architecture; conversely it is legitimate to build from one. The missing question is what the learning organization adds to the acquisition, reuse and preservation of actual task computations under equivalent resources.

## 2. Exact counterexample: a protected gate library can forget every task

Assume I and X execute perfectly for all theta, so their physical retention error is identically zero. An old input x0 is routed to I with probability 1-r_theta(x0) and X with probability r_theta(x0). On input state |0>, its output-one probability is r_theta(x0). A shared neural-router update changing r from0 to1 loses the old task completely while BOTH gates remain perfectly protected. No quantum-control error, calibration defect or noise is needed.

For a differentiable program mixture, the executed channel is

    E_theta(x)=(1-r_theta(x))*I_channel+r_theta(x)*X_channel.

Changing r by delta changes an observable probability by delta; on |0> the trace distance of the two outputs is |delta|. Thus gate invariance supplies no bound on the router's contribution. The same problem occurs if a shared encoder/readout changes the meaning of the logical state. Protecting a hardware operation is not automatically protection of a learned computational capability.

This counterexample applies to the naive transfer inference, not to all possible topology-protected architectures. The pilot's explicit IDs plus retained angle table avoid this issue by an immutable binding. A classical compiler/program library can use the same device at the same semantic-memory cost. Assigning it a neural router that forgets while granting ours a frozen ID table would be an unfair comparison.

## 3. A concrete topology-based repair to investigate

Let the shared router itself generate an executable loop in a punctured control space:

    gamma_theta(s,x) = z0 + sum_(k=1)^K a_(theta,k)(x)
                                           *(exp(2pi i k s)-1),
    s in[0,1], z0 fixed and nonzero.

The endpoint is pinned. A loop's homotopy class determines an operation through a specified monodromy representation; nontrivial semantics, its physical realization and resource price must be supplied, not inferred from the word topology. This does not require quantum hardware, although protected quantum transport is one possible carrier.

For an old input domain D_j, suppose an actual certificate gives

    m_j <= inf_(x in D_j,s in[0,1]) |gamma_theta(s,x)|,
    sup_(x in D_j) sum_k |a_(theta',k)(x)-a_(theta,k)(x)|
                                                       < m_j/2.       (1)

The pointwise loop difference is <m_j because |exp(2pi i k s)-1|<=2. Its straight interpolation cannot hit the puncture, by the reverse triangle inequality. It is a based homotopy; therefore the ENTIRE router's old homotopy class, not merely a supplied gate, stays fixed for all x in D_j. With a stable monodromy representation and its separate finite execution bound, this protects the old task-to-operation binding.

The scope is the binding, not the whole architecture automatically. Encoder, readout and code identification must remain fixed, or be jointly transported with an explicit semantic condition. For example a bijective code transport T preserves the overall function if Encode'=T Encode, Execute'=T Execute T^-1 and Read'=Read T^-1 on the retained domains. Estimating/implementing T and maintaining all three identities have their actual costs; this elementary commuting-diagram condition is not a new algorithm or free protection of arbitrary neural representations.

This is a genuine sufficient topology-based update rule. Its admission check can be conservative. It protects only the declared old domains, the chosen code interface and actual finite contracts. The margin is not free knowledge: finite samples do not certify an arbitrary continuum without model/regularity structure. A uniform old-domain bound can be expensive in high dimension. The update also needs a useful new-task direction; a tiny guaranteed step is not a capability advantage.

A prospective learning event is explicit: obtain new task outcomes; propose a finite update to shared coefficients; execute the new loop and obtain task reward; accept only if (1), the execution error/caps and the task evidence pass, otherwise reject/change the representation or buy information. For example symmetric finite proposals theta±delta*v produce the paid estimate (R_plus-R_minus)*v/(2delta), with a stated finite proposal distribution and bounds. This is ordinary feedback-guided search unless its protected binding structure demonstrably changes learning/resources. Winding reward has flat regions, so infinitesimal gradients alone cannot cross a class boundary. A valid structural proposal may jump between two valid controllers installed at idle; it must not physically execute a singular path through the puncture.

New operations can be formed on new input domains while old domains remain protected. But if all loops avoid the punctures for EVERY point of one connected input domain, continuity makes their discrete homotopy class locally constant across that domain. Multiple operation classes therefore require input-domain boundaries, an explicit compositional/discrete routing structure, or a representation change. Topology has not erased the ordinary learning/classification problem.

## 4. A complete small learning transition, and its exact conventional rival

Take K=1, z0=1, a_theta(x)=theta0+theta1*x real. The routed loop is a circle of center1-a and radius|a|. Its winding is1 if a>1/2 and0 if a<1/2; at a=1/2 it hits the puncture and is invalid. Negative a has winding0. Let the representation send winding0 to I and winding1 to X (this supplied finite representation is not claimed discovered).

Initially theta=(0,0). The old domain D_old=[-2,-1] uses I. A new task at x=1 rewards X. Propose the ONE shared update

    theta'=(eta/2,eta/2),        1/2<eta<1 (e.g. eta=3/4).

Then a_theta'(x)=eta(1+x)/2<=0 throughout D_old, so every old routed operation remains I. The initial old margin is1 and sup_Dold|delta a|=eta/2<1/2, satisfying (1) as well. At x=1 it is eta>1/2, so the new binding calls the EXISTING operation X; it does not create a new operation alphabet. For all intermediate shared parameters t*theta', t in[0,1], old loops still avoid the puncture. The new input class changes through a parameter value where its path would be singular, but that intermediate new-task controller is not installed/executed. This is a complete constructive old-domain retention/new-task-binding transition, rather than an unexplained discovery black box, and remains only a support step toward M0. Which eta to accept can be decided from actual paid task returns; no new feedback is assumed from loop preservation itself.

The strongest simple rival is explicit: a one-layer threshold classifier h_theta(x)=1[a_theta(x)>1/2] with a fixed I/X program library gives EXACTLY the same routing, update, memory and task outputs, without numerically transporting a loop. Both can retain the old interval by the same linear inequality. Our small router construction therefore has no demonstrated algorithmic learning or memory advantage. The loop can additionally carry a physical robustness property, but that must be priced against a physically appropriate competitor, not assumed from the symbolic representation. A test of this tiny example would merely rerun the equivalence, so it is NOT executed.

This comparison narrows this explicit repair; it does not prove that all topology-supported binding is ordinary threshold classification. A noncommutative, compositional program representation could have a different finite-budget acquisition structure, but its actual formation rule and strong rival remain to be constructed. Do not assume success by adding punctures or dimensions.

## 5. What remains consequential

Three different maps now need to be distinguished:

- physical perturbation -> harmful logical error (the projector selection rule in noise-boundary-p);
- shared parameter update -> changes in old operation semantics (the uniform chart controls a limited family);
- shared parameter update -> old input/representation/program binding (the counterexample and repair above).

A transferable learning mechanism must handle all three AND acquire new computational organization. A fixed number of protected primitive controls could reduce physical maintenance independently of the library size, but existing digital/program and robust quantum architectures can also reuse primitives. A learned router can have shared semantic-preserving deformations, but the explicit scalar example is already matched by an ordinary classifier. Neither isolated protection claim passes M0.

The next development uncertainty is therefore concrete: can an ACQUIRED compositional binding/execution representation give reusable protected update directions that materially improve later operation acquisition, beyond an equally informed compact program learner and constrained/modular optimizer? Need a representation/transition law where this difference can exist, not another supplied angle grammar, a universal-simulation dismissal, or a new gate-control benchmark. Keep the locality/error-projection work as support; do not silently add a favorable noise prior to the completed pilot.

CONTINUE inside human-locked topology. No M0 advantage or completed architecture is claimed. This file supplies an actual transfer counterexample, an explicit constructive binding rule, and a complete small transition/rival; it does not claim that the remaining architecture has been invented. The current finite version's advantageous-result claim remains closed while the research hypothesis stays OPEN.
