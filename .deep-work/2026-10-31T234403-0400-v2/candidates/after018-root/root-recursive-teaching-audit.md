# Recursive teaching: exact comparator, derivative repair and finite-depth boundary

RECONSTRUCTION / ATTACK / constructive support, same V2. Root read P candidate-v1 and source-v1 completely. No scientific code execution, training, formal compilation or novelty assertion. This is not a replacement candidate.

## 1. What the finite example actually establishes

With theta=(a,b,d) and actual coordinate SGD lessons, B(theta)=(b,d,d), B²(theta)=(d,d,d). For desired scalar t, execution losses of theta/B(theta)/B²(theta) are half squared errors of a/b/d. Thus (1,1,0) with t=1 has zero present and first-child loss, nonzero grandchild loss; the missing d derivative is supplied by the latter. The candidate's calculation is correct. It demonstrates a distinct measured capability, not an advantage over ordinary optimization or copying.

## 2. Complete differentiable state

Let z_t=(w_t,H_t,h_t) include child parameters, teaching history and any teacher/child fast state that influences later lessons. Actual transition is

    ell_t=T_theta(c,inspect(z_t));
    z_(t+1)=F(z_t,theta,ell_t).

The student's SGD partial holds each already emitted lesson fixed. The outer total derivative does not. For Z_t=D_theta z_t,

    Z_(t+1)=(F_z+F_ell T_z) Z_t + F_theta + F_ell T_theta.

Teacher-controlled learning rate/weight derivatives are inside this full F/T formula. A weight-only recurrence is sufficient only when histories/fast states are fixed, memoryless, or already folded into w. For a memoryless gradient G(w,theta)=partial_u loss(u;ell_theta(w))|u=w, D_w G includes the ordinary loss Hessian plus G_ell D_w ell and need not be symmetric. This corrects scope, not the finite example. Supervisor independently identified this repair before this note; message provenance retained, no claim of a saved review yet.

## 3. Same-information ordinary meta-learning graph

Construct a conventional learned teacher T_psi and ordinary student learner U. Give them exactly P's bounded lesson format, inspection, context/archive, initial states and external labels. Define

    Build(psi,c,epsilon)=U^K(w0; T_psi's adaptive lessons),
    w=Build(theta,c,epsilon),
    v=Build(w,c',epsilon'),
    J=L(theta)+alpha L(w)+beta L(v)+protected_loss+cost.

Tie the teacher and executor parameters at the same locations as P. Backpropagate through the same finite graph, including the full z state above. This is exactly P's state transitions, observations, parameter sharing, derivative and work, not a universal-computer simulation argument. Calling the ordinary meta-learner's retained parameters theta and reusing its child as next teacher changes no operation. Thus the supplied law is a particular recursively unrolled meta-objective. A useful inductive bias or cost/generalization consequence of this coupling remains possible, but the mere existence of this graph is not such a consequence. SEAL's inspected implementation is nearby but not this exact full derivative; do not fabricate source coverage.

## 4. Why adding assessed generations cannot alone establish closure

For any finite assessed depth d>=1, construct theta=(a0,...,a_(d+1)) and teaching map

    B(theta)=(a1,...,a_(d+1),a_(d+1)).

Each coordinate can be realized by a unit-rate half-squared-error lesson on a one-weight port, exactly as in P's example. Let EXEC return its first coordinate. Set a0=...=a_d=1 and a_(d+1)=0. Every descendant at depths0 through d solves target1 exactly, while depth d+1 fails. This refutes an inference from arbitrary finite tested depth to unlimited reliable transmission over this allowed family. It does not refute useful finite-depth objectives, nor show that all stable constructors are impossible. No need to run a shift-register toy.

## 5. Constructive stable-family condition and its tradeoff

For one fixed context/task and a declared complete state space, suppose actual Build=B maps a closed region R into itself and is kappa-contractive in a stated metric, 0<=kappa<1. If s* is its fixed point and execution error is L-Lipschitz there, then descendant n's error is at most error(s*)+L*kappa^n distance(s0,s*). Banach's theorem provides existence when R is complete and nonempty. Those are strong premises about the complete learning transition, not consequences of P's two losses. One may instead bound distance to an invariant family, but must actually establish invariance and a suitable transverse contraction.

This could repair the particular unlimited-transmission inference. It is not a new learning rule, and global contraction to one fixed state can erase task-specific distinctions. To support acquisition, context-conditioned stable families would need to change from real new information without collapsing previously needed distinctions; ordinary recurrent/control design and meta-learning can impose the same condition. No claim that this condition alone discovers new mechanisms.

## 6. Scoped copying and information boundaries

Where children share identical parameter semantics and exact memory copy is legal, Copy(theta)=theta preserves EXEC and TEACH without any reconstruction error for the same contexts. It costs reading/writing the retained state; a finite SGD teaching trajectory adds its forward/gradient work. This undercuts a pure same-instance transmission advantage. It does NOT prove copying is best for new tasks, differing architectures, compressed storage, or adaptation; permanent-teacher, distillation and ordinary meta-learning rivals must then receive the same information/cost constraints. Merely prohibiting copy to manufacture a benefit would not answer the broader question.

All generated lessons are functions of actual model state/context/history. They can reorganize evidence or access information already in that state; they cannot identify an independent hidden external target absent from those variables. The root/helper point-query bound is a specific rigorous instance, not a claim that inspecting an acquired network is useless.

## Decision interface

P-v1 is an explicit valid construction with a useful task/transmission distinction, but its stated transition/learning graph has an exact ordinary-meta-learning comparator. Give the requested v2 a serious constructive opportunity to identify a consequential different organization or consequence. Do not run an experiment solely to re-show the missing scalar gradient or finite-depth failure. Any later test must adjudicate acquisition beyond these known constructions at matched information and total cost.
