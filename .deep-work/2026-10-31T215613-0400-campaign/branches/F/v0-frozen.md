# F v0 — frozen before nearest-prior attack and all cross-branch exposure

Frozen 2026-10-31. AGENT PROPOSAL, not novelty claim. Working description: learned holonomy as an internal instruction representation.

## Computational opportunity

An internal system need not identify its immediate state-update fields with its useful computational operations. Closed excursions of a small controller/workspace can leave a structured transformation in data that the workspace carries. The loop is a persistent executable object; composing loops composes actual transformations, and noncommuting transformations can synthesize a new data operation while restoring intermediate registers. Learning could reshape the geometry so that useful operations become short loops. This connects to construction and composition in the Fundamental Question. It does not by itself improve global reachability, give evolution, or remove an externally specified learning signal.

## State, execution, adaptation, retention

Full state is (q,h,theta,L,c): q in R^2 is an internal controller state; h in R^d is payload; theta defines matrix fields A_i(q;theta); L is a finite library of signed closed polygonal paths; c is a finite search cursor and task-feedback statistics. theta and L are fixed during a single execution. The original external input is placed in h; outputs read h. Weights theta persist across episodes, and accepted loop descriptions persist in L. There is no evolutionary claim.

Continuous execution law:

  dq_i/dt = u_i(t),
  dh/dt = sum_i u_i(t) A_i(q;theta) h.

A_i can be outputs of a finite ReLU neural network reshaped as matrices. The operational state of a numerical implementation additionally contains the integrator time index; the integrator, time step, precision and full primitive-step cost must be specified. Reversing a path implements the inverse map only in the reversible driftless idealization; it is an assumption to test under dissipative/noisy implementations. Closed q does not imply closed h. No operation acts as a unit-cost instruction simply because it receives a name.

Primitive loop candidates are bounded polygons: from q=0, follow n learned displacement vectors v_1,...,v_(n-1), and use v_n=-sum_(i<n) v_i. This parameterization enforces closure, not functional success. The v_i are real parameters with a fixed amplitude bound. For each proposed loop execute the actual dynamics on supplied training examples. A fully explicit adaptation option is symmetric finite-difference coordinate descent on (theta,v): evaluate J(z+delta e_j), J(z-delta e_j) using actual executions, then update z_j <- clip(z_j-eta[J_plus-J_minus]/(2delta)); cycle coordinates in a documented order. J is external task loss plus stated path-length and stability penalties. This is expensive ordinary optimization, not a self-organization oracle.

After primitive training, propose compositions through length-first enumeration of signed accepted loop indices, excluding only immediate inverse cancellation. Execute every candidate on task feedback examples. Accept the first shortest word below fixed error tolerance on an independent validation set; append its complete expanded path (or DAG references plus full expansion cost) to L. This fixed syntactic proposal process supplies variation. It is a conventional search scaffold, explicitly charged, and may prove to be the real source of any apparent discovery advantage. No claim that the controller or search algorithm emerges.

## Exact small instantiation

Use d=3 and a two-feature ReLU connection:

  phi_0(x)=max(0,1-|x-1|),
  phi_1(x)=max(0,1-|x-3|),
  A_x(x,y)=y[phi_0(x) M_0 + phi_1(x) M_1],
  A_y=0.

Each M_k is an arbitrary trainable strictly lower-triangular 3x3 matrix, parameterized by three neural output weights. The two hats have disjoint interiors, integral 1 on [0,2] and [2,4]. Their support structure and triangularity are human/agent designed, not learned. Their use gives an exact finite probe, not a claim about general neural field discovery.

Loop A: (0,0)->(0,1)->(2,1)->(2,0)->(0,0).
Loop B: (0,0)->(2,0)->(2,1)->(4,1)->(4,0)->(0,0).
The only nontrivial segment of A or B traverses a single hat at y=1, so H_A=exp(M_0), H_B=exp(M_1). Since M_k^3=0, exp(M_k)=I+M_k+M_k^2/2 exactly. Initial M_k weights will be small random values. Training examples specify the two linear maps, not their logarithms: A*(a,b,c)=(a,b+a,c), B*(a,b,c)=(a,b,c+b). Finite-difference learning updates six weights from basis-vector executions; it does not install E_21/E_32 directly.

The later construction target is C*(a,b,c)=(a,b,c+a), which preserves the intermediate b. It is evaluated through input/output feedback; no named commutator rule is provided to the enumerator. Search words of length <=4 over {A,A^-1,B,B^-1} after primitive training. Include full physical loop segment costs and all candidates/evaluations, plus a flat word-search comparison using the identical learned actions. Expected exact-model solution is a four-operation commutator, but this expectation is a derivation, not an executed result.

Worked operation, assuming the primitive training reaches M_0=E_21, M_1=E_32: A then B then A^-1 then B^-1 sends

  (a,b,c) -> (a,b+a,c) -> (a,b+a,c+b+a)
            -> (a,b,c+b+a) -> (a,b,c+a).

The controller q returns to zero after each component. Example (a,b,c)=(2,5,7): (2,7,7), (2,7,14), (2,5,14), (2,5,9). The third register gains 2 and the intermediate register is restored. This is a standard elementary-matrix commutator independently reconstructed here, not a new algebraic fact.

## High-upside extension and fragile step

HYPOTHESIS: Jointly learning the connection and a library of closed loops could make a useful internal operator algebra short and reusable, while higher commutators create operations requiring no persistent scratch-state change. A larger nonlinear neural connection may supply operations not handchosen as these shears were. A causal comparison must hold payload dimension, trainable parameter count, primitive execution budget and task feedback fixed.

OPEN, hardest step: Learn a changing connection that both makes a later useful operation accessible and preserves the semantics of retained old loops. Closure alone does not protect those semantics. A library can become stale when theta changes. The proposal's significance survives only if this geometry gives a cost/robustness/composition advantage over a recurrent network with the same dynamics and a cached sequence library. No such advantage is established in v0.

## Predeclared lightweight probe

1. Fit six connection weights by explicit central-difference learning, at most 2500 epochs, 13 primitive-loss evaluations per epoch (12 perturbations plus center); deterministic seeds 0–4, fixed hyperparameters, no adaptive budget increases.
2. Enumerate reduced signed words length <=4, with the same fitted dynamics; report tested words, basis evaluations, minimal accepted length and held-out linear test error. This tests construction of C, not discovery of the primitive fields.
3. Causal negative control: constrain all primitive matrices to be scalar multiples of one common E_21. No sequence can change the third coordinate. Confirm target failure under the same word grammar.
4. Stability attack: perturb both trained matrices with predeclared isotropic parameter noise, sigma in {0,0.001,0.01,0.1}, 20 draws/seed. Report old primitive errors and retained-word errors. This tests the geometry's semantic fragility; it is not an evolvability test.
5. Cost comparison: flat cached word execution under the same actions is algebraically identical. Charge exact matrix primitive actions and geometric path edges; do not claim acceleration from renaming a word.
