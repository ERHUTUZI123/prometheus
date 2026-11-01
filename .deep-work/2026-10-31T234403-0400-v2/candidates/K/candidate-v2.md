# K v2: compact the learning transitions, retain and construct plasticity

HYPOTHESIS / AGENT PROPOSAL with an exact mathematical instance. Preserves v0/v1 and their negative findings. Requested profile pro; observed parent host gpt-6.1-sol / ultra; effective mode unknown. This is a continuation from v1's finding that current-function compilation can erase future learning directions.

## Important limitation and proposed change

Current capability and the mechanism that produces later capabilities are different objects. A model can have a cheap representation of its present function while its ability to change depends on parameter state invisible in that function. Keeping every original parameter protects this state at a potentially large cost. Keeping only the present function, or even its initial Jacobian kernel, can destroy it.

The proposed organization retains a compact **state sufficient for future learning transitions**, then permits the neural system to construct and modify the part of that state governing plasticity without first modifying current behavior. The compactor's target is a learning dynamical system, not only a forward computation. A learned constructor selects computational regions, proposes sufficient statistics and plasticity edits, and is trained on subsequent task learning and actual resource cost. Exact algebraic reduction is an allowed implementation component, not an entity that must itself emerge.

If successful beyond the exact class below, a network could replace a bulky internal implementation with a small executable state-update law, retain the ways that implementation can acquire functions, compose that law with other learned mechanisms, and use freed resources to create further mechanisms. The important prospective consequence is preservation and deliberate construction of future learning capacity under computational recycling. It is not an increase in the expressivity of a linear map.

## General mathematical primitive

Let source state q include parameters and optimizer memory, computation f(q,x), and an allowed learning action a (data/loss, step size, specified construction action). Its actual update is q'=U(q,a). A compact state z=C(q) is accepted with executable readout g and update V only if

    f(q,x)=g(C(q),x),
    C(U(q,a))=V(C(q),a)

for the stated source domain, all declared x and actions a. These equations imply equality of every finite future computation/learning trajectory by induction. A one-step identity at only the current q does not suffice. Approximate learned proposals remain approximate unless their error envelope and horizon are checked.

**Construction of C.** Start with observables needed for the present output. Pull them back through U. If their successors cannot be expressed using current observables, add the new coefficient/statistic programs exposed by that failure, and repeat. For polynomial source updates, coefficient extraction and linear dependence provide a concrete finite procedure; closure is not guaranteed. A neural proposer can prioritize programs/regions and learn from successful reductions. The polynomial fallback makes its proposals inspectable; it does not hide a correct statistic in task labels.

**What must survive.** An observable survives because it distinguishes future learning behavior under the declared action interface, even when it does not affect today's function. This can require more than a Jacobian snapshot. A caller may supply any gradient matrix at the module boundary, so the exact instance below covers unknown future tasks and nonlinear surrounding computations without a fixed future-task list.

## Exact instance: a wide learned matrix mechanism

Consider a two-factor module W=VU, U in R^(d_in x h) transposed convention below, V in R^(d_out x h). Precisely, use U in R^(h x d_in), so W=VU. There is no activation between these factors. The module may be inside an otherwise nonlinear network. It exports W, not its individual hidden factors.

Given any differentiable task loss depending on the factors through W, let G=dL/dW. Ordinary simultaneous finite-step SGD is

    V'=V-eta G U^T,
    U'=U-eta V^T G.

Define

    A=VV^T, B=U^T U,
    H=[[A,W],[W^T,B]], Z=[V;U^T], H=ZZ^T,
    M(G)=[[I,-eta G],[-eta G^T,I]].

The new executable learning mechanism is

    H'=M(G) H M(G)^T,
    output(x)=W x, with W the upper-right block of H.

Expanded laws are

    W'=W-eta(AG+GB)+eta^2 G W^T G,
    A'=A-eta(GW^T+WG^T)+eta^2 G B G^T,
    B'=B-eta(G^T W+W^T G)+eta^2 G^T A G.

They follow directly by multiplying the two actual factor updates. Hence, for any finite task/gradient sequence, finite h, and exact arithmetic, H preserves every future output and SGD update. H is positive semidefinite and stays so by congruence. It can be realized by factors of width at most d_in+d_out when a factor implementation is required.

Cost: original factor storage is h(d_in+d_out). H stores at most (d_in+d_out)(d_in+d_out+1)/2 independent entries; one-time construction costs O(h(d_in+d_out)^2), dense updates conservatively O((d_in+d_out)^3), and readout costs O(d_in*d_out). This helps only in an appropriate width/dimension regime and after amortizing construction. Bit lengths, data gradients, surrounding network execution and precision costs must be counted. It is not a claimed advantage for low-rank adapters when h is small.

The covariance reduction itself has close established prior work. It is a technically grounded instance of the proposed organization, not the claimed novel theorem. The research proposal is to discover/use such closed learning state as an internal mechanism and to co-learn construction actions on it.

## A worked later-learning consequence beyond initial NTK matching

Take d_in=d_out=2, W=0 and two realizable states:

    H_a: A=2I, B=2I;
    H_b: A=3I, B=I.

Both have the same current function and the same complete initial first-order response G -> AG+GB=4G. Thus current W plus initial Jacobian/NTK information cannot distinguish them. Both can be realized with eight rational hidden paths; no irrational initialization is needed.

Use eta=1/4 and the legal two-action loss sequence with constant gradients

    G1=diag(1,2), G2=E_12.

These gradients can arise from two successive linear losses L_j(W)=trace(G_j^T W); only a finite two-step claim is made. After the first step W1=-G1 in both cases. After the second,

    (W2_a)_12=-37/32,
    (W2_b)_12=-77/64.

The difference is -3/64. The quadratic finite-step terms change A and B differently even though their initial mobility agrees. H retains the information needed for this distinction. A frozen initial-kernel simulator cannot reproduce both futures from the same compressed state. This demonstrates the relevant failure of a particular compaction criterion; it does not prove superiority over every optimizer.

A simpler automatically discoverable instance uses w=sum_i a_i b_i. Under a_i'=a_i-u b_i, b_i'=b_i-u a_i for free control u=eta*dL/dw, pulling back w yields w'=w-u*s+u^2*w, where s=sum_i(a_i^2+b_i^2). Pulling back s gives s'=s-4u*w+u^2*s. A generic coefficient-closure routine starting with w can therefore discover s rather than receiving its name or formula as a target. This will be the small inspectable prototype of constructing update-sufficient state, not a claim that a neural proposer has already learned it.

## Construct plasticity while preserving the current computation

The learner may choose a bounded positive semidefinite block-diagonal edit

    D_psi(c)=diag(R_psi(c)R_psi(c)^T, S_psi(c)S_psi(c)^T),
    trace(D_psi)<=rho,
    H <- H + D_psi(c).

This leaves W, hence every current input/output, exactly unchanged. It changes subsequent updates through A and B. It is physically realizable by adding hidden columns that are nonzero on only one side of the factor pair. It is the same functional growth action used by an actual enlarged source network, represented without accumulating its inactive paths.

Here c contains past task/gradient summaries; the neural constructor's parameters psi are adapted through a finite later-learning objective,

    J(psi)=E_tau[L_val,tau(W_K(H+D_psi(c)))]+
           lambda*construction_and_execution_cost,
    psi <- psi-beta*grad_psi J.

The exact inner laws above make this meta-gradient explicit. Train/validation information and horizon K are specified by a future task contract, not supplied by an oracle in evaluation. rho, rank and norm constraints prevent a scalar speedup obtained by arbitrarily inflating the effective learning rate. A useful result needs matrix-direction or structural benefits under matched budgets.

For W=0, A=aI, B=bI and one-step squared target losses, the expected first-order benefit of adding output-side Delta A is proportional to trace(Delta A * E[T T^T]) when eta(a+b)<1. A trace-bounded PSD edit therefore favors the top eigendirections of the future target covariance. This gives a worked adaptation law and explains what information is learned. Current-gradient SVD growth and meta-learned preconditioners are strong actual comparators, not straw rivals.

## Composition and optimizer memory

Connect such modules through arbitrary specified differentiable nonlinear functions, for example W2 sigma(W1 x). Each source module and its compact replacement have the same W and therefore the same forward values and boundary gradients G. Applying their exact updates preserves the composed learning trajectory by induction, provided factors are not separately inspected, there are no hidden internal-factor penalties, and gradients use the same pre-update state. This is preservation under a declared computational/learning interface, not merely reuse on a training set.

Momentum can be handled without retaining width h. With stacked factor state Z and velocity P,

    P'=mu P+K(G)Z, Z'=Z-eta P',
    K(G)=[[0,G],[G^T,0]].

For C=[Z;P], C'=N(G)C where

    N(G)=[[I-eta K(G),-eta*mu I],[K(G),mu I]].

The Gram C C^T follows the same congruence law and retains factor/velocity cross-statistics. Adam's coordinatewise normalization generally lacks this shared-linear closure; no such extension is claimed. A bounded set of explicit nonlinear exceptions may remain uncompressed. A new source action that exposes hidden units or changes their individual nonlinearities requires refinement/recompilation of the learning-state contract.

## Scaffold, acquisition and main unresolved construction

Designed: update semantics, an arithmetic program grammar, symbolic coefficient expansion/exact identity checking, resource accounting, PSD/trace constraint and outer meta-gradient rule. Existing SGD, AD, covariance algebra and growth actions are permitted components.

Constructed in the proposed prototype: previously absent sufficient statistics and an executable closed update program obtained from source updates. A scalar symbolic closure routine can establish that construction without receiving s as a target. The full matrix H mechanism is mathematically derived here, not yet automatically discovered by a neural policy.

To be learned in the strong version: which regions admit valuable compaction, which action contracts/observables need preservation, which plasticity edits prepare for later tasks, and how newly constructed module laws can be reused. A successful build or exact Gram replay is not evidence that these choices have been learned or that the broad ambition has been achieved.

The hardest step is discovering useful, compact, composable update-sufficient state for mechanisms with richer nonlinear computation, rather than accumulating an observables hierarchy as large as the original implementation. The source update may have no cheap finite closure. This is the central scientific risk, not a reason to replace invention by further toy demonstrations.

## Actual established controls and scientific distinction

1. Full source training with the same growth/controller decisions: the exact trajectory reference and cost baseline.
2. Function-only folding/distillation into W with a conventional optimizer: preserves present capability but changes later dynamics.
3. Initial-Jacobian/NTK preservation: a stronger compaction criterion, separated by the two-step example.
4. Known covariance/intrinsic-dynamics reductions: the strongest theory comparator for the exact matrix class. K does not claim this reduction itself is new.
5. GradMax/network morphism: function-preserving physical growth that deliberately changes future training. Match added norm/trace, task information, architecture, total compute and growth opportunities. The contemplated change is retaining the growth's learning effect after compacting away its implementation.
6. Meta-Curvature or an equally strong learned preconditioner: compare equal meta-training information and matrix/trace/storage budgets. It may reproduce or outperform the proposed plasticity edits. Its independently existing goal is learned gradient transformation; it should not be credited with learned transition-state discovery without implementing that additional coupling, nor should K call a preconditioner difference a new computational paradigm.

The organizing proposal is this specific coupling: construct computational mechanisms -> expose future learning actions -> discover closed sufficient state -> retain that state through compaction -> meta-learn function-preserving construction of plasticity -> compose and repeat. A deterministic implementation is compatible with the proposal. The required evidence is that this organization changes acquisition, retention, composition or resources against these actual methods, not that no conventional components can implement it.

## Evidence boundary and next action

Mathematical laws and finite-step consequences above are RECONSTRUCTION awaiting independent checking. The candidate is not a breakthrough claim. Freeze this version and a bounded exact test contract before execution. The central local test is whether automatic coefficient closure constructs the missing statistic and whether independently implemented source SGD/momentum and compact laws agree while function/initial-kernel compaction fails the two-step witness. That can validate the primitive and expose errors; it cannot establish the ambitious learned-discovery claim.
