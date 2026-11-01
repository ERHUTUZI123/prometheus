# Frozen candidate v0: learning the algebra in which an internal computation runs

Status: AGENT PROPOSAL / HYPOTHESIS. Frozen before reading any previous or current campaign candidate report. This is a branch of the existing V2 campaign, not a new campaign. Context separation is instruction_only with shared filesystem. I read AGENTS.md and README.md Research Vision, and independently selected the original theoretical entry below. I have not loaded the campaign booklist/index or root's technical proposals. The inherited provider configuration is unchanged; effective provider mode is unavailable, and no Pro claim is made. No experiments, training, paid API, package installation, or model switching were performed.

## Judgment and scientific connection

The potentially consequential principle is that learning can change the *algebraic organization of the actual computation while preserving its exact input/output meaning*. This differs from only changing the answers produced by a fixed sequence of neural operations: the learned objects determine whether an apparently exponential global computation can run as a polynomial one. Neural control learns an internal change of computational coordinates; exact local cancellation makes the change safe under arbitrary composition.

The proposed concrete realization is restricted and may fail the ambition test: a neural controller finds local basis changes that turn an internal tensor computation into a certifiable tractable network. Its strongest rival is an ordinary algorithm that solves for that basis analytically. If that rival exhausts its useful content, this branch should be abandoned as a foundational proposal rather than renamed.

This bears on the Fundamental Question because the network learns which internal operations and global execution organization realize a computation, rather than learning only an input/output map. It does not establish open-ended evolution, self-sufficient evaluators, or new computability. The intended difference is finite resource accessibility.

## Independent entry and a reconstructed assumption

SOURCE: Leslie G. Valiant, *Holographic Algorithms*, SIAM Journal on Computing 37(5), 1565–1594 (2008); original preprint text: https://renyi.hu/~miklosi/RES2015Fall/Valiant-holographic_algorithms.pdf . I read introduction, §§3–5, and the statements around Theorem 4.1, not just the abstract. The source establishes a way to transform local signatures while preserving a global sum, then exploit planar perfect-matching computation. It also expressly discusses algebraic search for reductions and its difficulty. This mathematical organization is prior work, not our invention.

RECONSTRUCTION: The assumption being challenged is not that changes of representation alter the mathematical function. They do not. It is the practical assumption that an internal encoding merely stores equivalent values while leaving the kind of algorithm needed to combine them unchanged. Here an exactly equivalent local encoding can make a different global execution rule applicable. Learning that encoding could therefore constitute learning a computational mechanism, if its acquisition genuinely exceeds a routine learned preconditioner.

## State and computational primitive

An episode is a finite planar bipartite graph G=(L,R,E), with an ordered set of ports at each vertex and maximum degree k fixed by the run. Each edge has a two-dimensional rational state space. Vertex v carries a rational tensor A_v indexed by its incident Boolean edge labels. External evidence is represented by fixed boundary tensors. The required answer is

    Z(G,A) = sum_{x in {0,1}^E} product_v A_v(x restricted to incident(v)).

The internal state is

    S_t = (G, {A_v^(t)}, {Q_e^(t)}, C_t, phi_t),

where Q records accumulated basis changes, C records accepted exact local signatures/certificates, and phi is the neural proposal policy. The computation is an actual contraction of this network; its answer is not approximated by phi.

Primitive 1: exact rational tensor contraction of two adjacent tensors.

Primitive 2: paired internal basis change on one edge e=(u,v). For any invertible M in GL(2,Q), define

    A'_u(...,j,...) = sum_i A_u(...,i,...) M[i,j]
    A'_v(...,j,...) = sum_i A_v(...,i,...) (M^(-T))[i,j].

Every other tensor stays fixed. Because M (M^(-T))^T = I, summing the shared j restores the original contraction. Repeating the operation on any edges preserves Z exactly. Boundary interpretations are fixed or transformed with their paired inverse; nothing changes the question asked.

This invariant is a short algebraic proof, not an empirical result: for vectors a,b, (a^T M)(M^(-1)b)=a^T b. The global result follows by expanding the finite sums on each transformed edge. Rational arithmetic and exact nonzero determinant checks are required; floating-point closeness is not an exact certificate.

Primitive 3: a certified change of global execution rule. The controller may propose a bounded planar weighted matchgate H_v for each transformed local tensor. Compute its complete signature (all external-node deletion cases) and check exact equality with that tensor. If every local equality, port order, and planarity condition is certified, assemble the planar graph H and evaluate its weighted perfect-matching sum with the supplied classical polynomial algorithm. If certification fails, retain the original exact computation. The tractable evaluator and certificate checker are designed infrastructure; the learned object is the coordinate organization and its local realization.

## Neural adaptation and retention

Use a graph neural policy to assign a categorical proposal distribution to each permitted edge operation and local matchgate proposal:

    h_v^0 = encode(A_v, degree(v), port order, local cost)
    h_v^(r+1) = F_phi(h_v^r, sum_{u adjacent v} B_phi(h_v^r,h_u^r,port(u,v)))
    pi_phi(a_t | S_t) = softmax(score_phi(h_u^r,h_v^r, local trace)).

An action chooses an edge and an invertible rational matrix from a finite bounded-description catalogue, or proposes a bounded matchgate realization with rational weights. The catalogue includes identity, permutation, sign, elementary shear and diagonal factors; finite products can express more bases without assuming a new arithmetic primitive. Only finitely many actions are allowed per episode. Neural output does not bypass exact checks.

For episode cost, use

    J = B_search + B_verify + B_execute + lambda * description_length(transforms, certificates),

where each B counts actual bit operations or a declared bounded proxy, including rational bit growth. Correctness is invariant; reward is R=-J, with a fixed failure penalty when the episode exhausts its search/execute budget. An explicit permitted update is

    phi_(t+1) = phi_t + eta * (R_t-b_t) * sum_s grad_phi log pi_phi(a_s|S_s).

The baseline b_t is updated by an exponential running mean. This is conventional policy gradient, not a new learning rule. Successful shared transform generators and neural parameters are retained; instance answers are not the retained mechanism. Evaluation must include unseen graph sizes, unseen local signatures, and total acquisition plus deployment cost, rather than only amortized query latency.

The required mechanism evidence would be that a retained local transformation policy repeatedly exposes a tractable algebra on unfamiliar instances, with causal ablation of the transformations restoring the expensive computation while preserving the exact mathematical answer. This would establish organization change; generality or novelty would still require separate evidence.

## Fully worked finite process (hand derivation, not an executed experiment)

Take two arity-three vertices with corresponding ports connected by three edges. Let K=[[1,1],[1,-1]], so K^(-1)=K/2. Define even-parity tensors T and U, indexed in binary order 000,...,111:

    T = (1,0,0,2,0,3,5,0)
    U = (7,0,0,11,0,13,17,0).

The supplied internal tensors are instead

    A = K^(tensor 3) T = (11,1,-3,-5,-5,-3,1,11)
    B = (K^(-1))^(tensor 3) U = (6,0,-1,-3/2,-3/2,-1,0,6).

The initial exact computation is their eight-term inner product:

    66 + 0 + 3 + 15/2 + 15/2 + 3 + 0 + 66 = 153.

The neural controller proposes M=K/2 on each of the three edges. Since M^(-T)=K, the paired basis transition gives

    A' = (K/2)^(tensor 3) A = T
    B' = K^(tensor 3) B = U.

Now all odd-parity tensor entries vanish and the same answer is

    1*7 + 2*11 + 3*13 + 5*17 = 153.

This demonstrates a literal change of internal operations and cancellation, with exact meaning preserved. It does NOT show a speedup worth training: eight terms versus four is trivial, and this graph is tiny. The nontrivial success counterfactual concerns arbitrary large planar compositions of local tensors with certified matchgate realizations: a basis transformation would make the supplied global tractable evaluator applicable without enumerating 2^|E| configurations. The source theorem provides that conditional implication. This finite example does not establish discovery of the basis or superiority to an analytic solver.

## Important limitation that could change, and its exact boundary

HYPOTHESIS: A neural system can learn transformations that expose a useful closed computational algebra which is hidden in its initial local encoding, and transfer the discovery to previously unseen graph compositions. A successful general mechanism would move some learning effort from approximating answers to autonomously organizing exact algorithms. The importance would come from a changed resource scaling or access to previously infeasible computations, not merely reusable subroutine calling.

No universal improvement is claimed. Edgewise invertible gauges preserve matrix-flattening ranks and do not reduce graph treewidth. They cannot generally turn arbitrary hard counting into easy counting. The specific possible gain here is membership in a different tractable algebra, not lower entanglement, new functions, or magic removal of complexity. Search cost can exceed the savings. Matchgate applicability is narrow and planar composition is a real constraint.

## Strongest rival, fragile step, and decision boundary

ATTACK: Analytic holographic-basis discovery already searches for equivalent tractable descriptions. An ordinary algebraic solver given the same signatures and target family may recover the basis cheaply, with stronger guarantees. Tensor-network gauge optimization is another direct alternative. The neural proposal could be only an incremental heuristic on those methods.

The fragile step is consequently NOT the invariant (which is elementary) but the acquisition claim: can the learner discover a computationally important organization for which known algebraic search and simple exact solvers are meaningfully inadequate at matched total resources? The present mechanism neither demonstrates that such a family exists nor gives a reason to expect broad autonomous discovery outside the hand-supplied tractable family.

A second attack is that we manually supplied the execution algebra and a neural controller only finds a compatible encoding. That supports learning how to access an existing mechanism, not learning a new mechanism or inventing new algebraic laws. Extending the action space to arbitrary learned exact rewrites would need a different substantive mechanism, and would invite the strong equality-saturation / learned algorithm-discovery rivals. That extension is not silently part of this candidate.

Provisional decision at freeze: serious explicit candidate, but its consequential content is OPEN and its basic substrate is prior mathematics. Do not label it a foundational advance unless the strong rival leaves a specific important unresolved acquisition difference.

## Scientific state

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

Paired local transformations preserve the finite contraction | VERIFIED by the displayed finite algebra, not machine-checked | equations above | exact-arithmetic implementation absent | none executed | check nearest analytic basis-discovery results.

Learning such transformations is a consequential new computational organization | OPEN | explicit proposed substrate and hand example | target algebra supplied; closest algorithms may exhaust the idea | initial plain relation-propagation idea was not pursued because its best realization was ordinary learned constraint programming | compare original basis-discovery work before any implementation.
