# after012-b · Path-history closure and obstruction lifts · v0

Status: **substantial construction frozen; not a validated new principle**. Agent proposal/reconstruction. No code or experiments. The interrupted predecessor supplied only a seed (Chen/path-history algebra); no predecessor result is presumed. This context read AGENTS.md, README.md including Research Vision, and deep-work/WORKFLOW.md; it has not read old catalogs, old reports, or other live candidates. Context/file isolation is instruction_only; effective mode is unknown.

## Judgment and success counterfactual

An internal state can fail because it identifies histories that have the same present observation but different compositional consequences. A concrete constructive response is to **learn an extension of the state-composition law from the inconsistency**, rather than only change a readout. Two instances below are fully specified: acquire a signed-area accumulator by learning a central cocycle; and acquire a discrete sheet coordinate when real unipotent path algebras erase torsion.

If broadly learnable, the important change would be a system that acquires the *kind of memory needed by a computation*, including global discrete memory invisible to local differential statistics, from task feedback. It would change which histories its internal execution identifies. This bears directly on acquiring internal computational mechanisms. The finite examples do not establish that broad learnability, autonomous mechanism invention, or an advantage over conventional recurrent models.

**Main weakness:** the construction currently supplies the class of possible extensions and the geometric/concatenation interfaces. Ordinary system identification, weighted automata, a development network, and covering-space inference are strong matched explanations. The mathematical torsion obstruction is real, but is not a new discovery or a lower bound against arbitrary neural networks or path signatures.

## 1. Concrete first construction: acquire a composition coordinate

### State, input, and feedback

Inputs are piecewise linear paths from the origin in R^d, given as increments delta_1,...,delta_n. Concatenation translates the second path to the first endpoint. A starting mechanism retains only endpoint x = sum delta_i. Arithmetic and addition are allowed primitives. An evaluator returns a scalar F(p) for any queried complete path p. Acquisition is offline; nothing here investigates online organizational plasticity.

The learner can query paths p, q, and pq and compute

    r(p,q) = F(pq) - F(p) - F(q).

This is a derived contrast from the same scalar evaluator, not an additional oracle. Its use costs three labels unless some are cached. Suppose the residual depends only on retained endpoints g=x(p), h=x(q), and fits c(g,h). This assumption must itself be challenged with distinct paths sharing an endpoint.

Create a new retained scalar z and replace the old composition rule by

    (g,z) * (h,w) = (g+h, z+w+c(g,h)).                (1)

Associativity is equivalent to

    c(g,h)+c(g+h,k) = c(h,k)+c(g,h+k).               (2)

Thus the learned correction cannot be an arbitrary residual predictor. It must be a normalized 2-cocycle (c(0,g)=c(g,0)=0). Equation (2) follows by expanding both parenthesizations of (1); this is a checkable algebraic requirement, not a claim that all task residuals satisfy it.

With a general base group G, replace g+h by gh. The same identity yields an extension by an additive central coordinate. This first version only treats central extensions. Dependence on the pre-existing z or noncentral action is outside its scope.

### Actual acquisition rule

In the restricted, explicitly supplied family c(g,h)=g^T C h, every C satisfies (2). For basis segments E_i of displacement e_i, query F(E_i), F(E_i E_j), and F(empty). Set

    C_ij = F(E_i E_j)-F(E_i)-F(E_j),

after normalizing the empty-path output to zero. For arbitrary observations this is a least-squares fit to the outer products g h^T, with subsequent distinct-history and associativity checks. In the noiseless bilinear family the d^2 pair labels and d singleton labels identify C; this is not a claim of identifiability for unrestricted tasks. A rank-revealing factorization can reduce the cost of g^T C delta but does not create a new capability by itself.

Allocate z only if a held-out equal-endpoint contrast is nonzero and the proposed correction predicts additional concatenations. If the check fails, this specific extension family is rejected. No search over path words or unbounded synthesis oracle is used.

Given acquired C and a straight-segment output f(delta), execution is

    z <- z + f(delta) + x^T C delta
    x <- x + delta.                                 (3)

The order matters: the old x occurs in the first line. For the signed-area example below f=0. Formation costs O(d^2) feedback within the stated family and a finite linear solve. Dense execution costs O(d^2) arithmetic per increment and O(d^2) parameters, plus d+1 state scalars; for rank r, the bilinear term costs O(dr). No long trace needs to remain in execution memory. Formation, evaluator, and precision costs cannot be omitted from a later comparison.

### Complete acquisition and execution example

Take d=2. The evaluator is

    F(p)=A(p)=1/2 integral (x dy-y dx),

with coordinates relative to the start. The learner is not told this formula, but its candidate family above is supplied. Let X be displacement (1,0), Y displacement (0,1). Observations are

    F(X)=F(Y)=F(XX)=F(YY)=0,
    F(XY)=+1/2, F(YX)=-1/2.

The same endpoint (1,1) has incompatible outputs, so an endpoint-only readout cannot solve the task. Regression obtains

    C = [[0, 1/2],[-1/2, 0]].

The acquired local recurrence is therefore

    z <- z + (x delta_y-y delta_x)/2;
    (x,y) <- (x+delta_x,y+delta_y).

On the square L=X Y X^-1 Y^-1:

| Increment | New endpoint | Area increment | New z |
|---|---:|---:|---:|
| X | (1,0) | 0 | 0 |
| Y | (1,1) | 1/2 | 1/2 |
| X^-1 | (0,1) | 1/2 | 1 |
| Y^-1 | (0,0) | 0 | 1 |

The old state returns to its initial value; the extended state does not. Concatenating k copies has endpoint zero and z=k, by (1). The reverse loop gives z=-1. The state transition, not only the final decoder, now retains an ordered-history operation. This is exact for every finite polygonal path under the noiseless bilinear assumption, by segmentwise integration or induction using (1). No execution of this arithmetic was performed; the evidence is the displayed derivation.

Gauge changes matter. Replacing z by z+b(g) changes c to

    c'(g,h)=c(g,h)+b(g+h)-b(g)-b(h).

The symmetric part of a bilinear C can be removed by a quadratic b; the antisymmetric part cannot be removed this way. Thus not every new coordinate means an intrinsic new history distinction. This quotient by endpoint-only coordinate changes is part of the mechanism definition.

### What was acquired and what was designed

Designed: additive coarse state, concatenation queries, scalar feedback interface, bilinear central-extension family, test for adding a coordinate, arithmetic, and the ability to allocate state. Acquired: entries of C and therefore which earlier coordinates interact with which incoming increments in the new recurrence. Evaluated externally: F. Executed locally: (3). Retained: C and its accumulator rule. Neither algebraic primitives nor the candidate extension family emerged. A later task could use z, but no subsequent capability acquisition has been established. This is currently a constructive mechanism-identification example.

## 2. Strong internal attack: finite algebra closure is already matrix execution

Let A be a finite-dimensional unital associative real algebra, dim A=m. Let h(pq)=h(p)h(q), h(empty)=1, and let the output be a linear functional ell(h(p)). For each input atom a, right multiplication R_a:v -> v h(a) is a linear operator on the same m-dimensional vector space. Starting from 1 and multiplying these operators computes h(a_1...a_n) exactly. A linear readout produces the same output.

This is a direct same-dimension conversion to a matrix-product / weighted-automaton representation, not an appeal to universal simulation. It preserves scalar precision and can preserve sparse arithmetic. The algebra language alone offers no representational resource separation. For continuous increments, the same atom map can be used by the comparator; if its exponential is learned, it is a path-development/CDE construction.

For the area example, v=(1,x,y,z)^T obeys

    v_next = [[1,0,0,0],
              [a,1,0,0],
              [b,0,1,0],
              [0,b/2,-a/2,1]] v

for delta=(a,b). This conventional four-state linear recurrence has exactly the candidate's state, arithmetic, and outputs. Its matrix is exp(a B_x+b B_y), since the corresponding N has N^2=0. A trainable development layer can therefore contain the mechanism. For a finite alphabet containing X,Y and inverses, the function A has Hankel rank exactly four: concatenation gives rank at most four, while prefixes/suffixes (empty,X,Y,L) give the matrix

    [[0,0,0,1], [0,0,1/2,1],
     [0,-1/2,0,1], [1,1,1,2]],

whose determinant is -1/4. No smaller real weighted automaton with linear output realizes this exact function. The candidate meets that bound rather than beating it. The learning rule could still have an acquisition advantage over an unstructured fit, but a comparator given the same bilinear prior can implement the very same regression. There is presently no such advantage.

## 3. Substantive repair: acquire torsion-sensitive memory, not another polynomial level

The above attack is not repaired by naming the algebra differently or adding more signature coordinates. A more consequential limitation occurs if the target history distinction is torsion in the fundamental group. The source reading below changed the candidate: real/complex *homotopy-invariant finite iterated-integral functions* only see a specific nilpotent completion. They need not separate all homotopy classes.

Here is the key elementary special case. Let the loop group be Gamma=Z/2={e,g}. In R[Gamma], u=g-e spans the augmentation ideal I, and

    u^2 = -2u, hence I^n=I for every n>=1.

Every finite augmentation quotient R[Gamma]/I^(n+1) is therefore just R. It erases g versus e. Chen's checked finite-filtration identification makes this a barrier for homotopy-invariant iterated integrals in this class. Equivalently, an order-two element cannot map nontrivially to a real unipotent group: if U=I+N, N nilpotent, and U^2=I, then N(2I+N)=0 and invertibility of 2I+N gives N=0.

**Exact boundary:** this is not a barrier for arbitrary full signatures of embedded paths, for non-invariant features followed by nonlinear readouts, for general CDEs, or for finite matrices with eigenvalue -1. It is not a neural-network impossibility theorem. The repair must change the class of retained path representation, not claim that topology is unavailable to conventional computation.

### Actual discrete acquisition rule

Suppose the observed-state manifold M is supplied with a finite good cover and a finite nerve's 2-skeleton K. A trajectory determines a sequence of overlapping charts. The evaluator labels closed loops by a homomorphism y:pi_1(M)->F_2. This is a task assumption, not inferred from a few positive examples. Local geometry and sufficient path sampling must reliably determine chart transitions.

Unknown learned parameters are edge bits t_ij=t_ji. Enforce the triangle constraints

    t_ij+t_jk+t_ki=0 mod 2

for every filled triangle, so contractible local deformations cannot change output. Fix vertex gauge using a spanning tree and set its edge bits to zero. Each remaining edge e determines a fundamental cycle p_e. Query y(p_e) and assign t_e=y(p_e). If any filled-triangle constraint fails, reject the assumed flat double-cover model or flag noisy labels. This uses at most E-V+1 loop labels; solving the binary constraints and constructing a tree have explicit finite polynomial costs. One can instead solve a general binary linear system from any sufficiently informative loop labels.

At execution retain the current base/chart state and one sheet bit s. On a chart transition i->j:

    s <- s+t_ij mod 2.

The predicted closed-loop class is s. Arbitrary subdivisions, backtracking, and replacements across filled triangles preserve it. No enumeration of complete path words is needed. The learned t determines a flat double cover (up to vertex-gauge changes). This is an implemented-by-equations acquisition rule, not a request to let a network discover a cover as a black box.

### Complete topology example

Take M=SO(3), whose double cover is the unit quaternions. Consider rotations R(theta) about a fixed axis n. Their continuous quaternion lift is

    q(theta)=cos(theta/2)+n sin(theta/2).

The loop theta:0->2pi returns R to I but ends at q=-1; theta:0->4pi ends at q=+1. In the standard one-cell presentation of the same fundamental-group distinction, a loop generator a has relation a^2=e. One label y(a)=1 learns the nontrivial edge bit; relation a^2=e is respected since 1+1=0 in F_2. Thus one execution gives s=1 and two give s=0. The continuous lift shows this is not an arbitrary marker on a word: it distinguishes actual rotation histories with identical observable endpoints.

For sampled rotations, a conventional exact execution also exists: choose the unique near-identity unit-quaternion lift of each relative rotation R_k^-1 R_(k+1), requiring every relative rotation angle <pi, and multiply it into q. This exposes both the mechanism and the strongest simple rival. Its SU(2) state separates the torsion erased by unipotent homotopy-invariant coordinates.

### Why this repair is substantial, and why it is unfinished

The residual can require a different coefficient group and global sheet state, rather than higher real polynomial order. Local differential structure alone does not select this extension. A local Lie-algebra fit cannot decide between SO(3) and SU(2), whose Lie algebras agree; a global loop observation is necessary for this distinction. That is a concrete organizational distinction with a structural reason to extend beyond the example.

But the learner was supplied a good cover/nerve, the candidate F_2 fiber, homotopy-invariance, and access to informative loop labels. Turning a neural system's self-generated representation into a reliable cover with task-relevant coefficient group is still open. The example acquires a cocycle in a supplied family, not unrestricted internal mechanism generation. Ordinary topology-based state lifting and a two-state parity transducer on the same chart events perform the same acquisition and execution. Giving the candidate known topology while making a rival infer it would be an invalid comparison.

## 4. Evidence and one decision

| Claim | Status | Evidence/artifact | Weakest remaining link | Failed branches | Next single decision |
|---|---|---|---|---|---|
| A bilinear composition residual acquires an exact area accumulator | RECONSTRUCTION; algebra checked on paper | Sections 1–2 | Supplied residual family; no empirical learning evidence | Endpoint-only memory | Do not implement this as a new-principle claim |
| Finite associative path algebra with linear output converts to equal-dimensional matrix execution | VERIFIED in displayed algebraic sense | Right-regular representation derivation | No general acquisition lower bound follows | Claim of new execution class from algebra closure alone | Treat weighted automata/development as mandatory matched rivals |
| Real homotopy-invariant unipotent memory cannot retain the Z/2 loop class | SOURCE plus special-case derivation | Chen 1975 Theorem 5.3; u^2=-2u proof | Exact invariance and representation restrictions | Increasing polynomial truncation within that restricted class | Use only as a precise boundary, not broad signature failure |
| Task labels can select a finite-sheet cocycle | RECONSTRUCTION; finite algorithm stated | Section 3 | Geometry, fiber, and informative global labels supplied | Purely local Lie-algebra acquisition cannot pick global cover | Compare against topology-based state lifting before further commitment |
| This is a consequential new learning principle beyond conventional models | OPEN / currently unsupported | No supporting resource or acquisition separation | Strong exact matched conventional construction | Area/signature-only novelty; unqualified topology impossibility | Root should decide whether the unsupplied cover-acquisition gap is worth a separate finite theoretical question; no implementation is authorized by this note |

Current recommendation: **do not promote the finite path-algebra seed to the main line**. Preserve the torsion-sensitive repair as a concrete source-informed question only if it exposes a gap absent from the current topology mechanism. Its present finite acquisition algorithm is conventional cohomology/cover inference; the missing principle is how a learner chooses and acquires the appropriate extension without being handed its topology or answer-bearing grammar.

Primary-source locations, reading boundaries, and failed fetches are in ../../sources/after012-b/reading.md. No novelty is claimed.
