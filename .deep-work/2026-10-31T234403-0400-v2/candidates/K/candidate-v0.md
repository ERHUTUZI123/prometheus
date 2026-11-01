# K v0: learn an execution algebra, with explicit exceptions

Status: HYPOTHESIS; substantive first freeze before cross-input. No neural acquisition result or novelty claim.

## Judgment and scientific opportunity

The candidate is to let a neural learner change the algebra in which its internal program executes. Keep a general, locally executable tensor program; learn coordinated changes of basis at its internal ports, then replace regions that satisfy a tractable algebra with an exact compiled mechanism. Permit arbitrary gates outside that algebra, accounting for the boundary through which those exceptions interact with the tractable region. Learn to reduce that boundary while learning the task.

The important possible change is that acquiring a capability need not require paying for the learner's original internal execution forever. A learner could first construct a useful but expensive internal mechanism, subsequently discover the relations that make it compressible, and retain an executable algebraic implementation as a reusable object. Optimization of the implementation can preserve the function exactly, so retention does not depend on re-estimating old task losses. The computational resource being organized is algebraic interaction across an exception boundary, rather than just the number of neurons or graph edges.

This is not yet a new principle established beyond existing holographic algorithms, tractable-backbone methods, and program superoptimization. The central unproved hypothesis is that this organization supplies a useful acquisition path: neural search can construct capabilities outside a current tractable core, then learn interface algebra that absorbs or localizes them, and reuse the result on new compositions. Merely recovering a known hidden parity or matchgate basis would not establish that hypothesis.

## Original theory selected and its challenge to the initial idea

I independently chose Valiant's *Holographic Algorithms* (author manuscript), an unfamiliar technical theory for this entry. Reading entered the definitions, Holant theorem, four-port signature identity and the polynomial-system construction, rather than relying on the abstract. The technical reading record is in `sources/K/source-record.md`.

SOURCE: Valiant's method already coordinates a basis with local gadgets; the global sum is preserved, with a polynomial evaluator for appropriate planar matchgate realizations. His Section 11 already proposes semi-mechanized search through polynomial systems. Thus “a network learns a basis revealing a known efficient algorithm” is insufficient as a new mechanism. Cai/Guo/Williams sharpen this attack: basis-finding for known affine/product signature classes is already algorithmic.

RECONSTRUCTION: The useful departure for the neural question is not a new holographic identity. It is to make the **currently available execution algebra, the location of exceptions, and certified reusable compiled regions** part of the learned state, with an explicit path from general neural computation to a cheaper retained mechanism. Whether that departure is scientifically substantive remains OPEN.

## State and actual computation

Use a finite planar bipartite port graph G=(U,V,E), with ordered binary ports, maximum degree d, and rational or explicitly represented algebraic tensor entries. Fixed d is an early restriction, not a universal claim. Inputs choose or parameterize local tensors. Outputs may be open ports; equivalence then means equality for every boundary assignment, not equality only on training examples.

The semantic program computes

    Z_theta(x) = sum_{s:E -> {0,1}} product_v F_v(s restricted to ports(v); x, theta).

This definition supplies an exact but potentially expensive executor. It is a sum-product computation, not a claim that arbitrary conventional nonlinear NNs admit a cheap tensor encoding. Small fixed-arity dense tensors and neural controllers generating their entries are legitimate starting representations. Conversion costs and tensor arities must be counted.

Learned state at step t:

    S_t = (G_t, F_t, phi_t, {B_e}, D_t, L_t).

Here phi is the neural construction/compiler policy; each B_e is an invertible port transformation; D is the set of uncompiled exceptions; and L is a library of compiled regions with boundary tensors, realizations, exact certificates and measured or derived execution costs. Both the semantic parameters F and the compiler/library are retained. There is no evolutionary inheritance or population-selection claim in v0.

For edge e=(u,v), define transformed tensors by applying B_e to the u port and B_e^{-T} to the v port. In indices,

    F'_u(...,i,...) = sum_a (B_e)_{ia} F_u(...,a,...),
    F'_v(...,i,...) = sum_b (B_e^{-T})_{ib} F_v(...,b,...).

Since sum_i (B_e)_{ia}(B_e^{-T})_{ib}=delta_ab, contracting that edge preserves the whole open-boundary tensor. Applying this separately to every edge remains exact. This identity is an independently reconstructed algebraic calculation, not a new theorem. Orthogonal rational B_e are an initially useful subset, because they avoid arbitrary norm-scaling cheats. General B require condition-number and bit-length accounting.

The compiled core initially uses planar matchgate signatures. For a normalized four-port even signature A, with bit strings in cyclic port order, the source relation is

    A_0000 A_1111 - A_0011 A_1100
        + A_0101 A_1010 - A_0110 A_1001 = 0,

with all odd components zero. Use the source's nonzero-chart conditions or an explicit matchgate realization; do not treat one residual test as a universal recognizer in higher arity. In v0 a core gate is accepted only with its actual realization/certificate. Each compiled connected region must expose its ordered boundary and preserve the required planar gluing conditions.

## Explicit transition and learning law

1. **Construct/adapt.** The neural policy proposes a new local tensor, rewiring, or module invocation using task feedback. This changes semantic computation and is evaluated on the task. Dense residual gates are allowed; do not discard a useful unfinished computation merely because it is outside the core.
2. **Re-express.** Holding the semantic program fixed, update B and propose connected regions to compile. Gauge moves preserve all outputs exactly. A proposal may contract a bounded region, inspect its boundary tensor, and seek a matchgate realization. Exact table comparison verifies finite fixed-arity replacements. Repeated use of this route must charge the exponential boundary-table discovery cost.
3. **Accept/retain.** Replace a region only after a certificate of its complete boundary behavior passes. Add the representation, port conversion, certificate and executor to L when its amortized cost improves. Later compatible compositions invoke it as an internal operation. No task-data-only equivalence certificate is accepted.
4. **Train the proposer.** For a proposed compiler action a use reward

       r(a) = C_before - C_after - C_discovery - C_verification,

   with actual or conservatively estimated amortization horizon specified. A basic implementable policy update is REINFORCE,

       phi <- phi + eta (r(a)-baseline) grad_phi log pi_phi(a | S).

   Continuous gauge search may use normalized matchgate-identity residuals as a search heuristic, but exact acceptance still requires a certificate. The semantic tensor parameters use task gradient or finite task-feedback search separately. The compiler reward does not certify that semantic learning has improved.

For fixed accepted B, D and core, exact execution cuts out D, enumerates the binary indices on its interface to the core, evaluates the defect tensors and the pinned core, and sums. Let b be the number of distinct interface/internal defect-edge bits actually enumerated. Cost is at most 2^b times a polynomial core evaluation cost plus exception evaluation. With k bounded-degree exceptional vertices, b <= dk (up to explicitly counted open-input handling). This is an elementary cutset-conditioning bound, not a new complexity result. It makes the exception cost explicit instead of silently calling a dense generic contraction efficient.

Composing two retained mechanisms retains their complete boundary semantics. Their efficient evaluators compose cheaply only if the glued core remains in the tractable family or the additional defect boundary is charged. “Modules are individually cheap” does not imply arbitrary compositions are cheap.

## Concrete execution, specified before code

One normalized four-port core tensor has

    A_0000=1; A_1100=1; A_1010=2; A_1001=3;
    A_0110=4; A_0101=5; A_0011=6; A_1111=8;

and all odd entries zero. Its identity is 1*8 - 6*1 + 5*2 - 4*3=0. Two copies joined along reversed cyclic port order form a planar two-vertex multigraph. Relabel the second tensor's ports accordingly; its scalar is sum_s A_s^2=156. This is a hand reconstruction, not an executed result at freeze.

Apply Q=(1/5)[[3,-4],[4,3]] on every paired port at both ends. Q^T Q=I, so the now generally dense tensors still compute 156. Change one original A_1111 to 9: the identity residual becomes 1, and the scalar becomes 164. This is a genuinely outside-core tensor in that chart; retaining it as one exception gives exact execution by conditioning on its four ports. Treating it as a core tensor by zeroing the residual changes the function. These finite examples exercise exact re-expression and exception accounting only. They do not demonstrate learned construction, task acquisition or a meaningful speed advantage.

## Scaffold versus acquired

Designed: binary port semantics, general tensor executor, neural proposal/update machinery, exact certificate kernel, initial matchgate theory/evaluator, cost accounting and external resource limits. The Pfaffian algorithm does not emerge in v0.

Potentially acquired: task tensors/topology, coordinated port bases, which regions admit compilation, locations and interfaces of exceptions, reusable compiled boundary mechanisms, and a policy for proposing these changes. No such acquisition has yet been demonstrated. To count as a stronger contribution, a later result must show a useful relationship/executor or reusable decomposition that was not placed in the constructor or task generator as the answer.

## Success counterfactual and significance boundary

If the strong version works, neural learning can construct an initially general internal program, identify an algebra making substantial portions exactly cheap, and preserve that mechanism across subsequent capability construction. Reuse would be evaluated on independent compositions and on additional semantic learning, with full compiler/discovery cost. This could change how neural systems allocate computation and retain algorithms: mechanism learning becomes movement between execution algebras with explicit hard residuals, rather than only parameter fitting inside a fixed execution scheme.

If the outcome is only faster basis recognition for a deliberately obfuscated known class, it is an incremental learned compiler. That would verify machinery but would not supply the hoped-for organizational principle. If good task performance needs an extensive exception boundary, the tractable core fails to change relevant computational resources. Those are real negative outcomes.

## Strongest rivals and hardest missing step

The strongest matched rival is a deterministic algebraic recognizer/compiler for the same signature class, plus the same cutset-conditioning executor and unrestricted task learner. Give it identical tensor entries, exact arithmetic, allowed rewrites, compiler budget and retained library capacity. A second rival is a neural task learner with the algebra and exception arrangement supplied directly, to identify whether the acquisition path itself adds anything. Equality-saturation superoptimization with learned rewrite selection is a third close mechanism; local rewrites/library retention are not novel merely because the proposal network is neural.

The hardest missing step is **acquisition of useful algebraic organization beyond a supplied tractability oracle**. Exact compiler moves cannot create task capability: they stay in a semantic equivalence class. Task learning must create useful computation, and the compiler must then uncover more than a relabelled known backbone or cheap cancellation planted by the task generator. The v0 laws specify a route, not evidence that this hard step is possible at useful cost.

Strong attacks: gauge transformations preserve tensor cut-ranks and do not magically lower generic contraction complexity; approximate algebraic identities can yield badly wrong global sums; numerically small residuals can be pure scaling artifacts; open-boundary realization discovery may be exponential; compiling a region can hide future adaptation costs; and learning a fixed Gaussian/parity core with sparse non-Gaussian exceptions has substantial prior art in quantum simulation and circuit optimization. No claim of open-ended evolution, universal self-improvement or paradigm-level result is justified.

Next single scientific decision after independent review: is there a concrete acquisition-and-composition phenomenon here that distinguishes the adaptive algebra/interface process from a deterministic compiler plus fixed tractable core? If none survives, close this candidate as an instructive reconstruction rather than escalate implementation.
