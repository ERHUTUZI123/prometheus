# Frozen repair v1: the local composition law and its state type are acquired

Status: AGENT PROPOSAL with a complete hand construction; no neural training or experiment. This repairs the fixed-algebra weakness in v0, not the significance weakness established below. This is an explicit different mechanism, but the present realization does **not** survive its strongest simple rival as a foundational proposal.

## Provenance and convergence

v0 was frozen before any campaign candidate inputs. Root then directed me to K/candidate-v0.md, K/candidate-v1.md and sources/K/source-record.md. I read those files. The paired inverse transformations, tractable evaluator, policy-gradient search and analytic-recognizer rival are substantive independent convergence with K. My v0 is weaker: K additionally specifies exceptions, editable-family retention and full semantic gradients. No inference that this convergence independently verifies mathematics is intended.

This repair changes the acquired mathematical object. v0 learns coordinates for a supplied algebra, preserving the entire function. v1 changes the composition law itself from observed causal behavior, and increases the dimension of its operand type after a genuine behavior-span witness. Its semantic transitions can therefore add a computational operation rather than only replace an equivalent implementation. The construction below shows this distinction precisely; it also exposes why the difference alone is insufficient for significance.

## Specific question and intended consequence

Can a neural learner form an internal domain of operations whose elements, multiplication law and necessary dimension are inferred from the operations' actual interactions, without preassigning a collection of human-named modules?

The ambitious success counterfactual is acquisition of a compact, closed internal arithmetic that makes newly encountered operations composable with retained ones, including interactions outside the initially represented span. Instead of predicting which library function to call, the learner would change what its operation-valued states mean and how those states combine. This bears directly on construction of internal computational mechanisms. It is not a claim that every useful computation has a small finite-dimensional algebra.

## Explicit state, execution and adaptation

At time t retain

    A_t = (d_t, C_t, e_t, encoder_phi, decoder_psi, probe_records),

where C is a d-by-d-by-d real structure tensor and e is the unit element. A state z is an element of R^d. The learned computational primitive is

    (z star_t w)_k = sum_(i,j) C[t,k,i,j] z_i w_j.

C is not restricted to be a basis transform of an initial multiplication table. A word of observed operations o_1,...,o_n executes as

    z_0 = e_t
    z_(r+1) = encoder_phi(o_(r+1)) star_t z_r
    y = decoder_psi(z_n,x).

If an operation denotes a linear map on an underlying vector input, the decoder may be linear in both the operation coordinates and x. That is a restriction used by the exact worked case, not an assumption that general neural computation is linear.

One explicit update objective is

    L = L_task
        + lambda sum_(a,b,x) ||D(C(E(a),E(b)),x)-D(E(a),D(E(b),x))||^2
        + mu L_unit + nu L_associativity + rho d^3.

The composition examples require actual interaction queries or supplied executions; they are additional information and must be equally available to rivals. L_associativity is the sum of squares of the coordinate equations

    sum_m C[m,i,j] C[k,m,l] = sum_m C[m,j,l] C[k,i,m]

for every i,j,k,l. Gradient steps update C, phi and psi. If exact associativity is required, project to the constraint variety with an explicitly bounded solver, or accept only a coefficient table whose polynomial identities pass exact checking. A small residual is not a proof. Associativity/unit constraints are supplied structure, but the table satisfying them is not fixed.

**Formation law:** do not infer dimension insufficiency from unsuccessful optimization. For the certified linear setting below, maintain response vectors R(T) of operation T on a known spanning probe set. These vectors are an injective representation of the permitted operation class. For candidate interaction T_i composed with T_j, compute

    r = R(T_i T_j) - P P^+ R(T_i T_j),

where columns of P are the current independent operation responses. In exact arithmetic, r != 0 and a rank-(d+1) minor certify that this interaction lies outside the current d-dimensional behavior span. Add a new basis element with response r (or the complete new response after a basis change), embed old states z -> (z,0), and reidentify the affected multiplication coefficients.

For observed products the coefficient update can be made concrete:

    C[:,i,j] <- argmin_c sum_x ||sum_k c_k R_k(x)-T_i(T_j(x))||^2.

With complete exact spanning probes, a full-rank P, and closure, this minimizer is unique. An external exact least-squares solver is legitimate but is then part of the scaffold, not a discovered neural ability. A neural proposal may select interactions or initialize coefficients; no advantage is inferred from placing it there.

If R is only finite sampled behavior of arbitrary nonlinear operations, a nonzero residual only establishes a mismatch on those probes. It does not prove global inequivalence, a universal dimension lower bound, or that adding a state variable is the right remedy. The witnessed necessity is relative to this model family and certified information source.

## Complete worked acquisition process

This is a hand derivation showing actual formation and refinement; it is not proposed as a new experimental benchmark or as a rediscovered-algebra result worth training. The learner is given access to two initially unnamed operations on R^2. The task interface guarantees they are linear and permits exact queries at e_1=(1,0), e_2=(0,1). Without this input guarantee the inference below would not be valid globally.

The returned responses are

    A e_1=e_1,  A e_2=-e_2
    B e_1=e_2,  B e_2=e_1.

The learner starts with states for the identity I, A and B. Its decoder is

    D(z,x) = (z_0 I+z_1 A+z_2 B) x.

Those three behavior columns are independent. The current type fits each primitive response, but the probe of the interaction A after B returns

    AB e_1=-e_2,  AB e_2=e_1.

In the response order (first output on e_1, second output on e_1, first output on e_2, second output on e_2),

    R(I)  = (1,0,0,1)
    R(A)  = (1,0,0,-1)
    R(B)  = (0,1,1,0)
    R(AB) = (0,-1,1,0).

The fourth vector is orthogonal to the first three and nonzero. Thus this is not an optimizer's failed fit: no three-coordinate linear behavior decoder spanning I,A,B can represent AB. The formation law creates a fourth element c=ab. The new internal operation was not supplied as a named library primitive; its behavior came from an actual interaction query. That is the sense in which this process constructs an internal computational element.

Further interaction probes (or direct composition of the now-identified linear maps) determine

    a*a=1, b*b=1, a*b=c, b*a=-c,
    c*c=-1, a*c=b, c*a=-b, b*c=-a, c*b=a.

The resulting acquired multiplication primitive on z=(z0,z1,z2,z3), w=(w0,w1,w2,w3) is

    (z*w)0 = z0*w0 + z1*w1 + z2*w2 - z3*w3
    (z*w)1 = z0*w1 + z1*w0 - z2*w3 + z3*w2
    (z*w)2 = z0*w2 + z2*w0 + z1*w3 - z3*w1
    (z*w)3 = z0*w3 + z3*w0 + z1*w2 - z2*w1.

The decoder becomes D(z,x)=(z0 I+z1 A+z2 B+z3 AB)x. Associativity follows from the injective map to operator composition; it can also be checked by the finite coefficient identities. No componentwise multiplication on the original three states could produce this table. It changes the computation law and operand type, not only the coordinates of an unchanged three-dimensional operation algebra.

As an untrained composition, let the runtime request the word A B A B (rightmost executes first). The learned primitive executes

    a*b=c, c*a=-b, (-b)*b=-1,

and returns -x for every input x. For the much longer word (AB)^n, the same acquired law gives the exact four-cycle 1,c,-1,-c. This consequence is not itself a new capability over matrix multiplication; it only demonstrates internally composable acquired behavior beyond the primitive observations.

Retention consists of the learned table, state encoder and decoder, not stored answers for those words. The dimension can later grow again on a certified independent behavior. There is no assertion that this growth continues indefinitely or becomes more computationally efficient.

## Strongest simple rival, with the same information

Now construct the rival rather than naming a distant literature. The exact primitive probes already identify

    A=[[1,0],[0,-1]], B=[[0,1],[1,0]].

Rival D stores these matrices and uses ordinary matrix multiplication for every composition. It obtains AB, the whole table, and every held-out word using the same input information. The four-dimensional learned algebra is exactly M_2(R) in the basis I,A,B,AB. It has four coefficients, the same as a generic 2-by-2 matrix, and the displayed product has the same constant-scale arithmetic cost as ordinary matrix multiplication. The dimension-refinement event is standard closure of an operator span; no neural discovery advantage is shown.

This rival escapes none of the rules imposed on the candidate. Giving D only final task labels while giving the candidate exact intervention probes would be an information mismatch. Excluding matrix multiplication because it is designed would contradict the project's permission to use existing arithmetic primitives.

For sequence-only scalar responses, the same construction becomes the search for a finite-dimensional representation of compositional behavior; weighted-automaton/Hankel realization is therefore a necessary close comparison. No claim is made here about which theorem covers arbitrary nonlinear/state-dependent operations. That extension has not been specified or earned by the finite example.

## Decision and remaining boundary

The repair is substantively different from K: it changes semantic composition and learns a type that is not available at initialization. It therefore escapes the *fixed-algebra, function-preserving compiler equivalence*.

It does **not** escape ordinary operator-algebra identification. The strongest finite realization recovers existing linear algebra with a neural proposal attached. It does not presently support paradigm-level significance, a new learning principle, or an advantage over a fixed neural architecture plus explicit operator representation. Running a training demo on this supplied linear task would not reverse that judgment.

A possible consequential successor would need an explicit mechanism where formation of an operation algebra changes acquisition or interaction capacity in a way the matched direct operator learner cannot already implement cheaply. Merely increasing dimensions, learning another multiplication table, observing associativity violations, or rebranding this as autonomous arithmetic is insufficient. No such successor mechanism is claimed here. This finite branch should stop rather than generate a harder version of the same example.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

Changing structure constants plus a witnessed type refinement can add semantic computation beyond gauge-only recompilation | VERIFIED in the stated finite linear model by explicit calculation, not machine checked | v1 worked process | exact spanning probes and linearity supplied | none executed | retain as a distinction, not a research direction.

This yields a consequential new neural acquisition principle | NOT SUPPORTED | direct matched matrix rival above | all demonstrated content covered by ordinary matrix representation and closure | v0 converges with K; v1's strongest worked case is ordinary operator-algebra identification | root decides whether a genuinely different scientific entry remains worth pursuing; no new probe authorized by this artifact.
