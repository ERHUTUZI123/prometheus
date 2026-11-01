# P v1a — one explicit acquisition law for an intervention interface

This supplements, rather than replaces, candidate-v1.md. It supplies an exact restricted family in which the learner can *acquire* a compact persistent-edit interface and generate a new coordinated update operation. It is not proposed as the main scientific task, a novel linear-algebra result, or evidence of a general neural solution. No code was run for this derivation.

## An edit response class

Let a bounded neural computer have n persistent binary control choices u in GF(2)^n. They can select routing, feature inclusion, parameter variants or other specified interventions; they are not new per-example inputs. Its forward computation may be rich, but assume the responses being modeled are affine in these controls:

    r_j(u) = a_j u XOR b_j,        a_j in GF(2)^(1 x n).              (1)

Each j may denote a particular task input/readout or an independently checked function coefficient. One legitimate neural realization is

    F_u(x) = h(x) XOR XOR_i [u_i AND g_i(x)],

where h and g_i are fixed acquired neural Boolean computations during this pass. This is arbitrary computation in x, restricted affine control in u. The restriction is designed/verified, not inferred solely from successful single-bit probes. Learning new h or g_i changes the model and triggers a new interface pass.

The learner does not receive a_j. At a common base u, it executes each single-control intervention e_i. Under (1),

    a_j[i] = r_j(u XOR e_i) XOR r_j(u).                              (2)

One baseline plus n executed control settings identifies all currently observed rows simultaneously. Costs count every forward execution/readout, including all task inputs. Across N examples, the learner stacks the corresponding rows; it never chooses a different persistent u for each example.

## Acquiring and expanding the interface

Maintain B, a row basis of the response rows whose distinctions are currently required. Define the edit interface

    sigma(delta_u) = B delta_u.

Every required response change is computable from this interface, since its row is a linear combination of rows of B. When a new executed operation/query yields row a:

* If a lies in rowspan(B), retain the interface and record the corresponding linear readout.
* Otherwise append a as a new independent interface coordinate and update the solver/repair routine.

This gives a concrete growing-language rule, with interface dimension bounded by n in this family. It stores the effects of proposed persistent edits, not merely their current forward values. Old coordinates are recovered by projection, matching v1's refinement law. Gaussian elimination is a designed internal operation here; a network may learn to invoke or implement it, but no such learning is claimed.

## Creating a new learning operation while retaining old responses

Suppose B contains all old response constraints to retain. To independently flip a newly demanded affine response a u XOR b, seek v satisfying

    B v = 0,             a v = 1.                                  (3)

Then the executable macro-edit

    ChangeNew(u) = u XOR v

preserves every old response and flips the new one exactly in the stated class. A new multi-parameter update action has been constructed from executed effects and retained constraints; future learning can call it as one operation with an inspectable physical effect.

**Existence criterion.** Such v exists iff a is not in rowspan(B). One direction is immediate: if a is a combination of rows of B, Bv=0 forces av=0. Conversely, if a is not in the rowspace, the linear map [B;a] has one greater rank than B; the equation with right side (0,...,0,1) is consistent. Equivalently, the annihilator of ker(B) is rowspan(B). This is established finite-dimensional linear algebra, used as a construction rather than a novelty claim.

There is no gratuitous new capacity. Once old constraints span all n control directions, no independent retained-behavior-preserving action exists. The learner must report the obstruction, change its representational/operation repertoire in an authorized constructive step, or reconsider a retention demand. A vanished nullspace is not proof that the entire neural computer cannot learn a new capability.

## Worked coordinated edit

Let u=(u1,u2,u3,u4), initially 0000, and retain

    r1=u1 XOR u2,
    r2=u2 XOR u3.

The new requested response is r3=u1, to be set to 1. B has rows 1100 and 0110; a=1000 is independent. Equation (3) forces v1=v2=v3=1. With Hamming edit cost, v=1110 is a minimum solution of cost 3. Executing it yields u=1110:

    old r1=0 remains 0;
    old r2=0 remains 0;
    new r3 changes 0 -> 1.

Neither flipping u1 alone nor treating the old response constraints independently gives this safe coordinated action. The macro is an update-space mechanism acquired from response relations. A later query r4=u4 adds another independent coordinate and can be controlled with 0001. A query r5=u1 XOR u3 is already the sum of the old two rows, so it cannot be independently flipped while both old responses remain fixed. The interface tells the learner the difference between new controllability and a relabeled old effect.

## What this resolves and what remains ambitious

The N-example repair relation does not require an alphabet of size 2^N in this class: at most n independent response rows suffice. Plain elimination solves feasible persistent updates in polynomial work. This explicitly realizes v1's compact-summary requirement for one closure family and constructs a subsequent learning action. It also states exactly which closure property is doing the work.

It does **not** show that the system learns useful h/g mechanisms, that arbitrary neural responses are affine in useful edit coordinates, or that a repair interface remains compact when such closure is absent. A fixed feature model with an exact linear solver is an equally strong control here. The affine family should not become the main research question merely because the proof is convenient.

The ambitious extension is constructive: can the neural computer discover parameterized *macro-edits and response coordinates* in which useful new nonlinear routines admit composable low-complexity repair, rather than merely projecting ordinary gradients or choosing an already supplied linear head? Interface selection then changes which coordinated modifications are cheap to discover. New routines could preserve that structure, introduce a genuinely new repair mode, or reveal a dependency requiring refinement. V1's executed repair conflicts and learned cost-summary composition specify a route, but no general acquisition law for this richer setting has yet been established.

## A sharp nonlinear attack

Take one response r(u1,u2)=u1 AND u2 at base 00. Both single-bit probes in (2) give zero, so an unrestricted learner would infer a=00 and predict no possible effect. The two-bit intervention 11 changes the output. Thus n+1 probes do not identify an arbitrary neural edit-response law. Exactness requires the affine scaffold/closure assumption, an independent proof of it, or complete finite coverage; a few successful checks cannot establish it. A higher-order residual would need an acquired interaction mode, not a claim that the old row basis remains sufficient.

This attack is central to the next constructive problem: how can a learner identify and package an interaction such as a coordinated two-bit change as a new repair operation, then reuse its conditional effects across new routines? A recipe that merely increases polynomial degree until it enumerates all 2^n interventions would lose the intended learning advantage.

## Strong actual neural comparison

Orthogonal Gradient Descent already preserves old neural outputs approximately by restricting updates relative to stored output-gradient directions. Its original Section 3 explicitly uses a local approximation at an old task's optimum. Thus a basis of old effects plus nullspace updates is not a new idea. See sources/P/source-record-v1a.md. V1a's finite exact law is a restricted reconstruction, not a novelty boundary against that method.

The continuing candidate is v1's acquisition and composition of *nonlinear, set-valued, cost-aware repair programs with changing interfaces*. Its affine component must be compared to OGD/ordinary projected optimization, its inverse composition to DTP and deductive synthesis, and its general structural inference to a source-graph-aware solver. A useful result must come from the proposed learned coupling, not withholding information or weakening these controls.

CLAIM | STATUS | EVIDENCE | WEAKEST LINK | NEXT DECISION
--- | --- | --- | --- | ---
An intervention interface and a new coordinated update action can be acquired from finite executions in the stated affine class | RECONSTRUCTION with explicit laws and proof | Equations (1)–(3), worked example | Known closure scaffold; no neural acquisition evidence | Find the nonlinear interaction mechanism that extends this family without exhaustive edit enumeration
This suffices for general neural self-improvement | NOT CLAIMED | Nonlinear AND attack | Compact nonlinear repair composition remains open | Continue v1's constructive interaction/coordination mechanism
