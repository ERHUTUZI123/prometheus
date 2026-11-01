# Write-control audit: a retained mode must be both persistent and movable by task-relevant inputs

RECONSTRUCTION / MATHEMATICAL SUPPORT for the same protected lineage. This note does not change the completed pilot, its contract, the carry audit, any candidate or campaign status. No experiment, training, parameter search or new label acquisition occurred. Root will freeze any subsequent acquisition process.

## Actual starting point and information boundary

I read the actual complete row and final evaluation rows in `probes/after020-local-contrast/pilot-v1.raw.jsonl`. The mean local-carry arm did succeed in the declared single-cue diagnostic: its final parameters are approximately A=1.388886, beta=0.315054 and v=1.236622, with terminal half-squared loss about 3.66e-16. The already-persistent-initialization endpoint control also succeeded. Therefore the possible dilution condition in the earlier audit is not a diagnosis of failure in this run and requires no repair now.

Both learned persistent models preserve the first sign in the reported changed-write evaluations. The raw record explicitly scores **both** first-cue and last-write interpretations. Those alternative target columns are evaluation conventions, not newly received task feedback. The complete row records two endpoint labels, the original single-cue pair. This note does not imply that opposite-write endpoint labels have already been obtained or that last-write semantics are the intended task.

The next mathematical issue, conditional on actual new task feedback, is whether the acquired dynamical mode can be controlled by an input that should change its stored value. Persistence alone does not imply that capability.

## 1. Exact opposite-pulse conditions

Use the actual ungated family

    r_next=tanh(A*r+beta*x),    answer=v*r,               (1)

with A>=0 and a fixed positive state/readout orientation. Let a prestate be r=-s*a, where s in {-1,+1}, a=|r|>=0. Suppose an observed pulse x=s should establish the new sign s. Its signed post-write state is

    s*r_next=tanh(beta-A*a).                             (2)

Thus:

    correct new sign  iff beta>A*a,
    new signed state >=m iff beta>=A*a+atanh(m),          (3)

for 0<m<1. Equality in the first inequality produces zero state, not an overwritten sign. The second condition is a margin statement about the actual next state, stronger than mere sign change.

For arbitrary previous states |r|<=R, the worst state opposes the pulse, so a sufficient and necessary uniform condition within this family is

    beta>=A*R+atanh(m).                                  (4)

Here R must be a justified reachable-state/precision bound. Taking R=1 is conservative for the ideal tanh recurrence. Fitting only long-gap prestates near a particular attractor does not establish (4) for all shorter gaps or repeated pulses.

The observed local-carry mode has a long-hold magnitude about 0.80865. A>1.38 and a>0.8 imply A*a>1.104, whereas beta<0.316. Its opposite pulse therefore cannot change sign under (3). This explains the reported behavior without treating that behavior as an error relative to labels that were never supplied. The persistent-initialization control has an even smaller beta relative to its retained state's recurrent drive.

## 2. Hold and write have a compatible feasible region

On zero input, (1) reduces to f_A(r)=tanh(A*r). The signed band [m,1] is forward invariant exactly when

    A>=A_hold:=atanh(m)/m.                               (5)

This follows from monotonicity: the minimum next magnitude in that band is tanh(A*m). The negative band has the same property by odd symmetry.

Combining (4) and (5), a robust sign-memory controller in this ideal family can satisfy

    A>=atanh(m)/m,
    beta>=A*R+atanh(m).                                 (6)

Then, by induction, every observed-sign pulse writes the required band and every subsequent zero input holds that band, until the next pulse writes a new one. This is a complete conditional write-hold-repeat execution argument, not a claim that the training has acquired it.

Under bounds A<=A_max and beta<=beta_max, this sufficient family is nonempty if

    A_max>=A_hold and beta_max>=R*A_hold+atanh(m).        (7)

For the existing m=0.4, A_hold is about 1.059. With R=1 and the existing bounds A_max=2.5, beta_max=4, the region is nonempty. Even at A=2.5 the universal margin condition needs beta at least about 2.924, within that bound. At a fixed A around the observed 1.389, beta around 1.813 suffices for every ideal prestate in [-1,1]. These are analytic feasibility values, not proposed settings or a search result.

This shows a specific useful fact: the current ungated family need not be replaced merely because its learned first-cue mode resists the tested pulse. Its present beta and its acquired semantics/learning signal are separate issues from write-hold representability. If future feedback instead distinguishes relevant writes from distractor pulses, the simple beta*x drive may be insufficient; no selective gate has been acquired by this argument.

## 3. What new endpoint feedback could license locally

Suppose a future authorized acquisition obtains actual completed histories with differing write events followed by the same remaining input word and available query context. Their endpoint labels are observed, rather than inferred from the input pulse. This may be a pair of opposite overwrites after different old signs, or histories sharing a prestate and then receiving distinct write pulses.

The general inference from such a pair is only that a correct deterministic computation must preserve whatever distinction produces the different future answers. It does **not** identify a unique scalar coordinate, sign or magnitude at the write checkpoint.

In the present restricted family, a stronger oriented inference is available under explicit conditions:

1. The common remaining word consists of zero inputs and the frozen A is positive, so subsequent dynamics preserve the sign of every nonzero state.
2. The readout at the endpoint is v*r with v fixed positive, or an equivalent orientation already established by replayed anchors. No later gate/drive/context operation reverses this mapping.
3. The new task endpoint label s requires output sign s, and the label is reliable.

Under these conditions, endpoint correctness requires the immediate post-write state to have sign s. An oriented local sign constraint is then a valid necessary condition of this family. A positive margin m remains a chosen robustness/learning convention; binary endpoint labels alone do not imply that particular numerical value.

If the later dynamics or readout can invert sign, or if the same raw pulse has a context-dependent task meaning, imposing s*r_write>=m by fiat can teach the wrong internal representation. A pair-separation loss is more general but has its own symmetry/credit limitations. The oriented construction should therefore carry its explicit continuation/readout assumptions instead of being promoted to arbitrary neural target inference.

The necessary actual states are obtained by replaying retained raw histories through the current candidate. They are not recovered from an already aliased state and are not supplied by an external hidden-state teacher. If raw histories or the relevant new labels are unavailable, this local supervision cannot be manufactured by self-inspection.

## 4. Detached one-transition hinge: exact direction and a hold tradeoff

At an actually replayed prestate r, let b=tanh(A*r+beta*x). For a licensed target orientation s, define a local squared hinge

    ell_write = (1/2)*(m-s*b)_+^2.                       (8)

Detach r and hold s fixed in the conditional one-transition derivative. Set d=(m-s*b)_+ and q=1-b^2. Then

    partial_beta ell_write = -d*s*x*q,
    partial_A ell_write    = -d*s*r*q.                   (9)

For an opposite pulse x=s and r=-s*a,

    partial_beta ell_write = -d*q,
    partial_A ell_write    = +d*a*q.                     (10)

A joint gradient step raises beta and lowers A. Its signed write drive delta=beta-A*a changes on this frozen prestate by exactly

    delta_next = delta + eta*d*q*(1+a^2),                (11)

before projections. Therefore it improves the actual one-step signed output, since tanh is increasing. This is the derivative of a conditional auxiliary objective, not of the full endpoint loss or the whole recomputed trajectory.

The same calculation exposes a potential problem: decreasing A to make a pulse effective can destroy the hold condition (5). A write-only objective can learn a reactive state that is easy to overwrite but decays across the next long gap. Full original task replay and carry checks cannot be removed merely because the local write loss improves.

### Observed hold Jacobians yield the beta-only choice, rather than a manual mode flag

At root's request, derive the parameter responsibility from actual retained transition rows. Let theta=(A,beta,v), and let `(r_i,x_i)` be the actual predecessor/input pairs in the retained common-future hold bank. The bank is selected from the observed witness/continuation, not by a hidden true-memory state. Detach those recorded predecessors and form

    J_H[i,:] = grad_theta tanh(A*r_i+beta*x_i)
             = sech^2(A*r_i+beta*x_i)*(r_i,x_i,0).

For Euclidean parameter coordinates, put

    P_H = I-J_H^dagger*J_H,    g_write_projected=P_H*g_write. (12a)

This familiar nullspace projection satisfies `J_H*g_write_projected=0` in exact arithmetic. It removes the first-order change to the measured hold transitions caused by the **write auxiliary update**, without supplying a hand-coded beta-only flag.

For the actual zero-input hold bank, each row is `(r_i*sech^2(A*r_i),0,0)`. If at least one row is nonzero, the row space is the A axis, so

    P_H=diag(0,1,1).

The write hinge has zero v derivative; hence its projected update is exactly

    Delta A=0,    beta_next=beta+eta*d*q,
    Delta v=0,    delta_next=delta+eta*d*q.               (12b)

In this specific family, this has an exact consequence stronger than a local tangent statement: the full zero-input map `r -> tanh(A*r)` is unchanged for **every** r because A is unchanged and beta multiplies zero. Thus an already-established autonomous hold map is preserved, even though written entry states intentionally change. Root and the independent supervisor supplied/checked the Jacobian-projection interpretation after the initial support derivation; this contribution is recorded rather than claimed as isolated invention.

The conditions matter:

- If every recorded r_i is zero, J_H has rank zero and P_H=I. The bank does not constrain A; a separated observed hold trajectory is needed to infer this responsibility. Numerical saturation can similarly make a row appear zero at the stated precision.
- A nonzero-input hold bank has rows proportional to `(r_i,x_i,0)`. Rank two can leave only the v direction, making the projected write-hinge update zero; rank one can give a mixed A/beta direction. There is no guaranteed beta-only or nonzero safe write direction for a different bank.
- For this scalar affine-preactivation map, any exact nullspace step preserves the measured one-step outputs at their frozen `(r_i,x_i)`, since zero Jacobian change with nonzero sech^2 means `r_i*Delta A+x_i*Delta beta=0`. It does not generally preserve unobserved contexts or trajectories with changed predecessor states. Only the zero-input/Delta A=0 specialization supplies the all-r autonomous-map statement above.
- For a general nonlinear neural transition, the projection gives first-order preservation on the measured bank, not an exact finite-step global contract. Curvature, rank approximation, parameter rescaling/metric, changed states and actual post-update replay matter.

Most importantly, an arm combining endpoint, carry and projected-write terms has an update such as

    Delta theta=-eta*(g_task+lambda_C*g_carry
                              +lambda_W*P_H*g_write),   (12c)

before its declared clipping/projection. The first two terms remain **unprojected unless the acquisition contract explicitly says otherwise**. They can change A, v and the hold behavior. Projection protects the write auxiliary's contribution, not the whole combined update. Scalar global norm clipping preserves a projected component's direction; componentwise projection, a preconditioner or momentum applied afterward need their actual effect checked rather than being assumed to preserve the nullspace.

### Scanning, rank and execution costs of that projection

Constructing J_H requires processing the actual hold bank and differentiating its one-step transitions. For M transitions and d parameters, storing a dense Jacobian costs O(Md); a dense SVD costs O(min(M*d^2,M^2*d)), in addition to actual derivative work. A declared singular-value tolerance is a numerical rank decision, not a learned semantic certificate, and must not be tuned after a desired outcome.

The exact scalar structure permits a paid closed form: scan the actual rows and accumulate `sum_i r_i^2*sech^4(A*r_i)`. If this is positive under the declared rank/precision rule, use the rank-one projector above. This costs O(M) row evaluations and does not require a hidden SVD/controller. The bank scan, derivatives, rank check and projection still count even if code later compiles them to a constant-size scalar action. If parameters or bank states change, refreshing or lawfully reusing the projector must be specified.

This is a familiar constraint/projection mechanism, not a novelty claim. Holding the autonomous transition fixed also does not preserve every finite task output automatically. Stronger pulses change initial states and short-gap amplitudes; a shared readout can change during later learning; nonzero distractors can be overwritten unintentionally. Original and new whole-task replay remain decisive. If a previously accepted mode is frozen for older consumers, further improvement must respect that version/ownership policy rather than silently editing their state dynamics.

## 5. Why the short local path can help, and what it cannot cure

For the opposite write, a terminal beta gradient propagated through K zero-input hold steps contains the factor

    product_(j=1)^K A*(1-r_j^2).                         (13)

Near a stable attractor that factor can be tiny even when the sign is confidently wrong for the new task. Equation (9) removes that subsequent hold product from the write-learning path. The local derivative still contains q=sech^2(A*r+beta*x): saturation in the write transition itself can make it small. A local hinge bypasses one specified credit path, not every source of poor learning.

A finite observed write pair also need not reach the universal bound R=1. Training after long neutral gaps may only show prestates near +/-r_star. A beta that overwrites those states can fail after several same-sign pulses or a short gap yielding a larger opposing magnitude. Pair-level success does not certify controllability over every reachable prestate. A proposed uniform claim must justify its prestate set or pay the corresponding broader task evidence/proof.

The numerical margin is not the output-amplitude objective either. With answer v*r, getting signed r>=m ensures sign robustness in the specified band, but it need not make v*r close to a unit endpoint target. The readout/amplitude loss must still be fitted and evaluated on full episodes.

## 6. Genuine controllable memory versus hand-imposed state

The defensible interpretation of a successful future step would be:

- actual new endpoint outcomes reveal that a stored distinction must change under particular observed inputs;
- the restricted continuation/readout structure translates those outcomes into a local necessary sign constraint, with a declared margin convention;
- an update derived from actual replayed prestates changes a real write transition;
- full replay verifies that persistence, readout and earlier observed task behavior survive and that the new input genuinely controls the retained mode.

It would be hand-imposed semantic supervision if an experimenter supplies the correct state sign at every intermediate time, chooses the desired poststate from a hidden latent bit, or treats every pulse as a write before the task feedback establishes it. It would also be misleading to count the two interpretation columns in the existing evaluation JSONL as newly acquired labels.

Ordinary auxiliary supervision, continuation-based state learning and recurrent/gated learners may use the same endpoint-derived constraints and actual traces. They are constructive same-information controls. A gain over endpoint-only BPTT would identify a useful credit organization in that condition, not historical novelty, uniqueness of neural mode formation, or superiority over those controls. Conversely their ability to receive this same learning signal is not a reason to stop investigating its actual effect in the protected lineage.

## 7. Costs and what remains for root's acquisition freeze

No extra labels have been obtained here. A later process must specify which new completed episodes supply them, their task interpretation/context, and which are training versus held-out evaluation. Distinguishing last-write from first-cue requires feedback on histories where those interpretations disagree; the old single-cue pair cannot do that.

Replay to a write checkpoint costs its actual prefix length and storage/recomputation; matching the shared continuation and labels costs its actual mining work. A detached local write derivative is one transition per candidate pair, but evaluating the resulting task still costs the full old/new episodes. State/readout versions, retained witnesses, failed updates, parameter projection and maintenance costs remain real. No free controller resets a prestate to the desired answer.

The exact result of this support note is conditional and useful: the current scalar family has a feasible write-hold region; the observed mode's pulse resistance follows directly from its parameters; and an endpoint-licensed local write hinge has a short, analyzable gradient whose joint update can trade off hold. Projection against actual detached hold-transition Jacobians yields a beta-only write-auxiliary step in this particular bank/family and preserves its autonomous map. That does not protect unprojected task/carry updates or establish generic global preservation. Root can use these facts when freezing the next acquisition step. This note supplies neither that freeze nor a predeclared outcome.
