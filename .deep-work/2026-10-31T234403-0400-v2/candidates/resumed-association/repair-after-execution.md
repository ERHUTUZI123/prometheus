# After actual acquisition execution: acquire transported updates, not Q ranking

RECONSTRUCTION / AGENT PROPOSAL. Read the actual v1a/pilot_v1a.py, v1a/summary.json, v1a/raw.jsonl, recovery contract and original audit provenance. No new code, seeds, fitting or experiments were executed for this artifact. This is a constructive scientific repair returned to root, not a campaign-completion decision.

**Latest MAIN steering:** the primary constructive recommendation is now section 6: learn the mapping from an observed source behavioral change to a recipient behavioral change, using actual paired external-learning probes. Sections 1–5 retain the joint-response construction and its strong attack as a possible later extension; joint action selection is not substituted for acquiring this mapping.

## Judgment from the executed evidence

The pilot has established a real finite-effect acquisition path with externally observed program labels, actual response transport and independent-position evaluation. It has not established a new learning principle. Q is an amortized contextual optimizer over a supplied single-recipient action set. The coordinator's aggregate of the actual summaries is consistent with the central limitation: exact cached replay has lower final loss than Q (about 0.0743 versus 0.0877), while local-only is about 0.1371 with substantially less transfer work. Equal labels do not establish equal computation or end-to-end efficiency after acquisition.

The actual seed3101/world0 record is especially instructive: Q's accumulated immediate sampled gains are about 1.718 versus replay's 1.240, but final untouched loss is about 0.1812 versus replay's 0.0819. These gains come from different adaptive checkpoints/batches and do not telescope into the same final objective. They demonstrate why a larger immediate utility total is not itself evidence of better later capability. More Q capacity or seeds would not repair that interpretation.

There remains a potentially important mechanism question, but it requires a different object: **experience can create a new callable coordinated update, thereby changing which collections of internal computations are treated as one unit of adaptation**. A predictor that only ranks a fixed menu does not demonstrate this. Finite joint packets are one concrete construction route. Their elementary algebra is already solved by strong simple alternatives, so a two-cell plateau escape alone would not justify the scientific ambition.

## 1. A structural gap the present pilot cannot contain

The executed consumer is

    Y=tanh(z),   z=a+0.7 s b+0.3ab,   s in {-1,+1},

with a,b themselves tanh-cell outputs. Therefore, at every finite cell state,

    partial z / partial a = 1+0.3b > 0.7,
    |partial z / partial b| = |0.7s+0.3a| > 0.4,
    partial Y / partial z > 0.

Neither cell is individually silent at the consumer interface. Parameter gradients can be small or cancel over examples, but the consumer has no exact jointly-only channel of the proposed kind. Thus adding a joint-action head to this same generator would not by itself test the proposed gap. A joint-silent study changes the scientific representation/claim and must be justified accordingly; it is not an excuse to keep tuning the old ranking pilot.

## 2. Concrete new persistent object: a learned joint response procedure

Retain the lawful parts of the pilot: external program labels, independent parameter owners, public semantic probes, actual distillation and paired future audits. Add a bounded archive E of effect templates obtained from real previous source experiences. An archive entry stores the retained response representation, its source/version/context and its acquisition/storage cost. It may store a small executable source-response routine or only a declared interpolated probe packet. Deleted source states do not become available again.

The new acquired object is

    R=(role pattern, owner-binding requirements, input packet types,
       a conditional joint proposal program, validity/failure history).

It is a program that takes source effects and current externally derived history, emits a joint staged edit to several owners, probes the realized responses and either commits under its finite contract or reports failure. It may call earlier joint procedures. It is not only a table assigning high value to an already supplied pair. The acquisition loop must construct a previously unavailable short program from successful coordinated traces and retain it for new bindings/compositions.

The initial language supplies ordinary raw edits, single-response distillation, bounded tuple construction and numerical primitives. It does not supply the correct source pair, recipient hyperedge, amplitude signs, local targets or success witness for a new program. A full-state solver receives the same graph, archive and primitives.

### A concrete construction/update law

1. From a paid external program history, choose a small candidate owner set B and source-template tuple using the current proposer plus explicitly randomized exploration. Choosing every correct B/template tuple for free is forbidden. Record actual proposal probabilities where used.
2. At a common checkpoint, construct each constituent distillation and its joint staged version. Resolve parameter owners before acting. Measure actual cell responses, not requested packet amplitudes. Check all protected callers affected by the union write set.
3. Obtain an interaction signature by evaluating known forward programs on public probes. For two owners, the mixed finite response is

       I_jk = Y(after both)-Y(after j)-Y(after k)+Y(before).  (1)

   These are unlabeled executions, not four new target labels. If source states are only retained on public probes, this signature is also only a public-probe claim. Arbitrary off-probe response requires retained executable state or a new approximation boundary.
4. Use fresh external program observations to measure the **joint** loss effect and compare a zero-transfer branch, each constituent alone and the joint proposal on the same data. Count the extra forward/proposal work and reuse one paid label lawfully. A nonzero I identifies interaction, not usefulness; only the externally labeled effect gives direction/utility.
5. If a coordinated trace works, abstract it into a conditional role-bound procedure, train its argument/pair selection on actual episodes, and test it on new owner bindings and program contexts. Retain the procedure only when future acquisition succeeds more cheaply after all acquisition/check/storage costs. Failed bindings update its conditions or trigger a newly acquired correction, rather than being silently called successful transfer.
6. Later constructed routines can call that procedure during learning. The library's action grammar has changed. Test later capability acquisition and retained performance separately; an immediate decrease in one joint audit loss is not the final acquisition outcome.

This construction rule is executable without local teacher targets. It is still a proposal: no such library was acquired by the completed Q pilot, and no generic portability result has been proved. The alias/prefix conditions in ownership-repair-v2.md apply unchanged.

## 3. Exact worked law, including its strongest simple solution

Consider P(v_1,v_2)=v_1v_2 at a checkpoint where both recipient outputs are zero. Let u(x),v(x) be two actually retained source-response templates. They are candidates from E, not oracle targets supplied for the recipients. A trial distills recipient changes d_1=a u and d_2=b v. In the exact-realization illustration,

    Y_after = ab u(x)v(x),
    Y_after_single_1 = Y_after_single_2 = 0,
    I_12 = ab u(x)v(x).

With external residual e=Y_before-y=-y and squared program loss, the finite joint improvement on the paid batch is

    U(a,b)=-ab mean[e u v] -(a^2 b^2/2) mean[(u v)^2].   (2)

The external history needed for this candidate interaction is only

    m_12=mean[e u v],    H_12=mean[(u v)^2].              (3)

No target for either recipient cell was observed. Single-cell histories proportional to e times the consumer Jacobian are zero at this checkpoint and cannot recover equation (3). The joint-effect channel therefore contributes genuinely missing information relative to the one-recipient interface. That is a precise interface gap, not a novelty claim.

**Strongest simple alternative:** set t=ab and C(x)=u(x)v(x). This is ordinary one-dimensional least squares on the candidate feature C. If H_12>0, the unconstrained optimum is

    t_star=-m_12/H_12.

A graph-aware block solver can choose a,b with this product, obey response-amplitude constraints, distill them and check the actual result. It need not learn Q or estimate a high-order derivative. A direct constructed feature C with a trainable coefficient t is an equally important representation control if permitted the same function-library/parameter budget. This elementary example therefore does NOT establish a fundamental advantage over optimization or feature/program composition.

For a current nonzero product state, the exact finite output change is

    D = a v_2 u + b v_1 v + ab u v.

The external loss is a small nonlinear least-squares problem in a,b. A second-order method, graph-aware block optimizer or finite candidate replay can solve it using the same data. With an outer tanh consumer, evaluate the finite tanh response directly; the one-dimensional closed form no longer automatically applies, but direct low-dimensional numerical optimization is still the proper simple rival.

The exact formulas assume the requested response shapes are realized. Actual neural distillation introduces residuals and saturation; evaluate D from actual recipient outputs, as in the executed pilot. A source template that is useful in one consumer may be harmful to another protected caller. Shared owners, protected behavior and failure handling remain physical constraints, not annotations.

## 4. What the repair must actually acquire

The elementary product law solves a candidate pair **after it has been chosen**. It does not select useful source features from a growing archive, determine which owners should co-adapt in a new composed program, compress that search into a reusable procedure, or show that the resulting procedure facilitates later learning.

That is the remaining candidate mechanism: acquire reusable *coordination programs* that choose and bind complementary source effects, rather than repeatedly solving an unrelated block-search problem. For example, a learned procedure could identify a familiar interaction role from current program structure and external response history, retrieve complementary effect types from E, fit their joint coefficient/sign using paid evidence, and call earlier corrections to protect shared callers. New labels still decide whether this association applies. Structural similarity alone is not a success oracle.

The potential important change is finite-budget accessibility of composed computations: acquiring component capabilities would also acquire a way to coordinate their future modifications. A newly constructed computation could inherit a usable adaptation procedure instead of encountering an unstructured combinatorial search over all parameter/source tuples. This makes the connection to the Fundamental Question explicit. It changes search organization, not computability.

There are two exact no-shortcut constraints:

* With r archived source effects, even pairs create O(r^2) candidate interaction features before considering recipient sets. A claimed cheap discovery must account for retrieval/search and the cost of learning the selector. Giving the learned method the correct pair while forcing the solver to search all pairs is invalid.
* Zero-initialized product cells are a convenient diagnostic but can be an artificial trap. Ordinary nonzero initialization, one factor initialized nonzero at safe growth, negative-curvature/block optimization or directly adding a product feature may remove it. A result solely defeating a deliberately silent initialization would be a local optimizer result, not the project's consequential mechanism.

Thus a pair/triple plateau example should be only a semantics/negative-control check. A serious success requires new program structures and changing relevant coordination sets, with a source archive learned from previous actual tasks and a fair resource budget. It should still matter when competent initialization and block solvers are present.

## 5. Closest actual organizational rival, not a weak comparator

The direct rival is a contextual learned block optimizer or program synthesizer with the same program graph, effect archive, external histories, joint action grammar, memory and ability to retain useful procedures. It can implement this construction. There is no broad representational separation merely because our object is called an association or repair interface.

An ordinary factor-/source-aware solver should first use equation (2) or its finite numerical counterpart, then retain successful block strategies as parameterized options/macros. A program-library learner may build the same joint procedure. Compare against that system, not only independent SGD or the old Q. If both obtain the same reusable organization, the result belongs to the shared acquired program content; no separate new principle has been established by renaming it.

The strongest causal test would freeze forward capability, archive E, observed data, budgets and bindings. Allow or remove only execution of the acquired coordination procedures, expanding them into charged primitive operations where needed. Measure later task acquisition on held-out program compositions. Then give the same retained procedures to the strong solver and finally let it learn its own from identical episodes. These distinguish useful acquired content from a privileged interface, and acquired-content reuse from ordinary history/compute advantages.

The objective also needs repair from the pilot: promotion cannot use summed immediate gains as its entire criterion. Use a finite subsequent learning sequence from a documented common checkpoint, specify success/retention at its end and charge all intermediate failed attempts. Immediate joint effect can supply training evidence, while later capability provides the scientific claim. No future outcome labels enter the proposal before they are actually observed.

## 6. Revised primary law: conditional finite-update transport with an observability channel

The old pilot learns whether to select a supplied scalar multiple of a source packet. Replace that head by a conditional map that **constructs a new recipient update**:

    T_eta(d_i,h_i,h_j,c) -> distribution over recipient response d_j.

d_i is the measured source before/after behavioral response, h_i,h_j are only observed histories and current response/consumer evidence, and c contains observed context/learning-step metadata. Position IDs and hidden teacher/group variables remain absent. Output uncertainty is meaningful because the same observable source change can correspond to different recipient changes.

### Honest supervision from real external learning probes

At an acquisition episode, obtain an ordinary externally labeled source batch and an ordinary externally labeled recipient batch under the declared sparse acquisition schedule. On cloned common checkpoints, let each position undergo its specified native program-learning update and measure its actual response change on public probes. Record

    (pre-update h_i,h_j,c, actual d_i, actual d_j,
     both external-label/gradient/forward costs).

The d_j supervision is an effect of real learning from a whole-program label, not a hidden local cell target or a supplied inverse. It is not free: this episode contains recipient labels and recipient native learning. Training can use randomly sampled co-occurring positions; it must NOT select “corresponding” pairs using hidden World.group or teacher coefficients. If a common external event/context really is observed, it may be an input, but it cannot secretly encode the correct relation. Unrelated sampled pairs remain noise/negative evidence rather than being deleted as inconvenient.

The learned map predicts d_j **before the next recipient learning labels are observed**. Its usefulness is judged by actual recipient program performance on fresh data after realizing the prediction, not merely by matching a native SGD trajectory. Native SGD is a paid source of training effects, not an optimality oracle. A source-free predictor T(h_j,c) must be tested: otherwise ordinary recipient-history prediction can be mistaken for source-to-recipient transfer.

### One concrete chart implementation, with its exact limit

An initial implementation can posit a low-dimensional latent update z and context-conditioned behavioral charts

    d_i=A_i(h_i,c) z,       d_j=A_j(h_j,c) z.

Learn A and an encoder jointly from the actual paired changes. New positions obtain charts from their observable histories/calibration records, not a new lookup-table row. A global latent-coordinate change is a harmless gauge if all charts change together; independently learned marginal charts have no automatically shared orientation. Paired actual episodes are what align them.

For fixed linear charts, a deterministic source-to-recipient map exists for all z **if and only if**

    ker(A_i) subset ker(A_j).                              (4)

Necessity: latent updates differing by v in ker(A_i) have the same source packet; their recipient packets differ by A_j v. Sufficiency: A_j factors through the image of A_i. The suggested map R_ji=A_j A_i^+ is valid on source packets precisely when A_j annihilates the source-nullspace component, equivalently A_j(I-A_i^+A_i)=0. Pseudoinversion does not supply missing latent information.

Composition also needs a condition. In general,

    R_kj R_ji=A_k A_j^+ A_j A_i^+

need not equal A_k A_i^+. A rank-deficient intermediate can discard a distinction relevant to k. Full column rank charts in a common latent coordinate are a sufficient exact condition; otherwise carry the original latent posterior/evidence through the chain instead of treating the intermediate response alone as sufficient. A cycle-consistency loss can encourage fits but cannot make an impossible factorization valid.

### Repair missing information constructively

Use an encoder/posterior q_eta(z | d_i,h_i,c), not a single unqualified pseudoinverse. Predict the target effect and its uncertainty from this posterior. For the diagnostic Gaussian linear case with isotropic latent prior and noiseless source observations, its unidentified contribution to recipient covariance is

    A_j (I-A_i^+A_i) A_j^T.

That is uncertainty, not a request to hallucinate a target update. When it is large, acquire an **additional source behavioral probe coordinate** using the actual retained source before/after snapshots, paying the forward cost, and update the posterior. Choose probes whose predicted response distinguishes latent directions relevant to the recipient. If snapshots were deleted, this information cannot be recovered. If a latent direction changes no source behavior at all, further source probes cannot reveal it; a new external observation/context is required or the policy must abstain/retain uncertainty.

This adds a concrete information-interface action: experience can teach which extra distinctions must travel with an update for a later recipient, rather than assuming a fixed packet is sufficient forever. It is not a proof that useful charts stay low-dimensional or that active probing is cheaper than direct recipient learning. A same-information baseline receives every added probe.

### Finite nonlinear version

A fixed linear chart should be a diagnostic model, not silently the final mechanism. Use a conditional finite decoder D_eta(h_j,c,z) and encode the actual source change through q_eta. In the present fixed-feature cells, a concrete realizable decoder can emit a coefficient update b_j in the canonical phi basis, giving

    D_eta(h_j,z)(P)=tanh(Phi(P)theta_j+Phi(P)b_j(h_j,z))
                        -tanh(Phi(P)theta_j).

This produces finite state-dependent behavioral changes and avoids proposing impossible tanh outputs at individual probes. The phi coordinates have a declared common meaning; this does not align arbitrary hidden neural parameters for free. For general modules, output a response packet and use bounded distillation, measuring the realization error as before.

Train on paired actual effects with predictive reconstruction, plus fresh external outcome probes:

    L = E_pairs E_{z~q}[ ||D_eta(h_j,z)-d_j||^2 ]
        + lambda E_paid_fresh[ L_program(after predicted update) ]
        + declared cost/complexity terms.

Recipient fresh labels in the second term are obtained by the sparse paid audit schedule. They are not visible when the tested update is proposed. Train the reverse direction on legitimately paired records if useful, but do not force a deterministic inverse where the histories show ambiguity. Later prediction errors update the shared map/history; this is actual relation revision, not just changing the acceptance threshold of a fixed source packet.

### Worked transfer/ambiguity process

For a transparent chart calculation, suppose actual paired learning episodes have established A_i=diag(1,2) and A_j=[[0,1],[-1,0]] in a shared two-coordinate update model. A future source response (0.2,0.6) implies latent update (0.2,0.3) and predicts recipient response (0.3,-0.2). The relation constructs a rotated/rescaled update, not one amplitude of the source packet. A new recipient position must obtain its chart from its paid observable calibration/history; these matrices are not supplied as hidden group labels. The finite decoder realizes the predicted response and a later external audit evaluates it.

Now change the source chart to [1,0] and recipient chart to [0,1]. Latent updates (1,1) and (1,-1) yield the same source response but opposite recipient updates. A deterministic relation cannot solve both. An additional source response coordinate [0,1], if genuinely observable from its retained before/after computation, resolves the ambiguity. If no such source behavior exists, querying more of the same source packet does not. This exact countermodel states what information the proposed extra channel must obtain.

These chart examples explain the algorithm and its admission conditions. They are not a new toy endpoint or evidence that nonlinear neural learning is governed by a shared low-rank latent update.

### Alignment scrambling that preserves the local update geometry

The supervisor's proposed orthonormal-frame control has a clean exact form. Let each local observed response frame E_j have r orthonormal columns and let O_j be a learned r-by-r orthogonal alignment. Put A_j=E_j O_j. Then

    R_ji=E_j O_j O_i^T E_i^T,
    R_jj=E_j E_j^T,
    R_kj R_ji=R_ki.

The last equality follows from E_j^T E_j=I and O_j^T O_j=I. Independently replacing the O_j by randomized orthogonal matrices preserves each local projector, rank, singular values and within-frame update norms while disrupting relative alignment. This is a specific causal intervention rather than changing local basis capacity or adding noise to raw weights.

Local unequal scales can be retained: A_j=E_j S_j O_j with invertible positive-definite S_j. Independent O scrambling then preserves A_j A_j^T=E_j S_j^2 E_j^T, hence the entire local latent-update covariance and singular spectrum, while changing cross-position transport. It tests whether learned relative orientation matters conditional on the local geometry. Do not substitute a norm-changing or out-of-scope intervention for this declared control.

These identities require the same retained latent dimension and full column rank. If an intermediate chart loses rank, projection through it can erase information and the composition identity no longer follows. Equation (4) and the posterior/active-probe repair still apply. An r-dimensional observed chart can also omit genuine update directions; the control is exact for the stated retained chart model, not a proof of global sufficiency.

### Strongest simple alternative and the one matched test

The linear version is latent-factor multi-output regression of actual learning effects. A contextual learned optimizer can implement the nonlinear version. Both are essential actual controls with the SAME paired episodes, calibration, source probes, recipient labels and memory. Give a direct conditional regressor the source packet and histories too; withholding those makes the comparison meaningless. Any efficiency claim must include the cost of obtaining paired recipient updates and active source probes.

One useful matched test freezes a stream of real paired external-learning episodes, trains the transport map and its simple regression/optimizer controls under the same resource envelope, then predicts updates on genuinely new positions/events before revealing recipient labels. A source-free model controls ordinary history prediction. A **relation-scrambled** model uses the same training marginals/history and compute but shuffles source/recipient correspondence within declared context/magnitude bins. Scrambling arbitrary decoder charts at deployment can create a state mismatch rather than isolate learned association; shuffling the observed pairing is the cleaner initial control.

Score actual fresh program learning/retention after the generated update, including failures; also report packet prediction error, realizability and probe/acquisition cost. A benefit over the scrambled/source-free controls would establish use of learned cross-position update information in this family. It would not establish novelty over the same-information conditional learned optimizer. If low-rank regression or the generic optimizer reproduces the whole behavior, report that shared mechanism rather than claiming a separate principle.

**Residual scientific judgment after this constructive pass:** I do not currently see a material distinction between the best precise conditional-transport variant above and Bayesian latent-factor update regression / contextual meta-learning with active observation. The observability and scrambling laws make its claims more honest and testable; they do not turn it into a new organization of computation. Nonlinear decoders, revisions from feedback and new-position generalization remain within that operational description. There is no substantiated foundational residual to protect because the charts are interpretable. A matched test could establish learned relative alignment as an intermediate result, but success would not by itself justify making this the main principle. Do not execute a cleaner version merely to sustain this branch.

## Decision returned to root (PIVOT recommendation, not MAIN completion)

Keep the completed pilot as evidence that the lawful source-effect/history/route pipeline runs, plus a negative boundary on its stronger interpretation. Do not spend the next allocation improving its MLP ranking.

The strongest constructed candidate is **acquired conditional transport of actual behavioral learning changes**, with paired external-learning supervision and an explicit observability/uncertainty channel. It is a concrete and bounded learning construction, but the same-information latent-factor/contextual optimizer implements its mechanism and state directly. The joint-response construction remains a possible extension when a real composed mechanism requires coordinated updates; its elementary effect is already covered by the explicit least-squares/feature alternative above. Neither construction currently supplies a meaningful remaining distinction at the project's ambition.

Recommend pivoting the primary principle search rather than executing the cleaner chart/scrambling study merely because it is now well specified. Preserve the actual Q results, the strongest transport construction, and the exact kernel/cycle/information limits as honest artifacts. The next main action belongs to the independent fresh construction directed by root. This is a branch-level scientific judgment, not a claim that learning update relations is useless or that the Fundamental Question is resolved. No further execution was performed; the MAIN campaign continues.
