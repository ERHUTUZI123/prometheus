# Form return ports from actual future credit

AGENT PROPOSAL / RECONSTRUCTION. New v2 artifact; candidate-v1 is unchanged. No experiment or neural acquisition result. CoBRAS, HM-LSTM and ACT are actual constructive precedents, not merely names used to close a proposal.

## Judgment and the stronger mechanism

RETURN plus private-state reset is already present in learned multiscale recurrent models. Exact derivative folding is ordinary AD. The stronger candidate is a **formation rule**: use paid episodes of actual later consumer demand to create or revise a procedure's return code, port width, parameter read set and return time, explicitly reducing the amount of learning state that must remain live.

The acquired object is a computational interface chosen partly for its subsequent learning consequences. It is not just a hidden-state autoencoder, a halting gate, a future-gradient predictor or a low-rank matrix. The proposed pressure is actual task performance together with live causal-credit cost. Sampled future adjoints supply a targeted constructive signal when an old interface is inadequate. An implementation could be an ordinary modular architecture learner; giving that learner this same new formation rule would reproduce the shared mechanism, not show that the rule has no content. The scientifically relevant comparison is with its actual existing formation objective, plus strong adjoint-aware/model-reduction alternatives.

If this works beyond a convenient example, an agent could acquire large internal computations whose small returned objects support both new downstream computation and timely later learning. Its retained learning state could follow the important future interfaces rather than the full private state or elapsed execution time. This changes a potentially important constraint on constructing and improving larger mechanisms. Whether the acquired formation rule yields that effect remains OPEN.

## 1. State and the object that changes

Keep v1's actual call frames, immutable parameter versions during a finite credit episode, explicit parameter owners and enforceable private-state access. A library entry additionally stores

    M=(body G_theta, boundary policy rho, return encoder C_phi,
       consumer adapters D_psi, declared parameter read sets,
       probation evidence, production version).

A production call computes private h, emits z=C_phi(h), then retires h. Consumers receive z and their own observed context. They cannot read h through an uncounted side channel. The interface may grow or be refactored for future calls; deleting one old instance's state is irreversible.

Unlike v1's generic resource-aware controller, the following rule specifies **what evidence proposes which new port computation**. The return encoder is executable neural/code state, not a collection of unrelated cotangent vectors assigned separately to each example.

## 2. Paid probation: actual future gradients enter only after they exist

At a candidate boundary, a bounded sampler can choose PROBATION_RETURN instead of destructive production return. Before seeing future consumers, it records the decision, current code/parameter version and candidate port. It temporarily retains the original h and the necessary original trace or restart/input data within a declared cap. This is real extra memory/work, charged to acquisition. It runs the old full-state continuation and a candidate port continuation on the same admissible subsequent observations.

When a real later consumer executes and its external outcome arrives, the probation record can now contain:

    h, actual consumer context c, actual external target/outcome,
    full and port outputs/losses, their consumed/released timestamps,
    g = derivative of an actual full-consumer scalar output/loss wrt h,
    lambda = derivative of the corresponding port-consumer scalar wrt z.

Output cotangents can be unit seeds or paid random projections of the **actually executed** output, so useful credit evidence need not vanish merely because the current task loss happens to be zero. External outcome loss remains necessary to reject a full model's confidently wrong behavior. These probes are not gradients from unexecuted arbitrary future tasks. Each VJP requires a retained/reconstructed actual consumer trace and its compute is charged.

The initial setting uses exogenous event streams/sequence prediction, so both continuations can lawfully see the same later data. In an interactive environment where a port changes actions and hence future observations, a common observational trace is not automatically a valid counterfactual. That setting needs paid replay/simulation with a specified coupling, or a weaker claim.

If a probation cap expires before its consumer/outcome, preserve it as censored/unverified evidence or terminate the trial. Do not mark a port sufficient because no failure has yet arrived. A production episode that already erased h cannot later obtain this full-state cotangent by assumption. New failures can improve later calls but cannot restore the old instance.

## 3. Formation signal: useful future credit, not activation variance alone

Let S_h=partial h/partial theta_P for the true producer parameter support P. For an executed scalar consumer, the original producer gradient is S_h^T g. A candidate port with Jacobian J_C=partial C_phi/partial h and an actual learned consumer adapter has producer gradient S_h^T J_C^T lambda. Define the local discrepancy

    r_credit = S_h^T [g-J_C^T lambda].                    (1)

Compute it by VJPs of the retained probation trace; a full dense S_h need not be materialized if the runtime can supply these products. The necessary original trace/recomputation still costs resources. The norm is relative to the specified parameterization/optimizer, not a coordinate-free physical invariant.

Equation (1) distinguishes two failure types. A port can reproduce sampled forward values yet suppress a producer direction needed for later adaptation. Conversely, a discrepancy in a private-state cotangent can be irrelevant to the currently reachable producer parameter directions. Actual output/loss error and parameter-weighted credit error should both enter the decision; neither substitutes for the other.

For diagnostic proposal selection one may solve for the best possible lambda in the port row span. A large minimum residual rejects that local candidate span. A small one is only a necessary local opportunity: it does not show the actual consumer adapter can compute that lambda, or that the field integrates into a globally valid return program. The training/promotion loss uses lambda from the actual executable adapter, rather than unconstrained per-example lambda witnesses.

### One operative update objective

On collected, versioned probation records train the encoder/body/adapters by

    L_form = L_external_future
             + alpha ||Y_port-Y_full||^2
             + beta ||r_credit||^2
             + gamma K_live_credit
             + delta K_compute
             + eta K_code/probation.

The reference-output term is optional self-distillation from the agent's existing continuation; it is not an external local teacher. External targets decide whether preserving that continuation is useful. The credit term is a proposal/training signal, not a certificate of arbitrary future behavior. Differentiating it can require second derivatives or finite-difference alternatives; this expense belongs in acquisition cost.

K_live_credit is measured from the chosen actual executor. A useful explicit proxy is the time integral over live returned values v of

    width(v) * |parameter_support(v)| * bytes_per_entry,

plus value storage, active/private derivative state, version metadata and decision-score traces. Parameter support follows true ancestor/read identities; it is not guessed from small gradients on one example. The executor should allow sparse/factored derivative representations and charge their actual storage/work. A dense-matrix proxy alone is not a lower bound: ordinary AD can exploit source factorization and make a seemingly dense interface cheaper.

Lifetime ends at an actual consumption/release event, not an oracle prediction that the value will never be needed again. Resource pressure may include per-event latency/deadline constraints as well as total byte-time. This matters when a long recomputation at delayed feedback is unaffordable even if its lifetime total FLOPs is acceptable.

## 4. Concrete formation actions after a failure

The constructor proposes one bounded action from actual residual/outcome records:

1. **Learn a new return coordinate.** Append b_omega(h) or replace an inadequate coordinate. Train it to reduce the actual consumer-output/loss and credit residual across retained examples. Its coefficient in each future consumer is computed by the executable context-conditioned adapter, not independently chosen per trace. A small neural function or a typed composition of existing arithmetic/neural routines is allowed.
2. **Move the boundary.** If no compact coordinate is learned under budget, keep the frame active longer or return before an expensive cross-dependency is introduced, provided actual consumers still receive their required results. The event policy learns this choice from measured future cost/outcomes.
3. **Extract/refactor a procedure.** Factor the producer-plus-encoder computation into a new callable routine and record its real parameter read set. Exact source slicing can eliminate truly dead instructions. A learned smaller implementation can instead be proposed from the agent's own generated examples and then retrained/tested on external tasks. Such distillation changes the implementation/parameters and does not preserve the old gradient by fiat. Its acquisition, retained old code and any still-shared parameters remain charged.
4. **Keep the old interface.** Failed construction is a legitimate outcome. Do not retire state merely to satisfy a memory target.

Candidate selection is bounded; running every possible port/boundary/refactor for free is prohibited. Promote a candidate only after new probation episodes with independent states and actual held-out consumer combinations satisfy the declared task/credit tolerances and complete resource objective. Then production uses the learned return code without a full shadow. Continue occasional paid probation so genuinely new consumer demand can revise future interfaces.

CoBRAS is a suitable initializer for a linear or kernel port: form state and actually sampled future-gradient snapshots and compute its balanced projection. Its future-gradient sensitivity and variance criterion are existing mechanisms. The proposed formation rule adds event/program-boundary placement, actual parameter-read/credit-lifetime accounting, executable coordinate growth/refactoring and recurring probation on changed consumers. It cannot claim that gradient-based coordinate selection itself is new.

## 5. Worked acquisition process, not another chain-rule check

Suppose a producer has N private recurrent state coordinates h, all influenced by a shared producer parameter set P. Initially its consumer family asks only for h_1, so the acquired return is C_0(h)=h_1. The rest of the private workspace is retired in production.

A newly encountered consumer family depends on

    q(h)=h_1 h_2+h_3,

for example emitting q or tanh(q), while the old h_1 consumers remain required. This formula specifies the worked functional example; it is not supplied as the learner's correct port. In a meaningful acquisition task, it must be discovered from actual neural continuations/probation evidence rather than being handed to an exact compiler as already-factored source code. If the source graph literally exposes q, static demand analysis/hoisting is a strong simple solution and must be included.

During a paid probation call the full continuation still has h. For an actual q output its measured cotangent is

    g_q=(h_2,h_1,1,0,...,0).

The old port has J_C0=(1,0,...,0), so no scalar adapter cotangent can reproduce the h_3 component. The counterexample contains a new useful credit direction; it does not tell the learner that the right new scalar is q. A CoBRAS/gradient initializer can identify a useful small subspace, and a trained/synthesized nonlinear coordinate can propose b(h)=h_1h_2+h_3. The constructor then tests the executable proposal on old and new consumer episodes.

A successful learned interface is

    C_1(h)=(h_1, q(h)).

The old adapter reads coordinate 1; the new adapter reads coordinate 2 or computes tanh of it. Their actual cotangents lie in the rows of J_C1. For example, at (h_1,h_2,h_3)=(0.2,0.5,-0.1), q=0 but g_q=(0.5,0.2,1). Even an exactly matched zero current output can conceal a lost future learning channel; an actual output VJP exposes it without inventing another task's future gradient.

The returned producer sensitivities are

    E_1=partial h_1/partial theta_P,
    E_2=h_2 partial h_1/partial theta_P
        +h_1 partial h_2/partial theta_P
        +partial h_3/partial theta_P.

After probation and acquisition costs have been paid, a production instance holds two value/credit rows while waiting for later consumers, rather than N full-state rows in a dense forward-sensitivity executor. New compositions of the two actual returned functions can reuse the interface and give their real cotangents to these rows. This is a candidate acquisition/retention/composition process: preserving old output, adding a newly required internal function and changing future learning-state cost.

At a generic state the output cotangents of h_1 and q are linearly independent. Any differentiable port exactly supporting both there must have width at least two, by the chain rule rank inequality. If the producer sensitivity is full row rank on these directions, the same local obstruction appears in producer-gradient space. This is a scoped local statement, not a global minimum-code theorem.

If later demand independently asks for h_2, C_1 may be insufficient. An actual probation record can justify another coordinate/refactor for later calls. Old production instances whose h_2 was destroyed can fail that query; the system reports the failure. It cannot obtain h_2 from the sensitivity rows or invent a previously erased state.

The full linear-gradient example does not prove the neural learner discovers q, that its discovered coordinate integrates globally, or that a broad program family admits a small port. Those are the substantial constructive uncertainties.

## 6. Important scaling prospect and its accounting

Consider many interleaved computations with large transient workspace, long waits for externally determined consumer contexts and a strict budget for responding/learning when feedback arrives. A useful procedure may need N state variables to compute its result but only r returned functions to support all encountered later consumers, with r much smaller than N. If the relevant derivative state is dense over p producer parameters, a forward-sensitivity representation waiting across B instances has an Np-versus-rp factor per instance. A learned small-parameter refactor may also reduce p, but that is an additional acquired implementation, not an automatic consequence of projection.

These are costs of specified executors, not universal complexity lower bounds. Reverse AD can retain inputs/checkpoints and recompute later; a compiler can preaccumulate a small output Jacobian before discarding its local tape. A learned program must be compared against these real alternatives on memory, total work AND per-event latency. The opportunity is to learn which output functions and boundaries make such preaccumulation worthwhile, instead of fixing a large recurrent state or a fixed summary width/layer hierarchy in advance.

Amortization must be explicit: probation shadows may cost more than their later savings. Report acquisition work, peak acquisition memory, saved production byte-time/latency and the number/distribution of subsequent uses. If new consumers arrive so often that the learner must keep full shadows indefinitely, or if port width/read sets grow back to full state, this proposed organization has not changed the important limitation.

## 7. Actual precedents and causal comparisons

HM-LSTM already learns boundaries and ejects a summary before hard reset. ACT already trades task error against adaptive computation cost. CCN already chooses recurrent organization for efficient exact online gradients. CoBRAS/goal-oriented reduction already uses future output sensitivities/adjoints to identify useful reduced coordinates. These are constituents and direct comparison methods, not novelty claims.

The candidate's specific new pressure is **forming procedural interfaces from observed future credit while charging their live parameter-supported learning state**. Test it constructively against:

* Actual HM-LSTM/ACT-style boundary learning with task/forward-compute costs, given matched architecture/action opportunities and a fair complete budget.
* The same program/port grammar and task/resource objective, trained without the cotangent-residual formation step; it can spend saved VJP/higher-order work on its own learning/search.
* Fixed or ordinarily learned boundaries with CoBRAS/goal-oriented port initialization and reduced-model learning. This isolates whether procedural boundary/readset formation adds value beyond known adjoint-aware reduction.
* Source-aware compiler liveness, demand analysis, Jacobian preaccumulation and dynamic checkpointing. It receives the same actual code, future-consumer observations and replay privileges.
* A modular architecture learner supplied this very formation rule as a replication/shared-mechanism control. Matching there attributes an effect to the rule rather than to a special runtime name. It is not by itself a scientific rejection of an otherwise consequential and distinct formation rule.

Useful ablations remove actual parameter-support/lifetime cost, replace future cotangents by activation reconstruction at equal work, or remove recurring probation after new consumers arrive. Keep source information, task labels, parameter precision, storage rules and total resources explicit. A win only against naive unrolled BPTT or a deliberately undersized return is insufficient.

The central acquisition outcome must concern learned new procedure/consumer compositions and retained later trainability under memory/latency bounds. It must not be reduced to confirming equation (1), PCA compression, or extracting an explicitly supplied symbolic q. Closest-work coverage still needs completion before novelty claims; no source presently establishes this pressure is new.

## Status returned to root

There is now one operative formation rule, with state, probation, actual future data acquisition, executable port updates, readset/refactor actions and a complete worked learning process. Its strongest prospect is an acquired *interface for future learning*, not RETURN/reset itself.

The most fragile step is obtaining a compact executable port/readset that transfers to genuinely new consumer compositions at an acquisition cost worth paying. A local cotangent span can guide that construction but cannot replace global program construction and actual outcomes. The remaining versioned-parameter/continuous-online limits in v1 remain in force. This v2 is a bold explicit hypothesis for main-level selection; no experiment, positive neural result or MAIN completion is claimed.
