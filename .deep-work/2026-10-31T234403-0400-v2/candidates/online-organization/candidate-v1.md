# Learn where an internal computation can return

AGENT PROPOSAL / RECONSTRUCTION. A fresh mechanism problem after sealed stage007. It does not extend the association, edit-frame or operator-algebra hypotheses. No experiment was run. The construction is developed here before its main attacks.

## Question and success counterfactual

Can a neural computer learn reusable internal procedures that perform large, variably scheduled recurrent computations, then **commit a small explicit result and retire their private working state**, while preserving a usable path for much later learning?

The important limitation is not inability to differentiate a while loop. It is the cost of continually retaining or reconstructing causal history when an agent constructs longer computations, interleaves jobs and receives feedback at unpredictable times. BPTT, RTRL and checkpointing offer different tradeoffs, but the organization of the forward computation determines what information those methods must retain. The proposal is to let learning change that organization by acquiring real return boundaries, not merely selecting another estimator for the same fixed recurrent state.

If successful, acquired capability would include both what a procedure computes and a small physical interface through which future computation and credit depend on it. New computations could compose these procedures without exposing their entire private dynamics to lifetime learning state. This would connect directly to the Fundamental Question: neural learning would construct and compose mechanisms whose internal organization changes the cost of improving them later. No computability expansion or general solution of long-term credit is claimed.

## 1. Machine state and actual forward transitions

A bounded interpreter holds

    S=(L, theta, F, V, E, rho, budget).

L is a library of neural procedure graphs. theta contains their versioned parameters. F is a set/stack of active frames. Each frame has explicit input ports z, private recurrent workspace h, an input-event cursor and a declared set of readable parameter owners. V contains committed values still available to consumers. E contains derivative/decision evidence for active frames and committed live values. rho is a learned controller for call/continue/commit/widen actions. The interpreter, access discipline, resource bounds and numerical primitives are designed.

A procedure frame may execute

    h_{k+1}=F_{theta_m}(h_k,z,x_k),

with the number and order of internal steps selected from current observations and available events. At a learned COMMIT action it emits

    z_out=C_{theta_m}(h_K,z),

then **physically destroys h and its access paths**. Subsequent consumers can read z_out and public context, never the retired workspace. This is an actual change in forward computation; it is not a claim that an arbitrary old h was exactly summarized for every possible future question.

Calls and committed values have stable identities. Independent frames can be scheduled in different orders. A consumer must await its actual input values. Changing the order of dependent events changes the computation and is not declared equivalent. Dynamic scratch allocation has an external cap and a termination path.

Parameter versions are explicit. The first exact version uses finite credit episodes: parameter values are frozen while that episode's outstanding forward values are evaluated, and learned updates apply when its external feedback arrives. Calls within an episode can have unpredictable durations and arbitrary legal interleavings. This does not yet solve exact current-parameter gradients for an indefinitely running learner that changes shared weights while old values remain live; section 7 treats that difficulty honestly.

## 2. The retained learning object at a return boundary

For one completed call m, calculate the derivative of its actual returned value with respect to its input ports and the parameters it actually used:

    A_m=partial z_out/partial z_in,
    B_m=partial z_out/partial theta_used.

They are computed from the executed transitions, not predicted by a critic or supplied as an inverse. Automatic differentiation is an allowed internal operation. One can use forward sensitivities inside a call or bounded reverse differentiation/checkpointing. All derivative work and stored input/restart data are charged.

If the current live input ports have influence matrix E_in=partial z_in/partial theta, return updates it by

    E_out=A_m E_in + Scatter(B_m).                       (1)

Scatter uses the true parameter identities, summing repeated uses rather than copying parameters. Conditional on the executed discrete control path and fixed parameter version, equation (1) is the ordinary chain rule and is exact. Retire A,B and private derivative work once their contribution has been folded into the chosen live-frontier representation and no other caller needs them.

For a call with n private state coordinates, r input/output port coordinates and p_m readable parameters, local forward sensitivities have sizes n-by-r and n-by-p_m. At return they become r-by-r and r-by-p_m maps. An existing global frontier E has r-by-p entries, where p is the total parameter count. Thus long private computation need not keep a time-indexed activation tape, but it can still have substantial local n(r+p_m) sensitivity cost and global rp cost. A procedure that freely reads every parameter loses the p_m advantage. Many simultaneously live returns cost proportionally many port/sensitivity records. These are actual costs, not hidden constants.

The scientific candidate is not equation (1). It is learning forward procedures and return policies that make the physically live frontier small without losing the capabilities demanded by later experience.

## 3. Actual acquisition and learning rule

Start with a generic bounded workspace procedure and narrow but expandable return ports. The controller chooses among CONTINUE, CALL(existing procedure), COMMIT(return encoder), and a bounded representation-growth action. It is not given the correct segmentation, final answer, local cell teacher or a proof that the current return is sufficient.

For an externally evaluated episode, define the measured objective

    J = task_loss
        + lambda_M * live_credit_byte_time
        + lambda_C * actual_forward_and_derivative_work
        + lambda_L * library_description_cost.

Here byte-time includes the data/derivatives retained while waiting for delayed feedback. It does not reward throwing away evidence and pretending its credit survived. External prediction/control loss supplies the functional pressure; resource terms select among successful organizations rather than replace the scientific task.

For continuous parameters, use exact pathwise derivatives through the committed ports. For discrete controller choices a_k, accumulate score terms

    s_rho=sum_k grad_rho log pi_rho(a_k | observed state_k),
    s_theta=sum_k grad_theta log pi_rho(a_k | observed state_k).

The second term is needed if policy observations depend on theta. It can be computed using the current local sensitivities and input-port influence, with its cost counted. A learner that simply detaches those observations is using a semi-gradient, not the full expected-objective gradient. With an action-independent baseline b and a sampled finite episode cost J, the score-function contribution is (J-b)s; add the executed-path derivative where present. This gives a specified estimator rather than an unspecified claim that a scheduler learns. Its variance may be prohibitive; no cheap variance bound is asserted.

Acquire a reusable procedure as follows. The constructor proposes factoring a recurring bounded event subgraph into a library body with explicit inputs, private state and a return encoder. This is a real source-graph action. Reuse its code/parameters across later call sites. Train the body and return policy from actual subsequent task loss, comparing its complete resource/capability effect on held-out episodes. Reject a factorization that passes only its old immediate output tests but fails later consumers. All trace collection, proposal search and retraining count toward acquisition.

A learned return can be wrong in a useful diagnostic way: if later demands need information it destroyed, the next acquisition step can keep the frame open longer, widen the physical port, or change what is computed before return. It cannot recover already deleted instance information retrospectively. The library change affects future executions; old missing evidence remains missing.

## 4. One full worked process

Consider a procedure with private scalar recurrent workspace

    h_0=0,
    h_1=tanh(b x_1),
    h_2=tanh(w h_1+b x_2),
    z=v h_2.

The controller has learned to return after these two relevant events and to expose z rather than keep the frame active during an unrelated stream. This learned stopping/return choice is an assumption of the worked execution, not a demonstrated acquisition result.

At return, the exact local sensitivities are

    d_b h_1=(1-h_1^2)x_1,
    d_w h_1=0,
    d_w h_2=(1-h_2^2)h_1,
    d_b h_2=(1-h_2^2)[w(1-h_1^2)x_1+x_2].

The live value stores z and its three-parameter derivative

    E_z=(v d_w h_2, v d_b h_2, h_2).

h_0,h_1,h_2 and the input events need not remain stored if they are not used elsewhere. The workspace can execute many other jobs; no identity gradient through their unrelated activations is inserted into this value's causal path. After an arbitrary permitted delay, a consumer returns an external target y and incurs ell=(z-y)^2/2. Its exact executed-path gradient for this parameter version is

    grad_(w,b,v) ell=(z-y) E_z.

The result does not depend on how many unrelated events occurred while z waited. A consumer that instead computes q=tanh(alpha z+u) supplies its actual derivative d_ell/d_z; the same E_z composes by equation (1). Later composition thus need not reopen the discarded call.

What was acquired must be tested separately: can the learner discover this procedure/return rule from external feedback, reuse it with new delays/consumers, and keep task-relevant information? The derivative calculation is exact reconstruction, not evidence it can.

For a failure case, suppose a later consumer asks for another independent property of the private sequence. A scalar z may no longer suffice. Under fixed finite-precision port storage, arbitrary independent instance information cannot be packed into one scalar without loss. The learner must allocate more physical return state or compute a different sufficient result before destruction. Correct gradients of the surviving z do not repair a failed forward representation. This distinguishes exact credit for an executed architecture from adequacy of that architecture.

## 5. Why schedule and structure matter

In a global-clock recurrent network, irrelevant work can remain part of a large recurrent state even when an internal procedure has logically finished. Here a return destroys that part of the forward state and its future influence. The causal boundary is enforced by the interpreter, so its derivative boundary is legitimate. Learned scheduling can interleave frames without turning the entire wall-clock stream into one undifferentiated state sequence.

The prospective gain is not “delay has disappeared.” Waiting values still cost port memory and sensitivity; parameter sharing across many live calls can be expensive; score-function credit for scheduling can have long variance horizons. The gain would be acquiring an organization where useful algorithmic work has a small live interface, so both forward memory and exact learning state follow the actual causal frontier rather than all transient workspace across elapsed time.

An important prospective task class is variable-depth neural computation over interleaved event streams, with later queries composed from intermediate results and with no permission to replay arbitrary external inputs for free. The key capability outcome is learning new internal procedures/compositions under a complete memory/compute budget, including unexpected later consumers. A fixed two-step recurrence alone is not a consequential endpoint.

## 6. Strong same-information controls

The strongest control receives the SAME source graph, parameter versions, call/return actions, external observations, resource accounting and representation-growth budget. It uses ordinary mixed-mode AD/RTRL and dynamic checkpointing on the same runtime graph. Given the same learned forward program, it computes the same sensitivities and can reproduce the procedure exactly. No algorithmic differentiation separation is claimed.

Required actual rivals are:

1. A deep-transition recurrent model with narrow persistent state and large transient computation, trained by competent RTRL/mixed-mode AD. This can already realize much of the proposed organization.
2. A dynamic-program/checkpointing runtime with learned or searched segmentation and the same library-growth actions. Unknown execution length and while loops are not grounds to deny it checkpointing.
3. Columnar-constructive RTRL, and a version allowed the same learned procedure/return search, with total functional and learning capacity matched rather than forcing every method to use the same hidden width.
4. Reversible recurrent computation where applicable, charging forgotten bits, stored inputs and reconstruction work. Reversibility does not make erased information or future feedback free.
5. A generic architecture/program learner optimizing the identical task/resource objective. If it acquires the same call structure, the mechanism belongs to that shared organization rather than a special name for it.

Approximate local-gradient methods are useful practical controls but cannot be the sole strongest rival. This proposal does not rely on spiking neurons, a biological-locality claim or a new e-prop rule. Implicit differentiation/control of a stable equilibrium solves a different executed-objective problem unless the procedure actually converges to that equilibrium; it cannot silently replace a transient program's gradient. Conversely, where a learned procedure is an equilibrium solver, implicit differentiation receives that structure and is an appropriate control.

## 7. Main attacks after the constructive pass

**The strongest immediate attack:** this may be ordinary learned modular recurrent architecture plus mixed-mode AD. Learning a commit/return boundary is an actual organizational change, but that broad fact does not establish novelty or foundational value. The direct program/architecture-learning control can implement it. The candidate needs a concrete important limitation altered beyond a convenient implementation, not only a lower tape count against naive global BPTT.

**Frozen-version scope:** exact derivatives above refer to executed calls at a specified parameter version. If shared parameters change while old values wait, applying their old gradient to current weights is delayed/stale optimization. A versioned mailbox is a valid learning rule, but it is not automatically the gradient of the current network. Exact differentiation through the intervening optimizer trajectory introduces additional state/higher-order work. This v1 does not hide that cost or claim fully continuous fresh gradients for arbitrary online updates.

**Resource accounting:** an r-by-p eligibility can be more expensive than retaining a short local tape; many open frames/returns can defeat the intended gain. The controller's resource objective must compare those alternatives using actual costs, rather than counting only forward activations. Packet/controller decision scores are learning memory too.

**Representation risk:** the method intentionally changes what the future can depend on. It trades forward capacity/information for credit cost, much as known structured recurrent architectures do. If new consumers repeatedly force ports/workspaces to grow until they recover the old full state, the acquired abstraction has not solved the important bottleneck.

**Learning risk:** the resource-aware controller may learn task shortcuts, an externally imposed fixed segmentation, or high-variance exploration that is worse than architecture search. A finite witness of good segmentation does not demonstrate it can acquire one. Delayed feedback can identify a useful result without identifying a cheap way to organize the internal computation.

## Present status and decision returned to root

There is a concrete state/transition/learning law and an exact worked gradient process. The substantive candidate is **acquisition of physical causal return boundaries as part of learning a computational mechanism**. The underlying credit calculus is known; it is not a revival of the sealed parameter-coupling hypotheses.

The fragile constructive question is whether a neural program can acquire and revise such boundaries for meaningfully new composed tasks, under an online resource budget, in a way that is not exhausted by ordinary modular architecture/program learning. Primary reading currently makes that rival serious rather than disproving the possibility by name. No significance or novelty result is established; no execution has occurred.

Before an experiment, root should decide whether a task family makes the best plausible success consequential after granting the strong controls the same learned call/runtime organization. A naive fixed-unroll comparison is insufficient. This returns an actual new construction for main-level review; it is not a MAIN completion decision.
