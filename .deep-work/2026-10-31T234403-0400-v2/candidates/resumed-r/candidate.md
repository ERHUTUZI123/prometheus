# R resumed: acquire finite plasticity programs, then learn through them

AGENT PROPOSAL / RECONSTRUCTION / OPEN. Version 1 of this resumed construction. No experiment executed. Earlier R v0/v2 and all negative records remain immutable. Requested host setting is the inherited authorized Ultra; effective provider mode is unknown. This is not a novelty claim.

## Judgment and success counterfactual

The concrete proposal is a **growing library of goal-conditioned internal update programs**. The learner can stimulate a set of local plasticity processes, but cannot directly assign every persistent weight. It tries finite, reversible cycles of those processes; task feedback selects useful interactions; successful cycles become parameterized learning actions. Later feedback adjusts their arguments rather than rediscovering the complete low-level sequence. Higher-order cycles can use previously acquired actions.

If this worked beyond specially chosen examples, a neural system could acquire a new way to change itself: an initially inaccessible first-order control of a useful internal response becomes a tunable action, while previously learned responses are restored at the end. It would change finite-budget accessibility and the horizon of subsequent credit assignment, not the global reachability set of the original controls. This is relevant to activity/modulation-controlled plastic substrates, distributed local-learning systems and a software neural substrate that deliberately exposes local update processes. It is **not** a benefit for an ordinary addressable scalar weight that an optimizer may simply overwrite. That stronger interface wins the worked case below.

The strongest same-interface alternatives are model-based planning and learned parameterized options, with the same retained histories, inverse controls, macro storage and reuse rights. The proposal's distinctive inductive bias is to search for task-relevant **cancellation programs**, then train through those programs. Classical nonholonomic control already explains the available interactions; parameterized options already explain reusable goal-conditioned policies. Whether this particular organization acquires useful update interfaces substantially more cheaply is OPEN. A favorable result against an unstructured short-horizon planner alone would not answer it.

## 1. Designed substrate and acquired state

Persistent neural state is s (fast weights, local traces and retained controller state); a forward program F_s supplies finite task/protected-output probes O_D(s). The designed local action interface consists of signed flows E_i(a), with E_i(0)=identity and E_i(a)^-1=E_i(-a) on a stated bounded domain. A controller can activate these processes and choose amplitudes. It is not given their joint state-transition equations. A flow can be an exactly reversible local shear implemented in software, or an idealization of a local plastic process. Generic SGD is not assumed to have this inverse. Approximate reversibility must be measured and becomes an additional limitation.

Designed resources: local flows and their bounds/inverses; literal sequencing; physical-owner handles; copying/staging when available; task/protected-output evaluators; a finite constructor grammar; numerical regression/optimization and a bounded archive. These are substantial resources. No interpreter, inverse, task signal or ability to allocate memory is claimed to emerge.

The acquired state at time t is:

* M_t: an acyclic library of literal physical edit programs, their argument ports, owner binding, domain guard, measured effects, costs and failure witnesses;
* H_t: a bounded set of actual transition/task episodes, not a reconstruction of deleted history;
* R_{m,theta}: a learned response predictor for each retained macro, predicting target and protected changes from observable context o and arguments alpha;
* A_{m,psi}: a learned argument map from (o, demanded effect r, protected contract) to alpha;
* a small exploration archive S_t of relevant but not yet safe interaction programs.

M_t and S_t have separate fixed capacities. A speculative seed is not called an acquired useful capability. Forgetting an episode really loses its information unless retained models/code/traces preserve it.

## 2. Exact program constructor and proposal law

Write a semicolon for chronological execution. For callable programs P(alpha), Q(beta), define the designed constructor

    Loop(P,Q; alpha,beta) = P(alpha); Q(beta); P(alpha)^-1; Q(beta)^-1.       (1)

The inverse of any program reverses its instruction list and replaces each primitive by its supplied inverse. Therefore every proposed loop has an executable inverse without inferring a target algebra. A nested call expands to real primitive executions. No compiled constant-time shortcut is assumed.

At each acquisition round, enumerate or sample ordered pairs from the current primitives, active library and seed archive, subject to maximum expanded length L and tree depth d. Amplitude candidates come from a fixed bounded grid initially, then from the learned argument model. The numerical demonstration must disclose this grammar: a commutator prior is handed in, but neither the useful pair, nesting, target effect, amplitude law nor scope is supplied.

For a proposal m, probe on a copied common starting state s, producing the actual change

    delta_m = O_D(T_m(s;alpha)) - O_D(s).                                   (2)

Let r be the desired target-output change and delta_m=(delta_T,delta_P). Define actual endpoint utility

    U(m) = ||r||^2 - ||r-delta_T||^2 - lambda ||delta_P||^2
           - eta * (actual primitive actions + probe work).                 (3)

The current committed state changes only after an actual endpoint contract passes. A model prediction cannot certify it. If staging is used, copy/commit/synchronization costs and the fact that intermediate states were hidden are charged. Without staging, only endpoint retention is claimed, and intermediate caller effects remain real.

The seed archive is needed because a lower-order interaction can have useful task alignment and collateral damage that a later loop cancels. Keep at most K_seed candidates with positive target progress and a nonredundant measured effect after projection against existing effect columns at the tested context. Rank by target progress per expanded action; record protected damage rather than concealing it in a success label. This is a designed stepping-stone policy and may waste its whole budget. It is not unrestricted novelty search and it does not guarantee preservation. The active archive uses the stricter promotion law below.

## 3. Learning the effect and the future update interface

After every actual probe, append e=(o,m,alpha,delta,cost,binding,contract,outcome) to H_t. Train response and argument models using ordinary numerical learning:

    theta_m <- theta_m - h grad_theta ||R_{m,theta}(o,alpha)-delta||^2;
    psi_m <- psi_m - k grad_psi [||R_T(o,A_{m,psi}(o,r))-r||^2
                                 + lambda ||R_P(o,A_{m,psi}(o,r))||^2].      (4)

The second gradient is through a learned predictor, not a claimed exact derivative of the unknown substrate. Real probes validate the result; exploitation of model error is a failure. Finite differences, least squares or a small neural network can instantiate these updates. The bounded test below uses least-squares response calibration and a numerical option optimizer; no neural advantage is assumed.

Promote a seed m to M only after replay from documented common checkpoints shows that **later adaptations using it** cost less than adaptation with the preexisting library, with failures and protected violations counted. Charge the probes used to discover, model, validate and promote it. The finite promotion score is

    Benefit(m) = sum_{e in promotion tasks} [C_old(e)-C_old+{m}(e)]
                 - C_discovery(m) - C_model(m) - C_validation(m)
                 - charged code/storage/compilation cost.                  (5)

Training/discovery and final transfer tasks are separate. A single favorable promotion estimate does not prove a distributional advantage. New routines can call an acquired macro as an internal learning operation. Their controllers see its argument ports, measured response and cost; they do not receive the derivation of its effect. New interaction proposals can in turn use that macro in (1). This is the proposed organizational feedback: task-selected update programs change the action basis and the horizon of subsequent learning, and subsequent tasks select which such programs survive.

The representation of a macro does not by itself save physical execution. Equation (5) may remain negative forever. The important empirical question is amortized acquisition across future tasks, not whether a word can be assigned a name.

## 4. Worked acquired operation: third-order control with lower-order restoration

This is an exact hand reconstruction of a substrate used to specify the test; these equations are not supplied to its learner. Let s=(x,y,z,w), with local plasticity controls

    X(a): x <- x+a;
    Y(b): (y,z,w) <- (y+b, z+b*x, w+b*x^2/2).                               (6)

The second update uses x unchanged during the action. It can be interpreted as a modulation b acting on three local eligibility features (1,x,x^2/2). These particular features are designed; activity-dependent local plasticity, rather than arbitrary direct synaptic writes, is the assumed interface. This is a polynomial control family, not a finding about typical neural networks.

For C(a,b)=Loop(X,Y;a,b), direct substitution gives

    C(a,b): (x,y,z,w) -> (x,y,z+ab,w+ab*x+a^2*b/2).                         (7)

Thus the first interaction is not a clean context-free multiplication: it has a collateral, state-dependent w effect. A seed can be useful without yet satisfying the protected z contract. Now use that acquired C in

    D(c,a,b) = X(c); C(a,b); X(-c); C(a,b)^-1.

Every expanded prefix is a physical action in the original binding. Equation (7) gives exactly

    D(c,a,b): (x,y,z,w) -> (x,y,z,w+abc).                                    (8)

D costs TEN primitive actions, not one. Fix a=c=h=1/2. Two actual response probes with opposite b fit the unknown slope beta=h^2; a future residual r is served by b=r/beta, within bounds. For example r=0.1 uses b=0.4. In a hidden family with w increment gamma*b*x^2/2, the slope is gamma*h^2 and must be acquired from observations.

At x=0 the primitive instantaneous w controls are zero. Under D the endpoint derivative d(w')/db=ac is nonzero, while x,y,z return. This is a specific change in later credit assignment: a scalar task residual can tune a useful persistent change through a retained program, without re-solving the ten-step action sequence each time. Full-horizon AD/planning can achieve the same endpoint; an addressable w write achieves it immediately. Those are limits, not impossibility claims.

To make the operation computational rather than a latent-coordinate excursion, let these fast weights control independent forward features F_s(u)=x f1(u)+y f2(u)+z f3(u)+w f4(u). A task requests a change in the f4 response while prior callers depend on f1,f2,f3. Held-out feature/input probes check the changed forward computation. This is still a small, designed module. Its role is to audit acquisition and retained use of an update procedure, not to claim that new forward architectures or open-ended innovation have been demonstrated.

## 5. Why the example generalizes mathematically, and where it does not

For smooth driftless local vector fields, a small four-flow loop produces a second-order Lie-bracket displacement. Nested loops can expose higher-order directions. Murray and Sastry's original control analysis already supplies this principle, including regularity assumptions; it is SOURCE, not our result. In chained/nilpotent polynomial systems some formulas are exact, as above. In general finite loops have higher-order errors and context-dependent effects. Small useful signals can scale as epsilon^2, epsilon^3 or worse while sensing noise, inverse error and physical action cost do not shrink accordingly.

The broader candidate class is **locally actuated plastic state with noncommuting updates**, recurring task/protection contracts and reusable interaction motifs. It is not all neural learning. Flat/commuting controls give no new direction. A loop producing only a gauge transformation gives no new forward capability. Arbitrary local-learning rules may be irreversible, inaccessible or unsafe to run backward. If ordinary optimizer access supplies every needed control, there is no actuation gap. The unproved consequential step is that meaningful neural substrates have enough reusable nonintegrability, and that task-guided acquisition finds it more economically than ordinary model-based options.

## 6. Ownership, callers and reuse scope

P's independent `candidates/resumed-p/ownership-repair-v2.md` is a substantive attack incorporated here. A valid original physical trace does not imply portability under new aliases. An identical endpoint requires preservation of the binding diagonal; transplanting the same executed program additionally requires prefix/observation realizability. Rebinding the primitive controls to new owners can be executable while changing their commutator law.

Each macro therefore retains physical-owner/caller scope. New owner graphs require quotient-first binding, affected-caller closure and fresh validation or an acquired specializer; neither endpoint success nor an isomorphic forward interface certifies it. The present bounded test does not claim transfer to arbitrary alias topology. P's learned conflict specializer is a separate constructive extension, not an oracle imported into this proposal.

## 7. Real rivals and falsification

The primary rival is ordinary **parameterized options plus planning**. It has the same primitive controls/inverses, snapshots, observations, training history, archive/storage limit and expanded horizon. It can optimize arbitrary signed action words, retain successful words as parameterized options, regress context/demand to option parameters and compose/reuse old options. It can discover repeated parameters and inverse ties from successful words; those are not reserved for R. The difference being tested is the structured loop proposal prior and its recursive retention, not whether the rival is allowed memory.

A stronger diagnostic rival fits low-degree primitive transition models from the same transitions, then uses model-based shooting/MPC with cached parameterized solutions. Polynomial identification is particularly strong in (6); its fitting, internal rollouts and oracle checks must be reported. An unrestricted gradient/direct-write learner is a deliberately richer-interface control, exposing whether the local-actuation restriction is doing all the work.

R fails its acquisition claim if its macros are prelisted, target laws are handed in, future task solutions are used to choose templates, only unnamed stored words distinguish the methods, or total future savings do not repay acquisition. It fails a broader significance claim if the best result is only a familiar control/option prior on a hand-picked polynomial plant. No current evidence rules out that outcome.

CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Finite acquired update programs can change the first-order action interface of later learning | RECONSTRUCTION for example; OPEN for acquisition | Equations (6)-(8), explicit acquisition/update laws (1)-(5) | Economical discovery/retention from feedback versus model-based options | Supplied A/M closure alone did not supply acquisition | Freeze a matched acquisition/transfer audit before execution
The program grammar can generate effects beyond supplied linear Eij | RECONSTRUCTION, not novelty | Nonlinear nested physical flows, explicit cost ten | Classical control already explains this, substrate is designed | No new reachability or free runtime claim | Check whether acquisition cost/transfer survives the real rival
This is a consequential new organization beyond parameterized options | OPEN | A distinct cancellation prior and learned learning interface are specified | May reduce to an existing useful inductive bias with no broader advantage | No neural win or foundational finding asserted | Let same-rights option/model rivals enter the bounded test early
