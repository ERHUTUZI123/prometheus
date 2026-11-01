# Diagnostic support: finite memory separation is different from gradient transport

RECONSTRUCTION / MATHEMATICAL DIAGNOSTIC. Support for the one selected task-driven persistent-dynamical-mode hypothesis, under the deployed continuous-development policy. This is not another candidate, a novelty judgment, a status dashboard, or a MAIN/PARK decision. No training, test, simulation or formal prover was run. Only this new support artifact was written.

## Immediate deployment check

I read the actual three-file git diff and the deployment-check JSON. Independent read-only comparisons against the pre-amendment tracked blocks confirmed: permanent principles byte-exact; stronger current AGENTS §5.2 byte-exact; optional cross-field role byte-exact; new continuity/window/review-layer headings each occur once. The actual hashes match the coordinator's record:

- AGENTS.md: 803a3d4ab6d4ce3c8ba6c538faa3dd44148285511cde0f3f38085b52ce032b9c
- deep-work/WORKFLOW.md: d3a2ca86de0471a5962e560c18d4951577d55325a9c2215d43a4134bdc096541
- deep-work/ROLES.md: e25d3586d4951f8d3a5630c30d251756b5b2e314ba441a108b91862ac7b01ca1

The deployed text preserves resource authorization, immediate correction of fatal factual errors, independent review before confirmation/increased commitment, and USER-PARK. I have read the updated policy and am supporting the selected hypothesis; earlier independent-candidate work is not resumed.

## What the closed behavioral-state construction already established

The actual CLOSED `after017-outside-principle/comparison-v1.md` was reread. Its precise reusable results are:

1. For a finite linear learned behavior `R_p p + R_f f = 0`, restricted to admissible pasts, equality of future-completion sets is equivalent to `R_p(p-p')=0`. A sufficient finite-future state can be formed as the corresponding quotient/rank factorization. This is known finite behavioral realization, not a newly acquired nonlinear mode principle.
2. A state sufficient for one next observation need not support continued execution. For `x_(t+2)=a*x_(t+1)+b*x_t`, the two-step future map from `(x_t,x_(t+1))` has determinant `b^2`. With a=b=1, histories `(1,0)` and `(0,1)` have the same next value but different following values.
3. Those facts concern a specified fixed behavior and future horizon. They do not preserve information under arbitrary later learning, prove nonlinear state acquisition, or establish that a neural mechanism found the necessary representation. Its exact linear same-information solver correspondence remains valid within that version; it does not close the present dynamical-mode hypothesis.

The useful warning for the current development is to choose a continuation that actually distinguishes the demanded histories. An immediate-output probe can miss the memory requirement entirely. No old DEN/growth source claims are needed for the diagnostic below.

## One diagnostic: paired continuation with finite-separation and tangent channels

Select two actual task histories that require different answers after an otherwise identical allowed continuation. Write their demanded cue as C in {-1,+1}, balanced for the diagnostic. This notation need not introduce a new task; it can index an existing task-relevant pair of histories.

Start both branches from a documented common checkpoint. Freeze slow learned parameters for the measurement, but retain every runtime state transition required by the mechanism, including declared fast plasticity. Use the same neutral inputs/distractors and specified common/noise coupling after the histories. At delay D, denote the complete causal runtime state by Z_D. Include fast weights, gate/allocated-mode state and other information that can affect the future answer. Do not include an external audit log unavailable to the computation, or grant free unlimited history.

There are two separate channels to inspect:

- **Finite-separation channel:** how distinguishable are the two complete resulting state distributions, and how much noise/quantization can they tolerate before the task-relevant distinction disappears?
- **Tangent channel:** how much derivative signal is carried from an infinitesimal perturbation at the cue-state cut to the later task loss?

These are two parts of one paired-continuation assay. The central mathematical result below is that they can differ radically, even with exactly matched tangent decay.

## 1. Exact information/aliasing statement

Let P_+ and P_- be the distributions of Z_D for the balanced two histories under the declared noise. For a finite-precision state space, the best possible classification error over arbitrary readouts of Z_D is

    epsilon_star(D) = (1/2) sum_z min(P_+(z),P_-(z))
                    = (1-TV(P_+,P_-))/2,                 (1)

where `TV=(1/2) sum_z |P_+(z)-P_-(z)|`. The derivation is pointwise: for each observed z choose the more likely cue; errors contribute the smaller probability. The same density formula applies to the continuous examples below.

Consequences with their proper scope:

- If the full causal-state distributions are identical, no frozen readout can recover the cue better than chance. This is information loss/aliasing at that checkpoint and horizon, not just slow output training.
- If they are distinguishable but the current answer mechanism performs badly, its excess error above (1) is use/readout/optimization error or an access/cost limitation. It does not establish absent state capacity.
- Equal current states for two histories do not prove that every allowed reorganization or retraining must identify them. That stronger capacity claim needs an architectural/finite-precision restriction valid across learning.
- Conversely, different exact real numbers do not imply a usable persistent computational distinction. A state can hide arbitrarily many bits in arbitrarily fine real precision. Robustness and numerical budget must be stated.

For M mutually distinguishable task histories and only b available bits of causal runtime state, more than 2^b distinct exact state classes are impossible. No analogous finite counting lower bound follows merely from the number of real-valued neurons without precision, range or robustness assumptions.

An arbitrary full-state readout in (1) is an information diagnostic, not a free computational module. If an audit reads weights or coordinates unavailable to the actual query interface, label its result as an upper envelope of recoverability rather than the system's achieved capability. A trained finite probe with poor error cannot prove P_+=P_-; it may be an inadequate probe. On a small declared finite state domain, exact comparison is possible; otherwise one gets bounded/sampled evidence, not a universal aliasing certificate.

## 2. Matched vanishing gradients, different robust memory: exact construction

Consider the scalar recurrent map

    z_(t+1)=tanh(kappa*z_t),      kappa>1.                (2)

It has nonzero fixed points +/-a, with `a=tanh(kappa*a)`, 0<a<1. Starting the two cue branches at +/-a leaves them there under every finite neutral delay. Their noiseless separation is exactly 2a, and their noiseless state distributions are disjoint.

The local derivative at either fixed point is

    r=kappa*(1-a^2),      0<r<1.                         (3)

One way to verify the strict upper bound is to use `kappa=atanh(a)/a` and

    atanh(a)=integral_0^a 1/(1-u^2) du < a/(1-a^2).

Thus the cue-state-to-later-state tangent Jacobian is exactly

    J_D=r^D -> 0.                                        (4)

Now compare a fading recurrent map

    z_(t+1)=r*z_t,      z_0=+/-a,                        (5)

using **the same r** as in (3). Its tangent Jacobian is also exactly r^D. But its finite cue separation is `2a*r^D`, rather than 2a.

The two systems therefore have identical vanishing tangent transport along these trajectories but different finite memory retention. In (2), contraction removes small within-basin perturbations while preserving which basin the cue selected. That can be a desirable persistent-mode organization. It is false to diagnose missing memory solely from `||J_D|| -> 0`, or to demand a unit-eigenvalue direction as a prerequisite for persistent categorical memory.

### A specified noise scale makes the distinction operational

For this calculation only, add independent Gaussian observation noise with standard deviation sigma at the query cut, rather than noise at every recurrent step. The two observed scalar distributions have means +/-m_D and common variance sigma^2. Their optimal balanced decision is the sign of the observation, with

    epsilon_star(D,sigma)=Phi(-m_D/sigma).                (6)

For (2), `m_D=a`, so this error is independent of D. For (5), `m_D=a*r^D`, so it tends to 1/2 as delay increases. This is an exact state-readout noise comparison, with identical tangent decay.

With a fixed rounding grid of step delta, the fading branches can also collapse to the same rounded state once their values lie inside the zero rounding cell. Merely retaining infinitesimal exact signs does not meet that finite-resolution contract.

If noise acts throughout the recurrence, (6) is not the answer. Noise can eventually cause a transition between attractor basins, and escape/mixing probabilities over the actual finite task horizon must be evaluated. The noiseless fixed-point proof does not give indefinite noisy reliability. This limitation belongs in the present hypothesis's error budget, not in a reason to discard its formation question.

## 3. Tangent credit and actual learning are also different statements

At a cue-state cut z_0, with later loss L and a differentiable runtime transition, the derivative transported specifically through that cut is

    grad_(z_0) L = J_D^T grad_(z_D) L.                  (7)

For shared recurrent parameters, the full parameter gradient also includes contributions injected at every later use of those parameters. Equation (7) is not a complete parameter-gradient formula. Nor does a small (7) show that structural growth, a finite perturbation across a basin boundary, or another allowed learning rule lacks a useful signal.

The diagnostic should therefore report finite branch separation and tangent transport separately. In particular:

- Good finite/noise-robust separation plus a small J_D is consistent with a successfully acquired stable mode.
- Small finite separation at the declared noise/precision scale indicates a representation/retention problem under that condition, even if a short-lag Jacobian is large.
- A remaining task error with recoverable state information but an ineffective cue-parameter update motivates a learning/credit-assignment investigation. It does not establish a capacity lower bound.

This does not prescribe K's acquisition law or require gradient learning to succeed at creating (2). It supplies a way to tell whether the law has actually formed the relevant persistent organization, rather than treating either a spectral plot or a task score as the whole mechanism.

## 4. Rare labels and noise: controls inside the same assay

If a demanded query occurs with probability q and other events produce no loss on the relevant memory parameters, then

    E[L_training] = q E[L_query | demanded pair],
    grad E[L_training] = q grad E[L_query | demanded pair]. (8)

An unconditional gradient can be small solely because q is small, with no temporal derivative attenuation. Balanced, query-conditional reporting removes this dilution for diagnosis. It does not give the learner extra free labels: if queries are sampled naturally, obtaining n such examples costs about n/q episodes in expectation, and any resampling/intervention cost must be charged.

When nondemanded events also update those parameters, the exact formula is instead

    g_total = q*g_query + (1-q)*g_other.

Cancellation/interference can then mimic small signal. Do not call the whole effect rare labels or vanishing temporal gradients without the conditional terms.

Likewise use distinct conditions for deterministic aliasing, readout noise and noise during state evolution. Equal mean states do not imply equal state distributions, and a mean-distance plot can miss information in variance or another runtime coordinate. Equation (1), or a clearly declared computable lower/upper approximation to it, identifies which claim is being made.

## Practical interpretation for the protected development interval

The one useful diagnostic target is a **finite task-relevant separation that persists under the declared error budget**, paired with an explicit account of the learning signal that formed it. The tanh/fading comparison shows why this target should not be replaced by nonvanishing gradients. The old finite behavioral-state result shows why one immediate prediction should not be substituted for continuation sufficiency.

The diagnostic can be instantiated later on K's actual state, formation rule and task witness. This note authorizes no test and supplies no new benchmark requirement. It neither claims novelty for attractor memory nor closes the selected hypothesis because familiar recurrent/gated mechanisms exist. Its information value is to distinguish *what is absent*: demanded state information, robust separation, usable readout, informative label frequency, or the specific learning signal.
