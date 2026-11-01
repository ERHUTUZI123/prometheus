# Carry-credit audit: averaging over fading traces can dilute a useful local update

RECONSTRUCTION / WITHIN-LINEAGE LEARNING DIAGNOSTIC. Read the actual `mechanism-v1.md` and `probes/after020-local-contrast/contract.md`. This supports the one protected mode-formation hypothesis. It is not a novelty review, a prediction of pilot failure, a contract amendment, or an allocation decision. No test, training, numerical sweep or external source search was run; only this new mathematical audit was written. The frozen pilot must retain its original mean-carry objective.

## Judgment

**Detached local carry avoids a terminal Jacobian product, but its useful gradient can still be diluted by the uniform time average.** For fixed contracting gain A<1, only a bounded early part of the actual replay retains appreciable contrast. Averaging those one-step gradients over L common steps gives an O(1/L) gain update, even while most late local losses remain near their maximal unmet-margin value. That is a learning-signal organization issue, not proof that the scalar candidate family lacks memory capacity.

It can coexist with genuine current state insufficiency at late times under the declared precision/noise scale. Reweighting cannot recover information from an already collapsed episode. Its legitimate role is to change parameters and then rerun the retained prefixes so that future trajectories do not collapse.

A trace-conditioned alternative is mathematically available: train the first actual transition that loses a currently available separation level, with relative normalization and an explicit resolution floor. This supplies a local gain condition independent of the number of irrelevant later steps. It is a possible future construction, not a silent repair of the currently frozen diagnostic.

## 1. Exact conditional gradient for the contract

The scalar family is

    r_t = tanh(A*r_(t-1)+beta*x_t),    r_0=0.

The two episodes have cue +/-1 and then 64 zero inputs. In this symmetric family, branch states are opposite; write their nonnegative half-separation as a. The incumbent first merges at tau=2. The 63 carry transitions start after this merger. Index them j=1,...,L, L=63, with actual replayed predecessor magnitude a_j and next magnitude

    b_j=tanh(A*a_j).

The contract uses one quarter of mechanism Eq3. Since the pair distance is 2b_j, each conditional carry loss is exactly

    ell_j=(m-b_j)_+^2,                 m=0.4.             (1)

The actual predecessor a_j is detached during this update. Therefore

    g_j = partial_A ell_j
        = -2 (m-b_j)_+ (1-b_j^2) a_j,                  (2)

and

    G_mean=(1/L) sum_j g_j.                              (3)

The carry term has **zero direct beta and v gradients** on these hold steps: x_t=0, previous state is detached, and v does not occur in the local separation objective. Seed and endpoint terms are responsible for their other parameter updates in the actual combined arm. This must not be confused with differentiating the full replayed carry trajectory with respect to beta.

For A>=0, a_j>0 and b_j<m, (2) is negative: the local update pushes A upward. There is no sign cancellation among these symmetric active carry steps. In this narrow zero-input scalar family with beta fixed, increasing A also increases every actual replayed positive branch state, so the direction agrees with reducing unmet replayed carry margins. Its small size can still matter, and other jointly trained terms can change the full learning trajectory.

## 2. An exact dilution bound, and the small-signal expression

Assume 0<=A<1 and let s=tanh(|beta|). At the cue, the magnitude is s. Because tanh(u)<=u for u>=0,

    a_j <= s*A^j.                                        (4)

Here j=1 corresponds to r_2, so the first exponent is one. Equations (2)-(4), (m-b)_+<=m and 1-b^2<=1 give the exact upper bound

    |G_mean| <= (2*m*s/L) * A*(1-A^L)/(1-A)
             <= 2*m*s*A/[L*(1-A)].                      (5)

For any fixed A bounded below one, this is O(1/L). No successful neural training or linear approximation is needed for that inequality.

In the early regime where states are much smaller than m, tanh is nearly linear and cumulative cubic corrections are small, for example |beta|^2/(1-A^2)<<1. Then

    a_j approximately |beta|*A^j,
    G_mean approximately
        -2*m*|beta|*A*(1-A^L)/[L*(1-A)].                 (6)

The relevant assumptions include a fixed A<1, small states relative to m, and the current scalar/symmetric/detached update. Formula (6) is not a uniform approximation as A approaches one or after seed learning substantially changes beta.

At the frozen initial values A=0.5, beta=0.01, m=0.4, L=63, (6) is approximately -1.27*10^(-4). The seed loss is `(r_2-m)^2`, whose initial A derivative is approximately -0.0079. Thus the initially averaged carry gradient is only about 1.6% of the seed A gradient under their equal objective coefficients. These are hand small-signal estimates at the initial checkpoint, not a training run or a forecast of the 512 updates.

Late b_j near zero still contributes loss approximately m^2=0.16 but gradient approximately zero. The mean local loss can therefore remain substantial while its derivative on A is supported almost entirely on the first few actually separated states. A large loss alone is not evidence of a usable local teaching signal at every step.

The global gradient cap applies after combining objective components in the actual pilot. Raw-component dilution and subsequent combined clipping should be distinguished using the already-contracted component logs; this note supplies no new numerical setting.

## 3. Conditions and countercases

### Dilution has a specific domain

Equation (5) describes a fixed contracting regime. If a positive fraction of the L predecessor states retain appreciable contrast below margin, then the sum in (3) can have order L and the mean can remain O(1). For a persistent fixed magnitude a_star in (0,m), late conditional gradients approach

    -2*(m-a_star)*(1-a_star^2)*a_star,

which does not decay with L. If all states satisfy the margin, zero carry gradient is appropriate, not dilution. Near-critical A depending on L can also invalidate a fixed-A O(1/L) interpretation; the geometric sum itself must be examined.

### Exact collapse is not repaired by normalization

At a_j=0, b_j=0 and (2) is exactly zero. No one-step gain change can distinguish identical inputs to the same deterministic hold transition. Carry alone cannot create a write from r=0. The seed/prefix replay, a different actual input, or another acquired structural action must create a distinction first. A divisor cannot supply the missing bit.

With finite precision or state noise, a very small measured separation may likewise be unusable or unreliable. Its resolution must be stated. Current late-state failure and averaging dilution are compatible explanations at different levels: the late runtime state is poor, while the learning rule also underweights earlier states that could teach prevention of that loss.

### State capacity is a different claim

The admitted parameter family includes A>1, so it can represent stable sign modes. Failure to reach that regime under one learning schedule is not a scalar-capacity lower bound. Conversely a tiny nonzero real sign under A<1 is not proof of robust task memory. Terminal loss, readout limits, the chosen m and the declared error/precision budget retain their separate roles.

The single-cue training episodes still do not identify first-cue versus last-write semantics. Improving carry on them does not establish overwrite/reset acquisition. This audit leaves that information boundary unchanged.

## 4. A principled trace-conditioned repair for a future frozen version

Use only actual paired replayed states and the existing witness. At the seed checkpoint, let a_0 be the observed half-separation and define

    c = min(m,a_0).                                      (7)

Detach c for the conditional update. Require c to clear a declared numerical/noise resolution epsilon. epsilon is an explicit measurement/robustness assumption, not a success-tuned hidden memory target. If c is zero or unresolved, abstain from carry normalization and return the record to seed formation/replay; do not invent separated predecessor states.

Replay the actual common suffix and select its first transition satisfying

    a_j >= c,    b_j < c.                                (8)

This is the first documented loss of a currently available distinction. Train that actual transition with

    ell_frontier = ((c-b_j)_+/c)^2,                       (9)

holding a_j, c and the discrete selection fixed inside its one-step derivative. Its exact conditional gain derivative is

    g_frontier = -2*(c-b_j)_+*(1-b_j^2)*a_j/c^2.         (10)

For small states with a_j approximately c and b_j approximately A*c, this becomes

    g_frontier approximately -2*(1-A)_+.                 (11)

It has neither the factor 1/L nor the original small contrast amplitude. No additional intermediate truth label is used: c comes from actual seed separation; the need not to lose a distinction comes from the already retained matched-continuation witness. The margin convention remains a design choice rather than a recovered true latent state.

After changing parameters, recompute the real trajectories and locate the actual crossing again. Do not train indefinitely on a frozen invented predecessor. If no crossing appears within the observed suffix, the evidence is only that this level survived that finite suffix. It is not proof of indefinite persistence. If repeated actual transitions are desired instead of the first, their sampling rule must similarly state its conditional normalization and exact expected update.

This is a changed conditional learning law, partly an adaptive rescaling of the available signal. It is not the gradient of the original uniform mean, nor of the full endpoint objective. A future finite-difference check must freeze a_j, c and selection, just as the existing contract correctly freezes predecessor states. The rule earns its value only through actual subsequent task learning and cost, not from the larger analytic gradient.

## 5. Exact gain condition supplied by the repair

For a fixed separation level 0<c<1, the whole interval [c,1] is forward invariant under `f_A(a)=tanh(A*a)` if and only if

    A >= A_c := atanh(c)/c.                              (12)

Proof: f_A is increasing for A>=0. Its minimum on [c,1] is f_A(c), so interval preservation is exactly tanh(A*c)>=c, equivalent to (12).

If 0<=A<A_c and the current seed starts at magnitude at least c, a sufficiently long deterministic zero-input trajectory crosses below c. For A<=1 it tends to zero. For 1<A<A_c, its stable positive fixed point is below c; a trajectory starting above c descends toward that point and eventually crosses c. At the boundary A=A_c it can approach c without crossing. A finite observed suffix may miss a very late crossing near this threshold.

Thus (8) supplies a genuine violation witness for the retention level c. Every such selected transition has a positive predecessor at least c, and (10) pushes A upward. This is the mathematical reason state-conditioned selection can be better directed than averaging mostly extinguished states. It is not a convergence theorem for the combined seed/endpoint/carry updates.

Once seed learning supplies a_0>=m, c=m. In the contract m=0.4, the invariant-band threshold is atanh(0.4)/0.4, approximately 1.059. This is a condition for the designed absolute carry margin, not a universal requirement for classifying a noiseless sign.

### Countercase: preserving a tiny available contrast is not enough

For c very small, A_c=1+c^2/3+O(c^4), close to one. The normalized carry rule can preserve that tiny level without building the task's desired m-scale robustness. At A=1 and a_j=c, tanh(c)=c-c^3/3+O(c^5), so (10) is only about `-2*c^2/3`; normalization does not manufacture a strong signal arbitrarily close to neutral, perfectly tiny contrast.

Consequently seed strengthening and actual endpoint/readout evaluation remain essential. A carry-only success on a minute relative level does not establish a useful persistent mode. Dividing by c^2 also amplifies estimation error when c is near its resolution; this is why (7)'s admission condition is substantive. No hidden extra state or latent-bit label repairs that countercase.

## 6. Discovery, execution and information costs

The proposed conditioning uses the paired trajectories already required by the learner. Locating the first crossing costs O(L) comparisons and enough retained/recomputed state to evaluate it. Replaying both episodes still costs O(L) transition work after updates. One selected local derivative can cost less than 63 local derivatives, but no total O(1) learning claim follows: mining, replay, seed updates, endpoint updates, scanning and retention all count.

The two existing endpoint labels are reused; no new teacher intermediate is queried. The original witness's provenance and reliability remain necessary. A reference controller is not allowed to restore a_0 or replace an extinguished state with c during execution. Any future matched control gets the same witness, actual traces, selection/rescaling rule if the mechanism comparison requires it, and the same charged budget.

## Interpretation of whatever the frozen pilot returns

The frozen mean-carry pilot may succeed, fail, or reveal a different interaction. Its seed gradients can change beta and A, thereby altering the carry support; initial estimate (6) cannot decide that trajectory. The existing persistent-initialization control addresses another condition and is not a same-initial-state causal estimate.

This audit supplies one conditional learning fact to use when interpreting the records: **a uniform mean over a long fading trace can dilute the update supported by its early separated states, and this can be checked independently of whether the candidate family can represent persistence.** A first-loss, resolution-aware, relative update is a concrete same-lineage repair to consider only in a separately frozen later version if the actual records make that unknown informative. No pilot setting, result, contract or status was changed here.
