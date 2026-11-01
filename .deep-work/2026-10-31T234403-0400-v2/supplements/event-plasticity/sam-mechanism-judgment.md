# SAM and the post-compaction acquisition gap

**Judgment — RECONSTRUCTION:** SAM provides a concrete candidate for learning a credit-assignment interface within a fixed scaffold. It deserves constructive attention because its plastic cross-stream maps relax handwired neuron correspondence. It does not, by itself, supply a rule that restores deleted coordinate information, invents a new computational primitive, or proves that a new/perturbed interface becomes useful. The mathematically decisive open quantity is pathway alignment, not reciprocal weight matching. This is directly connected to the Fundamental Question through the acquisition of machinery that enables later learning.

The source record and exact reading locations are in `sam-primary-reading.md`. Everything below is our reasoning from those equations. It is neither an independent replication of the authors' experiments nor a novelty claim.

## Information and organization accounting

An output loss and its derivative inject task information into the downstream population. Current input activities, local gates, feedback activities and cross-stream synaptic currents then supply the four outer-product updates. Their locality is locality of the delivered signals: task information still originates at the output, and delayed feedback needs an activity trace. A synapse need not know a full downstream weight matrix, but it must receive its prescribed instructive signal. Consequently, task feedback, eligibility memory and the persistent feedback maps all count as resources; none is a reconstruction of deleted history.

Learning inference weights and learning the weights that teach inference are different organizational changes even when both are parameter updates. Thus “merely training” is an unhelpful dismissal. The precise limit is that the main mathematical system retains fixed layers, fixed population sizes, four allowed matrix types, fixed driving/modulatory roles, and a specified error port. Updating a zero-valued allowed coefficient could create an effective connection, but the algorithm does not specify graph insertion/deletion, neuron creation, or operation-type invention. The companion adds a genuinely separate representation/mutation/selection mechanism for graph change. We should count that outer search explicitly rather than attribute it to the four local updates.

## What the reciprocal identity establishes

Let E_l=G_l-C_{l-1}^T. With matched rates and decay, subtracting the two updates gives

\[
E_l^+=(1-\gamma)E_l.
\]

This is a real derivation, and it applies even while the common outer-product drive is changing. In discrete time contraction requires |1−γ|<1, i.e. 0<γ<2; conventional nonoscillatory decay uses 0<γ≤1. The continuous-time equation contracts for every γ>0. These statements should not be conflated. With rate mismatch, a forcing term (η_G−η_C)b_l a_l^T remains; with decay mismatch, another forcing term remains. Exact reciprocal matching is therefore conditional on coordinated local rules, not a proof that arbitrary plasticity discovers matching.

This is the same cancellation principle as the reciprocal-learning predecessor. SAM applies it to the cross-stream interface rather than requiring those cross-links to be identity maps. That distinction can matter under unequal population sizes. It does not prove that the signal arriving through the whole downstream network is an appropriate teaching signal.

## Reconstruction of the generalized-SGD argument

For one fixed sample and current gates, define T_l=D_l F_l^T and the pathway mismatch

\[
P_l=G_lT_l-\bar D_lR_lC_l^T.
\]

When G_l has full column rank, multiplying by its pseudoinverse gives

\[
T_l=G_l^+\bar D_lR_lC_l^T+G_l^+P_l.
\]

Compose these bridges from the readout downward. Each internal splice C_{l-1}^T G_l^+ equals the projector onto col(G_l), plus a term proportional to E_l. Projector leakage is proportional to P_l. This produces a finite exact decomposition of the SAM teaching operator into a preconditioned true-gradient operator plus transported error terms. No stochastic optimization convergence theorem is required for that algebra.

The resulting main data-update term is −η_F D(C C^T+C E)D ∇_F loss. When E=P=0 on the relevant chain, the error terms vanish and D C C^T D is PSD. Full-column-rank G requires downstream width at least upstream width; equality is permitted. Strict downstream widening is not a theorem requirement. Width alone does not imply rank, preserve rank during training, or control the pseudoinverse norm. Ill-conditioned G amplifies a small pathway mismatch.

There are three distinct boundedness requirements. First, bounded C makes C E vanish as E contracts. Second, bounded downstream products make the projector-leakage term vanish with E. Third, a useful uniform residual estimate needs controlled pseudoinverses, transported matrix products, output gradients and activities. Depth products can amplify small local errors. A finite sample-wise identity with explicit factors is weaker than a uniform-in-time or uniform-in-depth approximation.

C C^T itself is PSD at every instant. It is the generally nonsymmetric finite-error effective prefactor that approaches the gated PSD form under boundedness. PSD gives a nonopposing infinitesimal data direction; it does not ensure strict progress in every direction because gates can make it singular. Finite step sizes, residuals and the separate weight-decay term further prevent inference of monotone task loss. No theorem here establishes successful acquisition from arbitrary initialization, convergence of P_l, later-task retention, held-out composition, or discovery of a new operation.

There is an additional sample-dependence limit. At a fixed weight state with the default linear downstream gates, the feedback operator R_l C_l^T is independent of x, whereas the forward operator contains D_l(x). For two inputs x,x′,

\[
P_l(x)-P_l(x')=G_l[D_l(x)-D_l(x')]F_l^T.
\]

Consequently,

\[
\max(\|P_l(x)\|,\|P_l(x')\|)
\geq\tfrac12\sigma_{\min}(G_l)
\|[D_l(x)-D_l(x')]F_l^T\|.
\]

This uses the triangle inequality and injectivity of G_l. Thus exact whole-operator pathway alignment across genuinely different, consequential ReLU gate patterns is generally unavailable with fixed weights and linear feedback. The theorem is sample-wise and conditional, so this is no contradiction. It also does not prevent useful learning: matching the teaching operator's action on the encountered output-gradient directions can suffice even when a global operator-norm bound is pessimistic. A future acquisition claim should distinguish those two targets.

## ATTACK — exact reciprocal matching can coexist with ascent

This is an analytic state witness, not an experiment or a refutation of the conditional theorem. Consider two scalar forward weights, input x=1, a positive hidden unit, linear output, loss f²/2, identity downstream gates. Set

\[
w_0=0.1,\quad w_1=1,\quad G_1=C_0=1,\quad C_1=1,\quad R_1=-1.
\]

All relevant cross-maps have full rank, the output connector is identity, and E_1=0 exactly. The output is f=0.1; feedback activities are b_2=−0.1 and b_1=0.1. Hence

\[
\Delta w_0=0.1\eta-0.1\gamma,\qquad
\Delta w_1=-0.01\eta-\gamma.
\]

Choose γ=0.1η and 0<η<1. The new output is

\[
f^+=(0.1+0.09\eta)(1-0.11\eta)
=0.1+0.079\eta-0.0099\eta^2>0.1.
\]

The loss strictly increases. The pathway mismatch is P_1=1−(−1)=2, so the missing small-P premise is explicit. Reciprocal self-matching alone therefore supplies no sign guarantee for task learning, even with matched rates, positive decay, full rank and perfectly valid output feedback. This witness does not establish persistent failure: subsequent learning might repair the pathway.

## ATTACK — the learned metric does not close future plasticity

SAM's aligned metric H=C C^T has the same Gram form as the earlier compact covariance mechanism. Its source is different, but the post-compaction issue survives. Suppose the cross-interface receives fresh activities a,b and updates by C^+=αC+ηab^T, α=1−γ. Expanding the Gram matrix gives

\[
H^+=\alpha^2H+\alpha\eta(u a^T+a u^T)
+\eta^2\|b\|^2aa^T,\qquad u=Cb.
\]

Therefore H plus a alone is insufficient. Even H,a,b are insufficient unless u, or enough interface information to compute it, remains available. For example C=(1,0) and C'=(0,1) have the same H=1, but for b=(1,0)^T they give u=1 and u'=0 and hence different subsequent H. No probabilistic or asymptotic assumption is involved.

This identifies a constructive sufficient interface: current a, the instructive current u, and the feedback energy ||b||² close this particular Gram update. But if the original C is deleted, its current u cannot remain available by fiat. One must retain appropriate circuitry/state, derive the new current from retained information, or learn a new interface from fresh observations. Choosing a canonical factor is permissible new design; it does not reproduce the deleted coordinate-specific learning rule unless all dependent ports transform consistently. Coordinatewise nonlinear plasticity generally breaks Gram equivalence.

## HYPOTHESIS — useful connection and strongest rival

The credible proposal is to let a new module acquire its teaching interface from new input/output-error experience after compaction. Explicitly allocate its new state, feedback ports and calibration samples. The desired effect is later learnability without a manually supplied coordinate correspondence. This could change how computational mechanisms can be added and taught; it does not need to avoid gradients to matter. The theorem suggests what to inspect, but does not establish repair: both reciprocal matching and pathway alignment must recover, alongside task capability.

The strongest simple rival is an ordinary minimum-width factor network with the same growth/morphism, persistent state, feedback ports, fresh examples and trainable adapters. Reciprocal KP-style learning and fixed-feedback alignment supply additional mechanism controls. A fair comparison must charge SAM for all four matrix families, feedback width, forward traces, error information and calibration work. A gain over a baseline denied the new observations or adapters would identify an information/resource difference, not an organizational principle. A useful distinction would be a documented reduction in required handwired correspondence, retained interface state, or calibration resources under matched task acquisition and retention.

**Next single decision — OPEN:** whether this interface-acquisition question is consequential enough to replace the unresolved P/R mechanism, after comparison with the matched adapter alternative. This supplement authorizes no new training or diagnostic run. SAM motivates that question; it does not close the existing gap.

## Supervisor precision note added after reading

The phrase above that both reciprocal matching and pathway alignment must recover refers to invoking this particular generalized-SGD guarantee, not a necessary condition for all useful learning. Credit on encountered error directions can be useful with substantial whole-operator mismatch; feedback alignment need not satisfy exact reciprocal matching. Do not close a construction candidate solely because it fails a stronger sufficient theorem condition. The source equations and independent witnesses above are unchanged; no new experiment was performed.
