# Exact standard rival for Gram-mode promotion

**Judgment (RECONSTRUCTION):** a minimal-width ordinary factor network, given the same invariant promotion policy, matches every future residual/output/promotion/path-SGD behavior under the assumptions below. The Gram representation does not create a different learning trajectory against this rival. Function-preserving nonlinear growth remains a real operation, but its substantive difference must lie in a new growth/selection policy, cost advantage, or a broader action interface—not merely replacing wide factors by H.

## Rival and exact induction

Write residual output as `VUx`, with V of shape m×k and U of shape k×n. Set `Z=[V;U^T]`, `H=ZZ^T`, `d=m+n`, and `r=rank(H)`. Choose any thin Gram factor `Zbar` of shape d×r satisfying `Zbar Zbar^T=H`; split its rows into Vbar and Ubar^T. Width r is minimal for representing this *whole Gram state*, not just the cross block VU. Factoring only VU can change SGD dynamics.

For a current task-loss derivative `G=∂L/∂(VU)`, simultaneous plain SGD gives

`V+ = V - eta G U^T`, `U+^T = U^T - eta G^T V`,

`Z+ = M(G)Z`, where `M(G)=[[I,-eta G],[-eta G^T,I]]`.

Thus `H+=M(G) H M(G)^T`. This identity is exact for finite eta, including the eta² term. Ordinary AD/SGD on the thin factor induces exactly the same H update. Equal residual H and equal explicit nonlinear paths imply equal outputs on every input, equal G, and equal explicit-path gradients at the next step.

For a promotion boundary t with `q=t^THt>0`, define

`s=Z^Tt/sqrt(q)`, `p=Zs=Ht/sqrt(q)=[v;u]`.

Then `||s||=1`, `Zres=Z(I-ss^T)`, and

`Zres Zres^T=H-pp^T=Hres`.

The thin rival obtains the identical physical p using `sbar=Zbar^Tt/sqrt(q)`. A latent orthogonal Householder rotation with first column sbar makes p the first column of Zbar. Remove that column and retain the remaining r−1 columns as an ordinary residual factor. No semantic compiler or special optimizer is needed.

Promote the removed linear contribution to the explicit path

`h(x)=v*(z+c phi(z))`, `z=u^Tx`, `c=0` at birth.

The path initially equals `vu^Tx`, so total output is preserved. For output error delta, ordinary gradients are

`grad_v L=delta*(z+c phi(z))`,

`grad_u L=(v^Tdelta)*(1+c phi'(z))*x`,

`grad_c L=(v^Tdelta)*phi(z)`.

The rival has identical v,u,c and error, so these updates match. Residual equality and explicit-path equality are preserved by both SGD and promotion. Induction therefore covers any finite future sequence of data, learning rates and repeated promotions whose choices depend only on H, explicit paths and shared external observations/randomness. Nonlinear paths may later become nonzero and interact through the common loss; the induction still holds.

At q=0 the requested split is undefined and must be skipped/rejected in both systems. Exact arithmetic is assumed; numerical PSD, conditioning and rank tolerances require separate implementation checks.

## A finite-reservoir limitation

For q>0, `rank(Hres)=rank(H)-1`: extraction removes one nonzero direction of the thin factor by an orthogonal projection. Plain SGD has `rank(MZ)<=rank(Z)`. Consequently there are at most rank(H0)<=m+n nonzero promotions without a separate replenishment operation. Original hidden width k supplies no additional hidden reservoir beyond this rank. Replenishment may be legitimate, but its rule, state and costs must be added to both sides of the comparison.

## Costs that actually differ

| Operation | Gram state | Thin factor + ordinary AD |
|---|---|---|
| Residual storage | d(d+1)/2 independent scalars | dr scalars; may be smaller at low rank |
| Single-example forward/SGD | Dense cross-block forward; low-rank Gram updates generally O(d²) | O(dr) factor forward/backward |
| Promotion for a supplied t | Ht and rank-one downdate: O(d²) | Factor matvec and Householder split: O(dr) |
| Initial realization | H already present | PSD factorization, conventionally O(d³); counted once |

The actual boundary-selection policy may dominate these costs. A thin factor can evaluate Ht as `Zbar(Zbar^Tt)` without materializing H. It need not be refactored after each SGD step; latent orthogonal equivariance preserves the realization. Singular rank-drop events can be handled separately if continuously minimal width is required. Compared with a huge-width physical source, Gram compression can be substantial, but the fair thin-factor rival removes that artificial width disadvantage. At full rank the Gram form can save a constant in storage and offer a convenient invariant interface; no asymptotic or learning advantage follows here.

## Limits of the equality claim

The claim uses plain simultaneous SGD and identical structural decisions. It does not automatically cover coordinatewise Adam, per-neuron dropout/pruning, hidden-coordinate-dependent regularizers, untracked optimizer state, or growth choices based on original neuron identities. Such rules can break Gram closure. Scalar weight decay can be included by changing the diagonal blocks of M. Momentum would require additional state; it cannot be silently represented by the same H alone.

The rival copies the *specific invariant selection policy* to test whether H is a genuinely different dynamical organization. This does not show that the policy is already known or useless. What survives as a prospective scientific issue is whether an acquired boundary/promotion policy constructs useful computational primitives under matched information and total cost. The current equivalence says where that effect cannot come from.

## Closest checked original growth precedent

**SOURCE:** Wei, Wang, Rui and Chen, *Network Morphism*, ICML 2016, §3.3, equations (9)–(11), p. 568: <https://proceedings.mlr.press/v48/wei16.pdf>. The original passage defines a trainable activation family interpolating between identity and a nonlinear activation, initializes it at identity to preserve the network function, then learns the activation parameter. This is close mechanism precedent for identity-at-birth nonlinear promotion. Its canonical interpolation is not the identical `z+c phi(z)` parameterization, and the checked passage does not supply the proposed Gram-invariant mode-selection rule. Net2Net (Chen et al., 2016), already read in R's source record, is the broader function-preserving widening/deepening precedent.

No experiment or code was run for this derivation. R v2 and all its freezes are unchanged.
