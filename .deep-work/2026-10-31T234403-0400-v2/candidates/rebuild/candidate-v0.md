# Independent reconstruction v0 — frozen before executable results

Provenance: AGENT RECONSTRUCTION from `inputs/neutral-rebuild.md`, read before
any author candidate/code/result. This file states an analytic result, with the
executable check specified separately. No primary-literature or novelty claim.

## Question and assumptions

For positive integers m,n,h, real A in R^(m x h), B in R^(h x n), execution
uses W=AB and simultaneous SGD uses OLD A,B with externally supplied arbitrary
G in R^(m x n) and eta>0. Future G may also be determined by the preserved
network function and an external task sequence. No optimizer memory,
factor-specific losses, noise, clipping or factor-specific future intervention
is part of the stated law. Exact real arithmetic is the analytic domain.

The connection to the fundamental question is narrow: determine what internal
learning organization, invisible to today's forward function, must survive
an exact compaction. It does not demonstrate construction or discovery of an
internal mechanism by a neural learner.

## Sufficient state and closed law

Stack X=[A;B^T] in R^((m+n) x h). Define

    P=AA^T, W=AB, Q=B^TB,
    S=XX^T = [[P,W],[W^T,Q]],
    H(G,eta) = [[I_m,-eta G],[-eta G^T,I_n]].

The simultaneous update is X_next=H X, hence

    S_next = H S H^T.

This is an exact, positive-semidefinite closed state of size independent of h.
Its explicit blocks are

    W_next = W-eta(PG+GQ)+eta^2 G W^T G,
    P_next = P-eta(GW^T+WG^T)+eta^2 G Q G^T,
    Q_next = Q-eta(G^TW+W^TG)+eta^2 G^T P G.

Induction proves agreement of every future W, hence all forward outputs Wx.
If the surrounding network depends on this block only through W, identical
external states and inputs imply identical G and consequently the same
induction. The claim concerns mathematical trajectories, not bitwise equality
of floating-point implementations.

Every PSD S has a real factor Y with YY^T=S and width rank(S)<=m+n, by its
spectral decomposition. Partitioning Y into its first m and last n rows gives
new A and B^T. This is an exact factor compaction; preserving W alone need not
preserve learning. Maintaining S directly is another implementation. If the
original h is already small, no storage or speed benefit is implied. S requires
m(m+1)/2 + mn + n(n+1)/2 independent real entries.

## Current function, initial response, and future behavior

W alone is insufficient. In m=n=h=1, factors (a,b)=(1,1) and (2,1/2) both
give w=1 but give p+q=2 and 17/4. Their exact next products differ by
-(9/4) eta g whenever g is nonzero.

Given W, the complete first-order gradient-response operator is

    T(G)=PG+GQ.

Because the eta^2 term uses W only, W and T also determine every exact
one-step product for arbitrary G,eta. They need not determine the product
after two or more steps.

Here is the precise lost information. If two states have the same W and T,
then D_P G+G D_Q=0 for every G. Substituting the matrix units E_ij first
forces the off-diagonal entries of D_P,D_Q to vanish and then forces
(D_P)_ii+(D_Q)_jj=0 for all i,j. Therefore

    D_P=c I_m, D_Q=-c I_n

for one real c. Conversely this pair leaves T unchanged.

After a common first update (G,eta), these differences become

    D_P_next=c(I_m-eta^2 GG^T),
    D_Q_next=-c(I_n-eta^2 G^TG),
    D_W_next=0.

For a later gradient K, the difference between response operators is

    D_T_next(K)=c eta^2(KG^TG-GG^T K).

If c is nonzero and max(m,n)>1, one can choose rank-one G and K making this
nonzero. For n>1 use G=e_1 e_1^T and K=e_1 e_2^T in the rectangular shapes;
then GG^T K=K and KG^TG=0. For n=1,m>1 use G=e_1 and K=e_2; then
KG^TG=K and GG^T K=0. Thus different realizable Gram states are distinguishable
by some finite sequence of permitted inputs whenever max(m,n)>1: if W differs,
the current output distinguishes; if T differs, one step distinguishes; if
only c differs, two steps distinguish. This is behavioral state necessity,
not a lower bound against pathological encodings of real numbers.

A concrete realizable initial-response pair has m=1,n=2,h=6:

    A1=[1,1,0,0,0,0]
    B1^T=[[0,0,1,1,0,0],[0,0,0,0,1,1]]
    A2=[1,1,1,0,0,0]
    B2^T=[[0,0,0,1,0,0],[0,0,0,0,1,0]].

Both have W=0 and T(G)=4G, while their (P,Q) are (2,2I_2) and (3,I_2).
For G1=[1,0], eta1=1/2, both give W1=[-2,0]. For G2=[0,1], eta2=1/2,
their products are respectively [-2,-9/4] and [-2,-17/8], differing by
[0,1/8]. Even all exact one-step response information initially is insufficient.

There is a genuine scalar exception. If m=n=1, the state (w,r=p+q) closes:

    w_next=w-eta g r+eta^2 g^2 w,
    r_next=(1+eta^2 g^2)r-4 eta g w.

The difference p-q is behaviorally invisible for every permitted gradient
sequence. Therefore full Gram sufficiency should not be overstated as
universal minimality.

## Zero-product appends

Appending new paths with B_new=0,A_new=U changes P by UU^T and leaves W,Q
unchanged. Appending A_new=0,B_new=V changes Q by V^TV and leaves W,P
unchanged. A mixture has

    S_after_append=S+diag(UU^T,V^TV).

This captures these appends without retaining the count or identity of paths.
Its very next product differs by -eta(UU^TG+GV^TV). For a specified G this
can vanish; a nonzero unilateral append changes the response for some G.
Indeed two PSD additions cannot cancel for every G, by the preceding
characterization of the kernel of T. A path with both sides zero remains
zero under the stated SGD law forever. These statements concern externally
specified appends; the law does not discover, choose or evaluate them.

## Strongest attack, evidence boundary, success counterfactual

The strongest simple rival summary is W plus its full initial gradient
response; the two-step example refutes it except for the scalar exception.
The strongest scope objection is that real neural training may include other
state or factor-sensitive operations. That objection limits applicability;
this derivation is only for the stated micro-law. A changed optimizer would
require a new sufficient-state analysis.

If the construction is correct, an arbitrarily wide linear factor block can
be represented in a width-independent state preserving its entire specified
future learning behavior. It does not establish hardware efficiency, learned
mechanism invention, novelty, open-endedness, neural empirical success or a
foundational breakthrough. Its remaining scientific significance depends on
whether this organization and its modifications support a consequential
learning mechanism beyond the equivalence calculation.

Status at this freeze: analytic RECONSTRUCTION; executable claims UNTESTED.
