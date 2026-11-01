# Recipient evidence from externally labeled composed programs

RECONSTRUCTION / AGENT PROPOSAL. Technical contribution to root's ongoing acquisition pilot. No code/execution, new source claims or campaign-completion claim. The purpose is to supply a compact, lawful recipient-history interface when only whole-program targets are observed.

## Judgment

Use the program residual multiplied by the recipient's actual downstream sensitivity, together with a finite-response curvature term. These quantities are derived from a paid **external program label**, the known program graph and current executed cell outputs; they require no local cell teacher targets. They can forecast finite transfer rather than just gradient alignment. Their compactness is exact only for a stated response basis/current checkpoint. Stale summaries or arbitrary off-probe nonlinear behavior require additional retained evidence or an explicitly approximate predictor.

## 1. Setting and the exact finite-effect law

Cell j has independent persistent weights and output

    v_j(x)=f_{theta_j}(x)=tanh(sum_k theta_jk phi_k(x)).

An externally evaluated program has scalar output Y(x)=P(v_1(x),...,v_m(x)), with m=2 or 3, and observes only (x,y), where y is the whole-program target. The paid program batch B={(x_n,y_n)} contains no target for any v_j. Let

    e_n=P(v(x_n))-y_n,
    c_jn=partial P / partial v_j evaluated at current v(x_n).

For squared external loss ell=(Y-y)^2/2, let a source experience induce an actually realized finite response change d_j(x)=v_j^+(x)-v_j(x) after bounded distillation into recipient j. Define the corresponding actual program output change

    D_j(x)=P(v(x)+e_j d_j(x))-P(v(x)).

Here e_j denotes the coordinate unit vector, not the residual. With all other physical owners fixed, the external loss improvement is exactly

    U_B(j,d)=mean_n[-e_n D_j(x_n)-D_j(x_n)^2/2].          (1)

This follows by expanding (e+D)^2-e^2. It is exact for finite changes and any known consumer P, including saturating/nonlinear consumers. It is empirical utility on the paid batch, not automatically expected future utility.

### Addition/product programs with one occurrence of each cell

If P is built from additions and products and cell j occurs at most once, then P is affine in v_j conditional on the other cell outputs. Thus D_j=c_j d_j exactly, not approximately. Equation (1) becomes

    U_B(j,d)=mean_n[-e_n c_jn d_j(x_n)
                        - c_jn^2 d_j(x_n)^2/2].         (2)

The first term is signed external credit; the second measures finite overshoot cost. The residual and sensitivity are not a local teacher value. In particular, a small c_j means a locally large cell change can have little program effect, while its sign can reverse whether copying a source change helps.

Examples: for P=v_1 v_2+v_3, (c_1,c_2,c_3)=(v_2,v_1,1). For P=(v_1+v_2)v_3, the sensitivities are (v_3,v_3,v_1+v_2). All are computed from the executed graph and current outputs using one external label per example, rather than one label per cell.

If the same physical cell occurs twice, this affine condition need not hold. For P=v_j^2, D_j=2v_j d_j+d_j^2. For a nonlinear outer consumer, one may evaluate the finite consumer secant directly. Resolving aliases before differentiating/evaluating is necessary: all occurrences of one owner change together. It is not lawful to perturb one occurrence while freezing its alias.

## 2. A compact recipient-history measurement

Choose a shared, position-ID-free response basis b(x) in R^r, using common semantic input coordinates. This can be the declared phi features, a fixed random-feature bank, or a small public-probe interpolation basis. Choosing b does not assert that arbitrary tanh updates lie in its span.

From the paid external batch, store for each recipient

    m_j=mean_n[e_n c_jn b(x_n)],
    H_j=mean_n[c_jn^2 b(x_n)b(x_n)^T].                  (3)

H_j is positive semidefinite. A compact descriptor is

    h_j=(m_j,H_j, current cell/consumer probe responses,
         context, sample count, uncertainty/drift indicators).

The storage is r+r(r+1)/2 scalars per recipient, plus declared context/probe state. All recipients in one 2–3-cell program can derive their descriptors from the SAME paid external labels; cell evaluations and derivative/graph work are still counted. This is legitimate reuse of observed labels, not observation of fresh outcome labels for every possible future transfer.

Suppose a proposed realized response has d_j(x)=b(x)^T alpha. Equation (2) gives the exact paid-batch forecast

    U_B(j,alpha)=-m_j^T alpha - alpha^T H_j alpha/2.     (4)

Thus m_j alone is insufficient for a finite change, and one scalar mean residual is insufficient even at first order: the sign/magnitude of c_j and the correlation of d_j with x matter. Omitting x-resolved basis information can cancel useful and harmful regions.

### Obtaining alpha without local targets

Let P_public=(p_1,...,p_q) be the public unlabeled probes and B_P its q-by-r basis matrix. After actual recipient distillation, evaluate the realized response vector d_j(P_public). A fixed ridge projection

    alpha=(B_P^T B_P+lambda I)^(-1)B_P^T d_j(P_public)

produces the response coordinate. This requires no local teacher. The source packet's intended displacement cannot replace the realized d_j: recipient saturation, optimization error and representation limits can make them different. The coefficient and its fit residual are descriptor inputs, not a hidden success label.

Source-only probe values also do not determine a source response at arbitrary other inputs. If the current episode retains the actual source before/after model snapshots, extra source evaluations can be made and charged. If those states have been deleted and only probe values retained, arbitrary off-probe values cannot be recovered. Either restrict transport to the declared interpolated packet or acknowledge this uncertainty.

## 3. Approximation error and a useful repair

A generic finite tanh-cell response is not exactly linear in the original phi features. Write the true response d=b^T alpha+epsilon and let d_tilde=b^T alpha. Under the affine-consumer condition, the error of equation (4) on the SAME paid batch obeys the explicit bound

    |U_B(d)-U_B(d_tilde)|
      <= mean[ (|e c_j|+c_j^2 |d_tilde|)|epsilon|
                  + c_j^2 epsilon^2/2 ].              (5)

This is just the triangle inequality applied to equation (2), not a distributional generalization result. The residual can be measured on retained paid inputs if actual d is available there. A small fit residual only at public probes does not prove a small residual elsewhere.

There are two practical repairs, with different claims:

* **Current-checkpoint exact evidence:** retain a very small external-labeled coreset B (input and whole-program label), recompute current e,c and actual candidate d on it, and use equation (1)/(2) directly. This is bounded replay with explicit bytes/forward work. It is a strong baseline as well as a usable mechanism ingredient.
* **Compressed approximate history:** retain only m,H plus probe/drift summaries and learn a residual correction/uncertainty predictor from later sparse audits. This saves memory but cannot exactly revalue arbitrary future nonlinear states. New observations revise it; discarded external labels are not silently restored.

The second repair addresses a genuine staleness problem. Updating another cell changes both current program residual e and consumer sensitivity c_j. Old m,H therefore cease to be exact at the new checkpoint. Recomputing c_j on unlabeled probes is possible, but recomputing label-dependent e at old inputs is not possible after deleting the necessary y information. Timestamps/versions alone do not repair missing data. The pilot can initially freeze the checkpoint between measurement and transfer, then separately test bounded coreset versus compressed-history revision.

## 4. Worked program process: one external label distinguishes recipients

At one input, suppose independent cells currently output

    (v_1,v_2,v_3)=(0.5,-0.6,-0.3),
    P=v_1 v_2+v_3=-0.6,
    external observed target y=-0.2.

The paid external residual is e=-0.4 and sensitivities are (-0.6,0.5,1). No local target for the three cells has been supplied.

A source packet proposes a positive response displacement 0.8. Distillation can realize this for recipients 2 and 3, because their proposed outputs become 0.2 and 0.5. For recipient 1, an intended output 1.3 is outside tanh's range; use the actually realized displacement, for example 0.4 if its final output is 0.9. Its failed realization is observable without any local teacher.

Equation (2) gives:

| Recipient | Realized d | First-order predicted improvement -e c_j d | Exact finite improvement |
| --- | --- | --- | --- |
| 1 | 0.4 | -0.096 | -0.1248 |
| 2 | 0.8 | 0.16 | 0.08 |
| 3 | 0.8 | 0.32 | 0 |

Recipient 2 exactly reaches the external target: (0.5)(0.2)-0.3=-0.2. Recipient 3 overshoots to 0.2, leaving the same squared loss as before. Recipient 1 moves in the wrong program direction. A first-order rule prefers recipient 3 over 2, whereas the finite-effect history identifies overshoot. This is not an acquired scientific advantage by itself; a quadratic/finite-validation optimizer given the same data computes it too.

The process is lawful: pay for the external (x,y); execute cells; derive e,c and h; receive source finite response; distill and measure the actual d; forecast the recipient effect; then test prediction on a separate later audit batch. The single paid point does not establish which recipient improves over the input distribution. The example illustrates the interface, not the endpoint task.

## 5. Multiple recipients and new compositions

Do not sum equation (2) if a source update is sent to several cells simultaneously. For P=v_1 v_2+v_3,

    D=v_2 d_1+v_1 d_2+d_3+d_1 d_2.

The cross term can reverse conclusions from individual routes. The clean initial pilot sends one transfer to one physical recipient per event. A multi-recipient extension must evaluate/approximate the joint finite consumer effect with cross-response features and charge the added inference. Independence of parameter owners does not imply independence of their downstream effects.

New programs can reuse the same measurement law because c_j comes from their current graph rather than an absolute cell ID. Addition/product role changes modify c_j automatically. New positions can derive h from a small declared external program batch, using the same shared response basis, and obtain a prediction from the shared association learner. This is an actual observable route to generalization; a new position with no informative external evidence may remain ambiguous.

There is a sharp insufficiency: two environments can provide identical observed program histories yet assign different labels off those histories. No compact h can certify opposite future recipient utilities in those worlds. Also, at v_1=v_2=0 in P=v_1 v_2, both single-recipient sensitivities vanish. One-cell transfers then have zero program effect, while a joint transfer can matter. History derived only from single-cell first-order evidence cannot discover the joint mechanism without a joint exploration/proposal channel. The repair is a paid joint-effect probe or an explicit joint model, not inventing local target labels.

## 6. Relation to adaptive gradient kernels and the core test

The ordinary externally supervised parameter gradient is

    g_j=mean[e c_j (1-v_j^2) phi(x)].

For a small parameter update delta_theta_j, the first-order improvement is -g_j^T delta_theta_j. A source/recipient learned coupling acting on these gradients is an adaptive gradient kernel or learned optimizer. The m term in equation (4) is the same external credit expressed in response coordinates. It is not a separate principle. H captures a finite-output correction exactly for the conditional affine consumer; a second-order or direct finite-effect optimizer can use it as well.

The core technical comparison should therefore give every method the SAME small external labeled program batch and forbid local cell targets. Compare consumer-blind descriptors, m-only adaptive kernels, m+H finite-response forecasts, current-cohort exact finite validation, and a contextual learned optimizer receiving the same histories/source packet. All methods may use the labels for direct ordinary program learning. Include truly new positions and addition/product compositions, and test forecasts on fresh external data rather than the history batch.

One useful causal ablation keeps the current forward cell states and observed external history fixed while replacing only the acquired association predictor by the analytic adaptive-kernel rule or a fresh same-information predictor. If the candidate wins only against a consumer-blind or first-order-only baseline, the demonstrated effect is known nonlinear credit/curvature handling, not acquisition of a new computational organization. The prospective contribution remains learning and revising a reusable recipient relation over continuing experiences and new composed programs under the full information budget.

Root owns implementation, resource accounting and experimental decisions. This derivation supplies the recipient evidence path and its limits; it does not conclude the MAIN campaign.
