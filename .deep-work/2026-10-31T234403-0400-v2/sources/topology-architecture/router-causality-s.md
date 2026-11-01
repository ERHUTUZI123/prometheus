# Router causality S — competitive credit, collapse stability and semantic formation

2026-11-01. SOURCE READING + INDEPENDENT DERIVATION, same V2. MAIN=TOPOLOGY human-locked; M0 active NOT PASSED; M3 ultimate. Online USER-PARK; Persistent State standalone USER-PAUSED. No scientific code, checkpoint, diagnostic or experiment was executed for this note. Original papers were read; equations below are derivations, not additional measured outcomes. This supports current implementation, not a preapproval gate.

## Original bodies actually entered

- **Jacobs, Jordan, Nowlan and Hinton1991, Adaptive Mixtures of Local Experts:** [author-Berkeley original scan](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts-bitmap.pdf), printed pp80–83, Eq1.1–1.5 and surrounding explanation. The Toronto route supplied no readable text; the Berkeley body did. It distinguishes squared error of blended outputs from competition between complete expert predictions, then uses mixture likelihood and relative-fit responsibilities. No numerical experiment is imported.
- **Mena et al.2018:** [original arXiv1802.08665](https://arxiv.org/pdf/1802.08665), pp2–3, Eq1–4/Theorem1 and finite-iteration/temperature qualifications. Sinkhorn relaxes an individual matching problem; fixed-score low-temperature convergence needs the stated uniqueness conditions. This does not assert that separate rounding commutes with composition.
- **Shazeer et al.2017:** [original arXiv1701.06538](https://arxiv.org/pdf/1701.06538), §4, Eq6–7, load discussion and diversity footnote. It identifies self-reinforcing expert imbalance and penalizes the coefficient of variation of batch importance, separately treating load. Its sparse selection already imposes per-example sparsity; it reports not needing a separate gate-diversity term there.
- **Fedus, Zoph and Shazeer, Switch Transformers:** [original arXiv2101.03961v3](https://arxiv.org/pdf/2101.03961), printed pp7–8, Eq4–6. Its auxiliary objective couples hard dispatch fractions f_k to mean soft probabilities P_k as alpha*K*sum f_k P_k; gradients pass through P, not hard f. This is a utilization objective, not an operation-equivalence rule.

These are focused original readings, not exhaustive MoE/continual-learning novelty review. The source mechanisms are already established; the remaining campaign contribution must be a consequential computational/learning organization rather than their renaming.

## 1. Negative token–route MI is zero at collapse, including its gradient

Consider one slot and distinct observed(task,token) keys x with weights w_x (the current code weights present UNIQUE keys equally, not token occurrences). Let p_x=softmax(z_x/tau), p_bar=sum w_x p_x. The implemented loss is

    L_MI = sum_x w_x H(p_x) - H(p_bar)
         = -sum_x w_x KL(p_x || p_bar).

At p_x=p for every key, its value is0, its maximum. Its probability-space derivative is

    dL_MI/dp_xk = w_x[log(p_bar,k)-log(p_xk)] = 0.

The logit gradient is zero too. Therefore MI does NOT provide a nonzero first-order semantic gradient at perfectly identical routing. It supplies no correct expert/token correspondence.

Its local stability nevertheless changes. For interior p and perturbations p_x=p+epsilon*d_x, with each sum_k d_xk=0 and d_bar=sum w_x d_x,

    L_MI = -(epsilon^2/2) sum_x w_x sum_k (d_xk-d_bar,k)^2/p_k
           + O(epsilon^3).

Only differences BETWEEN keys acquire negative curvature; common routing changes remain flat. For z_x=z+epsilon*v_x and J_p=diag(p)-p p^T,

    L_MI = -epsilon^2/(2 tau^2)
                 sum_x w_x (v_x-v_bar)^T J_p (v_x-v_bar)
           + O(epsilon^3).

This follows from J_p diag(1/p) J_p=J_p. In the full network this curvature is accessible only through parameter directions that change keys differently. Exact symmetric initialization with no such first-order direction may still be stationary. As routing saturates to one expert, J_p approaches0, so logit-space escape can become weak despite the probability-space expression's1/p factors.

For terminal loss plus lambda*L_MI, a stationary collapsed solution keeps its zero first derivative, while its Hessian changes from H_terminal to H_terminal-lambda*G on accessible contrast directions. It becomes unstable only where this subtraction creates negative curvature; an already flat terminal direction is particularly susceptible. Hence “MI repairs collapse” is a conditional stability/optimization statement, not a theorem that every collapsed router escapes.

The actual MI05 first-stage repair is consistent with this mechanism. It does not alone measure the Hessian, prove which expert parameter first differentiated, or establish a general architecture advantage.

## 2. MI, competitive responsibility and load balancing are not the same rule

A uniform soft router p_x=uniform has perfectly balanced marginal usage but I(key;route)=0. A deterministic balanced partition has the SAME marginal usage but positive MI. Thus marginal balancing alone does not reproduce the current objective. Conversely different keys that implement the same function can maximize MI by using different routes, wasting semantic reuse. MI identifies input keys, not computational meanings.

For a probabilistic competitive expert model with likelihood l_k(y|x), define

    Z=sum_k pi_k(x) l_k(y|x),
    q_k=pi_k(x) l_k(y|x)/Z.

The negative-log-likelihood gate-logit derivative is pi_k-q_k (or divided by tau for temperature-scaled logits). This relative-fit credit is the relevant reconstruction of the Jacobs mechanism. It depends on the observed desired output, while token MI does not. Experts with identical predictions give q=pi, so likelihood alone need not break their symmetry. This explains the possible role of a diversity conditioner without mistaking it for extra task information.

Shazeer's importance penalty and Switch's hard/soft utilization coupling address capacity/dispatch imbalance. They neither certify that experts implement distinct useful operations nor that aliases reuse one implementation. The Switch surrogate is not algebraically identical to MI, and its hard-dispatch boundary requires care; no general optimization theorem is imported from its practical uniform-load motivation.

## 3. Rounding an acquired whole operator is not rounding its factors

For a single bank, row/column Sinkhorn normalizations add row/column constants in log space. Every permutation assignment uses each row/column once, so these constants do not change the matching maximizers. Thus, in exact arithmetic, matching the original bank logits and matching the log-normalized bank have the same maximizer set. Finite Sinkhorn iterations alone do not explain the structural composition mismatch.

The matching operator M is NOT a homomorphism: generally M(log(A B)) differs from M(log A)M(log B). An explicit example uses an8-cycle C and

    A=.6 I+.4 C,
    A^2=.36 I+.48 C+.16 C^2.

A is doubly stochastic. Its unique row/matching maximum is I; A^2's unique maximum is C. Thus the joint hard map is an8-cycle while the repeated independently hardened factor is I. No simulation is required for these inequalities.

Moreover no8-state permutation P satisfies P^2=C: such P commutes with C, whose centralizer is its powers, requiring2k=1 mod8. The observed successful MI model's displayed u route[2,2] and joint8-cycle therefore cannot be interpreted as a faithful repeated hard permutation expert. Actual slot mixtures need not be identical, so this example is a mechanism explanation, not a reconstruction of their exact numeric matrices.

Mena's fixed-score low-temperature result does not repair this automatically. If every factor and route converged with nonzero matching/choice margins, continuous multiplication would recover the product of their hard limits. The observed incompatible component program shows those conditions are not already met. Training may move scores or preserve mixtures as temperature changes.

This motivates native promotion of the MODEL-ACQUIRED full operator and a verified program rewrite. It does not supply teacher maps or show promotion beats ordinary compilation/program learning.

## 4. Terminal-only state AND expert responsibilities for a whole word

Flatten all non-PAD token/slot positions into m=1..M. Let latent state z_m have D values, expert k_m have K values (including skip), pi_mk be the router prior, O_kij its stochastic operator, E_i the start encoding and D_jy the decoder. For one observed terminal target y,

    p(y,z,k|x)=E_(z0) D_(zM,y)
                 product_m pi_(m,km) O_(km,z(m-1),zm).

Forward/backward messages are

    alpha_0(i)=E_i,
    alpha_m(j)=sum_(i,k) alpha_(m-1)(i) pi_mk O_kij,
    beta_M(j)=D_jy,
    beta_(m-1)(i)=sum_(k,j) pi_mk O_kij beta_m(j),
    Z=sum_j alpha_M(j)D_jy.

Provided Z>0, the complete transition responsibility is

    xi_m(i,k,j)=alpha_(m-1)(i) pi_mk O_kij beta_m(j)/Z,
    q_mk=sum_(i,j) xi_m(i,k,j),
    gamma_m(j)=alpha_m(j) beta_m(j)/Z.

These are inferred internal explanations of the ALREADY observed terminal label, not teacher intermediate targets. The future beta term matters: judging an early expert by its immediate output can reject an intermediate computation that a later operation completes.

Exact gate credit is

    d[-log Z]/d z_mk = (pi_mk-q_mk)/tau.

For freely row-softmaxed operator logits U_kij, put N_kij=sum_m xi_m(i,k,j) and N_ki=sum_j N_kij. Then

    d[-log Z]/d U_kij = (N_ki O_kij-N_kij)/tau.

For Sinkhorn banks the independent-row formula is NOT valid: use the responsibility gradient with respect to log O and differentiate through the actual coupled normalization. Encoder/decoder responsibilities similarly follow gamma_0/gamma_M.

**Information answer: this is exactly the terminal-likelihood gradient already computed by correct BPTT.** Explicit forward/backward responsibility code does not add information or supervision. A detached-posterior expected-complete-log-likelihood gradient at the SAME parameters equals BPTT. Multiple damped EM/block updates with posterior held fixed can change optimization/preconditioning, but do not reveal new data. Differentiating through a self-generated posterior target without detaching adds other terms and is no longer this identity. A hard/Viterbi path is another approximation and can lock an incorrect early program.

Implementation can store alpha/beta in O(MD) and accumulate counts in O(KD^2), rather than materializing every batch/time/state/expert tensor. It still pays the corresponding contractions and numerical normalization. Zero terminal probability requires a disclosed soft-support repair; inventing an intermediate correct state is not a remedy.

## 5. Semantic merge/split without repelling aliases

Apply these rules to acquired WHOLE executable operators/programs first, since soft components need not themselves be coherent primitives.

**Conservative merge:** identical learned full-state maps share an ID independent of input token/task. This is equality of model-acquired maps, not a claim that unknown teacher functions are globally equal. For a merge valid only on current old reachable behavior, construct the proposed rewrite and run the existing finite old/new decoded-state relation. Include slot-intermediate states if certifying component replacements; a whole-token relation avoids that omission. Preserve terminal validation/error contracts. Matching a few observed outputs only proposes a merge; it does not prove all-word equivalence.

**Split proposal from functional conflict:** group actual posterior edge counts by candidate context c and incoming state i. Let r_ci(j)=N_cij/N_ci and r_i be their pooled row distribution. The gain in the complete-data multinomial fitting surrogate from separate rows rather than one pooled row is exactly

    Delta Q=sum_(c,i) N_ci KL(r_ci || r_i)
           =sum_i N_i JS_(weights N_ci/N_i)({r_ci}).

Identical behavior distributions produce0 regardless of token identity. This can rank a bounded split proposal when one expert is serving incompatible conditional transitions. It is NOT proof of true semantic distinctness: posteriors depend on the current latent encoding and may themselves be wrong. Accept a fitted split only with actual terminal validation improvement/resource justification and retained-old graph contracts. No new route labels are granted.

Use a finite cost tradeoff such as terminal loss plus active-library/program/certificate cost. Temporary MI can bootstrap underused candidates, then yield to behavior-based equality/conflict. A permanent token-separation reward is unnecessary once aliases can share a proven executable operator. New splits/materializations/merges retain provenance and actual storage/search/maintenance cost.

Ordinary MoE, compact program learning and graph-equivalence constrained implementations must receive this same information and legitimate formation opportunity. Transformer remains the primary empirical baseline; those mechanisms are additional causal explanations, not replacements for a strong Transformer comparison.

## Consequence for the campaign

The MI05 run demonstrates a useful acquisition repair in its actual limited first-stage scope. It does not make MI, Sinkhorn, responsibility inference or native compilation novel; none alone establishes M0 or M3. The concrete architectural opportunity is to turn an acquired whole behavior into an executable reusable semantic unit, preserve old complete computations during later shared change, and merge/split by behavior rather than token names. Actual promotion implementation and its later runs require result-based supervision; this note neither approves them in advance nor supplies unseen outcomes.
