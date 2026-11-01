# Router repair: what the experiments changed, and the next executable mechanism

2026-11-01. Same V2, MAIN=TOPOLOGY human_only; M0 active, M3 ultimate. AGENT RECONSTRUCTION + IMPLEMENTABLE DEVELOPMENT, not a novelty or acceptance claim. The human paused new execution; root stopped MI05, preserved partial evidence, then the human explicitly resumed autonomous iteration. This artifact reasons from existing records, not newly fabricated experiments. Transformer remains the primary architecture baseline.

## 1. Evidence that constrains the explanation

Same N8/seed0/data/800 updates/batch64/lr.01: original topology hard IID=.513671875, soft=.548828125; Transformer and Replay=.921875. Train/IID dataset hashes match. Firststage replay has no old data. Transformer has40280 parameters versus topology1102, so this is equal information/update budget, not equal capacity or total compute. All three initially fail long-word evaluation (roughly .10–.13).

MI05 was launched before the pause and interrupted at step700. Complete curves exist only through600; there is no stage_end or supervisor completion receipt. From100 onward joint hard valid/IID/OOD/length16=1.0, and softIID=1.0 from200. Its updated source includes a temperature-guard fix, but no reference exists in this firststage: that guard cannot account for these firststage changes. This is a strong exploratory contrast, not independent seed replication or a complete continual-learning result. Earlier log/prob ablation and N4 joint rounding remain valid in their exact scopes.

Direct paths: `../../probes/topology-architecture/runs/gpu-firststage-n8-joint-lr01-seed0-v2/learn/records.jsonl`, `.../gpu-firststage-n8-transformers-lr01-seed0/learn/records.jsonl`, `.../gpu-firststage-n8-router-mi05-seed0/learn/records.jsonl`. Do not upgrade the interrupted summary to completed.

## 2. Capacity, credit, and formation are different questions

N8 stage0 is representable by the existing six-bank/two-slot model: learn one increment permutation and one rotation permutation, use the other slot as identity. This is a capacity witness, not a teacher route supplied to training. The empirical failure therefore does not establish insufficient state count. Stage2 hidden history may need additional distinguishable states; that is a separate later issue.

Write one soft slot as A_a=sum_k p_ak P_k and let G_a=dL/dA_a. Ignoring the independent skip for a moment,

    dL/dl_ak = p_ak <G_a, P_k-A_a>,
    dL/dP_k = sum_a p_ak G_a.

If all participating P_k coincide, routing gradients vanish, even when different symbols need different operations. If several symbols share p, their operation gradients are mixed into the same banks. Small random differences can be starved rather than amplified. This is a concrete cooperative-learning degeneracy; it is not proof that the actual trajectory remained exactly symmetric. The recorded original u/v do end with the same hard route and same joint map.

The implemented auxiliary term is

    L_div = mean_a H(p_a) - H(mean_a p_a) = -I(A;K).

For p_a=pbar+eps*v_a, with sum_a v_a=0 and sum_k v_ak=0, a Taylor expansion gives

    I(A;K) = eps²/(2m) sum_(a,k) v_ak²/pbar_k + O(eps³).

Thus exact collapse is still a stationary point, but -I supplies negative curvature along token-differentiation directions: existing perturbations can grow. It encourages both decisive per-symbol routes and diverse marginal use. Uniform utilization alone would allow every symbol to use the same uniform mixture.

This does not explain which experts should perform which computations. The auxiliary objective is independent of output semantics. Distinct aliases of the same operation can be unnecessarily separated. Expert identities can also permute without any computational change. Therefore the result first supports a trainability/operation-differentiation repair, not a topology-derived acquisition advantage.

## 3. A stronger finding: 100% joint behavior is not the stored native program

At complete MI05 step600 the saved encoder/decoder are identity, the joint increment is U=[1,2,3,4,5,6,7,0], while the displayed component call is [2,2]. Bank2 has zero canonical braid length, i.e. its hardened permutation is identity. More generally no permutation P of these same eight states satisfies P²=U:

    sign(P²)=+1, but sign(U)=(-1)^7=-1.

Consequently the joint compiler is creating an executable whole operation that the displayed component program cannot execute. Rounding is not a homomorphism:

    assignment(A B) need not equal assignment(A) assignment(B).

This is not merely a few incorrectly rounded entries. The call skeleton itself has an algebraic obstruction at D=8. With a larger latent cover and different encoding this argument no longer applies; it is not a general impossibility claim.

The pre-promotion hard banks at this checkpoint are identity except the rotation. Their generated subgroup contains only even permutations, so the odd increment is not obtainable by any word in that particular hardened library. Its information does exist in the terminal-trained SOFT weights and joint operator. A valid new operation must obtain it there or from new observations; it cannot be recovered from the old hardened library alone.

## 4. Selected development: learn a behavior, materialize an operation, then protect native use

Keep the successful acquisition model and its learned state encoder, operators, router and terminal loss. Do not replace it wholesale after one repair. Add a state-changing learning/compiler event, driven by the model's actually acquired full operator.

State: shared neural weights theta; latent state q in1..D; native bank {Pi_k}; trainable realizations W_k; trainable call head; old learned encoder/decoder/transition graphs; materialization provenance. No teacher operator or correct route is an input.

1. Terminal training forms soft effective A_a=product_slots(sum_k p_ask soft(W_k)). During initial formation use the declared temporary diversity term.
2. Form candidate C_a=assignment(log A_a) from trained parameters. Evaluate it separately from NATIVE component execution. Never call joint accuracy native accuracy.
3. If the candidate meets the declared validation admission but native calls disagree, reuse an existing bank with that complete learned class or append a new bank initialized to C_a with a declared finite score margin. This is operation birth from learned behavior, not inventing its answer from arithmetic.
4. Rewrite the shared call head to call that learned whole operation plus skip. Targets come only from the current acquired graph; old contexts use CURRENT NATIVE graphs, never an unprotected soft-old proposal. A CPU-double least-squares compiler uses learned hidden features, with rank/conditioning/output-margin checks. Failure rejects the compiler event rather than silently substituting a supplied program.
5. Check the before/after decoded relation: new native execution must match the learned candidate on its whole generator graph, and all old native graphs must remain equivalent. An exact finite closure check covers all old words, not only observed lengths. Then admit the operation and continue shared learning. All parameters remain trainable; optimizer reconstruction and materialization costs are charged.
6. After successful formation, stop forcing token-to-expert diversity for that acquired context. Deduplicate identical learned whole actions; preserve distinct composition order. Later compression may search for a cheaper existing program, but cannot replace actual native behavior by a favorable soft average.

This event is a concrete change in the executable library and the acquisition process. It still uses ordinary assignment, program compilation, shared weights and equivalence checks; those are legitimate mechanisms and strong conventional explanations, not evidence of historical novelty. A program-learning system receiving the same learned operators and budget can implement the same compiler. Transformer remains the main empirical baseline, with replay/adapters retained for continual tests.

## 5. Terminal credit can become explicit without inventing supervision

For a flattened token/slot sequence r, let A_r=sum_k p_rk P_k. Forward alpha_r=alpha_(r-1) A_r and backward beta_(r-1)=A_r beta_r, with beta_end(i)=decoder[i,y]. Then

    xi_r(k,i,j)=alpha_(r-1)(i) p_rk P_k[i,j] beta_r(j) / Pr(y),
    gamma_r(k)=sum_(i,j) xi_r(k,i,j).

Gamma is the model's posterior responsibility given actual TERMINAL feedback, not an observed teacher trace. The exact gate gradient is p_r-gamma_r (with the relevant temperature factor). Merely rewriting BPTT this way adds no information and is not a new credit algorithm. A genuinely different blocked update must specify its changed optimization schedule and count all posterior/inner-solver work. It also cannot identify unused or observationally indistinguishable operations for free.

For semantic consolidation, compare learned full actions on reachable states, not symbol identity or Euclidean token distance. Exact equality permits immediate class reuse. Approximate equality on finite samples is insufficient for all-word merging: two transitions may agree today and differ after a suffix. Use the same closed decoded relation as the safety condition; uncertain merges stay provisional. This permits aliases to share while retaining noncommuting actions. Whether such formation-plus-protection improves later learning remains the central experiment, not an established consequence of these formulas.

## 6. Where topology acts and what it does not explain

Each acquired permutation has a braid/monodromy realization and a discrete semantic class. A margin-preserving realization change can leave that class invariant. Shared encoder/calls must ALSO satisfy the old reachable relation: protecting a gate alone does not preserve its task binding. The actual prototype checks native learned graphs, including the encoder and decoder.

This is software permutation/covering semantics, not a six-Majorana physical experiment or hardware-energy result. The same finite certificate is an ordinary graph-equivalence constraint; any unique topology benefit remains to be demonstrated against that explanation. The diversity repair itself invokes no topological protection. Its firststage success therefore cannot be attributed to protection.

Future hidden-state refinement must protect only old reachable behavior, not freeze unvisited workspace. Extra sheets, operators, stored graphs and discovery cost must be explicit. No expansion is being claimed as autonomous or scalable merely by increasing D.

## 7. Immediate falsifiable uncertainty and resource boundary

The next new experiment must ask whether a learned full operation can be materialized into a faithful native library, reused in longer compositions, and retained during later terminal-trained acquisition. It must separately report native, joint-proposal and soft performance, library growth, compiler failures, optimizer resets, protection rejects, total training/maintenance/storage/inference costs. Same Transformer task data and training budget remain the primary comparison. This is a consequential continuation of the same mechanism, not more seeds merely to repeat the already observed joint100%.

Hard limits remain external900s/internal850s (or explicitly frozen <=1800s), D<=64, native banks<=24, no generated-code networking or credential access, existing GPU30% allocator budget, file<=100MiB. No paid route or permission expansion. Missing partial receipts stay missing. M0 not passed; no campaign completion inferred from this development artifact.

## Original sources actually entered; limited reading, not a landscape review

- Mena et al., ICLR2018, https://arxiv.org/pdf/1802.08665 : pp2–3 Eqs(1)–(5), matching/Sinkhorn parameterization and temperature/finite-iteration caveats; pp4–5 beginning of probabilistic construction. These support using and questioning a relaxed matching interface, not a continual-topology advantage. Remainder unread this segment.
- Shazeer et al., ICLR2017, https://arxiv.org/pdf/1701.06538 : pp3–5 Eqs(1)–(7), soft/noisy gating and self-reinforcing utilization imbalance. This directly supplies an existing-method explanation for a routing repair. No large-model benchmark conclusions adopted. Later empirical sections not reviewed here.
- Rabiner1989, https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf : printed pp262–265, forward/backward, local-vs-path decoding and Eqs(36)–(40) reestimation. Our terminal-conditioned operator/expert formula is a reconstruction from that framework, not its quoted theorem for this architecture. Other chapters unread.
- Krishnamurthy et al.,2023, https://arxiv.org/pdf/2302.14703 : pp1–3 mechanism/problem statement and related work, pp7–8 utilization/regularization comparisons. It explicitly distinguishes specialization from using all experts. Its Euclidean sample similarity is not the semantic closed-relation rule proposed here. Section6 full mathematics not yet read, so no claim of a literature gap relative to it.
- Jacobs et al.1991 author-hosted PDF located, but root text extraction failed; do not claim full original reading from its abstract. Independent source analysis is separate.


## 2026-11-01 · Method integration: evidence boundaries retained

Canonical method is water/method.md only. Its prior complete wording is frozen in probes/topology-architecture/versions/native-v6-method-before-continuous-integration.md. Moving repeated limitations out of the method changes no scientific finding.

Architecture history: protocol-v1 already specifies a software quotient, matching chambers and old/new reachable graph relations; there is no recorded failed continuous realizer experiment. Native-v5 installs self-learned whole operations; v6 adds observed-history-driven workspace growth. Exact monodromy is locally constant away from degeneracies, so exact terminal semantic gradient alone cannot change its class. This is an unresolved interface, not a proved impossibility.

Retained boundaries: MI routing repair is ordinary expert specialization, joint matching need not factor as native calls, compiled primitives/graph equivalence are ordinary tools, and topo_off retains shared materialization/growth preservation. The step/temperature guards had no rejections in the recorded v6 run. The present permutation quotient discards pure braids; a canonical braid selector is not a homomorphism. A continuous realizer does not by itself establish a new acquisition rule, physical protection, or M0 advantage. Parameter growth, actual training cost, and repeated semantic checks must remain charged.

Actual v6b/error evidence: probes/topology-architecture/baseline-and-error-sync-manifest-20261101T205356Z.json; resource-extraction-v6b-20261101.json. Native stage2 has111/4096 training errors and one observed-history alias; the repaired neural checkpoint is not yet claimed at this entry.
