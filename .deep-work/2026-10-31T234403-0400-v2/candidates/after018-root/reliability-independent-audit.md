# Independent constructive audit of the reliability candidate

RECONSTRUCTION / ATTACK / PROPOSED REPAIR. Same V2 campaign. This is an independent read of the actual after018-unfamiliar/candidate-v2.md and its source-v2.md, not a MAIN decision. Only this new audit file was written. No experiment, circuit training, package installation, model/resource change or source-code execution was performed. Effective provider/model/mode are unavailable.

Actual input hashes checked after reading:

- candidate-v2.md: f935d82811550f876a2fda8572fbcbb740d59a21067f8590ef0c6410183ba07f
- source-v2.md: 5141d50a273736658bbbd32589397fee2c10035506c97d25ce082d80869b7410

## Judgment

**The seven-bit local contract and finite-depth stochastic bound are correct under the stated fault locations and budgets. They establish a real noisy-computation invariant, conditional on acquiring the specified circuits. The current code-growth law, however, is not yet the claimed operation-induced guard-demand law. In the worked case a researcher-chosen q=3 determines sevenfold repetition before operation behavior enters.**

This is repairable as an explicit algorithmic coupling: let actual encoded-operation error amplification create the guard demand, then rerun representation formation and physically faulty restoration. A worked projection-versus-AND comparison below demonstrates that the resulting demand can depend on operation behavior even at the same nominal arity. But an ordinary same-information compiler obtains the seven-bit witness directly, and can also make that five-versus-seven choice. Thus the important remaining question is economical formation of computation-compatible codes on large structured interfaces, not correctness of a repetition gadget or the broad existence of learned error correction.

Known coding, majority and threshold operations remain legitimate building blocks. Their familiarity does not invalidate the proposed partial mechanism. It identifies which consequence must come from acquisition and representation formation rather than the supplied reliability substrate.

## 1. Local contract: what the induction actually needs

Let E(s)=c_s and V_s=B_t(c_s). An admitted interface should satisfy disjointness of semantic basins, for example

    min_{s!=v} d_H(c_s,c_v) > 2t.                         (1)

Then a block in V_s denotes at most one incumbent value. The candidate's final seven-bit/t=1 interface satisfies (1). Early growth states need not; they must remain unadmitted construction states, rather than already reliable representations.

The operation contract is

    h_i in V_si, local allowed faults
       => R_out^xi(U_a^eta(h_1,...,h_k)) in V_Fa(s_1,...,s_k). (2)

With valid initial blocks and (2) for every invoked operation, topological induction proves the basin invariant on a finite DAG. A finite recurrent execution follows by time induction. Fan-out creates correlated copies of an erroneous block, but (2) quantifies over every valid input tuple, including equal/correlated inputs, so independence of incoming errors is unnecessary. Nothing in this induction proves that training established (2), that the incumbent is semantically correct, or that local stochastic budgets hold forever.

Fault assumptions matter concretely: the witness permits neuron-output flips, with reliable weights, addressing, timing and wires outside those locations. It needs threshold gates with fan-in seven available as individual physical neurons. If the physical fan-in cap were two, or a majority gate had to be decomposed into multiple noisy layers, the stated one-stage R proof would need a new fault analysis. This is a scope condition, not a reason to forbid a legitimate fan-in-seven substrate.

## 2. Seven-bit proof: checked and slightly sharpened

For each logical input block, let E_i be its set of erroneous positions, |E_i|<=1. For a coordinatewise deterministic Boolean operation, an output position can differ from the clean Boolean result only where at least one argument is wrong. A U output fault can add one further erroneous position. Therefore

    NOT: pre-restoration errors <= 1+1 = 2,
    AND: pre-restoration errors <= |E_1 union E_2|+1 <= 3. (3)

The AND bound is attained: clean inputs (1,1), errors at distinct positions in the two input blocks, and a U output flip at a third position produce three wrong coordinates. The fact that some other input tuples have a smaller bound does not defeat the worst-case contract.

All seven nonfaulty majority-of-seven neurons see the same y. With at most three wrong coordinates they all return the clean Boolean value. If at most one final R output flips, the output differs from the correct repetition code in at most one bit. Thus the candidate's physical R, not an invisible clean decoder, renews t=1.

The shared majority inputs do not generate an independence argument; this is a deterministic count. A later operation can use the result repeatedly, including feeding the same block into both AND inputs, without increasing the declared input radius.

A useful necessity check also supports the candidate's use of exact clean restoration targets. Suppose an n-bit R must tolerate up to e_R adversarial flips at its final output and still land in B_t(c_s), with t<n. If its fault-free output is at distance d_0 from c_s, the worst such flip pattern reaches distance min(n,d_0+e_R). Thus

    d_0 <= t-e_R.                                        (4)

In the witness t=e_R=1, every clean R output must equal c_s exactly. Approximate restoration to an arbitrary member of B_1(c_s) is insufficient before the allowed R fault. Hence overlapping input restoration demands really are incompatible under this robust contract. For a generic loss_t objective without the corresponding final-output fault requirement, overlap of input q-balls alone would not prove incompatible outputs if the target t-balls themselves overlapped.

## 3. Stochastic bound and complete witness costs

For seven independently flipping neuron outputs with probability p, the event of at least two flips is contained in the union of the 21 faulty-pair events. Consequently its probability is at most 21p^2. There are two seven-neuron stages per invocation. A union bound over H invocations yields

    P(any declared local budget is violated) <= min(1,42 H p^2). (5)

On the complementary event, the deterministic invariant proves correct encoded execution. The bound is valid even though logical/activation errors become correlated across layers; independence is needed for each physical pair's probability p^2, not to treat successive logical outputs as independent. It is a sufficient finite-depth bound, not a nonzero-noise/unbounded-depth reliability theorem.

The final output remains an encoded block. A required last scalar neuron with independent flip probability p still supplies its own error floor. Valid initial input blocks are another premise: if raw inputs need a noisy encoder, its circuit, fault events and work must be added rather than hidden in (5).

The exact per-invocation witness costs are:

| Operation | U neurons | R neurons | U input edges | U-to-R edges | Live output bits |
|---|---:|---:|---:|---:|---:|
| NOT | 7 | 7 | 7 | 49 | 7 |
| AND | 7 | 7 | 14 | 49 | 7 |

With one stored number per weight and threshold, the NOT gadget has 70 such circuit parameters and AND has 77, before graph handles/scheduling: 14 or 21 in U, plus 56 in R. Constants can be stored/shared more cheaply if the implementation permits it, equally for both methods. Block storage is seven bits per live logical wire. Gate fan-in, synchronization and trusted routing are part of the substrate account.

Acquisition additionally pays for the incumbent/reference parameters and evaluations, encountered-state bank, conflict graph, cut search, encoder fits, every U/R training and failed proposal, fault search or enumeration, code-version migration, and preservation/re-encoding of live state. A fixed per-stage one-fault budget is not a fixed physical failure probability when node count changes: more nodes create more possible faulty pairs. Bit growth cannot be presumed to monotonically improve a stochastic end-to-end cost/reliability score.

## 4. The score-function update is valid with explicit conditions

For a frozen code/graph and a training context e (clean tuple, permitted input corruption and exogenous fault sampling law), let b_j denote the pre-fault Bernoulli decision of stochastic node j. Its conditional inputs include actual upstream faults/decisions. The trajectory likelihood factors as

    p_theta(tau|e) = p_exogenous(e) product_j
                    p_theta(b_j | actual_inputs_j).

If fault probabilities do not depend on theta and the clean target code is held fixed, differentiation of the expected trial loss gives

    grad_theta E[L] = E[L sum_j grad_theta log p_theta(b_j|inputs_j)], (6)

plus any explicit differentiable dependence of L on theta that is not mediated by the sampled trajectory. For fixed graph execution count/storage and frozen code targets, those discrete quantities have no missing continuous derivative. The candidate's formula is therefore a valid likelihood-ratio estimator for the stated stochastic objective, not a proof of convergence or of the deterministic freeze contract.

The baseline must be chosen correctly. A baseline measurable before the trial, or an appropriate node-specific baseline measurable before that node's sampled decision, has zero expected score contribution. An arbitrary scalar computed from the completed current trajectory and reused as a supposedly constant baseline can bias (6). State this condition instead of leaving 'baseline' unrestricted.

Also distinguish mathematical continuous logit parameters during acquisition from the later finite-precision threshold circuit. If deterministic weight quantization is inside the acquisition map, an ordinary derivative through that quantizer is not made exact by the score trick. The proposal may use continuous differentiable logit parameters during trusted acquisition and round/freeze afterward, or declare its surrogate/discrete parameter proposals. Its subsequent frozen-circuit checks remain necessary. Shared parameters and noisy upstream inputs are otherwise handled by including every relevant sampled node score.

## 5. Main mismatch: fixed-q separation is not yet operation-driven growth

The current representation deficit is

    D_code = sum_sv w_sv max(0,2q+1-d(c_s,c_v)).           (7)

On a fixed encountered bank and for a fixed q, once every relevant pair has d>=2q+1, every summand is zero. Increasing failure-derived w_sv cannot create any new positive deficit. Therefore an encoded operation can amplify an allowed incoming error beyond q, fail restoration, and still supply no code-growth gradient/trigger through (7). Its failure only feeds circuit retraining unless another rule changes q, the conflict definition or the bank of values/pairs.

In the two-value example q=3 is supplied before learning. The maximum possible distance with n bits is n, so the separation target alone forces at least seven bits. Every positive cut puts the two values on opposite sides. Repeated additions reproduce repetition/complemented repetition independently of which Boolean operation will run. The constructive trajectory is valid, but it is fixed guard-radius coding. Operation trials help train U/R; they do not select the redundancy level in this witness.

For a larger state set, the candidate says failures change pair weights, but the rule for mapping a failed off-code output to a specific incompatible pair is not yet specified. Different failure frequencies can change a finite cut, but that is weaker than a demonstrated law linking an operation's actual amplification to a required guard radius. A codebook may be well separated and still be computationally incompatible with the learned U under the declared input/fault set.

## 6. A defensible repair, with a worked operation-sensitive consequence

Make the missing demand explicit, retaining the incumbent reference and finite-resource boundary. For a frozen proposed code and U define

    A_a(t,e_U) = max d_H(U_a^eta(h_1,...,h_k), c_Fa(s_1,...,s_k)), (8)

where the maximum is over the declared tuple domain, h_i in their admitted t-balls, and allowed U fault masks. A finite encountered bank gives only an observed/checked lower bound on the global maximum. Full exact evaluation is a finite but potentially huge paid operation; no statistical sample maximum is promoted to the universal A_a.

Let the output-interface guard demand be at least the maximum checked A_a over its producing operations. When that demand increases, (7)'s actual separation target changes and may require new code bits. A recorded intermediate y can also supply a direct restoration conflict if it is demanded by two distinct incumbent output values. After changing code, encoder or U, recompute/recheck the amplification demand: appending bits changes the object being measured and does not preserve the old bound for free.

This is a genuine coupling between execution and representation formation. It is not a guarantee of convergence or cheap search. A new bit may fail to fit, new U may amplify more, or the budget may be exhausted. Growth is accepted only after actual frozen replay reduces the stated contract failures/margins at its charged cost. Sampling gives finite-bank evidence; a global contract needs a proof or complete finite check.

Importantly A_a is a property of the particular encoded U and code, not an intrinsic minimum redundancy of F_a. A poorly learned U can have A_a=n even when a cheap better U exists. The learner must compare paid U/R repair at the current code with paid code growth, rather than declare more bits necessary merely because its current U is bad. The five-versus-seven calculation below is exact for the specified coordinatewise realization, not a universal lower bound over every possible neural implementation of projection or AND.

### Same nominal arity, different demand: projection versus AND

Consider two separate closed Boolean interface families with t=1, one permitted U output fault and one R output fault. Both accept two input blocks. One computes projection P(b1,b2)=b1 coordinatewise; the other computes AND coordinatewise.

For projection, only the first input's erroneous position can affect the output. Thus A_P(1,1)<=2, attained by placing a U fault elsewhere. A five-bit repetition code with five majority-of-five outputs repairs every such case and tolerates one R output fault. For AND, the attained bound in (3) is three. Five bits fail: clean (1,1), two distinct incoming error positions and a third U fault leave a majority of wrong coordinates. Seven bits suffice, as proved above.

Thus operation behavior can drive q=2/n=5 versus q=3/n=7, even with the same two-input signature. This is an exact constructive consequence of repair (8), not an experiment. The five-bit gadget costs five U plus five majority neurons and 25 U-to-R edges rather than seven plus seven and 49. It makes operation-sensitive demand observable and consequential.

The two families are separate here so no conversion is free. If a single shared interface type must support both operations, its guard is their maximum and seven bits remain necessary for this coordinatewise construction. Per-operation short/long encodings require paid converters and a new composition proof. Moreover an ordinary compiler can inspect the projection's actual dependency on one input and derive the same savings. The example repairs the causal coupling; it does not establish a novel or superior acquisition algorithm.

## 7. Strong conventional SAMEINFO alternative

For the actual Boolean witness, a fixed-code compiler is a concrete stronger alternative, not a generic simulator of the proposed entire learner:

1. Inspect the incumbent Boolean NOT/AND graph. If truth-table reference queries are needed and all Boolean tuples are legal, at most two unary and four binary evaluations identify these two functions. If counterfactual tuples are not legal, use the exact inspected incumbent code or restrict the guarantee to the same encountered tuple domain. Do not supply a free expanded domain only to the rival.
2. Assign Boolean 0/1 to sevenfold repetition and replace NOT/AND by seven independent coordinatewise copies.
3. Attach seven majority-of-seven threshold outputs with weights 1 and threshold 3.5. NOT and AND use their ordinary one- or two-input threshold realizations.
4. Store the explicit proof obligations (1)-(3) and apply the local-budget invariant. Charge construction, graph expansion, storage, all fourteen physical neurons and all wires. No neural fitting/cut search is needed for this witness.

It has the same runtime state, physical gate fault model, trusted reference/write/version ports and output interface as the candidate. Its runtime cost is equal; formation is simpler. It knows a coding algorithm, just as the candidate is supplied a weighted-cut prior, Hamming balls, restoration grammar and likelihood-ratio optimizer. Denying that known coding construction only to the rival would weaken the scientific comparison.

A broader bounded-arity Boolean compiler gives the same basic contract without special semantic reliance on AND: use independent coordinatewise copies of an exact Boolean circuit. With at most k t incoming erroneous coordinates and e_U physical faults, and no cross-copy fan-out, pre-restoration error is bounded by k t+e_U. Internal faults can corrupt only their own output coordinate. Majority length n>=2(k t+e_U)+1 followed by at most t faulty restoration outputs yields the interface. Runtime gate count includes all n copies of the actual Boolean circuit; under independent per-neuron p its larger physical fault-event count must be included. This is not a claim that lowering an arbitrary finite-precision neural graph into such a circuit is always cheap.

For large multi-valued neural interfaces, that compilation expansion can be expensive and a learned computation-compatible code may genuinely offer a better organization. The witness does not establish that advantage. The appropriate simple competing learning method also gets the same internal reference tuples and trains code/operation/repair jointly under the same fault distribution; terminal-only task training would have less information. A concrete counterexample-guided codec/operation co-design algorithm, allowed to reuse the same amplification demands, is another fair rival to the repaired rule.

## 8. One independently entered primary, and the consequential uncertainty

I independently entered the complete original text of Judd and Munro, *Nets with Unreliable Hidden Nodes Learn Error-Correcting Codes* (NIPS 1992), especially methodology pp.90-91 and structured-corruption pp.94-95. [Original paper](https://proceedings.neurips.cc/paper_files/paper/1992/file/c361bc7b2c033a83d663b8d9fb4be56e-Paper.pdf).

SOURCE: their experiments corrupt a hidden channel, learn encoder/decoder representations, and add a fault-free decoding layer for a structured fault example. This supports the candidate's stated boundary: learned coding is prior work, while faults inside the repair computation and reusable operation contracts require their own analysis. It does not establish that the particular proposed co-design loop is novel. No source plots or quantitative learning claims are used here.

Access provenance: web PDF retrieval failed; the existing /opt/local/bin/gs successfully extracted the PDF fetched into process memory, with -dSAFER and a finite timeout. No local PDF/image/source file was saved and no package was installed. The PDF skill had already been read in this session. This was source reading, not an experiment on the mechanism.

**The one consequential uncertainty:** can feedback about an operation's actual amplification select and acquire a substantially cheaper, composition-compatible neural representation over a large structured value space, after every reference/search/fitting/verification/versioning cost, where ordinary reliable-bit compilation and matched joint noise-aware learning do not already obtain the same benefit? The fixed-q seven-bit example does not answer it. Repair (8) makes the proposed coupling testable enough to discuss, but it remains untrained and lacks a finite-budget acquisition argument.

Recommendation to root: retain the valid local-contract reconstruction and the independent operation-demand repair; correct the description of the current growth law before promoting its formation claim. Do not turn the repetition witness into a campaign benchmark merely because its proof is clean. Whether the repaired operation-aware formation question deserves the next allocation is a scientific decision for root/human, not an authorization supplied by this audit.

CLAIM | STATUS | EVIDENCE | WEAKEST LINK | NEXT SINGLE DECISION

The seven-bit noisy NOT/AND plus noisy restorer renews t=1 under the declared local budgets | HAND-CHECKED in stated sense; no execution | (1)-(5), gate/edge accounting | acquisition and global fault-model relevance absent | retain as a correct known-principle witness.

Current code growth is operation-induced | NOT YET SUPPLIED by the fixed-q law | zero-deficit observation (7) | operation failure does not itself change q or growth once separated | root decides whether to adopt an explicit operation-amplification demand rule.

Operation behavior can require different redundancy even at equal nominal arity | HAND RECONSTRUCTION | projection five-bit versus AND seven-bit comparison | same distinction is available to an ordinary compiler; large-state acquisition remains OPEN | assess that single consequential uncertainty before any experiment.
