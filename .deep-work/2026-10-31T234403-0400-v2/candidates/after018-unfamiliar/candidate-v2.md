# V2: acquire a restorable computational representation, not only a robust message code

**AGENT PROPOSAL / CONSTRUCTIVE PASS; untrained and unexecuted.** Same Prometheus V2, root sole coordinator. No MAIN decision, experiment, installation, paid route, model override or online-organization continuation. Provider-effective mode unknown. Earlier v0/v1 and comparisons remain unchanged. This direction was selected before root supplied Judd–Munro (1992) as a nearest-source lead; that original body was then read. Source details are in `sources/after018-unfamiliar/source-v2.md`.

## Judgment and important limitation

An acquired computation may be correct on clean, short executions yet become unusable when its approximate/noisy internal states are repeatedly composed. A separated representation alone does not solve this: computation can amplify existing errors, the correction circuit can fail, and a final noisy scalar readout still has a noise floor. The candidate is a learner that constructs **code bits, encoded operations and an executable restoration operation together**, using its own inspectable incumbent computation as the semantic reference.

The proposed formation unit is an **operation plus a restorable state interface**. Redundancy is created when actual code neighborhoods demand incompatible restorations; a new learned bit is selected to separate those internal values. Encoded operations are then learned to return their outputs to the same restorable interface. The intended acquisition benefit is composition at new depths without retraining the whole computation at each depth. Neither that learning benefit nor novelty is established here.

This connects to the Fundamental Question through improvement and reorganization of existing internal mechanisms. It does not explain their initial acquisition from a bare environment. It constructs an additional internal operation—restoration—and changes the representation in which later computation executes. Reliable-computation theory and learned error-correcting representations are established. The scientific question left is whether this **joint operational formation law** acquires useful reliable computation beyond ordinary noise-aware training and fault-tolerant compilation under the same resources.

## 1. Information source, designed substrate and actual state

Start with an incumbent learned computational graph. Its local operations `F_a:S_1 x ... x S_k -> S_out` and the values on its interfaces can be inspected and run. The finite-precision wire-value sets may be large. We retain a finite bank of actual encountered values and operation tuples; no exhaustive reachable-state catalogue is silently supplied. Interface grouping and operation boundaries initially come from this graph. Discovering those boundaries is outside this version.

Use a declared higher-precision/reference execution of the incumbent during acquisition. It generates clean internal tuples `(s_1,...,s_k,F_a(s_1,...,s_k))`. This is legitimate additional access to **our own acquired computation**, not access to a hidden environmental A or latent truth. Every reference evaluation, checkpoint and retained parameter is charged. If reference execution is itself noisy, its errors must be modeled; this version assumes a trusted reference route and gives the same route to every rival. It does not claim training under noise everywhere with a pristine teacher hidden inside it.

Faithfulness to this reference inherits its systematic semantic mistakes. Repairing execution noise does not establish task correctness or discover unknown facts. If the incumbent was inferred only from the negative point-oracle transcript in the query audit, inspecting its code cannot reveal the unknown positive point.

For each incumbent interface type tau, the learned representation contains:

- n_tau binary code channels and a learned encoder `E_tau(s)=c_s in {0,1}^n`;
- a shared learned restoration network `R_tau:{0,1}^n -> {0,1}^n`;
- one learned encoded operation `U_a` for each retained incumbent operation;
- a finite trace/counterexample bank, integer correction/guard radii, and actual graph/weight/code storage.

The runtime state consists of encoded blocks c on the computational graph's live interfaces. One operation executes

`(h_1,...,h_k) --U_a with physical faults--> y --R_out with physical faults--> h_out`.

The reference F and clean s labels are absent from this runtime transition. The intended semantic decoder is used to state correctness; it is not a free reliable runtime module. The encoded U/R networks consist of bounded-fan-in finite-precision threshold neurons and explicit wires. Each physical output can misfire according to the declared fault model, including outputs inside R. Layer count, fan-in, synchronization, storage and every executed neuron count. The host's reliable addressing/scheduling and the permitted fault locations are substantial scaffolding.

## 2. Actual code creation: separate incompatible restoration demands

Let `B_t(c_s)` be the Hamming ball of radius t around the current code for value s. Distinct values that the incumbent can distinguish at this interface need distinct decoded codes. This version preserves their full incumbent value, rather than claiming to discover the minimal behavioral quotient.

Pick a declared guard radius q for restorer training. If `d_H(c_s,c_v) <= 2q` for distinguishable s,v, their q-balls overlap: some y would be required to restore to both clean codes. More decoder capacity cannot satisfy those incompatible demands. Code growth addresses this conflict.

Maintain conflict weights `w_sv >= 0` from such explicit overlaps and from actual failed restoration/operation trials. The required code-separation deficit is

`D_code = sum_(s,v) w_sv * max(0, 2q+1-d_H(c_s,c_v))`.

First allow the existing encoder bits to improve this objective. When the bounded attempt does not resolve a recorded conflict and the external channel budget permits growth, choose a new binary assignment z to the **encountered incumbent values**, by maximizing

`sum_(s,v with positive deficit) w_sv * 1[z(s)!=z(v)] - charged bit/encoder cost`.

This is a finite weighted-cut problem on the encountered-value graph, not an oracle for all future values. An exact bounded search or a declared approximate solver can instantiate it; all search counts and failure to improve is recorded. The cut prior is supplied by the researcher. It is not itself claimed as a new coding algorithm.

Train a new encoder channel on these internally selected z labels using the actual incumbent values as inputs. Its labels are created by the conflict optimization, not by an externally supplied code constructor. Freeze the channel to its actual binary decision and accept it only if its realized codes reduce the recorded deficit under the frozen evaluation. Failure to fit is a failed formation proposal. The same bit may be a nonlinear feature of incumbent state; fitting and input inspection are charged. Extend every affected encoded operation/restorer by an output channel and retrain its circuit within the declared node/edge budget. This structural step creates actual redundancy and costs additional computation; no free replication is assumed.

In the two-value worked case below, every improving cut assigns opposite bits to the two values. Repeated growth therefore creates repetitions or complemented repetitions **as a consequence of the conflict law**. Sevenfold coding is not preinstalled as the initial representation. For many values, weighted cuts may generate poor codes or expensive encoders; no efficient general construction is proved.

## 3. Learning encoded operations and physical repair

Let c_s denote the current frozen encoder's clean code. Build training trials by selecting actual incumbent tuples, perturbing each input within the declared input radius, and injecting faults at allowed physical neurons of U and R. Targets are the incumbent's corresponding clean output codes. Include restoration-only trials around each encountered value as well as operation-plus-restoration trials; this prevents R from being merely a communication decoder used once.

For a target value v, define `loss_t(h,v)=max(0,d_H(h,c_v)-t)`. A possible finite training objective is

`J = E[loss_t(R^xi(U_a^eta(h_1,...,h_k)), F_a(s_1,...,s_k))]`

`    + gamma * E[loss_t(R^xi(y),s)] + lambda * actual_work + mu * stored_bits`,

where `h_i in B_t(c_si)`, `y in B_q(c_s)`, and eta,xi are actual fault masks. The expectation is over the declared finite training distribution and recorded adversarial trials, not an unexamined universal quantifier. Code-distance conflict D_code is an additional representation-formation objective. The clean encoder target is held fixed inside a circuit-training step; afterward a new code version can be proposed and replayed globally.

Hard thresholding and correction do not provide ordinary backpropagation through their true zero-a.e. Jacobians. One valid learning implementation uses stochastic Bernoulli threshold-neuron outputs during acquisition, with logits determined by their finite weights. The pre-fault sampled decisions are logged, and a fixed fault mask is applied afterward. For exact expected-loss optimization use the ordinary likelihood-ratio update

`theta <- theta - eta_lr * (trial_loss-baseline) * sum_nodes grad_theta log p_theta(sampled_decision | actual_inputs)`.

Independent randomness and supplied fault-mask probabilities give the stated estimator; no derivative of a bit flip or a correct teacher intermediate is invented. Optimize the encoder-channel classifier separately against the internally chosen cut labels. A straight-through surrogate could be another method, but would be labeled a surrogate and would not establish this exact update. Circuit structure can undergo bounded add-node/add-edge proposals selected by the same actual replay objective.

After training, freeze deterministic gates and test their actual operation/restoration behavior on a finite frozen contract bank. Stochastic training success is not evidence that argmax gates satisfy a worst-case contract. Fault search can sample or propose concrete bad masks; failure to find one is not a proof. Full enumeration is allowed only when its real finite cost is paid. The hand witness below supplies a mathematical example of the relevant contract without claiming a learner has acquired it.

### Intentional writes and learning are separate from faults

An encoded U_a deliberately moves one or more blocks to the next value's basin; R then restores that new basin. R is not given an oracle saying which old value to preserve. A raw write outside this protocol can be undone as an error. Parameter/code learning occurs in the trusted outer acquisition phase between frozen trials, not by pushing gradients through R during noisy execution. A new codebook/R version requires retraining affected operations and an explicit paid re-encoding of retained runtime state before switching. No seamless cross-version composition or autonomous noisy online plasticity is claimed. A digital baseline gets the same trusted write/version port.

## 4. The local contract and its actual compositional consequence

The desired admitted contract for an operation is

`for all h_i in B_t(c_si), for all faults in declared local budget:`

`R_out^xi(U_a^eta(h_1,...,h_k)) in B_t(c_(F_a(s_1,...,s_k)))`.       (C)

If every executed operation satisfies (C), induction on an acyclic composition gives the same basin guarantee at every generated interface, independent of circuit depth, **provided every local fault budget holds**. A recurrent machine has the corresponding finite-step induction. This is a deterministic invariant of learned-target operations if they actually satisfy the contract. It is not a claim that the finite training objective establishes its universal premises.

Storage denoising alone is insufficient. U must transport the whole incoming basin into a region that R can restore, and R's own permitted faults must return to an accepted output basin. The common interface is what permits a newly composed computation to use an acquired operation without retraining all upstream states.

## 5. Worked code formation and noisy execution, including a faulty restorer

This is a hand construction for Boolean-valued interfaces and already acquired Boolean NOT/AND operations, not an executed learning result. There are two incumbent values 0 and 1. Require input/output basin radius t=1 and a restorer guard radius q=3. Their q-balls need separation at least seven.

Start with the one-bit codes c_0=0,c_1=1. The conflict deficit is six. Each improving new-bit cut separates the same two values; after six accepted idealized channel additions, the codes have seven opposed positions. Complementing selected positions is an irrelevant re-encoding, so choose `c_0=0000000`, `c_1=1111111` for the readable witness. This traces the structural rule's choices. It assumes each bit fit succeeds; it is not a claim that the neural optimizer found them.

Define each U_NOT output as the complement of the corresponding input bit; define U_AND coordinatewise on two blocks. Define R to contain **seven separate majority-of-seven threshold neurons**, each reading the seven input lines. Its threshold is 3.5 with unit weights. These are permitted finite-precision neurons; their weights/wiring are the witness target, not installed initial repair rules.

Fault model for each invocation: at most one U output neuron misfires, and at most one R output neuron misfires. Inputs may each already contain one wrong bit from earlier restoration. No independence assumption is needed for the following calculation.

- NOT preserves at most one incoming wrong position; one U fault makes at most two wrong positions before R.
- AND is wrong only at positions where at least one input differs from its correct bit, giving at most two wrong positions; one U fault makes at most three.
- With at most three wrong inputs, **every nonfaulty majority-of-seven neuron** returns the correct logical bit. At most one R neuron misfires, so the restored output lies within radius one of the right code.

Thus the same noisy physical restorer satisfies the required output interface for both operations. The majority computations share inputs, but the proof uses a deterministic error-count bound; it does not pretend their errors are independent. Any finite NOT/AND circuit composed from these blocks preserves its logical internal values under the stated local budgets. In particular a NOT result can feed both AND inputs or a later NOT without a growing Hamming-error allowance. A bare repeated representation without restoration can accumulate new errors across operations and leave its decoding basin.

Each logical operation costs seven U neurons plus seven majority neurons, with seven wires into each majority neuron; NOT's first stage is simpler but still counted. Block storage is seven bits per logical wire. There is no constant-cost invisible decoder. The code-growth channels, reference examples, weights, all failed circuits and restorer learning are additional acquisition cost.

### A finite-depth stochastic consequence, not an unlimited threshold theorem

Suppose instead every one of these fourteen neuron outputs flips independently with probability p. A seven-neuron stage violates its one-fault budget with probability at most `binom(7,2)*p^2=21p^2`, by a union bound on faulty pairs. For H logical operation invocations, another union bound gives logical encoded failure probability at most `42 H p^2`, clipped at one. This is conservative and conditioned on initially valid blocks. It shows one possible reduction in the **order of accumulated fault probability** at explicit redundancy cost.

It does not yield indefinitely reliable execution at fixed p and unbounded H. Correlated common-mode faults can violate the budget; recurring systematic U errors are not independent hardware faults. Concatenating the whole construction is not supplied as a theorem: the majority-of-seven gate has higher arity than the two-input witness, so blindly applying the same gadget recursively would be invalid. No threshold value or physical power advantage is claimed.

The claimed output is an encoded block. If the task finally insists on one physical scalar output neuron that independently flips, its final error floor remains p. An external clean majority observer used to assess logical correctness is an evaluation interface, not an extra reliable device inside the machine. Further reliable output protocols would be a separate paid construction.

## 6. Source boundary, strongest same-information alternatives and fragile bridge

**SOURCE:** von Neumann's original pp.70–72 already separate executive organs from restoring organs and warn about correlated lines in restoration. Judd–Munro's original 1992 body already learns separated error-correcting codes by injecting faults into hidden-channel units, including a structured-corruption example requiring another decoding layer. Learned coding, majority correction and reliable composition are not invented here. The finite witness is a reconstruction of established principles.

The proposal's specific formation content is the loop linking **operation-induced guard demand, overlap-driven code-bit creation, learned encoded operations, and a physically faulty shared restoration interface**. That is an explicit architecture/learning proposal, not a priority claim. Closest coverage by recurrent learned error-correcting machines, fault-tolerant compilation and computation-compatible codes remains to be checked; a good name or an unsearched intersection is no evidence of novelty.

Strong same-information alternatives are:

1. Quantize the incumbent graph and apply ordinary fixed-code/fault-tolerant compilation. It receives the same trusted reference, precision, neuron fault model, write port, graph access and total budget. The seven-bit witness itself is available to that rival. A reliable digital computer with error-free components is a different substrate, but reliable quantization cannot be forbidden only to the rival.
2. Train the same-sized recurrent/graph neural computation with noise injection, including correction layers and parameter redundancy, using the same internal reference tuples. Merely outperforming terminal-only training would confound information access.
3. Learn an autoencoder/code and distill each incumbent operation into it with ordinary end-to-end training. If its actual update law and budgets match ours after eliminating the conflict bookkeeping, the proposal is covered at that level.

The most fragile bridge is not existence of a correct code. It is **economical acquisition of a computation-compatible code and repair circuit**, especially when the incumbent's value spaces are large, the encountered-state bank is incomplete, and a learned operation can amplify faults in ways a storage code does not anticipate. Bit growth may merely memorize finite values, while fixed coding wins outright. A restorer can also erase legitimate intermediate distinctions if the interface partition is wrong. This version preserves full incumbent values to expose that cost instead of assuming a perfect quotient.

## 7. Success counterfactual and finite next decision

If the formation rule reliably acquired local contracts that transfer to substantially deeper and differently composed executions, the learner would have formed a **reliable computational alphabet** rather than only a noise-resistant representation. This could change how neural systems obtain stable discrete/internal algorithmic mechanisms from approximate units. The importance would lie in a measured acquisition-and-composition consequence under honest substrate costs, not in new computability or in recovering a prescribed repetition code.

What would reverse the prospect: the strongest plausible success is only a known code compiler rediscovered expensively; code growth gives no advantage over matched noise-aware distillation; the trusted reference and full value bank do all the work; faults in repair/readout remove the claimed benefit; or only manually supplied intermediate semantic partitions make acquisition possible. Such outcomes would justify closing or changing the representation, not merely training longer.

**Next single decision for root:** inspect and attack whether the overlap-driven representation formation plus noisy local contract supplies a consequential learning problem beyond known encoder training/compilation, and which missing acquisition step deserves a constructive repair. No experiment is admitted by this worker. The seven-bit witness should not become a toy benchmark for the campaign.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`Code channels can be created from incompatible restoration neighborhoods without a preinstalled codeword list | EXPLICIT FORMATION RULE, untrained | weighted-cut/channel-fit rule §§2–3 | search/fitting cost and unseen-value generalization | no empirical branch | assess whether it differs usefully from ordinary code learning`

`A noisy operation+restorer can renew the same input error allowance after each operation | HAND RECONSTRUCTION | seven-bit NOT/AND proof §5 | restrictive fault locations/budgets; final scalar floor; learned acquisition absent | naïve independence/indefinite reliability not claimed | independent contract/learning audit`

`The learner acquires useful reliable composition more economically than matched coding/distillation | HYPOTHESIS | full state, formation/update rules and success counterfactual | no comparative evidence or established novelty | v0/v1 overlaps retained separately | root judges next constructive step, no test automatic`
