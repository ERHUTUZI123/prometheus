# Six actual paired-versus-shuffled tails: scientific supervision

Reviewer scope: same Prometheus V2; current human objective is broad, reproducible capability–parameter–complete-cost advantage. TODO3 remains highest priority; Monodromy is optional; no Primary freeze follows from these results. This review closes neither the broader memory-driven computation hypothesis nor the campaign. The reviewer read actual `AGENTS.md`, README Research Vision, Method §§3.7–3.11, the actual typed transition, fresh/tail trainers, CUDA graph engine, ordinary baseline classes/restoration, TRAIN index generator, frozen protocols, results and saved predictions. No neural execution, SSH, checkpoint deserialization, Method edit or other worker-file edit was performed.

**Judgment: the proposed same-prefix pairing did not repair useful query selection in these actual continuations.** It substantially harmed typed Block and Transformer acquisition; GRU became more query-sensitive without consistent correct query orientation. The shuffled typed continuation recovered the two-candidate computation but its argmax ignored query reversal on every VALID pair. This sharpens the learning gap: acquiring a function of two stored values and assigning their query-dependent roles are separable. The next informative action is the already proposed frozen three-NN VALID event audit, not more tail updates. This is a development result, not a confirmation of broad architecture superiority or impossibility.

## Checked intervention and its scope

`DIRECT`: six prospective 600-update tails, three models × two schedules. Within each model, parent stage2 NN/Adam/RNG/Replay128 and post-growth initial NN hashes match. Actual per-tail index arrays equal the selected frozen schedule; the global 28,800-current-row multisets match exactly, and each model's two actual 600×16 Replay draw payloads match. All 3,600 recorded updates have sequential steps 1–600 and finite recorded NLL/gradient norms. All six frozen-version source maps and all 72 TEST-prediction-file hashes were independently checked. Saved-array arithmetic independently reproduced every soft/native TEST accuracy and NLL: 144 metric pairs over 589,824 saved log-probability scalars, maximum NLL discrepancy below 1.48e-7. Baseline native is a documented alias of the same ordinary neural interface, not an extra architecture/pass. Prediction fields and finite updates verify the stated observations; they do not independently verify every optimizer state byte.

The selected schedule archive SHA is `9a0b33d7a7a10dd03ecdec8dd3c45b45cdf85247bbe185b8161dc2bcf650fecf`. TRAIN pairing uses only `(initial, active tokens[:-4])`; it retains all 4,096 rows/2,048 groups and does not filter equal-label pairs. The four-token suffix boundary is explicit task prior. The actual trainer keeps mean terminal NLL/cross-entropy, Adam .003, clip 1, 48 current +16 Replay and the original labels. No new teacher or auxiliary loss enters these six tails.

`PARTIAL`: the intervention changes minibatch dependence, the number of independently sampled prefixes in a batch, and temporal order jointly. It does not isolate gradient variance, a specific causal mechanism inside Adam, or general-purpose learned contrast construction. Equal full-pool exposure does not imply equal padded attention work. `OUT_OF_SCOPE`: arbitrary-key content addressing, learned mechanism novelty, five-seed formal benchmarking, broad real-data performance and complete-training-FLOP advantage.

## Actual performance and query structure

P/S mean paired/same-pool-shuffled. Values are proportions, not percentages. TEST columns describe frozen observations; the constructive diagnosis below uses VALID.

| Model/schedule | P | New VALID | New TEST IID/long/unseen | Old TEST IID 0/1/2 |
|---|---:|---:|---|---|
| Typed/P | 10,991 | .261719 | .279297/.271484/.246094 | .847656/.863281/.873047 |
| Typed/S | 10,991 | .625000 | .580078/.625000/.625000 | .888672/.861328/.871094 |
| GRU32/P | 14,024 | .615234 | .628906/.617188/.625000 | .824219/.835938/.812500 |
| GRU32/S | 14,024 | .626953 | .625000/.623047/.625000 | .853516/.857422/.832031 |
| Transformer32/P | 26,760 | .285156 | .253906/.224609/.250000 | .494141/.687500/.714844 |
| Transformer32/S | 26,760 | .490234 | .466797/.335938/.462891 | .470703/.691406/.701172 |

The generator gives `y0=(3v0+v1) mod8`, `y1=(3v1+v0) mod8` (`memory_live_stage_data.py:69–94`). They agree iff `2(v0-v1)=0 mod8`, on 16/64 value pairs. A deterministic predictor whose answer is invariant to query order, even with all other prefix information, therefore has conditional ceiling `(16×2+48)/128=5/8`. This is an exact finite-grid ambiguity calculation, not chance accuracy, an admission threshold, or a claim about every model scoring near .625.

| Model/schedule | VALID outputs in {y0,y1}, /512 | Nondegenerate query pairs changing prediction, /192 | Both correct / both wrong, /192 | Write-order flips, /256 |
|---|---:|---:|---|---:|
| Typed/P | 214 | 7 | 0/111 | 118 |
| Typed/S | 512 | 0 | 0/0 | 32 |
| GRU32/P | 512 | 49 | 22/27 | 55 |
| GRU32/S | 512 | 5 | 3/2 | 13 |
| Transformer32/P | 223 | 11 | 0/122 | 142 |
| Transformer32/S | 402 | 16 | 3/42 | 107 |

“Both wrong” includes unchanged predictions outside the two candidates when candidate coverage is incomplete. For the GRU arms, full candidate coverage makes the interpretation exact: paired has 22 correctly oriented flips and 27 swaps into the wrong orientation; shuffled has 3 correct and 2 wrong flips. Overall VALID counts obey `320 + both_correct − both_wrong`: 315 versus 321. Counting only the increase to 22 correct pairs would hide its matched failure. Write-order changes should preserve the target, so their increase is adverse binding evidence.

Zero argmax flips do not mean zero probability-level query response. Typed/S mean total-variation distance across query reversal is .002964 (maximum .017574); Typed/P mean is .017647. For the nondegenerate pairs, the correct ordered-candidate log-odds difference is positive on 95/192 Typed/S and 97/192 Typed/P pairs. GRU/P and GRU/S are both positive on 96/192 pairs. These are descriptive finite-grid diagnostics, not independent Bernoulli trials or statistical significance certificates. Each VALID split has only two independently sampled distractor templates.

## Why pairing alone need not create the missing mechanism

`RECONSTRUCTION`: ordinary paired mean loss has the same population objective as rowwise sampling. If a pair currently has the same logits `z` despite distinct targets `a,b`, its shared-logit gradient is `softmax(z) − (e_a+e_b)/2`. This trains the conditional mixture unless a query-dependent Jacobian supplies a useful differential direction. Pair membership is not an additional gradient channel.

A concrete counterexample is binary logistic prediction `z=b+uvq`, with balanced query `q∈{-1,+1}`, target `(q+1)/2`, and initialization `b=u=v=0`. A complete pair has zero gradient in `b`; gradients in `u,v` vanish because each is multiplied by the other. Paired Adam remains at the query-blind solution despite a representable perfect solution. Randomly ordered samples may move `b`, but cannot repair the zero interaction Jacobian either. This is a counterexample to guaranteed repair by pairing, not an explanation established for the actual Block. Where the query-dependent Jacobian exists, paired gradients can be informative. The six outcomes cannot distinguish missing/weak interaction credit, action suppression, noisy candidate acquisition and optimizer ordering by themselves.

## Full relevant resource boundary

Every tail pays 600 new updates, 28,800 current and 9,600 replay label reads, with 1,013,912 active token steps. Each effective path also inherits its OWN paid first 1,800 updates; those updates were not rerun. Original random tails, admissions, failed versions and other configurations remain historical costs, outside these new-tail totals. No table here is a whole-campaign bill.

| Model/schedule | New update wall s | Scientific driver s | SSH scope s | Charged forward proxy G MAC | B64 soft inference p50 ms |
|---|---:|---:|---:|---:|---:|
| Typed/P | 11.7401 | 31.7126 | 37.9088 | 20.2113 padded | 42.0405 |
| Typed/S | 11.7077 | 32.3330 | 39.7690 | 20.2113 padded | 42.5723 |
| GRU32/P | 4.8433 | 10.3241 | 16.2612 | 12.4688 active | 1.6490 |
| GRU32/S | 5.1764 | 10.3865 | 16.1695 | 12.4688 active | 1.6366 |
| Transformer32/P | 6.0142 | 11.0493 | 16.8868 | 58.1376 cropped-dense | 1.0803 |
| Transformer32/S | 6.9513 | 11.9157 | 18.0422 | 60.2446 cropped-dense | 1.0602 |

New-tail sums are update wall 46.4331s, driver 107.7211s, six distinct SSH scopes 145.0376s. These three scopes are nested, not additive. Historical first-1,800 update walls are Typed 13.1883s, GRU 11.5663s, TF 14.2351s; adding once per effective path yields Typed 24.9284/24.8960s, GRU 16.4096/16.7427s, TF 20.2493/21.1864s. Do not count shared history twice in spending totals.

Pair formation is shared once: CPU driver .109588s, SSH4.992074s, no NN update; archive210,448B, array payload877,568B. CPU baseline/typed restoration admissions additionally cost SSH8.835945/9.283725s. Admission synthetic optimizer checks are discarded, not research updates; they remain paid checks. Their scientific CPU-time aggregate is not reconstructed here. Typed graph setup is 1.10645/1.11934s, including three no-SGD warmups .45860/.48360s; the reported subscopes are nested. Six actual reference/graph checks per tail after admission cost 1.06627/1.21098s. Typed replay maintenance costs .04845/.04886s; baseline maintenance .05136/.05256/.05646/.05847s. Checkpoints, reload, VALID, TEST, inference, I/O and logging are included in each driver; the raw result retains their separate measurements.

Model/Adam payload bytes: Typed43,964/88,008; GRU56,096/112,248; TF107,040/214,252 plus16,640B positional buffer. All have55,296B Replay arrays plus separately reported metadata (~1.74KB). Typed carry is320 FP32/example =1,280B, B64=81,920B, excluding backward workspace. Peak CUDA allocated bytes are Typed51,525,120/51,528,192; GRU22,766,080/22,944,768; TF48,002,048/48,002,560. Peaks include diagnostics. Checkpoint files are Typed526,515B each, GRU1,318,517B each, TF1,522,145/1,522,209B; schemas store different index provenance, so file-size differences alone are not model-storage efficiency. At this review's filesystem census, the six run directories contain21,905,245B of evidence/checkpoints; subsequent verifier outputs and copied inputs/code/history are additional storage.

Pairing changes baseline padded token totals1,667,776 versus1,717,952. TF consequently executes3.50% fewer dense forward proxy MACs in the paired arm. Typed always uses1,843,200 padded steps and the same proxy cost; its active proxy is11.1223G for either schedule. Forward MACs exclude nonlinearities, backward, Adam and workspace and are not measured training FLOPs. Full training FLOPs, provider-effective compute and energy remain unknown. Inference is an actual B64/Lmax39 process measurement with three warmups/20 timed passes, not a hardware-independent complexity result. The present typed version saves parameters but has no complete resource advantage: its update time, explicit carry/workspace and measured eager inference are costly relative to these baselines.

## Constructive decision

The typed state transition can represent a useful short route: KEY updates address for the following read/write; STORE uses old raw `h`; LOAD directly returns `r`; MIX can learn a continuous computation of current and read values. This follows from the transition, not a proof that terminal learning finds it (`continuous_primitive_block_typed_memory_actions.py:70–103`). No representational impossibility has been demonstrated.

Continue with the already scoped frozen stage2/paired/shuffled VALID event audit. Separate value availability, content corruption, address preservation, query attention, action mass and final value mixing; compare native/soft where already authorized. Decoded hidden coordinates need not literally equal task state, and near-zero linear-readout accuracy does not prove absence of stored information. Identify whether pairing destroys content acquisition before asking why query selection fails. Relate candidate coverage and correct query orientation to causal memory use; neither alone establishes memory dependence.

One conditional minimal architecture repair is a separate type distribution `{COMPUTE,LOAD,STORE,ADDRESS}` plus a conditional four-expert distribution. Currently the same compute logits determine both within-compute preference and total compute mass (`typed core:76–85`); old expert confidence competes with new data actions. If the frozen audit shows inappropriate suppression of the needed data actions, decoupling these two choices is an explicit learnability hypothesis with unchanged raw values/primitives/terminal information. If the audit instead shows healthy action mass but content/address loss, this repair is not supported; target the observed state-path failure. Do not implement a new head solely because the aggregate score is low, hardcode token roles, add more current-tail steps, or repackage prior address-only attempts without new evidence.

### ROOT proposal: identical-forward removal of direct compute-allocation credit

Addendum recorded **2026-11-02 11:50:43 UTC**, after the actual six-tail results and ROOT's subsequent proposal; this was not part of the earlier frozen protocol. No implementation has been chosen, and the three-NN event audit is still pending at this addendum.

After the six-tail review, ROOT proposed an alternative with no new parameters: `a=logsumexp(z_C)`, `q=softmax(z_C)`, `beta=softmax([stopgrad(a),m_LOAD,m_STORE,m_ADDRESS])`. The compute probabilities are `beta_C*q`; they are mathematically identical to the original seven-way softmax at every fixed parameter vector. The four-compute conditional distribution and seven-action transition can therefore be retained. The extra logsumexp/type-softmax execution still has a small cost; zero additional parameters is not zero additional execution cost. An independent type row would add33 parameters (32 hidden weights plus bias), not97, and cannot generally reproduce a nonlinear logsumexp with one linear row at initialization.

`RECONSTRUCTION`: at fixed incoming features let `G_k=partial L/partial q_k` and `H=partial L/partial a=beta_C(partial L/partial beta_C - sum_j beta_j partial L/partial beta_j)`. The exact original logit derivative is

`partial L/partial z_k = q_k(G_k - sum_l q_l G_l) + q_k H`.

The proposed stop-gradient removes exactly `q_k H`; memory-logit derivatives are unchanged. The three memory columns of the four-way softmax Jacobian have rank3 at finite logits, so all local type-probability directions remain accessible through those logits. This is a concrete backward intervention on a proposed interference channel, not an extra-forward-capacity advantage. It also removes that branch into shared caller hidden features, controllers and embeddings; its effect is not restricted to four output rows.

It must be described as a **surrogate update**, not the exact gradient of unchanged terminal NLL. For binary compute-versus-memory selection `beta_C=sigmoid(z-m)` with compute-only loss `-log beta_C`, the surrogate field is `g_z=0, g_m=1-beta_C`; its cross-partials disagree (`partial g_z/partial m=0`, `partial g_m/partial z=-beta_C(1-beta_C)`). Thus it generally is not the gradient of any scalar objective over both logits. Exact original gradient descent increases the correct selection gap `z-m` by `2 eta(1-beta_C)`, whereas this surrogate increases it by `eta(1-beta_C)`: a simple case where it slows useful learning. No generic improvement follows from removing a correct gradient.

Nor does it fully decouple future compute mass from expert learning. The retained q-gradient sums to zero over logits, but `Delta a` under direct SGD is approximately `-eta sum_k q_k^2(G_k-sum_l q_l G_l)`, generally nonzero; Adam's coordinate preconditioning adds another route to common-offset movement. The precise hypothesis is removal of **direct compute-allocation credit**, not independent type/expert learning.

The strongest alternative remains memory coadaptation: `partial r/partial l_j = attention_j(V_j-r)` vanishes when slots coincide, while useful content/address credit is multiplied by subsequent LOAD/read use. Loading initially zero memory can be actively penalized. A backward change to expert logits does not remove this singularity. With initially zero memory-action head weights it can also remove substantial allocation credit into shared features. The frozen audit should decide whether direct allocation pressure is a plausible leverage point or whether this would be an unsupported repair. No implementation or new neural run was authorized/performed by the reviewer.

### Actual three-NN event audit and next causal question

Addendum recorded **2026-11-02 11:54:47 UTC**, after ROOT executed and synchronized `parameter-typed-three-nn-valid-actions-cpu-v1`. The reviewer independently read its actual source and checked all three frozen source hashes and NPZ hashes. Standard-library arithmetic reconstructed11,776 saved event transitions: action softmax, attention softmax, weighted read, and h/V/address updates all agree within5.77e-7 absolute error. Saved action means agree with the float32 result reductions within3.18e-6. These checks touch raw data and equations; they are not a second-model vote.

| Event/model | Stage2 STORE | Paired STORE | Shuffled STORE | Shuffled ADDRESS | Shuffled COMPUTE |
|---|---:|---:|---:|---:|---:|
| First WRITE | .0938055 | .0005482 | .0047965 | .4173853 | .5715487 |
| Second WRITE | .0957889 | .0005866 | .0049647 | .2385332 | .7518823 |
| KEY before READ | .0542240 | .0001862 | .0014567 | .7179415 | .2801774 |
| READ | .0403193 | .0003561 | .0030358 | .0048753 | .9918273 |
| MIX | not executed | .0002353 | .0082315 | .0171918 | .9718133 |

Paired MIX has mean LOAD .6738602; shuffled first KEY has ADDRESS .9738050. These observations contradict a blanket diagnosis that all memory actions are suppressed by compute mass. The dramatic reduction in WRITE STORE mass is real, but small amplitudes can carry information and learned downstream computation can amplify them. Conversely sharp attention is not proof that physical slots implement the intended logical keys. **COMPUTE itself receives the read vector r** in every primitive's input, so READ choosing COMPUTE does not establish a memory bypass; LOAD is a direct-copy option, not the only retrieval route. Training toward researcher-assigned action roles would add designed semantics not supplied in these runs.

The most justified single next intervention is therefore frozen **post-prefix memory-content dependence**, before a generic gate-gradient repair. Use the saved before-KEY-for-READ carries of paired/shuffled NN and replay only each row's original four-token suffix. Freeze three conditions: exact carry/sham; set V to zero; replace V with another prefix group's saved V via one fixed input-only cyclic permutation. Give the same donor to both query variants of each identical-prefix group; retain the recipient h/controller/address/cache. This needs2×3×512×4=12,288 suffix transitions, no SGD, no search and no teacher/action labels in execution. Report output distance, terminal NLL, candidate coverage and correctly/incorrectly oriented query flips, not just accuracy. The exact sham must recover the saved CPU endpoint. No sensitivity would support a bypass after the prefix; donor sensitivity would establish content dependence, not correct logical binding. This test cannot exclude earlier prefix use of memory, and the zero intervention may be out of distribution. ROOT retains the decision whether to execute it.

This audit is50,944 active transitions, three logical dataset passes,24 batch initializations,0 SGD; CPU wall2.681532s/CPU time2.681869s, SSH7.758544s, peak RSS560,398,336B. Stage2 has10,975P/43,900B; final NNs each10,991P/43,964B; historical Adam/replay are stored but no live optimizer is instantiated. Forward proxy558,739,456MAC; full FLOPs/provider compute remain unknown. Three compressed raw files total26,402,470B. Independent ZIP/NPY accounting finds actual array payload13,709,312B per file, total41,127,936B: source line77 reports40,960B less per file because it omits event_positions and original/runtime_lengths from the payload subtotal. Preserve the original result; correct the derived accounting without rerunning the NN. Audit result SHA256 is `f39259856563ba24168efa6b0d72a91109868a59cc1065a49a8dd08bd74f5311`.

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED BRANCH | NEXT SINGLE DECISION`: paired minibatches repair query-role acquisition | not supported in six actual dev tails; typed/TF acquisition harmed | raw results/predictions/index arrays and actual source checks above | internal causal bottleneck still unobserved, one training seed/schedule | this pairing recipe under current mean-NLL continuation | inspect frozen three-NN VALID event audit, then choose one justified mechanism change. TODO3 remains development; TODO1/4/5 formal new-version retraining follows eventual TODO3 validation and Primary freeze; TODO2 may remain negative/optional; TODO6 preserves these adverse results and all costs.

## Exact result provenance

All paths are below campaign `probes/topology-architecture/runs/`; each named directory contains `learn/result.json`, raw predictions and `records.jsonl`.

| Namespace | result.json SHA256 |
|---|---|
| parameter-paired-tail-gru32-paired-n8-seed0-600-v1 | d2b9ce9211d009836364a750880357974a9c95ec2e36ab1d6cbd6108ee61e88e |
| parameter-paired-tail-gru32-same-pool-shuffle-n8-seed0-600-v1 | 76e7cbaf32365091dfc70ed1fcf28e4dcced6aa8f6d2a42c1768f64a8f447983 |
| parameter-paired-tail-transformer32-paired-n8-seed0-600-v1 | 836e7b0597068c34600a3393b288322c72b77589446587fdaec7aeaca99b356e |
| parameter-paired-tail-transformer32-same-pool-shuffle-n8-seed0-600-v1 | 2f8f92e3a33dddc4b0d73c7f10cd580eca6d7c377bfe1a9dadaadc70035a9e68 |
| parameter-typed-paired-tail-paired-n8-seed0-600-v1 | aebfb41c41946aee611ece3e7e92bee0cf2673978b4182412f718f289462b028 |
| parameter-typed-paired-tail-same-pool-shuffle-n8-seed0-600-v1 | c669043dd45a68ef53e20f1cd76db08012140745e28e06e95da737723311a976 |

Typed old 2,400-update namespace `parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2` gives10,991P/.3828125 new IID and is preserved separately. Historical13,397P RAW posterior-memory/no-projection results (.62109375 new IID) are a different mechanism; neither belongs to this paired-versus-shuffled contrast. This reviewer used standard-library ZIP/NPY arithmetic only for all saved-output checks. Aggregate review CPU/read cost is unknown; no neural or remote execution cost was incurred by this review.
