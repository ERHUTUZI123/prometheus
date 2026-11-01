# Acquisition repair S — late formation of a reusable operation library

2026-11-01. **DESIGN UPDATED FROM ACTUAL EXISTING EVIDENCE; UNEXECUTED.** Human pause remains active: no new experiment, training, checkpoint/model execution, diagnostic run, SSH or model/train edits. MAIN=TOPOLOGY human-locked; M0 active NOT PASSED; M3 ultimate. Online USER-PARK; Persistent State standalone USER-PAUSED. Only this design is updated. This is not review approval, a new result or a novelty claim.

## New evidence changes the recommendation

ACTUALLY READ: current topology_model.py/router_regularizer and train.py, and all38 existing rows in `runs/gpu-firststage-n8-router-mi05-seed0/learn/records.jsonl`. Configuration is N8, seed0, LR.01, joint compilation, log credit, router-MI weight.05. Hard validation/IID/OOD/length16 accuracy is1.0 from step100 through600; soft IID is1.0 from200. There is NO stage_end record; the run was interrupted with a step700 checkpoint. No post700 outcome or completed-stage/retention result is claimed. At600, soft longer-length accuracy remains lower (OOD.861328125, length16 .591796875), despite perfect hard execution. Guard checks are0 before first admission.

This overturns the earlier recommendation that replacing the bank architecture was needed to repair N8 acquisition. Diversity alone already repaired this observed seed/scale. The previous direct-table/EM proposal is now a contingent alternative, NOT the required next architecture or an allocated implementation. Log/prob credit and more capacity are not the demonstrated decisive change in this comparison.

The capacity reasoning survives: two learned N-state operations plus skip fit stage0 inside six banks/two slots. The learned target maps/routes were not supplied to training. Stage2 hidden memory is a separate capacity question and cannot explain stage0 collapse. No reproducibility across independent seeds/scales, continual retention or M0 advantage follows from this one interrupted first-stage repair.

## A remaining concrete semantic defect

OBSERVED step600 graph: the jointly compiled u operation is the single8-cycle `[1,2,3,4,5,6,7,0]`, but the displayed hard call route is `[2,2]`. There is NO permutation P on eight states whose square is that8-cycle. If P²=C, P commutes with C, so P=C^k and2k=1 mod8, impossible.

Therefore the successful whole-token joint compiler is doing substantive semantic work: its u is not the faithful execution of that displayed repeated hard-bank call. This does not refute bank capacity; another learned route can express u. It identifies the missing abstraction step. Distinct tokens currently acquire useful whole operators without necessarily acquiring a coherent reusable library of the component operators named by their routes.

The MI regularizer maximizes mutual information between observed(task,token) keys and slot call choices, without labels. It prevents collapse but can also reward separating semantic aliases and spending experts on input identities. It does not literally guarantee one expert per token, and correct whole-token behavior can still use soft mixtures. Permanent token diversity is not the desired semantic organization.

## One concrete upgrade: acquire, materialize, then quotient by behavior

Keep the now-successful neural acquisition model and its terminal-only CE. Treat token-MI as a BOOTSTRAP conditioner for unacquired behavior, not as the permanent library definition. Add an explicit learned semantic library and a bounded compiler transaction AFTER actual acquisition.

State:

    S = (shared neural proposer theta, semantic library L,
         token/task programs P, old learned graph references R, task evidence E).

L entries are executable learned permutations with trainable implementations. P contains programs over semantic entries; entries are not reserved by token identity. Both are derived from trained model behavior. The supplied construction operations (compose, compare and install a learned finite map) remain disclosed; this is not a claim that the system invented those meta-rules.

### 1. Materialize the acquired FULL operator

After the current task meets a declared terminal validation criterion, extract each whole-token effective operator from its learned encoder/calls/banks using the current joint compiler. This uses model parameters, not teacher arithmetic or correct intermediate labels. First check whether that map already equals a short program of existing semantic entries (including skip). If it does, bind the token to that program. If it does not, install the acquired whole map as ONE new expert, using an unused/duplicate expert slot where available.

The new expert logits can be initialized to strongly represent the MODEL-INFERRED permutation, not the teacher permutation. Rewrite its program to `[new_expert,skip]`. In the observed u case this removes the impossible square-root factorization while preserving the acquired8-cycle. Installing an internally inferred operator is actual mechanism formation from learned behavior; the constructor and finite-state representation are still supplied.

Because current calls are produced by shared neural weights, do not assume a local route edit leaves all other tokens unchanged. Use a small compiled program record as a temporary executable target/fallback, and distill the shared router onto its OWN inferred rewrite. Those routing targets are generated by the trained model/compiler, not teacher program supervision, and their extra compute/storage must be reported. Remove the fallback only after the actual hard router/executor matches its programs. Soft/hard behavior is reported separately throughout.

### 2. Identify semantic aliases, not token aliases

Canonicalize full learned maps by their action on the declared finite latent state space. Identical maps share one semantic ID even when input tokens/tasks differ. Identity uses skip. For more permissive equivalence on reachable states, run the existing finite old/new decoded graph relation; do not infer global equivalence merely from a few matching terminal samples.

A merge/rewrite is a transaction: snapshot all old learned hard graphs; propose expert reuse/materialization and program rewrites; accept only if each old graph remains related and the measured task validation/error contracts remain satisfied. Reject or retain the previous representation otherwise. This is the existing ordinary equivalence certificate opportunity, not a new oracle. If comparison is done at component level, include slot-intermediate reachable states, not only states at token boundaries; whole-token graph equivalence is the safer existing interface.

After a successful semantic consolidation, anneal token-MI away for those consolidated keys. New aliases may then call existing experts freely. The next task learns programs over the reusable library plus a bounded reserve of uncommitted experts. Bootstrap diversity may temporarily separate candidates, but semantic reuse wins once terminal behavior is acquired. No permanently imposed one-key/one-expert rule is needed.

### 3. Preserve plasticity without confusing it with frozen programs

Store the admitted semantic contracts, but allow shared neural routing and expert logits to change inside updates passing the actual old decoded-state relation. The library is a protected semantic interface, not an excuse to freeze the entire model. A new task may reuse an existing composition or produce a genuinely new learned whole operator; count these separately. A bounded frequent-composition macro can later be materialized from learned programs, but mere renaming is not new functional capability.

A compiler/program learner can use exactly this materialization, equivalence test and reuse policy. A frozen modular library is also a necessary rival. Thus the design creates a concrete, executable operation-formation stage missing from the current soft factorization; it does not establish a uniquely topological algorithm or an important advantage over those explanations.

## Forward and credit after the upgrade

The neural proposer retains the current differentiable soft forward and terminal cross-entropy (plus temporary bootstrap MI). The deployed hard executor runs P's programs over L's executable maps. Router distillation teaches the proposer only its own already acquired semantic rewrites; no teacher route or intermediate state is introduced. Candidate rewrites are checked against the complete current finite graph and actual task evidence. Full-map assignment, expert installation, routing self-distillation, graph checks, fallback storage and rejected transactions are all costs.

The next architecture-level hypothesis is specific: semantic materialization/reuse might release redundant expert capacity and support later acquisition while old multistep behavior stays fixed. That is stronger than saying a scalar diversity term improves optimization, but it remains UNTESTED. The existing MI run establishes successful first-stage acquisition only. It does not demonstrate alias merging, library reuse, new operation formation by this transaction, multi-stage retention, scale growth, M0 passage or M3 value.

No new run is proposed as currently authorized. The concrete next implementable object is this late semantic-library transaction on an already acquired model; further work awaits the user's pause decision. Do not discard the successful bank model or resume the superseded wholesale replacement merely because it was written first.
