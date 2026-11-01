# Six-task terminal stream: parameter-efficiency benchmark schema

Version: `parameter-efficiency-terminal-stream-v1`. MAIN=TOPOLOGY is human locked; parameter growth is the current comparison, M0 is unpassed and M3 remains the final goal. Transformer is the primary architecture baseline; ordinary constraints, freezing and modular policies must receive the same files.

## Executable interface

`generate_stream(n_states=8|16, seed=0)` returns six task dictionaries and a public manifest. `save_stream(out, n_states, seed)` writes a **new** directory with `datasets/task{0..5}-{train,valid,iid,ood,length16}.npz` and `manifest.json`.

Each NPZ exposes only:

| Key | dtype / shape | Meaning |
|---|---|---|
|tokens|int64 `[B,18]`|Symbols1..6, right-padded with0|
|starts|int64 `[B]`|Observed initial value0..N−1|
|task_ids|int64 `[B]`|Public context0..5|
|targets|int64 `[B]`|Observed final output0..N−1|
|metadata|scalar JSON string|Public seed, dimensions, sampling and hashes; no teacher tables or latent labels|

Train has8,192 rows; each other split512. Every policy must use identical arrays and feedback. Task ID is necessary: the same symbol sequence can have different semantics in different task contexts.

Example construction commands, **not executed as part of this delivery**:

```sh
python parameter_efficiency_tasks.py --out NEW_N8_DATA --scale 8 --seed 0
python parameter_efficiency_tasks.py --out NEW_N16_DATA --scale 16 --seed 0
```

Only NumPy and the standard library are needed. No training, SSH, networking, subprocesses or installation occur in the generator.

## Private benchmark construction

Tasks0–2 each draw six independent unknown permutations on N states, with observable noncommutative two-step behavior. Tasks3–5 each draw independent reversible operations on2N internal states; readout exposes only the state index modulo N. One privately chosen symbol crosses the two state layers using independently random permutations in its two directions; the other five operate with independent permutations inside each layer. Its symbol ID is independently randomized per task. Neither its identity, a mode label, its transition table nor a correct program is serialized for the learner. In particular, crossing twice is not imposed to be identity and crossing need not preserve the observed output.

Each teacher is one deterministic random draw, never resampled to improve learner behavior or profile sufficiency. Visible noncommutativity is checked; a rare failure raises an explicit construction error rather than quietly replacing the draw. Hidden-memory instances have2N physical internal states, but minimal distinguishability and sufficiency of a finite response profile must be established from actual terminal data. If short profiles merge states that longer words distinguish, retain that learning failure rather than consulting teacher tables or changing the seed.

The private oracle provides only terminal labels during dataset construction. Models should consume the NPZ interface, with the generator/oracle excluded from their execution inputs by the coordinator. Python's private naming is not an OS isolation guarantee.

## Sampling and heldout composition

Public coverage includes **every** `(start,word)` of lengths0 through3, identically for all tasks and independently of the hidden crossing symbol. This consumes259N rows (2,072 for N8,4,144 for N16). Remaining training rows use random starts and words of lengths4..8. Rows are shuffled; no coverage-row, state or route annotation is given. An empty word is represented by all PAD0 tokens and has an ordinary final-output label; learners must support this legal input.

This supplies a public observation-table interface H=`(start,prefix)` with prefix in `{empty,1..6}` and suffix tests S=`{empty,1..6}`. For each h in H, each next symbol a and each suffix s in S, the required terminal record for `prefix(h)+[a]+s` is present in TRAIN. A learner can therefore construct response profiles and proposed transition closure without an oracle. It must still check complete original training words and independently heldout longer words: finite profile agreement is not automatically all-word equivalence. Prefix search, merges, closure checks and any refinement costs belong to the policy.

`valid` and `iid` are independently seeded fresh short-word tests with lengths1..cap. Here `iid` means an in-range random test, **not** exact equality to the deliberately coverage-stratified training distribution. Exact input overlap is counted in metadata rather than hidden. OOD words have lengths9..12 and the length16 split has length16, strictly exceeding all training lengths; these input sequences are disjoint from the training set by construction. They test composed execution rather than more angle estimates or supplied programs.

## Reproducibility and evaluation scope

Teacher RNG and sample RNG use independent deterministic SeedSequence namespaces indexed by scale, master seed, task and split. The manifest records the generator source hash, NumPy version, seed, actual lengths, overlap counts, canonical four-array SHA256 and compressed NPZ SHA256. No transition table is written.

A natural capability unit is acquisition of a complete new task contract on independently heldout words while retaining prior task contracts. Count actual retained contracts and parameter increments per newly retained contract; do not equate a new task ID or larger state count with successful acquisition. Parameter count, nonparameter operation/state tables, replay memory, constraint history, metadata and checkpoints must be reported separately.

Independent random task functions intentionally avoid a supplied arithmetic compiler or a guaranteed semantic reuse curriculum. They do not imply that every task lies outside the group generated by earlier permutations. Complete short-label coverage also gives ordinary program/state-identification methods a legitimate strong route to acquisition; those methods must remain comparison arms. This benchmark alone does not establish a general architecture, open-ended growth, topology-specific advantage or M0 passage.
