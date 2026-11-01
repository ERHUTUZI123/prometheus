# Generated-program replay certificate

Scope: the coordinator separately authorized author comparison and one bounded
independent exact replay of emitted JSON programs after receiving the complete
micro-law-only freeze. That earlier freeze is preserved unchanged.

Read-only comparison inputs are exactly bound by
`generated-replay-inputs.sha256`. The independent interpreter and contract
were inspected and frozen in `generated-replay-pre-execution.sha256` before
the only new execution. All those hashes were checked OK afterward, as were
all artifacts in the original `full-freeze.sha256`.

Execution used the supplied network-denied/write-restricted isolated runner,
timeout 90 seconds, script `probes/rebuild/generated_replay.py`, log
`probes/rebuild/generated-replay-execution.log`, write-dir `probes/rebuild`.
Exit 0. The saved summary and terminal raw record both say PASS. There was
one execution, no repair/retry, no author-code import, and no author writes.

Evidence: two actual generated programs; six scalar and six matrix commands;
every emitted state coordinate and source output checked exactly after every
command; two deliberate coefficient mutations detected; serialization smoke
before calculations plus initial/final state roundtrips. Raw records were
flushed/fsynced incrementally. State selection used every emitted event;
expected target dimensions never selected the retained state.

`generated-replay-full-freeze.sha256` binds comparison/certificate, contract,
interpreter, input manifest, pre-execution manifest and all new JSON/JSONL/log
artifacts. It does not include itself. This scoped certificate makes no
novelty, training-success, superiority or foundational scientific claim.
