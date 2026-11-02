# Review 042 — contextual execution and first character-LM development

This is a review of completed local evidence, not permission for the next run or a campaign conclusion. The current architecture remains under development; these are not frozen Primary results. Transformer remains the primary architecture baseline. This review does not interpret a pending Block language-model run.

## Evidence actually read

I checked the frozen source, configuration, all update records, final reports, reload records and sync manifests for the eight completed `parameter-context-n8-seed0-*-stage1-v{1,2}` configurations listed below. Each has exactly 600 contiguous actual updates and the same common batch-index hash, beginning `3918937bbb2e`. All eight record training freeze before the TEST result, and actual tensor/output reload checks pass. The respective manifests contain 15 or 16 files; all recorded sizes and SHA-256 values agree with the local bytes.

For `parameter-char-transformer-300-v1` and `parameter-char-gru-300-v1`, I checked all 15 manifest entries per run, the original frozen driver and model sources, all 300 update records, seven VALID measurements, actual final checkpoint metadata and summary. Both independent exit receipts report return code 0. The 15 entries of `parameter-char-data-v1` also match their manifest. This review executed no model inference, training, SSH, or downloaded code. Owned checkpoint metadata and selected saved tensors were inspected using the existing restricted inert ZIP reader; this is not an independent model implementation.

Disclosure: I authored the character-data preparation and training driver, and subsequently added an unused-for-these-baselines optional Block initialization flag. Checking the original bytes, arithmetic and records is useful evidence, but should not be described as independent code authorship or independent replication.

## Completed contextual results

All entries are seed 0, one stage, 600 updates. Each accuracy pair is soft/native. The three TEST columns measure IID, held-out program combinations and longer sequences, respectively; they are not a continual-retention experiment.

| Configuration | Parameters | TRAIN | VALID | TEST IID | TEST unseen | TEST long |
|---|---:|---:|---:|---:|---:|---:|
| dynamic, original iid v1 | 4,949 | .13623/.13330 | .16016/.11719 | .14062/.12891 | .15039/.12695 | .12109/.13086 |
| static, original iid v1 | 4,949 | .13623/.11938 | .16016/.10547 | .14062/.12695 | .15039/.11719 | .12109/.13086 |
| dynamic, carry v2 | 4,949 | .29419/.29419 | .27734/.27734 | .29883/.29883 | .25000/.25000 | .12891/.12891 |
| static, carry v2 | 4,949 | .29419/.29419 | .27734/.27734 | .29883/.29883 | .25000/.25000 | .12891/.12891 |
| expanded static, carry v2 | 5,973 | .28711/.27783 | .26562/.30078 | .29688/.29688 | .15039/.27148 | .12891/.12891 |
| two-layer GRU64 | 52,552 | 1/1 | .99609/.99609 | 1/1 | .99219/.99219 | .62109/.62109 |
| three-layer Transformer64 | 102,600 | .69385/.69385 | .62109/.62109 | .60938/.60938 | .19336/.19336 | .15039/.15039 |
| implemented recurrent SSM baseline | 100,168 | .91113/.91113 | .78125/.78125 | .77930/.77930 | .21875/.21875 | .11328/.11328 |

The carry modification preserves a public no-op prior; it does not supply a task transition or correct call. The completed dynamic and static versions have identical measured accuracies. This version therefore does not demonstrate a benefit from contextual CALL. The GRU result establishes that the supplied task and terminal feedback are learnable under this 600-update configuration; the Block failure cannot be explained solely by an intrinsically impossible task or universally insufficient feedback.

The dynamic carry raw gradient record is especially informative. The primitive gradient norm is `2.3708413e-8` at soft step 200 and exactly zero at straight-through steps 201, 300 and 600. At those latter points, the last caller weight gradient norm is approximately `7.09e-14`, `2.83e-13` and `8.07e-14`. Independently decoded saved native traces contain 240 active calls, all identity, and 144 inactive padding entries. The trace covers eight examples, not the whole dataset. Together with the selected-ST source, this supports a concrete dead-credit mechanism: the hard selected identity call excludes unused primitive banks from that forward derivative. It does not prove that every failure originated at that switch, nor that the dynamic representation lacks adequate expressivity. Carry-vs-iid initialization also changes the learning trajectory, and its improvement is not topology-specific evidence.

The source-level dense-ST repair supplies the Jacobian of the joint soft local mixture while preserving the hard forward mixture. This is an implementable optimizer/credit modification, not an already demonstrated acquisition or protection improvement. Its completed results must be judged separately; this review neither replaces them with a proof nor infers success from the formula.

## First completed character-LM baselines

Both baselines use the same V=100 TRAIN-only vocabulary, 8,192 TRAIN prefixes of length 32, batch 64, Adam learning rate .003, gradient clipping at 1, exactly 300 updates, and the identical actual batch-index SHA-256
`f8285c27e91c43ab9be0d6c3bcf1ad8691fc24ae100953ae728a21f7360a1530`.
The original source is `versions/code-parameter-char-{transformer,gru}-300-v1/`; driver SHA is `0bb40fbf60745456a8b4a40fa5dd33845f38344dfc04103e78663aa2c3872ad2`. Both configurations use the same input-array hashes for every split.

| Model | P | VALID NLL / PPL | TEST context 32 PPL / accuracy | context 64 PPL / accuracy | context 128 PPL / accuracy |
|---|---:|---:|---:|---:|---:|
| Transformer64 | 119,908 | 2.607729 / 13.568199 | 12.361826 / .302734 | 62.071153 / .095703 | 59.464703 / .083984 |
| GRU64 | 75,748 | 2.367264 / 10.668169 | 9.801371 / .378906 | 9.801382 / .378906 | 9.801378 / .378906 |

These are 512 fixed sampled next-character targets, not corpus-wide or word-level perplexity. I checked that the three TEST files have exactly the same 512 target positions and byte-identical target arrays. Their contexts differ, so the three columns are dependent paired measurements. The preparation code constructs `ids[p-context:p] -> ids[p]` within each original split; the sampled minimum TEST position is 297, above every tested context length. The TRAIN, VALID and TEST arrays are separately source-hashed. The original Gutenberg corpus SHA is `eb61e33cab23dbe712ad57091ded6c7292ba91ac3bb2764f51a3223ec0068fe5`.

The final checkpoint is saved after update 300 and before the first TEST file is opened. There is no best-TEST checkpoint selection. Inference reload checks actual tensor equality and identical outputs on 64 VALID examples; exact training resume is explicitly untested. Independent inert inspection confirms step 300, the expected architecture specifications, saved optimizer/RNG records and the original source hashes.

VALID NLL falls from 4.970523 to 2.607729 for Transformer and 4.779171 to 2.367264 for GRU; all 300 losses and recorded gradient norms are finite. No numerical-failure explanation is needed for these measured outcomes. The GRU's nearly equal three-length results are compatible with dependence mainly on recent context, but this is a hypothesis rather than an identified effective memory length. The Transformer deterioration outside its trained prefix length is a genuine result of this fixed configuration. It is not an implementation-validity threshold or evidence against the whole Transformer family. Its prefix-only bidirectional attention is legal for a terminal next-character target because every input precedes the target; this is not a standard all-position causal GPT training protocol.

## Costs and interpretation

For each LM, training consumes 19,200 target observations and 614,400 input characters. Reported training forward-MAC proxies are 72,818,688,000 (Transformer) and 30,321,868,800 (GRU); these exclude full backward/optimizer work. Recorded optimizer tensor storage is 959,436 and 606,040 bytes. Model parameter storage is 479,632 and 302,992 bytes, respectively; the Transformer additionally stores 33,280 buffer bytes. Actual training times are 3.7754 and 2.5563 seconds and whole-driver times 8.2332 and 6.9316 seconds. External runner times, import overhead, evaluation, checkpoint/reload and concurrent workload must remain separate. These are measured configuration costs, not an isolated hardware-throughput comparison or a full lifecycle bound.

The synthetic Block's smaller P cannot currently be called parameter efficiency: it has not acquired the tested functionality. Conversely, its failure is a valid configuration result and a concrete basis for repairing credit flow. The language-model baselines establish the development pipeline and useful comparison points, not formal five-layer publication evidence or a new Primary benchmark.

**Assessment: CONTINUE the already-running targeted credit/development work.** The next interpretation should compare the actual soft-only and dense-ST training traces, and then the already-chosen carry-initialized Block LM, preserving these original trajectories. This is not a request to restart, expand a tuning search or change MAIN. No acquisition/retention or topology-specific advantage is established by these completed runs. No research milestone or campaign is closed by this review.
