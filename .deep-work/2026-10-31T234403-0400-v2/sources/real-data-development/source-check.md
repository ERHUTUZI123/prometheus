# Checked sources for development data — 2026-11-01

This is a source/provenance artifact, not a research plan or benchmark result.
The coordinator/readers are newly written and **unexecuted** by this worker.
No dataset has been downloaded here and no observed data SHA256 is invented.

## CIFAR-100

Primary source: [the authors' dataset page](https://www.cs.toronto.edu/~kriz/cifar.html), redirected by the source to `cave.cs.toronto.edu`.
The CIFAR-100 download table publishes the binary archive MD5
`03b5dce01913d631647c71ecec9e9cb8` (page lines93–107 when checked).
Binary records are coarse-label byte, fine-label byte, and3072 RGB bytes.
The binary release avoids loading the Python archive's pickled objects.

Official archive: `https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz`.
Coordinator verifies the published MD5 before extraction and records actual
SHA256 as an additional frozen byte identity. The fixed development split
holds50 of each class's500 official training images for VALID using seed0;
the official100-per-class test set stays separate. Actual index arrays, their
hashes, split seed, and NumPy version are recorded when Root executes.

## Speech Commands v0.02, all35 words

Primary dataset description: [Warden, Speech Commands, §§6–7](https://arxiv.org/html/1804.03209v1).
It specifies105829 utterances, one-second-or-shorter mono16-bit PCM at16kHz,
and the released `validation_list.txt` / `testing_list.txt` split files.
Its stated dataset license is CC BY4.0; extracted attribution/license material
is retained. The35-label protocol is not the separate12-label keyword test set.

Official TensorFlow checksum registry:
[speech_commands/checksums.tsv](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/speech_commands/checksums.tsv).
The full release entry gives exact2,428,923,189 bytes and SHA256
`af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58`.
The [official torchaudio implementation](https://raw.githubusercontent.com/pytorch/audio/main/src/torchaudio/datasets/speechcommands.py)
independently pins that digest and implements the released list-based split.
These originals were read as text; none of their code is downloaded/executed.

HTTPS data URL: `https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz`.
The coordinator uses the full archive, excludes background noise from the35
word indices, verifies split disjointness/coverage and speaker disjointness,
and records actual source/member/index hashes. It does not use TFDS's
10-word+unknown+silence remapping or its dedicated small test archive.

## Autoregressive text development

Chosen source is [Project Gutenberg ebook100](https://www.gutenberg.org/ebooks/100),
*The Complete Works of William Shakespeare*, not the TinyShakespeare subset.
The primary metadata marks it public domain in the USA and identifies the
plain-text download. Data URL:
`https://www.gutenberg.org/cache/epub/100/pg100.txt`.
No publisher checksum was located; the coordinator therefore records actual
SHA256, HTTPS URL, ETag/Last-Modified when present, and fetch time, without
claiming an independently published digest. A previous SHA can be required
with `--text-sha256`. Existing prepared bytes are never silently replaced.

The raw ebook and its terms/attribution remain stored. Tokenization uses the
body between standard ebook markers, strict UTF8, no other normalization.
Contiguous90/5/5 character splits are made before fitting a sorted TRAIN-only
character vocabulary; ID0 is UNK for previously unseen VALID/TEST characters.
Each next-token pair stays inside its own split. Evaluation can count every
target once, including the final short context. Report character NLL/PPL and
UNK rates, not word-level perplexity or a comparison to TinyShakespeare.

## Resource and trust boundary

The initial full selection is approximately2.6GB of public downloads, dominated
by Speech Commands, and should reserve10GiB disk for verified archives, raw
files, indexes, and receipts. LM alone is bounded at16MiB source and runs first.
The coordinator is intentionally network-enabled public-data tooling; training
remains offline and uses a read-only prepared-data allowance. No dependency,
pickle, shell command, executable archive member, symlink, device, path
traversal, or archive mode is executed/restored. Manual extraction has byte,
file-count and per-member limits. Incomplete artifacts remain explicit.

This supports immediate audio/vision/LM development under current human
authorization. It does not certify an architecture, fulfill a formal benchmark,
or replace the later Primary freeze and actual retraining requirements.
