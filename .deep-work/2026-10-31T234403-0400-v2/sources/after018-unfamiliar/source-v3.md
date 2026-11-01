# V3 original reading and construction provenance

No experiment or training. V2 files and hashes are unchanged. Effective provider mode unknown; same V2 run and instruction_only shared filesystem. The new artifacts are this source note and `candidate-v3.md`.

## Chronology and contribution boundary

Before v3 writing, this worker developed the Toffoli observable lift, noticed that a single xy-helper fault fans out, and constructed pullback closure as a way to obtain coordinate-permutation execution. Root/supervisor then supplied the operation-output sets Y_v and their intersection/guard analysis. That suggestion is recorded as external input in v3 §6. Root subsequently supplied the cycle/transposition S_N orbit-size discriminator; the candidate explicitly attributes it and reconstructs its proof. These are not independent discoveries by this worker.

Root also supplied the Function-Correcting Codes title and author PDF as a nearest-primary lead. The author PDF timed out, but the original arXiv v3 body was accessible. A targeted Koopman original was then read to compare the actual closure construction. No broad literature landscape was attempted. Prior von Neumann/Judd readings remain in source-v2 rather than being counted again as fresh entry.

## 1. Function-Correcting Codes — actual original body entered

Andreas Lenz, Rawad Bitar, Antonia Wachter-Zeh and Eitan Yaakobi, **Function-Correcting Codes**, arXiv:2102.03094v3 (22 May 2023), earlier ISIT 2021 version noted in the paper. [Original full body](https://arxiv.org/html/2102.03094). The supplied [author PDF](https://yaakobi.net.technion.ac.il/files/2024/02/80_FCC_IT23.pdf) failed retrieval in this worker.

Read §II Definitions 1–2 and basic properties; §III Definitions 3–5 and Theorem 1 with proof; §§III-A/B Corollary 1 and Theorem 2 with proof; the beginning of §III-C including the irregular Plotkin bound proof. Did not read all specialized-function constructions, appendix proofs or compare paper versions.

SOURCE summary: systematic message-plus-redundancy codes need sufficient distance only between messages whose desired function values differ. The optimal redundancy is related exactly to an irregular-distance requirement and an associated independent-set formulation. For bijective desired functions, correction is full-message correction. The encoding/decoding complexity is function-dependent. Therefore irregular function-sensitive distances already have a concrete theory; they cannot themselves supply this project's novelty.

RECONSTRUCTION boundary: v3 uses a reversible full-state block and concerns the transport of physical coordinate faults **during repeated encoded computation**, including a faulty restoration network. The inspected FCC definitions do not by themselves give that execution circuit or its acquisition rule. This is a difference in specified mechanism, not a historical novelty verdict. Related coded-computation and code-automorphism work may cover it.

## 2. Koopman invariant observables — actual original closure mechanism entered

Steven L. Brunton, Bingni W. Brunton, Joshua L. Proctor and J. Nathan Kutz, **Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems for Control**, *PLOS ONE* 11(2):e0150171 (2016). [Original article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150171).

Read original body sections “Background on Koopman analysis,” “Koopman invariant subspaces and exact finite-dimensional models,” and “Data-driven sparse identification of nonlinear observable functions,” especially equations (4)–(18) and the analytic/iterated-regression discussion. Also read the continuous polynomial-state lifting construction at equations (21)–(24) and the stated discussion of closure limitations. Some HTML equations were image objects; text statements and the recursive construction were readable. The candidate's finite F_2 formulas are independently specified and derived, not transcriptions of absent equation images.

SOURCE summary: Koopman acts on observables by composition with the state update. Choosing an invariant observable span yields a finite linear evolution in those coordinates. The paper derives or estimates additional nonlinear observables from known/identified dynamics and iterates the augmented regression toward closure. Thus nonlinear observable augmentation and closure are already actual computational methods. The paper's continuous-state restrictions must not be transplanted as theorems about a finite binary state space.

RECONSTRUCTION boundary: requiring that individual Boolean observables close by **coordinate permutation**, rather than by an arbitrary linear combination, supplies the Hamming-isometry property used in v3. The finite closure proof, eleven-coordinate example and faulty decoder are our hand construction. They are not source results or claims to invent the broader Koopman idea. Ordinary orbit construction is an exact same-information compiler rival.

## 3. What has actually been obtained

The candidate goes beyond v2's generic fixed-q code by making a nonlinear operation generate particular nonlinear observables and comparing two distinct representations' actual fault propagation. It does not establish efficient acquisition, favorable broad scaling or priority. Its local mathematical gain and global closure-cost boundary are separate. No parameter count, successful build or toy run is being offered as evidence.

The root-supplied S_N discriminator specifically attacks the exact pure-permutation closure representation. It does not prove that every learned code, every Koopman approximation, every context-dependent representation or every neural architecture is inefficient. Such alternatives would require new explicit laws and cost accounting; none is silently added to v3.
