# Original-source entry for after018 v2: reliable computation and learned codes

## Provenance and actual reading route

This direction was independently selected and sent to root before root supplied the Judd–Munro title. Root then supplied that exact nearest-source lead and later reported its own von Neumann/Gács reading. Those disclosures are source context, not independent verification by this worker. No current reliability candidate body or review was read before this construction. Earlier v0/v1 overlaps and P's query-information boundary were already known.

PDF skill was already read in this worker. Web PDF retrieval often failed. Existing `/opt/local/bin/gs` was available, so original PDFs were fetched into process memory and passed through Ghostscript with `-dSAFER` for text extraction or page rendering. No package was installed, no local PDF/image was saved, and no experiment was run. The source-processing commands are reading/rendering operations, not tests of the proposed mechanism.

## 1. von Neumann: executive and restoring organs are different jobs

J. von Neumann, **Probabilistic Logics and the Synthesis of Reliable Organisms from Unreliable Components**, in *Automata Studies* (1956), pp.43–98. [Original scanned chapter](https://www.mriedel.ece.umn.edu/wiki/images/a/af/Von_Neumann_Probabilistic_Logics_and_the_Synthesis_of_Reliable_Organisms_from_Unreliable_Components.pdf).

Actually rendered and visually read printed pp.70–72 (PDF one-based pages 28–30), §§9.1,9.2.1–9.2.2 and 9.2.3.1–9.2.3.2; Figure 30 and Figure 31; equation (14). Did not read the full original chapter or independently verify its later noisy-component bounds. Manchester re-typeset access timed out; the scanned original above succeeded in memory.

SOURCE summary: multiplexed messages use bundles and extreme activity fractions. An executive operation can worsen the error fraction, motivating a separate restoring operation. The restoration discussion explicitly requires appropriate randomness/independence among lines and warns that split lines are correlated. The displayed ideal majority map is `alpha* = 3 alpha^2 - 2 alpha^3`; below/above one half it moves toward the corresponding extreme. These statements are about the stated model, not automatic guarantees for an arbitrary learned noisy network.

What changed in the candidate: **RECONSTRUCTION** — code separation does not ensure safe computation between codewords. A reusable interface must include both an operation's error amplification and faults inside restoration. The seven-bit witness uses deterministic error-count bounds to avoid incorrectly iterating an independence formula across shared inputs. It does not claim the original majority scheme as new.

Root additionally reported reading other sections concerning noisy final restorers and Gács's later reliable cellular-automaton hierarchy. Those are not this worker's original readings. No claim about their detailed theorems is used to establish v2.

## 2. Judd–Munro: learning an error-correcting representation is already explicit prior work

Stephen Judd and Paul W. Munro, **Nets with Unreliable Hidden Nodes Learn Error-Correcting Codes**, *Advances in Neural Information Processing Systems 5* (NIPS 1992), printed pp.89–96. [Original paper](https://proceedings.neurips.cc/paper_files/paper/1992/file/c361bc7b2c033a83d663b8d9fb4be56e-Paper.pdf); [official record](https://proceedings.neurips.cc/paper_files/paper/1992/hash/c361bc7b2c033a83d663b8d9fb4be56e-Abstract.html).

Actually read the complete extracted original body, especially §2 methodology pp.90–91, §§3.2–3.3 pp.92–95, and §4 discussion. The PDF fetched successfully in memory (1,731,070 bytes) and Ghostscript text extraction exited successfully. Plots were not visually inspected, so no new quantitative plot inference is made.

SOURCE summary: feedforward encoder experiments inject sign-flip misfires into a hidden channel layer during training; the network learns coding/decoding representations. The structured six-channel example distinguishes fault patterns constrained to two triplets and uses an additional fault-free hidden decoding layer. The paper also discusses input frequency/corruption structure and code distances. It therefore covers the broad idea that noisy neural training can acquire error-correcting codes; v2 cannot claim that idea.

Useful borrowed mechanism: train with the relevant corruption instead of treating robust codewords as an external gift. The concrete limits relevant here are a channel/reconstruction setting and protected other layers in the inspected construction. V2's proposed operation/repair contract includes faults inside its own repair network and asks for repeated computational composition. That is a describable setting/mechanism difference, not proof of historical novelty; later work was not surveyed.

## 3. Negative access and inference boundaries

Evans–Schulman, *Signal Propagation and Noisy Circuits* (1999), was located, but author-hosted retrieval failed through the web route and a local TLS certificate check; library download returned 403. Its substantive body was **not** entered and no theorem from it is claimed as checked. The original von Neumann pages and complete Judd–Munro body above are the actual source basis.

No learned-coding result proves this candidate's acquisition efficiency, unseen-state coverage, all-fault local contract, or comparative significance. The candidate's weighted-cut growth rule, exact score-function interface, seven-bit reconstruction and finite-depth union bound are our explicit construction/derivations. The latter remain a known-principle reconstruction rather than a new reliability theorem. Threshold coding, noise-aware training and fault-tolerant compilation are real rivals, not disqualifications by universal simulation.

No training, theorem prover, numerical probe or implementation was run. Effective mode remains unknown; filesystem/context isolation is instruction_only. The source and candidate are new artifacts inside the existing V2 run; earlier files are unchanged.
