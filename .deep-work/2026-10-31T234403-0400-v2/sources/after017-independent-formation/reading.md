# Reading actually performed before independent formation v0

Scope: same V2 campaign; independent neutral-input pass. Read AGENTS.md, README.md Research Vision, and deep-work/WORKFLOW.md. No current candidate, current review, author code, old report, old campaign artifact, or raw-book content was opened. Provider-effective mode unknown. Shared filesystem isolation is instruction-only. No experiment/training was performed. Sources were read through the browser text extraction of primary PDFs; no PDF was saved or modified. The broad initial filename search returned names outside the project but no contents from them were read.

The available hf-cli research-reading skill was read. Its instructions did not require any installation or account action for this task; outside sources below were read directly from primary URLs. No HF command, authentication, cloud job, download, or paid API was used.

## 1. Moerman, Sammartino, Silva, Klin, Szynwelski — Learning Nominal Automata

Primary PDF: https://arxiv.org/pdf/1607.06268 . PDF has 15 pages; printed page numbers match PDF index plus one.

Actual body reading: printed p. 1, introduction paragraphs contrasting active and passive learning; p. 2, observation-table closedness/consistency and transition definition; p. 4, equality-orbit example and §3 support/orbit/equivariance definitions; pp. 6–7, Theorem 2 termination proof and Figure 2 discussion of support growth and broken local symmetry. Search excerpts also exposed portions of query-complexity statements; I did not reconstruct those bounds.

Original notation entered: atom set `A`; permutation action `pi . x`; `supp(x)`; `orb(x)`; dimension; observation rows; `nu L*`. Checked mechanism: counterexamples can reveal larger support or remove a symmetry, rather than merely add states. Assumptions include orbit-finite state and active teacher queries. I do not import the termination theorem into a growing graph with unbounded support.

Effect on candidate: separate finite local equality-pattern orbits from potentially unbounded global relational support. The candidate's support impossibility argument is my reconstruction, not a theorem attributed to this paper. It prevents a false acquisition guarantee. Unread: full nondeterministic algorithm proofs, implementation, empirical section, and references' primary bodies.

## 2. de Falco — An Explicit Framework for Interaction Nets

Primary PDF: https://lmcs.episciences.org/1108/pdf . 27 PDF pages; printed pagination starts at 1.

Actual body reading: printed pp. 1–3, principal-port reduction and the port-fusion issue; printed pp. 13–14, Definition 5.2, Proposition 5.3, proof, and Corollary 5.4. The proof's fresh-port disjointness requirement was read, including the renaming qualification.

Checked mechanism: distinct active pairs are disjoint; applying both replacements gives a common result, and a unique rule per symbol pair gives strong confluence up to renaming. Original objects `R`, `R1`, `R2`, replacement redex/net pairs, and interface gluing were inspected. I did not reconstruct the paper's complete execution-formula machinery.

Effect on candidate: wire splicing must be defined even for boundary-to-boundary edges; deterministic local rule sampling preserves the relevant commuting argument. Confluence is not evidence that truncated executions meet an endpoint budget. Unread: most of §§2–4, proof-net correspondence, categorical development, and implementation details. Lafont's original 1990 article was found bibliographically but its primary body was not opened; no specific theorem is credited to an unread Lafont passage.

## 3. Kurach, Andrychowicz, Sutskever — Neural Random-Access Machines

Primary PDF: https://arxiv.org/pdf/1511.06392 . 17 pages, ICLR 2016 version.

Actual body reading: §3, printed pp. 2–3, controller/register/modules and Eq. (1); §3.1–3.2, p. 4, restricted controller input and memory READ/WRITE; §3.3–3.4, pp. 5–6, endpoint loss and discretization discussion. The controller-table precomputation passage was also inspected.

Original objects: `R` registers, `M` memory size/matrix, module outputs, controller-produced softmax circuit choices, `P(r_i=0)`, READ `M^T p`, and halting distribution `p_t`. Checked rival property: pointer values can become exact in discretized execution; an alleged hard-wiring benefit cannot rely on denying that baseline.

Effect on candidate: use an exact-pointer learner as the strongest memory rival and give it the same structured exploration. Unread: complete appendix, experimental setup/results, and all task-specific module tables. I make no independent performance or replication claim.

## 4. Kaiser and Sutskever — Neural GPUs Learn Algorithms

Primary PDF: https://arxiv.org/pdf/1511.08228 . 9 pages, March 2016 version.

Actual body reading: printed pp. 2–3, unaligned arithmetic discussion and §2 GRU/CGRU state-update equations. Original state shape `[w,h,m]`, shared convolutional kernels, update/reset gates, and input initialization were entered.

Effect on candidate: a fixed lattice can still supply input-dependent working memory, so a finite-hidden-vector lower bound does not disqualify this rival. Unread: full optimization procedure, ablations, and test results; no quantitative result is used.

## 5. Plappert et al. — Parameter Space Noise for Exploration

Primary PDF: https://arxiv.org/pdf/1706.01905 . 18 pages, ICLR 2018 version.

Actual body reading: printed p. 2, policy/trajectory framework; pp. 2–3, §3 episode-fixed parameter perturbation and state-dependent exploration argument. Original update `theta_e = theta + N(0,sigma^2 I)` was inspected.

Effect on candidate: keeping a sampled operational law fixed through a rollout is not a new general exploration principle. The pointer rival receives the same opportunity. Unread: full algorithm adaptations, empirical comparisons, and appendices. My discrete per-descriptor table distribution is not claimed as this paper's algorithm.

## 6. Ellis et al. — DreamCoder

Primary PDF: https://arxiv.org/pdf/2006.08381 . 22 pages. This PDF is the conceptual/main article version served at the URL; no supplemental implementation text was assumed present.

Actual body reading: printed pp. 4–5, wake/sleep specification and Eq. (1); pp. 6–7, recognition training passage and Figure 3 refactoring-to-map example. Original objects `rho_x`, library `L`, task likelihood `P[x|rho]`, program prior `P[rho|L]`, and recognition proposal `Q(rho|x)` were entered.

Checked nearest-precedent property: it jointly improves program search and acquired reusable abstractions. Effect on candidate: naming learned operations or lowering search depth through reuse is insufficient differentiation. The remaining proposed distinction is learning a low-level routing law without a supplied semantic evaluator for binding. Unread: supplemental derivations, domain-specific primitive inventories, exact experimental resources, and quantitative evaluation. No claim that the proposed method is better is supported by this reading.

## Evidence limits and provenance

All scientific prose labeled RECONSTRUCTION in candidate-v0.md is newly derived in this pass. The memory/support arguments are mathematical arguments with explicit assumptions, not formalized or empirically checked results. The acquisition update is an unexecuted algorithm specification. No successful table, learned routing protocol, sample-efficiency advantage, or novel mechanism has been demonstrated. A second model's agreement would not repair these missing forms of evidence.

Search was targeted, not a complete landscape survey. I did not read the neural interaction-net or learned graph-rewriting literature exhaustively and do not infer novelty from its absence here. The intended comparison phase begins only after the candidate file is saved and its hash is sent to the root coordinator.
