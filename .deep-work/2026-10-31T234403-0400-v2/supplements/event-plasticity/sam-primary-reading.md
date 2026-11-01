# SAM primary reading — 2026-11-01

Read by the inherited host agent; provider-effective mode unavailable. No Pro claim. Source reading and algebra only; no training, installation, downloaded-code execution, or paid route. This supplements the existing V2 run.

## Checked sources and passages

- **SAM v2:** Qianli Liao, Liu Ziyin, Yulu Gan, Brian Cheung, Mark T. Harnett, Tomaso Poggio, *How Biological Synapses Self-Assemble Gradient Learning*, arXiv:2412.20018v2, 12 October 2026. [Version metadata](https://arxiv.org/abs/2412.20018v2). Independently read [§3.1–3.2](https://arxiv.org/html/2412.20018v2#S3.SS1), Eqs. 1–7; [Appendix A](https://arxiv.org/html/2412.20018v2#A1), Eqs. 8–58, especially Assumption 1, Lemmas 1–3, Theorem 2 and proof I1–I6; [Appendix F](https://arxiv.org/html/2412.20018v2#A6), Algorithms 1–2; §3.9, Appendix C; Appendix E's divergent-run discussion. Local original HTML: `sources/sam-v2.html`; parsed reading aid: `sources/sam-v2.txt`.
- **Earlier version:** same six authors, *Self-Assembly of a Biologically Plausible Learning Circuit*, [arXiv:2412.20018v1](https://arxiv.org/html/2412.20018v1), 28 December 2024. Read §2.2 and Appendix A, Eqs. 13–40. Local HTML/text: `sources/sam-v1.*`.
- **Closest reciprocal-learning predecessor actually read:** Mohamed Akrout, Collin Wilson, Peter C. Humphreys, Timothy Lillicrap, Douglas Tweed, *Deep Learning without Weight Transport*, NeurIPS 2019, [original proceedings PDF](https://papers.nips.cc/paper_files/paper/2019/file/f387624df552cea2f369918c5e1e12bc-Paper.pdf), §5.1–5.2, Eqs. 13–18 and Figure 2. [ArXiv HTML](https://arxiv.org/html/1904.05391), saved as `sources/sam-prior-akrout-1904.05391.html`. The 1994 Kolen–Pollack original was identified but not independently fully audited here.
- **Feedback alignment:** Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, Colin J. Akerman, *Random synaptic feedback weights support error backpropagation for deep learning*, Nature Communications 7, 13276 (2016). Read [original full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5105169/), Results “Random feedback weights can deliver useful teaching signals,” Eq. 1, and “Insight into the mechanics of feedback alignment,” Figure 5. Browser retrieval supplied the article; direct local fetch returned an access page, retained honestly as `sources/sam-prior-feedback-alignment.html` (not an article copy).
- **Topology-change companion, narrow reading:** Liu Ziyin, Isaac Chuang, Tomaso Poggio, *Heterosynaptic Circuits Are Universal Gradient Machines*, [arXiv:2505.02248v1](https://arxiv.org/html/2505.02248v1), 4 May 2025. Read §2.4 and Appendix D.6, Eqs. 85–90, plus explicit fixed-port description. Saved as `sources/sam-companion-2505.02248v1.html`. The companion's broad universality theorem was not audited.

## SOURCE — concise record (v2)

Forward inference precedes feedback; output feedback is the loss's negative output gradient. Four dense, fixed-size matrices are plastic. Cross-stream maps modulate updates rather than drive inference. Theorem assumptions: ReLU gates, full-column-rank V (downstream width ≥ upstream), matched interstream rates/decay, identity output connector, exact output cotangent, bounded pathway/interstream mismatches. The proof telescopes mismatches into an explicit residual. Only reciprocal interstream mismatch decays without a pathway-alignment hypothesis; pathway convergence remains unproved. Bounded feedback maps support the asymptotic PSD-prefactor statement. Algorithm 2 supplies gated theoretical updates; benchmark Algorithm 1 omits interstream postsynaptic gates. Appendix E reports divergence despite good interstream matching. Topology growth/pruning is attributed to the companion.

Mathematical transcription in renamed symbols: a=D h, b=downstream activity; F=W, R=bar-W, G=V, C=bar-V. All variables refer to the same sample before update:

\[
\begin{aligned}
\Delta F_l&=\eta_F D_{l+1}C_l b_{l+1}a_l^T-\gamma F_l,\\
\Delta R_l&=\eta_R\bar D_lG_l a_l b_{l+1}^T-\gamma R_l,\\
\Delta G_l&=\eta_G b_l a_l^T-\gamma G_l,\\
\Delta C_l&=\eta_C a_{l+1} b_{l+1}^T-\gamma C_l.
\end{aligned}
\]

## SOURCE — revision boundary (v1)

V1's theorem assumes full-rank cross-maps, downstream overparameterization, and small interstream updates near stationarity. V2 replaces that presentation with explicit finite pathway/interstream error budgets and PSD gated preconditioning. V1 should not be quoted as though its approximate-stationarity premise proved pathway convergence from arbitrary initialization. V2's conditional theorem governs this supplement.

## SOURCE — matched predecessors

Akrout et al. §5 constructs reciprocal local updates driven by the same forward activity and error signals. Matched rates and decay make forward/feedback weight differences contract. Feedback alignment instead fixes random feedback weights and changes forward weights; its primary explanation traces feedback information through the changing input weights into output weights. Neither source's reported performance is independently verified here.

## SOURCE — actual topology mechanism and reporting ambiguity

The companion's four-neuron regression uses fixed input/output/error roles, a binary connectivity mask, recurrence capped at two firings, and first/last activity products for weight updates. An outer evolutionary loop mutates/selects masks using post-training loss; the local inner rule updates weights. This is explicit structural selection, beyond main-paper SAM's four matrix updates. The text is insufficient for a faithful replication without repair: §2.4 specifies 100 inner steps, D.6 specifies 50; D.6 defines fitness as negative loss but says to select the lowest fitness; mutation density uses H where the mask is M. These are source ambiguities, not evidence that the reported experiment failed.

No direct quotations were used. Original downloads remain evidence artifacts; the records above distinguish their read scope from our deductions in `sam-mechanism-judgment.md`.
