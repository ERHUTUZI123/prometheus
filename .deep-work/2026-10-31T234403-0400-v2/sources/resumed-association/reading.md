# Original technical reading for the association construction

Root directly entered these original technical passages on 2026-11-01, not just their abstracts. No published benchmark was reproduced.

## MSR — SOURCE

[Zhou, Knowles and Finn, arXiv:2007.02933v3](https://arxiv.org/html/2007.02933v3), §4.1 Eq. (2), §4.2 Proposition 1, §4.3 Eqs. (3)–(4), Algorithm 1, and §5.1/Table 1 explanation. Eq. (2) factors vec(W)=Uv: U specifies sharing, v task parameters. Standard MSR updates v in task training, U through subsequent validation loss; test tasks freeze U. The finite-group representability result concerns a suitable U, not arbitrary learned U. MSR-Joint explicitly updates BOTH U and v on task training data; continuing adaptation is therefore not absent from all variants. We must not claim that learning a sharing matrix or adapting it within a task is a new contribution. Appendix approximations and numerical benchmarks have not been independently verified. Full source HTML previously saved as msr-2007.02933v3.html.

## TAG — SOURCE

[Fifty et al., arXiv:2109.04617v2](https://arxiv.org/html/2109.04617), §4.1 Eq. (1) and the definition immediately preceding it; introduction's explicit four-stage procedure. A source task updates shared parameters while recipient task heads and example batch remain fixed. Affinity is one minus the recipient post-update loss divided by its original loss. Positive affinity means improvement. Per-step measurements can be averaged before selecting task groupings, then the chosen networks are trained. Thus paired finite cross-task effects and asymmetric utility already have a direct precedent. Our proposed independently owned recipients, realized behavioral transports and sparse continuing history are details to develop and test, not evidence that this paper leaves a novel gap. No grouping theorem or experimental result was replicated; later theoretical conditions and appendices are unread for this resumed construction.

## Provenance boundaries

DEN/Net2Net original technical reading belongs to prior run records; not newly replicated here. ETAP arXiv:2602.18591 was found in search but its technical text is UNREAD and is not evidence for any mechanism claim in this pilot. P's history-derivation.md is an independent algebraic reconstruction, not a source result or an executed measurement. A generic learned optimizer's exact representation of this Q-based selector is a transparent algorithmic equivalence, not a proof that all future association mechanisms are uninteresting.
