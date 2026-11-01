# CD inspected primary sources and mechanism boundary

Only the passages listed here were read. This is not a systematic novelty search. Original food/ sources were untouched. Source claims and our inferences are separated.

## Interchange intervention training

Geiger et al. (2022), *Inducing Causal Structure for Interpretable Neural Networks*, ICML/PMLR 162. [Official paper](https://proceedings.mlr.press/v162/geiger22a/geiger22a.pdf). Local geiger2022-iit.pdf; actual selected-page extraction geiger2022-selected.txt. Read PDF pages 3–4 / sections 3 and Figure 1, including equation 4 and pseudocode lines 7–15. The opening/related-work passages were additionally inspected through official browser PDF extraction, not read end to end.

**SOURCE:** The objective compares aligned base/source-state interventions in a fixed high-level causal model and a trainable neural model. The high-level intervened output supplies the target. Figure 1 distinguishes correct ordinary outputs from agreement under state interventions; the update changes the neural realization while preserving the demonstrated task behavior.

**RECONSTRUCTION / effect on this proposal:** A local/counterfactual training signal that constructs causal organization already exists. V1 cannot claim this principle. The unresolved difference is learning the interface/model rather than receiving it. V1's finite transition observations are explicitly the substitute information source; no claim that they are free or weaker in all respects.

## Classical state assignment for sequential circuits

Villa and Sangiovanni-Vincentelli (1990), *NOVA: State Assignment of Finite State Machines for Optimal Two-Level Logic Implementation*, IEEE TCAD 9(9), 905–924. [Researcher-hosted original PDF](https://people.eecs.berkeley.edu/~alanmi/publications/other/nova.pdf). Local nova1990.pdf; actual selected-page extraction nova1990-selected.txt. Read PDF pages 1–3 / printed 905–907: introduction, encoding problem classes, multiple-valued minimization, and the start of exact hypercube embedding. No performance table was read/replicated; no reported percentages are used as our evidence.

**SOURCE:** Symbolic FSM states are assigned Boolean codes; their encoding changes the combinational next-state implementation cost. The paper formulates optimization from symbolic transition descriptions through Boolean representation and describes input/output encoding constraints. Its state variables are both inputs and outputs of the next-state relation. The classical method optimizes specific logic/PLA objectives and includes exact/heuristic algorithms.

**RECONSTRUCTION / effect on this proposal:** Our eight-state schema/codebook inference followed by local Boolean program synthesis is a small reconstruction of this substantive problem family. Calling its coordinates editable causal state does not establish a new mechanism. The tiny probe uses a changed-field-bit-width cost and full enumeration rather than NOVA's exact implementation; it is not a NOVA replication or comparative benchmark.

## Other consulted evidence and boundaries

The instructed C/D branch source reconstructions and attacks were read as provenance, including their original source locations. Their lens/DreamCoder comparisons were not independently rechecked end to end in CD. Their claims are not upgraded by agreement here.

Ahuja et al. (2023), *Interventional Causal Representation Learning*, [official PDF](https://proceedings.mlr.press/v202/ahuja23a/ahuja23a.pdf), introduction and setup/table through PDF p.3 inspected in official browser extraction. It studies identification under explicit decoder/intervention/support assumptions; no general claim of discovering arbitrary editable schemas is taken from it. It is contextual evidence that interventions need stated assumptions, not a theorem used to certify CD.

Search also identified De Micheli et al. (1985), *Optimal State Assignment for Finite State Machines*. Its search-extracted opening motivated reading the later NOVA source; no theorem or algorithm claim here depends on the unread remainder.

## Scope/provenance

Inherited user-authorized Ultra route, no exposed backend ID. The only numerical execution was the local inspected program through run_isolated.py, with network denied and credentials scrubbed. Web searches/downloads retrieved public papers; no external experiment or publication was performed. Source PDFs are allowed research artifacts, not planning documents.
