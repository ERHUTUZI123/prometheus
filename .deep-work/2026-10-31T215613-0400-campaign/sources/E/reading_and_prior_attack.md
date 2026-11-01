# E: source-grounded reconstruction and closest-prior attack

Read after E/v0_frozen.md was written, except the explicitly marked entry-book readings. No other branch or prior Prometheus report was consulted. Labels distinguish source results from our reconstruction.

## Entry passages actually read

- Sterling & Laughlin, *Principles of Neural Design* (2015), printed pp. 105–108, 117–124, 399–404; PDF pp. 130–133, 142–149, 424–429. Load-bearing content: molecular state transitions, Fig. 5.5's AND sequence, ligand/receptor/G-protein activation and shutdown, material/time/energy constraints on learning. The authors' broad neurogenesis claims on p. 400 are not needed or independently verified here. Their loose registration-bit thermodynamic bound is not adopted.
- Pfeifer & Bongard, *How the Body Shapes the Way We Think* (2006), printed pp. 107–109 and 123–125; PDF pp. 132–134 and 148–150. Concrete change: task and niche delimit which physics performs useful work; a passive trajectory is not an algorithm that discovers its own morphology. Timing, reset and downstream coupling therefore belong inside the candidate's accounting.
- Peliti & Pigolotti, *Stochastic Thermodynamics* (2021), printed pp. 38–41, 60–62, 110–117; PDF pp. 61–64, 83–85, 133–140. Equations 3.2–3.9 distinguish equilibrium, manipulation, driving and intrinsic rates. Equations 3.76–3.81 require channel-resolved entropy bookkeeping. Equations 5.18–5.23 show energetic and kinetic discrimination; p. 112 distinguishes copying from reset. This motivates a physically consistent kinetic primitive without implying novelty.

Text extracted into *_selected.txt. Text extraction sometimes corrupts mathematical glyphs; equations used here were reconstructed against their surrounding definitions. No long quote is used in the research output.

## Original sources substantively checked

### SOURCE: Warren & Allen (2012)

*Steady-state parameter sensitivity in stochastic modeling via trajectory reweighting*, JCP 136, 104106. Author manuscript [arXiv:1202.4704](https://arxiv.org/abs/1202.4704), local PDF pp. 2–4, equations 4 and 6–11. These derive a path score whose increment is event propensity's log derivative minus integrated escape-rate derivative. For a rate parameter it reduces to event count minus integrated propensity, divided by that parameter. They explicitly warn that score variance grows with time, requiring more trajectories. Their ghost-particle construction only counts events; the compensator remains necessary.

ATTACK: E's proposed eligibility is exactly the log-rate form of this known sensitivity estimator, summed over its two fuel channels. Multiplication by scalar reward and a softmax resource Jacobian is a standard policy-gradient use, not a new learning principle. Any novelty claim based on that equation is defeated.

### SOURCE: Sartori & Pigolotti (2013)

*Kinetic versus Energetic Discrimination in Biological Copying*, PRL 110, 188101. [arXiv:1209.4566](https://arxiv.org/abs/1209.4566), local PDF pp. 1–5, equation 1, Fig. 2, equations 2–6. The three-state copying model separates barrier bias from bound-state energy bias. Initial error depends on barriers; equilibrium error depends on energies. In its single-step setting, combining them cannot beat the better limit; a kinetic optimum requires fast arrest, unlike an energetic optimum. Multi-step proofreading changes the possibilities under specified intermediate states.

ATTACK: useful transient computation with unchanged equilibrium energy minima is established physical reasoning. Our use of it in a learned circuit is an application, not a discovery of kinetic computation. Arrest/read/reset are operational resources, not harmless details.

### SOURCE: Fil & Chu (2020)

*A thermodynamically consistent chemical spiking neuron capable of autonomous Hebbian learning*, [arXiv:2009.13207](https://arxiv.org/abs/2009.13207), local PDF pp. 2–6, Table 1 and section 2.4. Input species make an internal-state species; a chain of binding sites creates a nonlinear learning signal; coincidence turns input into weight molecules. Their temporal-correlation examples require sufficient nonlinearity. Increasing chain length can starve free internal-state molecules because bound molecules are withdrawn from the signal pool, distorting the activation function.

ATTACK: adaptive catalytic material implementing neural and temporal learning already exists as a concrete CRN proposal, including a resource/nonlinearity limit. Replacing scalar weights with chemical amounts is not sufficient novelty. More decisively, our ideal product gate ignores binding depletion; the source makes a load test necessary, and forbids claiming effortless composability from catalytic nonconsumption alone. We do not infer that microreversibility alone guarantees global thermodynamic consistency for an arbitrary CRN; our elementary channel ratios are checked explicitly.

### SOURCE: Arredondo & Lakin (2022)

*Operant conditioning of stochastic chemical reaction networks*, PLOS Computational Biology 18, e1010676. [Publisher full text](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010676), local PDF pp. 4–7, 11, 14, 16–17; Fig. 3A and Methods. A stimulus-gated approximate-majority network makes probabilistic decisions. Catalyst weight molecules bias outcomes. Good/Bad feedback converts one conserved weight species into another, and the decision species supplies context. Feedback is supplied while the original stimulus persists. Their discussion explicitly identifies ambiguity between computation performed internally and by an arbitrary environment callback.

ATTACK: conserved catalyst pools plus environmental reinforcement predate this candidate. The remaining proposed distinction is credit through an internal multi-step trajectory, rather than reward on a still-present decision, with explicit physical channels and a downstream load test. That is a sharply specifiable extension, not yet a demonstrated original contribution. Their cited Banda et al. (2014) is an additional close precedent; no exhaustive novelty search is claimed.

## Judgment after attack

The broad novelty package in v0 fails: kinetic discrimination, molecular computational modules, adaptive catalyst weights, conserved resource pools and the eligibility estimator all have close precedents. Universal emulation is not the reason for this judgment; specific mechanism-level antecedents are. The finite remaining scientific opportunity is whether *construction of reusable conditional kinetic mechanisms* has a favorable discovery/composition tradeoff under physical loading and full training/reset cost. Neither source comparison nor a two-bit example settles it.

One high-upside branch remains open, without extending the present experiment: local physical devices might use a signed transition-count/dwell-time trace to allocate scaffold material over multi-step internal programs. A fully autonomous implementation of that trace and credit for adding a new state are unsolved here. If the small probe cannot learn the motif under the frozen trajectory budget, that failure is evidence against the current discovery procedure, not against all physical learning.
