# Event state and computation — primary reading

Coordinator reading, 2026-11-01. No simulation or energy measurement. This supplements the completed V2 artifacts without changing their frozen conclusions.

## Gerstner et al., Neuronal Dynamics

Primary https://neuronaldynamics.epfl.ch/online/Ch1.S3.html ; entered §1.3.1–1.3.3 equations(1.5)–(1.14), selected input-response derivation. SOURCE: LIF retains voltage, integrates inputs and emits a threshold event followed by reset. Between events exponential decay can be evaluated from a saved state and timestamp. The book does not establish our proposed module-construction mechanism. Whole book unread.

## Wunderlich–Pehle, EventProp (2021)

Primary https://pmc.ncbi.nlm.nih.gov/articles/PMC8213775/ ; entered Results/Table1–2, equations(1)–(2), Discussion on critical spike events, Methods transition differentiation equations(35)–(38), and derivative-jump setup. SOURCE: exact gradients use a reverse-time adjoint and saved event information. Transverse threshold crossings permit implicit differentiation; spike-creation/deletion boundaries are singular. Event sparsity can reduce state bookkeeping, but this neither supplies unsupervised task credit nor proves hardware energy savings. No benchmark reproduced; remaining methods/proofs not certified as fully reconstructed.

## Independent reconstruction for the present gap

Use state q=(u,I,adaptation,weights,eligibility,delays/pending events,feedback state). Which components are needed depends on declared future operations. Replacing continuous units by threshold/reset units generally destroys the previous hidden-rotation equivalence: equal H does not imply equal coordinatewise event times. Thus spiking does not repair the old compactor by itself.

For a declared trace rule de/dt=−e/tau between events, retaining (e_at_last_event,last_event_time) permits exact lazy decay. This is an event execution implementation, not a new learning principle. A digital event queue can still be centrally scheduled; asynchronous model events do not prove physically decentralized execution.

If an update is w'=w+eta*m*e, states with identical retained w/u/I but e1!=e2 have different next weights whenever m!=0. A compact state deleting that distinction cannot reproduce both futures. Keeping e beforehand or acquiring NEW observations can supply information; local plasticity cannot reconstruct an erased historical trace from nothing. This elementary implication is our reconstruction, not a new theorem or a result attributed to EventProp.

Potential use: retain temporal edit state for a newly constructed module and update it only on relevant events. Main unsolved issue remains how to acquire useful new operations and credit routing under matched cost. Timing information can expand computation but does not by itself provide that acquisition law.
