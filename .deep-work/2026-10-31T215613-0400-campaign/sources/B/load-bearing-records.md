# Branch B: load-bearing reading records, frozen with candidate v0

Isolation: only campaign seed/contract, the Independent researcher role, the assigned technical books, public primary papers, and root-provided infrastructure information were read. No previous Prometheus architecture report or another branch's work was read.

Epistemic status: SOURCE records below support the specified statements, not the candidate as a whole. Complete PDFs were machine-indexed into this directory to locate chapters; substantive reading was selective. This is not a claim to have read four entire books.

## K1. Kelso, Dynamic Patterns, chapter 6

Local source: `food/Dynamic Patterns_ The Self-Organization of Brain and Behavior (Complex Adaptive Systems).pdf`. Source PDF is unchanged. PDF page numbers below are one-based; printed pages are given separately because the PDF contains duplicated/fragmented book pages.

Read printed pp. 160–164, 168–179 (PDF 198–202, 207–221; figure-only/fragment pages included). Central passages: printed p. 163 explicitly distinguishes existing intrinsic coordination tendencies from innate mechanisms and treats learning as changing those tendencies; pp. 171–174 describe scanning probes before/after 90-degree training, one-week retention, changes to neighboring phases and loss/creation of attractors; p. 179 reports an unpracticed symmetry partner and interprets transfer in terms of a relative phase relation.

SOURCE: The chapter argues for evaluating the initial coordination landscape and probes changes outside the practiced point. It reports behavioral transfer to a symmetry-related relative phase. This is a behavioral account, not a demonstrated neural mechanism that discovers arbitrary computational operations. The small motor study and its interpretation do not establish general neural compositional learning.

Changed assumption: I initially considered storing attractor states sufficient. After reading, I required the learned object to specify a relation that can be reinstantiated at different phases, and required separate tests for present operation, retained relation, and composition. The book itself does not supply the factor-discovery or composition rule below.

## I1. Izhikevich, Dynamical Systems in Neuroscience, chapter 10

Local source: `food/dsn.pdf`, read printed pp. 407–415, 417–420 (PDF 423–431, 433–436).

SOURCE: Equations (10.12)–(10.16), printed pp. 407–408, reduce weakly coupled stable oscillators to phases and then average to pairwise interaction functions. In slow time,

`phi_i' = omega_i + sum_j H_ij(phi_j-phi_i)`

with `H_ij` the cycle average of the adjoint response against the actual coupling. Printed p. 411, equation (10.17), gives the two-oscillator difference equation `chi'=omega+G(chi)` and the negative-slope stability test. Printed pp. 412–413 give the locking-range failure and distinguish smooth asymptotic convergence from special discontinuous relaxation models. Printed p. 415 describes the neutral global phase direction and sufficient stability/gradient conditions. Printed pp. 419–420 derive the quadratic integrate-and-fire PRC, including the finite-amplitude versus infinitesimal distinction.

Changed assumption: Arbitrary desired phase couplings cannot simply be asserted to emerge from ordinary weak pairwise synapses. I therefore explicitly budget a multiplicative higher-order interaction primitive. The candidate's exact phase-only dynamics are its defined computational substrate, not a reduction already proved for generic biological neurons. Global phase neutrality is a coordinate symmetry, not a missing instability.

## S1. Strogatz, Nonlinear Dynamics and Chaos, section 4.5

Local source: `food/11Nonlinear_Dynamics_and_Chaos_2018_Steven_H._Strogatz.pdf`, read printed pp. 106–108 (PDF 121–123).

SOURCE: The driven phase model is `theta_dot=omega+A sin(Theta-theta)`, with drive `Theta_dot=Omega`. The difference satisfies `phi_dot=Omega-omega-A sin(phi)`. Stable locking requires frequency mismatch smaller in magnitude than resetting strength; its phase offset is not generally zero. At the boundary the stable/unstable fixed points merge; above it the phase slips.

Changed assumption and calculation: A learned relational potential computes accurately only within a bandwidth/detuning margin. In the candidate's single free-port case, `e_dot=nu-K sin(e)`, hence the stable error is `asin(nu/K)` for `|nu|<K`, rather than exact zero for arbitrary streaming inputs. For `nu=0.2K`, the deterministic phase bias is approximately 0.20136 rad. The candidate must not advertise time-invariant algebra as instantaneous dynamic computation.

## G1. Gerstner et al., Neuronal Dynamics, chapter 19

Local source: `food/15Neuronal Dynamics From Single Neurons to Networks and Models of Cognition ( etc.) (z-library.sk, 1lib.sk, z-lib.sk).pdf`, read printed pp. 501–504 (PDF 514–517).

SOURCE: Equations (19.12)–(19.14) implement pair-based STDP with local decaying pre/post traces and event-triggered updates. The chapter then gives frequency/triplet limitations of pair-only rules, and equation (19.15) adds a slow postsynaptic trace to the potentiation product. This is a specific biological plasticity model, not a justification of arbitrary high-order correlation detectors.

Changed assumption: A learning rule should state the actual memory variables and observation events, not call an unspecified process Hebbian. The complex traces in the candidate are explicit sufficient statistics. Their higher-order products and the observed-versus-generated gate remain designed scaffolding.

## P1. Bick, Böhle, Kuehn, higher-order phase reduction

Primary source: https://arxiv.org/html/2305.04277 , sections 5.1–5.2, equations (5.4)–(5.6), and discussion. Read beyond abstract, particularly terms `sin(phi_i-2phi_k+phi_l+2alpha)` and `sin(phi_i+phi_k-2phi_l)` and tensors `a_kl a_ki`, `a_kl a_li`.

SOURCE: At second order, a pairwise coupled oscillator system can have effective nonpairwise phase terms. Their coefficients and hypergraph structure are constrained by the underlying graph and dynamical parameters. Three-body phase terms are not independently programmable free parameters of that original network. Stronger coupling or higher reductions create approximation and algebraic difficulties. Direct nonlinear multi-oscillator couplings can instead generate nonpairwise terms at first order.

Implication: The construction may legitimately posit direct nonlinear couplings as a candidate substrate. It cannot treat the existence of effective hyperedges as evidence that arbitrary learned four/five-body terms can be obtained cheaply from standard synapses.

## P2. Anwar et al., adaptive higher-order networks

Primary source: https://arxiv.org/html/2408.12235 , section III, equations (31)–(36).

SOURCE: The paper explicitly allows both pairwise and three-body coupling strengths to adapt, with slow relaxation driven by state-dependent functions; equation (33) updates a three-body tensor. The studied target is existence/stability of global synchronization, with specified topological and coupling conditions.

Implication: Adaptive higher-order oscillator networks are existing work. Neither adding hyperedges nor giving them Hebbian-looking adaptation is a novelty claim available here. The narrower candidate question is learned functional relations plus internally generated elimination shortcuts; equivalence to existing approaches still requires a dedicated prior-work comparison.

## P3. Wang, Wanjura, Marquardt, oscillator equilibrium propagation

Primary source: https://arxiv.org/html/2402.08579v1 , sections II and III.1, equations (1)–(7), not only abstract. Read the explicit XY energy, free/nudged equilibria, local parameter-gradient differences, XOR training and the multistability precautions.

SOURCE: Pairwise phase oscillators can be trained on prescribed input-output tasks using equilibrium propagation. Inputs are externally fixed, a nudge applies desired-output information, and differences between equilibria estimate parameter gradients. XOR is already a demonstrated task in this literature.

Implication: Learning XOR with oscillators would be no scientific contribution by itself. The branch uses continuous examples and reversible relation execution to inspect the candidate mechanism, not as a novelty benchmark. A serious comparison must include oscillator EP, high-order associative memory, Fourier relation learning, vector-symbolic binding/resonator systems, and constraint elimination.

## Retrieval limitations

The DOI landing page for oscillator EP was robots-blocked; its official arXiv full text was accessible and used. A search located Kymn et al., *Computing With Residue Numbers in High-Dimensional Representation*, DOI 10.1162/neco_a_01723, but substantive PMC access failed at a challenge page. No technical conclusion from that paper is treated as checked. It is a specific unclosed prior-work lead, not evidence of novelty or equivalence.
