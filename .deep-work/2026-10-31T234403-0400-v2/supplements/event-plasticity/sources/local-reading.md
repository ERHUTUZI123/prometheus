# Local plasticity: selected original technical readings

Read-only supplement to the existing run. SOURCE below means an original technical passage was inspected, not that its simulations or biological interpretation were independently reproduced. No training, installation or experiment was run.

## SOURCE — Bellec et al., 2020, e-prop

*A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications 11:3625, DOI 10.1038/s41467-020-17236-y. Original published PDF: <https://igi-web.tugraz.at/PDF/248.pdf>; original full text: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/>.

**Locators:** main Eqs. (1)–(4); Methods pp. 10–12, Fig. 6, Eqs. (13)–(14), (23)–(29); reward section pp. 12–13, Eq. (5), (30) onward.

The factorization is `dE/dW_ji = sum_t L_j^t e_ji^t`, with ideal `L_j^t=dE/dz_j^t`. Local eligibility uses `epsilon_t=D_h M epsilon_(t-1)+D_W M`, `e_t=D_h f epsilon_t`. Hidden-neuron states communicate through observable outputs in the declared graph. The local factor is exact there; the ideal L generally needs future/network credit. Practical online e-prop approximates L with neuron-specific output-error feedback, low-pass filtering eligibility for readout dynamics. Spiking implementations use pseudo derivatives; this is not an ordinary exact derivative of a hard threshold. Supervised targets or environment rewards supply task feedback; reward e-prop additionally uses actor–critic/neuron-specific signals. These equations update existing weight slots, not allocate edges. Per-synapse traces are retained learning state; slow neuronal adaptation can lengthen them. The reset/hidden-state conventions must match the stated graph.

## SOURCE — Izhikevich, 2007, dopamine-modulated STDP

*Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling*, Cerebral Cortex 17:2443–2452, DOI 10.1093/cercor/bhl152. Original author PDF: <https://www.izhikevich.org/publications/dastdp.pdf>.

**Locators:** Materials and Methods, pp. 2444–2445, Eqs. (1)–(2), Fig. 1 and its reward-delivery protocol.

The model retains synaptic tag c: `dc/dt=-c/tau_c+STDP(delta_t)*delta_event`, and changes strength s through `ds/dt=c*d`, where d is dopamine. Temporal pre/post coincidence writes the tag; later modulation gates its consolidation. The demonstration's reward is delivered by an external programmed contingency, including selected spike-order events; the local rule does not discover what counts as success. Its network/noise assumptions are part of the demonstration, not a universal causal-credit theorem. These equations change strengths on supplied connections; they are not an edge-allocation rule or a mechanism for recovering deleted traces.

## SOURCE — Zenke and Ganguli, 2018, SuperSpike

*SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks*, Neural Computation 30:1514–1541, DOI 10.1162/neco_a_01086. Original author manuscript: <https://zenkelab.org/wp-content/uploads/2018/03/ZenkeGanguli2018_SuperSpike.pdf>.

**Locators:** §2, manuscript pp. 4–5, Eqs. (1)–(5); §3.1 Eqs. (6)–(7); §4.1 and feedback construction, pp. 13–16.

Output targets define a filtered spike-train error. The update combines that error with eligibility `lambda_ij=alpha*[sigma'(U_i)*(epsilon*S_j)]`. It replaces the discontinuous spike derivative by a voltage-dependent surrogate and neglects a reset/self-history contribution in the displayed derivation, motivated by low firing rates. Hidden units receive routed feedback (symmetric/random/uniform variants). Voltage dependence allows learning in allocated units that initially emit no spikes; it does not create an absent unit or its input trace. LIF propagation remains weighted presynaptic spike arrivals, current/membrane dynamics, threshold/reset and delays. The inspected learning rule changes weights, not topology.

## SOURCE — Kappel et al., original author preprint v4, 2018

*A dynamic connectome supports the emergence of stable computational function of neural circuits through reward-based learning*, arXiv:1704.04238v4, January 2018; later eNeuro DOI 10.1523/ENEURO.0301-17.2018. Checked version: <https://arxiv.org/pdf/1704.04238>.

**Locators:** pp. 5–7, Eqs. (1), (5); p. 10 on disconnected synapses; Methods pp. 25–26, Theorem 1; pp. 28–32, Eqs. (15)–(16), (21)–(24).

A supplied potential-edge scaffold has latent theta per edge; the edge is functional for theta>0 and absent for theta<=0. Dynamics combine a structural prior, reward-gradient estimate and Wiener noise. Disconnected slots receive no activity/reward gradient: prior/noise or a waiting-time approximation causes birth. Active-edge eligibility and reward averages estimate the gradient. The stationary-distribution theorem assumes a strictly positive, continuous, twice continuously differentiable posterior and positive diffusion/learning parameters; the practical online estimator additionally assumes slow parameter change. The paper notes its threshold-map derivative approximation. Thus literal rewiring is explicit, but occurs within retained potential endpoints, not unlimited neuron/program invention; neither theorem nor reward modulation guarantees finite-budget discovery or lossless retention.

## Retrieval and scope

Nature's direct HTML endpoint redirected to inaccessible authorization; its published author-hosted PDF and PMC copy were read instead. eNeuro's full-text endpoint returned 403; Kappel's original author preprint v4 was inspected, without assuming identity to every final published detail. No Clopath-specific structural claim is made: a dedicated structural source above supplies the literal birth/deletion mechanism. This worker did not investigate the supplement's separate SAM source or make hardware-energy claims. Host/provider-effective mode remains unknown.
