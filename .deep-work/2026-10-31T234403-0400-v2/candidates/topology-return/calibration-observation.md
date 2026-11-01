# Paid calibration from quantum outcomes, not a direct gain oracle

ROOT CONSTRUCTIVE ACTION after actual review024. Same topology USER_LOCKED_MAIN. Manual analytical protocol only: no hardware, shots, simulation, training or unseen data. This instantiates one obtainable-observation model rather than narrowing U without evidence. It does not solve arbitrary nonlinear device calibration or claim quantum/learning advantage.

## Why not assume an amplitude-error observation

The abstract e0,L,r envelope does not say how it is learned. Direct queries y=e(h)+noise would be additional sensing access, not supplied by topology or by the present tools. A minimal quantum-outcome protocol instead starts with an explicitly assumed, restricted physical model and paid preparations/measurements. Finite observations cannot certify its uniform validity absent that model/regularity prior.

SOURCE actually entered by root: Granade et al., Robust Online Hamiltonian Learning, original arXiv1207.1655v2, §II and §VII.1–VII.2/Eq9–10. The body uses preparation/evolution/binary measurement likelihoods and distinguishes known versus unknown decoherence parameters. Root did not audit its algorithm's experimental curves or import its performance claims. The derivation below is our elementary protocol, not a new Hamiltonian-learning algorithm. The retrieved page's rendered date disagrees with the arXivv2 marker; cited identity is the original v2, not a claimed new2026 paper.

## One explicit observation interface

Assume all currently used correction channels have a COMMON positive multiplicative gain kappa in a justified prior interval [kappa_L,kappa_U], with 0<kappa_L and kappa_U<=3kappa_L. This common linear response is a strong model assumption across channel/amplitude/waveform range. A single-amplitude query does not establish it. Additive/nonlinear/timing/decoherence errors are excluded from this first derivation, not silently declared calibrated.

At a calibration idle configuration where the chosen pair gamma_i,gamma_j are zero modes, let B=i gamma_i gamma_j act as logical Z. Prepare its logical X+ state, apply a constant requested coupling h>0 for duration tau, and measure logical X. The code preparation, parity/routing/gap setting and X measurement are real required capabilities. They do not follow merely from writing a six-mode Hamiltonian and must be costed/available for every competing method. No stored unknown logical input is destructively reused as a calibration copy.

With a known common ground energy (irrelevant global phase), the likelihood is

    Pr(X+ | kappa,h,tau) = cos²(kappa h tau).

It is a binary Born outcome, not a direct readout of kappa. For channels involving the usual idle gapped partner, another charged calibration configuration is required; alternatively the common-gain prior transfers a measured channel only if that prior is justified. Independent channel gains require independent identifiable probes and a union confidence budget.

Choose the pulse so kappa h tau lies in[pi/8,3pi/8] over the WHOLE prior interval. Such a choice exists under the displayed prior ratio; it avoids aliasing. There

    |d Pr(X+)/d kappa| = h tau |sin(2kappa h tau)|
                       >= h tau/sqrt(2).

This identifies a finite informative experiment from actual permitted quantum observables. A phase/gain prior outside that monotone range needs a multitime/adaptive identification procedure, with its additional observations and computation; inversion is not unique for arbitrary unknown gain.

## Complete conditional query and control cost

Take n independent preparations/evolutions/readouts. Let p_hat be their observed X+ frequency. Under the stated likelihood, Hoeffding gives, with confidence>=1-delta,

    |p_hat-p| <= eps_p=sqrt(log(2/delta)/(2n)).

Intersect the probability confidence interval with the known monotone prior image and invert it, rather than pretending noisy frequencies always have an inverse. If the intersection is empty, record model/confidence failure and do not fabricate a narrowed interval. The resulting gain interval has a half-width no larger than

    eta_kappa <= sqrt(2)*eps_p/(h tau).

More conservatively, its full diameter is at most2sqrt(2)eps_p/(h tau); use the interval midpoint as kappa_hat. Command division by that positive estimate gives, on this same confidence event,

    |kappa/kappa_hat-1| <= eta_kappa/kappa_L,
    epsilon_gate <= A eta_kappa/kappa_L,    A=3pi/4,

for the concrete monotone braid family and its ONLY multiplicative-error model. Thus a sufficient fixed-time protocol for demanded gate norm error epsilon_gate is

    n >= [A²/(kappa_L² epsilon_gate² h² tau²)] log(2/delta).

This is an upper bound for this simple nonaliasing protocol, not an optimal information lower bound. Longer coherent/interative experiments can change the tradeoff. Calibration elapsed evolution alone is n*tau, in addition to n preparations/readouts and interval arithmetic/control normalization. A larger h or tau is not free hardware power, coherence or control access. Different channel gains multiply required calibration configurations and split delta; repeated drift needs renewed data or a justified drift envelope.

Known finite T2 changes the likelihood to the damped form in the original and attenuates its gain slope by exp(-tau/T2); this fixed-time shot bound grows accordingly by exp(2tau/T2). Unknown T2, SPAM bias and additional Hamiltonian terms require further identifiable controls/model bounds. A bounded probability bias b gives a nonvanishing calibration floor of order b/(h tau); increasing n alone cannot remove it. These are model/observation obligations, not evidence that topology fails.

## Connecting calibration to continuing SHARED learning

The shared controller's phase/gap profiles can now change while the same calibrated interval is retained ONLY while the common-response model remains valid over their amplitude/slew/context range. The actual admission rule uses the worst relative gain consistent with the paid interval and the old per-gate/word budgets. If a new waveform leaves the calibrated model domain, it requires further information, a justified extension or rejection. A newly acquired fast deadline joins the retained contracts.

This separates three things: topology protects the ideal braid in the fixed family; calibration supplies finite physical-error information; learning uses that freedom to improve a useful actual task. No component supplies the others for free. Every baseline gets the SAME prior, binary outcome access, prep/readout model and observations. A direct model-aware controller can reuse this scalar calibration too, so an N-independent calibration cost is not automatically a topology-specific advantage.

The present host has not supplied a physical device/these measurements. A later local simulation could implement this explicit Born-outcome interface, with all shots/preparation/propagation costs marked modeled and real numerical work counted. It would test a scoped learning/retention/resource hypothesis, not certify that real Majorana hardware has this response class. No such process is started here merely to verify a known binomial formula.

CURRENT OUTCOME: the error envelope is linked to an explicit quantum measurement protocol and a paid conditional observation/precision cost. Uniform nonlinear e(h), amplitude-dependent correction error and the general favorable-lifetime-cost claim remain OPEN. The next consequential comparison must use actual shared adaptation plus these paid observations against a direct robust-control strategy; it must not compare a paid learner to a baseline denied the physical prior, or a free certificate to costly replay.

Additional admission details: after command division, waveform amplitudes must stay inside the actually calibrated/hardware command range for every kappa_hat in the retained interval; reserve that range conservatively rather than hiding increased drive. The n bound uses a fixed predeclared shot count and constant gain during that batch. Optional adaptive stopping needs an anytime-valid confidence rule or a paid batchwise failure-probability allocation. For a constant common gain, one valid parameter-coverage event can cover all subsequent admitted profiles in the modeled domain; drift or a new amplitude/channel regime is a changed assumption requiring uncertainty expansion/new data. No independence across infinitely many controller updates is invented.

Independent follow-up audit read this entire actual protocol and reentered the original likelihood body. Its exact constants/interface conditions were checked in a real message, preserved in execution-024.md. A further conditional lifetime consequence follows: for worst-case word length L_max with required norm error epsilon_word, the current telescoping bound uses epsilon_gate<=epsilon_word/L_max. This fixed-time sufficient shot COUNT therefore scales as L_max² at fixed other factors. It is independent of the number of admitted contexts sharing the same valid gain event, not independent of demanded compositional accuracy. This is not an optimal physical information lower bound or a statement that every error-correction/calibration strategy has quadratic total cost.
