# Candidate v1 — acquire where finite learning effects propagate

AGENT PROPOSAL, 2026-11-01. Continues the main campaign after review005. Not a novelty or superiority claim. User's learning-association direction motivates this construction; the explicit behavioral transport/audit loop is an agent proposal, using established learned-optimizer, distillation and intervention components.

## Computational object and potential significance

Forward positions have independent neural parameters theta_j. Forward weight sharing is not required. The learned object is a rule for **transferring the effect of an experience**, rather than copying parameters or assigning a permanent shared owner. It can change later updates without changing the current forward function. If it acquired reusable edit semantics from few externally supervised experiences, it could reduce relearning across newly created internal computations without requiring fixed neuron-coordinate alignment. Neither that resource benefit nor paradigm-level value has been established. Generic contextual learned optimization can express the whole rule and is a serious alternative, not a universality objection.

## Exact state and transition

State: independent theta_1,...,theta_M; current executable consumer graph G; public unlabeled probes P; bounded externally supervised observation memories H_j; shared finite-effect predictor Q_psi; bounded audit replay. No teacher ingredient identity, desired sharing label or unobserved local cell target is a learner input.

1. An ordinary source-program loss supplies an actual gradient update to one active position i. Apply it first: B=Theta-eta E_i grad_i L_source. A measured packet is u_i=f_i(B;P)-f_i(Theta;P). This is a finite function change, not an assumed transferable raw gradient.
2. For each considered recipient j and amplitude a, start at the common B and perform r steps of local distillation:

       z_j = f_j(B;P)+a*u_i
       theta_j^(k+1)=theta_j^k-rho grad_theta [||f_j(theta_j^k;P)-z_j||^2/(2|P|)].

   The executed delta_j=theta_j^r-theta_j(B), and realized d_j=f_j(theta_j^r;P)-f_j(B;P), are recorded. Different hidden coordinates can implement the same requested behavioral change; the first pilot still uses a common cell parameterization and does not verify this extra portability claim. Every attempted distillation is charged, even when not selected.
3. H_j contains actual recent program input/output observations. From those observations and the current graph, compute observed external loss derivatives with respect to the current cell response: m_j(x)=e(x)c_j(x), k_j(x)=c_j(x)^2+e(x)h_j(x), where c_j=partial F/partial f_j and h_j=partial^2 F/partial f_j^2. Interpolate the sampled values onto P with a declared fixed kernel. This is a lossy history summary, not a sufficient statistic for arbitrary tasks or lost information recovery.
4. Q_psi receives current response, realized d_j, interpolated m_j,k_j, their products with d_j, and norms/work descriptors. It predicts the finite externally measured gain of that particular intervention. It receives no position index. Q is fitted to actual paired outcomes with squared loss; harmful and null cases are retained. For uniform exogenous training audits the importance weight is one. Adaptive deployment propensities are logged; the first pilot freezes psi at transfer rather than fitting a biased selectively observed objective.
5. Select an action from predicted positive gains with epsilon exploration (zero transfer is available). Query a fresh sparse batch of external consumer-program labels only AFTER the proposal and prediction. From exactly B, measure sham and edited losses on the SAME examples:

       Y = mean[.5(F_B-y)^2 - .5(F_(B+E_j delta_j)-y)^2].

   No accepted-only scoring: all selected proposals, including harms and abstentions, remain in the transcript. The pilot commits the proposed edit without oracle rejection, then every arm uses its purchased audit labels for the same native recipient SGD. H_j incorporates the newly observed examples. A negative effect changes future features through history; psi is fixed during transfer. Learning psi online is not tested here.
6. New independently initialized cells/consumer graphs initialize H through a paid small calibration batch. A shared Q transfers without teacher group IDs. After an environmental change, only newly observed labels can update history. Predictions made with current versus frozen stale history are compared at identical parameter checkpoints. Identical observable histories with opposite hidden targets cannot be resolved without new information.

Only one recipient is applied per event. Single-recipient effects are not summed as a certificate for a joint update.

## Worked complete process

A source cell changes its nine public responses after a real source-program SGD step. A recipient receives a requested half-size response change, runs the bounded local fit and reports the actual change, which can differ from the request. Its cached external error/consumer derivatives and the actual packet form Q's input. Q forecasts gain before a fresh label batch. The paid paired test yields Y, good or bad; the edit is committed in this pilot, native supervised learning then uses the audit labels, and future recipient histories change. On new positions the identical Q is used with new paid histories. The executable probe records this entire sequence, not merely Q classification accuracy.

For an affine-in-one-cell consumer with squared loss, expansion gives exactly Y=mean[-e*c*d-.5*c^2*d^2] on the measured batch. For a tanh consumer the finite change includes higher orders and off-probe effects. This motivates first-order, curvature and exact cached finite-replay alternatives; it does not prove that Q is needed.

## Strong alternatives and information boundary

MSR already learns sharing by reparameterization; MSR-Joint also adapts it within tasks. TAG already measures cross-task finite update effects. Distillation and learned optimizers are known. The proposed difference to test is continuing, state-dependent finite functional-effect routing among new independently parameterized internal positions from sparse external histories. This is an explicit organization of known operations, not substantiated novelty.

Controls in the pilot: no transfer with paid-label native learning; norm/action-family matched random propagation; first-order history score; gradient-plus-curvature history score; exact finite replay on the same cached labeled observations; learned Q. A contextual learned optimizer using Q as its action-value head is **exactly the same policy**, explicitly checked as an identity control. The pilot is not a reproduction of MSR, DEN, Net2Net or a trained GNN benchmark. It cannot establish superiority to those systems.

## Main technical gap

Useful information must be present in sparse observable histories. Distillation on fixed probes can hide off-probe damage, and Q may merely amortize cached replay or relearn a Taylor correction. Even good routing would not yet demonstrate discovery of a new operation type or consequential total-resource savings. The main-level judgment after execution must distinguish this finite capability from a new learning principle. Do not close solely on terminology overlap; inspect actual alternatives and the mechanism's strongest plausible result.
