# Frozen nonlinear promotion check

Candidate: candidates/K/candidate-v3.md; strategic decision supervision/reviews/review-001.md. This is exploratory, not runtime-held-out evaluation.

Question: Can a task-conditioned construction act through H without recovering hidden coordinates, create genuinely nonlinear behavior, preserve subsequent training and support a second promotion? Does an ordinary minimum-width factor realization with the same morphism reproduce it?

Three independently updated states: source residual factors Z of width128 plus explicit nonlinear paths; compact residual H plus paths; minimum-width factor realization of H plus paths. Same information, batches, step size, promotion opportunities and rule. Each computes its own forward errors/gradients/selector. Source residual and minimum-factor updates use old factors, not compact H. Explicit path gradients are separately checked by finite differences at a nonsingular nonzero-gate state.

Fixed seeds 9101/9102/9103, m2,n3, batch24,100 SGD steps, eta0.005, promotions at steps8,55. Inputs uniform[-1,1], targets specified quadratic functions. Phi(z)=z^2. Target tasks switch at step50; all information available equally. Selector uses current error direction and error-weighted input second moment, eigenvector sign fixed deterministically. It receives no hidden factor identity, future labels or target coefficients.

Success: every step output, actual boundary gradient, residual Gram and explicit parameters agree within1e-9 relative/absolute max; both births preserve every tested current output within1e-10; final nonlinear finite-difference witness >1e-8; second promotion uses updated residual; wrong fold-to-H removes an available nonlinear gate derivative. Compare actual state counts, proposal counts and factorization cost class. No required efficiency or accuracy win. If the strong ordinary alternative matches, state that result.

Budget: ONE inspected process <=90s wall/CPU through existing offline isolated runner; <=3*100 batches, no sweeps/retries/tuning. JSON serialization smoke before work, incremental per-step evidence with flush, atomic summary. Stop on mismatch, failed finite-difference derivative, invalid PSD/rank or timeout. A failed program remains evidence; any correction needs a new version/contract. No training of a neural constructor, dependencies, network, purchase or old artifact mutation.
