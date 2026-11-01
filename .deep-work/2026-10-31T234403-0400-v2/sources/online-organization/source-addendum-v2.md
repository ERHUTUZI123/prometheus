# V2 constructive primary-source addendum

Original technical passages read through the linked primary manuscripts. No paper code was executed. These sources supply ingredients and strong actual comparisons; they are not a complete novelty survey.

## Chung, Ahn and Bengio, Hierarchical Multiscale Recurrent Neural Networks

Primary: https://arxiv.org/html/1609.01704 , v7. Read §3.2 equations (1)–(9), including gated top-down/bottom-up paths; §3.3's boundary training. UPDATE incorporates new lower-level information, COPY preserves states, and FLUSH ejects a summary then hard-resets the old layer state. Boundary variables are learned; the reported training uses a straight-through estimator. Therefore learned returns, resets and input-dependent segmentation are already explicit neural mechanisms. V2's proposed difference must be the observed-consumer-credit formation objective, procedural/readset changes and measured live learning-state cost; it cannot claim ejection/reset or unsupervised boundaries broadly.

## Graves, Adaptive Computation Time for Recurrent Neural Networks

Primary: https://arxiv.org/html/1603.08983 , §2 equations (3)–(11). ACT generates intermediate recurrent states and halting activations, uses a normalized halting-weighted mean of states/outputs, and adds a ponder cost based on update count plus remainder. This is a concrete existing task/resource pressure that learns computation depth. It supplies both a controller baseline and a design lesson: charge actual computation and specify how a relaxed halting object corresponds to the executed state. V2's future-credit storage/lifetime pressure is a proposed additional formation objective, not a claim that adapting internal work from task feedback is new.

## Otto, Padovan and Rowley, CoBRAS

Primary: https://arxiv.org/html/2207.14387 , v4, §2 equations (4)–(20), Theorems 2–3 and §2.2. State covariance and sampled gradients of future output sequences determine an oblique reduced coordinate. Factorized covariance balancing uses an SVD of the state/gradient snapshot cross matrix; random output projections provide adjoint samples. The stated output-error bound requires its conditional Gaussian/smoothness assumptions and concerns the specified distribution/output family. This is direct prior art for selecting coordinates with future sensitivity rather than state variance alone. V2 can use it as a paid probation initializer; it must not present future-gradient basis selection, nonlinear kernel features or that scoped bound as its own result. Its proposed additional object is a learned procedural boundary/readset/port formation rule under live credit costs and new consumers.

## Willcox, Ghattas, van Bloemen Waanders and Bader, goal-oriented model-based reduction

Primary: https://acdl.mit.edu/bgvw_cdc05.pdf , §III equations (18)–(30). The reduced basis is selected by minimizing output discrepancy subject to reduced governing equations; reduced state and adjoint solves yield basis gradients. The text distinguishes reconstructing stored state snapshots from accuracy of the reduced dynamics/output. This distinction is load-bearing here: small local port-gradient/reconstruction residual does not certify a valid reduced executable continuation. V2 must train/test its actual port consumers and future outcomes, while charging the state/adjoint data needed for formation.
