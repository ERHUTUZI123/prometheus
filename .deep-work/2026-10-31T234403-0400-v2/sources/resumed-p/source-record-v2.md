# Resumed P: targeted source boundary

Original technical passages were read through arXiv HTML. This is a targeted control check, not a complete novelty map. No paper code was run.

## Fong, Spivak and Tuyéras, Backprop as Functor (2019), arXiv:1711.10455v3

Primary: https://arxiv.org/html/1711.10455 . Read Introduction's missing-request argument; Definition II.1; Example V.4, Weight tying. A learner contains parameter space, implementation, update and request functions. Their composition constructs a learning algorithm for a composed implementation; the backpropagation theorem realizes ordinary gradient descent. Example V.4 treats shared parameters as one source whose update sums incoming requests. Therefore neither carrying learning semantics alongside a computation nor compositional handling of tied parameters is new. P's candidate cannot claim those principles broadly. Acquiring finite, conditional edit programs and changing their vocabulary is a different proposed operational detail, not a demonstrated novelty boundary.

## Andrychowicz et al., Learning to learn by gradient descent by gradient descent (2016), arXiv:1606.04474v2

Primary: https://arxiv.org/html/1606.04474 . Read Section 2, equations (2)–(3), and surrounding BPTT discussion. A recurrent optimizer emits updates and recurrent state, trained on a distribution of optimization trajectories. Intermediate losses make partial-horizon training possible; a stated approximation drops derivatives of optimizee gradients through the optimizer parameters. This is actual precedent for acquiring nonlinear stateful learning computations. A repair program's use of feedback, memory, meta-gradients or finite update sequences is not itself a novelty claim. The fair modern control should also receive graph/owner information and multi-parameter communication, rather than only the paper's coordinatewise restriction.

## Ellis et al., DreamCoder (2020 original preprint), arXiv:2006.08381v1

Primary: https://arxiv.org/html/2006.08381 . Read wake/sleep description, abstraction objective and refactoring discussion, including equation (1). Solved programs are refactored to expose reusable semantic patterns; library updates trade library size against refactored program length. A neural recognition model guides subsequent search, trained from replays and sampled programs. Thus growing a callable language, compressing successful traces, and improving later finite search are established mechanisms. P's direct control must allow analogous abstraction over its update DSL. Merely moving library learning to an optimizer DSL does not establish foundational value. Whether persistent ownership and recursively constructed repair protocols offer a distinct unresolved limitation is OPEN.

These source checks sharpen the strongest attack. No source evidence currently establishes that candidate-v2 is novel or that its promised organization yields a total-resource advantage.
