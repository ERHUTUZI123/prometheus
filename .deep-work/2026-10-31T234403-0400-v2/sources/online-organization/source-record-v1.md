# Targeted original reading for physical return boundaries

These are checked technical passages, not abstract/name-based novelty exclusions. No paper code was run and no complete literature map is claimed.

## Marschall, Cho and Savin, A Unified Framework of Online Learning Algorithms for Training Recurrent Neural Networks

Primary technical text: https://arxiv.org/html/1907.02649 . Read §2.1 equations (4)–(6), §3.1 and §4.2 equations (24)–(26). RTRL carries the state-to-parameter influence matrix forward and obtains current-loss gradients from it. The text distinguishes past-facing influence accumulation from future-facing credit prediction. DNI predicts future credit and trains it through a bootstrapped recurrence; an exact current influence and a predicted future adjoint are different evidence/algorithmic objects. The proposed return derivative is a factorization/application of ordinary forward sensitivities, not a new RTRL identity. Prediction of a missing future adjoint would introduce an approximation absent from the exact worked process.

## Javed, Shah, Sutton and White, Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks (JMLR 2023)

Primary: https://www.jmlr.org/papers/volume24/23-0367/23-0367.pdf . Read §3.1–§3.3 and Figure 2. Independent scalar recurrent columns give sparse influences. Constructive learning freezes previous incoming/recurrent weights while adding a new recurrent feature; CCNs learn multiple independent new columns per stage, taking earlier frozen features as inputs. These constraints trade functional/learning flexibility for exact efficient gradients. Thus organizing neural computation to make online differentiation cheap is an existing research direction. The prospective distinction here is learned variable call/return boundaries and actual retirement of transient state, rather than fixed independent columns or stagewise freezing. No novelty follows until the same-information modular/program-learning rival is resolved.

## Siskind and Pearlmutter, Divide-and-Conquer Checkpointing for Arbitrary Programs with No User Annotation

Primary: https://arxiv.org/html/1708.06799 . Read §1's execution/program-interval distinction and checkpoint trees; §3.5–§3.6 interruption/resumption interface; the listed fixed-space/fixed-time/logarithmic tradeoffs. The implementation supports arbitrary execution points rather than only loop/call boundaries, storing restart capsules and recomputing intervals. Consequently unknown control flow or variable execution length is not a valid reason to exclude checkpointing from the control. A learned physical return differs from a checkpoint because it changes future access to forward information; checkpointing preserves/reconstructs the original computation. Both incur real data/work costs. The proposed comparison must give checkpointing the same source and replay privileges, and cannot treat rerunning nonretained external observations as free.

The separate reversible-RNN source was found during selection but not entered as a load-bearing technical result here; no detailed reversible-learning novelty conclusion rests on its abstract. Its role in candidate-v1 is a prospective matched control with explicit information/work costs.
