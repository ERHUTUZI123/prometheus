# Opaque-name interface: one useful invariant, and what it does not buy

AGENT PROPOSAL / RECONSTRUCTION by root to support raw-stream mechanism formation, not a claim of new nominal semantics. No code or experiment. Root entered Gabbay/Pitts conference original §§2–3 (PDFpp2–4) Eq2/finite-support/freshness passages; journal PDF extraction was corrupt and is NOT claimed read. Source https://www.gabbay.org.uk/papers/newaas.pdf .

## Actual optional state/transition revision
Let A be a countably infinite set of opaque atoms. An agent carries finite-precision code z and at most r atom references. Reserved syntax tokens are separate. A neural pair kernel receives the two z codes and the equality matrix among their at-most2r atoms. It never receives their host integer addresses, order, spelling or learned embedding. An emitted bounded graph specifies new z codes and atom-reference selectors: copy a parent payload slot, or allocate a fresh atom. A fresh atom is a new identity outside all currently used identities; this is a generic reference primitive, NOT a BIND/SUBSTITUTE/LOOKUP operation. Heap/data pointers, exact equality and fresh allocation have charged finite representations; an infinite atom set is a mathematical abstraction, not free memory.

This changes what information reaches a learned local law. It can still learn arbitrary wrong control logic, but it cannot exploit identifier spelling as a semantic shortcut. The binding/environment graph still has to be constructed by execution.

## Checked finite equivariance argument
Fix theta and an input graph. Apply a bijection pi to the finite set of input-name atoms. Couple the same discrete neural choices on corresponding local events. At every finite step, corresponding pairs have identical z and equality patterns, hence choose the same code/wiring/selector outputs. Copy selectors preserve the partial bijection. When each execution allocates a fresh name, extend the finite bijection to pair those two unused names. Pointer gluing respects correspondence. Induction gives isomorphic code/port states modulo this growing atom bijection after every corresponding event. Complete normal forms therefore give name-covariant outputs when the decoder also only copies/tests equality and task semantics respects those names.

The guarantee is **up to renaming newly allocated atoms**, not strict equality under one predetermined global permutation. There cannot be a deterministic fully equivariant choice fresh(empty): a permutation moving its chosen atom while fixing empty would contradict equivariance. Concrete lowest-integer allocation can therefore be hidden behind the abstract interface only when its integer is unobservable and new names are compared modulo fresh renaming. Fixed-width identifiers can exhaust; stop explicitly rather than silently alias.

## Crucial limitation: global name equivariance is NOT learned lexical binding
Even a wrong lookup policy that returns the earliest matching binding is globally name-equivariant. Consider distinct ground values A,B:

 P = ((lambda x. ((lambda x.x) B)) A)
 Q = ((lambda x. ((lambda y.y) B)) A).

P and Q are alpha-equivalent by renaming only the inner bound x. Correct lexical execution returns B for both. An earliest-binding-with-that-name policy returns A for P and B for Q. It nevertheless commutes with every *global* atom permutation: matching equality and binding order are unchanged.

Thus the opaque-name interface neither supplies scope nor proves full alpha-equivalence behavior. A scoped alpha conversion is not generally a global name permutation when the same spelling is shadowed. Any claim that alpha-correct binding was acquired needs actual shadow/capture/independent-use evidence, rather than renamed-token success alone. This is a central formation discriminator, not a proposed new benchmark or a theorem against neural approaches.

## Prospect and matched alternative
This provides a clean acquisition bias over names and protects the test from vocabulary memorization; it does not establish a foundational principle. A conventional register/pointer learner can use the same copy/equality/fresh interface. NeuralGPU input preprocessing can canonicalize global identifiers, and must be charged/given the same information if used. The remaining question is whether local generated semantics acquires correct *scope and reusable function organization*, not whether names were canonicalized. Keep this invariant as scaffold rather than the claimed discovery.
