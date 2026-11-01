# Exact coordinate-isometric execution: a consequential representation cost bound

RECONSTRUCTION / ATTACK / constructive boundary, same V2. Derived after the constructor reported its substantive v3 direction and before root read a saved v3. The independent supervisor checked the group/orbit calculation from the stated mechanism; no saved review020 is presumed. This is a mathematical derivation, not a machine experiment or novelty theorem. Candidate may change its representation; the bound does not demand universal exact closure as an admission prerequisite.

## Mechanism being examined

The reported repair appends Boolean observables phi∘F whenever absent, so each invertible incumbent operation F acts as a permutation of a finite list of code coordinates. The resulting execution is Hamming-isometric before physical faults. This could prevent correlated error amplification rather than merely increasing code distance. It is a different actual representation rule from v2's fixed global guard.

The following statement applies only if the dictionary contains every actual pullback needed for exact execution of the declared operation library on its full finite state domain. Equality on a sampled bank is not that premise. Distinct Boolean functions count as distinct coordinates; duplicate copies do not remove the bound. Arbitrary nonlinear decoders/readout can change the representation and its cost, but do not make this specific closure free.

## Short incumbent library

Let m>=1, N=2^m, S={0,...,N-1}. There are two bijective operations:

    c(s)=s+1 mod N,
    tau(s)=1-s for s in {0,1}, otherwise s.

They have ordinary O(m)-size bounded-fan-in Boolean circuits when values are represented by m bits. Ripple increment uses the carries given by successive ANDs of lower bits, followed by XORs. tau toggles the low bit iff all higher bits are zero, leaving the others unchanged. AND/NOT suffice to implement all these gates with a constant expansion. No claim of a strict reversible-gate count without ancilla accounting: the benchmark operation is bijective but its ordinary Boolean circuit may use scratch. Both candidate and compiler can inspect/run the same incumbent circuits. No full N-entry truth table is required by this rival.

## The generated group and coordinate orbit

Conjugates c^j tau c^-j swap adjacent states j and j+1 (indices modulo N). In particular they supply the adjacent swaps along the line0,...,N-1, which generate the full symmetric group S_N. The inverse cycle is c^(N-1), so forward-pullback closure under the finite library also includes inverse and all group words.

Let C be a finite coordinate family of Boolean functions phi:S->{0,1}, closed under pullbacks of c and tau. If E(s)=(phi(s))_(phi in C) is injective, at least one phi is nonconstant. Write k=|phi^-1(1)|, 1<=k<=N-1. Full permutation of S sends that support to every k-subset, producing exactly

    |Orbit(phi)|=binom(N,k)

distinct Boolean functions. Closure requires all of them in C. Consequently

    n=|C| >= binom(N,k) >= N = 2^m.

This is tight as a bare representation-size order: the N state-indicator functions are a closed one-hot family. Their distance2 is not by itself an error-correcting code for radius1; further protection has its own cost. The theorem does not assert that the raw m-bit execution already tolerates noise.

If the append-only learner retains an initial raw bit, its support has k=N/2, giving the much stronger bound

    n >= binom(N,N/2) ~ 2^N sqrt(2/(pi N)).

For m=3 this particular closed orbit has70 functions, not just the reported11 for one Toffoli transformation. This is a hand value C(8,4)=70, not a new run. Abandoning that initial bit avoids this stronger bound but still leaves the N-coordinate minimum under these exact premises.

## Concrete conventional alternative under comparable execution assumptions

Ordinary reliable compilation need not linearize every operation as a coordinate permutation. It can keep m logical bits and compile each O(m) Boolean circuit into constant-size noisy-gate gadgets with local restoration. Here is an explicit conservative construction showing why the comparison is not universal simulation.

Use13 repeated physical bits per logical bit, input allowance t=2 per block. For a two-input AND, coordinatewise gates introduce <=4 errors from inputs. Up to2 faulty U output gates give <=6 errors. NOT gives <=2+2<=4. A majority-of13 returns the correct logical value from <=6 errors.

R has13 **private** output circuits, each computing majority-of13. Each circuit can be implemented using a fixed finite fan-in2 Boolean circuit; there is no need to grant a larger fan-in than the candidate. Do not share internal gates between these13 circuits. Any <=2 physical gate failures inside R can then corrupt at most2 of its outputs, so it restores the t=2 allowance. A majority13 implementation may have a large constant; it is explicitly part of the cost, not a reliable hidden decoder. Its fixed Boolean size is independent of m. No stochastic independence is required for this local deterministic count.

Induction through the incumbent's ordinary circuit handles <=2 faults in each gadget's U phase and <=2 in its R phase. In particular it handles the candidate's stricter total of two physical faults per whole incumbent invocation, wherever those fall, provided initialization is a valid encoded state and faults are the same gate-output deviations. Constant fan-in, actual redundant gates, scratch, synchronization and retained logical wires are all paid. Circuit gate count and live encoded state are O(m) times fixed constants, whereas exact pullback coordinates cost at least2^m. The exact constants or iid failure rate are not measured here. This does not claim a particular compiler's practical superiority for the three-bit example.

## What the bound decides and what it does not

It refutes an unqualified prospect that this exact coordinate-isometry compilation provides a compact general representation for short reversible computations. The source of cost is the semantic operation group's action on coordinate functions, not insufficient seeds, insufficient code length tuning or failure to learn a seven-bit code. Running more paths/degrees is unnecessary to obtain this information.

It leaves real possibilities open: restricted operation families with small feature orbits, approximate/local closure with a stated error budget, changing a non-permutation operation representation, or an acquired organization that avoids paying this orbit while still controlling error amplification. Those would be different substantive mechanisms needing actual rules, not automatic repairs. Context-dependent codes can move the original computation into encoding/decoding; those computations must be executed and counted. Full state equivalence must not be replaced silently by task-specific equivalence or lost information assumed recoverable.

The next constructive opportunity is to address this precise cost/fault-spread tradeoff in the actual v3, rather than add more features or train on the same small Toffoli witness. No new experiment admitted by this note. No general impossibility of learned reliable computation or of the Prometheus ambition follows.
