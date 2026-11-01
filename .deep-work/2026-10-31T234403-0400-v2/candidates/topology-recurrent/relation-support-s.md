# Relation support S — partial derivation, stopped on experiment-first steering

2026-11-01. **UNFINISHED / analytical support only.** Work stopped on Root's relayed latest human EXPERIMENT-FIRST instruction. This file preserves reasoning completed before that steering; it is not a finished architecture, frozen experiment, review029, approval prerequisite or result. No further theory/literature work is allocated by this artifact.

MAIN=TOPOLOGY; PRIORITY=USER_LOCKED; MAIN_CHANGE=HUMAN_ONLY; M0 active NOT PASSED; M3 ultimate. Online USER-PARK; Persistent State standalone USER-PAUSED. No code, experiment, membership query, training or literature lookup was executed for this support task. The binary observations below are hypothetical values derived by hand from the explicitly described example, not acquired data.

## Partial operational interface

A finite terminal-feedback quotient could provide a compact relation without directly observing target hidden states. Given access prefixes P and distinguishing suffixes E, associate a reached state with its terminal-output signature `(output(pe): e in E)`. Distinct signatures distinguish states. Learning still needs reset/word-query access, adequate distinguishing suffixes and either a known finite state bound reached by the discovered quotient or another justified conformance guarantee. A closed/consistent finite table alone cannot certify every unseen word of an unrestricted hidden system; an unobserved extra state can agree on all queried words and differ later. No equivalence oracle was granted here.

The desired implementation relation has two separate obligations: terminal signatures identify corresponding old/new behavioral states, while finite generator-level invariance checks establish that the implementation stays inside those corresponding state regions after every old symbol. Existence of a semiconjugacy alone is not a procedure for obtaining it from terminal data. This support had not completed the general learning rule, neural region certificate or resource comparison when stopped.

## Hand-derived noncommutative three-state example

For an illustrative hidden deterministic target with at most THREE states, let its start be0, actions a=(01) and b=(12), and terminal observation be1 only in state0. These state/action definitions describe the hypothetical target for analysis; learner feedback would be binary word outputs, not state labels or target programs.

A prospective access set is P={empty,a,ab} and suffix set E={empty,a}. The three signatures are:

| Access prefix | Empty-suffix observation | Suffix-a observation |
|---|---:|---:|
|empty|1|0|
|a|0|1|
|ab|0|0|

Transition signatures give a: `(10)<->(01)`, fixing `(00)`; b fixes `(10)` and swaps `(01)<->(00)`. Thus the inferred generators are noncommutative. In particular terminal output on `aba` is0, while on `baa` it is1. A suffix is needed because terminal outputs on `ab` and `ba` alone are both0.

The table over `(P union P{a,b}) E` uses11 distinct hypothetical binary word queries:

    empty:1, a:0, ab:0, b:1, aa:1, aba:0,
    abb:0, ba:0, aaa:0, abaa:0, abba:1.

If THREE is a valid upper state bound and all three distinct signatures are reached, they exhaust the target's states. Exact transition signatures then identify its finite transition table without an equivalence oracle. Without that state-bound premise this conclusion is unsupported. An adaptive discovery procedure for P/E, noisy-output guarantees and a complete observation-cost account had not been developed in this allocation.

## Unfinished implementation sketch and boundary

A possible continuous realization uses separated state regions and plateau functions equal to1 on each region and0 outside a disjoint larger neighborhood; inside each region a generator maps into the learned target region. Shared region locations could change while old generator inclusions and terminal readout identification are checked. This was only a sketch: no actual recurrent F_theta, learned state relation, isolation proof, neural optimization or new-operation acquisition trajectory was completed.

The ordinary finite-state/program learner receives the same terminal information and can use the same learned transition table. This example therefore supplies an operational observability calculation, not an acquisition/retention advantage claim. It does not establish the required general architecture or replace the root's actual experiment-first prototype.

NEXT: no further work from this support artifact. Return to Root; inspect actual raw experiment records only when assigned. Do not count unfinished support as a passed mechanism or restart it automatically after a failed prototype.
