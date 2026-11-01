# A source record frozen with v0

## Provenance and scope

No older Prometheus architecture report, other branch candidate, `fire/`, root research dashboard or full conversation was inspected. The seed and contract and the independent-researcher section of `deep-work/ROLES.md` were read. Local PDF extraction/OCR helpers and generated text live here; extraction is not counted as reading.

Substantive passages actually read:

- Banzhaf and Yamamoto, *Artificial Chemistries* (2015), local `food/Artificial Chemistries.pdf`: PDF 63–67 / printed 46–50, matrix chemistry; PDF 274–278 / printed 257–261, organization definitions; PDF 328–335 / printed 311–318, coupling, Jain–Krishna and higher-level entities. Some adjacent contents/introductory text appeared in extraction output. Only these named passages bear source claims below. Text of the whole PDF was mechanically extracted, not studied in full.
- Kauffman, *The Origins of Order* (1993), local long-named PDF: front matter/contents/themes PDF 1–16, then PDF 387–400 / printed 371–384, grammar-carried operators, machine assembly, infinite Boolean representation, finite resource bounds, inhibition and kinetics. These 14 technical pages were OCRed and read. The local PDF uses printed-page offset +16 here.
- von Neumann, edited/completed by Burks, *Theory of Self-Reproducing Automata* excerpt, `food/4TheoryofSelfReproducingAutomata.pdf`: title; PDF 2–8 / printed 64–70 on virtual memory and mixed analog/digital mechanisms; PDF 16–25 / printed 78–87 on complication, kinematic components, A/B/C constructor and heritable D. OCR was also produced for PDF 9–15 but that range was not substantively read. This is a 25-page excerpt, not the whole book. Burks's square-bracket reconstruction must not be attributed as a verbatim completed von Neumann design.
- Graves, Wayne and Danihelka, *Neural Turing Machines* (2014), official arXiv PDF https://arxiv.org/pdf/1410.5401, PDF 4–9 / printed 4–9, especially §§3.1–3.3: read, erase/add write, content/location addressing. Read via the web tool. This source materially limits novelty of A's initial soft scope model. No reliance on the abstract for the mechanism comparison. A search result for modern Hopfield networks was seen, but its technical paper was not read and it supports no claim here.

## 1. Matrix chemistry: a concrete constructive transition, and a source conflict

SOURCE, PDF 63–67 / printed 46–50, equations (3.1)–(3.17): a bit string s of length N=m² is folded into m×m operator P_s. With the canonical row-major fold, the operator acts separately on m length-m blocks of another string t:

    out[i+k*m] = σ(sum_j P_s[i,j]*t[j+k*m] - Θ),
    σ(z) = 1 if z >= 0 else 0.

The chemical transition keeps both input strings and creates the product:

    s + t -> s + t + out.

The product is another bit string in the same code space; it is eligible to fold and act later. This is genuine executed object construction, not merely an observer assigning a label to a stable concentration pattern. The fold and interpreter are hand-designed, and replication abundance is not task utility.

VERIFIED SOURCE CONVENTION CONFLICT: at Θ=0 every sum is nonnegative, so σ yields 1 at every output coordinate. Exhaustive evaluation of the 256 ordered length-4 pairs in `probes/A/v0_calculation.py` gives only 1111. The same printed page nevertheless calls Θ=0 the Boolean case and zero the destructor which can copy itself. The inspected source image is `matrix-p48.png`. The source's step-at-zero and threshold prose cannot all be used literally together.

RECONSTRUCTION WITH EXPLICIT REPAIR: choose Θ=1 (alternatively use a strict positive step at Θ=0). For s=(1,1,0,1), P_s=[[1,1],[0,1]], and t=(1,0,0,1), the two blocks are (1,0), (0,1). Their products after threshold are (1,0), (1,1). Thus out=(1,0,1,1), whose future operator is [[1,0],[1,1]]. This is a small actual calculation of a new executable operator; the positive threshold is our repair, not a claim that the printed source executed that convention.

What changed in A's reasoning: “state is code” is a precise established mechanism, but an implicit universal code space does not establish acquired useful primitives. Selection/dilution and task relevance must be distinguished; a changed string is not automatically a functional innovation.

## 2. Organization and fast/slow dynamics do not supply useful computation

SOURCE, PDF 277 / printed 260: in the book's simplified algebraic definition, O is an organization if it is closed under its reaction operation and every member has at least one producing tuple within O. These are set-level conditions. They do not establish positive concentrations under arbitrary kinetics, robustness, or task-solving computation.

SOURCE, PDF 329–330 / printed 312–313: the Jain–Krishna model described there has fast normalized species dynamics

    dx_i/dt = (C x)_i - x_i * sum_k (C x)_k,
    x_i >= 0, sum_i x_i=1,

then a slow operation removes a minimum-concentration node and randomly rewires its incoming/outgoing interactions. The source has binary adjacency, diagonal zero. No task-error input selects rewiring.

RECONSTRUCTION, not a new scientific result: take the weighted generalization C=[[0,2],[1,0]]. Its dominant eigenvalue is sqrt(2), and its normalized positive eigenvector is

    x*=(sqrt(2)/(1+sqrt(2)), 1/(1+sqrt(2)))
      =(0.5857864376, 0.4142135624).

Substitution gives Cx*=sqrt(2)x*, hence the normalized flow is zero. The executed substitution residual is <4e-16. Weighted C is our explicit extension, not the book's binary model. The calculation demonstrates an attractor under a selected interaction law; it does not demonstrate learned construction or future computational evolvability. I did not verify the original Jain–Krishna paper in this branch and therefore do not elevate the textbook's empirical claims to independently checked primary results.

SOURCE, PDF 334–335 / printed 317–318: the symbol y for a robust higher-level attractor equivalence class need not have its own reproductive strength. This is the sharp premise used in A: a learned “scope” must demonstrably alter causal access or protect independent state, rather than merely be a cluster that an observer names.

## 3. Kauffman: grouping changes kinetic accessibility, not the formal universe

SOURCE, PDF 391–392 / printed 375–376: an ordered aggregate of five string transformations acts as one machine; in a repertoire of 100 types there are 100^5=10^10 possible ordered length-five aggregates. Assembly changes the timescale and frequency of the particular compound transformations. This is already an explicit precedent for reusable computation by aggregation.

RECONSTRUCTION: choosing an exact five-step sequence by five independent uniform draws from 100 choices has probability 10^-10; choosing an already stored aggregate from a library of size K has probability 1/K. This is not a fair *total* learning-cost comparison until discovery, assembly, storage, library competition and interpretation costs are included. The saved routing search is real only after acquisition. A cannot call ordinary macro caching a new computational principle.

SOURCE, PDF 395–397 / printed 379–381: Kauffman's canonical family restricts maximum machine size, input bundle size, output spray and finite length amplification. The “all possible transformations occur” Boolean idealization is distinct from a resource-limited collision process. PDF 397–399 / printed 381–383 explicitly discusses inhibitors removing otherwise possible machine action; enzymatic sites can carry grammar operators and matching aggregates can self-assemble.

What changed: the new opportunity is a learned *domain of action* and its permeability, which changes which interactions occur, rather than simply adding a macro's name. This must still be compared to learned addressing. The source's speculative consequences about economics or life are not evidence for the proposed neural architecture.

## 4. von Neumann/Burks: constructor/copy accounting and what actually persists

SOURCE, PDF 22–25 / printed 84–87: A consumes φ(X) while constructing X. B consumes a description and produces copies. C sequences these actions and attaches a description to a finished object. Put U=A+B+C. For X=U, an automaton containing U and φ(U) can produce another U+φ(U); for X=U+D, a viable mutation confined to the description of D can be inherited as D'.

RECONSTRUCTION WITH RESOURCE ACCOUNTING: the excerpt's main double-copy discussion leaves two bare U automata but only one φ(U), because A consumes one copy. Burks's bracketed note at PDF 23 / printed 85 explicitly repairs continued reproduction by having B copy thrice:

    initial:  1 U, 1 description
    B:        1 U, 3 descriptions
    A:        2 U, 2 descriptions
    C:        two complete (U + φ(U)) systems.

This accounting is checked by the finite probe. If a description has L symbols and the machine has K parts, this construction requires at least K new machine parts and L net additional description symbols; B writes 3L symbol positions and A consumes L positions in this convention. These are representation/material counts, not a measured wall-clock estimate.

What changed: copying a transient state and preserving a rule that reconstructs the state are different. In A, keys/affinity laws that were learned between episodes can persist through saved parameters; the old activity field does not magically move when those keys change. A training update in an episode is not inherited merely because a runtime scope exists.

## 5. Close neural precedent prevents a broad novelty claim

SOURCE, NTM PDF 5–8, eqs. (1)–(9): read is a weighted sum of memory vectors; write applies differentiable erase and add masks; addressing is produced using similarity and location transformations. The paper makes memory interference a stated architectural concern.

RECONSTRUCTION: on a fixed finite grid, A's Gaussian field write/read is an unnormalized kernel-weighted memory write/read. Rescaling the add/read values maps it into normalized weightings. Thus the worked Gaussian case by itself is a special case of familiar differentiable memory. A different kinetic interpretation is not a novelty proof. The more ambitious scope proposal below must earn a distinction through learned local causal barriers for *execution, parameter updates and construction together*, or be classified as an incremental memory architecture.

## Artifacts and cost

The finite check script uses Python standard library only and reported ~0.002 seconds internal time on its successful second execution. It checks 256 bit-string pairs, five derivative comparisons, two 500-update one-dimensional adaptations, copy accounting and one eigenpair substitution. The first attempt failed an overoptimistic convergence-tolerance assertion; its exact script and failure are retained. Source extraction/OCR was local Swift PDFKit/Vision at scale 2, with no cloud inference or purchases; its wall time was not instrumented, so no quantitative total reading-compute cost is claimed.
