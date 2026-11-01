# after012-b · Primary reading record

No old catalog, old candidate, or other live candidate was read before v0. The seed was reconstructed independently, then sources entered during construction; this is not an independent empirical verification. No downloaded PDF or execution artifact was created. Effective mode unknown; context/file isolation instruction_only.

## Chen 1975: the limitation that changed the candidate

Kuo-Tsai Chen, *Iterated integrals, fundamental groups and covering spaces*, Transactions AMS 206 (1975), 83–98. DOI 10.1090/S0002-9947-1975-0377960-0. Original paper read through a mirror: https://scispace.com/pdf/iterated-integrals-fundamental-groups-and-covering-spaces-26273pvlt9.pdf . AMS PDF endpoints for Chen 1958 and 1977 returned 403; their text was not read and they do not support substantive claims here.

Read: printed pp.83–87 definitions and multiplication formula (1.2); pp.91–94 augmentation filtration, Lemma 5.1 and Theorem 5.3; pp.96–98 Proposition 7.1, Theorem 7.2 and covering-space Theorem 9.2. The spectral-sequence comparison proof has not been independently reconstructed.

SOURCE: over k=R or C, A' is the complex of iterated integrals restricted to based loops; A'(s) restricts length. Under the stated hypotheses, Theorem 5.3 identifies H^0(A'(s)) with linear functionals on Z pi_1(X) modulo its augmentation ideal's (s+1)-st power. Theorem 7.2, for finitely generated pi_1, characterizes separation by these homotopy-invariant functions through residual torsion-free nilpotence. Theorem 9.2 concerns separation on the universal cover with the additional base-function separation assumption.

RECONSTRUCTION: the special group Z/2 has augmentation generator u=g-1, u^2=-2u, so its characteristic-zero finite augmentation quotients erase the nonidentity element. This induced the move from only real nilpotent coordinates to discrete-sheet memory. It does not rule out nonlinear readouts of arbitrary signatures or non-unipotent matrix representations. Those extrapolations would be false or unestablished.

## Lou, Li, Ni 2024: exact conventional development rival

Hang Lou, Siran Li, Hao Ni, *Path Development Network with Finite-dimensional Lie Group*, TMLR 2024; original arXiv PDF v2: https://arxiv.org/pdf/2204.00740 . Read printed pp.1–6: introduction, equation (1), Definition 2.2, Example 2.1, Lemma 2.1, canonical extension equation (4). Training proofs, claimed empirical superiority, and universality proofs were not independently audited and are not evidence used here.

SOURCE: the trainable layer updates z_(n+1)=z_n exp(M_theta(delta x_n)), z_0=identity, using a matrix Lie group and a learned linear map into its Lie algebra. Their definition is the linear CDE dZ=Z M(dX). Concatenation becomes multiplication. Their canonical tensor-algebra extension replaces tensor words by products of generator matrices. Therefore describing learned finite path representations and concatenation as new is already specifically covered at the execution level.

RECONSTRUCTION: the area recurrence's homogeneous 4x4 matrix is a particular such development, without appealing to universal neural approximation. The proposed cocycle acquisition schedule is not claimed to occur in this paper. Whether it improves any fair learning resource remains open.

## Balle, Panangaden, Precup 2015: exact finite-state rival

Borja Balle, Prakash Panangaden, Doina Precup, *A Canonical Form for Weighted Automata and Applications to Approximate Minimization*, LICS 2015. Original author-hosted paper: https://www.cs.mcgill.ca/~prakash/Pubs/lics2015.pdf . Read Section II.B, printed pp.3–4: WFA definition, conjugacy, Hankel matrix, Theorems 1–2; introduction's description of spectral learning. Approximate-minimization results were not audited or used.

SOURCE: a real WFA evaluates a word by initial vector, product of letter matrices, and final vector. Its minimal dimension equals the rank of the word function's Hankel matrix H_f(p,s)=f(ps). A finite subblock's factorization is the basis for spectral reconstruction in the cited learning framework. This source is not evidence for a particular sample bound in the present continuous-input task.

RECONSTRUCTION: right multiplication on any m-dimensional associative algebra gives a same-dimensional linear representation. For the signed-area function, the displayed four-by-four Hankel submatrix has determinant -1/4, while concatenation gives rank <=4. This establishes the exact four-state comparison on the finite stated alphabet, on paper rather than by executed determinant code.

## Unread leads and boundaries

Searches for learned group extensions/cocycles did not establish an exhaustive prior-work boundary. Search results or abstracts about other group-extension work were not used as primary claims. No claim that a topology-based state-lifting acquisition algorithm is novel follows from these searches. The finite-good-cover/cocycle construction and SO(3) lift in v0 are mathematical reconstructions; their broader acquisition and neural implementation remain untested.
