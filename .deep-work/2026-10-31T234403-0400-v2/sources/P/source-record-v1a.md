# P v1a targeted comparison

## S11 — Farajtabar, Azizan, Mott and Li, Orthogonal Gradient Descent for Continual Learning, AISTATS 2020

Original PDF: https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf

Read Section 3, equations (7)–(9), Algorithm 1 and its discussion of locality. OGD stores a basis of previous model-output gradient directions, rather than only loss gradients, and projects new-task gradients onto the orthogonal complement. The paper explicitly notes that old-task inputs are unavailable and approximates current old-output gradients by those at the old optimum in a nearby parameter region. Its algorithm grows the stored basis at task boundaries. This is prior art for effect-preserving update subspaces; it is not an exact arbitrary nonlinear retention guarantee.

Boundary: P's affine control derivation cannot claim a new nullspace learning principle. It is an exact conditional construction supporting the more specific hypothesis of acquired/composed nonlinear repair interfaces. No source-code execution or numerical benchmark was performed.
