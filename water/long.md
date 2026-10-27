# Prometheus · Long

**Period:** 2026-10-27 → 2026-11-26 (rolling one-month plan)  
**Time budget:** ~4–6 hours/week; flexible, without catch-up debt  
**Mode:** Reading, reasoning, Markdown notes. No Python implementation required.

## Monthly question

> How can simple local interaction rules produce stable organization, memory, and useful computation?

**This month's boundary:** Understand specific known mechanisms of emergence and learning. Do not attempt to solve open-ended evolution, design Prometheus' final architecture, or prove that intelligence will emerge.

## Three anchor sources

1. **Growing Neural Cellular Automata** (Mordvintsev et al., 2020) — https://distill.pub/2020/growing-ca/  
   Focus: local perception, cell update, growth, persistence, regeneration, and the *externally supplied training objective*.
2. **Neuronal Dynamics** (Gerstner et al.) — https://neuronaldynamics.epfl.ch/online/  
   Focus: §19.1–19.3 (Hebbian learning, STDP, unsupervised learning), with §17.1–17.2 for memory and attractor dynamics.
3. **A Movable Architecture for Robust Spatial Computing** (Ackley, Cannon & Williams) — https://www.cs.unm.edu/~williams/acw.pdf  
   Focus: local asynchronous updates, spatial computation, redundancy and self-healing. Do not assume the model already achieves open-ended evolution.

All other books and papers are **references**, not a reading backlog. Open one only when a concrete question in an anchor source requires it.

## Weekly route

| Dates | Focus | Required reading | Test of understanding |
|---|---|---|---|
| Oct 27–Nov 1 | Local rules → organized form | Growing NCA, sections Model / Learning to Grow / persistence; Neuronal Dynamics §19.1–19.2 | Explain which mechanisms are predesigned, trained, or emergent |
| Nov 2–8 | Local plasticity → memory | Neuronal Dynamics §19.3 and §17.1–17.2 | Explain Hebbian/STDP updates and the distinction between state persistence and retrievable memory |
| Nov 9–15 | Local events → robust computation | Ackley et al.: introduction, MFM model, self-healing examples | Describe how information moves and computation survives local changes without global coordination |
| Nov 16–22 | Synthesis | Revisit the three sources; no new major book | Compare the assumptions, mechanisms, demonstrated outcomes, and limits |
| Nov 23–26 | Buffer and next question | Fill a concrete knowledge gap only if necessary | Write one precise question worth studying next month |

## Output (small on purpose)

- One concise mechanism note for each anchor source (roughly one page each; can be shorter).
- One comparison table: **What is fixed? What is learned? What emerges? What is measured?**
- One unresolved *specific* question to guide the next month. A mathematical hypothesis is optional, not mandatory.

## Method: prevent both failure modes

**Against directionless textbook reading**
- Before reading, write the question this source is supposed to answer.
- Read the relevant sections, not whole books sequentially.
- If a definition blocks progress, consult a single supporting section and return.

**Against endless abstract speculation**
- After every reading block, close the PDF and reconstruct one mechanism from memory: diagram, equation, or 150–250 words.
- For every claim about emergence, name the source and distinguish results from conjecture.
- Put exciting tangents into a brief `Later` paragraph; revisit only at the weekly review.

**Guardrail:** Do not judge success by page count, idea count, novelty, or conference potential. Judge whether one mechanism is clearer than last week.

## Reading-note template

```markdown
# Source / Mechanism
Question I brought to this source:

What the system is allowed to do (state, interactions, updates):

What is specified / learned / emergent:

One mechanism I can explain without the paper:

One mathematical detail I understand (or need to resolve):

What the paper does NOT show:

Next concrete question:
```

## Not in scope this month

Full-text reading of Kandel, Kauffman, Simon, Ashby, von Neumann, etc.; exhaustive citation gathering; Python prototypes; learning Chinese greetings; proposing a complete new neural architecture.
