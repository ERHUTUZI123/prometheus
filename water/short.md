# Prometheus · Short

**Week:** 2026-10-27 → 2026-11-01  
**Time budget:** ~4 hours, flexible  
**One question:** How does Growing Neural Cellular Automata turn local cell interactions into a stable global structure, and what did the researchers have to predefine?

## Three sessions

- [ ] **Tue–Wed · 90 min — Study Growing NCA.** Read Introduction, Model, and Learning to Grow: https://distill.pub/2020/growing-ca/  
  Identify: state of each cell; what a cell perceives; how updates occur; where the target image/loss enters. Draw a small update diagram.
- [ ] **Thu–Fri · 90 min — Learn one biological learning mechanism.** Read *Neuronal Dynamics* §19.1–19.2: https://neuronaldynamics.epfl.ch/online/  
  Explain Hebbian plasticity or pair-based STDP, including which signals a synapse needs to update. Focus on understanding one equation, not finishing the textbook.
- [ ] **Weekend · 60 min — Close the PDFs and explain.** Write a short note from memory: what in Growing NCA is designed, learned, or emergent? What does local plasticity change? Then revisit the PDF to check your understanding.

**Optional:** A 20-minute meal-time PDF read is fine; it does not create a new mandatory task.

## Three questions to answer by Sunday

1. What information can an individual NCA cell access, and what can it change?
2. Why does a global pattern emerge, and what role do externally supplied training targets play?
3. Why does self-organized shape formation not by itself demonstrate self-generated learning goals or new computational capabilities?

## Required output: one Markdown note, maximum ~1 page

```markdown
# Week 44 — Local Rules and Structure

## Mechanism
[Explain cell state → perception → update → collective behavior.]

## Design vs emergence
- Predetermined:
- Learned through training:
- Emergent in execution:

## Connection to synaptic plasticity
[One similarity; one important difference.]

## Still unclear
[One specific question to carry into next week.]
```

## Reading rules

- Start with the question, not page one of a large book.
- Spend more time reconstructing a mechanism than collecting related papers.
- When an exciting unrelated idea appears, note it in two lines and resume reading.
- First reading is yours. Use Codex afterward for checking misconceptions or navigating notes, not for replacing the initial reading.
- If only the first session is completed, continue from there next week; no catching up by racing through extra papers.

**Done means:** You can explain the mechanism clearly, including what is imposed by the experimenter. No code or architecture proposal is needed.
