# Prometheus Research Instructions

## Research Direction
Scalable Skill Acquisition × Architecture-Aware Reinforcement Learning.

Goal: Investigate original learning algorithms that improve acquisition, composition, and transfer of previously difficult reasoning skills. The exact method is undecided.

## Research Rules
- Verify claims against primary papers and actual source code. Record precise references and code revisions; mark unverified claims explicitly.
- Distinguish existing research results, hypotheses, and original contributions.
- Do not declare novelty without reviewing relevant prior work.
- Preserve train/test isolation and experimental reproducibility. Fix split definitions; never use test examples or feedback for training, tuning, or model selection.
- Record model IDs, package versions, git commit and uncommitted changes, random seeds, evaluation metrics, and planned and actual compute budgets for experiments. Also record data versions, split definitions, configurations, commands, and hardware/runtime details.
- Implement tests and execute them before claiming something works. State what was tested and distinguish code correctness from scientific validity.
- Never infer experiment success from generated code or documentation. Report measured results only from actual executions with retained provenance.
- Never ingest proprietary Huawei code, data, credentials, or logs into this personal repository.
- Do not launch training, use cloud GPUs, install broad dependencies, publish code, or push commits without explicit permission.
- Never modify unrelated files. Keep secrets, raw data, large checkpoints, and transient run outputs out of git.
- Keep the workflow minimal; do not build an agent orchestration framework.
- Prioritize scientific insight over engineering complexity.

## Workflow
Read → Understand → Hypothesize → Design → Implement → Evaluate → Critique.

The research question and methodology may evolve based on evidence.

Record substantive decisions in `research/decisions.md`. Preserve durable experiment configurations and result provenance outside ignored run outputs.
