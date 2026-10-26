# Prometheus

Independent ML research on **Scalable Skill Acquisition × Architecture-Aware
Reinforcement Learning**. Investigate original learning algorithms that improve
acquisition, composition, and transfer of previously difficult reasoning skills.
The exact method is undecided. No experiments or results are recorded yet.

## Layout

- `research/`: direction, related work, hypotheses, decisions, and a note template.
- `experiments/configs/`: versioned experiment configurations.
- `src/`: research implementations.
- `tests/`: implementation and evaluation checks.
- `runs/`: ignored local run outputs, except its directory placeholder.

## Workflow

Follow [AGENTS.md](AGENTS.md). Verify primary papers and source code, formulate
falsifiable hypotheses, then implement and test before requesting training permission.
No runtime or dependencies have been selected. Once selected, pin package versions
and document environment setup and exact test, training, and evaluation commands.

Record model IDs, package versions, git commit and uncommitted changes, random
seeds, evaluation metrics, planned and actual compute budgets, data versions,
split definitions, configurations, commands, and hardware/runtime details.
Keep durable configurations and result provenance in version control outside
ignored `runs/` outputs. Never commit secrets, raw data, or large checkpoints.
