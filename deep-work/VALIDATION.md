# Local v2 migration validation

This section records deployment checks, not a new research run or provider-mode verification.

- The working tree was clean at entry. Local AGENTS differed from the package's declared baseline, so clauses were merged semantically; no blind patch or whole-file replacement of AGENTS was used.
- Original local helper suite: 12 offline tests passed before migration. Unmodified update-package validation: 21 offline tests and its package invariants passed.
- Merged helper suite: 40 offline tests passed, including legacy receipt API/fields, malformed data, current-run authorization, unknown effective modes, per-worker receipt binding, path containment and exploratory evaluation labels.
- Additional deployment checks cover JSON, Python syntax without bytecode writes, exact permanent principles, retained local evidence/permission clauses, unique NOW, historical content, links, diff and protected-file fingerprints. Actual receipts and backups are in [the migration directory](../.deep-work/migration-2026-10-31T233014-0400/).
- No scientific experiment, training, live model/API probe, paid call, Git commit or global-setting change was performed. Archived campaigns and their original/failed audits are not rewritten to satisfy the new schema.
- These checks do not establish scientific novelty, workflow optimality, actual runtime isolation, real user consent or provider-effective mode. Structured authorization receipts bind declared scope to a run; human authorization remains a separate prerequisite.

# Historical v2 package validation (before local installation)

- 21 offline unit tests passed for mode provenance, current-run authorization, paths, dependencies and declared task records.
- Package invariants passed: exact permanent principles, discovery/validation split, P/K/R entries, one principal plus one high-risk alternative, no mandatory novel low-level primitive, no inherited prior-run mode permission.
- All JSON parsed; all Python files compiled for syntax.
- AGENTS.patch applied successfully to the specified Library AGENTS(4).md baseline in a temporary Git repository. The result exactly matched AGENTS.revised.md.
- Local machine AGENTS.md was not accessed; deploy by semantic merge if its content differs.
- No live model/API call, paid request, local user-repository edit, research training or scientific experiment was started.
- These tests do not establish scientific novelty, optimality, enforced isolation, provider mode, or correctness of the research campaign.
