# AE-v2 provenance and preserved failure

Fresh constructive repairer, explicitly authorized current-Ultra follow-on; no verified Pro/model-effort attribution. No children. This context received AE-v1 and A/E frozen mechanisms rather than being independent of them.

Read root AGENTS.md; deep-work/ROLES.md (constructive repairer governs); A candidate-v0, E v0_frozen, AE candidate-v1/provenance/source record/contract/probe/results and existing isolation runner. No independent_rebuild materials were accessed. A filename inventory was used only to locate relevant A/E files. No other candidate's substantive contents were read. New primary reading: Bello et al. §4 and §6; recorded under sources/AE-v2. Williams appeared in source search results; no technical claim depends on a newly verified Williams paper.

The initial construction, parameters, three seeds, 2,000 episodes/method/seed, contexts, comparisons, numerical sanity thresholds and partial-acquisition criterion were written in contract.md before execution. acquire.py was written, syntax checked and read before numerical execution. No outcome-driven parameter change was made.

Executed command:

    python3 .deep-work/2026-10-31T215613-0400-campaign/probes/infrastructure/run_isolated.py .deep-work/2026-10-31T215613-0400-campaign/probes/AE-v2/acquire.py --timeout 120 --log .deep-work/2026-10-31T215613-0400-campaign/probes/AE-v2/execution.log

Exit 1 at final json.dumps: numpy.bool_ is not JSON serializable. All six final policy files and reference.npz existed before the failure. Per-episode records and first internal duration were not saved. No training was repeated.

Parent explicitly authorized an evaluation-only recovery after this failure. recovery-contract.md and recover_evaluation.py were written before execution; the recovery loads the inspected definitions, saved reference table and final weights. It never calls Policy.update. Eight BDF comparisons/step checks and one finite-difference derivative check were repeated, and frozen policies were evaluated under seed+10000. Initial random-policy observations use original seeds. The ordinary allocator is deterministic replay of the original fixed candidate-generation rule, not an extra tuned search. Missing original training data remain unavailable.

Executed command:

    python3 .deep-work/2026-10-31T215613-0400-campaign/probes/infrastructure/run_isolated.py .deep-work/2026-10-31T215613-0400-campaign/probes/AE-v2/recover_evaluation.py --timeout 120 --log .deep-work/2026-10-31T215613-0400-campaign/probes/AE-v2/recovery.log

Exit 0. Recovery runtime 7.237562 seconds. Successful results: results_recovered.json. No original failed artifact was overwritten; acquire.py remains the exact original program with its serialization error. The recovery sanity conservation numbers apply to the specified checked batch, since the first table's original aggregate sanity counters were not saved.

Final SHA256 values (recorded after execution; not an independently timestamped freeze):

    contract.md 80682d785d2c85b00f5832ffe7b11db653f23d2d848e5f45961338024fd745c6
    acquire.py 4745afb9586389cf3681e5eeae4b856a05b82a01393758ac54a76d00db7f9058
    recovery-contract.md 1d60c3b678b82abea604c815bd8f81ca232e0e1ef8a82776dd815a0c59709660
    recover_evaluation.py 336f737cf7b09f1a809342e6ef378abef2ab2df7ea9863b3f3dcf8739294784e
    results_recovered.json 8f59d0ea5431dfd76c61e5cf95ad0a1e7f6bba0efaec952ab338fbf1fe113333
    reference.npz 02c4a2ce82c02111fc28804e5afc641946c030070f7ca4da8dd6ed452e3b97e7

AGENT PROPOSAL: finite neural material-installation policy and conditional-use criterion. INHERITED SOURCE/EXECUTED RESULT: E recipe and AE load driver. VERIFIED OBSERVATION: saved policies improve finite task scores. ATTACK: zero-load overspending and near-invariance to removal of load information. OPEN: genuinely load-sensitive constructor acquisition, learned boundaries and the full neural self-construction objective.
