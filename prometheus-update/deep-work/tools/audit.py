#!/usr/bin/env python3
"""Audit research records, not scientific validity.

All artifact paths are relative to the run-record JSON's directory. The audit
checks declared records and file existence; it cannot establish actual sandbox
isolation, source comprehension, novelty, or experimental honesty.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

STATES = {"pending", "running", "checkpointed", "completed", "blocked", "closed"}
MODE_KINDS = {"provider_metadata", "host_metadata", "ui_user_attested"}


def audit(record: dict[str, Any], root: Path, final: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    root = root.resolve()

    def artifact(value: Any, label: str) -> None:
        if not isinstance(value, str) or not value:
            errors.append(f"{label}: missing artifact path")
            return
        p = Path(value)
        if p.is_absolute():
            errors.append(f"{label}: must be relative to run root")
            return
        resolved = (root / p).resolve()
        if root not in resolved.parents or not resolved.is_file():
            errors.append(f"{label}: absent, not a file, or outside run root")

    # v2 validates an explicit per-run authorization without fabricating
    # provider-effective mode from a host selection or user confirmation.
    policy = record.get("mode_policy")
    has_current_policy = policy is not None
    if policy is None:
        policy = {"requested_mode": "pro", "authorized_observed_modes": ["pro"]}
    elif not isinstance(policy, dict):
        errors.append("mode_policy must be an object")
        policy = {}
    preferred_mode = policy.get("requested_mode", "pro")
    allowed = policy.get("authorized_observed_modes", ["pro"])
    if not isinstance(allowed, list) or not allowed or any(not isinstance(x, str) or not x for x in allowed):
        errors.append("authorized_observed_modes must be a nonempty list of labels")
        allowed = []

    def mode(value: Any, label: str) -> None:
        if not isinstance(value, dict):
            errors.append(f"{label}: missing mode evidence")
            return
        requested = value.get("requested_mode")
        effective = value.get("effective_mode")
        observed = value.get("observed_mode")
        kind = value.get("evidence_kind")
        if not isinstance(requested, str) or not requested:
            errors.append(f"{label}: missing requested mode")
        elif requested not in allowed and requested != preferred_mode:
            errors.append(f"{label}: request not covered by current policy")
        if kind not in MODE_KINDS:
            errors.append(f"{label}: unacceptable mode evidence kind")
        if not value.get("model"):
            errors.append(f"{label}: missing model identifier/visible model label")
        if kind == "provider_metadata":
            evidenced = effective
            if not isinstance(effective, str) or not effective:
                errors.append(f"{label}: effective provider mode not available; use truthful host/attestation evidence instead")
        else:
            evidenced = observed
            if effective is not None:
                errors.append(f"{label}: host selection/attestation must not manufacture provider effective_mode")
            if kind in {"ui_user_attested", "host_metadata"}:
                warnings.append(f"{label}: host/attestation selection only; provider effective mode remains unknown")
        if evidenced not in allowed:
            errors.append(f"{label}: observed/effective mode is not authorized for this run")
        elif evidenced != "pro":
            if not has_current_policy:
                errors.append(f"{label}: non-Pro requires an explicit current-run policy")
            warnings.append(f"{label}: authorized alternative {evidenced}; do not call this Pro")
        artifact(value.get("receipt_path"), f"{label}.receipt_path")

    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        errors.append("missing run_id")
    if record.get("status") not in STATES:
        errors.append("invalid run status")
    tasks = record.get("tasks", [])
    if not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks):
        return errors + ["tasks must be a list of objects"], warnings
    ids = [t.get("task_id") for t in tasks]
    if any(not isinstance(t, str) or not t for t in ids) or len(set(str(t) for t in ids)) != len(ids):
        errors.append("missing or duplicate task IDs")
    lookup = {str(t.get("task_id")): t for t in tasks}
    contexts: dict[str, str] = {}
    completed = [t for t in tasks if t.get("status") == "completed"]
    if completed or final:
        if has_current_policy:
            if policy.get("authorization_scope") != "this_run":
                errors.append("mode_policy: prior-run authorization is not current-run authorization")
            artifact(policy.get("authorization_receipt"), "mode_policy.authorization_receipt")
        mode(record.get("mode_evidence"), "run")
    for t in tasks:
        name = str(t.get("task_id"))
        status = t.get("status")
        if status not in STATES:
            errors.append(f"{name}: invalid status")
        deps = t.get("depends_on", [])
        if not isinstance(deps, list):
            errors.append(f"{name}: depends_on must be a list")
            deps = []
        for dep in deps:
            if dep not in lookup:
                errors.append(f"{name}: missing dependency {dep}")
            elif status == "completed" and lookup[dep].get("status") != "completed":
                errors.append(f"{name}: completed before dependency {dep}")
        if status != "completed":
            continue
        if t.get("substantive", True):
            mode(t.get("mode_evidence"), name)
        artifact(t.get("output_path"), f"{name}.output_path")
        artifact(t.get("input_manifest"), f"{name}.input_manifest")
        if t.get("independent_branch", False):
            context = t.get("context_id")
            if not isinstance(context, str) or not context:
                errors.append(f"{name}: independent branch lacks context ID")
            elif context in contexts:
                errors.append(f"{name}: independent context reused by {contexts[context]}")
            else:
                contexts[context] = name
            if t.get("prior_report_seen_before_freeze", True):
                errors.append(f"{name}: previous proposal not quarantined before freeze")
            if t.get("isolation_kind") not in {"runtime_enforced", "instruction_only"}:
                errors.append(f"{name}: missing isolation declaration")
            elif t.get("isolation_kind") == "instruction_only":
                warnings.append(f"{name}: context independence relies on instructions, not access controls")
    # Detect dependency cycles even among unfinished tasks.
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(task_id: str) -> None:
        if task_id in visiting:
            errors.append(f"dependency cycle at {task_id}")
            return
        if task_id in visited or task_id not in lookup:
            return
        visiting.add(task_id)
        deps = lookup[task_id].get("depends_on", [])
        for dep in deps if isinstance(deps, list) else []:
            if isinstance(dep, str):
                visit(dep)
        visiting.remove(task_id)
        visited.add(task_id)
    for task_id in lookup:
        visit(task_id)
    probes = record.get("probes", [])
    if not isinstance(probes, list):
        errors.append("probes must be a list")
        probes = []
    for i, probe in enumerate(probes):
        if not isinstance(probe, dict):
            errors.append(f"probe {i}: must be an object")
            continue
        if not probe.get("executed", False):
            continue
        name = str(probe.get("probe_id", i))
        for field in ("contract_path", "code_path", "config_path", "log_path", "result_path"):
            artifact(probe.get(field), f"probe {name}.{field}")
        if not probe.get("command"):
            errors.append(f"probe {name}: missing execution command")
        if probe.get("claims_held_out", False) and probe.get("evaluation_isolation") != "runtime_enforced":
            errors.append(f"probe {name}: held-out claim lacks declared runtime isolation; label exploratory")
    if final:
        artifact(record.get("final_report"), "final_report")
        if record.get("status") != "completed":
            errors.append("final audit requires completed run status")
        if not completed:
            errors.append("final audit has no completed tasks")
    warnings.append("Passing this audit validates declared bookkeeping only, not novelty, correctness or actual isolation.")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
    except (OSError, ValueError) as exc:
        print(f"Cannot read record: {exc}", file=sys.stderr)
        return 2
    errors, warnings = audit(record, args.record.parent, args.final)
    print(json.dumps({"passed": not errors, "errors": errors, "warnings": warnings}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
