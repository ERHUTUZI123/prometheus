#!/usr/bin/env python3
"""Check declared research records. Does not establish correctness or actual isolation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any

STATES = {"pending", "running", "checkpointed", "completed", "blocked", "closed"}


def audit(record: dict[str, Any], root: Path, final: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    root = root.resolve()

    def artifact(value: Any, label: str) -> None:
        if not isinstance(value, str) or not value:
            errors.append(label + ": missing artifact path")
            return
        raw = Path(value)
        p = (root / raw).resolve()
        if raw.is_absolute() or root not in p.parents or not p.is_file():
            errors.append(label + ": absent, not a file, or outside run root")

    def mode(value: Any, label: str) -> None:
        # USER OVERRIDE: current Ultra explicitly authorized, never relabelled Pro.
        # This replaces only the original Pro evidence policy. Other checks remain.
        if not isinstance(value, dict):
            errors.append(label + ": missing mode evidence")
            return
        if value.get("requested_mode") != "user_authorized_ultra" or value.get("effort") != "ultra":
            errors.append(label + ": current Ultra authorization/effort missing")
        if value.get("effective_mode") is not None or value.get("provider_effective_mode_verified") is not False:
            errors.append(label + ": unsupported provider-effective mode claim")
        if value.get("evidence_kind") != "host_metadata_plus_user_authorization":
            errors.append(label + ": inappropriate evidence kind")
        artifact(value.get("receipt_path"), label + ".receipt_path")
        try:
            receipts = json.loads((root / value["receipt_path"]).read_text())
            receipt = next(x for x in receipts if x.get("context_id") == value.get("context_id"))
            selected = receipt["host_selected"]
            if selected.get("model") != value.get("model") or selected.get("effort") != "ultra":
                errors.append(label + ": host receipt does not match selected route")
            policy = json.loads((root / "manifest.json").read_text())["mode_policy"]
            if policy.get("superseding_user_authorization") != ["Ultra就可以开始", "现在这个mode就可以开始"]:
                errors.append(label + ": explicit user override missing")
        except (OSError, ValueError, KeyError, StopIteration, TypeError):
            errors.append(label + ": malformed/unmatched host receipt or override")
        warnings.append(label + ": authorized host-selected Ultra; provider-effective reasoning.mode unknown, no Pro claim")

    if not isinstance(record.get("run_id"), str) or not record["run_id"]:
        errors.append("missing run_id")
    if record.get("status") not in STATES:
        errors.append("invalid run status")
    tasks = record.get("tasks", [])
    if not isinstance(tasks, list) or any(not isinstance(t, dict) for t in tasks):
        return errors + ["tasks must be objects in a list"], warnings
    task_ids = [t.get("task_id") for t in tasks]
    if any(not isinstance(x, str) or not x for x in task_ids):
        return errors + ["invalid task IDs"], warnings
    if len(set(task_ids)) != len(task_ids):
        errors.append("duplicate task IDs")
    lookup = {t["task_id"]: t for t in tasks}
    contexts: set[str] = set()
    completed = [t for t in tasks if t.get("status") == "completed"]
    if completed or final:
        mode(record.get("mode_evidence"), "run")
    for t in tasks:
        name = t["task_id"]
        if t.get("status") not in STATES:
            errors.append(name + ": invalid task status")
        deps = t.get("depends_on", [])
        if not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            errors.append(name + ": invalid dependencies")
            deps = []
        for dep in deps:
            if dep not in lookup:
                errors.append(name + ": missing dependency " + dep)
            elif t.get("status") == "completed" and lookup[dep].get("status") != "completed":
                errors.append(name + ": completed before dependency " + dep)
        if t.get("status") != "completed":
            continue
        if t.get("substantive", True):
            mode(t.get("mode_evidence"), name)
        artifact(t.get("output_path"), name + ".output_path")
        artifact(t.get("input_manifest"), name + ".input_manifest")
        if t.get("independent_branch", False):
            context = t.get("context_id")
            if not isinstance(context, str) or not context or context in contexts:
                errors.append(name + ": missing/reused independent context ID")
            else:
                contexts.add(context)
            if t.get("prior_report_seen_before_freeze", True):
                errors.append(name + ": old report not quarantined before freeze")
            isolation = t.get("isolation_kind")
            if isolation not in {"runtime_enforced", "instruction_only"}:
                errors.append(name + ": missing isolation declaration")
            elif isolation == "instruction_only":
                warnings.append(name + ": instruction-only isolation")
    visited: set[str] = set()
    visiting: set[str] = set()
    def visit(name: str) -> None:
        if name in visiting:
            errors.append("dependency cycle at " + name)
            return
        if name in visited or name not in lookup:
            return
        visiting.add(name)
        deps = lookup[name].get("depends_on", [])
        if isinstance(deps, list):
            for dep in deps:
                if isinstance(dep, str):
                    visit(dep)
        visiting.remove(name)
        visited.add(name)
    for name in lookup:
        visit(name)
    probes = record.get("probes", [])
    if not isinstance(probes, list):
        errors.append("probes must be a list")
        probes = []
    for index, probe in enumerate(probes):
        if not isinstance(probe, dict):
            errors.append(f"probe {index}: invalid object")
            continue
        if not probe.get("executed", False):
            continue
        label = "probe " + str(probe.get("probe_id", index))
        for key in ("contract_path", "code_path", "config_path", "log_path", "result_path"):
            artifact(probe.get(key), label + "." + key)
        if not probe.get("command"):
            errors.append(label + ": missing command")
        if probe.get("claims_held_out") and probe.get("evaluation_isolation") != "runtime_enforced":
            errors.append(label + ": held-out claim lacks declared runtime isolation; use exploratory label")
    if final:
        artifact(record.get("final_report"), "final_report")
        if record.get("status") != "completed" or not completed:
            errors.append("final audit requires completed run and actual completed tasks")
    warnings.append("Audit checks declared bookkeeping and file existence, not novelty, truth, real isolation or source comprehension.")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    try:
        data = json.loads(args.record.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("record must be an object")
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    errors, warnings = audit(data, args.record.parent, args.final)
    print(json.dumps({"passed": not errors, "errors": errors, "warnings": warnings}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
