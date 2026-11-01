"""Extract only host session/mode metadata for this project's campaign contexts.

Host configuration is not provider-side effective-mode verification. Never copy
messages, tool payloads, credentials, or unrelated project sessions into receipts.
"""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[2]
PROJECT = RUN.parents[1]
parent_id = "01a18aa7-d28f-7192-aeef-fde8a9b8e4ed"
allowed = {"/root/" + name for name in (
    "branch_a", "branch_b", "branch_c", "branch_d", "branch_e", "branch_f",
    "cross_ae", "cross_cd", "cross_bf", "independent_rebuild", "final_attack",
    "ae_acquisition_repair")}
receipts = {}
for day in ("2026/10/29", "2026/10/31", "2026/11/01"):
    directory = Path("/Users/xiaoyangliu/.codex/sessions") / day
    for path in sorted(directory.glob("*.jsonl")):
        with path.open() as handle:
            try:
                first = json.loads(next(handle))
            except (StopIteration, ValueError):
                continue
            meta = first.get("payload", {})
            if first.get("type") != "session_meta" or meta.get("cwd") != str(PROJECT):
                continue
            source = meta.get("source")
            spawn = (source.get("subagent", {}).get("thread_spawn", {})
                     if isinstance(source, dict) else {})
            context = spawn.get("agent_path")
            if not (meta.get("id") == parent_id or context in allowed):
                continue
            latest = {}
            for line in handle:
                if '"turn_context"' not in line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if item.get("type") == "turn_context":
                    p = item.get("payload", {})
                    latest = {"model": p.get("model"), "effort": p.get("effort"),
                              "reasoning_mode": p.get("reasoning", {}).get("mode")
                              if isinstance(p.get("reasoning"), dict) else None}
            context = context or "/root"
            receipt = {
                "context_id": context, "session_id": meta.get("id"),
                "parent_thread_id": spawn.get("parent_thread_id"),
                "host_log": str(path), "host_selected": latest,
                "provider_effective_reasoning_mode": None,
                "evidence_scope": "host configuration only; not provider-effective Pro attestation"
            }
            old = receipts.get(context)
            # Prefer the canonical original root log over replay files carrying
            # the same root id. Child contexts have unique session ids.
            if old is None or (context == "/root" and parent_id in path.name):
                receipts[context] = receipt
(RUN / "probes/preflight/host-metadata-refreshed.json").write_text(
    json.dumps(list(receipts.values()), indent=2) + "\n")
print(json.dumps({k: v["host_selected"] for k, v in receipts.items()}, indent=2))
