"""Read-only verification and joint P/cost ledger for actual fourth-stage runs.

No model execution, fitting, experiment selection, network or TEST changes.
Raw evidence remains immutable; the output is a derived evidence artifact.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("dynamic", "static", "gru", "transformer")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    evidence = {}
    for arm in ARMS:
        name = f"parameter-memory-live-stage-n8-seed0-{arm}-stage4-v1"
        directory = ROOT / "runs" / name / "learn"
        result_file, raw_file = directory / "result.json", directory / "records.jsonl"
        result = json.loads(result_file.read_text())
        events = [json.loads(line) for line in raw_file.read_text().splitlines()]
        updates = [r for r in events if r["kind"] == "actual_update"]
        assert [r["step"] for r in updates] == list(range(1, 601))
        assert result["new_training_updates"] == 600 and result["old_training_updates_repeated"] == 0
        assert result["formal_Primary"] is False and result["no_test_based_selection"] is True
        assert all(r["current_training_examples"] == 48 and r["replayed_training_examples"] == 16 for r in updates)
        costs = result["costs"]
        assert costs["SGD_updates"] == 600 and costs["training_examples"] == 38400
        assert costs["training_forward_MAC_proxy"] == sum(r["forward_MAC_proxy"] for r in updates)
        config = next(r for r in events if r["kind"] == "configuration")
        assert config["resources"]["trainable_parameters"] == result["resources"]["parameters"]
        receipt = result["checkpoint"]
        assert sha(directory / receipt["file"]) == receipt["sha256"]
        assert (directory / receipt["file"]).stat().st_size == receipt["bytes"]
        parent_result = json.loads((ROOT / "runs" / result["parent_run"] / "learn/result.json").read_text())
        test = result["frozen_test_metrics"]
        before = next(r["all_seen_test_metrics"] for r in parent_result["frozen_test_reports"]
                      if r["learned_through_stage"] == 2)
        coordinator = json.loads((ROOT / (name + "-coordinator-receipt.json")).read_text())
        assert coordinator["returncode"] == 0
        inference = result["inference"]["soft"]
        row = dict(
            arm=arm, run=name, parameters=result["resources"]["parameters"],
            training_trainable_parameters=config["resources"]["trainable_parameters"],
            added_token_embedding_parameters=result["growth"]["parameters_added"],
            VALID_new_soft=result["final_TRAIN_VALID"]["metrics"]["3"]["valid"]["soft"]["accuracy"],
            VALID_new_native=result["final_TRAIN_VALID"]["metrics"]["3"]["valid"]["native"]["accuracy"],
            new_IID_soft=test["3"]["test_iid"]["soft"]["accuracy"],
            new_long_soft=test["3"]["test_long"]["soft"]["accuracy"],
            new_unseen_soft=test["3"]["test_unseen"]["soft"]["accuracy"],
            new_IID_native=test["3"]["test_iid"]["native"]["accuracy"],
            parameter_payload_bytes=result["resources"]["parameter_bytes"],
            optimizer_tensor_bytes=result["resources"]["optimizer_tensor_bytes"],
            replay_array_bytes=result["resources"]["replay_bytes"],
            replay_metadata_bytes=result["resources"]["replay_metadata_JSON_payload_bytes"],
            extra_position_buffer_bytes=result["growth"]["registered_position_buffer_bytes_added"],
            training_seconds=costs["training_wall_seconds"],
            training_CPU_seconds=costs["training_CPU_seconds"],
            training_CUDA_stream_seconds=costs["CUDA_stream_elapsed_seconds"],
            training_forward_MAC_proxy=costs["training_forward_MAC_proxy"],
            complete_launch_SSH_coordinator_seconds=coordinator["wall_seconds"],
            TRAIN_VALID_evaluation_seconds=costs["evaluation_wall_seconds"],
            TRAIN_VALID_evaluation_forward_MAC_proxy=costs["evaluation_forward_MAC_proxy"],
            replay_maintenance_seconds=costs["replay_maintenance_wall_seconds"],
            restore_growth_seconds=next(r["restore_and_growth_wall_seconds"] for r in events if r["kind"] == "actual_parent_restore_and_storage_growth"),
            checkpoint_write_seconds=costs["checkpoint_write_wall_seconds"],
            frozen_checkpoint_bytes=receipt["bytes"],
            inference_B64_p50_seconds=inference["wall_latency_p50"],
            inference_B64_p95_seconds=inference["wall_latency_p95"],
            inference_total_measurement_seconds=result["inference_total_wall_seconds"],
            CUDA_peak_allocated_bytes=result["resources"]["cuda_peak_allocated_bytes"],
            CPU_peak_RSS_bytes=result["resources"]["cpu_peak_RSS_bytes"],
            extra_new_training_labels=costs["current_training_label_reads"],
            replay_training_label_exposures=costs["old_replay_training_label_reads"],
            exact_total_training_FLOPs=None, energy=None,
            deltaP_per_successfully_retained_new_capability=None)
        for phase in range(3):
            key = str(phase)
            row[f"old{phase}_IID_before"] = before[key]["test_iid"]["soft"]["accuracy"]
            row[f"old{phase}_IID_after"] = test[key]["test_iid"]["soft"]["accuracy"]
            row[f"old{phase}_IID_drop"] = row[f"old{phase}_IID_before"] - row[f"old{phase}_IID_after"]
        # Include every measured test/audit and historical cost without silently
        # treating overlapping timers or shared generation as additive totals.
        evidence[arm] = dict(result_sha256=sha(result_file), records_sha256=sha(raw_file),
            checkpoint=receipt, raw_costs=costs,
            frozen_test_measurement_costs=result["frozen_test_measurement_costs"],
            reload_audit=result["reload"], generator=result["offline_generator_costs_and_labels"],
            parent_run=result["parent_run"], last_saved_event_wall_seconds=events[-1]["wall"],
            last_event_scope="last raw event precedes frozen TEST evaluation; never interpreted as full driver time",
            complete_launch_SSH_coordinator_seconds=coordinator["wall_seconds"],
            protocol_sha256=sha(ROOT / (name + "-protocol.json")),
            source_sha256=result["source_sha256"], inference=result["inference"],
            actual_successful_updates=600, active_training_token_steps=sum(r["active_token_steps"] for r in updates))
        rows.append(row)
    out = ROOT / "memory-live-stage-parameter-complete-cost-evidence-v1.json"
    if out.exists():
        raise FileExistsError("append-only derived evidence; inspect or use a new revision")
    payload = dict(schema="joint-parameter-and-scoped-complete-cost-v1", rows=rows, evidence=evidence,
        generator_scope="one shared actual bounded CPU generation, reported separately, not multiplied by arm",
        comparison="same original task information, own phase2 parents, 600 new updates, 48new+16old, Replay128; single seed development only",
        scope="new phase learning costs shown; prior own paid training preserved in linked parent records, not erased or silently re-run",
        missing="unmeasured exact backward/total FLOPs, energy and physical price remain null; timings/counters have distinct nonadditive scopes",
        capability_scope="added token embedding is not a successfully retained capability; no ΔP/ΔCapability success claim is established",
        baseline_rule="actual fixed-budget evidence valid regardless of accuracy; no additional tuning authorized by low score")
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with out.with_suffix(".csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(output=str(out), rows=rows), sort_keys=True))


if __name__ == "__main__":
    main()
