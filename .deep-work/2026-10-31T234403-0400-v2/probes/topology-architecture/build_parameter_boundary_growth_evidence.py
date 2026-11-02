"""Build frozen evidence tables from OWN JSON/raw/ZIP metadata, no unpickling.

No model import, inference, training, SSH, literature search or Method update.
Existing measurements stay labelled with their scopes and missing values.
"""
import csv
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
RUNS = ROOT/"runs"
SPLITS = ("iid", "ood", "length16")
THRESHOLDS = dict(iid=.95, ood=.90, length16=.90)
INPUTS = [
    ("N8_Topology_V1", "parameter-n8-seed0-topology-through0", (0,)),
    ("N8_Topology_V1", "parameter-n8-seed0-topology-resumed", (1,2,3,4,5)),
    ("N8_Topology_boundary", "parameter-n8-seed0-topology-through0", (0,)),
    ("N8_Topology_boundary", "parameter-n8-seed0-topology-resumed-consistent", (1,2,3,4,5)),
    ("N8_Constraint_compact_boundary", "parameter-n8-seed0-constraint_compact-full", (0,)),
    ("N8_Constraint_compact_boundary", "parameter-n8-seed0-constraint_compact-resumed-consistent", (1,2,3,4,5)),
    ("N8_Modular_boundary", "parameter-n8-seed0-modular-full-consistent", (0,1,2,3,4,5)),
    ("N8_Freeze_readout_boundary", "parameter-n8-seed0-freeze-full-consistent", (0,1,2)),
    ("N8_Freeze_readout_boundary", "parameter-n8-seed0-freeze-resumed-consistent", (3,4,5)),
] + [(f"N8_{arm}", f"parameter-n8-seed0-{arm}-full", tuple(range(6)))
     for arm in ("transformer64", "replay64", "transformer96", "replay96")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def relative(path):
    return str(path.relative_to(ROOT))


def passed(metrics):
    return {int(t): all(m[s][mode] >= threshold for s,threshold in THRESHOLDS.items()
                       for mode in ("neural", "deployed")) for t,m in metrics.items()}


def minimum(metrics, mode, old_before=None):
    values = [m[s][mode] for t,m in metrics.items() if old_before is None or int(t)<old_before
              for s in SPLITS]
    return min(values) if values else None


def cache_snapshot(resources):
    geometry = resources.get("geometry", [])
    def total(path):
        values=[]
        for g in geometry:
            value=g
            for key in path:
                value=value.get(key) if isinstance(value,dict) else None
            if value is not None:values.append(value)
        return sum(values) if values else 0 if not geometry else None
    return dict(geometry_members=len(geometry),
        resident_homotopy_payload_bytes=total(("homotopy_lift_cache","packed_payload_bytes")),
        resident_homotopy_map_bytes_SUBSET=total(("homotopy_lift_cache","map_data_bytes")),
        resident_homotopy_certificate_bytes_SUBSET=total(("homotopy_lift_cache","certificate_bytes")),
        resident_current_map_view_bytes_MAY_ALIAS=total(("costs","map_cache_bytes_current")),
        geometry_reports=geometry,
        scope="snapshot; map bytes/certificates are subsets of packed payload; current views may alias; reports' counters inherit earlier phases, not new per-stage work")


def main():
    rows, curves, sources, run_cache = [], [], {}, {}

    def source(path):
        key=relative(path)
        if key not in sources:
            sources[key]=dict(path=key,sha256=sha(path),bytes=path.stat().st_size,
                              hash_scope="local bytes at this evidence build; execution-era checkpoint hashes checked separately")
        return key

    def run(name):
        if name in run_cache:return run_cache[name]
        directory=RUNS/name;learn=directory/"learn"
        raw_path=learn/"records.jsonl";raw=[json.loads(x) for x in raw_path.read_text().splitlines()]
        source(raw_path)
        path=learn/"summary.json" if (learn/"summary.json").exists() else learn/"failure.json"
        info=load(path);source(path)
        marker=directory/"sync-hashes.json"
        if marker.exists():
            manifest=load(marker)
            for folder,files in manifest.items():
                if not folder.startswith("outputs/"):continue
                for file,record in files.items():
                    item=directory/file
                    if item.stat().st_size!=record["bytes"] or sha(item)!=record["sha256"]:
                        raise ValueError(f"synchronized evidence mismatch: {item}")
            source(marker)
        point=dict(directory=directory,learn=learn,raw=raw,info=info,summary_path=path,
            integrity="completed sync-hashes verified" if marker.exists() else
                      "existing raw/checkpoint event hash + local source hashes; no new sync marker claimed")
        run_cache[name]=point;return point

    def add(series,name,stages,scale=8,seed=0):
        packet=run(name);reports={int(r["stage"]):r for r in packet["info"].get("reports",[])}
        raw=packet["raw"];directory=packet["learn"]
        for stage in stages:
            stored=reports.get(stage)
            cc=[x for x in raw if x["kind"]=="curve" and int(x["stage"])==stage]
            updates=[x for x in raw if x["kind"]=="actual_update" and int(x["stage"])==stage]
            if not cc or len(updates)!=600:raise ValueError(f"incomplete raw stage: {name}/{stage}")
            if stored is None:
                # Actual ordinaryC stage0 was saved after600 updates, then its
                # old reload helper failed BEFORE a completed report existed.
                if not (name=="parameter-n8-seed0-constraint_compact-full" and stage==0):
                    raise ValueError("unrecognized missing stage report")
                final=next(c for c in cc if c["step"]==600)
                metrics,resources=final["metrics"],final["resources"]
                P=resources["parameters"]["total"];success=passed(metrics)
                admitted=all(metrics["0"]["valid"][m]>=.95 for m in ("neural","deployed"))
                stable=[0] if admitted and success[0] else []
                caps=dict(stable=stable,total_stable=len(stable),currently_successful=[t for t,v in success.items() if v],
                    admitted=[0] if admitted else [],delta_parameters=P,
                    deltaP_per_new_retained_capability=float(P) if stable else None)
                caps_source="independently derived from actual600-step curve using the frozen common thresholds; not a recovered missing report"
            else:
                metrics,resources=stored["final_metrics"],stored["resources"]
                P=stored["parameters"];caps=stored["capabilities"]
                caps_source="actual saved stage report"
            event=next((x for x in raw if x["kind"]=="checkpoint" and x["stage"]==stage and x["step"]==600),None)
            if event is None:raise ValueError("actual600-step checkpoint event required")
            checkpoint=directory/event["file"]
            if sha(checkpoint)!=event["sha256"]:raise ValueError("actual neural checkpoint hash mismatch")
            source(checkpoint)
            with zipfile.ZipFile(checkpoint) as archive:
                zip_sizes=dict(checkpoint_zip_pickle_metadata_bytes=sum(x.file_size for x in archive.infolist() if x.filename.endswith("data.pkl")),
                    checkpoint_zip_tensor_storage_bytes=sum(x.file_size for x in archive.infolist() if "/data/" in x.filename),
                    scope="ZIP central-directory sizes only; not unpickled; pickle metadata includes args/RNG/history, not just model-extra")
            for c in cc:
                for task,split_metrics in c["metrics"].items():
                    for split,m in split_metrics.items():
                        curves.append(dict(series=series,scale=scale,seed=seed,stage=stage,step=c["step"],
                            evaluated_task=int(task),split=split,rows=m["n"],neural_accuracy=m["neural"],
                            deployed_accuracy=m["deployed"],terminal_CE=m["terminal_CE"],
                            boundary_after_refresh=bool(stored and stored.get("boundary_consistency_refresh") and c["step"]==600),
                            source_raw=relative(directory/"records.jsonl"),source_raw_sha256=sha(directory/"records.jsonl")))
            intermediate=[c for c in cc if c["step"]<600]
            old_values=[minimum(c["metrics"],"neural",stage) for c in intermediate]
            old_values=[v for v in old_values if v is not None]
            old_deployed=[minimum(c["metrics"],"deployed",stage) for c in intermediate]
            old_deployed=[v for v in old_deployed if v is not None]
            formation=next((x for x in raw if x["kind"]=="actual_neural_formation" and x["stage"]==stage),None)
            observation=directory/f"stage{stage}-observations.json"
            candidate=directory/f"stage{stage}-formation.json"
            oa=load(observation) if observation.exists() else None
            ca=load(candidate) if candidate.exists() else None
            if oa is not None:source(observation)
            if ca is not None:source(candidate)
            first=next(c for c in cc if c["step"]==0)
            is_replay="replay" in series.lower()
            current_episodes=(600*(32 if is_replay and stage else 64))
            old_episodes=600*32 if is_replay and stage else 0
            if stored:
                for key,expected in (("actual_current_training_episodes",current_episodes),
                                     ("actual_old_training_episodes",old_episodes)):
                    if key in stored and stored[key]!=expected:
                        raise ValueError("saved episode count disagrees with the frozen batch allocation")
            install_source=stored.get("formation",{}) if stored else formation.get("report",{}) if formation else {}
            install_seconds=install_source.get("costs",{}).get("seconds") if install_source else None
            row=dict(series=series,scale=scale,seed=seed,stage=stage,source_run=name,
                actual_optimizer_updates=len(updates),actual_current_training_episodes=stored.get("actual_current_training_episodes",current_episodes) if stored else current_episodes,
                actual_old_training_episodes=stored.get("actual_old_training_episodes",old_episodes) if stored else old_episodes,
                episodes_source="saved field if available; otherwise direct arithmetic from actual600 updates and frozen64/(32+32) batch rule",
                P_total=P,P_trainable=resources["parameters"].get("trainable"),
                deltaP=stored["delta_parameters"] if stored else P,
                Qstable=caps["total_stable"],current_pass_count=len(caps["currently_successful"]),
                current_pass_tasks=caps["currently_successful"],stable_tasks=caps["stable"],
                deltaP_per_new_retained_capability=caps.get("deltaP_per_new_retained_capability"),
                capability_score_source=caps_source,capabilities=caps,
                all_seen_neural_final_min=minimum(metrics,"neural"),all_seen_deployed_final_min=minimum(metrics,"deployed"),
                current_task_neural_final_min=minimum({str(stage):metrics[str(stage)]},"neural"),
                current_task_deployed_final_min=minimum({str(stage):metrics[str(stage)]},"deployed"),
                old_intermediate_neural_min=min(old_values) if old_values else None,
                old_intermediate_deployed_min=min(old_deployed) if old_deployed else None,
                intermediate_scope="recorded steps0/100/300, old tasks only, IID/OOD/L16; not a minimum over every update",
                new_task_valid_after_formation_before_SGD=first["metrics"][str(stage)]["valid"] if formation else None,
                formation_precedes_SGD=bool(formation),
                final_metrics=metrics,
                times=dict(train_seconds=stored.get("train_seconds") if stored else None,
                    old_guard_seconds=stored.get("guard_seconds") if stored else None,
                    boundary_refresh_seconds=stored.get("boundary_consistency_seconds",0.) if stored else None,
                    reload_seconds=stored.get("reload_seconds") if stored else None,
                    evaluation_seconds=stored.get("eval_seconds") if stored else None,
                    entire_phase_wall_seconds=stored.get("phase_wall_seconds") if stored else None,
                    observation_index_seconds=oa.get("seconds") if oa else None,
                    candidate_constraint_seconds=ca.get("formation_seconds") if ca else None,
                    neural_installation_seconds=install_seconds),
                time_scope="phase wall includes nested components; do not sum phase wall with them; concurrent workloads prevent controlled throughput ranking; NULL means missing original measurement",
                storage=dict(parameter_tensor_bytes=resources["parameters"]["bytes"],
                    model_extra_json_bytes=resources.get("model_extra_json_bytes"),
                    reference_json_bytes_SUBSET=resources.get("complete_reference_json_bytes"),
                    reference_array_bytes_SUBSET=resources.get("reference_array_bytes"),
                    registered_buffer_bytes=resources.get("registered_buffer_bytes"),
                    replay_tensor_bytes=resources.get("replay_tensor_bytes"),
                    optimizer_tensor_bytes=resources.get("optimizer_tensor_bytes"),
                    checkpoint_bytes=checkpoint.stat().st_size,checkpoint_zip=zip_sizes,
                    cache=cache_snapshot(resources),
                    scope="logical snapshots, JSON proxies and full serialized checkpoint have different overlapping scopes; no summed universal memory number"),
                complete_resources=resources,
                reload_verified=stored.get("actual_reload_verified") if stored else None,
                reload_resources=stored.get("reload_resources") if stored else None,
                boundary_consistency_refresh=stored.get("boundary_consistency_refresh") if stored else None,
                source_summary=relative(packet["summary_path"]),source_raw=relative(directory/"records.jsonl"),
                source_checkpoint=relative(checkpoint),source_checkpoint_execution_event_sha256=event["sha256"],
                integrity=packet["integrity"],
                missing_original_stage_report=stored is None,
                historical_status_preserved="unrefreshed V1 failures remain; generic consistency/readout repairs not topology-exclusive")
            rows.append(row)

    for series,name,stages in INPUTS:add(series,name,stages)
    # Optional already completed N16 chunks are separate PARTIAL rows. No wait
    # for current training and no inference of absent stages.
    n16_chunks=[]
    for directory in sorted(RUNS.glob("parameter-n16-seed*-*")):
        summary=directory/"learn/summary.json"
        if not directory.is_dir() or not (directory/"sync-hashes.json").exists() or not summary.exists():continue
        info=load(summary)
        if not info.get("reports") or info.get("arm") not in ("topology","transformer64","replay64","transformer96","replay96"):continue
        seed=int(info.get("seed",0));arm=info["arm"]
        series=f"N16_{arm}_"+("boundary" if "consistent" in directory.name else "partial")+"__"+directory.name
        stages=tuple(int(r["stage"]) for r in info["reports"])
        n16_chunks.append(dict(series=series,source_run=directory.name,stages=list(stages)))
        add(series,directory.name,stages,scale=16,seed=seed)
    rows.sort(key=lambda r:(r["scale"],r["series"],r["seed"],r["stage"]))
    payload=dict(schema="frozen-parameter-boundary-growth-evidence-v1",MAIN="TOPOLOGY_USER_LOCKED",
        current_scope="fixed six paper TODOs; no progress-percentage inference",
        comparison_scope="N8/seed0 completed basic synthetic streams; any N16 chunks explicitly listed and not presumed complete",
        thresholds=dict(iid=.95,ood=.90,length16=.90,both_interfaces=True,valid_admission=.95),
        scoring_boundary="Qstable is task-end checkpoint stability; intermediate soft dips remain visible",
        resource_boundary="equal600 gradients is not equal total formation/learning cost; no concurrent wall-time ranking; no nested-cost sum",
        n16_synced_chunks=n16_chunks,rows=rows,sources=list(sources.values()),
        provenance=dict(builder_file=Path(__file__).name,builder_sha256=sha(Path(__file__)),
            no_model_import=True,no_unpickling=True,no_new_training_or_inference=True,
            shared_phase0_sources_are_reused_not_independent_replications=True,
            no_summing_costs_across_versions_as_if_independent_lineages=True),
        unsupported_claims=["topology-exclusive parameter advantage over matched ordinaryC",
            "old neural correctness at every training update","full N8/N16/N32 three-seed evidence",
            "generic or real-data architecture advantage","random-performing Transformer is a strong solved-task baseline"])
    stem=ROOT/"parameter-boundary-growth-evidence"
    stem.with_suffix(".json").write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    fields=("series","scale","seed","stage","P_total","P_trainable","deltaP","Qstable","current_pass_count",
        "deltaP_per_new_retained_capability","all_seen_neural_final_min","all_seen_deployed_final_min",
        "old_intermediate_neural_min","old_intermediate_deployed_min","actual_optimizer_updates",
        "actual_current_training_episodes","actual_old_training_episodes","source_run","missing_original_stage_report")
    timekeys=("train_seconds","old_guard_seconds","boundary_refresh_seconds","reload_seconds","evaluation_seconds",
              "entire_phase_wall_seconds","observation_index_seconds","candidate_constraint_seconds","neural_installation_seconds")
    bytekeys=("parameter_tensor_bytes","model_extra_json_bytes","reference_json_bytes_SUBSET","reference_array_bytes_SUBSET",
              "registered_buffer_bytes","replay_tensor_bytes","optimizer_tensor_bytes","checkpoint_bytes")
    cachekeys=("resident_homotopy_payload_bytes","resident_homotopy_map_bytes_SUBSET",
               "resident_homotopy_certificate_bytes_SUBSET","resident_current_map_view_bytes_MAY_ALIAS")
    def flat(r):
        d={k:r[k] for k in fields};d.update(r["times"])
        d.update({k:r["storage"][k] for k in bytekeys});d.update({k:r["storage"]["cache"][k] for k in cachekeys})
        return {k:"NULL" if v is None else v for k,v in d.items()}
    with stem.with_suffix(".csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields+timekeys+bytekeys+cachekeys);w.writeheader();w.writerows(flat(r) for r in rows)
    with (ROOT/"parameter-boundary-learning-curves.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    lines=["# Frozen parameter/boundary-growth evidence", "",
        "Built only from actual local JSON/raw and ZIP sizes; no model was imported, unpickled, trained or evaluated. This is a frozen evidence artifact serving the six paper TODOs, not another current Method or plan.", "",
        "N8/seed0 has nine policies/versions below. Topology V1 failures remain V1 failures. Boundary maintenance and frozen-readout repair are generic rules available to the controls. Transformer failure is an acquisition problem, not evidence of a strong successful baseline being beaten. Shared stage0 sources are reused, not independent replications; do not sum versions as independent training lineages.", "",
        "|Series|Stage|P|ΔP|Qstable|Current pass|Final neural min|Final deployed min|Old intermediate neural min|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["scale"]!=8:continue
        def fmt(x):return "NULL" if x is None else f"{x:.6g}" if isinstance(x,float) else str(x)
        lines.append("|"+"|".join(fmt(r[k]) for k in ("series","stage","P_total","deltaP","Qstable","current_pass_count",
            "all_seen_neural_final_min","all_seen_deployed_final_min","old_intermediate_neural_min"))+"|")
    lines.extend(["", "Scores use task-end IID≥.95, OOD/L16≥.90 for both neural/deployed; VALID≥.95 controls admission. ‘Old intermediate’ is the observed minimum at0/100/300, not every update. Initial P is fully charged. Ratio is NULL when an old stable task is lost or no positive retained increment occurs.", "",
        "The C stage0 checkpoint exists after600 updates, but its old reload helper failed before the completed stage report. Its P/metrics/scoring are explicitly derived from real curve/checkpoint events; missing train/eval/reload/phase-wall measurements remain NULL, and successful reload is not backfilled.", "",
        "All times/storage columns are in the CSV and unabridged source resources in JSON. Phase wall encloses train/guard/refresh/eval/reload components and must not be added to them. Reference bytes overlap model-extra JSON; cache map/certificate bytes are subsets and current views may alias; checkpoint size includes optimizer/RNG/history. No single summed ‘total memory’ or concurrent throughput ranking is fabricated.", "",
        "Formation precedes Adam for the native learners; its observation, constraint-solving and NN-installation costs are retained where actually measured. Equal600 gradient updates does not remove these extra data passes or formation work. Current/old sample allocation for Replay is reported separately.", "",
        "N16 rows, if any, are only the explicitly synced completed chunks listed in JSON. Absent stages/scales/seeds are not silently filled. No new N32, complex-program, general Block or real-data success follows from this evidence snapshot.", "",
        "Files: `parameter-boundary-growth-evidence.json`, `parameter-boundary-growth-evidence.csv`, `parameter-boundary-learning-curves.csv`. Every used source and actual600-step checkpoint has an integrity hash in JSON; new sync markers are verified where present. The builder is `build_parameter_boundary_growth_evidence.py`.", ""])
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(dict(status="FROZEN_EVIDENCE_WRITTEN",stage_rows=len(rows),curve_rows=len(curves),
        source_files=len(sources),n16_chunks=n16_chunks,no_unpickle=True),sort_keys=True))


if __name__=="__main__":main()
