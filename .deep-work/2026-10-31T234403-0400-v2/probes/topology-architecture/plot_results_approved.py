"""Read actual architecture records; export provenance tables and Matplotlib plots.

Usage: python plot_results.py --runs RUNS_DIRECTORY --out NEW_OUTPUT_DIRECTORY
No training, model/checkpoint loading, SSH or inference. No reported result is
hard-coded. CPU diagnostics and first-stage ablations remain separate strata.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict


SPLITS = ("valid", "iid", "ood", "length16")
MODES = ("soft", "hard")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def read_json(path, warnings):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError) as error:
        warnings.append(dict(path=str(path), reason=str(error)))
        return {}


def sidecars(run_root, learn, warnings):
    choices = [run_root.parent / (run_root.name + ".runner-status.json"),
               run_root / "runner-status.json", run_root / "receipt.json",
               run_root / "manifest.json", run_root / "source-manifest.json",
               learn / "manifest.json"]
    found = []
    for path in dict.fromkeys(choices):
        if path.is_file():
            found.append((path, read_json(path, warnings)))
    return found


def read_run(learn, root, warnings):
    records_path, summary_path = learn / "records.jsonl", learn / "summary.json"
    rows, sources = [], []
    if records_path.is_file():
        before = records_path.stat()
        digest = hashlib.sha256()
        with records_path.open("rb") as stream:
            for line_no, line in enumerate(stream, 1):
                digest.update(line)
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        row = dict(row, _line=line_no)
                        rows.append(row)
                except (ValueError, UnicodeError) as error:
                    warnings.append(dict(path=str(records_path), line=line_no,
                                         reason="invalid/partial JSONL: " + str(error)))
        after = records_path.stat()
        sources.append(dict(path=str(records_path), sha256=digest.hexdigest(),
                            bytes_read=after.st_size,
                            stable_during_read=(before.st_size, before.st_mtime_ns)
                            == (after.st_size, after.st_mtime_ns)))
    summary = read_json(summary_path, warnings) if summary_path.is_file() else {}
    if summary_path.is_file():
        sources.append(dict(path=str(summary_path), sha256=sha256(summary_path)))
    configuration = next((r for r in rows if r.get("kind") == "configuration"), {})
    args = configuration.get("args", summary.get("args", {}))
    if not isinstance(args, dict):
        args = {}
    run_root = learn.parent if learn.name == "learn" else learn
    side = sidecars(run_root, learn, warnings)
    for path, value in side:
        sources.append(dict(path=str(path), sha256=sha256(path)))
    source_hashes = configuration.get("source_sha256", summary.get("source_sha256", {}))
    for _, value in side:
        candidate = value.get("source_sha256", value.get("source_hashes", {}))
        if candidate:
            source_hashes = candidate
            break
    if not isinstance(source_hashes, dict):
        source_hashes = {"recorded_source": str(source_hashes)} if source_hashes else {}
    explicit_version = args.get("version", configuration.get("version", summary.get("version")))
    if source_hashes:
        version = hashlib.sha256(canonical(source_hashes).encode()).hexdigest()[:12]
        version_evidence = "recorded_source_sha256"
    elif explicit_version is not None:
        version = str(explicit_version)
        version_evidence = "explicit_label_without_source_hashes"
    else:
        # NEVER assume that current local source was the source of an old run.
        version = "unknown-" + hashlib.sha256(str(run_root).encode()).hexdigest()[:10]
        version_evidence = "unrecorded; isolated per run to avoid false version pooling"
    stages = args.get("stages", 3)
    device = str(configuration.get("device", args.get("device", "unknown")))
    scope = ("CPU_DEBUG" if device.startswith("cpu") or args.get("pilot")
             else "FIRST_STAGE_ABLATION" if stages == 1
             else "SMALL_SCALE_DEBUG" if args.get("scale") == 4
             else "MULTISTAGE_EXPERIMENT_NOT_M0_CERTIFICATE")
    identity = dict(version=version, scale=args.get("scale"), seed=args.get("seed"),
                    steps=args.get("steps"), stages=stages, batch=args.get("batch"),
                    lr=args.get("lr"), latent_factor=args.get("latent_factor"),
                    compilation=args.get("compilation", "unrecorded"),
                    credit=args.get("credit", "unrecorded"),
                    protection_timing=args.get("protection_timing", "unrecorded"),
                    device=device, scope=scope)
    condition = hashlib.sha256(canonical(identity).encode()).hexdigest()[:12]
    run_id = str(run_root.relative_to(root)) if run_root.is_relative_to(root) else str(run_root)
    execution_status = next((value.get("status") for path, value in side
                             if path.name.endswith(".runner-status.json") and value.get("status")),
                            summary.get("status", "UNRECORDED"))
    return dict(run_id=run_id, execution_status=execution_status, run_root=str(run_root), learn=str(learn), rows=rows,
                summary=summary, args=args, sources=sources, identity=identity,
                condition_id=condition, source_hashes=source_hashes,
                version_evidence=version_evidence,
                source_records=str(records_path) if records_path.is_file() else None,
                records_sha256=next((x["sha256"] for x in sources if x["path"] == str(records_path)), None),
                source_summary=str(summary_path) if summary_path.is_file() else None)


def provenance(run, row):
    return dict(run_id=run["run_id"], condition_id=run["condition_id"],
                **run["identity"], source_records=run["source_records"],
                records_sha256=run["records_sha256"], source_summary=run["source_summary"],
                source_line=row.get("_line"), source_pointer=row.get("_pointer"),
                source_hashes=run["source_hashes"], version_evidence=run["version_evidence"])


def metric_map(row):
    metrics = row.get("terminal_metrics")
    if isinstance(metrics, dict):
        return metrics
    result = dict(row.get("old", {}))
    if isinstance(row.get("current"), dict):
        result[str(row.get("stage", 0))] = row["current"]
    return result


def metric_rows(run, row, origin):
    result = []
    stage = int(row.get("stage", 0))
    for task, splits in metric_map(row).items():
        if not isinstance(splits, dict):
            continue
        try:
            task = int(task)
        except (TypeError, ValueError):
            continue
        for split, metric in splits.items():
            if not isinstance(metric, dict):
                continue
            for mode in MODES:
                accuracy = number(metric.get(mode + "_acc"))
                if accuracy is None:
                    continue
                if not 0 <= accuracy <= 1:
                    raise ValueError("accuracy outside [0,1] in " + run["run_id"])
                result.append(dict(**provenance(run, row), model=row.get("model"),
                    training_stage=stage, task=task, task_seen_by_stage=task <= stage,
                    step=row.get("step", run["args"].get("steps")), split=split,
                    mode=mode, accuracy=accuracy, n=metric.get("n"),
                    terminal_loss=number(metric.get("loss")),
                    hard_mode=metric.get("hard_mode"), temperature=number(row.get("temperature")),
                    origin=origin, stage_complete=origin == "stage_end"))
    return result


def collect(runs):
    curves, stage_metrics, resources, failures, datasets = [], [], [], [], []
    for run in runs:
        stage_rows = {}
        model_begins = {}
        for row in run["rows"]:
            kind, model = row.get("kind"), row.get("model")
            if kind == "model_begin":
                model_begins[model] = row
            if kind == "curve":
                curves.extend(metric_rows(run, row, "curve"))
            elif kind == "stage_end":
                stage_rows[model, int(row["stage"])] = row
            elif kind in ("run_failed", "model_interrupted", "interrupted_checkpoint_failed"):
                failures.append(dict(**provenance(run, row), model=model,
                    training_stage=row.get("stage"), step=row.get("step"),
                    kind="recorded_training_failure", error=row.get("error"),
                    reason=row.get("reason"), event=kind))
            elif kind == "dataset":
                datasets.append(dict(**provenance(run, row), task=row.get("task"),
                    split=row.get("split"), dataset_file=row.get("file"),
                    dataset_sha256=row.get("sha256"), array_bytes=row.get("array_bytes"),
                    metadata=row.get("metadata"), collisions_with_train=row.get("collisions_with_task_train")))
        # Summary is fallback, never a second observation of the same stage.
        for model_index, completed in enumerate(run["summary"].get("completed", [])):
            for stage_index, row in enumerate(completed.get("stages", [])):
                key = completed.get("model"), int(row.get("stage", stage_index))
                if key not in stage_rows:
                    stage_rows[key] = dict(row, model=key[0], stage=key[1],
                        _pointer=f"/completed/{model_index}/stages/{stage_index}")
        for (model, stage), row in stage_rows.items():
            stage_metrics.extend(metric_rows(run, row, "stage_end"))
            params = row.get("parameters", model_begins.get(model, {}).get("parameters", {}))
            guard = row.get("guard", {})
            resources.append(dict(**provenance(run, row), model=model, training_stage=stage,
                parameters_total=params.get("total"), parameters_requires_grad=params.get("requires_grad"),
                parameters_active_after_masks=params.get("active_after_row_masks"),
                gpu_peak_bytes=row.get("gpu_peak_bytes"), optimizer_tensor_bytes=row.get("optimizer_tensor_bytes"),
                replay_memory_tensor_bytes=row.get("replay_memory_tensor_bytes"),
                representation_state_bytes=row.get("representation_state_bytes"),
                train_seconds=number(row.get("train_seconds")), wall_seconds=number(row.get("wall_seconds")),
                eval_seconds=number(row.get("eval_seconds")),
                guard_measured_seconds=number(guard.get("measured_seconds")),
                guard_reported_seconds=number(guard.get("reported_seconds")),
                guard_snapshot_seconds=number(guard.get("snapshot_seconds")),
                guard_calls=guard.get("calls"), guard_rejected=guard.get("rejected"),
                guard_accepted_full=guard.get("accepted_full"),
                guard_accepted_backtracked=guard.get("accepted_backtracked"),
                train_current_samples=row.get("train_current_samples"),
                train_old_samples=row.get("train_old_samples"),
                consolidation_seconds=number(row.get("consolidation_seconds"))))
    forgetting = []
    grouped = defaultdict(list)
    for row in stage_metrics:
        if row["split"] == "iid" and row["task_seen_by_stage"]:
            grouped[row["run_id"], row["model"], row["task"], row["mode"]].append(row)
    for group in grouped.values():
        group.sort(key=lambda row: row["training_stage"])
        final, earlier = group[-1], group[:-1]
        best = max((r["accuracy"] for r in earlier), default=None)
        forgetting.append(dict(**{k: v for k, v in final.items() if k not in
            ("accuracy", "n", "terminal_loss", "origin", "stage_complete")},
            final_stage=final["training_stage"], final_accuracy=final["accuracy"],
            earlier_best=best, forgetting_signed=None if best is None else best - final["accuracy"],
            forgetting_positive=None if best is None else max(0., best - final["accuracy"]),
            definition="max earlier completed stage-end IID - last completed stage-end IID; no future-task pretraining points",
            earlier_completed_stages=[r["training_stage"] for r in earlier]))
    for row in stage_metrics:
        if row["accuracy"] < 1:
            failures.append(dict(**provenance_from_flat(row), model=row["model"],
                training_stage=row["training_stage"], task=row["task"], split=row["split"], mode=row["mode"],
                kind="aggregate_terminal_mismatches", accuracy=row["accuracy"], n=row["n"],
                mismatch_count_estimate=None if not row["n"] else round(row["n"] * (1 - row["accuracy"])),
                reason="individual input/prediction failures were not logged; no cases fabricated"))
    costs = []
    cost_groups = defaultdict(list)
    for row in resources:
        cost_groups[row["run_id"], row["model"]].append(row)
    for group in cost_groups.values():
        final = max(group, key=lambda row: row["training_stage"])
        sums = ("train_seconds", "wall_seconds", "eval_seconds", "guard_measured_seconds",
                "guard_reported_seconds", "guard_snapshot_seconds", "consolidation_seconds",
                "guard_calls", "guard_rejected", "guard_accepted_full", "guard_accepted_backtracked",
                "train_current_samples", "train_old_samples")
        peaks = ("parameters_total", "parameters_requires_grad", "parameters_active_after_masks",
                 "gpu_peak_bytes", "optimizer_tensor_bytes", "replay_memory_tensor_bytes", "representation_state_bytes")
        summary = dict(**provenance_from_flat(final), model=final["model"],
                       completed_stages=sorted({r["training_stage"] for r in group}),
                       recorded_stage_count=len(group))
        for key in sums:
            values = [number(r.get(key)) for r in group if number(r.get(key)) is not None]
            summary[key + "_sum"] = sum(values) if values else None
        for key in peaks:
            values = [number(r.get(key)) for r in group if number(r.get(key)) is not None]
            summary[key + "_max"] = max(values) if values else None
        costs.append(summary)
    gaps = []
    pairs = defaultdict(dict)
    for row in stage_metrics:
        pairs[row["run_id"], row["model"], row["training_stage"], row["task"], row["split"]][row["mode"]] = row
    for pair in pairs.values():
        if "soft" in pair and "hard" in pair:
            soft, hard = pair["soft"], pair["hard"]
            gaps.append(dict(**provenance_from_flat(soft), model=soft["model"],
                training_stage=soft["training_stage"], task=soft["task"], split=soft["split"],
                soft_accuracy=soft["accuracy"], hard_accuracy=hard["accuracy"],
                soft_minus_hard=soft["accuracy"]-hard["accuracy"], hard_mode=hard["hard_mode"]))
    return dict(curves=curves, task_by_stage=stage_metrics, forgetting=forgetting,
                resources=resources, costs=costs, soft_hard_gaps=gaps,
                failure_cases=failures, datasets=datasets)


def provenance_from_flat(row):
    return {k: row.get(k) for k in ("run_id", "condition_id", "version", "scale", "seed", "scope",
            "source_records", "records_sha256", "source_summary", "source_line", "source_pointer", "source_hashes")}


def diagnoses(root, warnings):
    result = []
    for path in sorted(root.rglob("diagnosis.json")):
        data = read_json(path, warnings)
        common = dict(source_diagnosis=str(path), diagnosis_sha256=sha256(path),
                      checkpoint_sha256=data.get("checkpoint_sha256"),
                      data_sha256=data.get("data_sha256"), scale=data.get("n_states"),
                      temperature=data.get("temperature"),
                      scope="MATCHED_CHECKPOINT_INFERENCE_DIAGNOSIS_NOT_RETRAINING")
        for item in data.get("component_ablation", []):
            result.append(dict(common, kind="component_ablation", **item))
        for item in data.get("joint_compiler", []):
            result.append(dict(common, **item))
    return result


def csv_file(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def label(identity):
    return (f"{identity['scope']} · N={identity['scale']} seed={identity['seed']} "
            f"v={identity['version']}\n{identity['compilation']} / {identity['credit']} / "
            f"protection={identity['protection_timing']} · budget={identity['steps']} updates/stage")


def save_figure(figure, stem, out, plt, artifacts):
    for suffix in ("svg", "png"):
        path = out / (stem + "." + suffix)
        figure.savefig(path, dpi=180, bbox_inches="tight")
        artifacts.append(dict(path=str(path), sha256=sha256(path), format=suffix))
    plt.close(figure)


def make_plots(tables, runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    artifacts = []
    identities = {r["condition_id"]: r["identity"] for r in runs}
    partial_conditions = {r["condition_id"] for r in runs
                          if r["execution_status"] not in ("completed", "complete", "COMPLETED")}
    by_condition = defaultdict(list)
    for row in tables["curves"]:
        if row["task"] == row["training_stage"] and row["split"] == "iid":
            by_condition[row["condition_id"]].append(row)
    for condition, rows in by_condition.items():
        stages = sorted({r["training_stage"] for r in rows})
        models = sorted({r["model"] for r in rows})
        colors = {model: plt.get_cmap("tab10")(i % 10) for i, model in enumerate(models)}
        fig, axes = plt.subplots(len(stages), 2, figsize=(11, 3 * len(stages)), squeeze=False)
        for i, stage in enumerate(stages):
            for j, mode in enumerate(MODES):
                ax = axes[i, j]
                for model in models:
                    points = sorted([r for r in rows if r["model"] == model
                                     and r["training_stage"] == stage and r["mode"] == mode],
                                    key=lambda r: (r["run_id"], r["step"]))
                    by_run = defaultdict(list)
                    for p in points:
                        by_run[p["run_id"]].append(p)
                    for series in by_run.values():
                        ax.plot([r["step"] for r in series], [r["accuracy"] for r in series],
                                color=colors[model], label=model,
                                linestyle="-" if mode == "soft" else "--", linewidth=1.6)
                ax.set(xlabel="Updates within stage", ylabel="IID terminal accuracy", ylim=(-.02, 1.02),
                       title=f"Stage {stage + 1}: {mode} execution")
                temperatures = sorted({(r["step"], r["temperature"]) for r in rows
                                       if r["training_stage"] == stage and r["temperature"] is not None})
                if temperatures:
                    twin = ax.twinx()
                    twin.plot([x for x, _ in temperatures], [y for _, y in temperatures],
                              color=".55", linestyle=":", linewidth=1, label="logged temperature")
                    twin.set_ylabel("Logged temperature", color=".45")
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8, ncol=2)
        partial = "\nINTERRUPTED RUN: recorded points only" if condition in partial_conditions else ""
        fig.suptitle(label(identities[condition]) + partial + "\nMeasured curves; completion is not M0 passage", y=1.02)
        fig.tight_layout()
        save_figure(fig, "learning-curves-" + condition, out, plt, artifacts)

    matrices = defaultdict(list)
    for row in tables["task_by_stage"]:
        if row["split"] == "iid" and row["task_seen_by_stage"]:
            matrices[row["condition_id"]].append(row)
    for condition, rows in matrices.items():
        models = sorted({r["model"] for r in rows})
        height = 2.5 * len(models) + 1.5
        fig, axes = plt.subplots(len(models), 2, figsize=(9, height), squeeze=False)
        for i, model in enumerate(models):
            for j, mode in enumerate(MODES):
                selected = [r for r in rows if r["model"] == model and r["mode"] == mode]
                nstages = 1 + max(r["training_stage"] for r in rows)
                ntasks = 1 + max(r["task"] for r in rows)
                matrix = np.full((ntasks, nstages), np.nan)
                for r in selected:
                    matrix[r["task"], r["training_stage"]] = r["accuracy"]
                ax = axes[i, j]
                image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap="viridis", aspect="auto")
                for t in range(ntasks):
                    for s in range(nstages):
                        if np.isfinite(matrix[t, s]):
                            ax.text(s, t, f"{100 * matrix[t,s]:.1f}", ha="center", va="center",
                                    color="white" if matrix[t, s] < .55 else "black", fontsize=8)
                ax.set(title=f"{model} / {mode}", xlabel="Completed training stage", ylabel="Task",
                       xticks=range(nstages), xticklabels=range(1, nstages + 1),
                       yticks=range(ntasks), yticklabels=range(1, ntasks + 1))
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=.015, pad=.03, label="IID accuracy")
        fig.suptitle(label(identities[condition]) + "\nTask-by-stage retention; blanks are unlearned/unrecorded")
        fig.subplots_adjust(hspace=.7, top=1 - 1.5 / height, right=.88)
        save_figure(fig, "forgetting-heatmap-" + condition, out, plt, artifacts)
        frows = [r for r in tables["forgetting"] if r["condition_id"] == condition and r["earlier_best"] is not None]
        if frows:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), squeeze=False)
            for j, mode in enumerate(MODES):
                values = [r for r in frows if r["mode"] == mode]
                positions = range(len(values))
                axes[0, j].bar(positions, [100 * r["forgetting_signed"] for r in values])
                axes[0, j].set_xticks(list(positions), [f"{r['model']}\nT{r['task']+1}" for r in values], rotation=55, ha="right")
                axes[0, j].axhline(0, color="black", linewidth=.7)
                axes[0, j].set(title=mode, ylabel="max earlier − final IID (percentage points)")
            fig.suptitle(label(identities[condition]) + "\nSigned forgetting; low earlier accuracy is not acquired retention")
            fig.tight_layout()
            save_figure(fig, "forgetting-deltas-" + condition, out, plt, artifacts)

    # Compare first-stage compilation/protection ablations, holding source,
    # scale/seed/budget/credit/device fixed. Never pool them into multistage data.
    ablations = defaultdict(list)
    for row in tables["task_by_stage"]:
        if row["training_stage"] == 0 and row["task"] == 0 and row["stages"] == 1:
            key = tuple((k, row.get(k)) for k in ("version", "scale", "seed", "steps", "lr",
                                                 "batch", "latent_factor", "credit", "device", "scope"))
            ablations[key].append(row)
    for key, rows in ablations.items():
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), squeeze=False)
        arms = sorted({(r["model"], r["compilation"], r["protection_timing"]) for r in rows})
        for i, mode in enumerate(MODES):
            ax = axes[i, 0]
            width = .18
            for si, split in enumerate(SPLITS):
                values = []
                for arm in arms:
                    found = [r["accuracy"] for r in rows if (r["model"], r["compilation"], r["protection_timing"]) == arm
                             and r["split"] == split and r["mode"] == mode]
                    values.append(float(np.mean(found)) if found else np.nan)
                x = np.arange(len(arms)) + (si - 1.5) * width
                ax.bar(x, values, width=width, label=split)
            ax.set_xticks(range(len(arms)), [" / ".join(arm) for arm in arms], rotation=25, ha="right")
            ax.set(ylim=(0, 1.04), ylabel="Terminal accuracy", title=mode + " execution")
            ax.legend(ncol=4)
        fig.suptitle(f"FIRST-STAGE ABLATION ONLY · N={dict(key)['scale']} seed={dict(key)['seed']} · "
                     f"{dict(key)['steps']} update budget\nNo later acquisition/retention or M0 claim")
        fig.tight_layout()
        stem = hashlib.sha256(canonical(key).encode()).hexdigest()[:12]
        save_figure(fig, "firststage-ablation-" + stem, out, plt, artifacts)
    diagnostic_groups = defaultdict(list)
    for row in tables.get("checkpoint_diagnosis", []):
        diagnostic_groups[row["source_diagnosis"]].append(row)
    for source, rows in diagnostic_groups.items():
        names, values = [], []
        for row in rows:
            name = ("E/O/C/D=" + "".join(str(row.get(k)) for k in
                    ("hard_encoder", "hard_operations", "hard_calls", "hard_decoder"))
                    if row["kind"] == "component_ablation" else row["kind"])
            names.append(name)
            values.append(row["accuracy"])
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(range(len(names)), values)
        ax.set_xticks(range(len(names)), names, rotation=65, ha="right")
        ax.set(ylim=(0, 1.03), ylabel="Terminal accuracy",
               title=f"Matched checkpoint diagnosis · N={rows[0]['scale']} · tau={rows[0]['temperature']}\n"
                     "Inference component switches / compilation; no parameter update")
        fig.tight_layout()
        stem = hashlib.sha256(source.encode()).hexdigest()[:12]
        save_figure(fig, "checkpoint-ablation-" + stem, out, plt, artifacts)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.runs.resolve(strict=True)
    out = args.out.resolve() if args.out else root.parent / "architecture-plots-v1"
    if out.exists():
        parser.error("output must be a NEW directory; existing evidence will not be overwritten")
    learns = sorted({p.parent for p in root.rglob("records.jsonl")}
                    | {p.parent for p in root.rglob("summary.json") if p.parent.name == "learn"})
    if not learns:
        parser.error("no records.jsonl or learn/summary.json found")
    warnings = []
    runs = [read_run(learn, root, warnings) for learn in learns]
    tables = collect(runs)
    tables["checkpoint_diagnosis"] = diagnoses(root, warnings)
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in tables.items():
        csv_file(out / (name + ".csv"), rows)
    artifacts = make_plots(tables, runs, out)
    payload = dict(schema="topology-architecture-plot-results-v1", M0="NOT_ESTABLISHED_BY_PLOTS",
        plot_script=dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__))),
        inputs=[{k: v for k, v in r.items() if k not in ("rows", "summary")} for r in runs],
        tables=tables, warnings=warnings, figures=artifacts,
        limitations=["Configured update counts are budgets; interrupted curves contain only logged points",
            "CPU/debug, first-stage ablation and multistage experiments are separate strata",
            "Source-unrecorded versions are isolated per run, never imputed from current local code",
            "Soft execution and discrete execution differ; standard Transformer hard_mode may be soft_logits_argmax",
            "IID train/evaluation collisions are retained in datasets.csv; IID does not imply unseen input strings",
            "Forgetting uses completed stage ends; final task has no earlier-stage forgetting measurement",
            "Individual incorrect input/prediction pairs cannot be reconstructed from aggregate counts",
            "Memory fields are recorded tensor-byte categories, not complete runtime RAM or hardware energy",
            "Plots establish no M0 milestone, causal mechanism or baseline adequacy"])
    with (out / "results.json").open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(out), runs=len(runs), figures=len(artifacts),
                          rows={k: len(v) for k, v in tables.items()}, warnings=len(warnings))))


if __name__ == "__main__":
    main()
