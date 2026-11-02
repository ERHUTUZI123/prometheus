"""ROOT-RUN bounded figures from frozen actual JSON/JSONL, no model code.

Only TRAIN/VALID acquisition/retention and mixed-batch raw training losses are
plotted. The canonical cost ledger remains the source; plotted-data is a figure
extract, never another ledger. Optional frozen LM graph costs are a DIFFERENT
task, with the same weights/P across backends within each arm. No confidence
bands or treatment of correlated steps/timing draws as independent seeds.
No Torch, neural checkpoint/archive, installation, network or subprocess.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import time
import traceback

ARMS = ("dynamic", "static", "gru", "transformer")
GRAPH_ARMS = ("dynamic", "static", "detached", "gru", "transformer")
LABELS = dict(dynamic="Dynamic Block", static="Static CALL Block", detached="Detached CALL Block",
              gru="GRU44", transformer="Transformer32")
COLORS = dict(dynamic="#0072B2", static="#D55E00", detached="#009E73", gru="#8A5AA6", transformer="#444444")
BACKENDS = ("original_eager", "tensor_eager", "cuda_graph")
BACKEND_LABELS = ("Original eager", "Tensor eager", "CUDA Graph")
BACKEND_COLORS = ("#666666", "#E69F00", "#009E73")
RESOURCE_KEYS = ("parameters", "added_token_embedding_parameters", "training_trainable_parameters",
    "training_forward_MAC_proxy", "training_seconds", "complete_launch_SSH_coordinator_seconds",
    "training_CPU_seconds", "training_CUDA_stream_seconds", "TRAIN_VALID_evaluation_seconds",
    "TRAIN_VALID_evaluation_forward_MAC_proxy", "restore_growth_seconds", "replay_maintenance_seconds",
    "checkpoint_write_seconds", "inference_total_measurement_seconds", "parameter_payload_bytes",
    "optimizer_tensor_bytes", "replay_array_bytes", "replay_metadata_bytes", "extra_position_buffer_bytes",
    "registered_buffer_bytes",
    "frozen_checkpoint_bytes", "CUDA_peak_allocated_bytes", "CPU_peak_RSS_bytes",
    "extra_new_training_labels", "replay_training_label_exposures", "exact_total_training_FLOPs", "energy",
    "deltaP_per_successfully_retained_new_capability")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def near(a, b, name):
    if not math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-9):
        raise ValueError("actual evidence arithmetic mismatch: " + name)


def records(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def load_bundle(folder):
    """Stdlib preflight is also run locally before any plotting execution."""
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("schema") != "frozen-stage4-development-figure-inputs-v1":
        raise ValueError("exact bounded figure input schema required")
    inventory = []
    total = 0
    if not 1 <= len(manifest["files"]) <= 64:
        raise ValueError("bounded source file count required")
    for leaf, expected in manifest["files"].items():
        rel = Path(leaf)
        file = folder / rel
        if (rel.is_absolute() or ".." in rel.parts or file.is_symlink()
                or file.suffix not in (".json", ".jsonl") or not file.is_file()
                or file.stat().st_size != expected["bytes"] or file.stat().st_size > 16 << 20
                or sha(file) != expected["sha256"]):
            raise ValueError("source byte/hash/type contract failed: " + leaf)
        total += file.stat().st_size
        inventory.append(dict(file=leaf, **expected))
    if total > 64 << 20:
        raise ValueError("total JSON source byte cap")
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("unlisted input; model/archive/source substitution forbidden")
    def read(leaf):
        return json.loads((folder / leaf).read_text())
    ledger = read(manifest["ledger"])
    if ledger.get("schema") != "joint-parameter-and-scoped-complete-cost-v1":
        raise ValueError("canonical scoped cost ledger required")
    ledger_rows = {r["arm"]: r for r in ledger["rows"]}
    if set(ledger_rows) != set(ARMS) or set(manifest["stage4"]) != set(ARMS):
        raise ValueError("fixed four arms required")
    stages = {}
    for arm in ARMS:
        entry = manifest["stage4"][arm]
        result = read(entry["result"])
        raw = list(records(folder / entry["raw"]))
        updates = [r for r in raw if r["kind"] == "actual_update"]
        curves = [dict(step=r["step"], metrics=r["metrics"]) for r in raw if r["kind"] == "TRAIN_VALID_curve"]
        configurations = [r for r in raw if r["kind"] == "configuration"]
        if (result.get("status") != "DECLARED_FOURTH_STAGE_DEVELOPMENT_COMPLETED"
                or result.get("arm") != arm or result.get("seed") != 0 or result.get("formal_Primary") is not False
                or [u["step"] for u in updates] != list(range(1, 601))
                or [c["step"] for c in curves] != list(range(0, 601, 100))
                or curves != result["curves"] or len(configurations) != 1):
            raise ValueError("actual fixed600 update/raw-curve result contract failed: " + arm)
        if configurations[0]["source_sha256"] != result["source_sha256"]:
            raise ValueError("actual start/end source mismatch")
        if any(not math.isfinite(u["terminal_loss"]) or u["current_training_examples"] != 48
               or u["replayed_training_examples"] != 16 for u in updates):
            raise ValueError("finite actual mixed48+16 training stream required")
        costs = result["costs"]
        for raw_key, cost_key in (("forward_MAC_proxy", "training_forward_MAC_proxy"),
                                  ("training_wall_seconds", "training_wall_seconds"),
                                  ("training_CPU_seconds", "training_CPU_seconds"),
                                  ("CUDA_stream_elapsed_seconds", "CUDA_stream_elapsed_seconds")):
            near(sum(u[raw_key] for u in updates), costs[cost_key], arm + ":" + cost_key)
        row = ledger_rows[arm]
        remote, local = read(entry["remote_receipt"]), read(entry["local_receipt"])
        if remote["returncode"] != 0 or local["returncode"] != 0:
            raise ValueError("actual completed original run receipts required")
        for field, actual_value in (("parameters", result["resources"]["parameters"]),
                ("added_token_embedding_parameters", result["growth"]["parameters_added"]),
                ("training_forward_MAC_proxy", costs["training_forward_MAC_proxy"]),
                ("training_seconds", costs["training_wall_seconds"]),
                ("complete_launch_SSH_coordinator_seconds", local["wall_seconds"]),
                ("optimizer_tensor_bytes", result["resources"]["optimizer_tensor_bytes"]),
                ("parameter_payload_bytes", result["resources"]["parameter_bytes"]),
                ("replay_array_bytes", result["resources"]["replay_bytes"]),
                ("frozen_checkpoint_bytes", result["checkpoint"]["bytes"])):
            near(row[field], actual_value, arm + ":ledger:" + field)
        for mode in ("soft", "native"):
            near(row["VALID_new_" + mode], curves[-1]["metrics"]["3"]["valid"][mode]["accuracy"], arm + ":VALID")
        stages[arm] = dict(run=entry["run"], parent=result["parent_run"], source_sha256=result["source_sha256"],
            parameters=row["parameters"], delta_token_parameters=row["added_token_embedding_parameters"],
            curves=curves, updates=[{k: u[k] for k in ("step", "terminal_loss", "forward_MAC_proxy",
                "training_wall_seconds", "training_CPU_seconds", "CUDA_stream_elapsed_seconds")} for u in updates],
            resource_row=dict({key: row.get(key) for key in RESOURCE_KEYS},
                              registered_buffer_bytes=result["resources"]["registered_buffer_bytes"]),
            registered_buffer_bytes=result["resources"]["registered_buffer_bytes"],
            paid_work=dict(updates=600, examples=38400, new_label_exposures=28800, replay_label_exposures=9600,
                           replay_rows=128),
            scope="current fourth phase only; parent costs are not pooled into this phase")
    graphs = {}
    if manifest.get("graphs"):
        if set(manifest["graphs"]) != set(GRAPH_ARMS):
            raise ValueError("optional graph panel uses all five fixed arms")
        for arm in GRAPH_ARMS:
            entry = manifest["graphs"][arm]
            result = read(entry["summary"])
            raw = [r for r in records(folder / entry["raw"]) if r["kind"] == "actual_latency_draw"]
            if (result.get("status") != "FROZEN_CUDA_GRAPH_INFERENCE_AUDIT_COMPLETED"
                    or result["arm"] != arm or not result["all_parameters_buffers_and_checkpoint_unchanged"]
                    or result["model_state_before_sha256"] != result["model_state_after_sha256"]
                    or result["training_updates"] != 0 or result["label_reads"] != 0):
                raise ValueError("actual frozen graph inference source contract failed")
            raw_map = {(r["backend"], r["batch"], r["context"], r["mode"], r["repeat"]): r for r in raw}
            if len(raw_map) != len(raw):
                raise ValueError("duplicate graph raw draws")
            total_draws = 0
            selected = []
            for case in result["cases"]:
                if case["total_parameters"] != result["resources"]["total_parameters"]:
                    raise ValueError("P changed between inference backends")
                for backend in BACKENDS:
                    draws = case[backend]["draws"]
                    if len(draws) != 12 or [r["repeat"] for r in draws] != list(range(12)):
                        raise ValueError("exact12 timing draws required")
                    for draw in draws:
                        key = (backend, case["batch"], case["context"], case["mode"], draw["repeat"])
                        actual_draw = raw_map.get(key)
                        if actual_draw is None or any(actual_draw.get(k) != v for k, v in draw.items()) or draw["wall_seconds"] <= 0:
                            raise ValueError("graph summary/raw draw disagreement")
                    total_draws += len(draws)
                    for name, fraction in (("p50_wall_seconds", .5), ("p95_wall_seconds", .95)):
                        near(case[backend][name], percentile([d["wall_seconds"] for d in draws], fraction), arm + ":latency")
                if case["batch"] == 1 and case["mode"] == "soft":
                    graph = case["cuda_graph"]
                    components = dict(warmup_and_construction=graph["side_stream_warmup_wall_seconds"] + graph["graph_construction_wall_seconds"],
                        capture=graph["capture_wall_seconds"], instantiation=graph["instantiation_wall_seconds"],
                        output_verification=graph["output_validation_wall_seconds"] + graph["postdraw_validation_wall_seconds"],
                        pool_cleanup=case["pool_cleanup_wall_seconds"])
                    near(sum(components.values()), graph["complete_graph_lifecycle_wall_seconds"], arm + ":graph lifecycle")
                    saving = case["tensor_eager"]["p50_wall_seconds"] - graph["p50_wall_seconds"]
                    expected_calls = math.ceil(graph["complete_graph_lifecycle_wall_seconds"] / saving) if saving > 0 else None
                    actual_calls = case["amortization"]["primary_complete_graph_lifecycle"]["tensor_eager"]["calls"]
                    if actual_calls != expected_calls:
                        raise ValueError("complete-lifecycle amortization mismatch")
                    selected.append(dict(context=case["context"], total_parameters=case["total_parameters"],
                        latency={b: {f: case[b][f] for f in ("p50_wall_seconds", "p95_wall_seconds")} for b in BACKENDS},
                        lifecycle_components=components, complete_lifecycle_seconds=graph["complete_graph_lifecycle_wall_seconds"],
                        break_even_calls_vs_tensor_eager=actual_calls,
                        allocator_capture_peak_delta_bytes=graph["memory_after_instantiation"]["cuda_peak_allocated_bytes"] - graph["memory_before_capture"]["cuda_allocated_bytes"]))
            if total_draws != len(raw_map) or sorted(c["context"] for c in selected) != [64, 128, 256]:
                raise ValueError("graph case/draw coverage mismatch")
            graphs[arm] = dict(run=entry["run"], parameters=result["resources"]["total_parameters"], cases=selected,
                               scope="different frozen char-LM task; same P/weights across inference backends within this arm")
    return dict(manifest=manifest, input_inventory=inventory, stage4=stages, graphs=graphs,
                ledger_scope=ledger["scope"], ledger_missing=ledger.get("missing"),
                ledger_sha256=sha(folder / manifest["ledger"]),
                notes=["Single seed0 DEVELOPMENT, not Primary; no seed uncertainty bands.",
                       "Inherited lineages differ between arms; dynamic/static are not paired-weight CALL ablations.",
                       "delta token P is vocabulary storage growth, not P per acquired capability.",
                       "Raw mixed-batch losses and every TRAIN/VALID curve point are retained; no TEST performance plotted.",
                       "Training, SSH, allocator and checkpoint scopes differ/overlap; no isolated speed/full-resource advantage claim."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=100)
    args = parser.parse_args()
    if args.out.exists() or not 1 <= args.max_seconds <= 100:
        raise ValueError("new output and bounded<=100s analysis required")
    args.out.mkdir(parents=True)
    began = time.perf_counter()
    def alarm(_sig, _frame):
        raise TimeoutError("bounded CPU figure generation100s")
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    artifacts = []
    try:
        data = load_bundle(args.inputs)
        os.environ["MPLCONFIGDIR"] = str(args.out / ".matplotlib-cache")
        dependency = Path("/opt/prometheus-analysis-python")
        if not dependency.is_dir():
            raise RuntimeError("existing /opt/prometheus-analysis-python is required; no installation/fallback")
        sys.path.insert(0, str(dependency))
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
            "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
            "figure.dpi": 110, "savefig.dpi": 200, "pdf.fonttype": 42,
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
            "grid.alpha": .18, "text.usetex": False})
        source_copy = args.out / "source-json"
        source_copy.mkdir()
        for item in data["input_inventory"]:
            dest = source_copy / item["file"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.inputs / item["file"], dest)
            if sha(dest) != item["sha256"]:
                raise RuntimeError("source JSON preservation failed")
        shutil.copyfile(args.inputs / "manifest.json", source_copy / "manifest.json")
        def save(fig, stem, note, bottom=.14, top=.91):
            footer = textwrap.fill(note, width=155)
            fig.text(.015, .014, footer, ha="left", va="bottom", fontsize=7.6, linespacing=1.3)
            fig.tight_layout(rect=(0, bottom, 1, top))
            for extension in ("pdf", "png"):
                file = args.out / (stem + "." + extension)
                fig.savefig(file, bbox_inches="tight", facecolor="white")
                artifacts.append(dict(file=file.name, sha256=sha(file), bytes=file.stat().st_size))
            plt.close(fig)
        def accuracy(ax):
            ax.set_ylim(0, 1.02); ax.set_yticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100"])
            ax.set_xlim(0, 600); ax.set_xticks([0, 200, 400, 600]); ax.set_xlabel("New fourth-phase updates")
            ax.set_ylabel("Accuracy (%)")
        stages = data["stage4"]
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.1))
        fig.suptitle("Fourth-task acquisition | seed0 development, not Primary", fontsize=13)
        handles = []
        for ax, arm in zip(axes.flat, ARMS):
            row = stages[arm]
            steps = [c["step"] for c in row["curves"]]
            for split, color in (("train", "#0072B2"), ("valid", "#D55E00")):
                modes = ("soft", "native") if arm in ("dynamic", "static") else ("soft",)
                for mode in modes:
                    ax.plot(steps, [c["metrics"]["3"][split][mode]["accuracy"] for c in row["curves"]],
                        color=color, ls="-" if mode == "soft" else "--", marker="o", ms=3,
                        label=split.upper() + (" soft" if mode == "soft" else " hard CALL"))
            ax.axhline(.125, color=".65", ls=":", lw=1)
            ax.set_title(LABELS[arm] + f" | P={row['parameters']:,}")
            accuracy(ax)
            if arm in ("gru", "transformer"):
                ax.text(.03, .94, "One neural interface", transform=ax.transAxes, va="top", fontsize=8)
            if not handles:
                handles = ax.get_legend_handles_labels()
        fig.legend(*handles, ncol=4, loc="upper center", bbox_to_anchor=(.5, .94), frameon=False)
        save(fig, "stage4_acquisition", "Same4096 TRAIN/512 VALID rows; 600 updates, each48 new +16 replay. Dotted line=balanced1/8 chance. Soft/hard CALL share weights; ordinary baselines have one interface. Lineage-specific inherited weights; no independent-seed error bands.", top=.87)
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.8))
        fig.suptitle("Old-task VALID retention during fourth-task learning | seed0 development", fontsize=12.5)
        for task, ax in enumerate(axes):
            for arm in ARMS:
                row = stages[arm]
                for mode in (("soft", "native") if arm in ("dynamic", "static") else ("soft",)):
                    ax.plot([c["step"] for c in row["curves"]],
                        [c["metrics"][str(task)]["valid"][mode]["accuracy"] for c in row["curves"]],
                        color=COLORS[arm], ls="-" if mode == "soft" else "--", marker="o", ms=3,
                        label=LABELS[arm] + (" hard CALL" if mode == "native" else ""))
            ax.set_title("Old task " + str(task)); accuracy(ax)
        h, labels = axes[0].get_legend_handles_labels()
        fig.legend(h, labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, .91), frameon=False, fontsize=8)
        save(fig, "stage4_old_valid_retention", "Each point is an actual256-row VALID evaluation; connected points are one trajectory, not replicates. Step0 starts differ because parent checkpoints differ. Soft/hard CALL are executions of the same learned weights; baseline native aliases are omitted.", bottom=.19, top=.78)
        fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.8))
        fig.suptitle("Every actual mixed-batch TRAIN loss | 600 updates per arm", fontsize=13)
        for ax, arm in zip(axes.flat, ARMS):
            updates = stages[arm]["updates"]
            values = [u["terminal_loss"] for u in updates]
            smoothed = []
            for value in values:
                smoothed.append(value if not smoothed else .95 * smoothed[-1] + .05 * value)
            ax.plot(range(1, 601), values, color=COLORS[arm], lw=.65, alpha=.35, label="All600 raw losses")
            ax.plot(range(1, 601), smoothed, color=COLORS[arm], lw=1.6, label="EMA0.95 (descriptive)")
            ax.set_title(LABELS[arm]); ax.set_xlabel("Actual optimizer update"); ax.set_ylabel("Mixed-batch terminal NLL")
            ax.set_ylim(bottom=0); ax.set_xlim(1, 600); ax.legend(frameon=False)
        save(fig, "stage4_raw_train_losses", "Loss includes48 current-task +16 old replay examples per update; it is not a pure new-task loss. Raw600 observations remain visible and saved. Fixed EMA0.95 is a display summary; no smoothing used for selection, inference or independent-replicate statistics.")
        x = np.arange(4); names = ["Dynamic", "Static", "GRU44", "TF32"]
        rows = [stages[a]["resource_row"] for a in ARMS]
        colors = [COLORS[a] for a in ARMS]
        fig, axes = plt.subplots(2, 3, figsize=(12.7, 7.4))
        fig.suptitle("Fourth-phase paid resources | canonical scoped cost ledger", fontsize=13)
        ax = axes[0, 0]
        ax.bar(x, [r["parameters"] for r in rows], color=colors)
        for i, r in enumerate(rows):
            ax.text(i, r["parameters"], f"{r['parameters']:,}\n+{r['added_token_embedding_parameters']} token P", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(r["parameters"] for r in rows) * 1.23); ax.set_title("All model P + vocabulary-row growth"); ax.set_ylabel("Parameter count")
        axes[0, 1].bar(x, [r["training_forward_MAC_proxy"] / 1e9 for r in rows], color=colors)
        axes[0, 1].set_title("Training forward MAC proxy"); axes[0, 1].set_ylabel("G MAC (backward excluded)")
        ax = axes[0, 2]
        ax.bar(x - .18, [r["training_seconds"] for r in rows], width=.36, color="#0072B2", label="Training subset")
        ax.bar(x + .18, [r["complete_launch_SSH_coordinator_seconds"] for r in rows], width=.36, color="#D55E00", label="Complete SSH coordinator")
        ax.set_yscale("log"); ax.set_title("Recorded wall time (nested scopes)"); ax.set_ylabel("Seconds (log)"); ax.legend(frameon=False, fontsize=7)
        ax = axes[1, 0]; bottom = np.zeros(4)
        for field, label, color in (("parameter_payload_bytes", "Weights", "#0072B2"), ("optimizer_tensor_bytes", "Adam", "#E69F00"),
                                    ("replay_array_bytes", "Replay arrays", "#009E73"), ("replay_metadata_bytes", "Replay metadata", "#CC79A7"),
                                    ("registered_buffer_bytes", "Buffers", "#999999")):
            values = np.array([r[field] for r in rows]) / 1024
            ax.bar(x, values, bottom=bottom, color=color, label=label); bottom += values
        ax.set_title("Separate live payloads"); ax.set_ylabel("KiB"); ax.legend(frameon=False, fontsize=6.8)
        axes[1, 1].bar(x, [r["frozen_checkpoint_bytes"] / 1024 for r in rows], color=colors)
        axes[1, 1].set_title("Saved checkpoint (overlaps payload)"); axes[1, 1].set_ylabel("KiB on disk")
        axes[1, 2].bar(x, [r["CUDA_peak_allocated_bytes"] / 2**20 for r in rows], color=colors)
        axes[1, 2].set_title("Observed cumulative CUDA allocated peak"); axes[1, 2].set_ylabel("MiB; includes evaluations/audits")
        for ax in axes.flat:
            ax.set_xticks(x, names); ax.set_axisbelow(True)
        save(fig, "stage4_paid_resources", "All arms:600 updates,38400 label exposures=28800 new+9600 replay; Replay128 arrays=55296B. Added token P is NOT P per acquired/retained capability. SSH time includes training/evaluation/launch work; do not add nested fields or infer isolated throughput. Exact full training FLOPs/energy and capability-normalized delta P remain unknown; full scopes/CPU RSS/buffers retained in plotted-data and canonical ledger.", bottom=.16)
        if data["graphs"]:
            graphs = data["graphs"]
            fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
            fig.suptitle("Frozen character-LM inference | identical weights/P within each arm", fontsize=13)
            for ax, arm in zip(axes.flat, GRAPH_ARMS):
                graph = graphs[arm]
                ordered = sorted(graph["cases"], key=lambda c: c["context"])
                for backend, label, color in zip(BACKENDS, BACKEND_LABELS, BACKEND_COLORS):
                    median = np.array([c["latency"][backend]["p50_wall_seconds"] for c in ordered]) * 1000
                    p95 = np.array([c["latency"][backend]["p95_wall_seconds"] for c in ordered]) * 1000
                    ax.errorbar([64, 128, 256], median, yerr=np.array([np.zeros(3), p95 - median]),
                                color=color, marker="o", capsize=3, lw=1.3, label=label)
                label = "Transformer36" if arm == "transformer" else LABELS[arm]
                ax.set_title(label + f" | P={graph['parameters']:,}")
                ax.set_yscale("log"); ax.set_xticks([64, 128, 256]); ax.set_xlabel("TRAIN-token tiled window length")
                ax.set_ylabel("Resident B1 soft latency (ms, log)")
            axes[1, 2].axis("off")
            handles = [Line2D([0], [0], color=c, marker="o", label=l) for c, l in zip(BACKEND_COLORS, BACKEND_LABELS)]
            axes[1, 2].legend(handles=handles, loc="upper left", frameon=False)
            axes[1, 2].text(.02, .48, "12 timing draws per point\nLine = p50; upper whisker = p95\nWithin-run timing spread, not seed CI\nDifferent task from stage4 training", transform=axes[1, 2].transAxes, fontsize=9, va="top", linespacing=1.5)
            save(fig, "frozen_lm_graph_resident_latency", "No training or labels in this frozen audit. B1, soft, first originalTRAIN windows tiled to64/128/256; outputs verified. Original-to-tensor removes checks AND all-active indexing; tensor-to-graph preserves the same tensor function. Resident-input only; excludes H2D/serving/cached sequential generation and makes no training-speed claim.")
            cases = [next(c for c in graphs[a]["cases"] if c["context"] == 256) for a in GRAPH_ARMS]
            xx = np.arange(5); nn = ["Dynamic", "Static", "Detached", "GRU44", "TF36"]
            fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.8))
            fig.suptitle("Paid CUDA Graph lifecycle | B1, L256, soft, frozen LM weights", fontsize=13)
            bottom = np.zeros(5)
            for field, label, color in (("warmup_and_construction", "Warmup + construction", "#999999"),
                    ("capture", "Capture", "#0072B2"), ("instantiation", "Instantiate", "#E69F00"),
                    ("output_verification", "Output verification", "#009E73"), ("pool_cleanup", "Pool cleanup", "#CC79A7")):
                v = np.array([c["lifecycle_components"][field] for c in cases]); axes[0].bar(xx, v, bottom=bottom, color=color, label=label); bottom += v
            axes[0].set_title("Complete measured graph lifecycle"); axes[0].set_ylabel("One-time seconds"); axes[0].legend(frameon=False, fontsize=6.4)
            for i, c in enumerate(cases):
                calls = c["break_even_calls_vs_tensor_eager"]
                if calls is None:
                    axes[1].text(i, 1, "No saving", rotation=90, ha="center", va="bottom", fontsize=8)
                else:
                    axes[1].bar(i, calls, color=COLORS[GRAPH_ARMS[i]]); axes[1].text(i, calls, str(calls), ha="center", va="bottom", fontsize=8)
            axes[1].set_yscale("log"); axes[1].set_title("Break-even vs tensor eager p50"); axes[1].set_ylabel("Calls (log); includes teardown/verification")
            peak = [c["allocator_capture_peak_delta_bytes"] / 2**20 for c in cases]
            if any(v < 0 for v in peak):
                raise ValueError("invalid pre-capture/peak allocator scope")
            axes[2].bar(xx, peak, color=[COLORS[a] for a in GRAPH_ARMS]); axes[2].set_title("Capture allocator peak above baseline"); axes[2].set_ylabel("MiB; not exact executable/driver bytes")
            for ax in axes:
                ax.set_xticks(xx, nn, rotation=25, ha="right")
            save(fig, "frozen_lm_graph_paid_lifecycle", "Complete lifecycle=side-stream warmup+construction+capture+instantiation+first-output verification+postdraw verification+pool cleanup. Break-even uses complete lifecycle/(tensor-eager p50 - graph p50); undefined if no saving. Shared input/load checks are charged separately in original audit. Private pool per shape/case; no free persistent multi-shape cache or FLOP reduction is claimed.", bottom=.23)
        # Derived figure data never replaces the source ledger or raw evidence.
        plotted = dict(schema="stage4-development-figure-extract-v1", canonical_ledger_sha256=data["ledger_sha256"],
            stage4=stages, graphs=data["graphs"], notes=data["notes"], ledger_missing=data["ledger_missing"],
            figure_extract_only_not_a_cost_ledger=True, no_TEST_performance_plotted=True,
            single_seed=True, formal_Primary=False, raw_loss_EMA_display_only=.95)
        with (args.out / "plotted-data.json").open("x") as stream:
            json.dump(plotted, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        with (args.out / "plotted-resource-fields.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("arm",) + RESOURCE_KEYS)
            writer.writeheader(); writer.writerows(dict(arm=a, **stages[a]["resource_row"]) for a in ARMS)
        outputs = [dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size) for p in sorted(args.out.iterdir()) if p.is_file()]
        result = dict(status="FROZEN_STAGE4_DEVELOPMENT_FIGURES_COMPLETED", driver_sha256=sha(Path(__file__)),
            inputs_manifest_sha256=sha(args.inputs / "manifest.json"), input_files=data["input_inventory"],
            outputs=outputs, figures=artifacts, actual_new_updates_by_arm={a:600 for a in ARMS},
            actual_TRAIN_VALID_points_by_arm={a:7 for a in ARMS}, graph_cost_panel_included=bool(data["graphs"]),
            numpy_version=np.__version__, matplotlib_version=matplotlib.__version__, wall_seconds=time.perf_counter()-began,
            visual_QA="Root will inspect actual generated PNG/PDF; this JSON is not a visual pass",
            formal_Primary=False, training_updates_executed=0, model_inference_executed=0,
            notes=data["notes"])
        with (args.out / "provenance-manifest.json").open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
        print(json.dumps(dict(status=result["status"], figure_files=len(artifacts), out=str(args.out), wall_seconds=result["wall_seconds"])))
    except BaseException as error:
        (args.out / "exception.txt").write_text(traceback.format_exc())
        with (args.out / "failure.json").open("x") as stream:
            json.dump(dict(status="FAILED_OR_BOUNDED_PARTIAL_PLOT", error_type=type(error).__name__, error=str(error),
                completed_figure_files=artifacts, wall_seconds=time.perf_counter()-began), stream, indent=2); stream.write("\n")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
