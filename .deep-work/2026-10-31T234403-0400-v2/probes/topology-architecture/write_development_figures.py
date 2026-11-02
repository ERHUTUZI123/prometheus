"""ROOT-RUN fixed exploratory figures from frozen JSON/JSONL bytes only.

NumPy/Matplotlib analysis; no Torch, network, subprocess, model loading, training,
selection or source mutation. --inputs holds manifest.json and RUN/ or runs/RUN/
directories. Every supplied JSON/JSONL, including failures, is retained by bytes
with hashes. Curves use exact declared runs, not best/latest outcomes. Single
trajectories have no confidence bands. Root supplies external CPU isolation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import textwrap
import time
import traceback


COLORS = {"dynamic":"#2166ac", "static":"#e08214", "detached":"#238b45", "gru":"#8073ac", "transformer":"#b2182b", "cnn":"#8c510a"}
CONTEXT = {
    "dynamic": ("Continuous dynamic", "parameter-context-n8-seed0-dynamic-stage1-v1-soft_only-continuous", "parameter-context-n8-seed0-dynamic-stage3-v2-soft_only-continuous-resume"),
    "static": ("Continuous static", "parameter-context-n8-seed0-static-stage1-v1-soft_only-continuous", "parameter-context-n8-seed0-static-stage3-v2-soft_only-continuous-resume"),
    "gru": ("GRU44", "parameter-context-n8-seed0-gru-stage1-v1-width44", "parameter-context-n8-seed0-gru-stage3-v2-width44-resume"),
    "transformer": ("Transformer32", "parameter-context-n8-seed0-transformer-stage1-v1-width32", "parameter-context-n8-seed0-transformer-stage3-v2-width32-resume"),
}
LM = {
    "dynamic": ("Continuous dynamic", "parameter-char-block-300-v1-continuous"),
    "static": ("Continuous static", "parameter-char-block-300-v1-continuous-static"),
    "gru": ("GRU44", "parameter-char-gru-300-v1-width44"),
    "transformer": ("Transformer36", "parameter-char-transformer-300-v1-width36"),
}
REAL = (("dynamic","block_dynamic","Dynamic", "v1"), ("static","block_static","Static","v1"),
        ("gru","gru","GRU64","v1"), ("transformer","transformer","Transformer64","v1"),
        ("cnn","cnn","CNN","v1"), ("detached","block_detached","Detached (v2)","v2"))
SPLITS = (("test_iid","IID"),("test_unseen","Unseen combinations"),("test_long","Long"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-seconds",type=float,default=30);args=p.parse_args()
    if not 1<=args.max_seconds<=30:p.error("fixed analysis bound <=30seconds")
    args.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def timeout(_s,_f):raise TimeoutError("development figure analysis30second bound")
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    inventory=[];outputs=[];notes=[];used=set();series=[];bars=[];resources=[];execution=[]
    try:
        if not (args.inputs/"manifest.json").is_file():raise ValueError("Root-supplied inputs/manifest.json required")
        if args.inputs.resolve()==args.out.resolve() or args.out.resolve().is_relative_to(args.inputs.resolve()):raise ValueError("output must be outside read-only input tree")
        source_root=args.out/"source-bytes";source_root.mkdir();total=0
        for source in sorted(args.inputs.rglob("*")):
            if source.is_symlink():raise ValueError("input symlinks are outside this frozen analysis contract")
            if not source.is_file() or source.suffix not in (".json",".jsonl"):continue
            if source.stat().st_size>64<<20:raise ValueError("individual JSON/source cap")
            total+=source.stat().st_size
            if total>256<<20 or len(inventory)>=1000:raise ValueError("total bounded evidence input cap")
            rel=source.relative_to(args.inputs);destination=source_root/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,destination)
            digest=sha(source)
            if sha(destination)!=digest:raise RuntimeError("source byte preservation failed")
            inventory.append(dict(file=str(rel),sha256=digest,bytes=source.stat().st_size,copied_to=str(destination.relative_to(args.out))))
        analysis_python=Path("/opt/prometheus-analysis-python")
        if analysis_python.is_dir():sys.path.insert(0,str(analysis_python))
        os.environ["MPLCONFIGDIR"]=str(args.out/".matplotlib-cache")
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":9.5,
            "axes.labelsize":8.5,"legend.fontsize":7.5,"figure.dpi":110,"savefig.dpi":160,
            "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none",
            "text.usetex":False,"axes.grid":True,"grid.alpha":.18})
        root=args.inputs/"runs" if (args.inputs/"runs").is_dir() else args.inputs
        def read(run,relative):
            path=root/run/relative
            if not path.is_file():return None
            used.add(str(path.relative_to(args.inputs)))
            return json.loads(path.read_text())
        def save(fig,name,footnote):
            wrapped=textwrap.fill(footnote,width=160)
            fig.text(.01,.012,wrapped,fontsize=7,ha="left",va="bottom")
            fig.tight_layout(rect=(0,.035+.018*len(wrapped.splitlines()),1,.95))
            for extension in ("png","pdf","svg"):
                path=args.out/(name+"."+extension);fig.savefig(path,bbox_inches="tight")
                outputs.append(dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size))
            plt.close(fig)
        def bounded_accuracy(axis):axis.set_ylim(0,1.02);axis.set_yticks([0,.25,.5,.75,1])
        def unprovided(axis,text):axis.text(.5,.5,text,transform=axis.transAxes,ha="center",va="center",fontsize=8,color=".45")
        def numeric(value):return value is not None and np.isfinite(value)
        # Preserve observed execution status and partial curves for EVERY input
        # run, including earlier failed resume revisions not selected as repairs.
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not folder.name.startswith("parameter-"):continue
            found=[];actual_updates=0
            for path in sorted(folder.rglob("*.json")):
                if path.name in ("failure.json","summary.json","result.json","coordinator-exit.json"):
                    value=json.loads(path.read_text());found.append(dict(file=str(path.relative_to(args.inputs)),status=value.get("status"),
                        error=value.get("error",value.get("reason")),returncode=value.get("returncode")))
            for path in sorted(folder.rglob("records.jsonl")):
                for line in path.read_text().splitlines():
                    value=json.loads(line)
                    if value.get("kind") in ("actual_update","actual_LM_update"):actual_updates+=1
                    if value.get("kind") in ("run_failed","actual_LM_failure","failure"):
                        found.append(dict(file=str(path.relative_to(args.inputs)),status="raw_failure",error=value.get("error",value.get("reason"))))
                    # All partial TRAIN/VALID points survive as machine-readable
                    # plot data, rather than silently truncating failed versions.
                    if value.get("kind") in ("TRAIN_VALID_curve","VALID_curve"):
                        series.append(dict(group="all-provided-raw-curves",run=folder.name,event=value))
            execution.append(dict(run=folder.name,actual_update_records=actual_updates,observed_records=found))
        # 1. Exact original phase0 plus repaired resume-v2: task0 retention.
        context_data={};fig,axes=plt.subplots(1,2,figsize=(10,3.65));fig.suptitle("Contextual computation: old-task retention during new-task learning (N=8, seed 0)")
        for key,(label,first,resumed) in CONTEXT.items():
            documents=[read(first,"learn/result.json"),read(resumed,"learn/result.json")]
            reports=[]
            for run,document in zip((first,resumed),documents):
                if document:
                    for report in document.get("training_reports",[]):reports.append((run,report))
                else:notes.append(dict(group="context",run=run,status="not_provided; no zero result inferred"))
            context_data[key]=(label,reports,documents[-1])
            for run,report in reports:
                stage=int(report["stage"])
                for interface,axis in zip(("soft","native"),axes):
                    xs=[];ys=[]
                    for row in report["curves"]:
                        value=row["metrics"]["0"]["valid"][interface]["accuracy"]
                        x=600*stage+int(row["step"]);xs.append(x);ys.append(value)
                        series.append(dict(group="context-task0",run=run,arm=key,stage=stage,step=row["step"],cumulative_updates=x,split="VALID",interface=interface,accuracy=value))
                    axis.plot(xs,ys,color=COLORS[key],marker="o",markersize=2.7,lw=1.35,label=label if stage==0 else None)
                    axis.scatter(xs[0],ys[0],s=27,facecolors="none",edgecolors=COLORS[key],zorder=3)
        for axis,interface in zip(axes,("Soft execution","Native execution")):
            bounded_accuracy(axis);axis.set_xlim(0,1800);axis.set_xlabel("Cumulative updates (600 per stage)");axis.set_ylabel("Old task 0 VALID accuracy");axis.set_title(interface)
            for boundary in (600,1200):axis.axvline(boundary,color=".6",ls=":",lw=.8)
        handles=[Line2D([0],[0],color=COLORS[k],label=v[0]) for k,v in CONTEXT.items()];axes[0].legend(handles=handles,loc="best")
        save(fig,"context_task0_retention","Single trajectories; no error bands. Stage curves are separate segments. Open circles retain each stage-step0, including duplicate boundary x values. Old-task data are evaluation-only in later stages.")
        fig,axes=plt.subplots(2,3,figsize=(10.5,5.65),sharey=True);fig.suptitle("Contextual computation after stage 2: all three tasks")
        for row,interface in enumerate(("soft","native")):
            for col,(split,title) in enumerate(SPLITS):
                axis=axes[row,col];width=.19;x=np.arange(3)
                for offset,(key,(label,reports,document)) in enumerate(context_data.items()):
                    final=next((r for r in document.get("frozen_test_reports",[]) if r["learned_through_stage"]==2),None) if document else None
                    if final is None:continue
                    ys=[final["all_seen_test_metrics"][str(task)][split][interface]["accuracy"] for task in range(3)]
                    axis.bar(x+(offset-1.5)*width,ys,width,color=COLORS[key],label=label)
                    for task,value in enumerate(ys):bars.append(dict(group="context-final",arm=key,task=task,split=split,interface=interface,accuracy=value))
                bounded_accuracy(axis);axis.set_xticks(x,["Task 0","Task 1","Task 2"]);axis.set_title(title+" / "+interface)
                if col==0:axis.set_ylabel("TEST accuracy")
        axes[0,0].legend(fontsize=7)
        save(fig,"context_final_three_tasks","Exact phase0 + resume-v2 lineages; 1,800 paid updates per model. Terminal-task accuracy, not proof of general computation. Earlier failed revisions remain in execution inventory and preserved source bytes.")
        # 2. Matched-parameter LM: use actual dedicated serial inference audits.
        fig,axes=plt.subplots(2,2,figsize=(10.5,6.1));fig.suptitle("Matched-size character development: probability and frozen inference")
        lm_rows=[]
        for key,(label,parent) in LM.items():
            training=read(parent,"learn/summary.json");audit=read(parent+"-inference-audit-v1","audit/summary.json")
            if not audit:
                notes.append(dict(group="LM",run=parent+"-inference-audit-v1",status="not_provided"));continue
            parameter_count=audit["resources"]["parameters"]
            for mode in (("soft","native") if key in ("dynamic","static") else ("soft",)):
                x=[];ppl=[];acc=[]
                for split,extent in (("test_iid",32),("test_unseen",64),("test_long",128)):
                    record=audit["results"][split][mode];x.append(extent);ppl.append(record["sampled_character_PPL"]);acc.append(record["accuracy"])
                    lm_rows.append(dict(group="LM-probability",arm=key,label=label,parameters=parameter_count,mode=mode,context=extent,
                        sampled_character_PPL=record["sampled_character_PPL"],next_character_NLL=record["next_character_NLL"],accuracy=record["accuracy"],source_run=parent+"-inference-audit-v1"))
                linestyle="--" if mode=="native" else "-";name=label+f" ({parameter_count:,} P)"+(" hard CALL" if mode=="native" else " soft")
                axes[0,0].plot(x,ppl,color=COLORS[key],ls=linestyle,marker="o",markersize=3,label=name)
                axes[0,1].plot(x,acc,color=COLORS[key],ls=linestyle,marker="o",markersize=3,label=name)
                for size,axis in ((1,axes[1,0]),(64,axes[1,1])):
                    entries=sorted([r for r in audit["latency"] if r["mode"]==mode and r["batch_size"]==size],key=lambda r:r["context"])
                    axis.plot([r["context"] for r in entries],[1000*r["p50_seconds"] for r in entries],color=COLORS[key],ls=linestyle,marker="o",markersize=3)
                    for r in entries:
                        axis.vlines(r["context"],1000*r["p50_seconds"],1000*r["p95_seconds"],color=COLORS[key],lw=2,alpha=.55)
                        lm_rows.append(dict(group="LM-latency",arm=key,mode=mode,batch_size=size,context=r["context"],p50_seconds=r["p50_seconds"],p95_seconds=r["p95_seconds"],source_run=parent+"-inference-audit-v1"))
            if not audit.get("frozen_weights_unchanged"):raise ValueError("LM inference audit did not establish unchanged weights")
            if training:resources.append(dict(group="LM-training",arm=key,run=parent,parameters=training["resources"]["parameters"],training_wall_seconds=training["costs"]["training_wall_seconds"],training_forward_MAC_proxy=training["costs"]["training_forward_MAC_proxy"]))
        axes[0,0].set_ylabel("Sampled next-character PPL");axes[0,0].set_yscale("log");axes[0,0].legend(loc="best",fontsize=6.5)
        axes[0,0].set_title("Normalized probability: soft vs hard CALL");axes[0,1].set_title("Next-character accuracy");axes[0,1].set_ylabel("Accuracy");bounded_accuracy(axes[0,1])
        for size,axis in ((1,axes[1,0]),(64,axes[1,1])):axis.set_title(f"Resident-input latency / batch {size}");axis.set_ylabel("Forward wall time (ms)");axis.set_yscale("log")
        for axis in axes.flat:axis.set_xticks([32,64,128]);axis.set_xlabel("Prefix context length")
        save(fig,"lm_probability_and_inference","Same 512 paired TEST positions across contexts; sampled-prefix character PPL, not full-corpus or word PPL. Hard CALL keeps continuous readout. Lines: latency p50; vertical segments: within-run p50–p95, not seed uncertainty. No optimizer updates during audit.")
        series.extend(lm_rows)
        # 3. Official AUDIO/VISION static development; all original5 + new v2.
        real_data={};fig,axes=plt.subplots(2,2,figsize=(11,6.4));fig.suptitle("Real-data development: fixed 300-update trajectories and frozen TEST")
        for row,dataset in enumerate(("speech","cifar")):
            real_data[dataset]={}
            for order,(key,arm,label,revision) in enumerate(REAL):
                run=f"parameter-real-{dataset}-{arm}-300-{revision}";document=read(run,"learn/summary.json")
                if document is None:
                    notes.append(dict(group="real",run=run,status="not_provided; unavailable is not zeroaccuracy"));continue
                real_data[dataset][key]=(label,run,document)
                for split,style in (("TRAIN",":"),("VALID","-")):
                    xx=[r["step"] for r in document["curves"]];yy=[r[split]["accuracy"] for r in document["curves"]]
                    axes[row,0].plot(xx,yy,color=COLORS[key],ls=style,marker="o",markersize=2.3,lw=1.25,label=label if split=="VALID" else None)
                    for step,value in zip(xx,yy):series.append(dict(group="real-learning",run=run,dataset=dataset,arm=key,split=split,step=step,accuracy=value,revision=revision))
                final=document["final"]["TEST"];position=order
                axes[row,1].bar(position-.14,final["accuracy"],width=.28,color=COLORS[key])
                hard=final.get("native_hard_CALL_accuracy")
                if hard is not None:axes[row,1].bar(position+.14,hard,width=.28,color=COLORS[key],alpha=.55,hatch="//",edgecolor=COLORS[key])
                bars.append(dict(group="real-final",dataset=dataset,arm=key,revision=revision,run=run,soft_accuracy=final["accuracy"],native_hard_CALL_accuracy=hard,rows=final["rows"]))
                resources.append(dict(group="real-resources",dataset=dataset,arm=key,revision=revision,run=run,
                    parameters=document["resources"]["parameters"],training_wall_seconds=document["costs"]["training_wall_seconds"],
                    training_forward_MAC_proxy=document["costs"]["training_forward_MAC_proxy"],resources=document["resources"],costs=document["costs"]))
            axes[row,0].set_title(("Speech Commands v2" if dataset=="speech" else "CIFAR-100")+" / TRAIN dotted, VALID solid")
            axes[row,0].set_xlim(0,300);axes[row,0].set_xlabel("Actual optimizer updates");axes[row,0].set_ylabel("Accuracy");bounded_accuracy(axes[row,0]);axes[row,0].legend(ncol=2,fontsize=6.7)
            axes[row,1].set_title("Frozen TEST / solid soft; hatched hard CALL");axes[row,1].set_xticks(range(len(REAL)),[r[2] for r in REAL],rotation=25,ha="right");axes[row,1].set_ylabel("Accuracy");bounded_accuracy(axes[row,1])
            missing=[r[2] for r in REAL if r[0] not in real_data[dataset]]
            if missing:axes[row,1].text(.01,.98,"Not provided: "+", ".join(missing),transform=axes[row,1].transAxes,ha="left",va="top",fontsize=6.7,color=".4")
        save(fig,"real_data_learning_and_accuracy","Static first10-class diagnostics only; public35/100-class output heads included. Original five methods retained; detached is a separate v2 gradient ablation. Same source examples/features/batches per dataset; no TEST-based selection or confidence bands.")
        fig,axes=plt.subplots(2,3,figsize=(11,5.7));fig.suptitle("Real-data resources: actual model size and charged work")
        for row,dataset in enumerate(("speech","cifar")):
            for col,(field,title,scale) in enumerate((("parameters","All model parameters",1),("training_wall_seconds","Recorded train wall (s; concurrent)",1),("training_forward_MAC_proxy","Training forward MAC proxy (G)",1e9))):
                axis=axes[row,col]
                for order,(key,arm,label,revision) in enumerate(REAL):
                    if key not in real_data[dataset]:continue
                    _,run,document=real_data[dataset][key]
                    value=document["resources"][field] if field=="parameters" else document["costs"][field]
                    axis.bar(order,value/scale,color=COLORS[key])
                    if field=="parameters":axis.text(order,value/scale,f"{int(value):,}",ha="center",va="bottom",fontsize=6,rotation=25)
                axis.set_ylim(bottom=0);axis.set_title(title);axis.set_xticks(range(len(REAL)),[r[2] for r in REAL],rotation=30,ha="right")
                if col==0:axis.set_ylabel("Speech" if dataset=="speech" else "CIFAR")
        save(fig,"real_data_resources","P counts the full network including learned adapters and full public heads. Feature/optimizer/checkpoint storage remains in plotted-data JSON. Training wall includes actual concurrent conditions: do not interpret these bars as exclusive-throughput rankings. MAC proxies exclude backward/optimizer/nonlinearities.")
        plotted=dict(context_runs=CONTEXT,LM_runs=LM,real_configuration=REAL,series=series,bars=bars,resources=resources,notes=notes)
        with (args.out/"plotted-data.json").open("x") as stream:json.dump(plotted,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        # Flat scalar tables preserve exact values; nested raw curves remain JSON.
        for filename,rows in (("performance-bars.csv",bars),("resources.csv",[{k:v for k,v in r.items() if not isinstance(v,(dict,list,tuple))} for r in resources]),("learning-curves.csv",[r for r in series if "event" not in r])):
            keys=sorted({k for r in rows for k in r})
            with (args.out/filename).open("x",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
        with (args.out/"execution-and-failures.json").open("x") as stream:json.dump(dict(runs=execution,notes=notes),stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        generated_artifacts=[dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size) for path in sorted(args.out.iterdir()) if path.is_file()]
        manifest=dict(status="completed_fixed_development_figures",analysis_driver_sha256=sha(Path(__file__)),
            inputs_manifest_sha256=sha(args.inputs/"manifest.json"),input_manifest_policy="Root supplied opaque provenance retained verbatim; all actual JSON/JSONL bytes independentlyhashed/copied",
            input_files=inventory,files_used_for_selected_figures=sorted(used),figures=outputs,generated_artifacts=generated_artifacts,
            exact_configurations=dict(context=CONTEXT,LM=LM,real=REAL),numpy_version=np.__version__,matplotlib_version=matplotlib.__version__,
            single_seed=True,error_bands="none; latency ranges are within-run timing p50–p95 only",
            source_scope="all provided failures/partialcurves retained; fixed declared repairs not selected by score",
            boundary_policy="stage0end and nextstage0 points kept separately at same cumulativeupdate; no invented interpolation",
            formal_Primary=False,no_significance_or_advantage_claim=True,wall_seconds=time.perf_counter()-started)
        with (args.out/"provenance-manifest.json").open("x") as stream:json.dump(manifest,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
        print(json.dumps(dict(status=manifest["status"],figures=len(outputs),input_files=len(inventory),out=str(args.out),wall_seconds=manifest["wall_seconds"])))
    except BaseException as error:
        (args.out/"exception.txt").write_text(traceback.format_exc())
        with (args.out/"failure.json").open("x") as stream:json.dump(dict(status="failed_or_bounded_partial_analysis",error=f"{type(error).__name__}: {error}",input_files=inventory,completed_figure_files=outputs,wall_seconds=time.perf_counter()-started),stream,indent=2);stream.write("\n")
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=="__main__":main()
