"""Plot only exported actual Step3 evidence; no training or data generation."""
import argparse
import csv
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seconds",type=float,default=60.)
    args=parser.parse_args()
    if args.out.exists() or not 0<args.seconds<=90:
        parser.error("new output and<=90s internal bound required")
    began=time.monotonic()
    import sys
    dependency_root=Path("/opt/prometheus-v2-python")
    if dependency_root.is_dir():
        sys.path.insert(0,str(dependency_root))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    rows=list(csv.DictReader(args.curves.open()))
    for row in rows:
        row["task"]=int(row["task"]);row["relative_step"]=int(row["relative_step"])
        row["accuracy"]=float(row["accuracy"])
        row["total_gradient_updates"]=None if not row["total_gradient_updates"] else int(row["total_gradient_updates"])
    ledger=json.loads(args.ledger.read_text())
    args.out.mkdir(parents=True)
    manifest=[]
    def save(fig,name):
        if time.monotonic()-began>args.seconds:raise TimeoutError("bounded plotting deadline")
        fig.tight_layout(rect=(0,.025,1,.94))
        for extension in ("png","svg"):
            path=args.out/(name+"."+extension)
            fig.savefig(path,dpi=170,bbox_inches="tight");manifest.append(str(path))
        plt.close(fig)
    def series(selected,xfield):
        ordered=sorted(selected,key=lambda r:r[xfield])
        return [r[xfield] for r in ordered],[100*r["accuracy"] for r in ordered]
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),sharey=True)
    for task,ax in enumerate(axes):
        for arm in ("Prometheus","transformer","replay","adapter"):
            interface="neural_soft" if arm=="Prometheus" else "Transformer_neural"
            selected=[r for r in rows if r["study"]=="successful_chain" and r["arm"]==arm
                and r["task"]==task and r["split"]=="iid" and r["execution_interface"]==interface
                and r["total_gradient_updates"]>=(0,800,1800)[task]]
            x,y=series(selected,"total_gradient_updates")
            ax.plot(x,y,marker=".",markersize=4,label=arm)
        ax.set(title=f"Task {task}: actual neural IID",xlabel="Cumulative gradient updates",ylim=(-2,103))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Accuracy (%)");axes[2].legend(fontsize=8)
    fig.suptitle("N16 seed0: successful repaired lineage vs matched Transformer baselines (2800 updates)")
    fig.text(.01,.005,"Formation can change the program at the same update count; no new gradient step is implied by a vertical jump.",fontsize=8)
    save(fig,"matched-learning-lineage")
    fig,ax=plt.subplots(figsize=(8.6,4))
    for interface,label in [("neural_soft","Neural soft"),("native_graph","Native graph (logged repair nodes)"),
                            ("continuous_lift_cached","Actual lift-derived cached execution")]:
        selected=[r for r in rows if r["study"]=="successful_chain" and r["arm"]=="Prometheus"
            and r["task"]==2 and r["split"]=="iid" and r["execution_interface"]==interface
            and r["total_gradient_updates"]>=1800]
        x,y=series(selected,"total_gradient_updates")
        ax.plot(x,y,marker=".",label=label)
    ax.axvline(2600,color=".5",ls=":",label="History formation before extra SGD")
    ax.set(xlabel="Cumulative gradient updates",ylabel="Accuracy (%)",ylim=(-2,103))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("N16 task2: separate measured execution interfaces")
    fig.text(.01,.005,"Cached continuous execution includes learned maps obtained by actual cold lifts; this is not a fresh lift per request.",fontsize=8)
    save(fig,"execution-interfaces")
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,interface in zip(axes,("neural_soft","continuous_lift_cached")):
        for arm in ("continue_only","native_recompile","one_step","history_full","history_no_explicit_reuse"):
            selected=[r for r in rows if r["study"]=="acquisition" and r["arm"]==arm
                and r["task"]==2 and r["split"]=="iid" and r["execution_interface"]==interface]
            x,y=series(selected,"relative_step");ax.plot(x,y,marker="o",markersize=3,label=arm)
        ax.set(title=interface,xlabel="Additional SGD updates",ylim=(-2,103));ax.grid(alpha=.2)
    axes[0].set_ylabel("Task2 IID accuracy (%)");axes[1].legend(fontsize=7)
    fig.suptitle("Acquisition attribution: same source and200 updates, differing formation resources")
    fig.text(.01,.005,"History arms already reach100% at step0 after formation. Later SGD tests maintenance; it did not produce that initial jump.",fontsize=8)
    save(fig,"acquisition-attribution")
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,task in zip(axes,(0,2)):
        for arm in ("monodromy-guard-on","direct-guard-on","monodromy-guard-off","direct-guard-off"):
            selected=[r for r in rows if r["study"]=="monodromy" and r["arm"]==arm
                and r["task"]==task and r["split"]=="iid" and r["execution_interface"]=="actual_selected_backend"]
            x,y=series(selected,"relative_step")
            ax.plot(x,y,marker=".",ls="--" if arm.startswith("direct") else "-",label=arm)
        ax.set(title=f"Task {task}: actual selected backend",xlabel="Additional updates",ylim=(-2,103));ax.grid(alpha=.2)
    axes[0].set_ylabel("Accuracy (%)");axes[1].legend(fontsize=7)
    fig.suptitle("Monodromy vs direct execution × binding guard")
    fig.text(.01,.005,"At a fixed guard setting,200 raw proposal hashes and final parameters match across backends; overlapping curves are expected.",fontsize=8)
    save(fig,"execution-binding-attribution")
    names=["Prometheus","transformer","replay","adapter"]
    train=[ledger["total_native_synced_train_s"]]+[ledger["matched_baseline_chains"][k]["summed_train_s"] for k in names[1:]]
    params=[ledger["successful_native_chain"][-1]["parameters"]]+[ledger["matched_baseline_chains"][k]["phases"][-1]["parameters"] for k in names[1:]]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.7))
    axes[0].bar(names,train);axes[0].set(ylabel="Reported synchronized training seconds",title="Selected2800-update lineage")
    axes[1].bar(names,params);axes[1].set(ylabel="Actual final parameter scalars",title="Persistent model parameters")
    for ax in axes:ax.tick_params(axis="x",rotation=15)
    fig.suptitle("Recorded resource components; separate from total lifecycle cost")
    fig.text(.01,.005,"Training sums exclude separately recorded search/guards/evaluation/setup. Single resumed lineage; no controlled end-to-end speed claim.",fontsize=8)
    save(fig,"resource-components")
    (args.out/"manifest.json").write_text(json.dumps(dict(figures=manifest,source_curves=str(args.curves),source_ledger=str(args.ledger),
        scientific_scope="N16 seed0 observed evidence, M0 not passed",wall_seconds=time.monotonic()-began),indent=2)+"\n")


if __name__=="__main__":main()
