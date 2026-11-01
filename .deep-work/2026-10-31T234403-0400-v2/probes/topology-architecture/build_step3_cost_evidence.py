"""Assemble read-only saved-cost evidence; no models, training or simulation."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import zipfile


P = Path(__file__).resolve().parent
R = P / "runs"
SOURCES = {}


def load(name):
    path = R / name / "learn/summary.json"
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def records(name):
    path = R / name / "learn/records.jsonl"
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [json.loads(line) for line in path.read_text().splitlines()]


def stage_row(stage, arm, source, phase):
    return dict(arm=arm, phase=phase, source=source, updates=800,
                parameters=stage["parameters"]["total"], train_s=stage["train_seconds"],
                eval_s=stage["eval_seconds"], guard_s=stage["guard"]["measured_seconds"],
                temperature_guard_s=stage["temperature_guard_seconds"],
                wall_s=stage["wall_seconds"], wall_scope="stage wall, not entire source invocation",
                optimizer_bytes=stage["optimizer_tensor_bytes"],
                reference_bytes=stage["representation_state_bytes"],
                replay_bytes=stage["replay_memory_tensor_bytes"], raw_stage=stage)


def repair_row(summary, phase, source):
    costs = summary["costs"]
    last = summary["curves"][-1]
    return dict(arm="Prometheus", phase=phase, source=source, updates=200,
                parameters=summary["actual_neural_parameters"], train_s=costs["train_seconds"],
                eval_s=costs["evaluation_seconds"], guard_s=costs["guard_seconds"],
                temperature_guard_s=0., wall_s=summary["wall_seconds"],
                wall_scope="standalone repair internal wall including initialization/search",
                optimizer_bytes=last["optimizer_tensor_bytes"], reference_bytes=None,
                reference_bytes_boundary="not separately measured in this repair logger; do not substitute current graphs for retained references",
                replay_bytes=0, formation_search=summary["formation_audit"]["costs"],
                neural_install_s=costs["neural_materialization_seconds"],
                raw_costs=costs, raw_materialization=summary["materialization"]["costs"],
                final_checkpoint=summary["final_checkpoint"],
                final_checkpoint_bytes=(R/source/"learn"/summary["final_checkpoint"]).stat().st_size)


def main():
    primary = load("continuous-primary-v7-n16-seed0")
    initial_baselines = load("continuous-primary-v7-n16-seed0-baselines")
    first_repair = load("n16-observed-formation-v1")
    second_repair = load("n16-history-formation-v1")
    v8 = load("n16-repaired-stage2-v8")
    matched = load("n16-matched-baselines-v1")
    acquisition = load("n16-acquisition-contribution-v1")
    monodromy = load("n16-monodromy-contribution-v1")
    native = next(x for x in primary["completed"] if x["model"] == "topo")
    v8_model = v8["completed"][0]
    chain = [stage_row(native["stages"][0], "Prometheus", "continuous-primary-v7-n16-seed0", "task0-source800"),
             stage_row(native["stages"][1], "Prometheus", "continuous-primary-v7-n16-seed0", "task1-source800"),
             repair_row(first_repair, "task1-formation-extra200", "n16-observed-formation-v1"),
             stage_row(v8_model["stages"][0], "Prometheus", "n16-repaired-stage2-v8", "task2-repaired-parent800"),
             repair_row(second_repair, "task2-history-extra200", "n16-history-formation-v1")]
    chain[3]["source_whole_model_wall_s"] = v8_model["model_wall_seconds"]
    costs = chain.copy()
    baseline_chains = {}
    for completed in matched["completed"]:
        name = completed["model"]
        old = next(x for x in initial_baselines["completed"] if x["model"] == name)
        rows = [stage_row(s, name, "continuous-primary-v7-n16-seed0-baselines", f"task{s['stage']}-source800")
                for s in old["stages"][:2]]
        for phase in completed["phases"]:
            c = phase["costs"]
            rows.append(dict(arm=name, phase=phase["phase"], source="n16-matched-baselines-v1",
                updates=c["actual_gradient_updates"], parameters=phase["parameters"]["total"],
                train_s=c["train_seconds"], eval_s=c["eval_seconds"], guard_s=0.,
                temperature_guard_s=0., wall_s=c["phase_wall_seconds"],
                wall_scope="continuation phase, excludes model restore/final reload",
                optimizer_bytes=phase["optimizer_tensor_bytes"], reference_bytes=0,
                replay_bytes=phase["replay_memory_tensor_bytes"], raw_costs=c))
        baseline_chains[name] = dict(phases=rows, total_gradient_updates=sum(x["updates"] for x in rows),
            summed_train_s=sum(x["train_s"] for x in rows),
            final_metrics=completed["final_metrics"],
            new_continuation_whole_model_wall_s=completed["model_wall_seconds"])
        costs.extend(rows)
    extra = [dict(run="continuous-primary-v7-n16-seed0", arm="topo", phase="discarded original task2",
                  updates=800, raw_stage=native["stages"][2]),
             dict(run="continuous-primary-v7-n16-seed0", arm="topo_off", updates=2400,
                  model_wall_s=next(x for x in primary["completed"] if x["model"]=="topo_off")["model_wall_seconds"])]
    for completed in initial_baselines["completed"]:
        extra.append(dict(run="continuous-primary-v7-n16-seed0-baselines", arm=completed["model"],
                          phase="discarded original task2 before altered-parent continuation", updates=800,
                          raw_stage=completed["stages"][2]))
    for name, updates in [("n16-cache-training-v1",800), ("n16-acquisition-contribution-v1",1000),
                          ("n16-monodromy-contribution-v1",800), ("projection-numeric-v1",1)]:
        receipt = R/(name+".runner-status.json")
        measured = json.loads(receipt.read_text())
        extra.append(dict(run=name, updates=updates, receipt=str(receipt),
                          external_runner_s=measured["elapsed_seconds"],
                          scope="additional development/attribution, not selected successful parent chain"))
    overhead = []
    for name, c in [("Prometheus",native)] + [(x["model"],x) for x in initial_baselines["completed"]]:
        overhead.append(dict(arm=name, source_model_wall_s=c["model_wall_seconds"],
            all_source_stage_wall_sum_s=sum(s["wall_seconds"] for s in c["stages"]),
            unallocated_source_model_overhead_s=c["model_wall_seconds"]-sum(s["wall_seconds"] for s in c["stages"]),
            scope="source invocation also executed discarded task2; overhead not arbitrarily prorated"))
    index_checks = []
    for source, phase in [("n16-observed-formation-v1","task1_extra200"),
                          ("n16-history-formation-v1","task2_extra200")]:
        with zipfile.ZipFile(R/source/"learn/additional-training-indices.npz") as a, \
             zipfile.ZipFile(R/"n16-matched-baselines-v1/learn"/(phase+"-indices.npz")) as b:
            left, right = a.read("indices.npy"), b.read("current.npy")
            index_checks.append(dict(source=source, matched_phase=phase, exact_saved_current_array_equal=left==right,
                                     sha256=hashlib.sha256(left).hexdigest()))
    ledger = dict(schema="n16-successful-chain-costs-v1", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        M0="NOT_PASSED", MAIN="TOPOLOGY_USER_LOCKED", primary_baseline="Transformer",
        successful_native_chain=chain, total_native_gradient_updates=2800,
        total_native_synced_train_s=sum(x["train_s"] for x in chain),
        total_native_recorded_eval_s=sum(x["eval_s"] for x in chain),
        total_native_guard_s=sum(x["guard_s"] for x in chain),
        matched_baseline_chains=baseline_chains, matched_extra_index_checks=index_checks,
        successful_chain_and_extra_development_separate=True, extra_development=extra,
        unallocated_source_overhead=overhead, source_files=SOURCES,
        common_data=dict(train_rows=12288, evaluation_rows=6144, total_terminal_rows=18432,
            array_bytes=3096576, native_total_training_example_uses=179200,
            transformer_and_adapter_total_training_example_uses=179200,
            replay_current_example_uses=115200, replay_old_example_uses=64000,
            Transformer_padded_input_positions_per_example=20,
            Transformer_total_training_input_positions=3584000,
            new_external_labels_during_formations=0,
            semantics="Formation additionally scans/reuses TRAIN labels; zero new labels is not zero feedback-use/search cost. Same samples and gradients do not imply same representation or computation."),
        timing_scope="Do not add nested materialization/continuous-evaluation to enclosing eval/wall. Selected scopes span multiple resumed invocations; no single measured end-to-end successful-chain runtime exists.",
        missing_costs=["process warmup/data generation shared overhead allocation", "complete CPU RSS/VRAM and minimum deployment memory", "FLOPs/energy/hardware quantum costs", "repeated controlled runtime confidence intervals", "all prior campaign/HPO monetary cost", "future-task calibration and scaling"],
        acquisition_boundary="Both successful repairs already score100% at relative step0 after formation, before the200SGD updates. Those updates test continued learning/retention.")
    (P/"n16-successful-chain-costs-20261101.json").write_text(json.dumps(ledger,indent=2)+"\n")
    unified = dict(schema="step3-contribution-results-v1", M0="NOT_PASSED", MAIN="TOPOLOGY_USER_LOCKED",
        primary_baseline="Transformer", acquisition=acquisition, monodromy=monodromy,
        matched_baseline_continuation=matched, successful_chain_costs="n16-successful-chain-costs-20261101.json",
        original_source_baseline_costs_included=True, total_success_chain_updates_per_arm=2800,
        source_files=SOURCES, acquisition_boundary=ledger["acquisition_boundary"])
    (P/"step3-contribution-results.json").write_text(json.dumps(unified,indent=2)+"\n")
    curves=[]
    def add(study,arm,phase,relative,global_step,metrics,formation=False):
        for task, splits in metrics.items():
            for split,m in splits.items():
                interfaces={"soft_acc":"neural_soft","native_acc":"native_graph",
                    "continuous_native_acc":"continuous_lift_cached","graph_native_acc":"native_graph",
                    "backend_native_acc":"actual_selected_backend","accuracy":"Transformer_neural"}
                for key, interface in interfaces.items():
                    if key in m:curves.append(dict(study=study,arm=arm,phase=phase,relative_step=relative,
                        total_gradient_updates=global_step,task=int(task),split=split,
                        execution_interface=interface,accuracy=m[key],formation_already_applied=formation))
    for study, summary, key in [("acquisition",acquisition,"results"),("monodromy",monodromy,"completed")]:
        for arm in summary[key]:
            for c in arm["curves"]:add(study,arm["arm"],"additional200",c["step"],None,c["metrics"],
                arm["arm"] in ("history_full","history_no_explicit_reuse"))
            c=arm["costs"]
            costs.append(dict(arm=study+":"+arm["arm"],phase="additional200",updates=200,
                parameters=arm.get("final_parameters",arm.get("parameters")),
                train_s=c["train_seconds"],eval_s=c["evaluation_seconds"],guard_s=c["guard_seconds"],
                wall_s=arm.get("wall_seconds",arm.get("model_wall_seconds")),wall_scope="whole component arm"))
    for source in ("continuous-primary-v7-n16-seed0","n16-repaired-stage2-v8"):
        for c in records(source):
            if c["kind"]!="curve" or c["model"]!="topo" or (source.startswith("continuous-primary") and c["stage"]>1):continue
            offset=(0,800,1800)[c["stage"]]
            metrics={t:{s:dict(soft_acc=m["soft_acc"],continuous_native_acc=m["hard_acc"]) for s,m in splits.items()}
                     for t,splits in c["terminal_metrics"].items()}
            add("successful_chain","Prometheus",f"task{c['stage']}-source800",c["step"],offset+c["step"],metrics)
    for summary,phase,offset in [(first_repair,"task1-formation-extra200",1600),
                                 (second_repair,"task2-history-extra200",2600)]:
        for c in summary["curves"]:add("successful_chain","Prometheus",phase,c["step"],offset+c["step"],c["metrics"],True)
    for arm in matched["completed"]:
        for phase in arm["phases"]:
            offset={"task1-extra200":1600,"task2-original800":1800,"task2-extra200":2600}[phase["phase"]]
            for c in phase["curves"]:add("successful_chain",arm["model"],phase["phase"],c["phase_updates_completed"],offset+c["phase_updates_completed"],c["metrics"])
    for c in records("continuous-primary-v7-n16-seed0-baselines"):
        if c["kind"]=="curve" and c["stage"]<2:
            metrics={t:{s:dict(accuracy=m["hard_acc"]) for s,m in splits.items()} for t,splits in c["terminal_metrics"].items()}
            add("successful_chain",c["model"],f"task{c['stage']}-source800",c["step"],c["stage"]*800+c["step"],metrics)
    with (P/"step3-learning-curves.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    fields=["arm","phase","source","updates","parameters","train_s","eval_s","guard_s","temperature_guard_s","wall_s","wall_scope","optimizer_bytes","reference_bytes","replay_bytes"]
    with (P/"step3-costs.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(costs)
    lines=["# N16 成功链成本证据（非运行计划）", "", "M0 未通过。记录当前成功链与开发额外消耗，主要基线为 Transformer。", "",
        "成功链：原阶段0的800步 → 原阶段1的800步 → 数据约束形成后200步 → 修订父检查点上阶段2的800步 → 历史形成后200步。合计2800次梯度更新。", "",
        "| 片段 | 参数 | 同步训练秒 | 评估秒 | 保护秒 | 已记录 elapsed 秒及范围 |", "|---|---:|---:|---:|---:|---|"]
    for row in chain:lines.append(f"| {row['phase']} | {row['parameters']} | {row['train_s']:.4f} | {row['eval_s']:.4f} | {row['guard_s']:.4f} | {row['wall_s']:.4f}; {row['wall_scope']} |")
    lines += ["",f"上述同步训练片段之和 {ledger['total_native_synced_train_s']:.4f} 秒；评估 {ledger['total_native_recorded_eval_s']:.4f} 秒。它们不是单次端到端成功运行的总耗时。",
        "", "形成在追加SGD前的step0已经达到100%；后续200步检查能力保持与持续更新。额外搜索、重复标签访问、神经安装与连续执行全部在JSON中单列；嵌套计时不可重复相加。",
        "", "三个强基线亦合计2800次更新，包括原0/1阶段成本；新增1200步的实际样本索引已与对应修订逐字节核对。Replay使用一半current、一半old，标签重复访问与普通学习方式不同。",
        "", "原失败阶段2、OFF消融、缓存重训、形成/执行归因和数值探针属于额外开发消耗，已分列，未从成本记录删除。原共享调用的初始化/收尾开销未武断分摊。",
        "", "尚缺重复端到端计时、能耗、完整RAM/VRAM、FLOPs及长期扩展成本；当前不能宣称完整资源优势。详见同名JSON及 step3-contribution-results.json / step3-learning-curves.csv / step3-costs.csv。"]
    (P/"n16-successful-chain-costs-20261101.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(native_train_s=ledger["total_native_synced_train_s"],curve_rows=len(curves),cost_rows=len(costs),baseline_train_s={k:v["summed_train_s"] for k,v in baseline_chains.items()},extra_current_indices_all_equal=all(x["exact_saved_current_array_equal"] for x in index_checks))))


if __name__=="__main__":main()
