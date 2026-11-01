# Prometheus Deep Work v2

本包修订科学调度，不另造执行平台。唯一宪法是项目根 AGENTS.md。`CONSTITUTION.md` 只作旧链接兼容指针。

- `WORKFLOW.md`：探索/验证双模式；三个独立思考入口；成功后的价值检查；一主线与一个机制不同的高风险备选；证据与恢复。
- `ROLES.md`：按需加载的任务合同，不全部塞入每个进程。
- `DEPLOY.md`：运行准备和真实宿主交接。模型名称/供应商参数不硬编码。
- `runtime.example.json`：本工作流的调度规格，不是 Codex 或供应商配置。
- `inputs/seed.md`：不含旧答案和书单分类的共同输入。
- `inputs/books.md`：保留的可选导航，不默认加载给所有入口。
- `templates/`：简短的机制、候选、测试、运行记录。
- `tools/audit.py`：记录检查；区分请求、观察、有效模式和本轮授权。
- `tools/pro_probe.py`：上一版的可选 API 探针保留用于兼容，不是默认部署步骤。只有核查当前接口兼容性并获单独调用授权后才能使用 `--live`，不得拿它探测新收费路径。

先使用[更新包 INSTALL.md](../prometheus-update/INSTALL.md) 安装/合并规则；再在新会话使用 [START.md](../prometheus-update/START.md)。设计与安装不等于研究已经执行。辅助工具的单元测试也不验证发现能力。

本地兼容：`pro_probe.make_receipt()` 与 v2 的 `receipt_from()` 都保留；旧回执字段仍可读。新运行显式设置 `mode_policy` 时，`authorization_receipt` 指向本运行内的 JSON 对象，必须包含与记录一致的 `run_id`、`requested_mode`、`authorization_scope: "this_run"` 和 `authorized_observed_modes`。这是授权记录的绑定检查，不是用户身份或同意的自动认证。模式回执能解析时核对模型、context、effort 与已声明模式；未提供可机读字段时明确警告，不将宿主选择推断成服务端 Pro。旧 campaign 使用其归档审计器，不回写旧证据来迁就新 schema。
