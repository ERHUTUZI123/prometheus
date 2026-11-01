# v2 迁移完成（仅部署）

入场工作区干净；真实 AGENTS SHA256 与包中 Library 基线不同，因此按条款语义合并，没有直接套用补丁或用 revised 文件覆盖根宪法。所有待改文件已预先备份到本目录 backup/，入场 diff/status 也已记录。

## 实际修改

- 根 AGENTS 保留本地愿景锚点、七条协作约束、操作性结果/因果证据规则与权限边界；加入范式级目标、开放原著阅读、组件中立性、DWS-E/DWS-V。宪法不写局部分支数量。
- 同步 v2 WORKFLOW、ROLES、DEPLOY、seed、候选/测试/记录模板与 runtime。P/K/R 是可调入口，集中一条主线和不同机制的备选；没有活跃的旧六线准则。CONSTITUTION 仅作指向根 AGENTS 的兼容入口。
- 保留本机额外元数据、可选跨线重构角色、书目导航和来源历史。probe 保留 make_receipt()/旧字段，添加 receipt_from() 兼容；未运行探针。
- audit 按新语义区分请求/观察/有效模式，仅接受本轮作用域的替代授权。结构化授权回执绑定 run_id、模式与作用域；结构化 worker 回执核对所属 context/model/effort，不能借用另一 worker。修复畸形字段崩溃，增加回归测试。这仍不是用户同意或服务端模式的真实性认证。
- 唯一主 NOW 仅调整当前阶段、活跃 DWS 规则和下一动作。Vision、历史 Science 内容及两个 Completed 结果逐字保留；历史“唯一待判断”不再决定下一轮路线。Git 路径沿用 water/now.md；本机显示 NOW.md，未改名或另建主文档。
- VALIDATION 区分作者打包时的历史说明与本机真实迁移验收。

修改的18个跟踪文件：

- `AGENTS.md`
- `deep-work/CONSTITUTION.md`
- `deep-work/DEPLOY.md`
- `deep-work/README.md`
- `deep-work/ROLES.md`
- `deep-work/SOURCES.md`
- `deep-work/VALIDATION.md`
- `deep-work/WORKFLOW.md`
- `deep-work/inputs/seed.md`
- `deep-work/runtime.example.json`
- `deep-work/templates/candidate.md`
- `deep-work/templates/mechanism.md`
- `deep-work/templates/probe.md`
- `deep-work/templates/run-record.json`
- `deep-work/tools/audit.py`
- `deep-work/tools/pro_probe.py`
- `deep-work/tools/test_tools.py`
- `water/now.md`

## 验收结果

- 原有12项离线测试、未修改更新包21项测试及包内不变量、最终合并40项离线测试均通过。首次合并38项也通过；独立复核发现 worker/context 绑定缺口后新增两项，最终40项通过。
- 6份配置 JSON 解析通过，7份活动/包内 Python 文件语法通过；未执行研究代码。
- 三条永久研究原则完整区块逐字一致；本地附加证据规则及执行权限边界保留。
- Research Vision、food、旧 campaign（含原失败审计）、更新包共8,373个受保护文件的原始指纹一致；书目导航未改。
- 新文件链接、唯一 NOW、规则冲突检查、修改范围、staged 状态与 git diff --check 均通过。详细证据见 validation.json、merge-decisions.json 和测试日志。

剩余部署阻塞：无。未检查或宣称下一轮实际能运行 Pro；那属于下一次明确启动时的预检。上一轮 Ultra 授权未变成默认许可。

下一步：在新会话使用 prometheus-update/START.md 明确启动。到此结束部署；未启动科研、训练、模型/API探针、付费调用或后台运行，未提交 commit，未修改全局设置。
