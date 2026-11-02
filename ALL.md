# Prometheus — Roadmap to a Foundational Architecture Paper

**Ultimate Goal:** Develop and publish an original, general-purpose computational learning architecture with technical significance and experimental evidence comparable to *Attention Is All You Need* at its initial submission.

**Research Direction:** Self-Organizing Computation  
**MAIN:** TOPOLOGY

## TODO 1 — 完成可靠的计算能力增长机制

- 修复当前六阶段的 Soft / Native 双接口遗忘。
- 让系统稳定形成、调用、复用、修订计算原语，并根据经验扩展内部状态。
- 验证新能力获得后旧能力仍能执行。
- 在 N8、N16、N32 和多个随机种子下检验稳定性。

**完成标准：** 已学计算机制能持续积累，完整执行链可训练、可复现。

## TODO 2 — 建立真正由 Monodromy 驱动的核心学习原理

- 将原语形成、状态增长、共享参数更新与同伦保护统一为一套数学和算法机制。
- 让 Monodromy 直接影响学习自由度、参数增长或保护成本。
- 构造可重复使用的核心计算算子，给出清晰公式、更新规则和关键性质证明。
- 与普通约束保护、冻结、直接编译比较，建立拓扑机制的因果贡献。

**完成标准：** 拥有一项明确、可实现、有实测优势的原创核心计算机制。

## TODO 3 — 构建通用 Prometheus Block

- 将现有操作形成、调用、状态更新及拓扑保护统一成可训练模块。
- 支持任意长度的计算组合、多层堆叠、原语复用与动态状态增长。
- 扩展到非置换、非可逆或连续状态计算，明确可逆提升等实现的资源成本。
- 不为每个新任务重新设计核心架构。

**完成标准：** 同一 Prometheus Block 能作为不同任务的基础计算单元。

## TODO 4 — 证明现实任务迁移

使用同一核心架构完成两类实验：

1. **算法与组合推理：** 多步程序执行、状态记忆、未见组合泛化。
2. **真实数据学习：** 例如 Speech Commands v2 的连续类别学习，验证真实输入表示、记忆和能力保持。

允许更换输入输出适配层，不重新发明核心学习机制。

**完成标准：** 在合成计算任务之外，展示可重复的真实任务价值和跨任务适用性。

## TODO 5 — 正面击败有竞争力的现有架构

**主要 Baseline：Transformer。**

强竞争对手包括：

- Transformer + Replay / Adapter
- DEN、Hope / Nested Learning、PackNet
- Mamba-2、Titans（适合对应任务时）

首先确认对手能够有效学习任务，给予合理训练、调参和相应扩容机会。

全部比较：

- 新能力获取、旧能力保持、跨任务迁移
- 参数总量及每项新能力的边际参数增长
- 训练时间、推理延迟、存储、优化器与维护开销
- 模块、保护、原语形成和 Monodromy 的消融
- 多规模、多种子及统计稳定性

**完成标准：** 在至少一个重要、广泛适用的能力或资源维度展示显著且可复现的强基线优势，其他代价完整透明。

## TODO 6 — 建立可投稿的理论与实验证据

- 提炼核心数学定义、计算算子及算法。
- 明确原创贡献，与现有编织计算、程序学习、持续学习和动态架构区分。
- 完成关键证明、消融、扩展曲线及完整成本分析。
- 独立复核代码、结果、数据划分和主要科学主张。
- 形成完整论文、可复现代码与实验材料；公开前解决知识产权归属。

**完成标准：** 一篇能够经受顶级研究者独立审查、具有完整新架构与强实验结果的论文。

---

## Execution Principles

- 六项任务构成固定论文路线，不再使用 M0–M3 百分比推动研究。
- 允许并行推进理论、架构和实验，但始终沿同一 Topology MAIN。
- 所有实质性方法更新继续维护在唯一的 `water/method.md`。
- 已有成果、负面结果和原始证据全部保留。
- 不为追求单项准确率牺牲架构通用性和参数效率。
- 不因一次局部失败换方向，也不因完成一个小实验就停止研究。

**Success = Original Principle + General Architecture + Important Competitive Advantage + Reproducible Evidence.**