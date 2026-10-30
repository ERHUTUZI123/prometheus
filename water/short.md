# Prometheus · Short

**Period:** 2026-10-30 → 2026-11-12（两周）  
**Budget:** 每周约 4–6 小时；弹性安排，no catch-up debt  
**Mode:** Read → Reconstruct → Compare → Extract foundations。暂不要求 Python、完整架构或正式 PRP 研究轮次。

## Research anchor · 研究锚点

**Field:** Self-Organizing Neural Computation  
**Fundamental question:** *How can neural networks learn to construct, compose, and improve their own internal computational mechanisms?*

**Two-week question / 两周主问题：**

> **In Neural Cellular Automata, what organizes itself during execution, what is supplied by the designer or external training, and which mechanisms can we reconstruct as reusable research foundations?**
>
> 在 NCA 中，运行期究竟出现了什么自组织行为？哪些条件由设计者或外部训练提供？哪些机制足够明确，能够成为后续研究使用和质疑的“基石”？

**Scope:** 只深入比较 **Growing NCA (2020)** 与 **Growing Isotropic NCA (2022)**。两周内研究 F1、F2、F4；对 F3（可塑性）只记录待回答的问题，留到之后的 Hebbian/STDP 和 SAM 阅读。不急于声称发现了新的计算机制。

---

## Week 1 · Oct 30 – Nov 5

### Question · How does a Growing NCA work?

> 一个细胞怎样根据邻域更新自身状态？全局目标图案和学习信号在哪里进入系统？

**Source:** [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Mordvintsev et al. (2020)

- [ ] **Session 1 · 90 min — State and interaction.** 阅读 *The Model*。识别 cell state、隐藏通道、neighborhood perception、shared update network、stochastic update mask。画出单个 cell 的更新路径；分清局部可见量和全局状态。
- [ ] **Session 2 · 90 min — Learning and external objectives.** 阅读 *Learning to Grow*。追踪 target image、loss、training samples / sample pool、optimizer 和参数更新。特别指出 *local state update* 与 *training of the shared update rule* 的位置和时序差异。
- [ ] **Session 3 · 60–90 min — Persistence and regeneration.** 阅读 *What Persists, Exists* 及 regeneration 实验，确认论文展示的是何种稳定性与恢复现象；查清它们与训练设置的关系。
- [ ] **Session 4 · 30–45 min — Reconstruct without the PDF.** 独立写出一条更新方程或流程图，然后回到原文逐项核对。所有尚未理解的参数直接标为 `open`。

**Week 1 outputs / 本周产出：**

1. `water/mechanisms.md`：增加 Growing NCA 一行，区分 `fixed by design / externally optimized / locally executed / observed emergent behavior / demonstrated function`。
2. `water/foundations.md`：从文献中提取 **F1 · State & Representation** 和 **F2 · Local Interaction**；可以先只做一张合并卡片。

**Week 1 check / 完成判断：** 关掉文章，能说明 NCA 中谁更新 cell state、谁优化共享更新规则，以及一个局部单元能访问哪些信息。

---

## Week 2 · Nov 6 – Nov 12

### Question · Which organizing behaviors depend on design constraints?

> 改变方向性与对称性假设，会怎样影响图案的形成？哪些组织能力依旧依赖外部目标和训练程序？

**Source:** `food/#6` — *Growing Isotropic Neural Cellular Automata* (2022)；按需回读 Growing NCA 的 persistence / regeneration。

- [ ] **Session 1 · 90 min — Identify the constraint.** 从 Growing NCA 写出原始局部感知中的方向相关假设。阅读 IsoNCA 对 isotropy、orientation 和 symmetry breaking 的定义；不要把 isotropy 当成“系统没有任何设计偏置”。
- [ ] **Session 2 · 90 min — Inspect evidence.** 阅读论文的机制与关键实验，检查 initial state、learned rule、training objective 和 observed patterns 各自承担的作用；对照原始 NCA 的实验设置。
- [ ] **Session 3 · 60 min — Comparison and foundation.** 填完 NCA 与 IsoNCA 的最小比较表。提取 **F4 · Self-Organization** 的一张基石卡片，写清楚组织形成的条件和证据边界。
- [ ] **Session 4 · 30–45 min — Synthesis.** 不看原文，写出一个两篇论文支持的具体结论、一个尚未解决的机制问题，以及一个可能区分两种解释的干预或对照。

**Comparison / 最小对照表：**

| Question | Growing NCA (2020) | Isotropic NCA (2022) |
|---|---|---|
| State / local information | | |
| Perception / directional bias | | |
| Fixed constraints | | |
| External training / objective | | |
| Observed organization / persistence | | |
| What the evidence does **not** establish | | |

**Week 2 check / 完成判断：** 能指出两篇论文至少一项明确的机制差异，说明它怎样改变可产生的行为，同时避免将“出现复杂图案”等同于“自主发明可复用计算机制”。

---

## Foundation extraction · 基石提取

与 `water/long.md` 保持一致：基石是已有文献中能够明确写出状态、输入、交互、更新、约束和实验证据的**具体机制**，不是未来架构中的固定模块。

两周重点处理：

| ID | 要从文献中得到什么 | 最低可接受证据 |
|---|---|---|
| **F1 · State & Representation** | cell state / hidden channels；哪些信息允许保存与读取 | 状态定义及来源位置 |
| **F2 · Local Interaction** | neighborhood / perception / shared update / update scheduling | 一条更新式或图示，标出固定与可训练部分 |
| **F4 · Self-Organization** | 图案形成、持续或再生所需条件；方向性/对称性约束 | 一项原文实验及其未覆盖的情况 |

**每张卡片最多半页，格式：**

```markdown
## F# — Mechanism · source/section
Question:
State / input / topology:
Update / learning rule:
Fixed / externally trained / locally adaptive:
Demonstrated behavior:
Boundary / missing evidence:
Status: source-established / reconstructed / speculative
```

只记录实际理解的内容；不为凑齐卡片而增加论文。`source-established` 表示原文报告的内容，`reconstructed` 是自己的重建，`speculative` 是尚待检验的想法；同一张卡片不同条目可以有不同状态。

## Two-week checkpoint · 两周检查

在 `water/mechanisms.md` 末尾留下 **150–250 字** 的判断，回答：

> **What is the strongest demonstrated example of self-organization in the two NCA papers? What additional evidence would be needed to claim that the system learned to construct a new, reusable computational mechanism?**
>
> 两篇 NCA 论文最扎实的自组织证据是什么？若要进一步声称系统学会构造新的、可复用的内部计算机制，还缺少什么证据？

**最低产出：** 一条自己能重建的更新机制；一张两篇 NCA 的对照表；至少一张有来源和证据边界的 F1/F2/F4 卡片；一个明确的未解决问题。没弄懂的部分可以标 `open`，不要求卡片全部完成。

**PRP readiness / 研究协议准备度：** 可以把那个未解决问题保存为未来 Research Packet 的候选题目，标明依赖的基石 ID、最近先例和可能反例。**本期不启动 Explore → Attack → Verify 的完整 PRP**；只有基石可独立重建、问题定义清楚时才开启，不按日历强制运行。

**Next:** 11 月中旬转向 `food/#15` *Neuronal Dynamics*（Hebbian/STDP）和 SAM，研究 F3（运行期可塑性）与外部优化的界限。若 NCA 仍有重要概念未厘清，就先解决它。

## Guardrails · 执行边界

- 第一遍阅读自己完成；AI 在独立重建之后用于核对方程、查找反例与厘清概念。
- `food/` 是参考库，两周只以两篇 NCA 为主；其他书遇到具体定义障碍时才按需翻查。
- 不以笔记数量、论文页数和新假设数量衡量进度。先保证一个机制真的理解。
- 不要求写 Python、提出架构或开展额外研究轮次；没有 catch-up debt。
- `water/short.md` 只安排两周的实际行动；`water/long.md` 保留三个月方向与 PRP 全流程。
