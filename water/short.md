# Prometheus · Short

**Period:** 2026-10-29 → 2026-11-11（两周）  
**Budget:** 每周 4–6 小时；灵活安排，no catch-up debt  
**Mode:** Reading → Reconstruction → Comparison。只做阅读、推导和 Markdown 笔记；不要求 Python。

## 两周主问题 · Research Question

> **What actually emerges in Neural Cellular Automata, and which capabilities depend on externally chosen objectives, learned parameters, and architectural constraints?**
>
> NCA 中究竟有什么是涌现的？哪些能力依赖外部训练目标、已学习的参数和人为设定的架构约束？

两周只研究 **NCA 这一类具体系统**。不提前扩展到完整的人工生命、记忆理论或最终架构。

## Week 1 · Oct 29 – Nov 4

### 问题 · How does a Growing NCA work?

> 一个细胞只接收邻域信息，为什么能参与形成全局图案？这个过程中的「学习」发生在哪里？

**读：** [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)（2020）

1. **The Model**：识别 cell state（包括隐藏通道）、local perception、shared update network、stochastic update mask。
2. **Learning to Grow**：识别目标图案、loss、训练过程，以及 learned parameters 何时更新。
3. **What Persists, Exists / regeneration**：先看相关实验与结论；若时间不够，下周接着读。

**读后独立重建（20–30 分钟）：**

- 画一张 *neighborhood → perception → state update → global pattern* 的流程图，或写出一条状态更新公式。
- 在图旁分别标明 **fixed by design / externally trained / locally executed / emergent outcome**。
- 用 100–200 字回答：*Why is a learned local update rule not automatically a self-invented learning rule?*

**本周完成条件：** 不看原文，能解释 NCA 如何更新状态，以及全局训练目标怎样影响局部规则。只需半页至一页笔记。

## Week 2 · Nov 5 – Nov 11

### 问题 · Which structures depend on the designer's assumptions?

> 如果改变局部感知中的方向信息，哪些组织仍能形成？方向信息和对称性究竟从哪里来？

**读：** `food/#6` — *Growing Isotropic Neural Cellular Automata*（2022，8 页）

1. 先写下原始 NCA 中哪些感知操作具有方向依赖（direction dependence）。
2. 阅读 isotropy、symmetry breaking、seed / initialization 相关论述与实验。
3. 对照原始 [Growing NCA](https://distill.pub/2020/growing-ca/) 的 persistence / regeneration，确认哪些结果依赖训练方法，哪些与各向同性有关。**不要默认 IsoNCA 已展示更强的自主学习能力。**

**读后比较（20–30 分钟）：** 填一张小表：

| Question | Growing NCA (2020) | Isotropic NCA (2022) |
|---|---|---|
| Perception / directional bias（感知与方向约束） | | |
| Fixed architecture（固定架构） | | |
| External objective / training（外部目标与训练） | | |
| Emergent behavior（观察到的组织） | | |
| What remains unproven（尚未证明） | | |

**本周完成条件：** 能明确指出两篇 NCA 在归纳偏置（inductive bias）上的至少一项具体差异，并解释这种差异对形成组织意味着什么。

## 两周结束时 · Checkpoint

留下一段自己的判断（最多 150–250 字，中文即可，技术词保留英文）：

> **What is the strongest evidence for self-organization in these systems, and what would I still need to see before claiming self-construction of a new computational mechanism?**

判断时区分：**observed / inferred / still unknown**。

**下一步：** 11 月 12 日起再转向 `food/#15` *Neuronal Dynamics*（Hebbian learning、STDP）和 SAM，研究局部执行规则与局部可塑性究竟有什么差别。

## 使用边界 · Guardrails

- 只读以上两篇主文献；遇到障碍时按需查一个定义，不扩展阅读清单。
- 每次读完关掉 PDF，重建一个机制；理解优先于笔记篇幅。
- 未完成的部分顺延，无补课债；不牺牲睡眠和足球。
- `water/thoughts.md` 只记录真正改变理解的疑问或推导；其他灵感写入两行以内的 `Later`。
