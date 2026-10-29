# Prometheus · Long

**Period:** 2026-10-27 → 2027-01-26（三个月研究路线；每月滚动检查）  
**Time budget:** 每周约 4–6 小时，弹性安排；未完成的时间不补课（no catch-up debt）  
**Mode:** 阅读一手文献、重建机制、数学推导、比较已有方法、记录 Markdown。现阶段不强制写 Python。

## 研究目标 · Purpose

未来三个月只回答一个决策问题：

> **Does Self-Organizing Neural Computation contain a specific, tractable, and falsifiable question that I want—and can realistically attempt—to investigate for another 6–12 months?**
>
> 自组织神经计算中，是否存在一个足够具体、可以验证、值得我继续独立研究 6–12 个月的问题？

围绕三个逐步收紧的问题展开：

1. **Mechanism — 机制：** 已有系统到底实现了什么？哪些规则由研究者设定，哪些参数通过外部目标训练，哪些变化真正发生在运行中的系统内部？
2. **Computation — 计算：** 怎样区分形态出现（pattern formation）、状态保持（persistence）、记忆（memory）与任务相关计算（task-relevant computation）？
3. **Research gap — 研究缺口：** 最接近的已有方法已经做到哪里？有什么可以被测量、被反驳的新问题？

**边界：** 不急于设计 Prometheus 的最终架构；不试图证明通用智能会涌现；不把 `food/` 当作必须读完的书单。科学问题可以改变，证据标准不能降低。

---

## Month 1 · 2026-10-27 → 2026-11-26

### 主问题 · What do local rules actually generate?

> **What emerges from local interaction rules, and what is supplied by the designer or the training process?**
>
> 局部相互作用究竟能生成什么？哪些能力事先被设计或训练过程提供了？

### 阅读路线 · Read in this order

**1. [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) — Mordvintsev et al., 2020（在线原文）**

- **读什么：** *The Model*、*Learning to Grow*、*What Persists, Exists* 及再生实验（regeneration）的相关内容。
- **带着什么问题：** 局部感知（local perception）、细胞隐状态（hidden state）和共享更新规则（shared update rule）分别扮演什么角色？目标图案、loss、训练样本池由谁提供？
- **理解目标：** 画出一个细胞从邻域状态到下一状态的更新流程，清楚区分 **local execution** 与 **externally optimized learning**。

**2. `food/#6` — *Growing Isotropic Neural Cellular Automata*（2022）**

- **读什么：** 原始 NCA 的方向性假设、isotropy、对称性与 symmetry breaking。
- **带着什么问题：** 如果去掉人造的方向信息，系统还能形成什么？什么结构依赖初始条件，什么结构依赖模型归纳偏置？
- **理解目标：** 解释两篇 NCA 在 *design constraints* 上的具体差异，以及这种差异为什么值得研究。

**3. `food/#15` — *Neuronal Dynamics*（Gerstner et al.）**

- **读什么：** §§19.1–19.3，关注 Hebbian learning、STDP 与 unsupervised adaptation；只有当记忆问题需要时才补 §§17.1–17.2。
- **带着什么问题：** 局部突触更新怎样依赖前后神经元的活动？状态改变与学习规则的改变有什么区别？
- **理解目标：** 不看原文写出并解释一种局部可塑性更新规则，说明它能产生什么、尚未证明什么。

**4. [How Biological Synapses Self-Assemble Gradient Learning](https://doi.org/10.1073/pnas.2606245123) — Liao et al., 2026**

- **读什么：** 问题定义、Self-Assembling Motif（SAM）机制，以及最关键的理论/实验主张。难证明先标记，别因为一段推导中断整条路线。
- **带着什么问题：** 所谓 *self-assembly* 具体发生在哪一级？哪些局部规则、结构或环境条件仍被预先给定？
- **理解目标：** 准确表述论文证明或演示的范围，避免将其直接理解为「系统自主发明任意学习算法」。

### 本月应当追问 · Questions to carry

- **Fixed / externally trained / locally adapted / emergent：** 四类机制在每篇文章中分别是什么？
- 局部更新（local update）是否意味着局部学习（local learning）？训练目标、误差信号和 credit assignment 从哪里来？
- 一个有序结构出现时，有什么证据表明它承担了 *useful function*？
- 如果改变训练目标或初始状态，最终组织是否仍会出现？这说明了什么，又不能说明什么？

### 月末产出 · Evidence of understanding

在 `water/` 中形成一张简洁的 **`mechanisms.md` 对照表**：

| System | Fixed by design | Externally optimized | Locally adaptive | Emergent behavior | Demonstrated function |
|---|---|---|---|---|---|
| Growing NCA | | | | | |
| Isotropic NCA | | | | | |
| Hebbian / STDP | | | | | |
| SAM | | | | | |

另选 **两条更新方程或机制流程图**，关掉论文后独立重建。可以记录不理解的部分，无须伪装成已经掌握。

**Gate 1 — 月末判断：**

> Can I distinguish **running a learned local rule** from **learning to construct a new functional rule or circuit**?
>
> 我能否在至少一个具体模型中清楚地区分「执行已学得的局部规则」与「自主形成新的功能性规则或回路」？

---

## Month 2 · 2026-11-27 → 2026-12-26

### 主问题 · When does organization become computation?

> **When does an emergent structure constitute memory, computation, or a reusable mechanism?**
>
> 一个涌现的结构在什么条件下才算记忆、计算，或可复用的内部机制？

### 阅读路线 · Read in this order

**1. `food/#14` — *The Evolution of Emergent Computation*（Crutchfield & Mitchell）**

- **读什么：** 这篇短文可以通读；重点看 density-classification task、时空信息传递结构，以及作者如何解释系统完成的计算。
- **带着什么问题：** 输入、输出、任务成功标准分别是什么？哪些时空结构可能携带或处理信息？能否通过干预检查它们的因果作用？
- **理解目标：** 以一个具体任务为例，说明为什么「有复杂图案」还不足以判断「实现了有用计算」。

**2. `food/#9` — *Structure and Interpretation of Computer Programs*（SICP）**

- **读什么：** 按问题选读 §§1.1、1.3、3.1，聚焦 procedures、composition、abstraction 与 state。无需顺序完成练习。
- **带着什么问题：** 什么使一个操作成为可调用、可组合的计算过程？怎样区分新增操作与原有参数的变化？
- **理解目标：** 提出一个临时的 *computational mechanism* 定义，并给出至少一个容易误判的反例。

**3. `food/#11` — *Nonlinear Dynamics and Chaos*（Strogatz；按需查阅）**

- **读什么：** fixed points、stability、linearization 的相关章节；只在解释持续性或吸引子时遇到困难才读。
- **带着什么问题：** 长时间保持状态、扰动后恢复状态、维持任务表现，分别需要哪些稳定性条件？
- **理解目标：** 能用最小动力系统解释「稳定」究竟指什么。

**备选：[A Movable Architecture for Robust Spatial Computing](https://www.cs.unm.edu/~williams/acw.pdf) — Ackley, Cannon & Williams（MFM）**

如果实际问题转向 *asynchronous local updates*、self-healing 或分布式计算的鲁棒性，可用 MFM 替换第 3 项，而非增加阅读负担。

### 本月应当追问 · Questions to carry

- **Memory：** 持久状态在什么条件下才成为可以被读取、利用的记忆？
- **Mechanism：** 机制应当由计算图、状态转移、功能映射，还是别的对象来刻画？
- **Composition：** 两个内部过程怎样组合，才算形成了可验证的新能力？
- **Robustness：** 扰动后恢复的是外观、状态、任务性能，还是承担任务的内部机制？
- **Causality：** 怎样排除某个被观察到的结构只是伴随现象（correlate），而非计算的因果组成部分？

### 月末产出 · Operational definitions

写一份简短的 **`definitions.md`**，为以下四个概念各给出：

1. 一个暂定的 *operational definition*（可以被观察或测量的定义）；
2. 一个已读文献中的正例；
3. 一个看起来相似、实际上不符合定义的反例（false positive）；
4. 一个尚未确定的边界问题。

四个概念：**mechanism / memory / composition / improvement**。

**Gate 2 — 月末判断：**

> Could another researcher use my criteria to distinguish **emergent organization** from **emergent useful computation** in a toy system?
>
> 如果把这些定义交给另一个研究者，他能否用它们判断一个小系统有没有产生任务相关的计算功能？

**两个月节点：** 如遇课程、工作或精力限制，可以停在这里。留下一个具体未解决的问题和它最接近的已有结果，就已经形成有效积累；不强制进入实现。

---

## Month 3 · 2026-12-27 → 2027-01-26

### 主问题 · What remains to be discovered—and how would we test it?

> **What do existing methods already achieve, and what observation could distinguish a genuinely new claim?**
>
> 最接近的研究已经解决了什么？什么实验结果能够支持或推翻一个新的具体主张？

### 阅读路线 · Read in this order

**1. [Differentiable Plasticity: Training Plastic Neural Networks with Backpropagation](https://proceedings.mlr.press/v80/miconi18a.html) — Miconi et al., 2018**

- **读什么：** 学习到的 plasticity 参数、任务执行时发生的在线更新、训练与测试阶段各自承担的功能。
- **带着什么问题：** 一个网络在任务中改变突触权重，是否等于它学会创造新的学习规则？可塑性本身是谁设计和优化的？

**2. [AutoML-Zero: Evolving Machine Learning Algorithms From Scratch](https://proceedings.mlr.press/v119/real20a.html) — Real et al., 2020**

- **读什么：** primitive operations、search space、fitness evaluation、发现的算法，以及研究者提供的外部搜索过程。
- **带着什么问题：** 自动发现算法已经达到什么程度？如果我想强调 *internal self-construction*，与外部 evolutionary search 的差异究竟在哪里？

**3. 回读最相关的两篇旧文献**

从 NCA、SAM、Crutchfield–Mitchell 或其他确实与候选问题相关的材料中选择。不以广泛补充新论文为目标；只核查与自己想法最近的先例及其限制。

### 本月应当追问 · Questions to carry

- 我的候选主张已经被哪个方法在什么层次实现过？是 weights、plasticity、learning rules、circuit structure，还是 externally evolved programs？
- 最接近的方法依赖哪些人工设定的 objective、search space、credit assignment 或 architectural constraints？
- 哪个**最小 toy task** 能让两个竞争性解释给出不同预测？
- 观察到改进时，怎样排除参数更多、训练更久、搜索预算更大、读出层更强等替代解释？
- 如果我无法负担验证成本，能否缩小假设，而不是扩大叙事？

### 月末产出 · A researchable question

写一页 **`question.md`**，无需强行提出原创结论，但要尽力填出：

- **Claim / 假设：** 一个具体的候选命题，标注 *provisional*。
- **Closest prior work / 最近先例：** 2–3 个方法，分别写明已经做到什么。
- **Minimal test / 最小验证：** toy task 的输入、输出、内部状态、允许的更新与观测指标。
- **Controls / 对照：** 相同预算的固定架构；适用时加入仅参数学习版本；再加入简单或随机更新/搜索基线。
- **Falsification / 反证条件：** 什么观察会迫使我否定或修正该命题？
- **Feasibility / 可行性：** 手算、数学推导、笔记本级别或低算力实验是否足够？如不足，明确指出瓶颈。

**Gate 3 — 三个月判断：**

> Is there a question that is **well-defined, distinguishable from close baselines, and feasible to investigate independently**?
>
> 是否已经找到一个定义清楚、能与已有方法做出区分、并且现实中可以独立验证的具体问题？

有，则考虑进入下一阶段的实现或更深入理论研究。没有，则诊断障碍出在 **knowledge / definition / novelty / testability / resources** 哪一项，再调整方向；不因此直接否定整个研究领域。

---

## `food/` 的使用原则 · Reference, not backlog

这 20 份本地 PDF 是长期参考库，不是三个月阅读任务。按问题查阅：

| 遇到的问题 | 可查阅资料 |
|---|---|
| 自组织、适应性与形态失稳 | `#1` Kauffman、`#2` Ashby、`#3` Turing |
| 信息量、编码与传递如何度量 | `#10` Cover & Thomas |
| 稳定性、闭环与抗扰动 | `#11` Strogatz、`#12` Feedback Systems |
| 不熟悉的神经编码或学习模型 | `#16` Dayan & Abbott |
| 如何识别分布式表示中的内部计算 | `#19` SAE、`#20` Toy Models of Superposition |
| 生成结构与层次组织 | `#8` Simon、`#13` The Algorithmic Beauty of Plants |
| 物理系统是否真的发生内部学习 | `#17` Memristive Networks、`#18` Photonic Reservoir Computing |

`#4` von Neumann（本地仅节选）、`#7` Hofstadter、`#21` McGinn 等继续保留；只有具体问题确实需要时才打开。尤其要把 **cognitive closure** 的哲学猜想与神经网络的实验性主张分开。

**规则：** 每次补充阅读最多先选一节或一篇；读完回到当月主问题。不要因为一本书有趣就把它自动升级成新的学习任务。

## 每周如何执行 · Weekly practice

- **开始前 5 分钟：** 写下一个问题：*What evidence or derivation could change my understanding today?*
- **阅读中：** 优先弄懂一个机制、方程或可区分的例子。遇到前置知识缺口，仅查足够解决问题的一小节。
- **结束前 10–15 分钟：** 关掉 PDF，按四项记录：**source claim / my reconstruction / uncertainty / next question**。能画流程图或写方程时，优先于长篇感想。
- **每月底 20 分钟：** 对照当月 Gate，记录阻碍属于概念、数学、文献先例、实验设计还是资源。下一月计划可以据此调整。

**Non-negotiables：** 不欠阅读债；不以页数或笔记数衡量进度；不强迫每周出现新想法；没有清晰实验之前不要求代码；不为维持计划牺牲课程、睡眠或足球。

## 三个月后的判断 · Decision checkpoint

把这句话写清楚即可，未必已经知道答案：

> **What might a neural system learn to construct that the closest existing system cannot? How would I detect it? What would falsify my explanation?**
>
> 一个神经系统可能学会构建什么，是最接近的已有系统还未展示的？我怎样观察到这一点？什么结果会推翻我的解释？

评估三件事：**scientific specificity（科学问题是否具体）／empirical or theoretical testability（能否验证）／personal feasibility（个人条件是否允许）**。

如果三者都有初步依据，下一阶段可集中做实验或理论研究；若仍有缺口，就缩小问题、补充必要知识，再决定后续六个月的投入。**长期研究兴趣可以稳定，具体假设始终允许被证据推翻。**
