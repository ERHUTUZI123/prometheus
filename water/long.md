# Prometheus · Long

**Period:** 2026-10-27 → 2027-01-26（三个月研究路线；每月滚动检查）  
**Time budget:** 每周约 4–6 小时，弹性安排；未完成的时间不补课（no catch-up debt）  
**Mode:** 阅读一手文献、重建机制、数学推导、比较已有方法、记录 Markdown。现阶段不强制写 Python。

## Research anchor · 研究锚点

**Field:** Self-Organizing Neural Computation  
**Fundamental question:** *How can neural networks learn to construct, compose, and improve their own internal computational mechanisms?*

Prometheus 研究神经系统如何通过学习自主形成、组织、组合、保留、复用和改进内部计算机制，并由此获得与迁移日益复杂的计算能力。

- **研究对象：** 内部计算机制自身的形成和变化；不预设它们必然对应固定的人类设计模块。
- **方法边界：** 严格区分状态更新规则、参数学习规则、外部优化及运行期自适应。
- **证据标准：** 能力产生（creation）、机制识别（identification）、因果解释（understanding）分别检验；生物类比只作为候选机制来源。
- **远期愿景：** 发现新的学习原理或神经架构；当前不预选架构，也不把 cognitive closure 作为技术前提（见 `fire/cognitive-limits.md`）。

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

## Foundational Components · 基石库

“基石”指已有文献中可明确说明其**状态、输入、更新、约束、可测行为和证据来源**的具体机制。它可以是数学构造、学习规则或可复现的实验现象；不等于未来系统的预制模块。

| ID | 基石问题（提取具体机制，而非综述） | 当前优先来源 | 对应月份 |
|---|---|---|---|
| F1 · State & Representation | 允许什么状态、什么表征、什么局部可见信息？ | Growing NCA、Neuronal Dynamics | M1 |
| F2 · Local Interaction | 谁与谁交互？同步/异步、邻域和信息传播如何限定？ | NCA、IsoNCA；MFM 按需 | M1–M2 |
| F3 · Plasticity & Adaptation | 哪些变量可在运行期改变？学习信号和信用分配来自何处？ | Hebbian/STDP、SAM；Miconi | M1、M3 |
| F4 · Self-Organization | 什么局部规则生成并稳定维持特定整体结构？设计者固定了什么？ | NCA、IsoNCA；Turing/Kauffman 按需 | M1 |
| F5 · Memory & Stability | 保存的状态何时可读取、可抗扰动并影响未来任务？ | Neuronal Dynamics、Strogatz 按需 | M2 |
| F6 · Computation & Composition | 什么证据表明结构执行操作、被复用、被组合？ | Crutchfield–Mitchell、SICP | M2 |
| F7 · Selection & Feedback | 哪些反馈选择了有用机制？评估/搜索过程位于系统内部还是外部？ | SAM、AutoML-Zero、Differentiable Plasticity | M3 |

**记录办法：** 按阅读进度填写一个 `water/foundations.md`。每个已经理解的基石只需一张卡片：

```markdown
## F# — Name / 来源（具体论文、章节、页码）
Question: 此机制要回答的一个问题
State / input / topology: 状态、输入、允许的交互
Update / learning rule: 一条方程或伪代码；哪些量固定、可学习、运行期可变
Demonstrated behavior: 原文做了什么实验，观察到了什么
Boundary: 原文没有证明什么；是否存在外部目标/训练器
Reuse hypothesis (optional): 未来可能与哪个基石形成可检验组合？
Status: source-established / reconstructed / speculative
```

**不要为填表而补读七个领域。** 当月只处理该月文献出现的基石；未知栏位留空。重点是掌握可使用、可被质疑的机制，而不是积累术语。

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

另选 **两条更新方程或机制流程图**，关掉论文后独立重建。可以记录不理解的部分，无须伪装成已经掌握。 同步在 `water/foundations.md` 填写本月已经理解的 F1–F4 卡片；不要求四项都填满。

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

四个概念：**mechanism / memory / composition / improvement**。 在 `water/foundations.md` 补 F5、F6，并从已读基石中提取 1 个可以被形式化的候选问题；条件具备时可试运行 PRP。

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

在 F7（选择/反馈机制）的对照阅读完成后，写一页 **`question.md`**，无需强行提出原创结论。允许 AI 在 PRP 中提供备选假说和最小实验设计，由本人检查来源、数学推导与基线。尽力填出：

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

## AI-assisted Research Protocol · 有界研究尝试

可参考 [OpenAI 2026-10-06 数学发布](https://openai.com/index/sharing-ai-progress-in-mathematics/) 与其 [数学仓库](https://github.com/openai/math) 的**固定程序、候选问题筛选、结果记录、形式验证**等公开原则。其完整固定程序、所用内部模型和运行配置未公开；平均「约三小时 ChatGPT Pro Thinking 等效计算量」不是我们可以直接复现的模型用量或计时承诺。以下 PRP 是 **Prometheus 自己设计的流程**，不应称为 OpenAI 的原始算法。

**触发条件：** 一旦掌握至少两张可重建的基石卡片，并出现值得比较的具体机制问题，便可开展一次尝试。通常 **每两周至多一次**，不强制排期；优先确保个人阅读和推导质量。若没有清晰问题，就不启动 AI 研究轮次。

### PRP · One bounded research attempt

**0. Research packet — 研究输入（1–2 页）**  
由本人汇编：一个精确的问题、有关基石和更新式、已知最接近的结果（附引用）、允许的假设、需要证实或推翻的主张、可验证目标，以及可用计算和时间资源。AI 不直接从整个 `food/` 猜任务，也不根据长期愿景空想架构。

**1. Formalize — 明确问题**  
让 AI 将问题写成「状态—交互—学习规则—环境—功能指标—已知边界」，指出含糊定义和隐藏的全局信息通道。本人决定是否认可该问题表述。

**2. Explore — 两次独立尝试**  
使用同一份 packet，分别探索两条明确不同的机制假设或理论路径（可以用独立会话避免相互污染）。必须列出前提、关键推导、与现有基石的关系及可能失败方式。每条路径只追求一个可检查的中间结果。

**3. Attack — 反例审查**  
选择最有希望的一条，要求 AI 主动构造反例、查找已有工作、检查非局部依赖、额外参数、外部训练或目标泄漏；不允许只做赞成性解释。失败也要记录。

**4. Verify — 独立验证**  
本人重新推导关键式子、核查原始出处，必要时设计极简手算例子；进入实验期后才使用程序测试或数学形式化工具。计算检验不等于因果解释；所有未检验结论标注 `hypothesis`。

**5. Consolidate — 形成可继承的研究结果**  
只保留：已核实的机制事实、可复现的推导、明确被否定的猜想、未解决的定义、下一次尝试可以使用的问题。输出一个简短 `water/research-log.md` 条目。避免每次让模型重新消化全部聊天记录。

### 一次尝试的产出格式

```markdown
## PRP-001 — Candidate question
Research packet: 相关 foundations ID / 精确来源
Claim: 精确且可被否定的主张
Known facts: 已核实结果与适用假设
Candidate A / B: 各自的一条机制路径或推导
Adversarial checks: 反例、最近先例、隐藏设计假设
Verification: 人工或程序到底检查了什么？
Status: verified / partially checked / falsified / open
Next smallest question: 唯一的下一步
```

### First pilot · 第一轮试运行（可在 M2 末或 M3 开始）

> **在完全相同的局部状态更新框架下，允许运行期局部可塑性，是否会产生可重复利用、能够影响第二个任务的新内部功能？怎样与只改变隐藏状态或单纯增加容量区分？**

这是一个用于训练提问方式的 **provisional question**，尚无新颖性结论。其目的在于强制把 NCA、STDP/SAM、记忆和机制定义接到同一个可验证问题上；先写出最小假想任务和反例，不要求立刻实现模型。可与 Miconi 等近邻工作比较后修改或否定。

**资源原则：** 每次尝试采用 2 条候选路径 + 1 轮强反例审查 + 1 次本人验证的有限流程。`~3h Pro` 是 OpenAI 未公开系统的比较口径，**不能用来推算 ChatGPT 当前消息额度、实际研究耗时或成功率**。每周 4–6 小时仍是整个项目预算，不为研究尝试另设加班。

---

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
