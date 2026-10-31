# Prometheus · NOW

**Rolling week: 2026-10-31 → 2026-11-06 · human budget 4–6 h/week · no catch-up debt**  
**唯一活跃研究文档。** 本文记录当前科学问题和行动；`AGENTS.md` 是常驻研究宪法；`food/` 与原始代码/证明是资料和研究产物。不要恢复多层计划或独立 Deep Work 台账。

## Vision · 不随每周计划改变的方向

**How do computational mechanisms emerge, become causally functional, and change the future accessibility of new computational capabilities?**

我们研究自组织计算、Computational Evolvability 与开放式演化的交叉。保留原始神经科学问题 *How can neural networks learn to construct, compose, and improve their own internal computational mechanisms?*，但不预选神经网络为最终计算载体。

**World A**：依赖真实执行、可追踪和可干预的底层状态转移；希望研究复制、遗传、选择、生态反馈与计算机制积累如何发生。世界中的实际演化不等于生成式叙述。持续开放创新、复杂性增长或超人智能都**不预设**会出现。

## Current judgment · 本周选什么、不选什么

**唯一科学问题：** *When does an emergent spatiotemporal structure in a local dynamical system warrant a causal interpretation as a computational mechanism?*  
局部动力系统中的时空结构，何时能够被因果地识别为计算机制？

- **读 `food/#14` Crutchfield–Mitchell，先于 Avida。** 本周重点是分清图案、任务表现和因果计算机制；这篇论文在有外部指定目标的 density-classification CA 中分析分布式计算，不等于无目标开放演化。
- **停止 IsoNCA 深读**，需要具体局部机制或对称性对照时回查。放弃进一步深读是边际收益判断，不是认定神经网络不能自组织。
- **暂缓 Avida 与系统性的 Strogatz 学习**：等功能机制判据明确到可以用在遗传与选择研究时，再进入数字生物与非线性动力学。若一周后此路径无区分力，就改换入口。
- **本周不创建数字生态模拟器，也不为了使用 Lean 而形式化。** 先找到真正能改变判断的小问题。

## Science · 事实、假设、攻击

**SOURCE** — Crutchfield & Mitchell, *The Evolution of Emergent Computation*（1995；本地 `food/#14` 为技术报告版本）：研究在预先定义的 density-classification 任务下演化获得的 CA 规则，分析其时空信息处理结构。具体结构与结论须对原文逐项核对。

**HYPOTHESIS** — 如果我们声称一种时空结构承担计算功能，就应提出可区分的输入—信息流—输出关系，并预测针对该结构的干预如何影响结果；允许分布式、暂时性的机制。

**ATTACK** — 观察者可以给任意活动模式命名；扰乱任意关键位置都可能降低准确率。单纯“打坏后性能下降”无法证明特定结构承担所声称的计算功能。需要匹配扰动与竞争性解释。

**OPEN** — 如何选择不依赖任意表示坐标的干预对象？哪些证据足以区别信息传递、普通状态传播和真正的任务相关计算？

**下一篇候选（本周非必读）：** Ofria & Wilke (2004), *Avida: A Software Platform for Research in Computational Evolutionary Biology*；研究可执行复制、遗传变异、资源规则与数字计算功能演化。

## Deep Work · 什么时候值得让 AI 进行更长探索？

**决策：引入有界 Deep Work Session（DWS），暂不启动无目标长跑。** 按 `AGENTS.md` 第 5 节执行。DWS 只有在一个精确问题、检查过的前提、独立验证器、资源上限与终止条件全部就绪时才能启动。研究时长和 token 消耗不作为进展指标。

**本周候选试运行（有前提）：** 读完 `food/#14` 并锁定一个具体 CA 规则、初始条件集合和可观察输出后，尝试一次小型 DWS：

> 对某个候选时空结构，给出 **两种竞争性计算解释**，设计一个能区分它们的干预；如可在有限状态模型上实施，则验证预测。若不能区分，产出最小反例或明确阻碍。

**启动前检查**：规则和任务是否来自已核对的原文？干预及其匹配对照是否可执行或清楚说明？什么结果会推翻机制解释？若这些问题未回答，先完成普通研究，不让 agent 自行扩张任务。

**首轮预算**：至多两条不同假设路线、一次强反驳、一次决定性验证尝试。发现错误前提、验证失败或连续两轮无新证据时立即停止/改题。结果只用 `CLAIM / STATUS / EVIDENCE / WEAKEST LINK / NEXT` 的短报告回传；必要的 `.lean`、测试代码和数据属于研究产物，不建新的文档管理系统。

**对 OpenAI 数学成果的校准**：公开的“平均约三小时 ChatGPT Pro thinking 等效计算量”来自未公开的内部模型和数千道问题的大规模评估，不能换算成我们可指定的每次思考时长、token 配额或成功率。借鉴的是**清晰问题 + 有界尝试 + 可检查结果**，不模仿时长。

## Action · 接下来七天

- [ ] **你独立阅读 `food/#14`**：90 分钟；重建任务、CA 更新、演化搜索目标与作者识别的时空结构。只记一处重要不确定性。
- [ ] **我独立攻击**：用原文和最接近先例核实结构解释；提出一个更平凡的竞争性解释或反例，不把推测写成原文事实。
- [ ] **共同做机制判据**：画出一个候选结构的输入—传播—输出关系，设计选择性干预及匹配扰动。纸面方案即可。
- [ ] **有明确验证器时试运行一次 DWS**；没有则记录阻碍，不补做、不为消耗算力寻找题目。
- [ ] **周末 15–20 分钟决策**：本周结果支持转向 Avida，还是先修正机制判据？只更新这里的下一周问题。

**完成标准：** 至少一个具体例子，使“漂亮的图案”“正确的行为”和“有因果证据的计算机制”三者的差异更明确。反例或发现不可判定边界同样算实质进展。

**评估 Deep Work 本身：** 试运行只记录是否获得可检查的新结果、发生了什么错误或空转，以及你花了多少精力核查；以后有可比的小任务时再与普通研究方式比较。未经实测，不把本协议视为效率提升的证明。

## Partnership & safety

双方具有独立研究判断权。AI 承担至少 50% 的**实质性智力工作**，优先探索、验证和攻击；你集中注意力检查关键前提与研究意义，不必阅读每条失败支线。遇到问题漂移、假设堆叠或连续空转时，AI 主动停下。已核实事实、独立重建、猜想、反例与未解问题严格分开。

现阶段仅做文献、数学、Lean 和有界离线玩具实验。数字演化原型不得自行联网、跨宿主复制、获取凭证、扩张权限或资源；隔离和终止能力必须由模拟之外的机制保障。

**来源：** [Crutchfield & Mitchell (1995)](https://pmc.ncbi.nlm.nih.gov/articles/PMC40688/) · [Ofria & Wilke (2004)](https://doi.org/10.1162/106454604773563612) · [OpenAI 数学成果说明 (2026-10-06)](https://openai.com/index/sharing-ai-progress-in-mathematics/) · [公开研究仓库](https://github.com/openai/math)
