# Prometheus · 跨学科参考文献索引

更新日期：2026-10-28。本索引以 `food/` 当前实际文件为准，共 **20 份本地 PDF**，沿用文件名前缀编号：1–4、6–21；第 5 号目前没有本地文件。另保留两项在线参考，不计入本地文件数量。

**研究主题：** Self-Organizing Neural Computation。围绕神经系统如何自主形成、组织、组合、复用和改进内部计算机制整理资料，与 [项目研究愿景](../README.md) 对应。技能获取与迁移、架构和学习规则是技术研究的切入点，具体方法仍未确定；认知封闭作为独立的长期猜想保留。

**用途：** 按问题查阅的长期资料库。计划见 [water/long.md](../water/long.md) 和 [water/short.md](../water/short.md)，推导与理解记录见 [water/thoughts.md](../water/thoughts.md)。收录不代表已读完或已复现；下文的“研究问题”是 Prometheus 的思考方向，不是原作者已证明的结论。

## 分类导航

| 主题 | 本地编号 | 查阅的问题 |
|---|---|---|
| 自组织与适应 | 1、2、3 | 局部相互作用怎样产生结构、稳定性与适应？ |
| 自动机与形态生成 | 4、6 | 局部规则、对称性与初始条件怎样影响整体组织？ |
| 认知、类比与设计 | 7、8 | 概念、抽象与层次组织怎样帮助构建计算机制？ |
| 程序抽象与信息论 | 9、10 | 怎样表示、组合与度量内部计算？ |
| 动力系统与反馈控制 | 11、12 | 怎样分析稳定性、扰动响应与功能维持？ |
| 生成规则与涌现计算 | 13、14 | 怎样从局部演化中识别可分析的计算功能？ |
| 计算神经科学 | 15、16 | 神经动力学与可塑性怎样联系到计算任务？ |
| 物理学习系统与储备池计算 | 17、18 | 物理动力学提供了什么计算能力，学习发生在哪里？ |
| 机制可解释性、叠加与分析单位 | 19、20 | 特征如何共享表示维度？提取出的特征能否视为内部计算单位？ |
| 认知边界与心灵哲学 | 21 | 解释的可达性是否依赖认知结构？怎样区分哲学论证与技术证据？ |

## 一、自组织与适应

### 1. The Origins of Order: Self-Organization and Selection in Evolution

- **作者与版本：** Stuart A. Kauffman；Oxford University Press，1993。
- **文件：** [本地 PDF][pdf-1]（731 页，扫描件；题名与年份见 PDF 第 1 页）。
- **主题：** 自组织与演化选择、随机布尔网络、适应度景观和生命系统中的秩序。
- **研究问题：** 哪些局部交互能形成稳定且具有功能的组织？这些结构是否有助于学习，仍需独立验证。

### 2. Design for a Brain: The Origin of Adaptive Behaviour

- **作者与版本：** W. Ross Ashby；修订第二版，1960；初版 1952。
- **文件：** [本地 PDF][pdf-2]（293 页，扫描件；本地版本见 PDF 第 1–2 页）。
- **主题：** 适应、稳定性，以及系统在环境扰动下维持行为的机制。
- **研究问题：** 系统如何通过内部反馈适应新条件，同时维持已有功能？

### 3. The Chemical Basis of Morphogenesis

- **作者与出处：** A. M. Turing；1952，*Philosophical Transactions of the Royal Society of London. Series B*，237(641)：37–72。
- **文件：** [本地 PDF][pdf-3]（37 页；书目信息见 PDF 第 1 页）。
- **主题：** 反应—扩散系统与均匀状态失稳所产生的空间模式。
- **研究问题：** 局部动力学与耦合如何共同产生结构？图案形成怎样才能进一步联系到可检验的计算功能？

## 二、自动机与形态生成

### 4. Theory of Self-Reproducing Automata（节选）

- **作者与版本：** John von Neumann；Arthur W. Burks 编辑整理；University of Illinois Press，1966。
- **文件：** [本地 PDF][pdf-4]（25 页，扫描节选；不是全书）。
- **实际内容：** 题名页之后为第四讲 *The Role of High and of Extremely High Complication*，从书内第 64 页开始。不能把本地文件当作完整的自复制自动机构造论述。
- **研究问题：** 复杂度、可靠性与自动机组织之间是什么关系？构造和复用计算结构需要哪些条件？

### 6. Growing Isotropic Neural Cellular Automata

- **作者与版本：** Alexander Mordvintsev、Ettore Randazzo、Craig Fouts；2022；本地为 arXiv:2205.01681v3，2022-06-01。
- **文件与来源：** [本地 PDF][pdf-6]（8 页）；[对应 arXiv 版本](https://arxiv.org/abs/2205.01681v3)。
- **主题：** 原始 Growing NCA 的方向依赖，以及感知规则各向同性、结构化种子和对称性破缺。
- **研究问题：** 哪些方向信息来自状态，哪些由架构或坐标系预先提供？改变这些约束会怎样影响可形成的组织？
- **辨别：** 这是 2022 年的 IsoNCA，区别于 2020 年 *Growing Neural Cellular Automata*；后者见下方在线补充。

## 三、认知、类比与设计

### 7. Fluid Concepts and Creative Analogies: Computer Models of the Fundamental Mechanisms of Thought

- **作者与年份：** Douglas R. Hofstadter 与 Fluid Analogies Research Group；1995。
- **文件：** [本地 PDF][pdf-7]（511 页）。
- **主题：** 概念、类比、灵活表示与 Copycat 等计算模型。
- **研究问题：** 跨任务的相似结构能否被抽象为可调用、可组合的内部操作？

### 8. The Sciences of the Artificial

- **作者与版本：** Herbert A. Simon；第三版，MIT Press，1996。
- **文件：** [本地 PDF][pdf-8]（241 页；版本见 PDF 第 2–4 页）。
- **主题：** 人工系统、设计、复杂性与层次组织。
- **研究问题：** 如何区分系统的结构、功能与目标？什么组织形式可能帮助复杂技能的形成？

## 四、程序抽象与信息论

### 9. Structure and Interpretation of Computer Programs（SICP）

- **作者与版本：** Harold Abelson、Gerald Jay Sussman，Julie Sussman 协作；第二版，1996；本地为 1999 年第四次印刷文本。
- **文件：** [本地 PDF][pdf-9]（521 页；版本见 PDF 第 2–3 页）。
- **主题：** 过程抽象、数据抽象、状态、解释器与元语言抽象。
- **研究问题：** 新内部操作怎样表示、调用和组合？怎样区分新操作与既有操作的参数调整？

### 10. Elements of Information Theory

- **作者与版本：** Thomas M. Cover、Joy A. Thomas；第二版，2006。
- **文件：** [本地 PDF][pdf-10]（774 页；第二版题名见 PDF 第 2 页）。
- **主题：** 熵、互信息、编码、压缩、信息传输与率失真理论。
- **研究问题：** 内部状态存储了什么信息，组件之间传递了什么信息？这些量与任务能力之间是什么关系？
- **辨别：** 信息量或压缩率不能单独证明模型获得了新的推理技能。

## 五、动力系统与反馈控制

### 11. Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering

- **作者与本地版本：** Steven H. Strogatz；版权页标注 2015 年由 Westview Press 出版、2018 年由 CRC Press 出版；ISBN 978-0-8133-4910-7。
- **文件：** [本地 PDF][pdf-11]（532 页；版权信息见 PDF 第 5 页）。本地文件不是 2024 年第三版。
- **主题：** 相空间、定点、稳定性、分岔、极限环与混沌。
- **研究问题：** 学习到的局部更新规则对应什么动力学？有限时间内表现正常，是否能推出长期稳定？

### 12. Feedback Systems: An Introduction for Scientists and Engineers

- **作者与本地版本：** Karl Johan Åström、Richard M. Murray；电子版 v2.10b，2009-02-22。
- **文件：** [本地 PDF][pdf-12]（408 页；版本见 PDF 第 1 页）。本地文件不是 2021 年第二版。
- **主题：** 反馈、闭环行为、稳定性、鲁棒性与控制系统设计。
- **研究问题：** 持续更新和局部扰动下，系统怎样维持功能？稳定性与适应性如何协调？

## 六、生成规则与涌现计算

### 13. The Algorithmic Beauty of Plants

- **作者与版本：** Przemysław Prusinkiewicz、Aristid Lindenmayer；原书 1990，本地为 2004 年电子版。
- **文件：** [本地 PDF][pdf-13]（240 页；电子版说明见 PDF 第 4 页）。
- **主题：** L-systems、生成规则、植物形态与分支结构。
- **研究问题：** 局部规则怎样组织出分层结构？生成结构与生成可复用计算技能之间，还缺少什么条件？

### 14. The Evolution of Emergent Computation

- **作者与版本：** James P. Crutchfield、Melanie Mitchell；本地为 SFI Technical Report 94-03-012，相关期刊论文发表于 1995 年。
- **文件：** [本地 PDF][pdf-14]（13 页；技术报告编号见 PDF 第 1 页）。
- **主题：** 演化产生的分布式计算，以及通过粒子式结构分析信息传递和处理。
- **研究问题：** 当局部交互产生全局任务行为时，怎样识别承担计算的时空结构，并用干预检验其作用？

## 七、计算神经科学

### 15. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition

- **作者与年份：** Wulfram Gerstner、Werner M. Kistler、Richard Naud、Liam Paninski；2014。
- **文件：** [本地 PDF][pdf-15]（590 页；作者与年份见 PDF 第 4–5 页）。
- **主题：** 神经元模型、脉冲、网络动力学、记忆、决策与突触可塑性。
- **研究问题：** 状态变化、神经元间耦合与可塑性如何共同形成任务行为？模型中哪些随机性有机制依据？

### 16. Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems

- **作者与年份：** Peter Dayan、L. F. Abbott；2001。
- **文件：** [本地 PDF][pdf-16]（463 页；作者见 PDF 第 1 页）。
- **主题：** 神经编码与解码、神经回路、适应和学习。
- **研究问题：** 如何把内部活动与可检验的计算任务联系起来？局部学习机制能否支持技能保持、组合与迁移？

## 八、物理学习系统与储备池计算

### 17. Self-Organising Memristive Networks as Physical Learning Systems

- **作者：** Francesco Caravelli、Gianluca Milano、Adam Z. Stieg、Carlo Ricciardi、Simon Anthony Brown、Zdenka Kuncic。
- **版本与类型：** Perspective；首版预印本 2025；本地为 arXiv:2509.00747v2，2026-04-25。
- **文件与来源：** [本地 PDF][pdf-17]（24 页）；[对应 arXiv 版本](https://arxiv.org/abs/2509.00747v2)；[期刊版本 DOI](https://doi.org/10.1038/s42254-026-00978-x)。
- **主题：** 自组织忆阻网络（SOMNs）的电导状态、集体非线性与适应性动力学，以及利用这些性质进行物理学习的路径。
- **研究问题：** 输入诱导的状态改变与任务驱动的学习如何区分？局部可塑性产生的记忆怎样读取、维持与复用？
- **辨别：** 这是一篇综述性观点文章。涉及具体实验效果时，应追溯其引用的原始实验；不能由物理自组织直接推出通用技能获取能力。

### 18. Scalable Photonic Reservoir Computing for Parallel Machine Learning Tasks

- **作者：** A. Aadhi、L. Di Lauro 等；R. Morandotti 为末位作者。
- **年份与出处：** 2026，*Nature Communications*，17：1225；DOI：10.1038/s41467-025-67983-z。
- **文件与来源：** [本地 PDF][pdf-18]（11 页）；[期刊原文](https://doi.org/10.1038/s41467-025-67983-z)。年份按论文页脚的 2026 记录，不按 DOI 中的“025”推断。
- **主题：** 基于非线性放大环镜（NALM）的光子储备池，结合时间复用与波分复用进行并行任务处理；正文第 2 页说明训练限于读出层。
- **研究问题：** 非线性动力学与衰减记忆提供了哪些特征？物理参数调节、读出层训练和内部学习分别起什么作用？
- **辨别：** 并行处理多个任务不等于学会组合技能；硬件吞吐量或能耗指标也不能替代技能获取与迁移的评价。

## 九、机制可解释性、叠加与分析单位

### 19. Sparse Autoencoders Do Not Find Canonical Units of Analysis

- **作者：** Patrick Leask、Bart Bussmann、Michael Pearce、Joseph Bloom、Curt Tigges、Noura Al Moubayed、Lee Sharkey、Neel Nanda；前两位为共同第一作者。
- **版本与出处：** ICLR 2025；本地为 arXiv:2502.04878v1，2025-02-07。
- **文件与来源：** [本地 PDF][pdf-19]（23 页）；[对应 arXiv 版本](https://arxiv.org/abs/2502.04878v1)。题名、作者、会议和版本见 PDF 第 1 页。
- **主题：** 稀疏自编码器（SAE）提取的特征是否构成唯一、完整且不可再分的规范分析单位。论文用 SAE stitching 比较不同字典大小的潜变量，并用 meta-SAEs 分解另一 SAE 的解码方向。
- **论文证据：** 在所研究的设置中，更大 SAE 存在较小 SAE 未捕获的信息；部分潜变量还能分解为更小的组成特征。这对“增大 SAE 就会得到完整、原子化的分析单位”的设想提出了挑战，依据见 PDF 第 1–3 页。
- **研究问题：** 如果要识别新形成的内部计算机制，怎样区分分析工具划分出的特征与模型实际执行的操作？不同字典大小、随机种子和因果干预能否支持同一解释？
- **辨别：** SAE 特征不自动等于技能或计算原语；解码方向的可分解性也不等于模型会执行相应的技能组合。该文不证明所有模型都不存在规范分析单位，也不否定 SAE 的实用价值；本次未核验代码或复现实验。

### 20. Toy Models of Superposition

- **作者与年份：** Nelson Elhage、Tristan Hume、Catherine Olsson、Nicholas Schiefer 等；2022-09-14；Transformer Circuits Thread。
- **文件与来源：** [本地 PDF][pdf-20]（62 页，网页导出版本）；[作者团队原文](https://transformer-circuits.pub/2022/toy_model/index.html)。完整作者名单与日期见 PDF 第 1 页；文件名的单数写法不等于原文题名。
- **主题：** 用稀疏合成特征和小型 ReLU 网络研究叠加表示（superposition）：模型如何用少于特征数量的表示维度存储特征，以及多义神经元、特征干扰和几何结构如何出现。
- **论文证据：** 开头概述了特征稀疏性如何影响叠加表示，并报告在有限玩具模型设置中可于叠加状态执行某些计算；作者明确提醒，对真实网络的推广仍不清楚。依据见 PDF 第 1–3 页。
- **研究问题：** 内部功能是否必须与单个神经元或固定模块一一对应？叠加表示怎样影响计算机制的识别与干预？
- **与第 19 项的关系：** 本文提供特征与神经元不一一对应的模型背景；第 19 项进一步考察 SAE 提取的潜变量是否完整、原子化。两者不能直接证明一种唯一的内部计算分解。
- **辨别：** 这里的 superposition 是特征表示的叠加，不是量子叠加。玩具模型中的结果不证明所有网络都采用相同机制，也不是认知封闭的证据；关联代码尚未锁定提交或复现。

## 十、认知边界与心灵哲学

### 21. Can We Solve the Mind–Body Problem?

- **作者与出处：** Colin McGinn；1989，*Mind*，98(391)：349–366。
- **文件与来源：** [本地 PDF][pdf-21]（19 页，含 JSTOR 说明页，正文为扫描件）；[期刊原文 DOI](https://doi.org/10.1093/mind/XCVIII.391.349)。书目信息见 PDF 第 1 页；认知封闭的定义与论证开端见 PDF 第 2–4 页，即期刊第 349–351 页。
- **主题：** 以意识与大脑的关系为对象，提出认知结构可能限制概念形成与解释可达性的哲学论证。作者将认知封闭表述为某类心智相对于某个性质或理论的限制。
- **研究问题：** 能够观察、计算或利用某个系统，是否足以保证能够形成对它的解释？怎样区分知识不足、资源限制和概念能力限制？
- **与项目的关系：** 为 [认知边界研究备忘录](../fire/cognitive-limits.md) 提供哲学背景，与 README 中独立保留的 Cognitive Closure 猜想对应。
- **辨别：** 这是一篇关于心身问题的哲学论文，不是神经网络不可解释性的实验或形式证明；不能将其论证直接推广为“人类无法理解某些人工计算机制”的既定结论。

## 在线补充：目前没有对应本地 PDF

### 原第 5 项：Autopoiesis and Cognition: The Realization of the Living

- **作者与年份：** Humberto Maturana、Francisco Varela；1980。
- **来源：** [Springer](https://link.springer.com/book/10.1007/978-94-009-8947-4)。保留原索引条目，本次未核读原书。
- **用途：** 自创生与组织维持的概念背景；相关类比仍需转化为明确的模型和可检验假设。

### Growing Neural Cellular Automata（原始 NCA，2020）

- **作者：** Alexander Mordvintsev、Ettore Randazzo、Eyvind Niklasson、Michael Levin。
- **来源：** [Distill 原文、演示与代码入口](https://distill.pub/2020/growing-ca/)；DOI：10.23915/distill.00023。
- **用途：** 介绍 16 通道状态、Sobel 感知、共享更新网络和生长／修复训练；与本地第 6 项 IsoNCA 对照阅读。
- **版本边界：** 本次索引更新未检出或核验关联代码提交，也未执行模型；后续复现需另行锁定源码版本。

## 使用与核对说明

- 索引累计核对了目录清单、PDF 页数、可提取的题名／版权页／摘要；扫描件 1、2、4 的题名页和开头页面已在此前更新时查看。页码均指 PDF 页序，不一定等于书内页码。
- 第 6、17、19 项此前核对了对应 arXiv 版本页；第 18 项依据本地期刊 PDF 第 1–2 页。第 20 项本次核读了 PDF 第 1–3 页；第 21 项核对了说明页、查看了扫描正文第 349–351 页，并核对出版社书目页。上述核对不代表已阅读全文或复现结果。
- 第 7、10、15、16 项未完整核对印次；第 11 项仅按本地版权信息记录。两项在线补充不计入 20 份本地 PDF。没有由文件是否收录推断阅读进度。
- 研究时分别记录：原作报告了什么、我们由此提出什么假设、实际实验测得了什么。尤其区分形态形成、状态记忆、读出层拟合、内部规则学习和技能迁移。

[pdf-1]: <1The Origins of Order - Self-Organization and Selection in Evolution (1993) (Stuart A. Kauffman) (z-library.sk, 1lib.sk, z-lib.sk).pdf>
[pdf-2]: <2Design for a Brain - The Origin of Adaptive Behavior.pdf>
[pdf-3]: <3The Chemical Basis of Morphogenesis.pdf>
[pdf-4]: <4TheoryofSelfReproducingAutomata.pdf>
[pdf-6]: <6Growing Isotropic Neural Cellular Automata.pdf>
[pdf-7]: <7Fluid concepts  creative analogies  computer models of the fundamental mechanisms of thought (Hofstadter, Douglas R., 1945- etc.) (z-library.sk, 1lib.sk, z-lib.sk).pdf>
[pdf-8]: <8Simon_Herbert_A_The_Sciences_of_the_Artificial_3rd_ed.pdf>
[pdf-9]: <9StructureandInterpretationofComputerPrograms.pdf>
[pdf-10]: <10Elements of Information Theory (Wiley Series in -- Thomas M_ Cover -- 2006 -- 286b48343eb6661eac613ef5dbcbe644 -- Anna’s Archive.pdf>
[pdf-11]: <11Nonlinear_Dynamics_and_Chaos_2018_Steven_H._Strogatz.pdf>
[pdf-12]: <12Feedback SystemsAn Introduction for Scientists and Engineers.pdf>
[pdf-13]: <13The Algorithmic Beauty of Plants.pdf>
[pdf-14]: <14The Evolution of Emergent Computation.pdf>
[pdf-15]: <15Neuronal Dynamics From Single Neurons to Networks and Models of Cognition ( etc.) (z-library.sk, 1lib.sk, z-lib.sk).pdf>
[pdf-16]: <16Theoretical Neuroscience _ Computational and Mathematical -- Peter Dayan, L_ F_ Abbott, Tomaso A_ Poggio, Terrence J_ -- 1st edition, December 1, 2001 -- isbn13 9780262041997 -- 1d522319f9643e38faec065cbd51ad7a -- Anna’s.pdf>
[pdf-17]: <17Self-Organising Memristive Networks as Physical.pdf>
[pdf-18]: <18Scalable photonic reservoir computing for parallel machine learning tasks.pdf>
[pdf-19]: <19SPARSE AUTOENCODERS DO NOT FIND CANONICAL.pdf>
[pdf-20]: <20ToyModelOfSuperPosition.pdf>
[pdf-21]: <21Can We Solve the Mind--Body Problem.pdf>
