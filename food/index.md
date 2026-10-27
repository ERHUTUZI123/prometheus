# Prometheus · 跨学科参考文献索引

**研究主题：** Self-Organizing Neural Computation  
**研究问题：** 神经网络能否通过学习，自主形成、组织、保留和复用新的内部计算机制？

本索引收录 **16 部参考作品**，横跨数学、物理、复杂系统、生物学、人工生命、认知科学、哲学、理论计算机科学、信息论、非线性动力系统、进化发育生物学、计算力学、控制论和计算神经科学。每条的「对 Prometheus 的启发」是**研究问题或类比**，不代表原作者已经证明相应的神经网络结论。

**用途：** 长期参考索引，非逐项阅读清单。近期执行以 `water/long.md`、`water/short.md` 为准；遇到具体的数学、概念或机制问题时，再有选择地查阅对应作品。

## 目录

- [一、数学、物理与复杂系统：自组织如何发生？](#一数学物理与复杂系统自组织如何发生)
- [二、生物学与人工生命：系统如何生成自己的组织？](#二生物学与人工生命系统如何生成自己的组织)
- [三、认知科学与哲学：新的内部概念如何产生？](#三认知科学与哲学新的内部概念如何产生)
- [四、理论计算机科学与信息论：如何表示和度量计算？](#四理论计算机科学与信息论如何表示和度量计算)
- [五、非线性动力系统与控制论：组织如何稳定和适应？](#五非线性动力系统与控制论组织如何稳定和适应)
- [六、进化发育生物学与计算力学：新的组织和计算如何出现？](#六进化发育生物学与计算力学新的组织和计算如何出现)
- [七、计算神经科学：神经动力学如何产生计算与学习？](#七计算神经科学神经动力学如何产生计算与学习)

---

## 一、数学、物理与复杂系统：自组织如何发生？

### 1. The Origins of Order: Self-Organization and Selection in Evolution

- **作者：** Stuart A. Kauffman
- **年份：** 1993
- **类型：** 专著
- **来源：** [Oxford University Press](https://academic.oup.com/book/53153)

**内容：** 研究复杂生命系统如何形成秩序，涉及随机布尔网络、适应度景观、基因调控网络和混沌边缘。

**对 Prometheus 的启发：** 神经计算结构是否可能通过局部相互作用，形成稳定且具有功能的高阶组织？

### 2. Design for a Brain: The Origin of Adaptive Behaviour

- **作者：** W. Ross Ashby
- **年份：** 1952
- **类型：** 专著
- **来源：** [完整原著 PDF](https://www.ashby.info/Ashby%20-%20Design%20for%20a%20Brain%20-%20The%20Origin%20of%20Adaptive%20Behavior.pdf)

**内容：** 控制论经典，研究一个由相互作用部件构成的系统如何在环境扰动下产生适应性与稳定性。

**对 Prometheus 的启发：** 能否设计学习系统，使其内部计算组织在面对新任务时自主适应，而无需人为指定每一个操作？

### 3. The Chemical Basis of Morphogenesis

- **作者：** Alan Turing
- **年份：** 1952
- **类型：** 研究论文
- **来源：** [剑桥大学 PDF](https://www.damtp.cam.ac.uk/user/gold/pdfs/teaching/turing1952.pdf)

**内容：** Turing 用反应—扩散方程说明，简单的局部动力学可以使原本均匀的系统产生空间结构。

**对 Prometheus 的启发：** 复杂的内部计算结构能否从相对简单、共享的神经状态更新规则中自发形成？

**本组思考方向：** 从动力系统和自组织机制出发思考架构设计，而不预设必须在现有 Transformer 上增加模块。

---

## 二、生物学与人工生命：系统如何生成自己的组织？

### 4. Theory of Self-Reproducing Automata

- **作者：** John von Neumann；Arthur W. Burks 整理出版
- **年份：** 1966
- **类型：** 专著
- **来源：** [MacTutor 著作介绍](https://mathshistory.st-andrews.ac.uk/Extras/Von_Neumann_books/)

**内容：** 研究形式化自动机如何构造自身的复制品，并涉及复杂机器的组织与构造逻辑。

**对 Prometheus 的启发：** 一个计算系统能否表示、组合和构造新的计算结构？这可以作为研究「新的计算原语如何产生」的理论参照。

### 5. Autopoiesis and Cognition: The Realization of the Living

- **作者：** Humberto Maturana、Francisco Varela
- **年份：** 1980
- **类型：** 专著
- **来源：** [Springer 原著](https://link.springer.com/book/10.1007/978-94-009-8947-4)

**内容：** 研究自创生系统如何通过内部过程持续生产和维持自身的组织结构。

**对 Prometheus 的启发：** 如果神经网络能够改变自己的内部计算结构，怎样让新的组织长期保持功能？如何定义系统的自主组织能力？

### 6. Growing Neural Cellular Automata

- **作者：** Alexander Mordvintsev、Ettore Randazzo、Eyvind Niklasson、Michael Levin
- **年份：** 2020
- **类型：** 交互式研究文章
- **来源：** [Distill：原文、交互演示与代码](https://distill.pub/2020/growing-ca/)

**内容：** 作者训练一套局部神经状态更新规则，让大量单元从一个初始细胞开始形成指定图案，甚至在遭到破坏后重新生成。

**对 Prometheus 的启发：** 复杂结构如何从共享的简单学习规则中形成、稳定和恢复？

**需要区分：** 形态的自组织与计算能力的自组织是两种不同的问题。后者需要定义如何识别新的计算操作，并证明其能够组合、保留和迁移。

---

## 三、认知科学与哲学：新的内部概念如何产生？

### 7. Fluid Concepts and Creative Analogies

- **作者：** Douglas Hofstadter & Fluid Analogies Research Group
- **年份：** 1995
- **类型：** 专著
- **来源：** [Hachette 出版商介绍](https://www.hachettebookgroup.com/titles/douglas-r-hofstadter/fluid-concepts-and-creative-analogies/9780465024759/)

**内容：** 探索概念形成、类比推理和认知中的灵活表示，介绍 Copycat 等计算模型。

**对 Prometheus 的启发：** 新的内部计算原语，是否可能来自系统识别并抽象出不同任务之间可复用的结构？

### 8. The Sciences of the Artificial

- **作者：** Herbert A. Simon
- **年份：** 1969；第三版 1996
- **类型：** 专著
- **来源：** [MIT Press](https://mitpress.mit.edu/9780262264495/the-sciences-of-the-artificial/)

**内容：** 探讨人工系统、复杂组织和设计科学。

**对 Prometheus 的启发：** 我们应当如何定义人工计算系统的结构、功能、目标和设计原则？什么样的组织形式有助于复杂智能的形成？

---

## 四、理论计算机科学与信息论：如何表示和度量计算？

### 9. Structure and Interpretation of Computer Programs (SICP)

- **作者：** Harold Abelson、Gerald Jay Sussman；Julie Sussman 协作
- **年份：** 1984；第二版 1996
- **类型：** 教材
- **来源：** [MIT Press：第二版](https://mitpress.mit.edu/9780262510875/structure-and-interpretation-of-computer-programs/)

**内容：** 以程序抽象、数据抽象、状态和解释器为主线，讨论如何从简单计算过程构造复杂计算系统，特别是第 4 章 *Metalinguistic Abstraction*。

**对 Prometheus 的启发：** 如果网络能够学习并保留新的内部操作，这些操作应如何表示、组合与调用？能否形式化地区分一项新的计算操作和既有操作的简单参数变化？

### 10. Elements of Information Theory

- **作者：** Thomas M. Cover、Joy A. Thomas
- **年份：** 1991；第二版 2006
- **类型：** 教材
- **来源：** [Wiley：第二版](https://www.wiley.com/en-us/elements-of-information-theory-2nd-edition-p-9780471241959)

**内容：** 系统介绍熵、互信息、数据压缩、率失真理论与信息传输等工具，并涉及 Kolmogorov 复杂性。

**对 Prometheus 的启发：** 怎样度量内部状态保存的信息、不同组件之间的信息传递，以及学习出的组织能否压缩、预测或复用任务结构？

**需要区分：** 信息量、信息压缩与计算能力不是同一个指标；若宣称系统形成了新的计算能力，还需定义任务和可检验的行为。

---

## 五、非线性动力系统与控制论：组织如何稳定和适应？

### 11. Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering

- **作者：** Steven H. Strogatz
- **年份：** 第三版 2024
- **类型：** 教材
- **来源：** [CRC Press / Routledge：第三版](https://www.routledge.com/Nonlinear-Dynamics-And-Chaos-With-Applications-To-Physics-Biology-Chemistry-And-Engineering/Strogatz/p/book/9780367026509)

**内容：** 介绍定点稳定性、相空间、极限环、分岔、混沌与同步，为理解复杂系统的时间演化提供数学工具。

**对 Prometheus 的启发：** 哪些局部更新规则会形成稳定的集体动力学？当参数、输入或拓扑发生变化时，新形成的组织能否维持或转变其功能？

### 12. Feedback Systems: An Introduction for Scientists and Engineers

- **作者：** Karl J. Åström、Richard M. Murray
- **年份：** 第二版 2021（作者提供的开放版本为 2020 年）
- **类型：** 教材
- **来源：** [作者网站：第二版与免费 PDF](https://fbswiki.org/wiki/index.php/Feedback_Systems%3A_An_Introduction_for_Scientists_and_Engineers)

**内容：** 以反馈系统模型研究稳定性、鲁棒性、控制器设计与动态系统的闭环行为。

**对 Prometheus 的启发：** 一个可重组的计算系统如何在持续学习、局部失效和环境扰动下维持已获得的功能？系统内部需要怎样的反馈机制？

**与既有文献的关系：** Ashby 的 *Design for a Brain* 提供早期适应与稳态思想；本书补充现代反馈与稳定性分析工具。

---

## 六、进化发育生物学与计算力学：新的组织和计算如何出现？

### 13. The Algorithmic Beauty of Plants

- **作者：** Przemysław Prusinkiewicz、Aristid Lindenmayer
- **年份：** 1990
- **类型：** 专著
- **来源：** [作者团队：全书 PDF 与分章下载](https://algorithmicbotany.org/papers/)

**内容：** 通过 L-systems 等形式化生成规则模拟植物的生长、分支和形态发育，展示相对简单的规则如何组织出复杂结构。

**对 Prometheus 的启发：** 能否把神经计算组织的形成描述为可组合、可分层的生成过程？局部生成规则能否产生不同功能的计算单元？

**需要区分：** 植物形态的生成规律可以提供结构形成的模型，但不能直接推导出神经网络会学到新的计算技能。

### 14. The Evolution of Emergent Computation

- **作者：** James P. Crutchfield、Melanie Mitchell
- **年份：** 1995
- **类型：** 研究论文（PNAS）
- **来源：** [PubMed Central：开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC40688/)

**内容：** 探索简单演化过程如何在空间分布式系统中产生涌现的信息处理，并分析模式形成系统中粒子式的逻辑与信息传递机制。

**对 Prometheus 的启发：** 如果局部相互作用产生了全局计算行为，怎样从时空动态中识别承担信息存储、传播与处理的结构，并验证其计算功能？

**研究价值：** 这是把「自组织」与「可分析的计算机制」直接连接起来的代表性案例，值得在理解 NCA 和 MFM 后回头比较。

---

## 七、计算神经科学：神经动力学如何产生计算与学习？

### 15. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition

- **作者：** Wulfram Gerstner、Werner M. Kistler、Richard Naud、Liam Paninski
- **年份：** 2014
- **类型：** 教材
- **来源：** [作者网站：书籍介绍](https://neuronaldynamics.epfl.ch/book.html)；[在线全文与目录](https://neuronaldynamics.epfl.ch/online/)

**内容：** 从单神经元的膜电位与脉冲发放出发，介绍神经元模型、随机发放、群体与网络动力学，以及记忆、决策和突触可塑性等主题。

**对 Prometheus 的启发：** 局部神经元动力学与突触变化如何共同形成具有计算功能的网络状态？哪些随机性具有明确的建模依据，哪些只是计算上的简化？

### 16. Theoretical Neuroscience: Computational and Mathematical Modeling of Neural Systems

- **作者：** Peter Dayan、L. F. Abbott
- **年份：** 2001
- **类型：** 教材
- **来源：** [MIT Press](https://mitpress.mit.edu/9780262541855/theoretical-neuroscience/)；[作者网站：目录与部分章节](https://www.gatsby.ucl.ac.uk/~dayan/book/toc.html)

**内容：** 分为神经编码与解码、神经元与神经回路、适应与学习三部分，涵盖脉冲统计、信息论、网络模型、突触可塑性、强化学习和表征学习。

**对 Prometheus 的启发：** 如何把神经活动、网络动力学和学习规则连接到可检验的计算任务？生物启发的局部机制能否为技能的获取、保持与迁移提供可分析的模型？

**需要区分：** 神经科学模型对特定生物现象的解释，不等于已经证明相应机制能改善人工学习系统；后者仍需独立的任务定义与实验验证。

---

## 贯穿阅读的研究问题

1. **形成（Emergence）：** 简单局部更新规则怎样产生具有功能的计算结构？
2. **创造（Construction）：** 一个系统怎样形成原本没有显式设计的计算原语？
3. **维持（Retention）：** 新形成的内部结构怎样长期保持功能？
4. **复用（Reuse & Composition）：** 内部计算操作能否被组合并迁移到陌生任务？
5. **适应（Adaptation）：** 新任务能否推动系统自主重组内部计算机制？
6. **度量（Measurement）：** 如何识别并量化系统确实获得了新的计算能力，而不只是形成了复杂图案或动态？

> 阅读原则：优先寻找清晰的数学机制与可证伪假设。跨学科类比只能启发研究，不能替代 ML 实验和理论证明。
