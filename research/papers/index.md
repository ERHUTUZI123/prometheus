# Prometheus · 跨学科参考文献索引

**研究主题：** Self-Organizing Neural Computation  
**研究问题：** 神经网络能否通过学习，自主形成、组织、保留和复用新的内部计算机制？

本索引整理数学、物理、复杂系统、生物学、人工生命、认知科学和哲学方向的八部参考作品。每条的「对 Prometheus 的启发」是**研究问题或类比**，不代表原作者已经证明相应的神经网络结论。

## 目录

- [一、数学、物理与复杂系统：自组织如何发生？](#一数学物理与复杂系统自组织如何发生)
- [二、生物学与人工生命：系统如何生成自己的组织？](#二生物学与人工生命系统如何生成自己的组织)
- [三、认知科学与哲学：新的内部概念如何产生？](#三认知科学与哲学新的内部概念如何产生)

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

## 贯穿阅读的研究问题

1. **形成（Emergence）：** 简单局部更新规则怎样产生具有功能的计算结构？
2. **创造（Construction）：** 一个系统怎样形成原本没有显式设计的计算原语？
3. **维持（Retention）：** 新形成的内部结构怎样长期保持功能？
4. **复用（Reuse & Composition）：** 内部计算操作能否被组合并迁移到陌生任务？
5. **适应（Adaptation）：** 新任务能否推动系统自主重组内部计算机制？

> 阅读原则：优先寻找清晰的数学机制与可证伪假设。跨学科类比只能启发研究，不能替代 ML 实验和理论证明。
