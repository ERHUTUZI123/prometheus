# Prometheus · Current Method

**唯一持续维护的方法正文 · 当前实现 native-v6 · 2026-11-01**

**MAIN=TOPOLOGY，人类锁定；当前目标 M0，最终目标 M3。主要 baseline 固定为 Transformer。** Replay / Adapter 是该基线的持续学习增强；程序学习与约束优化是机制对照，不取代 Transformer 的主要 baseline 地位。

本页随实质算法变化原位更新，不另建“最新方法 v2/v3”。研究方向与行动只见 [now.md](now.md)。旧版本、冻结协议、失败及原始数据留在同一 V2，作为历史证据，不倒改。

## 1. 方法核心与实际边界

当前机制：**终端反馈形成软计算 → 完整计算形成原生可调用操作 → 保持旧输入绑定与转移闭包并继续学习 → 由经验见证决定内部状态扩展。**

学习器只接收 $(x,w,c,y)$：初始操作数、符号序列、公开任务 ID、最终输出。没有正确原语、教师调用、隐藏模式或中间状态。各方法获得相同这些信息；短串的终端反馈也是合法训练数据。

当前是软件置换/编织语义原型；实际原生执行用已学置换索引，不把编织词当作已实测的物理执行，也不声称量子硬件优势。该方法尚不是已证新颖性或 M0 通过结果。

## 2. 状态、执行与终端学习

内部状态 $q\in\{1,\ldots,D\}$；可学习编码 $E$、解码 $R$、符号/任务嵌入 $e_a,t_c$；操作分数 $W_k\in\mathbb R^{D\times D}$；共享调用头产生两个槽，另有 identity skip。

$$
P_0=I,\quad P_k=\operatorname{Sinkhorn}(W_k/\tau),\qquad
p_{a,c,s}=\operatorname{softmax}\!\left(\operatorname{Call}_s(\tanh(e_a+t_c))/\tau\right).
$$

$$
A_{a,c}=\prod_{s=1}^{2}\left(\sum_{k=0}^{K}p_{a,c,s,k}P_k\right),\quad
\alpha_0=\operatorname{softmax}(E_x/\tau),\quad
\alpha_t=\alpha_{t-1}A_{w_t,c},\quad
p(y\mid x,w,c)=[\alpha_L\operatorname{softmax}(R/\tau)]_y.
$$

实现采用有限次 log-Sinkhorn 和 log-semiring 传播，PAD 跳过。初始化有观察接口偏置：编码/读出近 identity；不声称从零发现所有表示。初始 $K=6$，进程上限 $K\le24,D\le64$；实际增加均计价。

训练为 Adam、终端交叉熵、梯度范数上限 1。初始操作形成时使用

$$
\mathcal L=-\log p(y\mid x,w,c)+\lambda\mathcal L_{\rm div},\qquad
\mathcal L_{\rm div}=\frac1m\sum_a H(p_a)-H(\bar p)=-I(A;K),\quad \bar p=\frac1m\sum_ap_a.
$$

实际按 minibatch 中出现的不同 $(c,a)$ 统计并在槽间平均，当前 $\lambda=0.05$；原语形成后停止当前阶段该辅助项。它不是永久“不同符号必须不同专家”的语义规则。

## 3. 路由坍缩的机制推导

仅考虑终端项 $\mathcal L_{\rm term}=-\log p(y\mid x,w,c)$。单槽 $A_a=\sum_kp_{ak}P_k$，令 $l_{ak}=\operatorname{Call}_{ak}/\tau$ 为已除温度的分数，$G_a=\partial\mathcal L_{\rm term}/\partial A_a$：

$$
\frac{\partial\mathcal L_{\rm term}}{\partial l_{ak}}=p_{ak}\langle G_a,P_k-A_a\rangle,\qquad
\frac{\partial\mathcal L_{\rm term}}{\partial P_k}=\sum_ap_{ak}G_a.
$$

若对原始 Call 分数求导，第一式另乘 $1/\tau$；总损失还要加上直接依赖路由的多样性项梯度。相同专家使路由难以区分，相同路由又将不同操作的梯度混入同一专家。完全坍缩时负互信息的值及一阶梯度均为零，但对 $p_a=\bar p+\epsilon v_a,\sum_av_a=0$：

$$
-I(A;K)=-\frac{\epsilon^2}{2m}\sum_{a,k}\frac{v_{ak}^2}{\bar p_k}+O(\epsilon^3).
$$

它提供差异方向的负曲率，可放大已有扰动；不注入操作答案，也不保证饱和路由能脱困。普通专家分配/负载平衡是当前成功的首要已有方法解释。相同操作的不同别名也可能被不必要地分开，故后续复用依据计算语义而非符号不同。

## 4. 从完整行为形成原生操作

必须分开报告 **soft、joint proposal、native execution**：

$$
\Pi_k=\operatorname{Match}(W_k),\qquad
T^{\rm native}_{a,c}=\Pi_{r_{a,c,1}}\Pi_{r_{a,c,2}},\qquad
C_{a,c}=\operatorname{Match}(\log A_{a,c}).
$$

一般 $\operatorname{Match}(AB)\ne\operatorname{Match}(A)\operatorname{Match}(B)$。实际 8 状态结果中，joint 加一正确，但组件调用是同一专家两次。八循环 $U$ 为奇置换，$\operatorname{sgn}(\Pi^2)=+1$；在当时八状态、identity 编解码前提下，该程序不能实现 $U$。不能把联合编译准确率冒充原组件程序的能力。

**实际实现的形成事件：**

1. 每 100 次更新检查；当前 joint validation accuracy 达到 0.95 时提取完整候选 $C$。该阈值不是正确性证明。
2. 对已有硬操作类去重；不存在时由模型自己的 $C$ 初始化新原语。目标分数 $+8$、其余 $-8$；使用的已有原语只向其自身已学类锐化，不填教师算术答案。
3. 编译目标只来自旧上下文的**当前 native 图**与新任务的 joint 候选。旧任务不受保护的 soft proposal 不参与。
4. 对学得的隐藏特征 $H_{(c,a)}=[\tanh(e_a+t_c),1]$，用 $\widehat W=H^+Y$ 重写共享调用头。$Y$ 表示自生成的“完整操作一次，其余 skip”，分数 $\pm4$。
5. 要求满行秩、条件数不超过 $10^8$、double 拟合残差不超过 $10^{-5}$、实际路由 margin 至少为 1，并核对新旧 native 图的闭合关系。失败恢复原 Parameter 对象。
6. 成功后重建 Adam，记录编译/验证费用，停止当前多样性项并继续剩余训练。所有参数仍可训练。当前实现同时保存当前任务的 native 关系，可能保留尚未完全学好的行为，不能隐去这一限制。

形成后的操作真正进入原生调用库。候选可以按自身全操作类复用；是否能低成本复用更长程序仍需实测。普通程序编译/蒸馏可解释这些组成，不据此宣称新颖性。

## 5. 由终端经验决定 workspace 扩展

在固定任务 ID 下，历史 $h=(x,\text{prefix})$ 只保留实际观察的剩余后缀与终端标签。若两个历史有同一后缀 $s$ 且 $F(h_i s)\ne F(h_j s)$，它们不能对应同一确定性状态。两两有实际见证的 clique $\mathcal C$ 给出

$$
D_{\min}\ge|\mathcal C|.
$$

当前候选历史仅空 prefix 或单个新符号 prefix；最多 64 节点、10 万搜索访问、2 秒内限。返回每对历史的后缀、标签和源记录行，不声称找到了最大 clique。观测标签不一致时禁止据此扩容。

只有 $|\mathcal C|>D$ 才扩至覆盖下界的下一幂次，且不超过 64。事务保留旧参数块，新 workspace 用小随机参数及旧**已学**解码行中性拷贝初始化，追加一个无正确语义的 discovery expert。没有预置正确 toggle 或隐藏模式；其使用及转移继续从终端反馈学习。

扩容检查跨维数旧关系，旧图保留原维数。没有展开所有可能组合；新增状态、原语、发现与维护费用均计入。当前有限扩容不等于已建立可扩展架构。

## 6. 保护对象与消融的准确范围

旧能力保存为从已训练模型提取的编码、解码与旧符号闭合转移图。旧/新状态关系 $\mathcal R$ 要求初始编码相关，且

$$
(q,q')\in\mathcal R\Rightarrow
\operatorname{decode}(q)=\operatorname{decode}'(q'),\qquad
(T_aq,T'_aq')\in\mathcal R\quad\forall a\in\Sigma_{\rm old}.
$$

有限 BFS 覆盖所有旧词的已学行为，包含输入—调用绑定；不证明原模型对所有真实任务正确。每个 Adam 提案回溯

$$
\theta'=\theta+\alpha(\theta_{\rm Adam}-\theta),\quad
\alpha\in\{1,1/2,1/4,1/8,1/16,1/32,1/64,0\},
$$

直到旧关系成立。温度也是执行状态，需要语义检查；当前优化器内部状态不随回溯恢复。

置换/编织类是不跨语义类的接口，当前实际验证器同时是普通图等价约束。独有拓扑优势尚未建立。**现有 `topo_off` 仅关闭逐步/温度 guard，仍共享形成原语和扩容时的旧语义保持**；不是移除了所有保护。后续消融必须前瞻定义，不能故意损伤对手，也不能倒改已有解释。

## 7. Transformer 主 baseline 与当前实测

真实两层 Transformer Encoder：$d_{model}=48$、4 heads、FFN96、dropout0，学习输入/位置/任务编码。Replay 为同一 backbone/初始化，总 batch64 中分配 32 当前、32 旧样本，旧存储计价。Adapter 是待完成的增强对照，未执行代码不算结果。

共同数据：每阶段 4096 train、512 validation、512 IID、512 长串、512 length16，数据及 batch 哈希相同。每阶段 800 更新、batch64、当前开发学习率 0.01，仅终端监督。相同更新数不等于相同参数量或总计算。各方法都知道任务 ID；当前不是 task-free continual learning。

**N=8、seed0，实际三阶段检查点续训，旧阶段未重跑：**

| 阶段 | Transformer 当前 / 旧 | Transformer + Replay 当前 / 旧 | 当前原生架构 当前 / 旧 |
|---|---|---|---|
| 一 | 92.19% / — | 92.19% / — | 100% / — |
| 二 | 52.34% / 57.42% | 48.44% / 91.02% | 100% / 100% |
| 三 | 36.91% / 42.38%,31.64% | 33.79% / 89.06%,52.34% | 97.46% / 100%,100% |

逐步 guard ON/OFF 当前这些结果相同。第三阶段原生新任务长串 90.63%、length16 为 83.59%，不写成完全组合泛化。实际反馈产生 16 个两两可区分历史的证据，触发 $D:8\to16$，没有教师隐藏标签。最终原生参数 3606，Transformer40280；原生方法的匹配、头重写、扩容和维护另计，不能仅按参数更少声称总成本优势。

这是单规模单种子的开发证据，M0 未通过。程序表示、路由修正及共同编译保持可能解释当前收益；逐步 guard 没有拒绝/回溯，尚未显示额外因果收益。多规模、多种子、完整强基线及保护归因仍须完成。

## 8. 实现、原文与证据索引

同一运行：[`2026-10-31T234403-0400-v2`](../.deep-work/2026-10-31T234403-0400-v2/)。以下路径相对此运行：

- 实际代码：`probes/topology-architecture/{topology_model.py,semantic_promotion_model.py,workspace_refinement.py,train.py}`；Transformer 在 `torch_baselines.py`。
- 冻结协议：同目录 `native-promotion-protocol-v5.json`、`native-continuation-protocol-v5b.json`、`native-workspace-protocol-v6.json`；历史实际源码保存在 `versions/`。
- 原始数据、失败、检查点、曲线及收据：同目录 `runs/gpu-native-v5-*`、`runs/gpu-native-v5b-*`、`runs/gpu-native-v6-*`；哈希同步清单一并保留。remote overlay 不是已确认的持久盘，原物持续同步本地。
- 历史推导：`candidates/topology-architecture/root-routing-mechanism.md`、`sources/topology-architecture/router-causality-s.md`。它们保留版本史，不是另一份当前方法正文。
- 实际独立审查：`supervision/reviews/review-029-architecture.md` 已查前两阶段；第三阶段后续审查按实际发生记录，不补写成此前已审。

主要原文：普通路由坍缩与均衡解释见 [Shazeer2017 §4](https://arxiv.org/pdf/1701.06538)；匹配/连续松弛接口见 [Mena2018 pp2–3](https://arxiv.org/pdf/1802.08665)；终端条件 responsibility 的推导源于 [Rabiner1989 pp262–265](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf) 的 forward/backward 框架。后者与精确 BPTT 的等价推导已保存，**尚未替代实际 Adam 更新**，不认领新增反馈或新学习原理。完整实际阅读边界见运行内来源记录。

每次实质修改，原位更新本页的状态/更新规则、实现边界、对应版本与证据；旧事实留档、未实现思想不冒充结果。下一阶段的执行行动仍只在 NOW，M0–M3 的验收不由方法叙述替代。
