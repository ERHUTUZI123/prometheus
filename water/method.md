# Prometheus · Current Method

**唯一持续维护的方法正文 · native-v7 终端见证形成与连续 monodromy 融合 · 2026-11-01**

**MAIN=TOPOLOGY，人类锁定；当前 M0，最终 M3；Transformer 为主要 baseline。** 本文原位维护计算与学习方法；方向和行动见 [now.md](now.md)。冻结版本、原始结果和详细反例留在同一 V2 的证据记录。

## 1. 中心机制与开发目标

研究一个统一过程：**终端经验形成操作 → 操作成为原生可调用原语 → 在保护旧调用与组合的条件下继续学习 → 经验推动状态分化、原语修订与能力增长。** 连续 monodromy 层接入这个过程，承担已形成操作在共享实现变化中的语义保护。

核心假说是：共享参数连续变化时，能否利用同伦不变量保持旧计算能力，以更少的额外参数、重训及验证成本持续获得新能力？准确率、时间效率、参数增长与完整资源成本共同评价此假说。

学习样本为 $(x,w,c,y)$：初始操作数、符号串、公开任务 ID、终端输出。终端标签提供任务信息；系统自己学习编码、操作、调用与内部记忆。方法状态写为

$$
\mathfrak M_t=(D,K,\theta,\tau,\mathcal B,\mathcal P,\mathcal R_{\rm old};\omega),\qquad
\theta=(E,R,\{W_k\},\{e_a\},\{t_c\},\mathrm{Call}).
$$

$D$ 是内部状态数，$K$ 是原语数；$\mathcal B$ 保存学得的原生操作，$\mathcal P$ 是调用程序，$\mathcal R_{\rm old}$ 是旧行为关系；$\omega$ 是已注册在同一神经模型与优化器中的四维共享连续路径参数。

| 部分 | 当前状态 |
|---|---|
| 终端训练、可学习表示/操作/调用、原生原语形成、三阶段续训 | 已实现、已执行 |
| 旧输入绑定与转移闭包保护、经验触发 $D:8\to16$ 及 $16\to32$ | 两规模均已实现、已执行；保护以实际准入合同为范围 |
| 历史区分见证驱动的置换修订 | 当前v7也已安装真实神经权重并续训200步，全部15个已测划分三种执行均100% |
| 连续配置回路、共享形变与路径提升 | 已从stage0接入真实神经模型，完成N8/N16三阶段训练和强基线比较 |
| 同伦保护降低持续获取的总体成本 | 中心待验证假说；M0 尚未通过 |

## 2. 拓扑对象、执行语义与保护原理

### 2.1 配置空间与 monodromy

令

$$
\mathrm{Conf}_D(\mathbb C)=\{(z_1,\ldots,z_D):z_i\ne z_j\},\qquad
X_D=\mathrm{Conf}_D(\mathbb C)/S_D.
$$

标记其中一个点得到 $D$ 重覆盖

$$
\mathcal E_D=\{(C,z):C\in X_D,\ z\in C\}\xrightarrow{p}X_D.
$$

以固定配置 $C_*$ 为基点，回路 $\gamma$ 的提升把初始 sheet $q$ 送至终止 sheet $q\cdot\rho_D([\gamma])$。有

$$
B_D=\pi_1(X_D,C_*),\qquad
1\longrightarrow PB_D\longrightarrow B_D\xrightarrow{\rho_D}S_D\longrightarrow1.
$$

采用**右作用**：置换矩阵满足 $[\Pi]_{q,\Pi(q)}=1$，状态分布按 $\alpha\mapsto\alpha\Pi$ 更新。先执行 $b_1$ 再执行 $b_2$ 对应 $\rho_D(b_1)\rho_D(b_2)$，一般不可交换。

数学定义依据 [González-Meneses2011 §1.1/1.3、§2.1 Eq2.1](https://www.numdam.org/article/AMBP_2011__18_1_15_0.pdf)。当前执行器使用这个作用的**置换商语义**，并已通过连续配置回路的实际路径提升生成原生执行映射。

### 2.2 同伦保护与完整能力合同

若 $\gamma_\lambda$ 是固定基点的连续回路形变，且全过程处于 $X_D$，则

$$
\rho_D([\gamma_\lambda])=\rho_D([\gamma_0])\quad(0\le\lambda\le1).
$$

这是覆盖同伦提升给出的操作语义保持，参见 [Hatcher Prop1.30, printed pp60–61](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf)。它允许共享实现改变，而旧操作作用保持。

能力合同同时连接三个接口：**输入编码 → 操作调用及复合 → 输出读出**。为旧、当前执行器建立关系 $\mathcal R$，要求初始编码相关，并对每个旧符号闭合：

$$
(q,q')\in\mathcal R\ \Longrightarrow\
R_{\rm hard}(q)=R'_{\rm hard}(q'),\qquad
\bigl(T_{a,c}(q),T'_{a,c}(q')\bigr)\in\mathcal R.
$$

由词长归纳得到全部旧词的已学输出保持。当前用有限 BFS 实现此绑定合同；连续层负责操作作用的同伦保持，二者在同一准入事务中配合。

## 3. 可训练计算与终端信用分配

编码、读出及两个调用槽为

$$
\alpha_0=\operatorname{softmax}(E_x/\tau),\quad
D_R=\operatorname{softmax}(R/\tau),\quad
p_{a,c,s}=\operatorname{softmax}\!\left(\mathrm{Call}_s(\tanh(e_a+t_c))/\tau\right).
$$

置 $\widetilde P_0=I$。对每个 $W_k/\tau$ 做四轮 log-Sinkhorn 行/列归一化，最后再做行归一化，得到实际行随机矩阵 $\widetilde P_k$。执行

$$
A_{a,c}=\prod_{s=1}^{2}\left(\sum_{k=0}^{K}p_{a,c,s,k}\widetilde P_k\right),\quad
\alpha_j=\alpha_{j-1}A_{w_j,c},\quad
p_\theta(y\mid x,w,c)=[\alpha_LD_R]_y.
$$

代码使用 log-semiring，PAD 跳过。编码/读出以近 identity 初始化；初始 $K=6$，现有进程上限 $K\le24,D\le64$。

学习采用 Adam、梯度范数上限 1：

$$
\mathcal L=\mathcal L_{\rm term}+\lambda\mathcal L_{\rm div},\quad
\mathcal L_{\rm term}=-\log p_\theta(y\mid x,w,c),\quad
\mathcal L_{\rm div}=\frac1m\sum_aH(p_a)-H(\bar p)=-I(A;K).
$$

实际对 batch 中不同 $(c,a)$ 及各槽统计，$\lambda=.05$；当前阶段原语形成后停止该辅助项。

单槽 $A_a=\sum_kp_{ak}P_k$，令 $l_{ak}=\mathrm{Call}_{ak}/\tau$、$G_a=\partial\mathcal L_{\rm term}/\partial A_a$，则

$$
\frac{\partial\mathcal L_{\rm term}}{\partial l_{ak}}=p_{ak}\langle G_a,P_k-A_a\rangle,\qquad
\frac{\partial\mathcal L_{\rm term}}{\partial P_k}=\sum_ap_{ak}G_a.
$$

对原始 Call 分数求导另乘 $1/\tau$。此式定位了路由与操作共同坍缩的信用瓶颈。负互信息在差异方向提供负曲率，帮助已有扰动形成专门化；形成后的复用再依据操作语义去重，而非按符号强制分配。

## 4. 从学习行为形成、调用和复用原语

分别定义 soft 计算、完整行为候选、原生调用：

$$
C_{a,c}=\operatorname{Match}(\log A_{a,c}),\qquad
\Pi_k=\operatorname{Match}(W_k),\qquad
T^{\rm native}_{a,c}=\Pi_{r_{a,c,1}}\Pi_{r_{a,c,2}}.
$$

完整行为成为原语的事务为：

1. 每 100 步检查当前 joint validation；达到 .95 时提取模型自己的 $C_{a,c}$。
2. 若与现有原语同类则复用；否则新增分数表示 $W_{new}$，在 $C$ 的位置设 $+8$，其余 $-8$。已有原语只按自身已学类锐化。
3. 收集旧任务**当前 native 图**及新候选。以 $H_{(c,a)}=[\tanh(e_a+t_c),1]$、自生成调用目标 $Y$，解 $\widehat W=H^+Y$，将完整操作调用一次、另一槽 skip。
4. 要求满行秩、条件数 $\le10^8$、double 拟合残差 $\le10^{-5}$、实际路由 margin $\ge1$，并验证旧能力合同。
5. 接受后更新真实模型 Parameter、重建 Adam、记录形成事件并继续训练；失败回滚事务。

记 $G_t=\langle\Pi_1,\ldots,\Pi_K\rangle\le S_D$。新增原语可以扩大原生闭包，也可以压缩已有程序。已保存的首阶段例子中，旧硬原语均为偶置换，新形成的八循环为奇置换，给出该次原生闭包严格扩大的代数见证。

连续层在**同一形成事件**中接收 $\Pi_{new}$，建立对应回路 $\gamma_{new}$；调用头仍调用同一个原语 ID。新原语的共享形变、连续执行与旧绑定由统一合同管理。

### 4.1 用终端经验约束操作形成

当前N16诊断显示，路由已确定，但原语置换的匹配间隙很小。新的形成增量把真实一步终端样本接入同一材料化接口。令 $E_h,R_h$ 为当前学得的硬编码/读出，对新符号 $a$ 定义

$$
M_a(q,j)=\bigwedge_{(x,[a],y)\in\mathcal D_{train}:E_h(x)=q}
\mathbf1[R_h(j)=y],\qquad
\Pi_a^*=\underset{\Pi\in S_D,\ M_a(q,\Pi(q))=1}{\arg\max}
\sum_q\log A_{a,c}(q,\Pi(q)).
$$

未观察的行保持可塑，存在多个可行目的状态时使用当前软计算的评分；约束冲突或无完美匹配则返回表示问题。形成后沿用原生类复用/出生、共享调用头拟合、旧合同、验证准入、真实checkpoint及后续训练。没有按教师规则直接填入操作表。

当 $D=N$、编码/读出为双射且一步数据覆盖全部输入时，终端信息可唯一确定完整操作；当 $D>N$ 或覆盖不足时，较长终端历史仍承担状态区分作用。相同终端数据的普通转移识别是该增量的直接对照。

**已执行：** N16第二阶段已安装真实权重并续训200步，新旧任务全部已测划分三种执行均100%，参数2642→3254；形成事务0.1116s、训练2.9464s、保护0.4525s。修订前失败记录保留。

## 5. 经验驱动的状态增长与操作修订

### 5.1 从终端经验识别需要区分的历史

固定任务下，历史 $h=(x,\text{prefix})$ 的已观测后缀 $s$ 若满足

$$
F(h_i s)\ne F(h_j s),
$$

则两个历史必须对应不同的确定性内部状态。具有实际成对后缀见证的 clique $\mathcal C$ 给出 $D_{min}\ge|\mathcal C|$。证据来自原始训练记录的行号、后缀及标签。

当前候选为空 prefix 或一个新符号的 prefix；搜索限 64 节点、10 万访问、2 秒。只有 $|\mathcal C|>D$ 才扩至下一幂次。扩容保留旧块，初始化新 workspace 和中性 discovery expert，再由终端反馈学习其功能；旧能力合同跨维数重新检查。

### 5.2 已有容量中的分化修订

实际第三阶段表明：容量已达 16，仍可能把两个有区分证据的历史映射到同一状态。因此增长还需要**经验驱动的操作修订**，而非只增加状态数。

对发生别名的历史，取其实际 prefix 中的边 $q\xrightarrow{a}\Pi_k(q)$。枚举候选目的状态 $r$，令

$$
q'=\Pi_k^{-1}(r),\qquad
\Pi'_k(q)=\Pi_k(q'),\quad \Pi'_k(q')=\Pi_k(q),
$$

其余像保持。该提案保持置换结构。先验证全部旧任务闭合，再以当前训练终端误差和未消除的别名数选择严格改进；最终验证集只做一次准入。

接入要求是更新真实神经模型中的 $W_k$、保留原语 ID 与调用头，重建必要的优化器状态，更新经反馈修订的当前任务合同，并继续终端训练。执行图是检查接口，真实 checkpoint 保存可继续训练的参数、旧关系和优化器状态。

### 5.3 从历史区分到可执行转移闭包

当前机制将区分见证进一步用于**状态分配及操作形成**。对历史 $h=(x,p)$，由实际终端训练记录构造观测表 $\mathcal O(h)(s)=y$。若两个历史代表同一状态，则相同后缀响应必须一致；定义操作后继兼容关系

$$
C_a(i,j)=\bigwedge_{s\in\operatorname{dom}\mathcal O(h_i a)\cap\operatorname{dom}\mathcal O(h_j)}
\mathbf1[\mathcal O(h_i a)(s)=\mathcal O(h_j)(s)].
$$

选择历史到当前内部状态的注入 $\mu$：空前缀锚定到 $E_h(x)$，实际观察到的空后缀输出与 $R_h$ 相容，区分clique中的历史两两分离。随后在兼容关系与实际终端输出约束下构造可逆后继 $\sigma_a$，安装

$$
\Pi_a\bigl(\mu(h_i)\bigr)=\mu\bigl(h_{\sigma_a(i)}\bigr).
$$

学习分数提供可行匹配的偏好；已有操作也形成可检验复用候选。对旧模型实际学得的 $f_b(x)=R_{old}(T_b(E_{old}(x)))$，在观察到的前缀层 $p$ 中提出 $h(x,p)\mapsto h(f_b(x),p)$；由当前训练续串约束及完整终端误差选择，不提供正确隐藏模式或调用表。

当前有限实现处理满维数区分clique、两个观察前缀层、最多两个未定历史对齐与64个完整程序候选。所有枚举、旧操作复用、终端训练评估与歧义均记录；最终验证集只做一次准入，之后沿同一材料化、旧合同、连续执行和checkpoint流程继续训练。

**已执行：** N16在32个区分历史仍只使用16状态时，验证496对见证、4106条实际观测，两个对齐中仅一个可行；两个完整训练候选中选出零训练错误方案。形成5个新原生操作，库9→14，参数10984→16354；真实安装及200步续训后，三阶段全部15个已测划分 soft/native/连续执行均100%，实际重载通过。没有新增外部标签。被动自动机识别与约束程序学习是这项获取规则的直接强对照。

**同源机制消融已执行：** 固定原第三阶段step800、相同200批训练和末尾准入时机，仅续训/原生重编译/一步约束的新任务IID为46.48%/46.48%/47.66%；历史闭包与去除显式旧动作候选的历史闭包均100%，旧任务全部100%。两项历史方法在安装后、额外SGD之前已经达到全部已测划分100%；本实例的显式旧动作候选未提供额外唯一程序。历史分配与续串闭合当前作为一个联合增量，参数10984→16354；不将该比较解释为等容量因果分解。原始候选、逐步曲线及各项成本见 `runs/n16-acquisition-contribution-v1/`。

## 6. 共享学习期间的保护与可塑性

操作类在唯一匹配区域

$$
\mathcal U_\Pi=\{W:\langle W,\Pi\rangle>\langle W,Q\rangle\ \forall Q\ne\Pi\}
$$

内保持。当前软件层对每个 Adam 提案回溯

$$
\theta'=\theta+\alpha(\theta_{Adam}-\theta),\qquad
\alpha\in\{1,1/2,1/4,1/8,1/16,1/32,1/64,0\},
$$

直到旧输入—调用—读出合同成立。温度改变也检查合同。允许改变连续分数、共享表示与调用实现；保护对象是旧可执行语义。当前优化器内部状态不随回溯恢复，该行为保留在执行记录。

当前任务的形成与修订合同采用版本化准入；旧任务合同保持，当前未完成行为可以在真实终端反馈支持下修订。原语出生、状态扩容、修订及连续实现都经过同一事务接口，成本分别累计。

**实测的保护分工：** 同源200步四臂中，绑定guard接收前118次提案违反旧合同，回溯后零次；关闭guard时一项旧任务IID降至67.38%，开启时两项均100%。固定guard后，连续monodromy与直接执行的200步参数提案及最终神经权重逐位相同。当前数据将绑定保持归于闭合关系检查；连续同伦提供的操作实现不变量须通过下面的成本接口单独检验。

## 7. 连续 monodromy 接入现有 Method

### 7.1 从已学操作到可执行连续回路

固定基配置 $z_j=4(j-(D-1)/2)$。将模型学得的 $\Pi_k$ 写为相邻交换序列；每个交换段使用

$$
z_i(u)=m-2e^{i\pi h(u)},\quad z_{i+1}(u)=m+2e^{i\pi h(u)},\quad
h(u)=3u^2-2u^3,\quad u\in[0,1].
$$

其余点固定。无序配置构成回路 $C_k(t)$，点间距离至少 4。亦可写作覆盖方程 $F_k(z,t)=\prod_{c\in C_k(t)}(z-c)=0$。

执行器接收无标签配置点集，通过连续路径提升输送内部 sheet，读出提升终点。组合直接串接回路并携带同一 sheet，不在操作之间重置。当前接入模块采用因子化配置接口；神经操作 ID、调用与组合程序沿用第4节。

### 7.2 可持续变化的共享实现参数

用同一 $\omega$ 改变所有原语路径：

$$
\Phi_{\omega,t}(x+iy)=x+i\left[y+a(t)h_\omega(x)\right],\quad
a(t)=\sin^2(\pi t/T),\quad
h_\omega(x)=\sum_\ell\omega_\ell\tanh(v_\ell x+b_\ell).
$$

$v,b$ 固定，$\omega$ 为共享参数。该剪切对每个有限 $\omega$ 都是全局微分同胚，且操作端点为 identity。故 $\Phi_{(1-\lambda)\omega+\lambda\omega',t}$ 给出碰撞自由同伦：

$$
\rho_D([\Phi_{\omega',\cdot}(C_k)])=\rho_D([C_k])=\Pi_k.
$$

令 $B=\sum|\omega_\ell v_\ell|$、$H=\sum|\omega_\ell|$，有统一充分界

$$
\delta_{\rm pair}\ge\frac4{1+B},\qquad
V_{max}\le(1+B)3\pi+\frac{\pi H}{T},\qquad
V_{max}\Delta t<\frac{\delta_{\rm pair}}4.
$$

最后一个条件使无标签点集的近邻提升唯一，提供可计费的执行步长。它将保护转移到解析结构和路径执行，而非每个旧组合逐一重放。

### 7.3 学习接口：形成操作类与优化类内实现

精确 monodromy 在无退化区域内局部常值；因此采用两种相互接入的更新：

$$
\text{终端经验}\longrightarrow\text{操作/调用提案}\longrightarrow
(\Pi_{new},\gamma_{new}),\qquad
\omega\leftarrow\omega-\eta_\omega\nabla_\omega\mathcal J_{exec}.
$$

第一条由第3–5节获得、修订离散操作；第二条在同伦类内学习执行质量。现已执行的目标为当前学得原语路径的几何 kinetic action：

$$
\mathcal A(\omega)=\sum_{k\in\mathcal B_{active}}\int_0^{T_k}\sum_j|\dot z_{k,j,\omega}(t)|^2dt
\simeq c+2b^T\omega+\omega^TH\omega,
\qquad \mathcal L_{fused}=\mathcal L_{term}+.01\,\frac{\mathcal A(\omega)}{\max(1,c)}.
$$

每交换段16点 midpoint quadrature，按当前已学类重建矩；这是几何工作目标的数值积分。终端梯度更新表示、操作与调用，工作梯度更新共享路径形状，两者进入同一 Adam 与 checkpoint。旧调用绑定仍遵守第2.2节。

同信息的执行成本对照为零剪切 $\omega=0$ 及四维二次优化 $\omega_*=-H^+b$（需检查可行域及驻点残差）。它们与神经优化使用相同的工作模型，后续比较其发现、维护和执行成本。

**实际接入：** 修订后的同一神经模型已注册 `path_omega`，真实追加200步联合训练，参数3606→3610。共享权重由 $(.7,-.5,.4,-.3)$ 变为 $(.179184,-.052156,-.031577,.093362)$；全部三阶段15个数据划分的 soft/native/连续执行保持100%，实际重载验证通过。此段在三阶段获取之后接入。当前 `continuous_native_model.py` 已将同一机制接入从stage0开始的形成、调用、保护及扩容流程，N8与N16完整三阶段训练均已执行；默认零剪切初始化。

冷执行通过无标签配置提升获得每个原语的映射，随后已学调用使用这些实际提升映射的精确缓存。缓存签名包含维数、原语类及共享路径参数；变化后重新构造。几何工作、冷构造及热调用分别计价。最初假说的关键检验是：类内共享适应能否减少额外参数、重训或验证，同时新类仍能持续获得。

### 7.4 将同伦保证用于跨共享更新的保护成本（实现中，尚未执行）

同一操作类在整个允许剪切族内具有同一理想作用，因此将首次真实提升的结果存为

$$
\mathcal C[(\mathrm{family\_id},D,\Pi)]=(\widehat\rho_D(\gamma_{\Pi,\omega_0}),\mathrm{receipt}).
$$

`family_id`绑定固定基点、sheet标识、回路编译器、形变函数及版本。后续共享更新只核验当前参数仍属于同一有限数值执行域；类未变时复用已提升语义，新类执行首次真实提升，容量固定且溢出淘汰。维数/覆盖/编译器改变使用新命名空间。保存真实神经参数和策略元数据；本版本不持久化执行映射，重载后重新付费构造。

这把同伦不变量落实为具体维护规则：省去共享几何改变后的重复提升，而非冻结共享参数。复用的是理想操作作用，不声称新几何轨迹已数值执行或物理作用免费。与原精确几何缓存、同样允许类缓存的直接执行比较，并分别记录参数、首次提升、证书检查、存储、训练与旧绑定guard成本。普通闭合约束继续处理编码和调用变化。

## 8. 完整学习过程

```text
输入：任务流中的终端样本；当前神经 checkpoint 与旧能力合同
for 每个新阶段：
    从真实终端记录构造历史区分见证
    若容量下界超过 D：事务扩容并验证旧关系
    repeat 有界训练：
        软执行 → 终端损失 → Adam 提案
        保护旧编码/调用/转移/读出；记录接受、回溯与维护成本
        当前完整行为达到形成门槛时：
            复用/形成原生原语，更新真实权重与调用头
            接入该原语的连续回路；验证旧合同并保存 checkpoint
        若启用配套见证修订事务且历史仍被合并：
            生成置换修订候选 → 终端训练反馈选择 → 验证准入
            安装真实参数并继续训练，保留修订前后记录
        连续层接通时：在保护的同伦类内学习共享实现参数
    评价新任务、全部旧任务及未见组合；保存完整资源账目
```

`continuous_native_model.py` / `train.py --continuous-monodromy` 已完成从第一阶段开始的同一三阶段流程。原语形成、增长、连续执行与联合学习已进入v7主循环；见证修订通过配套显式事务调用，v7已通过显式事务安装该修订并续训；主批原800步结果保持冻结。

## 9. M0：能力、时间、增长和完整资源共同计量

### 9.1 每新增能力的成本向量

每阶段同时记录

$$
\mathbf C_t=(T_{train},T_{discover},T_{install},T_{protect},T_{retrain},
P_t,P_t^{trainable},\Delta P_t,S_{model},S_{optim},S_{retain},
\ell_{infer}^{50/95},N_{queries},N_{reject},M_{peak}).
$$

$T_{protect}$ 包括旧合同、路径认证、温度检查及回滚；存储分别计模型、优化器、旧关系/回放/模块和路径表示。推理计入编码、调用、真实执行与读出，区分冷启动和缓存后的成本。发现、调参、失败重试及维护单独记账并纳入总投入。

能力数 $C_t^{cap}$ 依据冻结任务成功和保持标准计算；原语数 $K$ 与能力数分开。若 $\Delta C_t^{cap}>0$，报告

$$
\frac{\Delta T_{total}}{\Delta C_t^{cap}},\qquad
\frac{\Delta P_t}{\Delta C_t^{cap}},\qquad
\frac{\Delta S_t}{\Delta C_t^{cap}};
$$

未获得新能力的阶段记录失败/删失，不以零成本表示。比较任务数、程序长度和状态规模增长下的成本曲线，以及同准确率/保持要求下的时间—参数—存储前沿。

Transformer 是主要架构基线；Replay、Adapter 使用同一 backbone。相同信息、训练预算、硬件和计时方式是共同条件，强基线调优及专有记忆/模块成本均保留。推理延迟用相同 batch/长度/热身和设备同步测量；评估整个训练集所用时间不替代独立推理延迟。

### 9.2 已执行的三阶段开发结果

数据每阶段 4096 train、512 validation/IID/OOD/length16；800 更新、batch64，只有终端监督。下表 Transformer/Replay/Adapter 使用已实际补跑的较强 LR=.003；Native 为 .01。规模 $N=8$、seed0。

| 阶段 | Transformer 新 / 旧 IID | +Replay 新 / 旧 IID | Native 新 / 旧 IID |
|---|---|---|---|
| 一 | 95.51% / — | 95.51% / — | 100% / — |
| 二 | 61.72% / 69.53% | 58.98% / 98.83% | 100% / 100% |
| 三 | 40.82% / 43.75%,29.10% | 39.84% / 97.07%,57.62% | 97.46% / 100%,100% |

Adapter 最终新任务 32.81%，旧任务 98.44%/47.66%；普通 program-v2 最终新任务 51.56%，旧任务均100%。原始 .01 Transformer 结果仍保留在历史记录。

**较早 checkpoint 的实际修订续训（独立预算）：** 在16状态内搜索15个见证驱动候选，接受一次原语像交换，训练错误111→0；随后更新真实 $W_{10}$、保留调用与编码、修订当前任务合同，保存实际神经 checkpoint 并追加200次 Adam。最终原三阶段全部 train/validation/IID/OOD/length16 的 soft 与 native 均为100%。参数仍3606；新增训练3.1178s、保护0.6550s、安装0.0229s。此新增预算单独记账，上表保留原800步对比，后续同预算基线需得到对应增量机会。

修订前 Native 第三阶段 OOD90.63%、length16为83.59%；实际误差分析与修订均针对有训练区分证据的状态别名。

**v7规模扩展实测：** N16三阶段新任务IID为100%→59.38%→31.45%；最后旧任务为100%/59.38%，其中第二项任务未被正式准入。关闭逐步guard的臂为100%→95.51%→46.09%，最后旧任务均100%。Transformer最终新任务38.87%、旧任务34.18%/38.28%；Replay32.03%、旧任务96.09%/55.86%；Adapter22.85%、旧任务94.73%/40.23%。

当前具体开发缺口是终端软计算到稳定原生操作的形成/准入。N16失败阶段800次保护提案全部完整接受，不能把失败直接解释成保护拒绝；实际单步训练数据已足够区分新操作。完整诊断及IID重复比例见证据报告，数值/匹配敏感性仍待参数级因果检验。现有 topo_off 只关闭逐步/温度 guard，共同形成与扩容仍保持旧语义；该消融边界与详细反例集中在证据记录。

**后续修订轨迹（单独计费）：** N8 v7图修订已实际安装神经权重并追加200步，全部15项三种执行均100%，参数3610。N16第二阶段从59.38%修到100%；从其真实step1000检查点只续训第三阶段800步，得到46.48%新任务、旧任务均100%；再按§5.3形成转移并续训200步，三阶段全部已测划分均100%。成功学习链是800/1000/1000次梯度更新，修订搜索、旧失败分支和几何维护另计；原800步主表未倒改。 同数据且额外批次逐字节匹配的Transformer/Replay/Adapter续训已执行，总梯度更新同为2800；最终新任务IID分别39.84%/34.77%/24.41%，旧任务分别47.27%与35.55% / 96.29%与53.71% / 94.73%与43.75%。这些对照未付与方法相同的离散搜索，不能将等更新次数当成等总计算。

### 9.3 已记录的资源与待补测量

| 规模 / 方法 | 三阶段训练 s | 模型调用总 s | 阶段参数增长 | 最终优化器 / 保持数据 B |
|---|---:|---:|---|---|
| N8 Native ON | 28.233 | 80.005 | 1220→1448→3610 | 28,912 / 1,728 |
| N8 Transformer | 17.544 | 24.770 | 40,280固定 | 322,372 / 0 |
| N8 Replay | 19.321 | 23.601 | 40,280固定 | 322,372 / 1,376,256回放 |
| N16 Native ON | 43.261 | 93.709 | 2642→2642→8836 | 70,720 / 640 |
| N16 Transformer | 16.389 | 23.847 | 41,056固定 | 328,580 / 0 |
| N16 Replay | 14.876 | 18.988 | 41,056固定 | 328,580 / 1,376,256回放 |
| N16 Adapter | 20.823 | 26.739 | 52,224固定，最终活跃4080 | 35,760 / 模块含在参数内 |

Native ON从N8到N16，完整模型调用80.005→93.709s；第三阶段几何矩重建新增323次、63,003,648个point-strands、12.959s，已成为主要额外成本。已付费缓存峰值768→1536B；N16全程逐步保护3.066s、温度保护1.802s、连续冷提升2.512s。几何矩、提升和维护计时包含于训练/评估等字段，不能与总时间重复相加。

**几何维护的同输入成本核查：** 按新训练实际记录的128个类变化事件重放，原全量矩计算5.438s，有界128项LRU为1.507s，数值核查通过；计算27,424,768→7,413,760 point-strands，缓存数值payload及序列化键47,616B。该3.61倍只属于配对计算核。随后同源、同800批真实训练对照已完成：训练子步骤16.634→14.420s，而模型调用总时间38.978→38.938s基本不变；全部任务指标相同，尚无端到端时间收益。

**实际独立推理测试：** 同一L40S、共同输入、batch1/长度16，神经前向p50：Native soft4.895ms、Transformer0.672ms、Replay0.666ms、Adapter1.327ms。完整连续原语库冷构造后调用p50为96.785ms，缓存实际提升映射后的CPU调用为0.112ms；前者包含全部唯一原语构造，后者依赖已付费缓存，分别报告。此项有48组batch/长度/接口计时，全部权重前后哈希保持；使用较早step1200融合checkpoint，未冒称在新v7权重上重新测过。

这些是实际开发成本，当前不同方法准确率及训练过程仍不同；下一轮比较统一阶段流程与增长趋势。完整内存、硬件能耗和等能力生命周期成本仍由相应测量决定。

## 10. 实现与证据

同一 V2：[`2026-10-31T234403-0400-v2`](../.deep-work/2026-10-31T234403-0400-v2/)。以下路径相对此目录：

- 核心：`probes/topology-architecture/{topology_model.py,semantic_promotion_model.py,workspace_refinement.py,train.py}`；Transformer 为 `torch_baselines.py`。
- 融合增量：同目录 `witness_operation_repair.py`、`integrate_native_repair.py`、`continuous_monodromy.py`、`train_monodromy_extension.py`、`continuous_native_model.py`；真实 checkpoint 与续训见 `runs/native-neural-repair-integration-v1/`、`runs/native-continuous-integration-v1/`。
- 冻结协议：`protocol-v1.md`、`native-promotion-protocol-v5.json`、`native-continuation-protocol-v5b.json`、`native-workspace-protocol-v6.json`、`baseline-supplement-protocol-v6b.json`。
- 原始训练、失败、曲线与收据：`probes/topology-architecture/runs/`；源码快照在 `versions/`。远端 overlay 持续同步本地。
- 本批完整结果、两规模曲线、时间/参数/存储、机器可读表格：[`final/topology-architecture/result.md`](../.deep-work/2026-10-31T234403-0400-v2/final/topology-architecture/result.md)。历史成本及本批来源哈希继续保留。
- 架构演变、数学反例、消融范围与详细审查：`candidates/topology-architecture/root-routing-mechanism.md`、`supervision/reviews/review-029-architecture.md`、`review-030-architecture-integration.md`、`review-031-continuous-integration.md`、`review-032-primary-v7-results.md`。

当前架构从初版协议起即采用置换商简化；native-v5 修复原生操作形成，v6 加入经验状态增长。连续层原先未接通并非已记录的连续 monodromy 实验失败；其学习—几何接口现已按第7节实际融合。

原文依据：路由与均衡见 [Shazeer2017 §4](https://arxiv.org/pdf/1701.06538)，连续匹配松弛见 [Mena2018 pp2–3](https://arxiv.org/pdf/1802.08665)，终端前后向信用见 [Rabiner1989 pp262–265](https://www.fceia.unr.edu.ar/prodivoz/Rabiner_1989.pdf)。实际阅读边界及与当前更新的区别留在来源记录。

**当前科学判断：** 连续层与真实学习已接通，N8获取/保持较好；N16暴露形成失败及几何维护变贵。经终端约束和历史闭包修订后，N16三阶段已测功能均达到100%；M0未通过，当前进入获取机制、Monodromy/普通约束与完整成本的分离检验。
